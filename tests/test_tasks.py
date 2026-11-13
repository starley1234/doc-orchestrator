"""Очередь фоновых задач: порядок, сохранение результата, сбои и возврат прерванных задач в очередь."""

import asyncio

import pytest
from fakeredis import FakeAsyncRedis, FakeServer

from app.config import Settings
from app.core.errors import InvalidFileError
from app.core.tasks import PROCESSING, QUEUE, Handler, TaskQueue
from app.schemas.common import DocType, StructuredResult, TaskStatus, ValidationResult


async def ok(file_bytes: bytes, filename: str, doc_type: DocType) -> StructuredResult:
    """Обработчик-заглушка: «извлекает» из файла его имя и размер."""
    data = {"filename": filename, "size": len(file_bytes)}
    validation = ValidationResult(is_valid=True, warnings=[])
    return StructuredResult(status=TaskStatus.COMPLETED, doc_type=doc_type, data=data, validation=validation)


@pytest.fixture
async def start_queue(redis: FakeAsyncRedis, settings: Settings):
    """Фабрика запущенных очередей: start_queue(обработчик, task_workers=1). Останавливаются после теста."""
    queues: list[TaskQueue] = []

    async def start(handler: Handler = ok, **overrides: float) -> TaskQueue:
        queue = TaskQueue(redis, handler, settings.model_copy(update=overrides))
        queues.append(queue)
        await queue.start()
        return queue

    yield start
    for queue in queues:
        await queue.stop()


async def wait_until(queue: TaskQueue, task_id: str, *statuses: TaskStatus) -> StructuredResult:
    """Ждёт (до 5 секунд), пока задача не окажется в одном из состояний."""
    for _ in range(250):
        task = await queue.get(task_id)
        if task and task.status in statuses:
            return task
        await asyncio.sleep(0.02)
    raise AssertionError(f"Задача {task_id} не перешла в {statuses}: {task}")


async def test_task_is_processed_and_result_is_stored(start_queue, redis: FakeAsyncRedis):
    queue = await start_queue()

    task_id = await queue.enqueue(b"%PDF-1.7", "упд.pdf", DocType.UPD)
    task = await wait_until(queue, task_id, TaskStatus.COMPLETED)

    assert task.task_id == task_id
    assert task.doc_type is DocType.UPD
    assert task.data == {"filename": "упд.pdf", "size": 8}
    assert task.validation == ValidationResult(is_valid=True, warnings=[])
    assert await redis.llen(QUEUE) == await redis.llen(PROCESSING) == 0


async def test_new_task_is_queued(redis: FakeAsyncRedis, settings: Settings):
    queue = TaskQueue(redis, ok, settings)  # воркеры не запущены

    task_id = await queue.enqueue(b"data", "scan.pdf", DocType.INVOICE)

    assert await queue.get(task_id) == StructuredResult(
        task_id=task_id, status=TaskStatus.QUEUED, doc_type=DocType.INVOICE
    )
    assert await redis.lrange(QUEUE, 0, -1) == [task_id.encode()]


async def test_file_is_deleted_after_processing(start_queue, redis: FakeAsyncRedis):
    queue = await start_queue()

    task_id = await queue.enqueue(b"x" * 1000, "scan.pdf", DocType.UPD)
    await wait_until(queue, task_id, TaskStatus.COMPLETED)

    assert await redis.hkeys(f"task:{task_id}") == [b"state"]  # файл в Redis больше не хранится


async def test_task_expires_even_if_never_processed(redis: FakeAsyncRedis, settings: Settings):
    queue = TaskQueue(redis, ok, settings.model_copy(update={"task_ttl_s": 100}))  # воркеры не запущены

    task_id = await queue.enqueue(b"x", "scan.pdf", DocType.UPD)

    assert 0 < await redis.ttl(f"task:{task_id}") <= 100


async def test_result_is_kept_for_full_ttl_after_completion(
    start_queue, redis: FakeAsyncRedis, settings: Settings
):
    idle = TaskQueue(redis, ok, settings)
    task_id = await idle.enqueue(b"x", "scan.pdf", DocType.UPD)
    await redis.expire(f"task:{task_id}", 5)  # задача долго ждала в очереди: срок хранения почти вышел

    queue = await start_queue(task_ttl_s=100)
    await wait_until(queue, task_id, TaskStatus.COMPLETED)

    assert 5 < await redis.ttl(f"task:{task_id}") <= 100  # отсчёт начался заново


async def test_tasks_run_in_order_of_arrival(start_queue, redis: FakeAsyncRedis, settings: Settings):
    order: list[str] = []

    async def record(file_bytes: bytes, filename: str, doc_type: DocType) -> StructuredResult:
        order.append(filename)
        return await ok(file_bytes, filename, doc_type)

    idle = TaskQueue(redis, ok, settings)  # задачи накапливаются в очереди до запуска воркера
    ids = [await idle.enqueue(b"x", f"{n}.pdf", DocType.UPD) for n in range(5)]
    queue = await start_queue(record, task_workers=1)
    await wait_until(queue, ids[-1], TaskStatus.COMPLETED)

    assert order == ["0.pdf", "1.pdf", "2.pdf", "3.pdf", "4.pdf"]


async def test_workers_run_tasks_in_parallel(start_queue):
    both_started = asyncio.Event()
    running = 0

    async def meet(file_bytes: bytes, filename: str, doc_type: DocType) -> StructuredResult:
        nonlocal running
        running += 1
        if running == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=3)  # завершится, только если задач в работе две
        return await ok(file_bytes, filename, doc_type)

    queue = await start_queue(meet, task_workers=2)
    ids = [await queue.enqueue(b"x", "scan.pdf", DocType.UPD) for _ in range(2)]

    for task_id in ids:
        assert (
            await wait_until(queue, task_id, TaskStatus.COMPLETED, TaskStatus.FAILED)
        ).status == "completed"


async def test_service_error_is_stored_as_failed(start_queue):
    async def reject(file_bytes: bytes, filename: str, doc_type: DocType) -> StructuredResult:
        raise InvalidFileError("PDF защищён паролем")

    queue = await start_queue(reject)
    task_id = await queue.enqueue(b"x", "scan.pdf", DocType.UPD)
    task = await wait_until(queue, task_id, TaskStatus.FAILED)

    assert task == StructuredResult(
        task_id=task_id, status=TaskStatus.FAILED, doc_type=DocType.UPD, error="PDF защищён паролем"
    )


async def test_unexpected_error_does_not_leak_details_or_stop_worker(start_queue):
    async def crash(file_bytes: bytes, filename: str, doc_type: DocType) -> StructuredResult:
        if filename == "crash.pdf":
            raise RuntimeError("секретные подробности сбоя")
        return await ok(file_bytes, filename, doc_type)

    queue = await start_queue(crash, task_workers=1)
    crashed = await queue.enqueue(b"x", "crash.pdf", DocType.UPD)
    next_one = await queue.enqueue(b"x", "next.pdf", DocType.UPD)

    assert (await wait_until(queue, crashed, TaskStatus.FAILED)).error == "Внутренняя ошибка сервиса"
    assert (await wait_until(queue, next_one, TaskStatus.COMPLETED)).data["filename"] == "next.pdf"


async def test_interrupted_task_is_resumed_after_restart(
    redis: FakeAsyncRedis, settings: Settings, start_queue
):
    started = asyncio.Event()

    async def hang(file_bytes: bytes, filename: str, doc_type: DocType) -> StructuredResult:
        started.set()
        await asyncio.sleep(3600)  # «долгая» задача: сервис остановят раньше
        raise AssertionError("сюда выполнение не дойдёт")

    first_run = TaskQueue(redis, hang, settings)
    await first_run.start()
    task_id = await first_run.enqueue(b"%PDF-1.7", "upd.pdf", DocType.UPD)
    await asyncio.wait_for(started.wait(), timeout=3)
    await first_run.stop()  # сервис остановили посреди задачи

    assert (await first_run.get(task_id)).status == "processing"
    assert await redis.lrange(PROCESSING, 0, -1) == [task_id.encode()]  # задача не потеряна

    second_run = await start_queue()  # сервис запустили снова
    task = await wait_until(second_run, task_id, TaskStatus.COMPLETED)

    assert task.data == {"filename": "upd.pdf", "size": 8}
    assert await redis.llen(QUEUE) == await redis.llen(PROCESSING) == 0


async def test_expired_task_is_skipped(start_queue, redis: FakeAsyncRedis, settings: Settings):
    idle = TaskQueue(redis, ok, settings)
    expired = await idle.enqueue(b"x", "old.pdf", DocType.UPD)
    await redis.delete(f"task:{expired}")  # срок хранения истёк, пока задача ждала в очереди

    queue = await start_queue()
    fresh = await queue.enqueue(b"x", "new.pdf", DocType.UPD)

    assert (await wait_until(queue, fresh, TaskStatus.COMPLETED)).data["filename"] == "new.pdf"
    assert await queue.get(expired) is None
    assert await redis.llen(QUEUE) == await redis.llen(PROCESSING) == 0


async def test_workers_survive_redis_outage(start_queue, redis_server: FakeServer):
    queue = await start_queue()

    redis_server.connected = False  # Redis «упал»
    await asyncio.sleep(0.1)
    redis_server.connected = True  # и вернулся

    task_id = await queue.enqueue(b"x", "scan.pdf", DocType.UPD)
    assert (await wait_until(queue, task_id, TaskStatus.COMPLETED)).status == "completed"


async def test_stop_is_quick_and_idempotent(redis: FakeAsyncRedis, settings: Settings):
    queue = TaskQueue(redis, ok, settings)
    await queue.start()

    async with asyncio.timeout(2):
        await queue.stop()
        await queue.stop()
