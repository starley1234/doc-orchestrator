"""Очередь фоновых задач на Redis: асинхронный режим извлечения (async_mode=true)."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from uuid import uuid4

from redis.asyncio import Redis

from app.config import Settings
from app.core.aio import stop_task
from app.core.errors import OrchestratorError
from app.schemas.common import DocType, StructuredResult, TaskStatus

log = logging.getLogger(__name__)

QUEUE = "tasks:queue"  # идентификаторы задач, ждущих обработки
PROCESSING = "tasks:processing"  # задачи в работе: если сервис остановится, они не потеряются

Handler = Callable[[bytes, str, DocType], Awaitable[StructuredResult]]


def _key(task_id: str) -> str:
    """Ключ хеша задачи. Поля: state (JSON состояния), file и filename (до конца обработки)."""
    return f"task:{task_id}"


class TaskQueue:
    """Хранит задачи и результаты в Redis и выполняет их фоновыми воркерами внутри сервиса."""

    def __init__(self, redis: Redis, handler: Handler, settings: Settings) -> None:
        self._redis = redis
        self._handler = handler  # обработчик задачи: StructuredPipeline.run
        self._ttl = settings.task_ttl_s
        self._poll_interval = settings.task_poll_interval_s
        self._workers_count = settings.task_workers
        self._workers: list[asyncio.Task[None]] = []

    async def enqueue(self, file_bytes: bytes, filename: str, doc_type: DocType) -> str:
        """Сохраняет задачу, ставит её в очередь и возвращает task_id."""
        task_id = str(uuid4())
        state = StructuredResult(task_id=task_id, status=TaskStatus.QUEUED, doc_type=doc_type)
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.hset(
                _key(task_id),
                mapping={"state": state.model_dump_json(), "file": file_bytes, "filename": filename},
            )
            pipe.expire(_key(task_id), self._ttl)
            pipe.lpush(QUEUE, task_id)
            await pipe.execute()
        return task_id

    async def get(self, task_id: str) -> StructuredResult | None:
        """Состояние задачи. None — такой задачи нет или срок её хранения истёк."""
        state = await self._redis.hget(_key(task_id), "state")
        return StructuredResult.model_validate_json(state) if state else None

    async def start(self) -> None:
        """Запускает воркеры. Задачи, прерванные прошлой остановкой сервиса, возвращаются в очередь."""
        # Рассчитано на один экземпляр сервиса: при нескольких экземплярах задача,
        # которую выполняет сосед, будет выполнена повторно (результат от этого не изменится).
        while await self._redis.lmove(PROCESSING, QUEUE, "LEFT", "RIGHT"):
            log.warning("Прерванная задача возвращена в очередь")
        self._workers = [
            asyncio.create_task(self._work(), name=f"task-worker-{n}") for n in range(self._workers_count)
        ]

    async def stop(self) -> None:
        await asyncio.gather(*(stop_task(worker) for worker in self._workers))

    async def _work(self) -> None:
        """Цикл воркера: берёт из очереди самую старую задачу и выполняет её."""
        while True:
            try:
                task_id = await self._redis.lmove(QUEUE, PROCESSING, "RIGHT", "LEFT")
                if task_id is None:
                    await asyncio.sleep(self._poll_interval)  # очередь пуста
                    continue
                await self._run(task_id.decode())
                await self._redis.lrem(PROCESSING, 1, task_id)
            except Exception:  # сбой Redis не должен останавливать воркер
                log.exception("Сбой воркера задач")
                await asyncio.sleep(self._poll_interval)

    async def _run(self, task_id: str) -> None:
        """Выполняет задачу и сохраняет её итог: результат или причину сбоя."""
        state, file_bytes, filename = await self._redis.hmget(_key(task_id), "state", "file", "filename")
        if state is None or file_bytes is None:  # срок хранения истёк либо задачу уже выполнили
            return
        task = StructuredResult.model_validate_json(state)
        await self._save(task_id, task.model_copy(update={"status": TaskStatus.PROCESSING}))
        try:
            result = await self._handler(file_bytes, filename.decode(), task.doc_type)
            result = result.model_copy(update={"task_id": task_id})
        except OrchestratorError as exc:
            result = task.model_copy(update={"status": TaskStatus.FAILED, "error": str(exc)})
        except Exception:
            log.exception("Задача %s: непредвиденная ошибка", task_id)
            result = task.model_copy(
                update={"status": TaskStatus.FAILED, "error": "Внутренняя ошибка сервиса"}
            )
        await self._save(task_id, result, finished=True)

    async def _save(self, task_id: str, state: StructuredResult, finished: bool = False) -> None:
        key = _key(task_id)
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.hset(key, "state", state.model_dump_json())
            if finished:
                pipe.hdel(key, "file", "filename")  # файл больше не нужен
            pipe.expire(key, self._ttl)  # результат хранится TTL после последнего изменения
            await pipe.execute()
