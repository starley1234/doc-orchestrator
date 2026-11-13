"""GPU-менеджер: одна задача на видеокарту, одна модель в VRAM GPU 0, возврат к mineru после простоя.

Нарушения ловит сам мок: вторая модель в VRAM — HTTP 500 (OOM), параллельный запрос к Qwen3-VL — HTTP 429.
"""

import asyncio
import time

import httpx
import pytest
from conftest import MOCK_URL
from fakeredis import FakeAsyncRedis, FakeServer

from app.clients.base import GpuEngineClient
from app.clients.paddle import PaddleClient
from app.clients.vlm import VlmClient
from app.config import Settings
from app.core.errors import EngineError, GPUBusyError
from app.core.gpu_manager import KEY_ACTIVE_MODEL, LOCK_GPU0, LOCK_GPU1, GPUResourceManager
from mocks.main import MockState

SCAN = (b"%PDF-1.7 scan", "scan.pdf")
Engines = dict[str, GpuEngineClient]


@pytest.fixture
def make_gpu(redis: FakeAsyncRedis, engines: Engines, settings: Settings):
    """Фабрика менеджеров с изменёнными настройками, например make_gpu(paddle_idle_ttl_sec=0.1)."""

    def make(**overrides: float) -> GPUResourceManager:
        return GPUResourceManager(redis, engines, settings.model_copy(update=overrides))

    return make


async def use_gpu0(gpu: GPUResourceManager, engines: Engines, model: str) -> None:
    """Типичная задача OCR: занять GPU 0 с нужной моделью и распознать документ."""
    async with gpu.acquire_gpu0(model):
        await engines[model].extract(*SCAN)


async def use_gpu0_without_queue(engines: Engines, model: str) -> None:
    """Та же задача без менеджера: сама выгружает соседнюю модель и загружает свою."""
    neighbour = "paddle" if model == "mineru" else "mineru"
    await engines[neighbour].unload()
    await engines[model].load()
    await engines[model].extract(*SCAN)


async def wait_for_model(gpu: GPUResourceManager, model: str) -> None:
    """Ждёт (до 3 секунд), пока фоновая проверка сменит модель на GPU 0."""
    for _ in range(150):
        if await gpu.active_model() == model:
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f"GPU 0 так и не перешёл на {model}")


# --- GPU 0: смена моделей ------------------------------------------------------------------


async def test_first_acquire_loads_model(gpu: GPUResourceManager, mock: MockState, redis: FakeAsyncRedis):
    assert await gpu.active_model() == "none"

    async with gpu.acquire_gpu0("paddle"):
        assert mock.loaded == {"paddle"}

    assert await gpu.active_model() == "paddle"
    assert await redis.get(KEY_ACTIVE_MODEL) == b"paddle"


async def test_switch_unloads_old_model_before_loading_new(gpu: GPUResourceManager, mock: MockState):
    async with gpu.acquire_gpu0("mineru"):
        mock.calls.clear()

    async with gpu.acquire_gpu0("paddle"):
        assert mock.loaded == {"paddle"}

    assert mock.calls == ["mineru/unload", "paddle/load"]
    assert mock.oom_errors == 0


async def test_same_model_is_not_reloaded(gpu: GPUResourceManager, engines: Engines, mock: MockState):
    for _ in range(3):
        await use_gpu0(gpu, engines, "mineru")

    assert mock.calls.count("mineru/load") == 1
    assert mock.calls.count("mineru/extract") == 3


async def test_recovers_when_redis_state_is_lost(
    gpu: GPUResourceManager, mock: MockState, redis: FakeAsyncRedis
):
    async with gpu.acquire_gpu0("paddle"):
        await redis.delete(KEY_ACTIVE_MODEL)  # Redis «забыл» состояние, а paddle остался в VRAM

    async with gpu.acquire_gpu0("mineru"):
        assert mock.loaded == {"mineru"}
    assert mock.oom_errors == 0


async def test_model_is_reloaded_after_engine_restart(
    gpu: GPUResourceManager, engines: Engines, mock: MockState
):
    await use_gpu0(gpu, engines, "mineru")
    mock.loaded.clear()  # движок перезапустился: в VRAM пусто, а в Redis записано «mineru»

    with pytest.raises(EngineError, match="MinerU not loaded"):
        await use_gpu0(gpu, engines, "mineru")
    assert await gpu.active_model() == "none"  # состояние сброшено: что в VRAM, теперь неизвестно

    await use_gpu0(gpu, engines, "mineru")  # следующая задача загружает модель заново
    assert mock.loaded == {"mineru"}


async def test_task_failure_unrelated_to_engines_keeps_the_model(gpu: GPUResourceManager, mock: MockState):
    with pytest.raises(RuntimeError):
        async with gpu.acquire_gpu0("paddle"):
            raise RuntimeError("задача упала")

    assert await gpu.active_model() == "paddle"
    assert mock.loaded == {"paddle"}


async def test_failed_load_leaves_consistent_state(
    redis: FakeAsyncRedis, mock_http: httpx.AsyncClient, engines: Engines, settings: Settings, mock: MockState
):
    broken = {**engines, "paddle": PaddleClient(mock_http, f"{MOCK_URL}/no-such-engine")}
    gpu = GPUResourceManager(redis, broken, settings)
    async with gpu.acquire_gpu0("mineru"):
        assert mock.loaded == {"mineru"}

    with pytest.raises(EngineError, match="PaddleOCR-VL вернул HTTP"):
        async with gpu.acquire_gpu0("paddle"):
            raise AssertionError("в блок нельзя входить, если модель не загрузилась")

    assert await gpu.active_model() == "none"  # mineru выгружен, paddle не загрузился
    assert mock.loaded == set()
    async with gpu.acquire_gpu0("mineru"):  # лок свободен, GPU 0 работает дальше
        assert mock.loaded == {"mineru"}


async def test_failed_unload_keeps_model_registered(
    redis: FakeAsyncRedis, mock_http: httpx.AsyncClient, engines: Engines, settings: Settings, mock: MockState
):
    broken = PaddleClient(mock_http, settings.paddle_url)
    gpu = GPUResourceManager(redis, {**engines, "paddle": broken}, settings)
    async with gpu.acquire_gpu0("paddle"):
        broken._base_url = f"{MOCK_URL}/no-such-engine"  # paddle перестал отвечать

    with pytest.raises(EngineError, match="PaddleOCR-VL"):
        async with gpu.acquire_gpu0("mineru"):
            raise AssertionError("mineru нельзя загружать, пока paddle остаётся в VRAM")

    assert await gpu.active_model() == "paddle"
    assert mock.loaded == {"paddle"}
    assert mock.oom_errors == 0


# --- Взаимное исключение --------------------------------------------------------------------


async def test_concurrent_tasks_never_share_gpu0(gpu: GPUResourceManager, engines: Engines, mock: MockState):
    await asyncio.gather(*(use_gpu0(gpu, engines, model) for model in ["mineru", "paddle"] * 4))

    assert mock.oom_errors == 0
    assert mock.gpu0_peak == 1  # движки ни разу не работали одновременно
    assert mock.calls.count("mineru/extract") == mock.calls.count("paddle/extract") == 4


async def test_without_manager_concurrent_tasks_crash(engines: Engines, mock: MockState):
    # Контрольный опыт: те же задачи без очереди к GPU приводят к OOM
    results = await asyncio.gather(
        *(use_gpu0_without_queue(engines, model) for model in ["mineru", "paddle"] * 4),
        return_exceptions=True,
    )

    assert any(isinstance(result, EngineError) for result in results)
    assert mock.oom_errors > 0


async def test_gpu1_requests_go_one_by_one(
    gpu: GPUResourceManager, mock_http: httpx.AsyncClient, settings: Settings, mock: MockState
):
    vlm = VlmClient(mock_http, settings.ollama_url, settings.vlm_model, settings.vlm_num_ctx)

    async def ask(n: int) -> str:
        async with gpu.acquire_gpu1():
            return await vlm.chat(f"Опиши схему {n}")

    answers = await asyncio.gather(*(ask(n) for n in range(5)))

    assert len(answers) == len(mock.prompts) == 5
    assert mock.gpu1_rejects == 0


async def test_gpu0_and_gpu1_are_independent(gpu: GPUResourceManager):
    async with gpu.acquire_gpu0("mineru"), asyncio.timeout(1), gpu.acquire_gpu1():
        assert await gpu.active_model() == "mineru"  # GPU 1 доступен, пока занят GPU 0


async def test_lock_is_shared_between_service_instances(
    engines: Engines, settings: Settings, mock: MockState
):
    # Два экземпляра оркестратора: у каждого своё подключение к одному и тому же Redis
    server = FakeServer()
    connections = [FakeAsyncRedis(server=server), FakeAsyncRedis(server=server)]
    first, second = (GPUResourceManager(redis, engines, settings) for redis in connections)

    jobs = [use_gpu0(gpu, engines, model) for gpu, model in [(first, "mineru"), (second, "paddle")] * 3]
    await asyncio.gather(*jobs)

    assert mock.oom_errors == 0
    assert mock.gpu0_peak == 1
    for redis in connections:
        await redis.aclose()


async def test_locks_use_timeouts_from_the_spec(redis: FakeAsyncRedis, engines: Engines):
    defaults = Settings(_env_file=None)
    gpu = GPUResourceManager(redis, engines, defaults)

    assert (defaults.gpu_lock_timeout_sec, defaults.gpu_lock_wait_sec) == (300, 300)
    assert (defaults.paddle_idle_ttl_sec, defaults.gpu0_check_interval_sec) == (120, 15)
    async with gpu.acquire_gpu0("mineru"), gpu.acquire_gpu1():
        for lock in (LOCK_GPU0, LOCK_GPU1):
            assert 299_000 < await redis.pttl(lock) <= 300_000  # TTL лока — 300 секунд
    assert await redis.exists(LOCK_GPU0, LOCK_GPU1) == 0  # после работы локи сняты


async def test_lock_ttl_is_separate_from_waiting_time(make_gpu, redis: FakeAsyncRedis):
    gpu = make_gpu(gpu_lock_timeout_sec=50, gpu_lock_wait_sec=7)

    async with gpu.acquire_gpu1():
        assert 49_000 < await redis.pttl(LOCK_GPU1) <= 50_000


async def test_waiting_too_long_raises_gpu_busy(make_gpu):
    gpu = make_gpu(gpu_lock_wait_sec=0.2)

    async with gpu.acquire_gpu1(), asyncio.timeout(60):  # ждём очередь не вечно
        started = time.monotonic()
        with pytest.raises(GPUBusyError, match="lock:gpu1"):
            async with gpu.acquire_gpu1():
                raise AssertionError("занятую видеокарту получить нельзя")

    assert 0.1 <= time.monotonic() - started < 1


async def test_lock_is_released_when_task_fails(gpu: GPUResourceManager):
    with pytest.raises(RuntimeError):
        async with gpu.acquire_gpu0("mineru"):
            raise RuntimeError("задача упала")

    async with asyncio.timeout(1), gpu.acquire_gpu0("mineru"):
        assert await gpu.active_model() == "mineru"


async def test_long_task_keeps_the_lock(make_gpu):
    # TTL лока (0.3 с) меньше длительности задачи (1 с): лок обязан продлеваться, пока задача работает
    gpu = make_gpu(gpu_lock_timeout_sec=0.3)
    events: list[str] = []

    async def task(name: str, duration: float) -> None:
        async with gpu.acquire_gpu1():
            events.append(f"{name}: начало")
            await asyncio.sleep(duration)
            events.append(f"{name}: конец")

    long_task = asyncio.create_task(task("долгая", 1.0))
    await asyncio.sleep(0.05)  # долгая задача уже держит лок
    await asyncio.gather(long_task, task("следующая", 0))

    assert events == ["долгая: начало", "долгая: конец", "следующая: начало", "следующая: конец"]


async def test_release_never_hangs_when_lock_is_being_extended(make_gpu):
    # Лок продлевается каждые 2 мс, поэтому выход из блока часто совпадает с командой продления.
    # Регрессия: отмена, «проглоченная» внутри клиента Redis, навсегда подвешивала выход из блока.
    gpu = make_gpu(gpu_lock_timeout_sec=0.006)

    async with asyncio.timeout(10):
        for n in range(100):
            async with gpu.acquire_gpu1():
                await asyncio.sleep(0.002 + n % 5 * 0.001)


# --- Возврат GPU 0 основной модели после простоя ---------------------------------------------


async def test_idle_paddle_is_replaced_by_mineru(make_gpu, mock: MockState):
    gpu = make_gpu(paddle_idle_ttl_sec=0.1)
    async with gpu.acquire_gpu0("paddle"):
        mock.calls.clear()

    assert not await gpu.release_idle_paddle()  # только что работал — не простаивает
    await asyncio.sleep(0.15)
    assert await gpu.release_idle_paddle()

    assert mock.calls == ["paddle/unload", "mineru/load"]
    assert mock.loaded == {"mineru"}
    assert await gpu.active_model() == "mineru"


async def test_idle_timer_counts_from_the_end_of_last_task(make_gpu, engines: Engines):
    gpu = make_gpu(paddle_idle_ttl_sec=0.2)
    await use_gpu0(gpu, engines, "paddle")
    await asyncio.sleep(0.15)
    await use_gpu0(gpu, engines, "paddle")  # новая задача обнуляет простой
    await asyncio.sleep(0.15)

    assert not await gpu.release_idle_paddle()
    assert await gpu.active_model() == "paddle"


async def test_idle_mineru_stays_loaded(make_gpu, mock: MockState):
    gpu = make_gpu(paddle_idle_ttl_sec=0.05)
    async with gpu.acquire_gpu0("mineru"):
        mock.calls.clear()
    await asyncio.sleep(0.1)

    assert not await gpu.release_idle_paddle()
    assert mock.calls == []  # mineru — модель по умолчанию, выгружать её незачем


async def test_paddle_is_not_unloaded_while_gpu0_is_busy(make_gpu, mock: MockState):
    gpu = make_gpu(paddle_idle_ttl_sec=0.05)
    async with gpu.acquire_gpu0("paddle"):
        mock.calls.clear()
    await asyncio.sleep(0.1)  # paddle «простаивает», но тут приходит новая задача

    async with gpu.acquire_gpu0("paddle"), asyncio.timeout(1):  # проверка не ждёт, пока GPU 0 освободится
        assert not await gpu.release_idle_paddle()

    assert mock.calls == []
    assert mock.loaded == {"paddle"}


async def test_background_watcher_returns_gpu0_to_mineru(make_gpu, engines: Engines, mock: MockState):
    gpu = make_gpu(paddle_idle_ttl_sec=0.1, gpu0_check_interval_sec=0.05)
    gpu.start()
    try:
        await use_gpu0(gpu, engines, "paddle")
        await wait_for_model(gpu, "mineru")
    finally:
        await gpu.stop()

    assert mock.loaded == {"mineru"}
    assert mock.oom_errors == 0


async def test_watcher_survives_engine_failure(
    redis: FakeAsyncRedis, mock_http: httpx.AsyncClient, engines: Engines, settings: Settings, mock: MockState
):
    # Первые попытки выгрузить paddle падают (движок «недоступен»), потом движок оживает
    flaky = PaddleClient(mock_http, f"{MOCK_URL}/no-such-engine")
    quick = settings.model_copy(update={"paddle_idle_ttl_sec": 0.05, "gpu0_check_interval_sec": 0.05})
    gpu = GPUResourceManager(redis, {**engines, "paddle": flaky}, quick)
    await redis.set(KEY_ACTIVE_MODEL, "paddle")
    mock.loaded.add("paddle")

    gpu.start()
    try:
        await asyncio.sleep(0.2)  # несколько неудачных попыток
        assert await gpu.active_model() == "paddle"
        flaky._base_url = settings.paddle_url  # движок снова доступен
        await wait_for_model(gpu, "mineru")
    finally:
        await gpu.stop()

    assert mock.loaded == {"mineru"}
