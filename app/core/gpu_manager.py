"""Очередь к видеокартам: распределённые локи в Redis и смена модели в VRAM.

GPU 0 делят MinerU (основная модель) и PaddleOCR-VL — вместе они в 8 ГБ VRAM не помещаются.
GPU 1 целиком занят Qwen3-VL в Ollama, который обрабатывает строго один запрос за раз.
"""

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Literal

from redis.asyncio import Redis
from redis.asyncio.lock import Lock
from redis.exceptions import LockNotOwnedError, RedisError

from app.clients.base import GpuEngineClient
from app.config import Settings
from app.core.aio import stop_task
from app.core.errors import EngineError, GPUBusyError

log = logging.getLogger(__name__)

Gpu0Model = Literal["mineru", "paddle"]
DEFAULT_MODEL: Gpu0Model = "mineru"  # к ней GPU 0 возвращается после простоя paddle

KEY_ACTIVE_MODEL = "gpu0:active_model"  # модель в VRAM GPU 0: «mineru», «paddle» или «none»
KEY_LAST_USED = "gpu0:last_used_ts"  # когда закончилась последняя задача на GPU 0 (unix-время)
LOCK_GPU0 = "lock:gpu0"
LOCK_GPU1 = "lock:gpu1"


class GPUResourceManager:
    """Гарантирует: на каждой видеокарте одна задача в единицу времени, в VRAM GPU 0 — одна модель."""

    def __init__(self, redis: Redis, engines: Mapping[str, GpuEngineClient], settings: Settings) -> None:
        self._redis = redis
        self._engines = engines  # клиенты моделей GPU 0: {"mineru": ..., "paddle": ...}
        self._lock_timeout = settings.gpu_lock_timeout_sec
        self._lock_wait = settings.gpu_lock_wait_sec
        self._idle_ttl = settings.paddle_idle_ttl_sec
        self._check_interval = settings.gpu0_check_interval_sec
        self._watcher: asyncio.Task[None] | None = None

    # --- Захват видеокарт -----------------------------------------------------------------

    @asynccontextmanager
    async def acquire_gpu0(self, target_model: Gpu0Model) -> AsyncIterator[None]:
        """Занимает GPU 0 и загружает в VRAM нужную модель, выгрузив другую."""
        async with self._exclusive(LOCK_GPU0):
            await self._switch_model(target_model)
            try:
                yield
            except EngineError:
                # Движок мог перезапуститься и потерять модель: следующая задача загрузит её заново
                await self._redis.set(KEY_ACTIVE_MODEL, "none")
                raise
            finally:
                await self._redis.set(KEY_LAST_USED, time.time())  # отсюда отсчитывается простой

    @asynccontextmanager
    async def acquire_gpu1(self) -> AsyncIterator[None]:
        """Занимает GPU 1: запросы к Qwen3-VL идут строго по одному."""
        async with self._exclusive(LOCK_GPU1):
            yield

    async def active_model(self) -> str:
        """Модель, загруженная сейчас в VRAM GPU 0."""
        value = await self._redis.get(KEY_ACTIVE_MODEL)
        return value.decode() if value else "none"

    # --- Возврат GPU 0 основной модели после простоя -----------------------------------------

    def start(self) -> None:
        """Запускает фоновую проверку простоя paddle."""
        self._watcher = asyncio.create_task(self._watch_idle(), name="gpu0-idle-watcher")

    async def stop(self) -> None:
        await stop_task(self._watcher)

    async def release_idle_paddle(self) -> bool:
        """Если paddle простаивает дольше TTL, выгружает его и возвращает mineru. True — модель сменилась."""
        if not await self._paddle_is_idle():
            return False
        with contextlib.suppress(GPUBusyError):  # GPU 0 занят — значит, он не простаивает
            async with self._exclusive(LOCK_GPU0, blocking=False):
                if await self._paddle_is_idle():  # перепроверяем под локом: нас могли опередить
                    await self._switch_model(DEFAULT_MODEL)
                    return True
        return False

    async def _paddle_is_idle(self) -> bool:
        active, last_used = await self._redis.mget(KEY_ACTIVE_MODEL, KEY_LAST_USED)
        return active == b"paddle" and time.time() - float(last_used or 0) > self._idle_ttl

    async def _watch_idle(self) -> None:
        while True:
            await asyncio.sleep(self._check_interval)
            try:
                await self.release_idle_paddle()
            except Exception:  # сбой Redis или движка не должен останавливать наблюдение
                log.exception("Не удалось вернуть GPU 0 модели %s", DEFAULT_MODEL)

    # --- Внутреннее ----------------------------------------------------------------------

    async def _switch_model(self, target: str) -> None:
        """Меняет модель в VRAM GPU 0. Вызывать только под локом GPU 0."""
        active = await self.active_model()
        if active == target:
            return
        log.info("GPU 0: смена модели %s -> %s", active, target)
        # Освобождаем VRAM от всех прочих моделей, а не только от активной: после сброса Redis
        # или рестарта движков запись в Redis могла разойтись с тем, что реально загружено.
        for name, engine in self._engines.items():
            if name == target:
                continue
            try:
                await engine.unload()
            except EngineError:
                if name == active:  # модель точно в VRAM: не выгрузив её, новую загружать нельзя
                    raise
                log.warning("GPU 0: не удалось выгрузить %s (по данным Redis она и не загружена)", name)
        await self._redis.set(KEY_ACTIVE_MODEL, "none")
        await self._engines[target].load()
        await self._redis.set(KEY_ACTIVE_MODEL, target)

    @asynccontextmanager
    async def _exclusive(self, name: str, blocking: bool = True) -> AsyncIterator[None]:
        """Распределённый лок Redis: дожидаемся своей очереди и держим лок до выхода.

        blocking=False — не ждать: если видеокарта занята, сразу GPUBusyError.
        """
        lock = self._redis.lock(name, timeout=self._lock_timeout, blocking_timeout=self._lock_wait)
        if not await lock.acquire(blocking=blocking):
            raise GPUBusyError(f"Видеокарта занята: очередь ({name}) не подошла за {self._lock_wait:g} с")
        keeper = asyncio.create_task(self._keep_alive(lock))
        try:
            yield
        finally:
            await stop_task(keeper)
            try:
                await lock.release()
            except RedisError as exc:  # лок уже не наш или Redis недоступен: он истечёт сам по TTL
                log.error("Не удалось освободить %s: %s", name, exc)

    async def _keep_alive(self, lock: Lock) -> None:
        """Продлевает лок, пока задача работает: TTL лока страхует только от падения процесса."""
        while True:
            await asyncio.sleep(self._lock_timeout / 3)
            try:
                await lock.reacquire()
            except LockNotOwnedError:
                log.error("Лок %s потерян: видеокарту могла занять другая задача", lock.name)
                return
            except RedisError as exc:
                log.warning("Не удалось продлить %s: %s", lock.name, exc)
