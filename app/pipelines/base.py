"""Общая часть пайплайнов: набор ML-движков и единые шаги работы с ними."""

from dataclasses import dataclass, field
from typing import Any

import httpx
from redis.asyncio import Redis

from app.clients.docling import DoclingClient
from app.clients.mineru import MineruClient
from app.clients.paddle import PaddleClient
from app.clients.vlm import VlmClient
from app.config import Settings
from app.core.aio import run_cpu
from app.core.classifier import RouteType, classify_file, decode_text
from app.core.gpu_manager import GPUResourceManager


@dataclass(frozen=True)
class OcrResult:
    """Итог распознавания документа."""

    route: RouteType
    markdown: str
    engine: str  # сработавший движок: docling, mineru, paddle или none (текстовый файл отдан как есть)
    figures: dict[str, bytes] = field(default_factory=dict)  # картинки, вырезанные MinerU: имя -> байты


@dataclass(frozen=True)
class Engines:
    """Всё, с чем работают пайплайны: настройки, клиенты движков и очередь к видеокартам."""

    settings: Settings
    gpu: GPUResourceManager
    docling: DoclingClient
    mineru: MineruClient
    paddle: PaddleClient
    vlm: VlmClient

    @classmethod
    def connect(cls, settings: Settings, redis: Redis, http: httpx.AsyncClient) -> "Engines":
        """Создаёт клиенты движков (адреса берутся из настроек) и менеджер GPU."""
        mineru = MineruClient(http, settings.mineru_url)
        paddle = PaddleClient(http, settings.paddle_url)
        return cls(
            settings=settings,
            gpu=GPUResourceManager(redis, {"mineru": mineru, "paddle": paddle}, settings),
            docling=DoclingClient(http, settings.docling_url),
            mineru=mineru,
            paddle=paddle,
            vlm=VlmClient(http, settings.ollama_url, settings.vlm_model, settings.vlm_num_ctx),
        )

    async def ocr(self, file_bytes: bytes, filename: str, use_paddle: bool = False) -> OcrResult:
        """Общий шаг пайплайнов: классифицирует файл и превращает его в Markdown подходящим движком.

        Скан распознаёт MinerU, а с use_paddle — PaddleOCR-VL (печати и сложные таблицы).
        GPU 0 занят только на время распознавания.
        """
        route = await run_cpu(classify_file, file_bytes, filename)
        if route is RouteType.TEXT:
            return OcrResult(route, decode_text(file_bytes) or "", "none")
        if route in (RouteType.DIGITAL_OFFICE, RouteType.DIGITAL_PDF):
            return OcrResult(route, await self.docling.convert(file_bytes, filename), "docling")  # CPU
        if use_paddle:
            async with self.gpu.acquire_gpu0("paddle"):
                return OcrResult(route, await self.paddle.extract(file_bytes, filename), "paddle")
        async with self.gpu.acquire_gpu0("mineru"):
            reply = await self.mineru.extract(file_bytes, filename)
        return OcrResult(route, reply.markdown, "mineru", reply.figures)

    async def ask_vlm(
        self, prompt: str, image: bytes | None = None, schema: dict[str, Any] | None = None
    ) -> str:
        """Единственный путь к Qwen3-VL: запросы идут через очередь к GPU 1, строго по одному.

        Картинка должна быть подготовлена smart_resize: более крупная не поместится в VRAM.
        """
        async with self.gpu.acquire_gpu1():
            return await self.vlm.chat(prompt, image, schema)
