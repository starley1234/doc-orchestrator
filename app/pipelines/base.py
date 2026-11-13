"""Общая часть пайплайнов: набор внешних движков и единые шаги работы с ними."""

from dataclasses import dataclass
from typing import Any

import httpx
from redis.asyncio import Redis

from app.clients.dlp import DlpClient
from app.clients.docling import DoclingClient
from app.clients.mineru import MineruClient
from app.clients.paddle import PaddleClient
from app.clients.vlm_gateway import VlmGatewayClient
from app.config import Settings
from app.core.aio import run_cpu
from app.core.gpu_manager import GPUResourceManager
from app.core.image_ops import resize_max_pixels


@dataclass(frozen=True)
class Engines:
    """Всё, с чем работают пайплайны: настройки, клиенты движков и очередь к видеокартам."""

    settings: Settings
    gpu: GPUResourceManager
    docling: DoclingClient
    mineru: MineruClient
    paddle: PaddleClient
    vlm: VlmGatewayClient
    dlp: DlpClient

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
            vlm=VlmGatewayClient(http, settings.vlm_gateway_url, settings.vlm_model),
            dlp=DlpClient(http, settings.dlp_url),
        )

    async def paddle_ocr(self, file_bytes: bytes, filename: str) -> str:
        """Скан с таблицами -> Markdown. PaddleOCR-VL работает на GPU 0, поэтому сначала занимаем его."""
        async with self.gpu.acquire_gpu0("paddle"):
            return await self.paddle.extract(file_bytes, filename)

    async def ask_vlm(
        self, prompt: str, image: bytes | None = None, schema: dict[str, Any] | None = None
    ) -> str:
        """Единственный путь к Qwen3-VL: DLP pre-check промпта -> очередь к GPU 1 -> Gateway.

        Картинка уменьшается до max_pixels: более крупная не поместится в VRAM.
        """
        prompt = (await self.dlp.inspect(prompt)).text
        if image:
            image = await run_cpu(resize_max_pixels, image, self.settings.max_pixels)
        async with self.gpu.acquire_gpu1():
            return await self.vlm.chat(prompt, image, schema)
