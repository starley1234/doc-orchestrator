"""Клиент MinerU: основной OCR сканов на GPU 0."""

from pydantic import Base64Bytes, BaseModel

from app.clients.base import GpuEngineClient, file_part


class Figure(BaseModel):
    """Картинка (схема, рисунок), вырезанная из документа."""

    name: str  # имя файла из плейсхолдера в Markdown, например fig_1.png
    data: Base64Bytes  # в JSON — base64, после разбора — байты изображения


class MineruReply(BaseModel):
    markdown: str  # текст с плейсхолдерами картинок: ![fig_1](fig_1.png)
    images: list[Figure] = []


class MineruClient(GpuEngineClient):
    name = "MinerU"

    async def extract(self, file_bytes: bytes, filename: str) -> MineruReply:
        """Скан -> Markdown и вырезанные из него картинки."""
        return await self._post_json("/extract", MineruReply, files=file_part(file_bytes, filename))
