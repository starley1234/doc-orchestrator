"""Клиент MinerU: основной OCR сканов на GPU 0."""

from pydantic import Base64Bytes, BaseModel

from app.clients.base import GpuEngineClient, file_part


class MineruReply(BaseModel):
    markdown: str  # текст с плейсхолдерами картинок: ![Схема](fig_1.png)
    figures: dict[str, Base64Bytes] = {}  # вырезанные картинки: имя файла -> байты (в JSON — base64)


class MineruClient(GpuEngineClient):
    name = "MinerU"

    async def extract(self, file_bytes: bytes, filename: str) -> MineruReply:
        """Скан -> Markdown и вырезанные из него картинки (схемы, рисунки)."""
        return await self._post_json("/extract", MineruReply, files=file_part(file_bytes, filename))
