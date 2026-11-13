"""Клиент PaddleOCR-VL: сканы со сложными таблицами и формами на GPU 0."""

from app.clients.base import GpuEngineClient, MarkdownReply, file_part


class PaddleClient(GpuEngineClient):
    name = "PaddleOCR-VL"

    async def extract(self, file_bytes: bytes, filename: str) -> str:
        """Скан -> Markdown с таблицами."""
        reply = await self._post_json("/extract", MarkdownReply, files=file_part(file_bytes, filename))
        return reply.markdown
