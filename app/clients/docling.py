"""Клиент Docling: цифровые документы на CPU."""

from app.clients.base import EngineClient, MarkdownReply, file_part


class DoclingClient(EngineClient):
    name = "Docling"

    async def convert(self, file_bytes: bytes, filename: str) -> str:
        """Цифровой документ (PDF с текстом, DOCX, XLSX, PPTX, HTML) -> Markdown."""
        reply = await self._post_json("/convert", MarkdownReply, files=file_part(file_bytes, filename))
        return reply.markdown
