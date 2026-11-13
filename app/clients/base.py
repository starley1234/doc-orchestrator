"""Общая часть HTTP-клиентов: отправка запросов, единая обработка ошибок, разбор ответов."""

import mimetypes
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.core.errors import EngineError

ReplyT = TypeVar("ReplyT", bound=BaseModel)


class MarkdownReply(BaseModel):
    """Ответ OCR-движка: распознанный документ в Markdown."""

    markdown: str


def file_part(file_bytes: bytes, filename: str) -> dict[str, tuple[str, bytes, str]]:
    """Файл для multipart-запроса (поле file)."""
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    return {"file": (filename, file_bytes, content_type)}


class EngineClient:
    """HTTP-клиент одного внешнего сервиса. Любой сбой превращает в EngineError."""

    name = "Движок"  # название сервиса для сообщений об ошибках

    def __init__(self, http: httpx.AsyncClient, base_url: str) -> None:
        self._http = http  # общий httpx-клиент приложения: пул соединений и таймауты
        self._base_url = base_url.rstrip("/")

    async def _post(self, path: str, **request: Any) -> httpx.Response:
        """POST-запрос. Сетевой сбой, таймаут и ответ 4xx/5xx — это EngineError."""
        try:
            response = await self._http.post(self._base_url + path, **request)
        except httpx.HTTPError as exc:
            raise EngineError(f"{self.name} недоступен: {type(exc).__name__} {exc}".rstrip()) from exc
        if response.is_error:
            raise EngineError(f"{self.name} вернул HTTP {response.status_code}: {response.text[:300]}")
        return response

    async def _post_json(self, path: str, reply: type[ReplyT], **request: Any) -> ReplyT:
        """POST-запрос с разбором JSON-ответа по Pydantic-модели."""
        response = await self._post(path, **request)
        try:
            return reply.model_validate_json(response.content)
        except ValidationError as exc:
            raise EngineError(f"{self.name} вернул некорректный ответ: {exc.errors()[0]['msg']}") from exc


class GpuEngineClient(EngineClient):
    """Движок на GPU 0: модель загружается в VRAM и выгружается из неё по команде."""

    async def load(self) -> None:
        await self._post("/load")

    async def unload(self) -> None:
        await self._post("/unload")
