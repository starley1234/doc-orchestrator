"""Клиент Ollama: вызовы Qwen3-VL Instruct (GPU 1) через POST /api/chat."""

import base64
from typing import Any

import httpx
from pydantic import BaseModel

from app.clients.base import EngineClient


class _Message(BaseModel):
    content: str


class ChatReply(BaseModel):
    message: _Message


class VlmClient(EngineClient):
    name = "Qwen3-VL"

    def __init__(self, http: httpx.AsyncClient, base_url: str, model: str, num_ctx: int) -> None:
        super().__init__(http, base_url)
        self._model = model
        self._num_ctx = num_ctx

    async def chat(
        self, prompt: str, image_png: bytes | None = None, schema: dict[str, Any] | None = None
    ) -> str:
        """Один запрос к модели: текст промпта, по желанию картинка (PNG) и JSON Schema ответа.

        Со схемой модель обязана ответить JSON-ом строго по ней (поле format).
        """
        message: dict[str, Any] = {"role": "user", "content": prompt}
        if image_png:
            message["images"] = [base64.b64encode(image_png).decode()]  # чистый base64, без префикса data:
        body: dict[str, Any] = {
            "model": self._model,
            "messages": [message],
            "stream": False,
            "options": {"num_ctx": self._num_ctx, "temperature": 0.0},  # извлечение данных: без «творчества»
        }
        if schema:
            body["format"] = schema
        reply = await self._post_json("/api/chat", ChatReply, json=body)
        return reply.message.content
