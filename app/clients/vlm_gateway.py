"""Клиент Gateway: вызовы Qwen3-VL (GPU 1) через OpenAI-совместимый чат."""

import base64
from typing import Any

import httpx
from pydantic import BaseModel, Field

from app.clients.base import EngineClient


class _Message(BaseModel):
    content: str


class _Choice(BaseModel):
    message: _Message


class ChatReply(BaseModel):
    choices: list[_Choice] = Field(min_length=1)


class VlmGatewayClient(EngineClient):
    name = "Qwen3-VL"

    def __init__(self, http: httpx.AsyncClient, base_url: str, model: str) -> None:
        super().__init__(http, base_url)
        self._model = model

    async def chat(
        self, prompt: str, image_png: bytes | None = None, schema: dict[str, Any] | None = None
    ) -> str:
        """Один запрос к модели: текст промпта, по желанию картинка (PNG) и JSON Schema ответа.

        Со схемой модель обязана ответить JSON-ом строго по ней (поле format).
        """
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if image_png:
            data_url = "data:image/png;base64," + base64.b64encode(image_png).decode()
            content.append({"type": "image_url", "image_url": {"url": data_url}})
        body: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,  # извлечение данных: без «творчества»
            "stream": False,
        }
        if schema:
            body["format"] = schema
        reply = await self._post_json("/v1/chat/completions", ChatReply, json=body)
        return reply.choices[0].message.content
