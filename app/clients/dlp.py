"""Клиент Regex-DLP: маскирование секретов и персональных данных в тексте."""

from pydantic import BaseModel

from app.clients.base import EngineClient


class DlpResult(BaseModel):
    text: str  # текст после маскирования
    masked: bool  # нашлось ли, что маскировать


class DlpClient(EngineClient):
    name = "DLP"

    async def inspect(self, text: str) -> DlpResult:
        """Маскирует секреты в тексте: в промпте перед VLM (pre-check) и в готовом результате (post-check)."""
        return await self._post_json("/inspect", DlpResult, json={"text": text})
