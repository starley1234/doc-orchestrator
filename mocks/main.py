"""mock-engines: заглушка внешних ML-сервисов (Docling, MinerU, PaddleOCR-VL, Qwen3-VL, DLP)."""

from fastapi import FastAPI

app = FastAPI(title="mock-engines")


@app.get("/health")
async def health() -> dict[str, str]:
    """Проверка живости для docker compose."""
    return {"status": "ok"}
