"""Точка входа FastAPI: сборка приложения и lifespan (подключение к Redis)."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.config import Settings, get_settings


def create_app(settings: Settings | None = None, *, redis: Redis | None = None) -> FastAPI:
    """Собирает приложение. Готовый клиент Redis можно передать снаружи (так делают тесты)."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.redis = redis or Redis.from_url(settings.redis_url)
        yield
        if redis is None:  # закрываем только то, что открыли сами
            await app.state.redis.aclose()

    app = FastAPI(title="doc-orchestrator", lifespan=lifespan)

    @app.get("/health", tags=["Служебное"])
    async def health() -> dict[str, str]:
        """Проверка живости: сервис отвечает и Redis доступен."""
        try:
            await app.state.redis.ping()
        except RedisError as exc:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Redis недоступен") from exc
        return {"status": "ok"}

    return app


app = create_app()
