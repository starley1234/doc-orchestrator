"""Точка входа FastAPI: сборка приложения и lifespan (подключение к Redis и движкам)."""

import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.api import enterprise, openwebui
from app.config import Settings, get_settings
from app.core.errors import OrchestratorError
from app.core.tasks import TaskQueue
from app.pipelines.base import Engines
from app.pipelines.markdown_pipe import MarkdownPipeline
from app.pipelines.structured_pipe import StructuredPipeline

log = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None, *, redis: Redis | None = None, http: httpx.AsyncClient | None = None
) -> FastAPI:
    """Собирает приложение. Готовые клиенты Redis и HTTP можно передать снаружи (так делают тесты)."""
    settings = settings or get_settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # иначе в лог попадает каждый запрос к движкам

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with AsyncExitStack() as stack:  # при остановке всё закрывается в обратном порядке
            # Закрываем только те подключения, которые открыли сами
            redis_client = redis or await stack.enter_async_context(Redis.from_url(settings.redis_url))
            http_client = http or await stack.enter_async_context(
                httpx.AsyncClient(timeout=settings.http_timeout_s)
            )

            engines = Engines.connect(settings, redis_client, http_client)
            structured_pipeline = StructuredPipeline(engines)
            tasks = TaskQueue(redis_client, structured_pipeline.run, settings)
            app.state.redis = redis_client
            app.state.markdown_pipeline = MarkdownPipeline(engines)
            app.state.structured_pipeline = structured_pipeline
            app.state.tasks = tasks

            engines.gpu.start()  # фоновый возврат GPU 0 к mineru после простоя paddle
            stack.push_async_callback(engines.gpu.stop)
            await tasks.start()  # воркеры асинхронных задач
            stack.push_async_callback(tasks.stop)
            yield

    app = FastAPI(title="doc-orchestrator", lifespan=lifespan)
    app.state.settings = settings
    app.include_router(openwebui.router)
    app.include_router(enterprise.router)

    @app.exception_handler(OrchestratorError)
    async def orchestrator_error(request: Request, exc: OrchestratorError) -> JSONResponse:
        """Ошибки сервиса: у каждой свой HTTP-статус и понятное описание."""
        log.warning("%s %s -> %d: %s", request.method, request.url.path, exc.status_code, exc)
        return JSONResponse({"detail": str(exc)}, status_code=exc.status_code)

    @app.exception_handler(RedisError)
    async def redis_error(request: Request, exc: RedisError) -> JSONResponse:
        log.error("%s %s: Redis недоступен: %s", request.method, request.url.path, exc)
        return JSONResponse({"detail": "Redis недоступен"}, status_code=503)

    @app.get("/health", tags=["Служебное"])
    async def health() -> dict[str, str]:
        """Проверка живости: сервис отвечает и Redis доступен."""
        await app.state.redis.ping()
        return {"status": "ok"}

    return app


app = create_app()
