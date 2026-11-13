"""engine-gateway: штатные серверы ML-движков за контрактом, которого ждёт оркестратор.

Оркестратор обращается к движкам так же, как к заглушке mock-engines (AGENTS.md §5):

    POST /docling/convert                    -> {"markdown": "..."}
    POST /mineru/extract                     -> {"markdown": "...", "figures": {"путь": "<base64>"}}
    POST /paddle/extract                     -> {"markdown": "..."}
    POST /{mineru|paddle}/load и /unload     -> смена модели в VRAM GPU 0

У настоящих серверов API другой, а команд load и unload нет вовсе. Шлюз переводит одно в другое:
распознавание — в родные вызовы серверов (gateway/upstreams.py), а load и unload — в запуск
и остановку их контейнеров (gateway/docker_ctl.py): иначе видеопамять у этих серверов не освободить.
"""

import asyncio
import base64
import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import Any, Literal

import httpx
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import JSONResponse

from gateway.config import GatewaySettings
from gateway.docker_ctl import ContainerControl
from gateway.errors import GatewayError
from gateway.upstreams import DoclingServe, EngineApi, MineruApi, PaddleApi

log = logging.getLogger(__name__)

GpuEngineName = Literal["mineru", "paddle"]


@dataclass(frozen=True)
class GpuEngine:
    """Движок GPU 0: его сервер и контейнер, в котором тот работает."""

    api: EngineApi
    container: str


def create_app(
    settings: GatewaySettings | None = None,
    *,
    http: httpx.AsyncClient | None = None,
    docker: httpx.AsyncClient | None = None,
) -> FastAPI:
    """Собирает приложение. HTTP-клиенты движков и Docker можно передать снаружи (так делают тесты)."""
    settings = settings or GatewaySettings()
    logging.basicConfig(
        level=settings.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)  # иначе в лог попадает каждый опрос движка

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with AsyncExitStack() as stack:  # закрываем только те подключения, которые открыли сами
            engines_http = http or await stack.enter_async_context(httpx.AsyncClient(timeout=60))
            docker_http = docker or await stack.enter_async_context(
                httpx.AsyncClient(
                    transport=httpx.AsyncHTTPTransport(uds=settings.docker_socket),
                    base_url="http://docker",  # хост условный: запросы идут в unix-сокет
                    timeout=settings.container_stop_sec + 30,  # остановка контейнера — долгий вызов
                )
            )
            app.state.containers = ContainerControl(docker_http, settings.container_stop_sec)
            app.state.docling = DoclingServe(engines_http, settings.docling_serve_url, settings)
            app.state.mineru = MineruApi(engines_http, settings.mineru_api_url, settings)
            app.state.paddle = PaddleApi(engines_http, settings.paddle_api_url, settings)
            app.state.gpu_engines = {
                "mineru": GpuEngine(app.state.mineru, settings.mineru_container),
                "paddle": GpuEngine(app.state.paddle, settings.paddle_container),
            }
            yield

    app = FastAPI(title="engine-gateway", lifespan=lifespan)

    @app.exception_handler(GatewayError)
    async def gateway_error(request: Request, exc: GatewayError) -> JSONResponse:
        log.warning("%s %s -> %d: %s", request.method, request.url.path, exc.status_code, exc.detail)
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)

    @app.get("/health", tags=["Служебное"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/state", tags=["Служебное"])
    async def state() -> dict[str, Any]:
        """Что сейчас запущено: состояние контейнеров GPU 0 и готовность серверов движков."""
        containers: ContainerControl = app.state.containers
        report: dict[str, Any] = {"docling": {"ready": await app.state.docling.is_ready()}}
        for name, engine in app.state.gpu_engines.items():
            report[name] = {
                "container": await containers.status(engine.container),
                "ready": await engine.api.is_ready(),
            }
        return report

    @app.post("/docling/convert", tags=["Движки"])
    async def docling_convert(file: UploadFile) -> dict[str, str]:
        """Цифровой документ -> Markdown (docling-serve, CPU)."""
        return {"markdown": await app.state.docling.convert(await file.read(), file.filename or "")}

    @app.post("/mineru/extract", tags=["Движки"])
    async def mineru_extract(file: UploadFile) -> dict[str, Any]:
        """Скан -> Markdown и вырезанные из него картинки (MinerU, GPU 0)."""
        markdown, figures = await app.state.mineru.extract(await file.read(), file.filename or "")
        return {
            "markdown": markdown,
            "figures": {path: base64.b64encode(data).decode() for path, data in figures.items()},
        }

    @app.post("/paddle/extract", tags=["Движки"])
    async def paddle_extract(file: UploadFile) -> dict[str, str]:
        """Скан -> Markdown с таблицами и текстом печатей (PaddleOCR-VL, GPU 0)."""
        return {"markdown": await app.state.paddle.extract(await file.read(), file.filename or "")}

    @app.post("/{name}/load", tags=["GPU 0"])
    async def load(name: GpuEngineName) -> dict[str, str]:
        """Запускает контейнер движка и ждёт, пока его сервер начнёт отвечать.

        Сервер, который работает, но сообщает о сбое, перезапускается. Так ведёт себя MinerU, когда
        у него упал внутренний движок (например, не хватило видеопамяти): сам он уже не оправится.
        """
        engine: GpuEngine = app.state.gpu_engines[name]
        containers: ContainerControl = app.state.containers
        title, container = engine.api.title, engine.container
        hint = f"смотрите docker logs {container}"
        await containers.start(container)
        restarted = False
        try:
            async with asyncio.timeout(settings.deadline_sec):
                while True:
                    reply = await engine.api.probe()
                    if reply is None:  # порт ещё не слушается: сервер грузит модели
                        if await containers.status(container) != "running":
                            # Процесс упал при старте — например, модели не хватило видеопамяти
                            raise GatewayError(502, f"Контейнер {container} остановился при запуске: {hint}")
                        await asyncio.sleep(settings.poll_interval_sec)
                    elif reply.status_code == 200:
                        break
                    elif restarted:
                        failure = f"HTTP {reply.status_code}: {reply.text[:300]}"
                        raise GatewayError(502, f"{title} и после перезапуска сообщает о сбое: {failure}")
                    else:
                        log.warning(
                            "GPU 0: %s сообщает о сбое (HTTP %d) — перезапускаю", title, reply.status_code
                        )
                        await containers.stop(container)
                        await containers.start(container)
                        restarted = True
        except TimeoutError:
            raise GatewayError(504, f"{title} не поднялся за {settings.deadline_sec:g} с: {hint}") from None
        log.info("GPU 0: контейнер %s запущен, движок готов", container)
        return {"status": "loaded"}

    @app.post("/{name}/unload", tags=["GPU 0"])
    async def unload(name: GpuEngineName) -> dict[str, str]:
        """Останавливает контейнер движка: его видеопамять освобождается."""
        engine: GpuEngine = app.state.gpu_engines[name]
        await app.state.containers.stop(engine.container)
        log.info("GPU 0: контейнер %s остановлен", engine.container)
        return {"status": "unloaded"}

    return app


app = create_app()
