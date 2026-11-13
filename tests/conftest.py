"""Общие фикстуры. Мок движков работает внутри процесса: тестам не нужны ни Docker, ни сеть."""

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import pytest
from fakeredis import FakeAsyncRedis, FakeServer
from fastapi import FastAPI
from make_samples import SAMPLES, SAMPLES_DIR

from app.clients.base import GpuEngineClient
from app.clients.mineru import MineruClient
from app.clients.paddle import PaddleClient
from app.config import Settings
from app.core.gpu_manager import GPUResourceManager
from app.main import create_app
from mocks.main import SENSITIVE_RE, MockState
from mocks.main import create_app as create_mock_app

MOCK_URL = "http://mock-engines"  # условный адрес: запросы перехватывает ASGI-транспорт


@pytest.fixture
def mock_app() -> FastAPI:
    """Свежий экземпляр mock-engines с короткими задержками."""
    return create_mock_app(delay_s=0.02)


@pytest.fixture
def mock(mock_app: FastAPI) -> MockState:
    """Состояние мока: что в VRAM, счётчики OOM/429, журнал вызовов."""
    return mock_app.state.mock


@pytest.fixture
async def mock_http(mock_app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    """HTTP-клиент, запросы которого попадают прямо в мок (без сети)."""
    transport = httpx.ASGITransport(app=mock_app)
    async with httpx.AsyncClient(transport=transport, base_url=MOCK_URL) as client:
        yield client


def engines_stub(vlm: str | dict = "{}", mineru: dict | None = None, failing: str = "") -> httpx.AsyncClient:
    """HTTP-клиент с простейшими движками — для сценариев, которых нет в моке.

    vlm — ответ Qwen3-VL: строка либо словарь «название JSON Schema -> ответ» (None — запрос без схемы);
    mineru — ответ MinerU; failing — часть пути движка, который отвечает ошибкой 500.
    DLP маскирует то же, что и мок; остальные движки возвращают короткий текст.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if failing and failing in path:
            return httpx.Response(500, text="engine crashed")
        if path.endswith("/chat/completions"):
            title = (json.loads(request.content).get("format") or {}).get("title")
            answer = vlm.get(title, "{}") if isinstance(vlm, dict) else vlm
            return httpx.Response(200, json={"choices": [{"message": {"content": answer}}]})
        if path.endswith("/inspect"):
            text, hits = SENSITIVE_RE.subn("[REDACTED]", json.loads(request.content)["text"])
            return httpx.Response(200, json={"text": text, "masked": hits > 0})
        if path.endswith("/mineru/extract") and mineru:
            return httpx.Response(200, json=mineru)
        return httpx.Response(200, json={"markdown": "Текст документа"})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.fixture
def settings() -> Settings:
    """Настройки оркестратора для тестов: движки — мок, ожидания и интервалы короткие."""
    return Settings(
        _env_file=None,  # файл .env разработчика на тесты не влияет
        docling_url=f"{MOCK_URL}/docling",
        mineru_url=f"{MOCK_URL}/mineru",
        paddle_url=f"{MOCK_URL}/paddle",
        vlm_gateway_url=f"{MOCK_URL}/gateway",
        dlp_url=f"{MOCK_URL}/dlp",
        gpu_wait_timeout_s=120,  # зависшая очередь к GPU уронит тест, а не подвесит его на 10 минут
        idle_check_interval_s=0.05,
        task_poll_interval_s=0.02,
    )


@pytest.fixture
def redis_server() -> FakeServer:
    """Сервер Redis в памяти процесса, чистый для каждого теста. connected=False имитирует его отказ."""
    return FakeServer()


@pytest.fixture
async def redis(redis_server: FakeServer) -> AsyncIterator[FakeAsyncRedis]:
    client = FakeAsyncRedis(server=redis_server)
    yield client
    await client.aclose()


@pytest.fixture
def engines(mock_http: httpx.AsyncClient, settings: Settings) -> dict[str, GpuEngineClient]:
    """Клиенты моделей, делящих GPU 0."""
    return {
        "mineru": MineruClient(mock_http, settings.mineru_url),
        "paddle": PaddleClient(mock_http, settings.paddle_url),
    }


@pytest.fixture
def gpu(redis: FakeAsyncRedis, engines: dict[str, GpuEngineClient], settings: Settings) -> GPUResourceManager:
    return GPUResourceManager(redis, engines, settings)


@pytest.fixture
def start_app(settings: Settings, redis: FakeAsyncRedis, mock_http: httpx.AsyncClient):
    """Запуск оркестратора целиком: движки — мок, Redis — в памяти.

    Пример: async with start_app(max_file_size_mb=1) as client: ...
    Аргументы переопределяют настройки; http — другой HTTP-клиент для запросов к движкам.
    """

    @asynccontextmanager
    async def start(http: httpx.AsyncClient | None = None, **overrides) -> AsyncIterator[httpx.AsyncClient]:
        app = create_app(settings.model_copy(update=overrides), redis=redis, http=http or mock_http)
        transport = httpx.ASGITransport(app=app)
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(transport=transport, base_url="http://doc-orchestrator") as client,
        ):
            yield client

    return start


@pytest.fixture
async def client(start_app) -> AsyncIterator[httpx.AsyncClient]:
    """HTTP-клиент работающего оркестратора с настройками по умолчанию."""
    async with start_app() as http:
        yield http


def _sample(name: str) -> bytes:
    """Тестовый файл из tests/samples/. Если файла нет, он создаётся генератором."""
    path = SAMPLES_DIR / name
    if not path.exists():
        SAMPLES_DIR.mkdir(exist_ok=True)
        path.write_bytes(SAMPLES[name]())
    return path.read_bytes()


@pytest.fixture(scope="session")
def digital_pdf() -> bytes:
    """PDF с текстовым слоем."""
    return _sample("digital.pdf")


@pytest.fixture(scope="session")
def scan_pdf() -> bytes:
    """PDF-скан: пустая картинка без текста."""
    return _sample("scan.pdf")


@pytest.fixture(scope="session")
def drawing_png() -> bytes:
    """Чертёж 3000x2000 с основной надписью."""
    return _sample("drawing.png")
