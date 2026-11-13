"""Общие фикстуры. Мок движков работает внутри процесса: тестам не нужны ни Docker, ни сеть."""

from collections.abc import AsyncIterator

import httpx
import pytest
from fakeredis import FakeAsyncRedis
from fastapi import FastAPI
from make_samples import SAMPLES, SAMPLES_DIR

from app.clients.base import GpuEngineClient
from app.clients.mineru import MineruClient
from app.clients.paddle import PaddleClient
from app.config import Settings
from app.core.gpu_manager import GPUResourceManager
from mocks.main import MockState
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
        gpu_wait_timeout_s=20,
        idle_check_interval_s=0.05,
    )


@pytest.fixture
async def redis() -> AsyncIterator[FakeAsyncRedis]:
    """Redis в памяти процесса, чистый для каждого теста."""
    client = FakeAsyncRedis()
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
