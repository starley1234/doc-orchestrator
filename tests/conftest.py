"""Общие фикстуры. Мок движков работает внутри процесса: тестам не нужны ни Docker, ни сеть."""

from collections.abc import AsyncIterator

import httpx
import pytest
from fastapi import FastAPI
from make_samples import SAMPLES, SAMPLES_DIR

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
