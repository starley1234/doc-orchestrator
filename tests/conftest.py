"""Общие фикстуры. Мок движков работает внутри процесса: тестам не нужны ни Docker, ни сеть."""

from collections.abc import AsyncIterator

import httpx
import pytest
from fastapi import FastAPI

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
