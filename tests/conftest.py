"""Общие фикстуры и генераторы тестовых файлов: PDF и PNG создаются на лету.

Мок движков работает внутри процесса: тестам не нужны ни Docker, ни сеть.
"""

import io
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import pymupdf
import pytest
from fakeredis import FakeAsyncRedis, FakeServer
from fastapi import FastAPI
from PIL import Image, ImageDraw

from app.clients.base import GpuEngineClient
from app.clients.mineru import MineruClient
from app.clients.paddle import PaddleClient
from app.config import Settings
from app.core.gpu_manager import GPUResourceManager
from app.main import create_app
from mocks.main import MockState
from mocks.main import create_app as create_mock_app

MOCK_URL = "http://mock-engines"  # условный адрес: запросы перехватывает ASGI-транспорт

# --- Генераторы тестовых файлов -----------------------------------------------------------

# Текст цифрового документа: заметно больше 100 непробельных символов на страницу
TEXT = (
    "Договор поставки № ДП-17 от 10.01.2025. ООО «Ромашка» (Поставщик) обязуется передать, "
    "а АО «Вектор» (Покупатель) — принять и оплатить подшипники и комплектующие по спецификации. "
    "Общая сумма договора составляет 34 923,00 руб., включая НДС 20 %. Срок действия — до 31.12.2025."
)
A4_MM = (210, 297)  # книжный лист
A1_MM = (841, 594)  # альбомный лист


def make_pdf(pages: list[str]) -> bytes:
    """PDF формата A4. pages — текст каждой страницы; страница без текста — скан (картинка во весь лист)."""
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            writer = pymupdf.TextWriter(page.rect)  # Base-14 шрифты через insert_text кириллицу не пишут
            writer.fill_textbox(pymupdf.Rect(60, 60, 535, 780), text, font=pymupdf.Font("helv"), fontsize=11)
            writer.write_text(page)
        else:
            page.insert_image(page.rect, stream=make_png(620, 877, color=245))
    data = doc.tobytes(garbage=3, deflate=True)
    doc.close()
    return data


def make_png(width: int, height: int, color: int = 255) -> bytes:
    """Одноцветная картинка PNG."""
    return encode(Image.new("L", (width, height), color))


def stamp_box(width: int, height: int, sheet_mm: tuple[int, int]) -> tuple[int, int, int, int]:
    """Место основной надписи 185×55 мм на листе: правый нижний угол рамки (поля — 5 мм)."""
    x_scale, y_scale = width / sheet_mm[0], height / sheet_mm[1]  # пикселей на миллиметр
    right, bottom = sheet_mm[0] - 5, sheet_mm[1] - 5
    return (
        round((right - 185) * x_scale),
        round((bottom - 55) * y_scale),
        round(right * x_scale),
        round(bottom * y_scale),
    )


def make_drawing(width: int, height: int, sheet_mm: tuple[int, int]) -> bytes:
    """Чертёж ЕСКД (PNG): рамка, «деталь» из двух диагоналей и чёрный прямоугольник на месте штампа."""
    image = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(image)
    x_scale, y_scale = width / sheet_mm[0], height / sheet_mm[1]
    draw.rectangle(
        (20 * x_scale, 5 * y_scale, width - 5 * x_scale, height - 5 * y_scale), outline=128, width=4
    )
    draw.line((0, 0, width, height), fill=128, width=6)
    draw.line((0, height, width, 0), fill=128, width=6)
    draw.rectangle(stamp_box(width, height, sheet_mm), fill=0)
    return encode(image)


def encode(image: Image.Image, fmt: str = "PNG", **options) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, fmt, **options)
    return buffer.getvalue()


def decode(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data))


@pytest.fixture(scope="session")
def digital_pdf() -> bytes:
    """Цифровой PDF: страница с текстовым слоем."""
    return make_pdf([TEXT])


@pytest.fixture(scope="session")
def scan_pdf() -> bytes:
    """PDF-скан: картинка без текста."""
    return make_pdf([""])


@pytest.fixture(scope="session")
def drawing_a4() -> bytes:
    """Чертёж A4, книжный лист 2480x3508 (300 dpi)."""
    return make_drawing(2480, 3508, A4_MM)


@pytest.fixture(scope="session")
def drawing_a1() -> bytes:
    """Чертёж A1, альбомный лист 4000x2800."""
    return make_drawing(4000, 2800, A1_MM)


# --- Мок движков --------------------------------------------------------------------------


@pytest.fixture
def mock_app() -> FastAPI:
    """Свежий экземпляр mock-engines с короткими задержками."""
    return create_mock_app(delay_scale=0.15)


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
    Остальные движки возвращают короткий текст.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if failing and failing in path:
            return httpx.Response(500, text="engine crashed")
        if path.endswith("/api/chat"):
            title = (json.loads(request.content).get("format") or {}).get("title")
            answer = vlm.get(title, "{}") if isinstance(vlm, dict) else vlm
            return httpx.Response(200, json={"message": {"role": "assistant", "content": answer}})
        if path.endswith("/mineru/extract") and mineru:
            return httpx.Response(200, json=mineru)
        return httpx.Response(200, json={"markdown": "Текст документа"})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# --- Оркестратор ----------------------------------------------------------------------------


@pytest.fixture
def settings() -> Settings:
    """Настройки оркестратора для тестов: движки — мок, интервалы короткие."""
    return Settings(
        _env_file=None,  # файл .env разработчика на тесты не влияет
        docling_url=f"{MOCK_URL}/docling",
        mineru_url=f"{MOCK_URL}/mineru",
        paddle_url=f"{MOCK_URL}/paddle",
        ollama_url=MOCK_URL,
        gpu_lock_wait_sec=120,  # зависшая очередь к GPU уронит тест, а не подвесит его надолго
        gpu0_check_interval_sec=0.05,
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
