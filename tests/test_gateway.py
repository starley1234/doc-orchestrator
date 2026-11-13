"""Шлюз движков: перевод контракта оркестратора в API штатных серверов и управление контейнерами GPU 0.

Серверы (docling-serve, MinerU, paddleocr-pdf-api) и Docker имитирует стенд из stock_engines.py.
"""

import asyncio
import base64
import io
import zipfile
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

import httpx
import pytest
from conftest import MOCK_URL, make_png
from fakeredis import FakeAsyncRedis
from fastapi import FastAPI
from stock_engines import MINERU, PADDLE, SAMPLE_TEXT, Stand, docker_client, engines_client, picture

from app.config import Settings
from app.main import create_app as create_orchestrator
from gateway.config import GatewaySettings
from gateway.main import create_app
from mocks.main import MERMAID

SCAN = b"%PDF-1.7 scan"  # содержимое стенду безразлично


@pytest.fixture
def stand() -> Stand:
    return Stand()


@pytest.fixture
def paddle_loaded(stand: Stand) -> None:
    """GPU 0 занят PaddleOCR-VL: его контейнер запущен, контейнер MinerU остановлен."""
    stand.containers.update({MINERU: "exited", PADDLE: "running"})


@asynccontextmanager
async def running_gateway(
    stand: Stand, http: httpx.AsyncClient | None = None, docker: httpx.AsyncClient | None = None, **overrides
) -> AsyncIterator[FastAPI]:
    """Работающий шлюз перед серверами стенда. overrides — настройки; http и docker — другие клиенты."""
    settings = GatewaySettings(**{"http_timeout_sec": 20, "poll_interval_sec": 0.01} | overrides)
    async with (
        http or engines_client(stand) as engines_http,
        docker or docker_client(stand) as docker_http,
    ):
        app = create_app(settings, http=engines_http, docker=docker_http)
        async with app.router.lifespan_context(app):
            yield app


@pytest.fixture
def start_gateway(stand: Stand):
    """Запуск шлюза: async with start_gateway(mineru_max_figures=2) as client: ..."""

    @asynccontextmanager
    async def start(**options) -> AsyncIterator[httpx.AsyncClient]:
        async with running_gateway(stand, **options) as app:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://engine-gateway") as client:
                yield client

    return start


@pytest.fixture
async def gateway(start_gateway) -> AsyncIterator[httpx.AsyncClient]:
    async with start_gateway() as client:
        yield client


async def send(client: httpx.AsyncClient, path: str, filename: str, data: bytes = SCAN) -> httpx.Response:
    return await client.post(path, files={"file": (filename, data)})


def failing(status: int = 0, body: dict | None = None) -> httpx.AsyncClient:
    """HTTP-клиент сервера, который отвечает одним и тем же; status=0 — соединение не устанавливается."""

    def handler(request: httpx.Request) -> httpx.Response:
        if not status:
            raise httpx.ConnectError("No such file or directory", request=request)
        return httpx.Response(status, json=body or {})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://docker")


# --- Docling ----------------------------------------------------------------------------------


async def test_docling_convert_returns_markdown(gateway, stand: Stand):
    response = await send(gateway, "/docling/convert", "Договор поставки № 17.pdf")

    assert response.status_code == 200
    assert response.json() == {"markdown": f"## Документ\n\n{SAMPLE_TEXT}"}
    [request] = stand.docling_requests
    assert request["filename"] == "document.pdf"  # расширение сохранено, имя — нейтральное
    assert request["content"] == SCAN
    assert request["options"] == {
        "to_formats": "md",
        "image_export_mode": "placeholder",
        "include_images": "false",
        "do_ocr": "false",
    }


async def test_docling_ocr_can_be_enabled(start_gateway, stand: Stand):
    async with start_gateway(docling_do_ocr=True) as gateway:
        await send(gateway, "/docling/convert", "report.docx")

    assert stand.docling_requests[0]["options"]["do_ocr"] == "true"


@pytest.mark.parametrize("encoding", ["utf-8", "cp1251", "utf-16"])
async def test_docling_gets_text_files_in_utf8(gateway, encoding: str):
    # Docling читает только UTF-8, а выгрузки из 1С приходят в Windows-1251
    text = "Остатки на складе: подшипник 6205-2RS — 10 шт."
    response = await send(gateway, "/docling/convert", "остатки.txt", text.encode(encoding))

    assert response.json() == {"markdown": text}


async def test_docling_failure_is_reported(gateway, stand: Stand):
    stand.docling_status = "failure"
    response = await send(gateway, "/docling/convert", "broken.docx")

    assert response.status_code == 502
    detail = response.json()["detail"]
    assert "status=failure" in detail
    assert "Input document is not valid" in detail


async def test_docling_http_error_is_reported(gateway, stand: Stand):
    stand.docling_http_error = 504  # документ не успел обработаться за DOCLING_SERVE_MAX_SYNC_WAIT
    response = await send(gateway, "/docling/convert", "big.pdf")

    assert response.status_code == 502
    detail = response.json()["detail"]
    assert detail.startswith("Docling: POST /v1/convert/file -> HTTP 504")
    assert "DOCLING_SERVE_MAX_SYNC_WAIT" in detail


async def test_docling_rejects_scan(gateway, stand: Stand):
    response = await send(gateway, "/docling/convert", "scan.jpg")

    assert response.status_code == 422
    assert response.json() == {"detail": "Docling не принимает файлы «.jpg»"}
    assert stand.docling_requests == []


# --- MinerU -----------------------------------------------------------------------------------


async def test_mineru_extract_returns_markdown_and_figures(gateway, stand: Stand):
    response = await send(gateway, "/mineru/extract", "Скан: лист 1.pdf")

    assert response.status_code == 200
    reply = response.json()
    assert reply["markdown"] == stand.mineru_markdown
    assert list(reply["figures"]) == ["images/scheme.jpg"]  # ключ — путь, как в Markdown
    assert base64.b64decode(reply["figures"]["images/scheme.jpg"]) == stand.mineru_images["images/scheme.jpg"]
    # Задача создана без лишних полей, файл загружен по выданной ссылке с её заголовками
    assert stand.calls == ["mineru POST /v1/tasks", "mineru PUT upload application/octet-stream"]
    [task] = stand.mineru_tasks.values()
    assert task["file_name"] == "document.pdf"  # двоеточие в имени MinerU отклонил бы
    assert task["content"] == SCAN


async def test_mineru_links_are_followed_on_own_host(gateway, stand: Stand):
    # За обратным прокси сервер называет себя в ссылках иначе, чем его видит шлюз
    stand.mineru_public_url = "https://ocr.example.org:8443/"
    response = await send(gateway, "/mineru/extract", "scan.pdf")

    assert response.status_code == 200
    assert response.json()["markdown"] == stand.mineru_markdown


async def test_mineru_gets_tif_as_tiff(gateway, stand: Stand):
    # MinerU знает расширение .tiff, но не .tif
    await send(gateway, "/mineru/extract", "scan.TIF")

    [task] = stand.mineru_tasks.values()
    assert task["file_name"] == "document.tiff"


async def test_mineru_rejects_office_file(gateway, stand: Stand):
    response = await send(gateway, "/mineru/extract", "report.docx")

    assert response.status_code == 422
    assert stand.calls == []


async def test_mineru_returns_only_figures_worth_describing(gateway, stand: Stand):
    stand.mineru_markdown = (
        "Текст.\n\n![](images/scheme.jpg)\n\n![логотип](images/logo.png)\n\n![](images/lost.jpg)\n\n"
        '![](images/broken.jpg)\n\n![](photo.jpg "подпись")\n\nКонец.'
    )
    stand.mineru_images = {
        "images/scheme.jpg": picture(640, 480),
        "images/logo.png": picture(60, 400, "PNG"),  # узкая: меньшая сторона меньше 100 px
        "images/broken.jpg": b"not an image",
        "images/photo.jpg": picture(300, 200),  # в Markdown путь без каталога
        "images/unused.jpg": picture(800, 600),  # в Markdown ссылки нет
    }
    reply = (await send(gateway, "/mineru/extract", "scan.pdf")).json()

    assert list(reply["figures"]) == ["images/scheme.jpg", "photo.jpg"]
    assert base64.b64decode(reply["figures"]["photo.jpg"]) == stand.mineru_images["images/photo.jpg"]
    # Ссылки на неотданные картинки убраны: в тексте не остаётся путей, по которым ничего нет
    assert (
        reply["markdown"]
        == 'Текст.\n\n![](images/scheme.jpg)\n\n\n\n\n\n\n\n![](photo.jpg "подпись")\n\nКонец.'
    )


@pytest.mark.parametrize(
    "limit, kept",
    [
        (2, ["images/b.jpg", "images/d.jpg"]),
        (0, ["images/a.jpg", "images/b.jpg", "images/c.jpg", "images/d.jpg"]),
    ],
)
async def test_mineru_keeps_largest_figures(start_gateway, stand: Stand, limit: int, kept: list[str]):
    sizes = {"a": (200, 200), "b": (500, 500), "c": (300, 300), "d": (900, 900)}
    stand.mineru_images = {f"images/{name}.jpg": picture(*size) for name, size in sizes.items()}
    stand.mineru_markdown = "".join(f"![](images/{name}.jpg)\n" for name in sizes)

    async with start_gateway(mineru_max_figures=limit) as gateway:
        reply = (await send(gateway, "/mineru/extract", "scan.pdf")).json()

    assert list(reply["figures"]) == kept  # самые крупные, в порядке появления в документе
    assert reply["markdown"] == "".join(
        f"![]({path})\n" if path in kept else "\n" for path in stand.mineru_images
    )


async def test_mineru_failed_task_is_reported(gateway, stand: Stand):
    stand.mineru_error = "CUDA out of memory"
    response = await send(gateway, "/mineru/extract", "scan.pdf")

    assert response.status_code == 502
    assert response.json() == {"detail": "MinerU не распознал документ: CUDA out of memory"}


async def test_mineru_timeout_cancels_task(start_gateway, stand: Stand):
    stand.mineru_polls = 10**9  # задача не завершается
    async with start_gateway(http_timeout_sec=0.3) as gateway:
        response = await send(gateway, "/mineru/extract", "scan.pdf")

    assert response.status_code == 504
    assert response.json() == {"detail": "MinerU не уложился в 0.27 с"}  # 90 % таймаута оркестратора
    assert stand.calls[-1] == "mineru POST cancel"
    assert next(iter(stand.mineru_tasks.values()))["status"] == "canceled"


async def test_mineru_stopped_container_is_reported(gateway, stand: Stand):
    stand.containers[MINERU] = "exited"
    response = await send(gateway, "/mineru/extract", "scan.pdf")

    assert response.status_code == 502
    assert response.json()["detail"].startswith("MinerU недоступен: ConnectError")


@pytest.mark.parametrize("archive_files", [None, {"middle_json.json": "{}"}])
async def test_mineru_broken_result_is_reported(gateway, stand: Stand, archive_files: dict | None):
    buffer = io.BytesIO(b"not a zip")
    if archive_files:
        with zipfile.ZipFile(buffer, "w") as archive:
            for name, text in archive_files.items():
                archive.writestr(name, text)
    stand.mineru_archive = buffer.getvalue()

    response = await send(gateway, "/mineru/extract", "scan.pdf")

    assert response.status_code == 502
    assert response.json()["detail"].startswith("MinerU вернул результат без markdown.md")


async def test_unexpected_engine_reply_is_reported(start_gateway):
    # Сервер ответил 200, но не тем, что описано в его API (например, сменилась версия)
    async with start_gateway(http=failing(200, {"unexpected": True})) as gateway:
        mineru = await send(gateway, "/mineru/extract", "scan.pdf")
        paddle = await send(gateway, "/paddle/extract", "scan.pdf")
        docling = await send(gateway, "/docling/convert", "report.pdf")

    assert mineru.status_code == paddle.status_code == docling.status_code == 502
    assert mineru.json() == {"detail": "MinerU вернул неожиданный ответ: KeyError 'task_id'"}
    assert paddle.json() == {"detail": "PaddleOCR-VL вернул неожиданный ответ: KeyError 'job_id'"}
    assert docling.json() == {"detail": "Docling вернул неожиданный ответ: KeyError 'status'"}


# --- PaddleOCR-VL -----------------------------------------------------------------------------


async def test_paddle_extract_joins_pages_in_order(gateway, stand: Stand, paddle_loaded):
    response = await send(gateway, "/paddle/extract", "УПД № 128.png")

    assert response.status_code == 200
    assert response.json() == {"markdown": "\n\n".join(stand.paddle_pages)}
    assert stand.calls == ["paddle POST /ocr document.png", "paddle DELETE job"]
    assert stand.paddle_jobs == {}  # сервер хранит задачи бессрочно: шлюз убирает за собой


async def test_paddle_failed_job_is_reported_and_deleted(gateway, stand: Stand, paddle_loaded):
    stand.paddle_error = "ResourceExhaustedError: Out of memory error on GPU 0"
    response = await send(gateway, "/paddle/extract", "scan.pdf")

    assert response.status_code == 502
    assert response.json() == {"detail": f"PaddleOCR-VL не распознал документ: {stand.paddle_error}"}
    assert stand.paddle_jobs == {}


async def test_paddle_timeout_cancels_job(start_gateway, stand: Stand, paddle_loaded):
    stand.paddle_polls = 10**9
    async with start_gateway(http_timeout_sec=0.3) as gateway:
        response = await send(gateway, "/paddle/extract", "scan.pdf")

    assert response.status_code == 504
    assert response.json() == {"detail": "PaddleOCR-VL не уложился в 0.27 с"}
    assert stand.calls == [
        "paddle POST /ocr document.pdf",
        "paddle POST cancel",
    ]  # работающую задачу не удаляем


async def test_paddle_rejects_unknown_format(gateway, stand: Stand):
    response = await send(gateway, "/paddle/extract", "scan.gif")

    assert response.status_code == 422
    assert response.json() == {"detail": "PaddleOCR-VL не принимает файлы «.gif»"}
    assert stand.calls == []


# --- GPU 0: load и unload — это запуск и остановка контейнеров -----------------------------------


async def test_load_starts_container_and_waits_for_engine(gateway, stand: Stand):
    stand.boot_probes = 3  # после запуска контейнера сервер ещё грузит модель

    response = await gateway.post("/paddle/load")

    assert response.status_code == 200
    assert response.json() == {"status": "loaded"}
    assert stand.containers[PADDLE] == "running"
    assert stand.booting[PADDLE] == 0  # ответ пришёл, когда сервер уже отвечал
    assert stand.docker_calls() == ["docker start paddle-gpu0"]


async def test_load_of_running_engine_changes_nothing(gateway, stand: Stand):
    response = await gateway.post("/mineru/load")

    assert response.status_code == 200
    assert stand.containers == {MINERU: "running", PADDLE: "created"}


async def test_load_reports_missing_container(gateway, stand: Stand):
    del stand.containers[PADDLE]
    response = await gateway.post("/paddle/load")

    assert response.status_code == 502
    assert response.json()["detail"].startswith("Контейнер paddle-gpu0 не создан")


async def test_load_reports_crashed_container(gateway, stand: Stand):
    stand.crash_on_start = {PADDLE}  # процесс упал при старте: модели не хватило видеопамяти
    response = await gateway.post("/paddle/load")

    assert response.status_code == 502
    assert response.json() == {
        "detail": "Контейнер paddle-gpu0 остановился при запуске: смотрите docker logs paddle-gpu0"
    }


async def test_load_times_out_when_engine_does_not_start_listening(start_gateway, stand: Stand):
    stand.containers[MINERU] = "exited"
    stand.boot_probes = 10**9

    async with start_gateway(http_timeout_sec=0.3) as gateway:
        response = await gateway.post("/mineru/load")

    assert response.status_code == 504
    assert response.json() == {"detail": "MinerU не поднялся за 0.27 с: смотрите docker logs mineru-api"}


async def test_load_restarts_engine_that_reports_failure(gateway, stand: Stand):
    # У MinerU упал внутренний движок: контейнер работает, сервер отвечает 503 и сам не оправится
    stand.mineru_engine_dead = "VLM engine exited unexpectedly"
    stand.boot_probes = 2

    response = await gateway.post("/mineru/load")

    assert response.status_code == 200
    assert stand.docker_calls() == [
        "docker start mineru-api",  # контейнер и так запущен
        "docker stop mineru-api t=30",
        "docker start mineru-api",
    ]
    assert stand.containers[MINERU] == "running"
    assert stand.booting[MINERU] == 0  # ответ пришёл, когда перезапущенный сервер уже отвечал


async def test_load_restarts_broken_engine_only_once(gateway, stand: Stand):
    stand.mineru_preload_error = "CUDA out of memory"  # модели не загружаются и после перезапуска

    response = await gateway.post("/mineru/load")

    assert response.status_code == 502
    assert response.json() == {
        "detail": "MinerU и после перезапуска сообщает о сбое: HTTP 503: "
        '{"error":{"code":"model_preload_failed","message":"CUDA out of memory"}}'
    }
    assert stand.docker_calls() == [
        "docker start mineru-api",
        "docker stop mineru-api t=30",
        "docker start mineru-api",
    ]


async def test_unload_stops_container(gateway, stand: Stand):
    response = await gateway.post("/mineru/unload")

    assert response.status_code == 200
    assert response.json() == {"status": "unloaded"}
    assert stand.containers[MINERU] == "exited"
    assert stand.docker_calls() == ["docker stop mineru-api t=30"]  # Docker ждёт завершения процесса


@pytest.mark.parametrize("state", ["created", "exited", "missing"])
async def test_unload_of_stopped_engine_is_not_an_error(gateway, stand: Stand, state: str):
    if state == "missing":
        del stand.containers[PADDLE]
    else:
        stand.containers[PADDLE] = state

    assert (await gateway.post("/paddle/unload")).status_code == 200


async def test_unknown_gpu_engine_is_rejected(gateway, stand: Stand):
    assert (await gateway.post("/docling/load")).status_code == 422
    assert (await gateway.post("/qwen/unload")).status_code == 422
    assert stand.calls == []


async def test_missing_docker_socket_is_reported(start_gateway):
    async with start_gateway(docker=failing()) as gateway:
        response = await gateway.post("/mineru/unload")

    assert response.status_code == 502
    detail = "Docker недоступен (ConnectError): в контейнер шлюза должен быть проброшен /var/run/docker.sock"
    assert response.json() == {"detail": detail}


async def test_docker_error_is_reported(start_gateway):
    async with start_gateway(docker=failing(500, {"message": "driver failed programming"})) as gateway:
        load = await gateway.post("/paddle/load")
        unload = await gateway.post("/paddle/unload")
        state = await gateway.get("/state")

    assert load.status_code == unload.status_code == state.status_code == 502
    assert load.json()["detail"].startswith("Docker: POST /containers/paddle-gpu0/start -> HTTP 500")
    assert "driver failed programming" in load.json()["detail"]
    assert unload.json()["detail"].startswith("Docker: POST /containers/paddle-gpu0/stop -> HTTP 500")


# --- Служебное ---------------------------------------------------------------------------------


async def test_state_shows_containers_and_engines(gateway, stand: Stand):
    assert (await gateway.get("/state")).json() == {
        "docling": {"ready": True},
        "mineru": {"container": "running", "ready": True},
        "paddle": {"container": "created", "ready": False},
    }

    stand.mineru_preload_error = "CUDA out of memory"  # сервер MinerU отвечает, но сообщает о сбое
    assert (await gateway.get("/state")).json()["mineru"] == {"container": "running", "ready": False}

    del stand.containers[PADDLE]
    assert (await gateway.get("/state")).json()["paddle"]["container"] == "missing"


async def test_health(gateway):
    response = await gateway.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# --- Боевая схема целиком: оркестратор -> шлюз -> штатные серверы ---------------------------------

GATEWAY_URL = "http://engine-gateway"
PNG = {"X-Filename": "scan.png"}
PADDLE_PNG = PNG | {"X-OCR-Engine": "paddle"}


@pytest.fixture
def start_chain(stand: Stand, settings: Settings, redis: FakeAsyncRedis, mock_app: FastAPI):
    """Оркестратор с адресами шлюза вместо mock-engines. Qwen3-VL (Ollama) по-прежнему отвечает из мока."""

    @asynccontextmanager
    async def start(**overrides) -> AsyncIterator[httpx.AsyncClient]:
        engine_urls = {name: f"{GATEWAY_URL}/{name}" for name in ("docling", "mineru", "paddle")}
        chain_settings = settings.model_copy(
            update={f"{name}_url": url for name, url in engine_urls.items()} | overrides
        )
        async with running_gateway(stand) as gateway_app:
            hosts = {"engine-gateway": gateway_app, httpx.URL(MOCK_URL).host: mock_app}
            async with engines_client(stand, extra=hosts) as http:
                app = create_orchestrator(chain_settings, redis=redis, http=http)
                transport = httpx.ASGITransport(app=app)
                async with (
                    app.router.lifespan_context(app),
                    httpx.AsyncClient(transport=transport, base_url="http://doc-orchestrator") as client,
                ):
                    yield client

    return start


@pytest.fixture
async def chain(start_chain) -> AsyncIterator[httpx.AsyncClient]:
    async with start_chain() as client:
        yield client


async def wait_until(condition: Callable[[], bool]) -> None:
    """Ждёт (до 60 секунд), пока условие не выполнится."""
    for _ in range(6000):
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("Условие не выполнилось за 60 секунд")


async def test_chain_scan_is_read_by_mineru_and_figures_by_vlm(chain, stand: Stand):
    response = await chain.put("/process", content=make_png(200, 200), headers=PNG)

    assert response.status_code == 200
    body = response.json()
    assert body["metadata"] == {
        "filename": "scan.png",
        "route": "SCAN_IMAGE",
        "engine_used": "mineru+qwen3-vl",
    }
    # Картинку, вырезанную MinerU, Qwen3-VL заменил описанием: ссылки на неё в тексте больше нет
    assert body["page_content"] == stand.mineru_markdown.replace("![](images/scheme.jpg)", MERMAID)
    assert SAMPLE_TEXT in body["page_content"]


async def test_chain_digital_document_is_read_by_docling(chain, stand: Stand, digital_pdf: bytes):
    response = await chain.put("/process", content=digital_pdf, headers={"X-Filename": "contract.pdf"})

    assert response.status_code == 200
    assert response.json()["metadata"]["engine_used"] == "docling"
    assert response.json()["page_content"] == f"## Документ\n\n{SAMPLE_TEXT}"
    assert stand.docker_calls() == []  # Docling работает на CPU: контейнеры GPU 0 не трогаем


async def test_chain_switches_gpu0_by_restarting_containers(chain, stand: Stand):
    await chain.put("/process", content=make_png(200, 200), headers=PNG)
    response = await chain.put("/process", content=make_png(200, 200), headers=PADDLE_PNG)

    assert response.status_code == 200
    assert response.json()["metadata"]["engine_used"] == "paddle"
    assert response.json()["page_content"] == "\n\n".join(stand.paddle_pages)
    assert stand.containers == {MINERU: "exited", PADDLE: "running"}
    assert stand.docker_calls() == [
        "docker stop paddle-gpu0 t=30",  # первая задача: в VRAM не должно быть ничего, кроме MinerU
        "docker start mineru-api",
        "docker stop mineru-api t=30",  # вторая: сначала освобождаем VRAM, потом запускаем PaddleOCR-VL
        "docker start paddle-gpu0",
    ]
    assert stand.vram_conflicts == 0


async def test_chain_returns_gpu0_to_mineru_after_idle(start_chain, stand: Stand):
    stand.boot_probes = 3  # контейнеры поднимаются не мгновенно
    async with start_chain(paddle_idle_ttl_sec=0.2) as chain:
        response = await chain.put("/process", content=make_png(200, 200), headers=PADDLE_PNG)
        assert response.status_code == 200
        assert stand.containers == {MINERU: "exited", PADDLE: "running"}

        await wait_until(lambda: stand.containers == {MINERU: "running", PADDLE: "exited"})

    assert stand.vram_conflicts == 0


async def test_chain_scanned_upd_is_read_by_paddle_and_validated(chain, stand: Stand, scan_pdf: bytes):
    response = await chain.post(
        "/api/extract-structured", files={"file": ("upd.pdf", scan_pdf)}, data={"doc_type": "upd"}
    )

    assert response.status_code == 200
    assert response.json()["validation"] == {"is_valid": True, "warnings": []}
    assert stand.containers == {MINERU: "exited", PADDLE: "running"}  # скан УПД читает PaddleOCR-VL
    assert stand.paddle_jobs == {}


async def test_chain_reports_engine_failure(chain, stand: Stand):
    stand.mineru_error = "CUDA out of memory"
    response = await chain.put("/process", content=make_png(200, 200), headers=PNG)

    assert response.status_code == 502
    assert response.json() == {
        "detail": 'MinerU вернул HTTP 502: {"detail":"MinerU не распознал документ: CUDA out of memory"}'
    }


async def test_chain_starts_mineru_when_nothing_is_running(chain, stand: Stand):
    # Сервер перезагрузили, пока оба контейнера были остановлены
    stand.containers.update({MINERU: "exited", PADDLE: "exited"})
    stand.boot_probes = 5

    response = await chain.put("/process", content=make_png(200, 200), headers=PNG)

    assert response.status_code == 200
    assert stand.containers == {MINERU: "running", PADDLE: "exited"}


async def test_chain_restarts_mineru_after_its_engine_died(chain, stand: Stand):
    assert (await chain.put("/process", content=make_png(200, 200), headers=PNG)).status_code == 200
    stand.mineru_engine_dead = "VLM engine exited unexpectedly"  # сервер работает, но распознавать не может

    failed = await chain.put("/process", content=make_png(200, 200), headers=PNG)
    retried = await chain.put("/process", content=make_png(200, 200), headers=PNG)

    assert failed.status_code == 502
    assert "engine_dead" in failed.json()["detail"]
    assert retried.status_code == 200  # перед следующей задачей шлюз перезапустил контейнер
    assert stand.docker_calls()[-2:] == ["docker stop mineru-api t=30", "docker start mineru-api"]
    assert stand.vram_conflicts == 0


async def test_chain_recovers_when_container_was_stopped_behind_its_back(chain, stand: Stand):
    assert (await chain.put("/process", content=make_png(200, 200), headers=PNG)).status_code == 200
    stand.containers[MINERU] = "exited"  # администратор остановил контейнер вручную

    failed = await chain.put("/process", content=make_png(200, 200), headers=PNG)
    retried = await chain.put("/process", content=make_png(200, 200), headers=PNG)

    assert failed.status_code == 502  # по данным оркестратора MinerU был загружен
    assert "MinerU недоступен" in failed.json()["detail"]
    assert retried.status_code == 200  # после сбоя оркестратор загружает модель заново
    assert stand.containers[MINERU] == "running"
