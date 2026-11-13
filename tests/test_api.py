"""HTTP-контракт сервиса: заголовки и API-ключ, коды ошибок, ответы модели не по схеме, async_mode."""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote

import httpx
import pytest
from conftest import engines_stub
from fakeredis import FakeAsyncRedis, FakeServer

from app.config import Settings
from app.main import create_app
from mocks.main import DOCUMENT, MockState

ROOT = Path(__file__).parent.parent  # корень репозитория
PDF = {"X-Filename": "scan.pdf"}


async def extract(client: httpx.AsyncClient, filename: str, data: bytes, **form: str) -> httpx.Response:
    return await client.post("/api/extract-structured", files={"file": (filename, data)}, data=form)


async def wait_for_task(client: httpx.AsyncClient, task_id: str) -> dict:
    """Опрашивает задачу, пока она не завершится."""
    for _ in range(500):
        task = (await client.get(f"/api/tasks/{task_id}")).json()
        if task["status"] in ("completed", "failed"):
            return task
        await asyncio.sleep(0.02)
    raise AssertionError(f"Задача не завершилась: {task}")


# --- PUT /process: заголовки и API-ключ ------------------------------------------------------


async def test_filename_is_percent_decoded(client, digital_pdf: bytes):
    # Open WebUI кодирует имя файла: quote("Счёт на оплату 57.pdf")
    headers = {"X-Filename": quote("Счёт на оплату 57.pdf")}
    response = await client.put("/process", content=digital_pdf, headers=headers)

    assert response.json()["metadata"]["filename"] == "Счёт на оплату 57.pdf"
    assert "Счёт на оплату 57.pdf" in response.json()["page_content"]  # с этим именем файл ушёл в движок


@pytest.mark.parametrize("sent_name", ["../../etc/report.pdf", "C:\\Документы\\report.pdf"])
async def test_filename_path_is_dropped(client, digital_pdf: bytes, sent_name: str):
    response = await client.put(
        "/process", content=digital_pdf, headers={"X-Filename": quote(sent_name, safe="")}
    )
    assert response.json()["metadata"]["filename"] == "report.pdf"


async def test_extension_is_taken_from_content_type(client, digital_pdf: bytes):
    headers = {"X-Filename": "document", "Content-Type": "application/pdf; charset=binary"}
    response = await client.put("/process", content=digital_pdf, headers=headers)

    assert response.status_code == 200
    assert response.json()["metadata"] == {
        "filename": "document.pdf",
        "route": "DIGITAL_PDF",
        "engine_used": "docling",
    }


async def test_missing_filename_header_is_rejected(client, scan_pdf: bytes):
    response = await client.put("/process", content=scan_pdf)
    assert response.status_code == 422


async def test_api_key_is_not_required_by_default(client, digital_pdf: bytes):
    headers = {"X-Filename": "report.pdf", "Authorization": "Bearer anything"}
    assert (await client.put("/process", content=digital_pdf, headers=headers)).status_code == 200


@pytest.mark.parametrize(
    ("authorization", "expected_status"),
    [(None, 401), ("Bearer wrong", 401), ("s3cret", 401), ("Bearer s3cret", 200)],
)
async def test_api_key_is_checked_when_configured(
    start_app, digital_pdf: bytes, authorization, expected_status
):
    headers = {"X-Filename": "report.pdf"} | ({"Authorization": authorization} if authorization else {})

    async with start_app(api_key="s3cret") as client:
        response = await client.put("/process", content=digital_pdf, headers=headers)
        health = await client.get("/health")

    assert response.status_code == expected_status
    assert health.status_code == 200  # проверка живости ключа не требует


# --- Ошибки ---------------------------------------------------------------------------------


async def test_broken_pdf_is_422(client, mock: MockState):
    response = await client.put("/process", content=b"%PDF-1.7 broken", headers={"X-Filename": "broken.pdf"})

    assert response.status_code == 422
    assert response.json()["detail"].startswith("Не удалось прочитать PDF")
    assert mock.calls == []  # до движков файл не дошёл


async def test_empty_file_is_422(client):
    response = await client.put("/process", content=b"", headers=PDF)
    assert (response.status_code, response.json()) == (422, {"detail": "Файл пуст"})


async def test_too_large_file_is_413(start_app):
    megabyte = 1024 * 1024
    async with start_app(max_file_size_mb=1) as client:
        fits = await client.put("/process", content=b"x" * megabyte, headers={"X-Filename": "big.docx"})
        too_large = await client.put(
            "/process", content=b"x" * (megabyte + 1), headers={"X-Filename": "big.docx"}
        )
        structured = await extract(client, "big.docx", b"x" * (megabyte + 1), doc_type="upd")

    assert fits.status_code == 200
    assert (too_large.status_code, too_large.json()) == (413, {"detail": "Файл больше 1 МБ"})
    assert structured.status_code == 413


async def test_engine_failure_is_502(start_app, digital_pdf: bytes):
    async with start_app(http=engines_stub(failing="docling")) as client:
        response = await client.put("/process", content=digital_pdf, headers={"X-Filename": "report.pdf"})

    assert response.status_code == 502
    assert response.json() == {"detail": "Docling вернул HTTP 500: engine crashed"}


async def test_service_recovers_after_engine_restart(client, scan_pdf: bytes, mock: MockState):
    assert (await client.put("/process", content=scan_pdf, headers=PDF)).status_code == 200
    mock.loaded.clear()  # движок перезапустился: модель из VRAM пропала

    failed = await client.put("/process", content=scan_pdf, headers=PDF)
    recovered = await client.put("/process", content=scan_pdf, headers=PDF)

    assert (failed.status_code, failed.json()["detail"]) == (
        502,
        'MinerU вернул HTTP 500: {"detail":"MinerU not loaded"}',
    )
    assert recovered.status_code == 200  # следующая задача загрузила модель заново


async def test_busy_gpu_is_503(start_app, scan_pdf: bytes, redis: FakeAsyncRedis):
    await redis.set("lock:gpu0", "занято другой задачей")

    async with asyncio.timeout(60), start_app(gpu_lock_wait_sec=0.2) as client:  # ждём очередь не вечно
        response = await client.put("/process", content=scan_pdf, headers=PDF)

    assert response.status_code == 503
    assert response.json() == {"detail": "Видеокарта занята: очередь (lock:gpu0) не подошла за 0.2 с"}


async def test_redis_failure_is_503(client, scan_pdf: bytes, redis_server: FakeServer):
    redis_server.connected = False
    process = await client.put("/process", content=scan_pdf, headers=PDF)
    health = await client.get("/health")
    redis_server.connected = True

    assert process.status_code == health.status_code == 503
    assert health.json() == {"detail": "Redis недоступен"}
    assert (await client.get("/health")).json() == {"status": "ok"}


# --- POST /api/extract-structured: проверка запроса -------------------------------------------


@pytest.mark.parametrize("doc_type", ["act", "contract", ""])
async def test_unknown_doc_type_is_422(client, scan_pdf: bytes, doc_type: str):
    assert (await extract(client, "act.pdf", scan_pdf, doc_type=doc_type)).status_code == 422


async def test_missing_form_fields_are_422(client, scan_pdf: bytes):
    assert (await client.post("/api/extract-structured", data={"doc_type": "upd"})).status_code == 422
    assert (await extract(client, "upd.pdf", scan_pdf)).status_code == 422


@pytest.mark.parametrize("sent_name", ["../../etc/upd.pdf", "C:\\Документы\\upd.pdf"])
async def test_uploaded_filename_path_is_dropped(client, digital_pdf: bytes, mock: MockState, sent_name: str):
    await extract(client, sent_name, digital_pdf, doc_type="upd")

    assert "Извлеки данные документа «upd.pdf»" in mock.prompts[0]


async def test_binary_file_of_unknown_format_is_415(client):
    response = await extract(client, "upd.zip", b"PK\x03\x04\x00\x00", doc_type="upd")
    assert (response.status_code, response.json()) == (
        415,
        {"detail": "Формат файла «upd.zip» не поддерживается"},
    )


@pytest.mark.parametrize("filename", ["shaft.dwg", "shaft.docx", "shaft.png"])
async def test_drawing_must_be_pdf_or_image(client, mock: MockState, filename: str):
    response = await extract(client, filename, b"AC1027 not an image", doc_type="drawing")

    assert response.status_code == 422
    assert response.json() == {"detail": "Файл не является ни PDF, ни изображением"}
    assert mock.prompts == []


# --- Ответ модели не по схеме: это предупреждения, а не сбой -------------------------------


async def test_answer_with_missing_fields(start_app, digital_pdf: bytes):
    async with start_app(http=engines_stub('{"number": "128", "date": "2025-03-14"}')) as client:
        response = await extract(client, "upd.pdf", digital_pdf, doc_type="upd")

    assert response.status_code == 200
    body = response.json()
    assert body["data"] == {"number": "128", "date": "2025-03-14"}  # данные как есть
    assert body["validation"] == {
        "is_valid": False,
        "warnings": [
            "Поле seller: обязательное поле не заполнено",
            "Поле buyer: обязательное поле не заполнено",
            "Поле items: обязательное поле не заполнено",
            "Поле total_with_vat: обязательное поле не заполнено",
        ],
    }


async def test_answer_with_wrong_value_type(start_app, digital_pdf: bytes):
    items = [{**DOCUMENT["items"][0], "qty": "десять"}]
    async with start_app(http=engines_stub(json.dumps({**DOCUMENT, "items": items}))) as client:
        body = (await extract(client, "upd.pdf", digital_pdf, doc_type="upd")).json()

    assert body["validation"]["is_valid"] is False
    (warning,) = body["validation"]["warnings"]
    assert warning.startswith("Поле items.0.qty: некорректное значение")
    assert body["data"]["items"][0]["qty"] == "десять"


@pytest.mark.parametrize(
    "answer", ["Извините, документ не читается", "[1, 2, 3]", '{"number": "128"', "null"]
)
async def test_answer_that_is_not_a_json_object(start_app, digital_pdf: bytes, answer: str):
    async with start_app(http=engines_stub(answer)) as client:
        response = await extract(client, "upd.pdf", digital_pdf, doc_type="upd")

    assert response.status_code == 200
    assert response.json() == {
        "status": "completed",
        "doc_type": "upd",
        "data": {},
        "validation": {
            "is_valid": False,
            "warnings": ["Модель вернула не JSON-объект: данные извлечь не удалось"],
        },
    }


async def test_bad_kpp_bik_and_row_math_are_reported(start_app, digital_pdf: bytes):
    broken = {
        **DOCUMENT,
        "seller": {**DOCUMENT["seller"], "kpp": "77070100"},  # 8 символов вместо 9
        "buyer": {**DOCUMENT["buyer"], "inn": "502415678"},  # 9 цифр
        "bank": {**DOCUMENT["bank"], "bik": "144525225"},  # начинается не с 04
        "items": [{**DOCUMENT["items"][0], "amount_without_vat": 2600.0}, *DOCUMENT["items"][1:]],
    }
    async with start_app(http=engines_stub(json.dumps(broken))) as client:
        body = (await extract(client, "invoice.pdf", digital_pdf, doc_type="invoice")).json()

    assert body["validation"] == {
        "is_valid": False,
        "warnings": [
            "КПП продавца 77070100 имеет неверный формат",
            "ИНН покупателя 502415678 не прошёл контрольную сумму",
            "БИК банка 144525225 имеет неверный формат",
            "Строка 1: кол-во × цена (2500.00) != сумма без НДС (2600.00)",
            "Строка 1: сумма без НДС + НДС (3100.00) != сумма по строке (3000.00)",
        ],
    }


async def test_individual_entrepreneur_has_no_kpp(start_app, digital_pdf: bytes):
    seller = {"name": "ИП Иванов И. И.", "inn": "771234567859"}  # ИП: ИНН из 12 цифр, КПП нет
    async with start_app(http=engines_stub(json.dumps({**DOCUMENT, "seller": seller}))) as client:
        body = (await extract(client, "upd.pdf", digital_pdf, doc_type="upd")).json()

    assert body["validation"] == {"is_valid": True, "warnings": []}
    assert body["data"]["seller"] == {
        "name": "ИП Иванов И. И.",
        "inn": "771234567859",
        "kpp": None,
        "address": None,
    }


async def test_drawing_answers_that_do_not_match_schema(start_app, drawing_a1: bytes):
    answers = {
        "DrawingStampCore": json.dumps({"title": "Вал приводной"}),  # нет обязательного обозначения
        "TechRequirementsList": json.dumps({"requirements": "не список"}),
    }
    async with start_app(http=engines_stub(vlm=answers)) as client:
        response = await extract(client, "shaft.png", drawing_a1, doc_type="drawing")

    assert response.status_code == 200
    body = response.json()
    assert body["data"] == {"title": "Вал приводной", "requirements": "не список"}  # как есть
    assert body["validation"]["is_valid"] is False
    stamp_warning, requirements_warning = body["validation"]["warnings"]
    assert stamp_warning == "Поле designation: обязательное поле не заполнено"
    assert requirements_warning.startswith("Поле requirements: некорректное значение")


async def test_drawing_without_tech_requirements_answer(start_app, drawing_a1: bytes):
    answers = {
        "DrawingStampCore": json.dumps({"designation": "АБВГ.1", "title": "Вал"}),
        "TechRequirementsList": "нет",
    }
    async with start_app(http=engines_stub(vlm=answers)) as client:
        body = (await extract(client, "shaft.png", drawing_a1, doc_type="drawing")).json()

    assert body["data"] == {"designation": "АБВГ.1", "title": "Вал"}  # что успели прочитать
    assert body["validation"] == {
        "is_valid": False,
        "warnings": ["Модель вернула не JSON-объект: данные извлечь не удалось"],
    }


# --- Асинхронный режим ------------------------------------------------------------------------


def blocked_vlm() -> tuple[httpx.AsyncClient, asyncio.Event, list[str]]:
    """Движки, у которых Qwen3-VL молчит, пока тест не разрешит: клиент, «шлагбаум» и журнал запросов."""
    release, started = asyncio.Event(), []

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/api/chat"):
            started.append(json.loads(request.content)["messages"][0]["content"])
            await release.wait()
            return httpx.Response(200, json={"message": {"content": json.dumps(DOCUMENT)}})
        return httpx.Response(200, json={"markdown": "Текст документа"})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler)), release, started


async def statuses(client: httpx.AsyncClient, task_ids: list[str]) -> list[str]:
    return [(await client.get(f"/api/tasks/{task_id}")).json()["status"] for task_id in task_ids]


async def wait_for_statuses(client: httpx.AsyncClient, task_ids: list[str], expected: list[str]) -> None:
    """Ждёт (до 20 секунд), пока задачи придут в ожидаемые состояния, и проверяет, что они в них остаются."""
    for _ in range(2000):
        if await statuses(client, task_ids) == expected:
            break
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.1)
    assert await statuses(client, task_ids) == expected


async def test_async_mode_returns_task_id_then_result(client, scan_pdf: bytes, redis: FakeAsyncRedis):
    response = await extract(client, "upd.pdf", scan_pdf, doc_type="upd", async_mode="true")

    assert response.status_code == 202
    queued = response.json()
    assert set(queued) == {"task_id", "status"}
    assert queued["status"] == "queued"
    assert 3590 < await redis.ttl(f"task:{queued['task_id']}") <= 3600  # состояние хранится час

    task = await wait_for_task(client, queued["task_id"])
    assert set(task) == {"task_id", "status", "doc_type", "data", "validation"}
    assert (task["task_id"], task["status"], task["doc_type"]) == (queued["task_id"], "completed", "upd")
    assert task["validation"] == {"is_valid": True, "warnings": []}
    assert task["data"]["seller"]["inn"] == "7707083893"
    assert 3590 < await redis.ttl(f"task:{queued['task_id']}") <= 3600  # результат — тоже


async def test_async_task_honors_has_seals(client, scan_pdf: bytes, mock: MockState):
    queued = (
        await extract(client, "invoice.pdf", scan_pdf, doc_type="invoice", has_seals="1", async_mode="1")
    ).json()

    await wait_for_task(client, queued["task_id"])

    assert "paddle/extract" in mock.calls


async def test_async_task_with_bad_data_completes_with_warnings(client, scan_pdf: bytes):
    queued = (await extract(client, "bad_inn.pdf", scan_pdf, doc_type="upd", async_mode="true")).json()

    task = await wait_for_task(client, queued["task_id"])

    assert task["status"] == "completed"
    assert task["validation"]["is_valid"] is False
    assert len(task["validation"]["warnings"]) == 2


async def test_async_task_failure_is_reported(client):
    queued = (await extract(client, "upd.pdf", b"%PDF-1.7 broken", doc_type="upd", async_mode="true")).json()

    task = await wait_for_task(client, queued["task_id"])

    assert set(task) == {"task_id", "status", "doc_type", "error"}
    assert task["status"] == "failed"
    assert task["error"].startswith("Не удалось прочитать PDF")


async def test_unexpected_error_in_async_task_does_not_leak_details(start_app, digital_pdf: bytes):
    def crash(request: httpx.Request) -> httpx.Response:
        raise RuntimeError("секретные подробности")

    async with start_app(http=httpx.AsyncClient(transport=httpx.MockTransport(crash))) as client:
        queued = (await extract(client, "upd.pdf", digital_pdf, doc_type="upd", async_mode="true")).json()
        task = await wait_for_task(client, queued["task_id"])

    assert (task["status"], task["error"]) == ("failed", "Внутренняя ошибка сервиса")


async def test_async_tasks_wait_for_a_free_slot(start_app, digital_pdf: bytes):
    http, release, started = blocked_vlm()

    async with start_app(http=http, task_concurrency=2) as client:
        task_ids = [
            (await extract(client, f"upd_{n}.pdf", digital_pdf, doc_type="upd", async_mode="true")).json()[
                "task_id"
            ]
            for n in range(5)
        ]
        # Работают две задачи, остальным свободного места пока нет
        await wait_for_statuses(client, task_ids, ["processing", "processing", "queued", "queued", "queued"])
        for _ in range(2000):  # ждём (до 20 секунд), пока первая задача дойдёт до Qwen3-VL
            if started:
                break
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.1)
        assert len(started) == 1  # вторая работающая задача ждёт своей очереди к GPU 1

        release.set()
        results = [await wait_for_task(client, task_id) for task_id in task_ids]
        assert [task["status"] for task in results] == ["completed"] * 5
        await asyncio.sleep(0.05)  # даём циклу событий убрать завершённые задачи
        assert client._transport.app.state.background_tasks == set()  # в памяти они не копятся


async def test_async_tasks_are_marked_failed_when_service_stops(
    start_app, digital_pdf: bytes, redis: FakeAsyncRedis
):
    http, _, _ = blocked_vlm()

    # Таймаут — чтобы сломанная остановка сервиса провалила тест, а не подвесила его
    async with asyncio.timeout(60), start_app(http=http, task_concurrency=1) as client:
        task_ids = [
            (await extract(client, "upd.pdf", digital_pdf, doc_type="upd", async_mode="true")).json()[
                "task_id"
            ]
            for _ in range(2)
        ]
        await wait_for_statuses(client, task_ids, ["processing", "queued"])
    # Сервис остановлен: и работавшая, и ждавшая задачи завершены с понятной причиной

    for task_id in task_ids:
        task = json.loads(await redis.get(f"task:{task_id}"))
        assert task["status"] == "failed"
        assert task["error"] == "Сервис был остановлен: отправьте документ повторно"


async def test_async_tasks_share_gpu_queue_with_sync_requests(
    client, scan_pdf: bytes, drawing_a1: bytes, mock
):
    files = [
        ("upd.pdf", scan_pdf, "upd"),
        ("shaft.png", drawing_a1, "drawing"),
        ("invoice.pdf", scan_pdf, "invoice"),
    ]
    queued = [
        (await extract(client, name, data, doc_type=kind, async_mode="true")).json()
        for name, data, kind in files
    ]
    sync = await asyncio.gather(*(client.put("/process", content=scan_pdf, headers=PDF) for _ in range(3)))

    tasks = [await wait_for_task(client, task["task_id"]) for task in queued]

    assert [response.status_code for response in sync] == [200] * 3
    assert [task["status"] for task in tasks] == ["completed"] * 3
    assert [task["doc_type"] for task in tasks] == ["upd", "drawing", "invoice"]
    assert (mock.oom_errors, mock.gpu1_rejects, mock.gpu0_peak) == (0, 0, 1)


@pytest.mark.parametrize("task_id", ["00000000-0000-4000-8000-000000000000", "no-such-task"])
async def test_unknown_task_is_404(client, task_id: str):
    response = await client.get(f"/api/tasks/{task_id}")

    assert response.status_code == 404
    assert response.json() == {"detail": "Задача не найдена или срок её хранения истёк"}


# --- Настройки и служебное --------------------------------------------------------------------


def test_env_example_lists_every_setting():
    lines = (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    names = {line.partition("=")[0].lower() for line in lines if line and not line.startswith("#")}

    assert names == set(Settings.model_fields)
    Settings(_env_file=ROOT / ".env.example")  # все значения образца проходят проверку типов


def test_default_limits_match_the_spec():
    defaults = Settings(_env_file=None)

    assert (defaults.max_pixels, defaults.overview_dpi, defaults.drawing_dpi) == (1_280_000, 150, 300)
    assert (defaults.task_ttl_sec, defaults.api_key) == (3600, "")
    assert defaults.redis_url == "redis://redis:6379/0"  # адреса по умолчанию — контейнеры docker compose
    assert defaults.docling_url == "http://mock-engines:8001/docling"


def test_service_starts_with_settings_from_environment():
    # Так сервис запускает uvicorn: импорт app.main собирает приложение из переменных окружения
    env = {**os.environ, "LOG_LEVEL": "debug", "PADDLE_URL": "http://paddle.prod", "API_KEY": "s3cret"}
    code = "from app.main import app; s = app.state.settings; print(s.paddle_url, s.api_key)"
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True)

    assert result.stdout.strip() == "http://paddle.prod s3cret", result.stderr


async def test_own_connections_are_configured_from_settings(settings: Settings):
    # Без готовых клиентов (как в бою) сервис сам открывает подключения к Redis и движкам и сам их закрывает
    custom = {
        "redis_url": "redis://redis.test:6380/5",
        "http_timeout_sec": 123,
        "gpu0_check_interval_sec": 60,
    }
    app = create_app(settings.model_copy(update=custom))

    async with app.router.lifespan_context(app):
        redis_options = app.state.redis.connection_pool.connection_kwargs
        http = app.state.markdown_pipeline._engines.docling._http
        assert (redis_options["host"], redis_options["port"], redis_options["db"]) == ("redis.test", 6380, 5)
        assert http.timeout == httpx.Timeout(123)

    assert http.is_closed


async def test_background_watcher_stops_with_the_service(start_app):
    def watchers() -> list[asyncio.Task]:
        return [task for task in asyncio.all_tasks() if task.get_name() == "gpu0-idle-watcher"]

    async with start_app():
        assert len(watchers()) == 1
    assert watchers() == []


async def test_openapi_describes_all_endpoints(client):
    paths = (await client.get("/openapi.json")).json()["paths"]

    assert {"/process", "/api/extract-structured", "/api/tasks/{task_id}", "/health"} <= set(paths)
    assert "put" in paths["/process"]


async def test_engine_urls_and_model_come_from_settings(
    start_app, mock_app, scan_pdf: bytes, drawing_a1: bytes
):
    # Каждому движку задаём свой хост и проверяем, что запросы уходят именно туда, куда велят настройки
    seen: set[str] = set()
    models: set[tuple[str, int]] = set()
    mock_transport = httpx.ASGITransport(app=mock_app)

    class Recorder(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            seen.add(f"{request.url.host}{request.url.path}")
            if request.url.path == "/api/chat":
                body = json.loads(request.content)
                models.add((body["model"], body["options"]["num_ctx"]))
            return await mock_transport.handle_async_request(request)

    hosts = {
        "docling_url": "http://docling.prod/docling",
        "mineru_url": "http://mineru.prod/mineru",
        "paddle_url": "http://paddle.prod/paddle",
        "ollama_url": "http://ollama.prod",
        "vlm_model": "qwen3-vl:8b-instruct",
        "vlm_num_ctx": 4096,
    }
    async with start_app(http=httpx.AsyncClient(transport=Recorder()), **hosts) as client:
        await client.put("/process", content=scan_pdf, headers=PDF)  # mineru + qwen
        await client.put("/process", content=scan_pdf, headers={**PDF, "X-OCR-Engine": "paddle"})
        await client.put("/process", content=b"PK", headers={"X-Filename": "report.docx"})  # docling

    assert seen == {
        "docling.prod/docling/convert",
        "mineru.prod/mineru/load",
        "mineru.prod/mineru/unload",
        "mineru.prod/mineru/extract",
        "paddle.prod/paddle/load",
        "paddle.prod/paddle/unload",
        "paddle.prod/paddle/extract",
        "ollama.prod/api/chat",
    }
    assert models == {("qwen3-vl:8b-instruct", 4096)}
