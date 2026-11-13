"""HTTP-контракт сервиса: заголовки и токен Open WebUI, коды ошибок, ответы модели не по схеме, async_mode."""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote

import httpx
import pytest
from conftest import MOCK_URL, engines_stub
from fakeredis import FakeAsyncRedis, FakeServer

from app.config import Settings
from mocks.main import INVOICE, UPD, MockState

ROOT = Path(__file__).parent.parent  # корень репозитория
PDF = {"X-Filename": "scan.pdf"}


async def extract(client: httpx.AsyncClient, filename: str, data: bytes, **form: str) -> httpx.Response:
    return await client.post("/api/extract-structured", files={"file": (filename, data)}, data=form)


async def wait_for_task(client: httpx.AsyncClient, task_id: str) -> dict:
    """Опрашивает задачу, пока она не завершится (до 5 секунд)."""
    for _ in range(250):
        task = (await client.get(f"/api/tasks/{task_id}")).json()
        if task["status"] in ("completed", "failed"):
            return task
        await asyncio.sleep(0.02)
    raise AssertionError(f"Задача не завершилась: {task}")


# --- PUT /process: заголовки и токен --------------------------------------------------------


async def test_filename_is_percent_decoded(client, scan_pdf: bytes):
    # Open WebUI кодирует имя файла: quote("Счёт на оплату 57.pdf")
    response = await client.put(
        "/process", content=scan_pdf, headers={"X-Filename": quote("Счёт на оплату 57.pdf")}
    )

    assert response.json()["metadata"]["filename"] == "Счёт на оплату 57.pdf"
    assert response.json()["metadata"]["engine_used"] == "paddle"  # «счёт» — документ с таблицей


@pytest.mark.parametrize("sent_name", ["../../etc/report.pdf", "C:\\Документы\\report.pdf"])
async def test_filename_path_is_dropped(client, digital_pdf: bytes, sent_name: str):
    response = await client.put(
        "/process", content=digital_pdf, headers={"X-Filename": quote(sent_name, safe="")}
    )
    assert response.json()["metadata"]["filename"] == "report.pdf"


async def test_extension_is_taken_from_content_type(client, digital_pdf: bytes):
    headers = {"X-Filename": "document", "Content-Type": "application/pdf"}
    response = await client.put("/process", content=digital_pdf, headers=headers)

    assert response.status_code == 200
    assert response.json()["metadata"] == {
        "filename": "document.pdf",
        "engine_used": "docling",
        "dlp_masked": False,
    }


async def test_missing_filename_header_is_rejected(client, scan_pdf: bytes):
    response = await client.put("/process", content=scan_pdf)
    assert response.status_code == 422


async def test_token_is_not_required_by_default(client, digital_pdf: bytes):
    headers = {"X-Filename": "report.pdf", "Authorization": "Bearer anything"}
    assert (await client.put("/process", content=digital_pdf, headers=headers)).status_code == 200


@pytest.mark.parametrize(
    ("authorization", "expected_status"),
    [(None, 401), ("Bearer wrong", 401), ("s3cret", 401), ("Bearer s3cret", 200)],
)
async def test_token_is_checked_when_configured(
    start_app, digital_pdf: bytes, authorization, expected_status
):
    headers = {"X-Filename": "report.pdf"} | ({"Authorization": authorization} if authorization else {})

    async with start_app(openwebui_api_token="s3cret") as client:
        response = await client.put("/process", content=digital_pdf, headers=headers)
        health = await client.get("/health")

    assert response.status_code == expected_status
    assert health.status_code == 200  # проверка живости токена не требует


# --- Ошибки ---------------------------------------------------------------------------------


async def test_unsupported_format_is_415(client):
    response = await client.put("/process", content=b"plain text", headers={"X-Filename": "notes.txt"})

    assert response.status_code == 415
    assert response.json() == {"detail": "Формат файла «notes.txt» не поддерживается"}


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
        structured = await extract(client, "big.docx", b"x" * (megabyte + 1), doc_type="contract")

    assert fits.status_code == 200
    assert (too_large.status_code, too_large.json()) == (413, {"detail": "Файл больше 1 МБ"})
    assert structured.status_code == 413


@pytest.mark.parametrize("engine", ["docling", "dlp"])
async def test_engine_failure_is_502(start_app, digital_pdf: bytes, engine: str):
    async with start_app(http=engines_stub(failing=engine)) as client:
        response = await client.put("/process", content=digital_pdf, headers={"X-Filename": "report.pdf"})

    assert response.status_code == 502
    assert "вернул HTTP 500: engine crashed" in response.json()["detail"]


async def test_dlp_failure_never_leaks_unchecked_text(start_app, scan_pdf: bytes):
    # DLP недоступен — текст наружу не отдаётся (ни из /process, ни из извлечения JSON)
    async with start_app(dlp_url=f"{MOCK_URL}/no-dlp") as client:
        markdown = await client.put("/process", content=scan_pdf, headers={"X-Filename": "table.pdf"})
        structured = await extract(client, "shaft.pdf", scan_pdf, doc_type="drawing")

    assert markdown.status_code == structured.status_code == 502
    assert "DLP" in markdown.json()["detail"]


async def test_busy_gpu_is_503(start_app, scan_pdf: bytes, redis: FakeAsyncRedis):
    await redis.set("lock:gpu0", "занято другой задачей")

    async with start_app(gpu_wait_timeout_s=0.2) as client:
        response = await client.put("/process", content=scan_pdf, headers=PDF)

    assert response.status_code == 503
    assert response.json()["detail"].startswith("Видеокарта занята")


async def test_redis_failure_is_503(client, scan_pdf: bytes, redis_server: FakeServer):
    redis_server.connected = False
    process = await client.put("/process", content=scan_pdf, headers=PDF)
    health = await client.get("/health")
    redis_server.connected = True

    assert process.status_code == health.status_code == 503
    assert health.json() == {"detail": "Redis недоступен"}
    assert (await client.get("/health")).json() == {"status": "ok"}


# --- POST /api/extract-structured: проверка запроса -------------------------------------------


async def test_unknown_doc_type_is_422(client, scan_pdf: bytes):
    assert (await extract(client, "act.pdf", scan_pdf, doc_type="act")).status_code == 422


async def test_missing_form_fields_are_422(client, scan_pdf: bytes):
    assert (await client.post("/api/extract-structured", data={"doc_type": "upd"})).status_code == 422
    assert (await extract(client, "upd.pdf", scan_pdf)).status_code == 422


async def test_unsupported_structured_file_is_415(client):
    assert (await extract(client, "upd.txt", b"text", doc_type="upd")).status_code == 415


async def test_drawing_must_be_pdf_or_image(client):
    office = await extract(client, "shaft.docx", b"PK\x03\x04", doc_type="drawing")
    garbage = await extract(client, "shaft.dwg", b"AC1027 not an image", doc_type="drawing")

    assert (office.status_code, office.json()["detail"]) == (
        415,
        "Чертёж принимается только как PDF или изображение",
    )
    assert garbage.status_code == 422
    assert garbage.json()["detail"].startswith("Не удалось прочитать изображение")


# --- Ответ модели не по схеме: это предупреждения, а не сбой -------------------------------


async def test_answer_with_missing_fields(start_app, digital_pdf: bytes):
    async with start_app(http=engines_stub('{"number": "128", "date": "2025-03-14"}')) as client:
        response = await extract(client, "upd.pdf", digital_pdf, doc_type="upd")

    assert response.status_code == 200
    body = response.json()
    assert body["data"] == {"number": "128", "date": "2025-03-14"}  # данные как есть
    assert body["validation"]["is_valid"] is False
    assert body["validation"]["warnings"] == [
        "Поле seller: обязательное поле не заполнено",
        "Поле buyer: обязательное поле не заполнено",
        "Поле items: обязательное поле не заполнено",
        "Поле document_total: обязательное поле не заполнено",
    ]


async def test_answer_with_wrong_value_type(start_app, digital_pdf: bytes):
    items = [{**UPD["items"][0], "qty": "десять"}]
    async with start_app(http=engines_stub(json.dumps({**UPD, "items": items}))) as client:
        body = (await extract(client, "upd.pdf", digital_pdf, doc_type="upd")).json()

    assert body["validation"]["is_valid"] is False
    (warning,) = body["validation"]["warnings"]
    assert warning.startswith("Поле items.0.qty: некорректное значение")
    assert body["data"]["items"][0]["qty"] == "десять"


@pytest.mark.parametrize("answer", ["Извините, документ не читается", "[1, 2, 3]", '{"number": "128"'])
async def test_answer_that_is_not_a_json_object(start_app, digital_pdf: bytes, answer: str):
    async with start_app(http=engines_stub(answer)) as client:
        response = await extract(client, "upd.pdf", digital_pdf, doc_type="upd")

    assert response.status_code == 200
    body = response.json()
    assert body["data"] == {}
    assert body["validation"]["is_valid"] is False
    assert body["validation"]["warnings"]


async def test_bad_kpp_bik_and_row_math_are_reported(start_app, digital_pdf: bytes):
    broken = {
        **INVOICE,
        "seller": {**INVOICE["seller"], "kpp": "77010100"},  # 8 символов вместо 9
        "buyer": {**INVOICE["buyer"], "inn": "502415678"},  # 9 цифр
        "bank": {**INVOICE["bank"], "bik": "144525999"},  # начинается не с 04
        "items": [{**INVOICE["items"][0], "amount_without_vat": 2600.0}, *INVOICE["items"][1:]],
    }
    async with start_app(http=engines_stub(json.dumps(broken))) as client:
        body = (await extract(client, "invoice.pdf", digital_pdf, doc_type="invoice")).json()

    assert body["validation"] == {
        "is_valid": False,
        "warnings": [
            "КПП продавца 77010100 имеет неверный формат",
            "ИНН покупателя 502415678 не прошёл контрольную сумму",
            "БИК банка 144525999 имеет неверный формат",
            "Строка 1: кол-во × цена (2500.00) != сумма без НДС (2600.00)",
            "Строка 1: сумма без НДС + НДС (3100.00) != сумма по строке (3000.00)",
        ],
    }


async def test_contract_parties_are_validated(start_app, digital_pdf: bytes):
    contract = {
        "number": "7",
        "date": "2025-02-01",
        "subject": "Оказание услуг",
        "parties": [
            {"name": "ИП Иванов И. И.", "inn": "771234567859", "role": "Исполнитель"},  # ИП: 12 цифр, КПП нет
            {"name": "ООО «Вектор»", "inn": "5024156783", "kpp": "502401001"},
        ],
    }
    async with start_app(http=engines_stub(json.dumps(contract))) as client:
        body = (await extract(client, "contract.pdf", digital_pdf, doc_type="contract")).json()

    assert body["validation"] == {
        "is_valid": False,
        "warnings": ["ИНН стороны 2 5024156783 не прошёл контрольную сумму"],
    }
    assert body["data"]["parties"][0] == {
        "name": "ИП Иванов И. И.",
        "inn": "771234567859",
        "kpp": None,
        "address": None,
        "role": "Исполнитель",
    }


# --- Асинхронный режим ------------------------------------------------------------------------


async def test_async_mode_returns_task_id_then_result(client, scan_pdf: bytes):
    response = await extract(client, "upd.pdf", scan_pdf, doc_type="upd", async_mode="true")

    assert response.status_code == 202
    queued = response.json()
    assert set(queued) == {"task_id", "status"}
    assert queued["status"] == "queued"

    task = await wait_for_task(client, queued["task_id"])
    assert set(task) == {"task_id", "status", "doc_type", "data", "validation"}
    assert (task["task_id"], task["status"], task["doc_type"]) == (queued["task_id"], "completed", "upd")
    assert task["validation"] == {"is_valid": True, "warnings": []}
    assert task["data"]["seller"]["inn"] == "7701234560"


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


async def test_async_tasks_share_gpu_queue_with_sync_requests(
    client, scan_pdf: bytes, drawing_png: bytes, mock
):
    files = [
        ("upd.pdf", scan_pdf, "upd"),
        ("shaft.png", drawing_png, "drawing"),
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
    assert (mock.oom_errors, mock.gateway_rejects, mock.gpu0_peak) == (0, 0, 1)


async def test_unknown_task_is_404(client):
    response = await client.get("/api/tasks/00000000-0000-4000-8000-000000000000")

    assert response.status_code == 404
    assert response.json() == {"detail": "Задача не найдена или срок её хранения истёк"}


async def test_malformed_task_id_is_422(client):
    assert (await client.get("/api/tasks/not-a-uuid")).status_code == 422


# --- Настройки и служебное --------------------------------------------------------------------


def test_env_example_lists_every_setting():
    lines = (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    names = {line.partition("=")[0].lower() for line in lines if line and not line.startswith("#")}

    assert names == set(Settings.model_fields)
    Settings(_env_file=ROOT / ".env.example")  # все значения образца проходят проверку типов


def test_service_starts_with_settings_from_environment():
    # Так сервис запускает uvicorn: импорт app.main собирает приложение из переменных окружения
    env = {**os.environ, "LOG_LEVEL": "debug", "PADDLE_URL": "http://paddle.prod"}
    code = "from app.main import app; print(app.state.settings.paddle_url)"
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True)

    assert result.stdout.strip() == "http://paddle.prod", result.stderr


async def test_openapi_describes_all_endpoints(client):
    paths = (await client.get("/openapi.json")).json()["paths"]

    assert {"/process", "/api/extract-structured", "/api/tasks/{task_id}", "/health"} <= set(paths)
    assert "put" in paths["/process"]


async def test_engine_urls_come_from_settings(start_app, mock_app, scan_pdf: bytes, drawing_png: bytes):
    # Каждому движку задаём свой хост и проверяем, что запросы уходят именно туда, куда велят настройки
    seen: set[str] = set()
    mock_transport = httpx.ASGITransport(app=mock_app)

    class Recorder(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            seen.add(f"{request.url.host}/{request.url.path.split('/')[1]}")
            return await mock_transport.handle_async_request(request)

    hosts = {
        "docling_url": "http://docling.prod/docling",
        "mineru_url": "http://mineru.prod/mineru",
        "paddle_url": "http://paddle.prod/paddle",
        "vlm_gateway_url": "http://gateway.prod/gateway",
        "dlp_url": "http://dlp.prod/dlp",
    }
    async with start_app(http=httpx.AsyncClient(transport=Recorder()), **hosts) as client:
        await client.put("/process", content=scan_pdf, headers=PDF)  # mineru + qwen + dlp
        await client.put("/process", content=scan_pdf, headers={"X-Filename": "table.pdf"})  # paddle
        await client.put("/process", content=b"PK", headers={"X-Filename": "report.docx"})  # docling
        await extract(client, "shaft.png", drawing_png, doc_type="drawing")

    assert seen == {
        "docling.prod/docling",
        "mineru.prod/mineru",
        "paddle.prod/paddle",
        "gateway.prod/gateway",
        "dlp.prod/dlp",
    }
