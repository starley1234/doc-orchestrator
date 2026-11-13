"""Проверка самой заглушки.

Мок обязан «ломаться» при нарушении лимитов GPU — иначе стресс-тесты оркестратора ничего не доказывают.
"""

import asyncio
import base64
import json

import httpx
import pytest
from conftest import decode, make_png

from app.schemas.drawing_stamp import DrawingStampCore, TechRequirementsList
from app.schemas.invoice import InvoiceSchema
from app.schemas.upd import UpdSchema
from mocks.main import MERMAID, SEAL_TEXT, STAMP, TECH_REQUIREMENTS, MockState

SCAN = {"file": ("scan.pdf", b"%PDF-1.7", "application/pdf")}


def chat(prompt: str, schema: type | None = None, image: bytes | None = None) -> dict:
    """Тело запроса к Ollama: промпт, необязательные JSON Schema ответа и картинка."""
    message: dict = {"role": "user", "content": prompt}
    if image:
        message["images"] = [base64.b64encode(image).decode()]
    body: dict = {"model": "qwen3-vl", "messages": [message], "stream": False}
    if schema:
        body["format"] = schema.model_json_schema()
    return body


async def ask(http: httpx.AsyncClient, prompt: str, schema: type | None = None) -> str:
    response = await http.post("/api/chat", json=chat(prompt, schema))
    return response.json()["message"]["content"]


async def test_docling_returns_markdown_with_filename(mock_http: httpx.AsyncClient):
    files = {"file": ("отчёт.docx", b"PK")}
    response = await mock_http.post("/docling/convert", files=files)
    assert response.json() == {"markdown": "# Цифровой документ (отчёт.docx)\nТаблица и текст из Docling."}


async def test_mineru_returns_placeholder_and_figure(mock_http: httpx.AsyncClient):
    await mock_http.post("/mineru/load")
    result = (await mock_http.post("/mineru/extract", files=SCAN)).json()

    assert "![Схема](fig_1.png)" in result["markdown"]
    assert "scan.pdf" in result["markdown"]
    figure = decode(base64.b64decode(result["figures"]["fig_1.png"]))
    assert (figure.format, figure.size) == ("PNG", (64, 64))


async def test_paddle_returns_upd_table_and_seal(mock_http: httpx.AsyncClient):
    await mock_http.post("/paddle/load")
    markdown = (await mock_http.post("/paddle/extract", files=SCAN)).json()["markdown"]

    assert "| № | Наименование |" in markdown
    assert "Всего к оплате: 34923.00" in markdown
    assert SEAL_TEXT == "ПЕЧАТЬ: ООО Ромашка ОГРН 1027700132195"
    assert SEAL_TEXT in markdown


@pytest.mark.parametrize(("model", "name"), [("mineru", "MinerU"), ("paddle", "PaddleOCR")])
async def test_extract_requires_loaded_model(mock_http: httpx.AsyncClient, mock: MockState, model, name):
    response = await mock_http.post(f"/{model}/extract", files=SCAN)

    assert response.status_code == 500
    assert response.json()["detail"] == f"{name} not loaded"
    assert mock.oom_errors == 0  # это не нехватка памяти


@pytest.mark.parametrize(
    ("loaded", "called", "detail"),
    [
        ("paddle", "mineru", "Simulated OOM: PaddleOCR is still in GPU 0 VRAM!"),
        ("mineru", "paddle", "Simulated OOM: MinerU is still in GPU 0 VRAM!"),
    ],
)
async def test_gpu0_oom_when_second_model_is_loaded(
    mock_http: httpx.AsyncClient, mock: MockState, loaded, called, detail
):
    await mock_http.post(f"/{loaded}/load")

    response = await mock_http.post(f"/{called}/load")

    assert response.status_code == 500
    assert response.json()["detail"] == detail
    assert mock.loaded == {loaded}
    assert mock.oom_errors == 1


async def test_gpu0_oom_when_both_models_are_in_vram(mock_http: httpx.AsyncClient, mock: MockState):
    mock.loaded |= {"mineru", "paddle"}  # так бывает, если оркестратор забыл выгрузить модель

    for model in ("mineru", "paddle"):
        response = await mock_http.post(f"/{model}/extract", files=SCAN)
        assert (response.status_code, response.json()["detail"]) == (500, "Simulated OOM")
    assert mock.oom_errors == 2


async def test_gpu0_works_after_unload(mock_http: httpx.AsyncClient, mock: MockState):
    await mock_http.post("/paddle/load")
    await mock_http.post("/paddle/unload")
    await mock_http.post("/mineru/load")

    response = await mock_http.post("/mineru/extract", files=SCAN)

    assert response.status_code == 200
    assert mock.loaded == {"mineru"}
    assert mock.oom_errors == 0


async def test_state_endpoint_reports_vram_and_violations(mock_http: httpx.AsyncClient):
    assert (await mock_http.get("/mock/state")).json() == {
        "mineru_loaded": False,
        "paddle_loaded": False,
        "oom_errors": 0,
        "gpu1_rejects": 0,
        "gpu0_peak": 0,
    }

    await mock_http.post("/paddle/load")
    await mock_http.post("/paddle/extract", files=SCAN)
    await mock_http.post("/mineru/load")  # OOM

    assert (await mock_http.get("/mock/state")).json() == {
        "mineru_loaded": False,
        "paddle_loaded": True,
        "oom_errors": 1,
        "gpu1_rejects": 0,
        "gpu0_peak": 1,
    }


async def test_vlm_rejects_parallel_requests(mock_http: httpx.AsyncClient, mock: MockState):
    requests = [mock_http.post("/api/chat", json=chat("Опиши схему")) for _ in range(3)]

    responses = await asyncio.gather(*requests)

    assert sorted(r.status_code for r in responses) == [200, 429, 429]
    rejected = next(r for r in responses if r.status_code == 429)
    assert rejected.json()["detail"] == "GPU 1 Busy: Concurrent VLM request!"
    assert mock.gpu1_rejects == 2


async def test_vlm_oom_on_oversized_image(mock_http: httpx.AsyncClient, mock: MockState):
    fits = await mock_http.post("/api/chat", json=chat("Опиши схему", image=make_png(1280, 1000)))
    too_big = await mock_http.post("/api/chat", json=chat("Опиши схему", image=make_png(1281, 1000)))

    assert fits.status_code == 200
    assert too_big.status_code == 500
    assert "Simulated OOM" in too_big.json()["detail"]
    assert mock.images == [(1280, 1000), (1281, 1000)]
    assert mock.oom_errors == 1


async def test_vlm_describes_figure_with_mermaid(mock_http: httpx.AsyncClient):
    response = await mock_http.post("/api/chat", json=chat("Переведи схему в Mermaid"))

    assert response.json()["message"] == {"role": "assistant", "content": MERMAID}
    assert MERMAID == "```mermaid\ngraph TD;\n  A[Вход] --> B[Обработка];\n```"


async def test_vlm_answers_drawing_schemas(mock_http: httpx.AsyncClient):
    stamp = json.loads(await ask(mock_http, "Штамп", DrawingStampCore))
    requirements = json.loads(await ask(mock_http, "Техтребования", TechRequirementsList))

    assert stamp == STAMP
    assert stamp == {
        "designation": "АБВГ.711141.001",
        "title": "Вал приводной",
        "material": "Сталь 45 ГОСТ 1050-2013",
        "mass": "12.5",
        "scale": "1:1",
        "organization": "АО КБ Маш",
    }
    assert requirements == TECH_REQUIREMENTS
    assert requirements["requirements"][1] == "2. Острые кромки притупить R0,5."


@pytest.mark.parametrize("schema", [UpdSchema, InvoiceSchema])
async def test_vlm_returns_valid_document_by_schema(mock_http: httpx.AsyncClient, schema):
    answer = json.loads(await ask(mock_http, "Файл: upd.pdf", schema))

    assert set(answer) == set(schema.model_fields)  # только поля запрошенной схемы
    document = schema.model_validate(answer)
    assert (document.seller.inn, document.seller.kpp) == ("7707083893", "770701001")
    assert sum(item.row_total for item in document.items) == document.total_with_vat
    if schema is InvoiceSchema:
        assert document.bank.bik == "044525225"


async def test_vlm_returns_broken_document_for_bad_inn(mock_http: httpx.AsyncClient):
    bad = json.loads(await ask(mock_http, "Файл: bad_inn.pdf", UpdSchema))

    assert bad["seller"]["inn"] == "7701234569"
    (item,) = bad["items"]
    assert (item["qty"], item["price"], item["amount_without_vat"]) == (2, 500, 1000)
    assert (item["vat_amount"], item["row_total"]) == (200, 1200)
    assert bad["total_with_vat"] == 1500
