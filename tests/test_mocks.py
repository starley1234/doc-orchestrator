"""Проверка самой заглушки.

Мок обязан «ломаться» при нарушении лимитов GPU — иначе стресс-тесты оркестратора ничего не доказывают.
"""

import asyncio
import base64
import io
import json

import httpx
import pytest
from PIL import Image

from mocks.main import MockState

SCAN = {"file": ("scan.pdf", b"%PDF-1.7", "application/pdf")}
CHAT_URL = "/gateway/v1/chat/completions"


def chat(text: str, schema_title: str | None = None) -> dict:
    """Тело запроса к Gateway: промпт и (необязательно) JSON Schema ответа."""
    body: dict = {
        "model": "qwen3-vl",
        "messages": [{"role": "user", "content": [{"type": "text", "text": text}]}],
    }
    if schema_title:
        body["format"] = {"title": schema_title, "type": "object"}
    return body


def answer(response: httpx.Response) -> str:
    return response.json()["choices"][0]["message"]["content"]


async def test_docling_returns_markdown(mock_http: httpx.AsyncClient):
    response = await mock_http.post("/docling/convert", files=SCAN)
    assert response.json() == {"markdown": "# Документ (Docling)\nТекст цифрового документа."}


async def test_mineru_returns_placeholder_and_image(mock_http: httpx.AsyncClient):
    result = (await mock_http.post("/mineru/extract", files=SCAN)).json()
    assert "![fig_1](fig_1.png)" in result["markdown"]
    assert result["images"][0]["name"] == "fig_1.png"


async def test_paddle_returns_upd_table(mock_http: httpx.AsyncClient):
    markdown = (await mock_http.post("/paddle/extract", files=SCAN)).json()["markdown"]
    assert "| № | Наименование |" in markdown
    assert "Всего к оплате: 34923.00" in markdown


@pytest.mark.parametrize(("loaded", "called"), [("paddle", "mineru"), ("mineru", "paddle")])
async def test_gpu0_oom_when_second_model_enters_vram(
    mock_http: httpx.AsyncClient, mock: MockState, loaded, called
):
    await mock_http.post(f"/{loaded}/load")

    for action, kwargs in (("extract", {"files": SCAN}), ("load", {})):
        response = await mock_http.post(f"/{called}/{action}", **kwargs)
        assert response.status_code == 500
        assert response.json()["detail"] == "Out Of Memory (Simulated)"
    assert mock.oom_errors == 2


async def test_gpu0_works_after_unload(mock_http: httpx.AsyncClient, mock: MockState):
    await mock_http.post("/paddle/load")
    await mock_http.post("/paddle/unload")

    response = await mock_http.post("/mineru/extract", files=SCAN)

    assert response.status_code == 200
    assert mock.loaded == {"mineru"}  # модель подгрузилась лениво
    assert mock.oom_errors == 0


async def test_gateway_rejects_parallel_requests(mock_http: httpx.AsyncClient, mock: MockState):
    responses = await asyncio.gather(*(mock_http.post(CHAT_URL, json=chat("Опиши схему")) for _ in range(3)))

    assert sorted(r.status_code for r in responses) == [200, 429, 429]
    assert "Too Many Concurrent Requests on GPU 1" in next(r.text for r in responses if r.status_code == 429)
    assert mock.gateway_rejects == 2


def with_image(body: dict, width: int, height: int) -> dict:
    """Добавляет в запрос PNG-картинку заданного размера."""
    buffer = io.BytesIO()
    Image.new("L", (width, height)).save(buffer, "PNG")
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    body["messages"][0]["content"].append({"type": "image_url", "image_url": {"url": url}})
    return body


async def test_gateway_oom_on_oversized_image(mock_http: httpx.AsyncClient, mock: MockState):
    fits = await mock_http.post(CHAT_URL, json=with_image(chat("Опиши схему"), 1280, 1000))
    too_big = await mock_http.post(CHAT_URL, json=with_image(chat("Опиши схему"), 1281, 1000))

    assert fits.status_code == 200
    assert too_big.status_code == 500
    assert too_big.json()["detail"] == "Out Of Memory (Simulated)"
    assert mock.images == [(1280, 1000), (1281, 1000)]
    assert mock.oom_errors == 1


async def test_gateway_describes_image_with_mermaid(mock_http: httpx.AsyncClient):
    response = await mock_http.post(CHAT_URL, json=chat("Опиши схему"))
    assert answer(response) == "```mermaid\ngraph TD; A-->B;\n```"


async def test_gateway_answers_by_schema(mock_http: httpx.AsyncClient):
    stamp = json.loads(answer(await mock_http.post(CHAT_URL, json=chat("Штамп", "DrawingStampSchema"))))
    assert stamp == {
        "designation": "АБВГ.123456.001",
        "title": "Вал приводной",
        "material": "Сталь 45 ГОСТ 1050-2013",
        "mass": "12.5",
    }

    good = json.loads(answer(await mock_http.post(CHAT_URL, json=chat("Файл: upd.pdf", "UpdSchema"))))
    bad = json.loads(answer(await mock_http.post(CHAT_URL, json=chat("Файл: bad_inn.pdf", "UpdSchema"))))
    assert good["seller"]["inn"] == "7701234560"
    assert bad["seller"]["inn"] == "7701234569"
    assert bad["document_total"] != good["document_total"]


async def test_dlp_masks_secret_and_passport(mock_http: httpx.AsyncClient):
    text = "Токен SECRET_TOKEN, паспорт 4510 123456, ИНН 7701234560"
    result = (await mock_http.post("/dlp/inspect", json={"text": text})).json()
    assert result == {"text": "Токен [REDACTED], паспорт [REDACTED], ИНН 7701234560", "masked": True}

    clean = (await mock_http.post("/dlp/inspect", json={"text": "Обычный текст"})).json()
    assert clean == {"text": "Обычный текст", "masked": False}
