"""HTTP-клиенты движков: разбор ответов мока и превращение любых сбоев в EngineError."""

import base64
import json

import httpx
import pytest

from app.clients.base import EngineClient, MarkdownReply
from app.clients.docling import DoclingClient
from app.clients.mineru import MineruClient
from app.clients.paddle import PaddleClient
from app.clients.vlm import VlmClient
from app.config import Settings
from app.core.errors import EngineError
from mocks.main import MERMAID, SEAL_TEXT, MockState

PDF = b"%PDF-1.7 test"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def client_with(handler) -> EngineClient:
    """Клиент, которому вместо сети отвечает функция handler(request)."""
    return EngineClient(httpx.AsyncClient(transport=httpx.MockTransport(handler)), "http://engine/api/")


def vlm_with(handler, model: str = "qwen3-vl:4b-instruct-q8_0") -> VlmClient:
    return VlmClient(
        httpx.AsyncClient(transport=httpx.MockTransport(handler)), "http://ollama:11434/", model, 8192
    )


# --- Ответы движков -----------------------------------------------------------------------


async def test_docling_convert_sends_file_with_its_name(mock_http: httpx.AsyncClient, settings: Settings):
    markdown = await DoclingClient(mock_http, settings.docling_url).convert(PDF, "договор.pdf")
    assert markdown == "# Цифровой документ (договор.pdf)\nТаблица и текст из Docling."


async def test_mineru_extract_decodes_figures(mock_http: httpx.AsyncClient, settings: Settings):
    mineru = MineruClient(mock_http, settings.mineru_url)
    await mineru.load()

    reply = await mineru.extract(PDF, "scan.pdf")

    assert "![Схема](fig_1.png)" in reply.markdown
    assert list(reply.figures) == ["fig_1.png"]
    assert reply.figures["fig_1.png"].startswith(PNG_SIGNATURE)  # base64 уже раскодирован в байты


async def test_mineru_reply_without_figures():
    http = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"markdown": "Текст"}))
    )
    reply = await MineruClient(http, "http://mineru").extract(PDF, "scan.pdf")
    assert (reply.markdown, reply.figures) == ("Текст", {})


async def test_paddle_extract(mock_http: httpx.AsyncClient, settings: Settings):
    paddle = PaddleClient(mock_http, settings.paddle_url)
    await paddle.load()

    markdown = await paddle.extract(PDF, "upd.pdf")

    assert "| № | Наименование |" in markdown
    assert SEAL_TEXT in markdown


async def test_gpu_engine_load_and_unload(mock_http: httpx.AsyncClient, settings: Settings, mock: MockState):
    paddle = PaddleClient(mock_http, settings.paddle_url)

    await paddle.load()
    assert mock.loaded == {"paddle"}
    await paddle.unload()
    assert mock.loaded == set()
    assert mock.calls == ["paddle/load", "paddle/unload"]


async def test_vlm_chat(mock_http: httpx.AsyncClient, settings: Settings, mock: MockState):
    vlm = VlmClient(mock_http, settings.ollama_url, settings.vlm_model, settings.vlm_num_ctx)

    assert await vlm.chat("Опиши схему") == MERMAID
    assert mock.prompts == ["Опиши схему"]


async def test_vlm_request_follows_ollama_chat_contract():
    sent: list[dict] = []

    def ollama(request: httpx.Request) -> httpx.Response:
        sent.append({"url": str(request.url), **json.loads(request.content)})
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "{}"}, "done": True})

    schema = {"title": "DrawingStampCore", "type": "object"}

    assert await vlm_with(ollama).chat("Прочитай штамп", image_png=PNG_SIGNATURE, schema=schema) == "{}"

    assert sent == [
        {
            "url": "http://ollama:11434/api/chat",
            "model": "qwen3-vl:4b-instruct-q8_0",
            "messages": [
                {
                    "role": "user",
                    "content": "Прочитай штамп",
                    "images": [base64.b64encode(PNG_SIGNATURE).decode()],  # без префикса data:image/...
                }
            ],
            "format": schema,
            "stream": False,
            "options": {"num_ctx": 8192, "temperature": 0.0},
        }
    ]


async def test_vlm_request_without_image_and_schema():
    sent: list[dict] = []

    def ollama(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "ok"}})

    await vlm_with(ollama, model="qwen3-vl:8b-instruct").chat("Опиши схему")

    assert "format" not in sent[0]
    assert sent[0]["model"] == "qwen3-vl:8b-instruct"
    assert sent[0]["messages"] == [{"role": "user", "content": "Опиши схему"}]


def test_default_vlm_settings_match_the_spec():
    settings = Settings(_env_file=None)

    assert settings.ollama_url == "http://mock-engines:8001"
    assert settings.vlm_model == "qwen3-vl:4b-instruct-q8_0"
    assert settings.vlm_num_ctx == 8192
    assert settings.http_timeout_sec == 300


# --- Сбои ---------------------------------------------------------------------------------


async def test_http_error_becomes_engine_error(mock_http: httpx.AsyncClient, settings: Settings):
    with pytest.raises(EngineError, match=r"MinerU вернул HTTP 500: .*MinerU not loaded"):
        await MineruClient(mock_http, settings.mineru_url).extract(PDF, "scan.pdf")


async def test_timeout_becomes_engine_error():
    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    with pytest.raises(EngineError, match="недоступен: ReadTimeout"):
        await client_with(slow)._post("/extract")


async def test_connection_failure_becomes_engine_error():
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(EngineError, match="недоступен: ConnectError"):
        await client_with(down)._post("/extract")


@pytest.mark.parametrize("body", ["<html>502 Bad Gateway</html>", '{"text": "нет поля markdown"}', ""])
async def test_malformed_reply_becomes_engine_error(body: str):
    client = client_with(lambda request: httpx.Response(200, text=body))

    with pytest.raises(EngineError, match="некорректный ответ"):
        await client._post_json("/convert", MarkdownReply)


@pytest.mark.parametrize("reply", [{"error": "model not found"}, {"message": {"role": "assistant"}}])
async def test_vlm_reply_without_content_is_rejected(reply: dict):
    with pytest.raises(EngineError, match="Qwen3-VL вернул некорректный ответ"):
        await vlm_with(lambda request: httpx.Response(200, json=reply)).chat("Опиши схему")


async def test_ollama_error_text_reaches_the_message():
    def no_model(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model 'qwen3-vl:4b-instruct-q8_0' not found"})

    with pytest.raises(EngineError, match=r"Qwen3-VL вернул HTTP 404: .*not found"):
        await vlm_with(no_model).chat("Опиши схему")


async def test_base_url_trailing_slash_is_ignored():
    urls: list[str] = []

    def record(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        return httpx.Response(200, json={})

    await client_with(record)._post("/load")
    assert urls == ["http://engine/api/load"]
