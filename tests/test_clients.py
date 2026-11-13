"""HTTP-клиенты движков: разбор ответов мока и превращение любых сбоев в EngineError."""

import base64
import json

import httpx
import pytest

from app.clients.base import EngineClient, MarkdownReply
from app.clients.dlp import DlpClient
from app.clients.docling import DoclingClient
from app.clients.mineru import MineruClient
from app.clients.paddle import PaddleClient
from app.clients.vlm_gateway import VlmGatewayClient
from app.config import Settings
from app.core.errors import EngineError
from mocks.main import MERMAID, MockState

PDF = b"%PDF-1.7 test"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def client_with(handler) -> EngineClient:
    """Клиент, которому вместо сети отвечает функция handler(request)."""
    return EngineClient(httpx.AsyncClient(transport=httpx.MockTransport(handler)), "http://engine/api/")


# --- Ответы движков -----------------------------------------------------------------------


async def test_docling_convert(mock_http: httpx.AsyncClient, settings: Settings):
    markdown = await DoclingClient(mock_http, settings.docling_url).convert(PDF, "report.pdf")
    assert markdown.startswith("# Документ (Docling)")


async def test_mineru_extract_decodes_figures(mock_http: httpx.AsyncClient, settings: Settings):
    reply = await MineruClient(mock_http, settings.mineru_url).extract(PDF, "scan.pdf")

    assert "![fig_1](fig_1.png)" in reply.markdown
    assert [figure.name for figure in reply.images] == ["fig_1.png"]
    assert reply.images[0].data.startswith(PNG_SIGNATURE)  # base64 уже раскодирован в байты


async def test_paddle_extract(mock_http: httpx.AsyncClient, settings: Settings):
    markdown = await PaddleClient(mock_http, settings.paddle_url).extract(PDF, "upd.pdf")
    assert "| № | Наименование |" in markdown


async def test_gpu_engine_load_and_unload(mock_http: httpx.AsyncClient, settings: Settings, mock: MockState):
    paddle = PaddleClient(mock_http, settings.paddle_url)

    await paddle.load()
    assert mock.loaded == {"paddle"}
    await paddle.unload()
    assert mock.loaded == set()


async def test_original_filename_reaches_engine(mock_http: httpx.AsyncClient, settings: Settings):
    # Мок дописывает секрет в текст файлов с «secret» в имени — значит, имя дошло, включая кириллицу
    markdown = await DoclingClient(mock_http, settings.docling_url).convert(PDF, "договор_secret.pdf")
    assert "SECRET_TOKEN" in markdown


async def test_dlp_inspect(mock_http: httpx.AsyncClient, settings: Settings):
    dlp = DlpClient(mock_http, settings.dlp_url)

    masked = await dlp.inspect("Токен SECRET_TOKEN")
    clean = await dlp.inspect("Обычный текст")

    assert (masked.text, masked.masked) == ("Токен [REDACTED]", True)
    assert (clean.text, clean.masked) == ("Обычный текст", False)


async def test_vlm_chat(mock_http: httpx.AsyncClient, settings: Settings, mock: MockState):
    vlm = VlmGatewayClient(mock_http, settings.vlm_gateway_url, settings.vlm_model)

    assert await vlm.chat("Опиши схему") == MERMAID
    assert mock.prompts == ["Опиши схему"]


async def test_vlm_request_carries_model_image_and_schema():
    sent: list[dict] = []

    def gateway(request: httpx.Request) -> httpx.Response:
        sent.append({"url": str(request.url), **json.loads(request.content)})
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "{}"}}]})

    http = httpx.AsyncClient(transport=httpx.MockTransport(gateway))
    vlm = VlmGatewayClient(http, "http://gateway/", "qwen3-vl:8b")
    schema = {"title": "DrawingStampSchema", "type": "object"}

    assert await vlm.chat("Прочитай штамп", image_png=PNG_SIGNATURE, schema=schema) == "{}"

    (body,) = sent
    text, image = body["messages"][0]["content"]
    assert body["url"] == "http://gateway/v1/chat/completions"
    assert body["model"] == "qwen3-vl:8b"
    assert body["format"] == schema
    assert text == {"type": "text", "text": "Прочитай штамп"}
    assert image["image_url"]["url"] == "data:image/png;base64," + base64.b64encode(PNG_SIGNATURE).decode()


async def test_vlm_request_without_schema_has_no_format_field():
    sent: list[dict] = []

    def gateway(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    vlm = VlmGatewayClient(
        httpx.AsyncClient(transport=httpx.MockTransport(gateway)), "http://gateway", "qwen3-vl"
    )
    await vlm.chat("Опиши схему")

    assert "format" not in sent[0]
    assert sent[0]["messages"][0]["content"] == [{"type": "text", "text": "Опиши схему"}]


# --- Сбои ---------------------------------------------------------------------------------


async def test_http_error_becomes_engine_error(mock_http: httpx.AsyncClient, settings: Settings):
    await PaddleClient(mock_http, settings.paddle_url).load()

    with pytest.raises(EngineError, match=r"MinerU вернул HTTP 500: .*Out Of Memory \(Simulated\)"):
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
    with pytest.raises(EngineError, match="некорректный ответ"):
        await client_with(lambda request: httpx.Response(200, text=body))._post_json(
            "/convert", MarkdownReply
        )


async def test_vlm_reply_without_choices_is_rejected():
    http = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"choices": []}))
    )

    with pytest.raises(EngineError, match="некорректный ответ"):
        await VlmGatewayClient(http, "http://gateway", "qwen3-vl").chat("Опиши схему")


async def test_base_url_trailing_slash_is_ignored():
    urls: list[str] = []

    def record(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        return httpx.Response(200, json={})

    await client_with(record)._post("/load")
    assert urls == ["http://engine/api/load"]
