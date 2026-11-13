"""Интеграционные и стресс-тесты: оркестратор целиком по HTTP, движки — мок, Redis — в памяти.

Мок «ломается» при нарушении лимитов железа (HTTP 500 OOM, HTTP 429), поэтому нули в его
счётчиках доказывают, что оркестратор правильно выстраивает очередь к видеокартам.
"""

import asyncio
import io
import json

import httpx
import pytest
from conftest import engines_stub
from make_samples import make_pdf
from PIL import Image

from app.core.gpu_manager import GPUResourceManager
from mocks.main import MERMAID, PNG_1X1, STAMP, TECH_REQUIREMENTS, UPD, VLM_MAX_PIXELS, MockState

DESCRIBE_PROMPT = "Опиши схему текстом или переведи блок-схему в синтаксис Mermaid"
PASSPORT = "4510 123456"


async def process(client: httpx.AsyncClient, filename: str, data: bytes) -> httpx.Response:
    """Запрос Open WebUI: сырые байты файла, имя — в заголовке."""
    return await client.put("/process", content=data, headers={"X-Filename": filename})


async def extract(client: httpx.AsyncClient, filename: str, data: bytes, doc_type: str) -> httpx.Response:
    """Запрос 1С / Teamcenter: файл и тип документа в multipart-форме."""
    return await client.post(
        "/api/extract-structured", files={"file": (filename, data)}, data={"doc_type": doc_type}
    )


def png(width: int, height: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("L", (width, height), 255).save(buffer, "PNG")
    return buffer.getvalue()


# --- Пайплайн А: Markdown для Open WebUI --------------------------------------------------


async def test_scan_gets_mermaid_instead_of_figure_placeholder(client, scan_pdf: bytes, mock: MockState):
    response = await process(client, "scan.pdf", scan_pdf)

    assert response.status_code == 200
    body = response.json()
    assert body["metadata"] == {"filename": "scan.pdf", "engine_used": "mineru+qwen3vl", "dlp_masked": False}
    assert MERMAID in body["page_content"]  # описание от Qwen3-VL вклеено...
    assert "![fig_1]" not in body["page_content"]  # ...на место плейсхолдера картинки
    assert body["page_content"].startswith("# Отчёт (MinerU)")
    assert mock.prompts == [DESCRIBE_PROMPT]


async def test_scan_secrets_are_masked_by_dlp(client, scan_pdf: bytes):
    # В «распознанном» тексте файла secret*.pdf мок оставляет токен и номер паспорта
    body = (await process(client, "secret_scan.pdf", scan_pdf)).json()

    assert body["metadata"]["dlp_masked"] is True
    assert "SECRET_TOKEN" not in body["page_content"]
    assert PASSPORT not in body["page_content"]
    assert body["page_content"].count("[REDACTED]") == 2
    assert MERMAID in body["page_content"]  # остальной текст не пострадал


async def test_digital_pdf_goes_to_docling_without_gpu(client, digital_pdf: bytes, mock: MockState):
    body = (await process(client, "contract.pdf", digital_pdf)).json()

    assert body["metadata"]["engine_used"] == "docling"
    assert body["page_content"] == "# Документ (Docling)\nТекст цифрового документа."
    assert mock.calls == []  # GPU 0 не понадобился
    assert mock.prompts == []  # GPU 1 тоже


async def test_office_file_goes_to_docling(client, mock: MockState):
    body = (await process(client, "report.docx", b"PK\x03\x04 docx")).json()

    assert body["metadata"] == {"filename": "report.docx", "engine_used": "docling", "dlp_masked": False}
    assert mock.calls == []


async def test_table_scan_goes_to_paddle(client, scan_pdf: bytes, mock: MockState):
    body = (await process(client, "scan_table.pdf", scan_pdf)).json()

    assert body["metadata"]["engine_used"] == "paddle"
    assert "| № | Наименование |" in body["page_content"]
    assert mock.calls[-2:] == ["paddle/load", "paddle/extract"]
    assert mock.prompts == []  # таблицы Qwen3-VL не описывает


async def test_photo_is_recognized_by_mineru(client, mock: MockState):
    body = (await process(client, "photo.jpg", png(800, 600))).json()

    assert body["metadata"]["engine_used"] == "mineru+qwen3vl"
    assert mock.calls[-2:] == ["mineru/load", "mineru/extract"]


async def test_gpu0_is_free_while_figures_are_described(client, scan_pdf: bytes, mock: MockState, redis):
    await process(client, "scan.pdf", scan_pdf)

    # Картинки описывает GPU 1: лок GPU 0 к этому времени уже отпущен, а в конце свободны оба
    assert await redis.exists("lock:gpu0", "lock:gpu1") == 0
    assert mock.gpu0_peak == 1


async def test_each_figure_description_replaces_its_placeholder(start_app, scan_pdf: bytes):
    mineru = {
        "markdown": "Введение.\n\n![](fig_1.png)\n\nСередина.\n\n![Рис. 2 — схема](fig_2.png)\n\nКонец.",
        "images": [{"name": name, "data": PNG_1X1} for name in ("fig_1.png", "fig_2.png", "fig_3.png")],
    }
    description = (
        "Схема: A -> B. Путь C:\\temp\\1 и группа \\g<0> — обычный текст"  # спецсимволы regex не мешают
    )

    async with start_app(http=engines_stub(vlm=description, mineru=mineru)) as client:
        body = (await process(client, "scan.pdf", scan_pdf)).json()

    assert body["page_content"] == (
        f"Введение.\n\n{description}\n\nСередина.\n\n{description}\n\nКонец."
        f"\n\n{description}"  # у fig_3 плейсхолдера нет: описание дописано в конец
    )


async def test_scan_without_figures_is_not_sent_to_qwen(start_app, scan_pdf: bytes):
    async with start_app(http=engines_stub(mineru={"markdown": "Только текст."})) as client:
        body = (await process(client, "scan.pdf", scan_pdf)).json()

    assert body == {
        "page_content": "Только текст.",
        "metadata": {"filename": "scan.pdf", "engine_used": "mineru", "dlp_masked": False},
    }


async def test_broken_figure_from_mineru_is_engine_error(start_app, scan_pdf: bytes):
    mineru = {"markdown": "![](fig_1.png)", "images": [{"name": "fig_1.png", "data": "bm90IGFuIGltYWdl"}]}

    async with start_app(http=engines_stub(mineru=mineru)) as client:
        response = await process(client, "scan.pdf", scan_pdf)

    assert response.status_code == 502  # виноват движок, а не файл пользователя
    assert response.json() == {"detail": "MinerU вернул повреждённую картинку fig_1.png"}


# --- Пайплайн Б: JSON для 1С и Teamcenter ---------------------------------------------------


async def test_good_upd_is_valid(client, scan_pdf: bytes, mock: MockState):
    response = await extract(client, "upd.pdf", scan_pdf, "upd")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"status", "doc_type", "data", "validation"}
    assert (body["status"], body["doc_type"]) == ("completed", "upd")
    assert body["validation"] == {"is_valid": True, "warnings": []}
    assert body["data"]["seller"] == {
        "name": "ООО «Ромашка»",
        "inn": "7701234560",
        "kpp": "770101001",
        "address": "г. Москва, ул. Садовая, д. 1",
    }
    assert body["data"]["date"] == "2025-03-14"
    assert len(body["data"]["items"]) == 3
    assert body["data"]["items"][2]["price"] == 320.5  # числа, а не строки
    assert body["data"]["document_total"] == 34923.0


async def test_bad_upd_returns_200_with_warnings(client, scan_pdf: bytes):
    response = await extract(client, "bad_inn.pdf", scan_pdf, "upd")

    assert response.status_code == 200  # ошибка в данных — не сбой сервиса
    body = response.json()
    assert body["status"] == "completed"
    assert body["validation"] == {
        "is_valid": False,
        "warnings": [
            "ИНН продавца 7701234569 не прошёл контрольную сумму",
            "Сумма по строкам (34923.00) != Итого (35923.00)",
        ],
    }
    # Данные отданы как есть: документ уйдёт человеку на ручную проверку
    assert body["data"]["seller"]["inn"] == "7701234569"
    assert body["data"]["document_total"] == 35923.0


async def test_scan_upd_goes_through_paddle_then_qwen(client, scan_pdf: bytes, mock: MockState):
    await extract(client, "upd.pdf", scan_pdf, "upd")

    assert mock.calls[-2:] == ["paddle/load", "paddle/extract"]  # шаг 1: OCR на GPU 0
    (prompt,) = mock.prompts  # шаг 2: один запрос к Qwen3-VL
    assert "upd.pdf" in prompt
    assert "| 1 | Подшипник 6205-2RS |" in prompt  # в промпте — распознанный текст с таблицей
    (width, height) = mock.images[0]  # и картинка первой страницы, ужатая до лимита пикселей
    assert 0.98 * VLM_MAX_PIXELS <= width * height <= VLM_MAX_PIXELS


async def test_digital_upd_is_read_by_docling(client, digital_pdf: bytes, mock: MockState):
    body = (await extract(client, "upd.pdf", digital_pdf, "upd")).json()

    assert body["validation"]["is_valid"] is True
    assert mock.calls == []  # GPU 0 не нужен: у документа есть текстовый слой
    assert "Текст цифрового документа." in mock.prompts[0]


async def test_office_document_is_structured_without_image(client, mock: MockState):
    body = (await extract(client, "invoice.xlsx", b"PK\x03\x04 xlsx", "invoice")).json()

    assert body["validation"]["is_valid"] is True
    assert mock.images == []  # офисный файл в картинку не превращается


async def test_invoice(client, scan_pdf: bytes):
    body = (await extract(client, "invoice.pdf", scan_pdf, "invoice")).json()

    assert body["doc_type"] == "invoice"
    assert body["validation"] == {"is_valid": True, "warnings": []}
    assert body["data"]["bank"]["bik"] == "044525999"
    assert body["data"]["payment_due"] == "2025-03-26"


async def test_contract(client, digital_pdf: bytes):
    body = (await extract(client, "contract.pdf", digital_pdf, "contract")).json()

    assert body["validation"] == {"is_valid": True, "warnings": []}
    assert [party["role"] for party in body["data"]["parties"]] == ["Поставщик", "Покупатель"]
    assert body["data"]["amount"] == 34923.0


async def test_dlp_masks_values_inside_json(client, digital_pdf: bytes, mock: MockState):
    body = (await extract(client, "secret_contract.pdf", digital_pdf, "contract")).json()

    # Post-check: значение замаскировано, структура JSON цела, документ отправлен на ручную проверку
    assert (
        body["data"]["subject"] == "Поставка подшипников и комплектующих (паспорт представителя: [REDACTED])"
    )
    assert body["data"]["number"] == "ДП-17"
    assert body["validation"] == {
        "is_valid": False,
        "warnings": ["DLP замаскировал часть данных: документ нужно проверить вручную"],
    }
    assert PASSPORT not in json.dumps(body, ensure_ascii=False)
    # Pre-check: секреты из распознанного текста не дошли и до Qwen3-VL
    assert "SECRET_TOKEN" not in mock.prompts[0]
    assert PASSPORT not in mock.prompts[0]
    assert mock.prompts[0].count("[REDACTED]") == 2


async def test_dlp_checks_every_string_including_short_ones_and_lists(start_app, drawing_png: bytes):
    answers = {
        "DrawingStampSchema": json.dumps({"designation": "SECRET_TOKEN", "title": "Вал", "mass": "12.5"}),
        "TechRequirementsSchema": json.dumps(
            {"tech_requirements": ["1. Острые кромки притупить.", "2. 4510 123456"]}
        ),
    }
    async with start_app(http=engines_stub(vlm=answers)) as client:
        body = (await extract(client, "shaft.png", drawing_png, "drawing")).json()

    assert body["data"]["designation"] == "[REDACTED]"
    assert body["data"]["tech_requirements"] == ["1. Острые кромки притупить.", "2. [REDACTED]"]
    assert body["data"]["title"] == "Вал"
    assert body["validation"]["is_valid"] is False


async def test_dlp_masks_values_quoted_in_warnings(start_app, digital_pdf: bytes):
    # «ИНН» продавца похож на номер паспорта: DLP маскирует его и в данных, и в тексте предупреждения
    answer = json.dumps({**UPD, "seller": {**UPD["seller"], "inn": PASSPORT}})

    async with start_app(http=engines_stub(vlm=answer)) as client:
        body = (await extract(client, "upd.pdf", digital_pdf, "upd")).json()

    assert body["data"]["seller"]["inn"] == "[REDACTED]"
    assert body["validation"]["warnings"] == [
        "ИНН продавца [REDACTED] не прошёл контрольную сумму",
        "DLP замаскировал часть данных: документ нужно проверить вручную",
    ]
    assert PASSPORT not in json.dumps(body, ensure_ascii=False)


async def test_long_ocr_text_is_truncated_for_qwen(start_app, digital_pdf: bytes, mock: MockState):
    async with start_app(vlm_max_text_chars=10) as client:
        await extract(client, "upd.pdf", digital_pdf, "upd")

    assert mock.prompts[0].endswith("Распознанный текст документа:\n# Документ")  # первые 10 символов текста


async def test_drawing_warnings_come_from_both_zones(start_app, drawing_png: bytes):
    answers = {
        "DrawingStampSchema": json.dumps({"title": "Вал приводной"}),  # нет обязательного обозначения
        "TechRequirementsSchema": json.dumps({"tech_requirements": "не список"}),
    }
    async with start_app(http=engines_stub(vlm=answers)) as client:
        response = await extract(client, "shaft.png", drawing_png, "drawing")

    assert response.status_code == 200
    body = response.json()
    assert body["data"] == {"title": "Вал приводной", "tech_requirements": "не список"}  # как есть
    assert body["validation"]["is_valid"] is False
    stamp_warning, tech_warning = body["validation"]["warnings"]
    assert stamp_warning == "Поле designation: обязательное поле не заполнено"
    assert tech_warning.startswith("Поле tech_requirements: некорректное значение")


async def test_drawing_stamp_and_tech_requirements(client, drawing_png: bytes, mock: MockState):
    response = await extract(client, "shaft.png", drawing_png, "drawing")

    assert response.status_code == 200
    body = response.json()
    assert body["doc_type"] == "drawing"
    assert body["data"] == {**STAMP, "scale": None, "organization": None, **TECH_REQUIREMENTS}
    assert body["validation"] == {"is_valid": True, "warnings": []}
    assert mock.images == [(1350, 500), (1350, 860)]  # два запроса: кроп штампа и кроп техтребований
    assert ["основная надпись" in prompt for prompt in mock.prompts] == [True, False]
    assert mock.calls == []  # GPU 0 чертежам не нужен


async def test_pdf_drawing_is_rendered_at_300_dpi(client, mock: MockState):
    await extract(client, "shaft.pdf", make_pdf(["Чертёж вала"]), "drawing")

    # A4 при 300 dpi — 2480x3508 пикселей; штамп — правые 45 % ширины и нижние 25 % высоты
    assert mock.images[0] == pytest.approx((2480 * 0.45, 3508 * 0.25), abs=2)


async def test_huge_drawing_crops_fit_vlm_memory(client, mock: MockState):
    response = await extract(client, "sheet_a0.png", png(9000, 6000), "drawing")

    assert response.status_code == 200
    assert len(mock.images) == 2
    assert all(0.98 * VLM_MAX_PIXELS <= width * height <= VLM_MAX_PIXELS for width, height in mock.images)
    assert mock.oom_errors == 0


# --- Стресс: очередь к видеокартам ---------------------------------------------------------


def mixed_requests(client: httpx.AsyncClient, scan_pdf: bytes, drawing_png: bytes) -> list:
    """Пять одновременных запросов разных типов: mineru, paddle, чертёж и УПД."""
    return [
        process(client, "scan_1.pdf", scan_pdf),  # GPU 0 (mineru) -> GPU 1
        process(client, "table.pdf", scan_pdf),  # GPU 0 (paddle)
        extract(client, "shaft.png", drawing_png, "drawing"),  # GPU 1, два запроса
        extract(client, "upd.pdf", scan_pdf, "upd"),  # GPU 0 (paddle) -> GPU 1
        process(client, "scan_2.pdf", scan_pdf),  # GPU 0 (mineru) -> GPU 1
    ]


async def test_five_concurrent_requests_never_break_gpu_limits(
    client, scan_pdf: bytes, drawing_png: bytes, mock: MockState
):
    responses = await asyncio.gather(*mixed_requests(client, scan_pdf, drawing_png))

    assert [response.status_code for response in responses] == [200] * 5
    assert mock.oom_errors == 0  # мок ни разу не ответил «500 Out Of Memory»
    assert mock.gateway_rejects == 0  # и ни разу — «429 Too Many Concurrent Requests»
    assert mock.gpu0_peak == 1  # MinerU и PaddleOCR-VL не работали одновременно
    # Все задачи действительно дошли до движков
    assert mock.calls.count("mineru/extract") == 2
    assert mock.calls.count("paddle/extract") == 2
    assert len(mock.prompts) == 5  # 2 картинки из сканов + 2 кропа чертежа + 1 УПД
    # Результаты не перепутались между запросами
    scan_1, table, drawing, upd, scan_2 = (response.json() for response in responses)
    assert scan_1["metadata"]["engine_used"] == scan_2["metadata"]["engine_used"] == "mineru+qwen3vl"
    assert table["metadata"]["engine_used"] == "paddle"
    assert drawing["data"]["designation"] == "АБВГ.123456.001"
    assert upd["validation"]["is_valid"] is True


async def test_without_gpu_queue_same_requests_break_the_limits(
    client, scan_pdf: bytes, drawing_png: bytes, mock: MockState, monkeypatch: pytest.MonkeyPatch
):
    # Контрольный опыт: отключаем локи и убеждаемся, что мок действительно ловит нарушения
    monkeypatch.setattr(GPUResourceManager, "_exclusive", lambda self, name, wait=None: asyncio.timeout(None))

    responses = await asyncio.gather(*mixed_requests(client, scan_pdf, drawing_png))

    assert 502 in [response.status_code for response in responses]
    assert mock.oom_errors + mock.gateway_rejects > 0


async def test_thirty_concurrent_requests(client, scan_pdf: bytes, drawing_png: bytes, mock: MockState):
    requests = [request for _ in range(6) for request in mixed_requests(client, scan_pdf, drawing_png)]

    responses = await asyncio.gather(*requests)

    assert [response.status_code for response in responses] == [200] * 30
    assert (mock.oom_errors, mock.gateway_rejects, mock.gpu0_peak) == (0, 0, 1)
    assert len(mock.prompts) == 30


async def test_two_service_instances_share_the_gpus(
    start_app, scan_pdf: bytes, drawing_png: bytes, mock: MockState
):
    # Два экземпляра оркестратора с общими Redis и движками: очередь к видеокартам у них одна на двоих
    async with start_app() as first, start_app() as second:
        requests = [
            request for client in (first, second) for request in mixed_requests(client, scan_pdf, drawing_png)
        ]
        responses = await asyncio.gather(*requests)

    assert [response.status_code for response in responses] == [200] * 10
    assert (mock.oom_errors, mock.gateway_rejects, mock.gpu0_peak) == (0, 0, 1)


async def test_idle_paddle_gives_gpu0_back_to_mineru(start_app, scan_pdf: bytes, mock: MockState):
    async with start_app(paddle_idle_ttl_s=0.1, idle_check_interval_s=0.05) as client:
        await process(client, "table.pdf", scan_pdf)
        assert mock.loaded == {"paddle"}

        for _ in range(150):  # ждём фоновую проверку простоя (до 3 секунд)
            if mock.loaded == {"mineru"}:
                break
            await asyncio.sleep(0.02)

    assert mock.loaded == {"mineru"}
    assert mock.calls[-2:] == ["paddle/unload", "mineru/load"]
    assert mock.oom_errors == 0
