"""Интеграционные и стресс-тесты: оркестратор целиком по HTTP, движки — мок, Redis — в памяти.

Мок «ломается» при нарушении лимитов железа (HTTP 500 OOM, HTTP 429), поэтому нули в его
счётчиках доказывают, что оркестратор правильно выстраивает очередь к видеокартам.
"""

import asyncio
import base64
from urllib.parse import quote

import httpx
import pytest
from conftest import engines_stub, make_pdf, make_png

from app.core.gpu_manager import GPUResourceManager
from app.pipelines.markdown_pipe import FIGURE_PROMPT
from app.schemas.drawing_stamp import DrawingStampSchema
from mocks.main import FIGURE_PNG, MERMAID, SEAL_TEXT, STAMP, TECH_REQUIREMENTS, VLM_MAX_PIXELS, MockState


async def process(client: httpx.AsyncClient, filename: str, data: bytes, **headers: str) -> httpx.Response:
    """Запрос Open WebUI: сырые байты файла, имя — в заголовке (в percent-кодировке)."""
    return await client.put("/process", content=data, headers={"X-Filename": quote(filename), **headers})


async def extract(
    client: httpx.AsyncClient, filename: str, data: bytes, doc_type: str, **form: str
) -> httpx.Response:
    """Запрос 1С:ДО / Teamcenter: файл и тип документа в multipart-форме."""
    return await client.post(
        "/api/extract-structured", files={"file": (filename, data)}, data={"doc_type": doc_type, **form}
    )


async def wait_for_task(client: httpx.AsyncClient, task_id: str) -> dict:
    """Опрашивает задачу, пока она не завершится."""
    for _ in range(500):
        task = (await client.get(f"/api/tasks/{task_id}")).json()
        if task["status"] in ("completed", "failed"):
            return task
        await asyncio.sleep(0.02)
    raise AssertionError(f"Задача не завершилась: {task}")


# --- Пайплайн А: Markdown для Open WebUI --------------------------------------------------


async def test_digital_pdf_goes_to_docling_without_gpu(client, digital_pdf: bytes, mock: MockState):
    response = await process(client, "contract.pdf", digital_pdf)

    assert response.status_code == 200
    assert response.json() == {
        "page_content": "# Цифровой документ (contract.pdf)\nТаблица и текст из Docling.",
        "metadata": {"filename": "contract.pdf", "route": "DIGITAL_PDF", "engine_used": "docling"},
    }
    assert mock.calls == []  # GPU 0 не понадобился
    assert mock.prompts == []  # GPU 1 тоже


async def test_scan_gets_mermaid_instead_of_figure_placeholder(client, scan_pdf: bytes, mock: MockState):
    response = await process(client, "scan.pdf", scan_pdf)

    assert response.status_code == 200
    body = response.json()
    assert body["metadata"] == {"filename": "scan.pdf", "route": "SCAN_PDF", "engine_used": "mineru+qwen3-vl"}
    assert (
        body["page_content"]
        == f"# Скан scan.pdf (MinerU)\n\nРаспознанный текст.\n\n{MERMAID}\n\nТекст после схемы."
    )
    assert "![Схема]" not in body["page_content"]  # блок Mermaid от Qwen3-VL встал на место плейсхолдера
    assert mock.calls == ["paddle/unload", "mineru/load", "mineru/extract"]
    assert mock.prompts == [FIGURE_PROMPT]
    assert (
        FIGURE_PROMPT
        == "Переведи схему на изображении в блок кода Mermaid или дай краткое техническое описание."
    )
    assert mock.images == [(64, 64)]  # в Qwen3-VL ушла картинка, вырезанная MinerU


@pytest.mark.parametrize(
    "filename", ["report.docx", "price.xlsx", "slides.pptx", "page.html", "README.md", "notes.txt"]
)
async def test_office_and_text_files_go_to_docling(client, mock: MockState, filename: str):
    body = (await process(client, filename, b"PK\x03\x04 content")).json()

    assert body["metadata"] == {"filename": filename, "route": "DIGITAL_OFFICE", "engine_used": "docling"}
    assert filename in body["page_content"]
    assert mock.calls == []


async def test_photo_is_recognized_by_mineru(client, mock: MockState):
    body = (await process(client, "photo.jpg", make_png(800, 600))).json()

    assert body["metadata"] == {
        "filename": "photo.jpg",
        "route": "SCAN_IMAGE",
        "engine_used": "mineru+qwen3-vl",
    }
    assert mock.calls[-2:] == ["mineru/load", "mineru/extract"]


async def test_ocr_engine_header_sends_scan_to_paddle(client, scan_pdf: bytes, mock: MockState):
    response = await process(client, "contract_with_seal.pdf", scan_pdf, **{"X-OCR-Engine": "paddle"})

    body = response.json()
    assert body["metadata"] == {
        "filename": "contract_with_seal.pdf",
        "route": "SCAN_PDF",
        "engine_used": "paddle",
    }
    assert "| № | Наименование |" in body["page_content"]
    assert SEAL_TEXT in body["page_content"]  # PaddleOCR-VL читает оттиски печатей
    assert mock.calls[-2:] == ["paddle/load", "paddle/extract"]
    assert mock.prompts == []  # Qwen3-VL здесь не нужен


@pytest.mark.parametrize("engine", ["auto", "mineru"])
async def test_default_ocr_engine_is_mineru(client, scan_pdf: bytes, mock: MockState, engine: str):
    body = (await process(client, "scan.pdf", scan_pdf, **{"X-OCR-Engine": engine})).json()

    assert body["metadata"]["engine_used"] == "mineru+qwen3-vl"
    assert "paddle/extract" not in mock.calls


async def test_ocr_engine_header_does_not_affect_digital_documents(
    client, digital_pdf: bytes, mock: MockState
):
    body = (await process(client, "contract.pdf", digital_pdf, **{"X-OCR-Engine": "paddle"})).json()

    assert body["metadata"]["engine_used"] == "docling"
    assert mock.calls == []


async def test_unknown_ocr_engine_is_rejected(client, scan_pdf: bytes):
    response = await process(client, "scan.pdf", scan_pdf, **{"X-OCR-Engine": "tesseract"})
    assert response.status_code == 422


async def test_gpu0_is_free_while_figures_are_described(client, scan_pdf: bytes, mock: MockState, redis):
    await process(client, "scan.pdf", scan_pdf)

    # Картинки описывает GPU 1: лок GPU 0 к этому времени уже отпущен, а в конце свободны оба
    assert await redis.exists("lock:gpu0", "lock:gpu1") == 0
    assert mock.gpu0_peak == 1


async def test_each_figure_description_replaces_its_placeholder(start_app, scan_pdf: bytes):
    mineru = {
        "markdown": "Введение.\n\n![](fig_1.png)\n\nСередина.\n\n![Рис. 2 — схема](fig+2.png)\n\nКонец.",
        "figures": dict.fromkeys(("fig_1.png", "fig+2.png", "fig_3.png"), FIGURE_PNG),
    }
    # Спецсимволы регулярных выражений в ответе модели — обычный текст
    description = "Схема: A -> B. Путь C:\\temp\\1 и группа \\g<0>"

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
        "metadata": {"filename": "scan.pdf", "route": "SCAN_PDF", "engine_used": "mineru"},
    }


async def test_large_figure_is_resized_before_qwen(client, scan_pdf: bytes, mock: MockState, monkeypatch):
    big_figure = base64.b64encode(make_png(3000, 2000)).decode()  # 6 Мп: в VRAM GPU 1 не поместится
    monkeypatch.setattr("mocks.main.FIGURE_PNG", big_figure)

    response = await process(client, "scan.pdf", scan_pdf)

    assert response.status_code == 200
    assert mock.images == [(1376, 896)]  # не больше 1,28 Мп, стороны кратны 32
    assert mock.oom_errors == 0


async def test_pixel_limit_setting_applies_to_every_image_for_qwen(
    start_app, scan_pdf: bytes, drawing_a1: bytes, mock: MockState, monkeypatch
):
    monkeypatch.setattr("mocks.main.FIGURE_PNG", base64.b64encode(make_png(1000, 1000)).decode())

    async with start_app(max_pixels=100_000) as client:
        await process(client, "scan.pdf", scan_pdf)  # картинка, вырезанная из скана
        await extract(client, "upd.pdf", scan_pdf, "upd")  # обзорная картинка первой страницы
        await extract(client, "shaft.png", drawing_a1, "drawing")  # два кропа чертежа

    assert len(mock.images) == 4
    assert all(60_000 < width * height <= 100_000 for width, height in mock.images)


async def test_broken_figure_from_mineru_is_engine_error(start_app, scan_pdf: bytes):
    mineru = {"markdown": "![](fig_1.png)", "figures": {"fig_1.png": "bm90IGFuIGltYWdl"}}

    async with start_app(http=engines_stub(mineru=mineru)) as client:
        response = await process(client, "scan.pdf", scan_pdf)

    assert response.status_code == 502  # виноват движок, а не файл пользователя
    assert response.json() == {"detail": "MinerU вернул повреждённую картинку fig_1.png"}


async def test_vlm_model_family_is_reported_in_engine_used(start_app, scan_pdf: bytes):
    async with start_app(vlm_model="minicpm-v:8b") as client:
        body = (await process(client, "scan.pdf", scan_pdf)).json()
    assert body["metadata"]["engine_used"] == "mineru+minicpm-v"


# --- Текстовые файлы: отдаются как есть, без движков --------------------------------------------


@pytest.mark.parametrize(
    ("filename", "data", "text"),
    [
        (
            "price.csv",
            "Товар;Цена\nВал приводной;12500,00\n".encode(),
            "Товар;Цена\nВал приводной;12500,00\n",
        ),
        ("выгрузка.xml", "<Документ Номер='128'/>".encode("cp1251"), "<Документ Номер='128'/>"),
        ("script.py", b"print('hello')\n", "print('hello')\n"),
    ],
)
async def test_text_file_is_returned_as_is(client, mock: MockState, filename: str, data: bytes, text: str):
    response = await process(client, filename, data)

    assert response.status_code == 200
    assert response.json() == {
        "page_content": text,
        "metadata": {"filename": filename, "route": "TEXT", "engine_used": "none"},
    }
    assert (mock.calls, mock.prompts) == ([], [])  # ни один движок не понадобился


async def test_binary_file_of_unknown_format_is_415(client):
    response = await process(client, "archive.zip", b"PK\x03\x04\x14\x00\x00\x00\x08\x00")

    assert response.status_code == 415
    assert response.json() == {"detail": "Формат файла «archive.zip» не поддерживается"}


# --- Пайплайн Б: JSON для 1С:ДО и Teamcenter --------------------------------------------------


async def test_good_upd_is_valid(client, scan_pdf: bytes):
    response = await extract(client, "upd.pdf", scan_pdf, "upd")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"status", "doc_type", "data", "validation"}
    assert (body["status"], body["doc_type"]) == ("completed", "upd")
    assert body["validation"] == {"is_valid": True, "warnings": []}
    assert body["data"]["seller"] == {
        "name": "ООО «Ромашка»",
        "inn": "7707083893",
        "kpp": "770701001",
        "address": "г. Москва, ул. Садовая, д. 1",
    }
    assert body["data"]["date"] == "2025-03-14"
    assert len(body["data"]["items"]) == 3
    assert body["data"]["items"][2]["price"] == 320.5  # числа, а не строки
    assert body["data"]["total_with_vat"] == 34923.0
    assert "bank" not in body["data"]  # в схеме УПД банковских реквизитов нет


async def test_bad_upd_returns_200_with_warnings(client, scan_pdf: bytes):
    response = await extract(client, "bad_inn.pdf", scan_pdf, "upd")

    assert response.status_code == 200  # ошибка в данных — не сбой сервиса
    body = response.json()
    assert body["status"] == "completed"
    assert body["validation"] == {
        "is_valid": False,
        "warnings": [
            "ИНН продавца 7701234569 не прошёл контрольную сумму",
            "Сумма по строкам (1200.00) != Итого по документу (1500.00)",
        ],
    }
    # Данные отданы как есть: документ уйдёт оператору на ручную проверку
    assert body["data"]["seller"]["inn"] == "7701234569"
    assert body["data"]["total_with_vat"] == 1500.0


async def test_scan_upd_goes_through_paddle_then_qwen(client, scan_pdf: bytes, mock: MockState):
    await extract(client, "upd.pdf", scan_pdf, "upd")

    assert mock.calls[-2:] == ["paddle/load", "paddle/extract"]  # шаг 1: скан УПД читает PaddleOCR-VL
    (prompt,) = mock.prompts  # шаг 2: один запрос к Qwen3-VL
    assert "upd.pdf" in prompt
    assert "| 1 | Подшипник 6205-2RS |" in prompt  # в промпте — распознанный текст с таблицей
    assert SEAL_TEXT in prompt
    # ...и обзорная картинка первой страницы: A4 при 150 dpi (1240x1754) после smart-resize
    assert mock.images == [(928, 1344)]


async def test_scan_invoice_is_read_by_mineru(client, scan_pdf: bytes, mock: MockState):
    body = (await extract(client, "invoice.pdf", scan_pdf, "invoice")).json()

    assert body["validation"] == {"is_valid": True, "warnings": []}
    assert mock.calls[-2:] == ["mineru/load", "mineru/extract"]
    assert len(mock.prompts) == 1  # только сборка JSON: схемы из скана здесь не описываются
    assert "Скан invoice.pdf (MinerU)" in mock.prompts[0]


async def test_scan_with_seals_is_read_by_paddle(client, scan_pdf: bytes, mock: MockState):
    body = (await extract(client, "invoice.pdf", scan_pdf, "invoice", has_seals="true")).json()

    assert body["validation"]["is_valid"] is True
    assert mock.calls[-2:] == ["paddle/load", "paddle/extract"]
    assert SEAL_TEXT in mock.prompts[0]


async def test_digital_upd_is_read_by_docling(client, digital_pdf: bytes, mock: MockState):
    body = (await extract(client, "upd.pdf", digital_pdf, "upd", has_seals="true")).json()

    assert body["validation"]["is_valid"] is True
    assert mock.calls == []  # GPU 0 не нужен: у документа есть текстовый слой
    assert "Таблица и текст из Docling." in mock.prompts[0]
    assert mock.images == [(928, 1344)]


async def test_office_document_is_structured_without_image(client, mock: MockState):
    body = (await extract(client, "invoice.xlsx", b"PK\x03\x04 xlsx", "invoice")).json()

    assert body["validation"]["is_valid"] is True
    assert mock.images == []  # офисный файл в картинку не превращается


async def test_text_document_is_structured_from_its_own_text(client, mock: MockState):
    xml = "<Файл><Документ НомерСчФ='128' ДатаСчФ='14.03.2025'/></Файл>".encode("cp1251")

    body = (await extract(client, "upd.xml", xml, "upd")).json()

    assert body["validation"]["is_valid"] is True
    assert mock.calls == []  # распознавать нечего: текст уже есть
    assert "<Документ НомерСчФ='128' ДатаСчФ='14.03.2025'/>" in mock.prompts[0]
    assert mock.images == []


async def test_image_scan_is_structured_with_the_image_itself(client, mock: MockState):
    body = (await extract(client, "upd.jpg", make_png(2000, 1000), "upd")).json()

    assert body["validation"]["is_valid"] is True
    assert mock.images == [(1600, 800)]


async def test_overview_dpi_setting(start_app, scan_pdf: bytes, mock: MockState):
    async with start_app(overview_dpi=72) as client:
        await extract(client, "upd.pdf", scan_pdf, "upd")

    assert mock.images == [(576, 832)]  # A4 при 72 dpi — 595x842; стороны округлены вниз до кратных 32


async def test_invoice(client, scan_pdf: bytes):
    body = (await extract(client, "invoice.pdf", scan_pdf, "invoice")).json()

    assert body["doc_type"] == "invoice"
    assert body["validation"] == {"is_valid": True, "warnings": []}
    assert body["data"]["bank"]["bik"] == "044525225"
    assert body["data"]["payment_due"] == "2025-03-28"
    assert "upd_status" not in body["data"]


async def test_long_ocr_text_is_truncated_for_qwen(start_app, digital_pdf: bytes, mock: MockState):
    async with start_app(vlm_max_text_chars=10) as client:
        body = (await extract(client, "upd.pdf", digital_pdf, "upd")).json()

    assert mock.prompts[0].endswith("Распознанный текст документа:\n# Цифровой")  # первые 10 символов текста
    assert body["validation"] == {
        "is_valid": False,  # модель видела не весь документ: его нужно проверить вручную
        "warnings": ["Документ слишком длинный: модель получила только первые 10 из 57 символов текста"],
    }


async def test_drawing_stamp_and_tech_requirements(client, drawing_a1: bytes, mock: MockState):
    response = await extract(client, "shaft.png", drawing_a1, "drawing")

    assert response.status_code == 200
    body = response.json()
    assert (body["status"], body["doc_type"]) == ("completed", "drawing")
    assert body["data"] == {**STAMP, **TECH_REQUIREMENTS}
    assert DrawingStampSchema.model_validate(body["data"]).designation == "АБВГ.711141.001"
    assert body["validation"] == {"is_valid": True, "warnings": []}
    # Два запроса к Qwen3-VL: кроп штампа (2000x840) и кроп техтребований (2000x1176) после smart-resize
    assert mock.images == [(1728, 704), (1472, 864)]
    assert ["основная надпись" in prompt for prompt in mock.prompts] == [True, False]
    assert all("shaft.png" in prompt for prompt in mock.prompts)
    assert mock.calls == []  # GPU 0 чертежам не нужен


async def test_portrait_drawing(client, drawing_a4: bytes, mock: MockState):
    body = (await extract(client, "shaft_a4.png", drawing_a4, "drawing")).json()

    assert body["validation"]["is_valid"] is True
    assert mock.images == [(1792, 704), (1504, 832)]  # штамп и ТТ — во всю ширину листа


async def test_pdf_drawing_is_rendered_at_300_dpi(client, mock: MockState):
    await extract(client, "shaft.pdf", make_pdf(["Чертёж вала"]), "drawing")

    # A4 при 300 dpi — 2480x3508 пикселей, то есть те же кропы, что у книжного чертежа-картинки
    assert mock.images == [(1792, 704), (1504, 832)]


async def test_drawing_hint_wins_over_file_extension(client, drawing_a1: bytes):
    body = (await extract(client, "shaft.docx", drawing_a1, "drawing")).json()  # картинка с чужим расширением
    assert body["data"]["designation"] == "АБВГ.711141.001"


async def test_huge_drawing_crops_fit_vlm_memory(client, mock: MockState):
    response = await extract(client, "sheet_a0.png", make_png(9000, 6000), "drawing")

    assert response.status_code == 200
    assert len(mock.images) == 2
    for width, height in mock.images:
        assert width % 32 == height % 32 == 0
        assert 0.93 * VLM_MAX_PIXELS <= width * height <= VLM_MAX_PIXELS
    assert mock.oom_errors == 0


async def test_async_drawing_is_polled_until_completed(client, drawing_a1: bytes):
    response = await extract(client, "shaft.png", drawing_a1, "drawing", async_mode="true")

    assert response.status_code == 202
    queued = response.json()
    assert set(queued) == {"task_id", "status"}
    assert queued["status"] == "queued"

    task = await wait_for_task(client, queued["task_id"])
    assert set(task) == {"task_id", "status", "doc_type", "data", "validation"}
    assert (task["task_id"], task["status"], task["doc_type"]) == (queued["task_id"], "completed", "drawing")
    assert task["data"] == {**STAMP, **TECH_REQUIREMENTS}
    assert task["validation"] == {"is_valid": True, "warnings": []}


# --- Стресс: очередь к видеокартам ---------------------------------------------------------


def six_requests(client: httpx.AsyncClient, scan_pdf: bytes, drawing: bytes) -> list:
    """Шесть одновременных запросов: 2 скана (mineru), 2 УПД с печатями (paddle) и 2 чертежа."""
    return [
        process(client, "scan_1.pdf", scan_pdf),  # GPU 0 (mineru) -> GPU 1
        extract(client, "upd_1.pdf", scan_pdf, "upd", has_seals="true"),  # GPU 0 (paddle) -> GPU 1
        extract(client, "shaft_1.png", drawing, "drawing"),  # GPU 1, два запроса
        process(client, "scan_2.pdf", scan_pdf),
        extract(client, "upd_2.pdf", scan_pdf, "upd", has_seals="true"),
        extract(client, "shaft_2.png", drawing, "drawing"),
    ]


async def test_six_concurrent_requests_never_break_gpu_limits(
    client, scan_pdf: bytes, drawing_a1: bytes, mock: MockState
):
    responses = await asyncio.gather(*six_requests(client, scan_pdf, drawing_a1))

    assert [response.status_code for response in responses] == [200] * 6
    assert mock.oom_errors == 0  # мок ни разу не ответил «500 Simulated OOM»
    assert mock.gpu1_rejects == 0  # и ни разу — «429 GPU 1 Busy»
    assert mock.gpu0_peak == 1  # MinerU и PaddleOCR-VL не работали одновременно
    # Все задачи дошли до движков, модели в VRAM GPU 0 переключались
    assert mock.calls.count("mineru/extract") == 2
    assert mock.calls.count("paddle/extract") == 2
    assert {"mineru/load", "mineru/unload", "paddle/load", "paddle/unload"} <= set(mock.calls)
    assert len(mock.prompts) == 8  # 2 схемы из сканов + 2 УПД + 2 x 2 кропа чертежей
    # Результаты не перепутались между запросами
    scan_1, upd_1, drawing_1, scan_2, upd_2, drawing_2 = (response.json() for response in responses)
    assert (scan_1["metadata"]["filename"], scan_2["metadata"]["filename"]) == ("scan_1.pdf", "scan_2.pdf")
    assert scan_1["metadata"]["engine_used"] == scan_2["metadata"]["engine_used"] == "mineru+qwen3-vl"
    assert upd_1["validation"] == upd_2["validation"] == {"is_valid": True, "warnings": []}
    assert drawing_1["data"] == drawing_2["data"] == {**STAMP, **TECH_REQUIREMENTS}


async def test_without_gpu_queue_same_requests_break_the_limits(
    client, scan_pdf: bytes, drawing_a1: bytes, mock: MockState, monkeypatch: pytest.MonkeyPatch
):
    # Контрольный опыт: отключаем локи и убеждаемся, что мок действительно ловит нарушения
    monkeypatch.setattr(
        GPUResourceManager, "_exclusive", lambda self, name, blocking=True: asyncio.timeout(None)
    )
    mock.delay_scale = 10  # «движки» работают по 1,5 секунды: запросы заведомо пересекаются на любой машине

    responses = await asyncio.gather(*six_requests(client, scan_pdf, drawing_a1))

    assert 502 in [response.status_code for response in responses]
    assert mock.oom_errors + mock.gpu1_rejects > 0


async def test_thirty_concurrent_requests(client, scan_pdf: bytes, drawing_a1: bytes, mock: MockState):
    requests = [request for _ in range(5) for request in six_requests(client, scan_pdf, drawing_a1)]

    responses = await asyncio.gather(*requests)

    assert [response.status_code for response in responses] == [200] * 30
    assert (mock.oom_errors, mock.gpu1_rejects, mock.gpu0_peak) == (0, 0, 1)
    assert len(mock.prompts) == 40


async def test_two_service_instances_share_the_gpus(
    start_app, scan_pdf: bytes, drawing_a1: bytes, mock: MockState
):
    # Два экземпляра оркестратора с общими Redis и движками: очередь к видеокартам у них одна на двоих
    async with start_app() as first, start_app() as second:
        requests = [
            request for client in (first, second) for request in six_requests(client, scan_pdf, drawing_a1)
        ]
        responses = await asyncio.gather(*requests)

    assert [response.status_code for response in responses] == [200] * 12
    assert (mock.oom_errors, mock.gpu1_rejects, mock.gpu0_peak) == (0, 0, 1)


async def test_idle_paddle_gives_gpu0_back_to_mineru(start_app, scan_pdf: bytes, mock: MockState):
    async with start_app(paddle_idle_ttl_sec=0.1, gpu0_check_interval_sec=0.05) as client:
        await process(client, "scan.pdf", scan_pdf, **{"X-OCR-Engine": "paddle"})
        assert mock.loaded == {"paddle"}

        for _ in range(150):  # ждём фоновую проверку простоя (до 3 секунд)
            if mock.loaded == {"mineru"}:
                break
            await asyncio.sleep(0.02)

    assert mock.loaded == {"mineru"}
    assert mock.calls[-2:] == ["paddle/unload", "mineru/load"]
    assert mock.oom_errors == 0
