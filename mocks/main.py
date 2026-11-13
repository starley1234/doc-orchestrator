"""mock-engines: заглушка внешних ML-сервисов для локального стенда и автотестов.

Имитирует по HTTP Docling, MinerU, PaddleOCR-VL, Gateway с Qwen3-VL и Regex-DLP.
Кроме ответов воспроизводит ограничения железа, чтобы тесты ловили ошибки оркестратора:
  * GPU 0: вторая модель в VRAM -> HTTP 500 Out Of Memory (Simulated);
  * GPU 1: второй одновременный запрос -> HTTP 429;
  * GPU 1: картинка крупнее 1 280 000 пикселей -> HTTP 500 Out Of Memory (Simulated).

Особые имена файлов меняют ответы:
  * «bad_inn» — УПД с неверным ИНН продавца и несходящимся итогом;
  * «secret»  — в распознанном тексте и в JSON договора появляются секрет и номер паспорта.
"""

import asyncio
import base64
import json
import re
import struct
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, UploadFile
from pydantic import BaseModel

# --- Тестовые данные -------------------------------------------------------------------

SELLER = {
    "name": "ООО «Ромашка»",
    "inn": "7701234560",
    "kpp": "770101001",
    "address": "г. Москва, ул. Садовая, д. 1",
}
BUYER = {
    "name": "АО «Вектор»",
    "inn": "5024156782",
    "kpp": "502401001",
    "address": "г. Красногорск, ул. Ленина, д. 5",
}


def _item(name: str, qty: float, price: float) -> dict[str, Any]:
    """Строка табличной части с НДС 20 %: все суммы сходятся."""
    amount = round(qty * price, 2)
    vat = round(amount * 0.2, 2)
    return {
        "name": name,
        "unit": "шт",
        "qty": qty,
        "price": price,
        "amount_without_vat": amount,
        "vat_rate": "20%",
        "vat_amount": vat,
        "row_total": round(amount + vat, 2),
    }


ITEMS = [
    _item("Подшипник 6205-2RS", 10, 250.00),
    _item("Вал приводной АБВГ.123456.001", 2, 12500.00),
    _item("Смазка Литол-24, 0,8 кг", 5, 320.50),
]
TOTAL = round(sum(item["row_total"] for item in ITEMS), 2)  # 34923.00

UPD = {
    "number": "128",
    "date": "2025-03-14",
    "seller": SELLER,
    "buyer": BUYER,
    "items": ITEMS,
    "document_total": TOTAL,
    "upd_status": 1,
    "basis": "Договор поставки № ДП-17 от 10.01.2025",
}
# «Галлюцинация» модели: неверная контрольная цифра ИНН и итог, не равный сумме строк
BAD_UPD = {**UPD, "seller": {**SELLER, "inn": "7701234569"}, "document_total": TOTAL + 1000}

INVOICE = {
    "number": "57",
    "date": "2025-03-12",
    "seller": SELLER,
    "buyer": BUYER,
    "items": ITEMS,
    "document_total": TOTAL,
    "bank": {
        "bank_name": "АО «Тестбанк»",
        "bik": "044525999",
        "account": "40702810900000001234",
        "corr_account": "30101810400000000999",
    },
    "payment_due": "2025-03-26",
}

CONTRACT = {
    "number": "ДП-17",
    "date": "2025-01-10",
    "parties": [{"role": "Поставщик", **SELLER}, {"role": "Покупатель", **BUYER}],
    "subject": "Поставка подшипников и комплектующих",
    "amount": TOTAL,
    "valid_until": "2025-12-31",
}

STAMP = {
    "designation": "АБВГ.123456.001",
    "title": "Вал приводной",
    "material": "Сталь 45 ГОСТ 1050-2013",
    "mass": "12.5",
}
TECH_REQUIREMENTS = {
    "tech_requirements": [
        "1. 42...48 HRC.",
        "2. Неуказанные предельные отклонения размеров: H14, h14, ±IT14/2.",
        "3. Острые кромки притупить.",
    ]
}

MERMAID = "```mermaid\ngraph TD; A-->B;\n```"
VLM_MAX_PIXELS = 1_280_000  # картинка крупнее не помещается в 8 ГБ VRAM GPU 1
PASSPORT = "4510 123456"  # тестовый номер паспорта
SECRET_TEXT = f"\n\nКлюч доступа: SECRET_TOKEN. Паспорт: {PASSPORT}."
PNG_1X1 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR42mP4//8/AAX+Av4zEpUUAAAAAElFTkSuQmCC"

MINERU_MARKDOWN = "# Отчёт (MinerU)\n\nРаспознанный текст скана.\n\n![fig_1](fig_1.png)\n\nТекст после схемы."


def _upd_markdown() -> str:
    """Markdown-таблица УПД — так её «распознаёт» PaddleOCR-VL."""
    rows = [
        f"| {n} | {i['name']} | {i['unit']} | {i['qty']:g} | {i['price']:.2f} "
        f"| {i['amount_without_vat']:.2f} | {i['vat_amount']:.2f} | {i['row_total']:.2f} |"
        for n, i in enumerate(ITEMS, start=1)
    ]
    return "\n".join(
        [
            f"# УПД № {UPD['number']} от 14.03.2025 (PaddleOCR-VL)",
            "",
            f"Продавец: {SELLER['name']}, ИНН {SELLER['inn']}, КПП {SELLER['kpp']}",
            f"Покупатель: {BUYER['name']}, ИНН {BUYER['inn']}, КПП {BUYER['kpp']}",
            "",
            "| № | Наименование | Ед. | Кол-во | Цена | Сумма без НДС | НДС | Всего |",
            "|---|---|---|---|---|---|---|---|",
            *rows,
            "",
            f"Всего к оплате: {TOTAL:.2f}",
        ]
    )


# --- Состояние стенда -------------------------------------------------------------------


@dataclass
class MockState:
    """«Железо» стенда и счётчики нарушений (их читают автотесты и GET /state)."""

    delay_s: float = 0.1  # имитация времени работы движка
    loaded: set[str] = field(default_factory=set)  # модели в VRAM GPU 0
    gateway_lock: asyncio.Lock = field(default_factory=asyncio.Lock)  # GPU 1: один запрос за раз
    oom_errors: int = 0  # сколько раз сымитирован OOM (GPU 0 или GPU 1)
    gateway_rejects: int = 0  # сколько раз сымитирован 429 на GPU 1
    gpu0_jobs: int = 0  # задач на GPU 0 прямо сейчас
    gpu0_peak: int = 0  # максимум одновременных задач на GPU 0
    calls: list[str] = field(default_factory=list)  # журнал вызовов движков GPU 0
    prompts: list[str] = field(default_factory=list)  # промпты, дошедшие до Qwen3-VL
    images: list[tuple[int, int]] = field(default_factory=list)  # размеры картинок, дошедших до Qwen3-VL


def get_state(request: Request) -> MockState:
    return request.app.state.mock


StateDep = Annotated[MockState, Depends(get_state)]
Gpu0Model = Literal["mineru", "paddle"]
router = APIRouter()


def _secret_tail(filename: str | None) -> str:
    """Для файлов с «secret» в имени дописывает в текст секрет и номер паспорта."""
    return SECRET_TEXT if "secret" in (filename or "").lower() else ""


def _claim_vram(state: MockState, model: str) -> None:
    """Загружает модель в VRAM GPU 0. Если там уже лежит другая — имитирует OOM."""
    if state.loaded - {model}:
        state.oom_errors += 1
        raise HTTPException(500, "Out Of Memory (Simulated)")
    state.loaded.add(model)


# --- Docling (CPU) ----------------------------------------------------------------------


@router.post("/docling/convert")
async def docling_convert(file: UploadFile, state: StateDep) -> dict[str, str]:
    """Цифровой документ -> Markdown."""
    await asyncio.sleep(state.delay_s)
    return {"markdown": "# Документ (Docling)\nТекст цифрового документа." + _secret_tail(file.filename)}


# --- MinerU и PaddleOCR-VL (делят GPU 0) --------------------------------------------------


@router.post("/{model}/load")
async def load_model(model: Gpu0Model, state: StateDep) -> dict[str, Any]:
    """Загрузка модели в VRAM."""
    state.calls.append(f"{model}/load")
    _claim_vram(state, model)
    await asyncio.sleep(state.delay_s)
    return {"model": model, "loaded": True}


@router.post("/{model}/unload")
async def unload_model(model: Gpu0Model, state: StateDep) -> dict[str, Any]:
    """Выгрузка модели из VRAM."""
    state.calls.append(f"{model}/unload")
    state.loaded.discard(model)
    return {"model": model, "loaded": False}


@router.post("/{model}/extract")
async def extract(model: Gpu0Model, file: UploadFile, state: StateDep) -> dict[str, Any]:
    """OCR скана. Модель подгружается лениво, поэтому конфликт VRAM проверяется и здесь."""
    state.calls.append(f"{model}/extract")
    _claim_vram(state, model)
    state.gpu0_jobs += 1
    state.gpu0_peak = max(state.gpu0_peak, state.gpu0_jobs)
    try:
        await asyncio.sleep(state.delay_s)
    finally:
        state.gpu0_jobs -= 1

    tail = _secret_tail(file.filename)
    if model == "mineru":
        # Markdown с плейсхолдером картинки + сама вырезанная картинка (PNG 1x1 в base64)
        return {"markdown": MINERU_MARKDOWN + tail, "images": [{"name": "fig_1.png", "data": PNG_1X1}]}
    return {"markdown": _upd_markdown() + tail}


# --- Gateway + Qwen3-VL (GPU 1) -----------------------------------------------------------


def _parts(body: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Части всех сообщений. Поле content — строка или список частей, как в OpenAI API."""
    for message in body.get("messages", []):
        content = message.get("content", "")
        yield from [{"type": "text", "text": content}] if isinstance(content, str) else content


def _png_size(data_url: str) -> tuple[int, int]:
    """Ширина и высота PNG из data URL: они записаны в заголовке IHDR. Не PNG — (0, 0)."""
    raw = base64.b64decode(data_url.partition(",")[2])
    return struct.unpack(">II", raw[16:24]) if raw[:8] == b"\x89PNG\r\n\x1a\n" else (0, 0)


def _vlm_answer(schema: dict[str, Any] | None, prompt: str) -> str:
    """Ответ «модели»: JSON под переданную схему или Mermaid, если просят описать картинку."""
    if not schema:
        return MERMAID
    contract = CONTRACT
    if "secret" in prompt.lower():
        contract = {**CONTRACT, "subject": f"{CONTRACT['subject']} (паспорт представителя: {PASSPORT})"}
    answers = {
        "UpdSchema": BAD_UPD if "bad_inn" in prompt.lower() else UPD,
        "InvoiceSchema": INVOICE,
        "ContractSchema": contract,
        "DrawingStampSchema": STAMP,
        "TechRequirementsSchema": TECH_REQUIREMENTS,
    }
    return json.dumps(answers.get(schema.get("title", ""), {}), ensure_ascii=False)


@router.post("/gateway/v1/chat/completions")
async def chat_completions(body: dict[str, Any], state: StateDep) -> dict[str, Any]:
    """OpenAI-совместимый чат. JSON Schema ответа передаётся в поле format (как в Ollama)."""
    if state.gateway_lock.locked():
        state.gateway_rejects += 1
        raise HTTPException(429, "Too Many Concurrent Requests on GPU 1")
    async with state.gateway_lock:
        parts = list(_parts(body))
        prompt = "\n".join(part.get("text", "") for part in parts if part.get("type") == "text")
        images = [_png_size(part["image_url"]["url"]) for part in parts if part.get("type") == "image_url"]
        state.prompts.append(prompt)
        state.images += images
        if any(width * height > VLM_MAX_PIXELS for width, height in images):
            state.oom_errors += 1
            raise HTTPException(500, "Out Of Memory (Simulated)")
        await asyncio.sleep(state.delay_s)
        answer = _vlm_answer(body.get("format"), prompt)
    return {
        "id": f"chatcmpl-{uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model"),
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}
        ],
    }


# --- Regex-DLP ----------------------------------------------------------------------------

# Секретный токен или номер паспорта РФ (серия и номер через пробел)
SENSITIVE_RE = re.compile(r"SECRET_TOKEN|\b\d{2} ?\d{2} \d{6}\b")


class InspectRequest(BaseModel):
    text: str


@router.post("/dlp/inspect")
async def dlp_inspect(body: InspectRequest) -> dict[str, Any]:
    """Маскирует секреты и персональные данные в тексте."""
    text, hits = SENSITIVE_RE.subn("[REDACTED]", body.text)
    return {"text": text, "masked": hits > 0}


# --- Служебное ----------------------------------------------------------------------------


@router.get("/health")
async def health() -> dict[str, str]:
    """Проверка живости для docker compose."""
    return {"status": "ok"}


@router.get("/state")
async def state_snapshot(state: StateDep) -> dict[str, Any]:
    """Что сейчас в VRAM и были ли нарушения лимитов GPU."""
    return {
        "mineru_loaded": "mineru" in state.loaded,
        "paddle_loaded": "paddle" in state.loaded,
        "oom_errors": state.oom_errors,
        "gateway_rejects": state.gateway_rejects,
        "gpu0_peak": state.gpu0_peak,
    }


def create_app(delay_s: float = 0.1) -> FastAPI:
    """Создаёт приложение с чистым состоянием (тесты поднимают свой экземпляр на каждый тест)."""
    app = FastAPI(title="mock-engines")
    app.state.mock = MockState(delay_s=delay_s)
    app.include_router(router)
    return app


app = create_app()
