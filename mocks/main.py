"""mock-engines: заглушка ML-движков для локального стенда и автотестов.

Имитирует по HTTP Docling-serve, MinerU, PaddleOCR-VL и Ollama с Qwen3-VL.
Кроме ответов воспроизводит ограничения железа, чтобы тесты ловили ошибки оркестратора:
  * GPU 0: вторая модель в VRAM -> HTTP 500 Simulated OOM;
  * GPU 0: распознавание без загруженной модели -> HTTP 500 «not loaded»;
  * GPU 1: второй одновременный запрос -> HTTP 429;
  * GPU 1: картинка крупнее 1 280 000 пикселей -> HTTP 500 Simulated OOM.

Особое имя файла: «bad_inn» в промпте — УПД или счёт с неверным ИНН продавца и несходящимся итогом.
"""

import asyncio
import base64
import json
import struct
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, UploadFile

# --- Тестовые данные -------------------------------------------------------------------

SELLER = {
    "name": "ООО «Ромашка»",
    "inn": "7707083893",
    "kpp": "770701001",
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
    _item("Вал приводной АБВГ.711141.001", 2, 12500.00),
    _item("Смазка Литол-24, 0,8 кг", 5, 320.50),
]
TOTAL = round(sum(item["row_total"] for item in ITEMS), 2)  # 34923.00

# Поля УПД и счёта на оплату вместе: «модель» отдаёт те из них, что есть в запрошенной схеме
DOCUMENT = {
    "number": "128",
    "date": "2025-03-14",
    "seller": SELLER,
    "buyer": BUYER,
    "items": ITEMS,
    "total_with_vat": TOTAL,
    "upd_status": 1,
    "basis": "Договор поставки № ДП-17 от 10.01.2025",
    "bank": {
        "bank_name": "ПАО «Тестбанк»",
        "bik": "044525225",
        "account": "40702810900000001234",
        "corr_account": "30101810400000000225",
    },
    "payment_due": "2025-03-28",
}
# «Галлюцинация» модели: неверная контрольная цифра ИНН и итог, не равный сумме строк (1200.00)
BAD_DOCUMENT = {
    **DOCUMENT,
    "seller": {**SELLER, "inn": "7701234569"},
    "items": [_item("Подшипник 6205-2RS", 2, 500.00)],
    "total_with_vat": 1500.00,
}

STAMP = {
    "designation": "АБВГ.711141.001",
    "title": "Вал приводной",
    "material": "Сталь 45 ГОСТ 1050-2013",
    "mass": "12.5",
    "scale": "1:1",
    "organization": "АО КБ Маш",
}
TECH_REQUIREMENTS = {
    "requirements": [
        "1. Неуказанные предельные отклонения размеров H14, h14, +/-IT14/2.",
        "2. Острые кромки притупить R0,5.",
    ]
}

MERMAID = "```mermaid\ngraph TD;\n  A[Вход] --> B[Обработка];\n```"
SEAL_TEXT = "ПЕЧАТЬ: ООО Ромашка ОГРН 1027700132195"
VLM_MAX_PIXELS = 1_280_000  # картинка крупнее не помещается в 8 ГБ VRAM GPU 1


# «Схема», вырезанная MinerU из скана: белая картинка PNG 64x64 в base64
FIGURE_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAAAAACPAi4CAAAAK0lEQVR42u3MMREAAAgEoNf+"
    "nTWEmwcBqMlNRyAQCAQCgUAgEAgEAsHzYAFQLQF/OJ8+2wAAAABJRU5ErkJggg=="
)
MINERU_MARKDOWN = (
    "# Скан {filename} (MinerU)\n\nРаспознанный текст.\n\n![Схема](fig_1.png)\n\nТекст после схемы."
)


def _upd_markdown() -> str:
    """Markdown-таблица УПД и текст круглой печати — так их «распознаёт» PaddleOCR-VL."""
    rows = [
        f"| {n} | {i['name']} | {i['unit']} | {i['qty']:g} | {i['price']:.2f} "
        f"| {i['amount_without_vat']:.2f} | {i['vat_amount']:.2f} | {i['row_total']:.2f} |"
        for n, i in enumerate(ITEMS, start=1)
    ]
    return "\n".join(
        [
            f"# УПД № {DOCUMENT['number']} от 14.03.2025",
            "",
            f"Продавец: {SELLER['name']}, ИНН {SELLER['inn']}, КПП {SELLER['kpp']}",
            f"Покупатель: {BUYER['name']}, ИНН {BUYER['inn']}, КПП {BUYER['kpp']}",
            "",
            "| № | Наименование | Ед. | Кол-во | Цена | Сумма без НДС | НДС | Всего |",
            "|---|---|---|---|---|---|---|---|",
            *rows,
            "",
            f"Всего к оплате: {TOTAL:.2f}",
            "",
            SEAL_TEXT,
        ]
    )


# --- Состояние стенда -------------------------------------------------------------------


@dataclass
class MockState:
    """«Железо» стенда и счётчики нарушений (их читают автотесты и GET /mock/state)."""

    delay_scale: float = 1.0  # множитель задержек: тесты ускоряют «работу» движков
    loaded: set[str] = field(default_factory=set)  # модели в VRAM GPU 0: «mineru», «paddle»
    gpu1_lock: asyncio.Lock = field(default_factory=asyncio.Lock)  # GPU 1: один запрос за раз
    oom_errors: int = 0  # сколько раз сымитирован OOM (GPU 0 или GPU 1)
    gpu1_rejects: int = 0  # сколько раз сымитирован 429 на GPU 1
    gpu0_jobs: int = 0  # задач на GPU 0 прямо сейчас
    gpu0_peak: int = 0  # максимум одновременных задач на GPU 0
    calls: list[str] = field(default_factory=list)  # журнал вызовов движков GPU 0
    prompts: list[str] = field(default_factory=list)  # промпты, дошедшие до Qwen3-VL
    images: list[tuple[int, int]] = field(default_factory=list)  # размеры картинок, дошедших до Qwen3-VL

    async def work(self, seconds: float) -> None:
        """Имитация времени работы движка."""
        await asyncio.sleep(seconds * self.delay_scale)

    def oom(self, detail: str) -> HTTPException:
        self.oom_errors += 1
        return HTTPException(500, detail)


def get_state(request: Request) -> MockState:
    return request.app.state.mock


StateDep = Annotated[MockState, Depends(get_state)]
Gpu0Model = Literal["mineru", "paddle"]
NEIGHBOUR: dict[str, str] = {"mineru": "paddle", "paddle": "mineru"}  # модели, которые делят GPU 0
ENGINE_NAMES = {"mineru": "MinerU", "paddle": "PaddleOCR"}
router = APIRouter()


# --- Docling (CPU) ----------------------------------------------------------------------


@router.post("/docling/convert")
async def docling_convert(file: UploadFile, state: StateDep) -> dict[str, str]:
    """Цифровой документ -> Markdown."""
    await state.work(0.05)
    return {"markdown": f"# Цифровой документ ({file.filename})\nТаблица и текст из Docling."}


# --- MinerU и PaddleOCR-VL (делят GPU 0) --------------------------------------------------


@router.post("/{model}/load")
async def load_model(model: Gpu0Model, state: StateDep) -> dict[str, Any]:
    """Загрузка модели в VRAM. Если там ещё лежит соседняя модель — памяти не хватит."""
    state.calls.append(f"{model}/load")
    neighbour = NEIGHBOUR[model]
    if neighbour in state.loaded:
        raise state.oom(f"Simulated OOM: {ENGINE_NAMES[neighbour]} is still in GPU 0 VRAM!")
    state.loaded.add(model)
    return {"model": model, "loaded": True}


@router.post("/{model}/unload")
async def unload_model(model: Gpu0Model, state: StateDep) -> dict[str, Any]:
    """Выгрузка модели из VRAM."""
    state.calls.append(f"{model}/unload")
    state.loaded.discard(model)
    return {"model": model, "loaded": False}


@router.post("/{model}/extract")
async def extract(model: Gpu0Model, file: UploadFile, state: StateDep) -> dict[str, Any]:
    """OCR скана. Модель должна быть загружена, а соседняя — выгружена."""
    state.calls.append(f"{model}/extract")
    if model not in state.loaded:
        raise HTTPException(500, f"{ENGINE_NAMES[model]} not loaded")
    if NEIGHBOUR[model] in state.loaded:
        raise state.oom("Simulated OOM")
    state.gpu0_jobs += 1
    state.gpu0_peak = max(state.gpu0_peak, state.gpu0_jobs)
    try:
        await state.work(0.15)
    finally:
        state.gpu0_jobs -= 1

    if model == "paddle":
        return {"markdown": _upd_markdown()}
    # MinerU: Markdown с плейсхолдером картинки и сама вырезанная картинка
    return {"markdown": MINERU_MARKDOWN.format(filename=file.filename), "figures": {"fig_1.png": FIGURE_PNG}}


# --- Ollama + Qwen3-VL (GPU 1) ------------------------------------------------------------


def _png_size(image_b64: str) -> tuple[int, int]:
    """Ширина и высота PNG: они записаны в заголовке IHDR. Не PNG — (0, 0)."""
    raw = base64.b64decode(image_b64)
    return struct.unpack(">II", raw[16:24]) if raw[:8] == b"\x89PNG\r\n\x1a\n" else (0, 0)


def _vlm_answer(schema: dict[str, Any] | None, prompt: str) -> str:
    """Ответ «модели»: JSON под переданную схему или Mermaid, если просят описать картинку."""
    if not schema:
        return MERMAID
    fields = schema.get("properties", {})
    if "designation" in fields:  # основная надпись чертежа
        answer = STAMP
    elif "requirements" in fields:  # технические требования
        answer = TECH_REQUIREMENTS
    else:  # УПД или счёт на оплату
        document = BAD_DOCUMENT if "bad_inn" in prompt else DOCUMENT
        answer = {name: value for name, value in document.items() if name in fields}
    return json.dumps(answer, ensure_ascii=False)


@router.post("/api/chat")
async def ollama_chat(body: dict[str, Any], state: StateDep) -> dict[str, Any]:
    """Чат Ollama. JSON Schema ответа передаётся в поле format, картинки — base64 в поле images."""
    if state.gpu1_lock.locked():
        state.gpu1_rejects += 1
        raise HTTPException(429, "GPU 1 Busy: Concurrent VLM request!")
    async with state.gpu1_lock:
        messages = body.get("messages", [])
        prompt = "\n".join(message.get("content", "") for message in messages)
        images = [_png_size(image) for message in messages for image in message.get("images", [])]
        state.prompts.append(prompt)
        state.images += images
        if any(width * height > VLM_MAX_PIXELS for width, height in images):
            raise state.oom("Simulated OOM: image is too large for GPU 1 VRAM!")
        await state.work(0.15)
        answer = _vlm_answer(body.get("format"), prompt)
    return {"model": body.get("model"), "message": {"role": "assistant", "content": answer}, "done": True}


# --- Служебное ----------------------------------------------------------------------------


@router.get("/health")
async def health() -> dict[str, str]:
    """Проверка живости для docker compose."""
    return {"status": "ok"}


@router.get("/mock/state")
async def state_snapshot(state: StateDep) -> dict[str, Any]:
    """Что сейчас в VRAM GPU 0 и были ли нарушения лимитов GPU."""
    return {
        "mineru_loaded": "mineru" in state.loaded,
        "paddle_loaded": "paddle" in state.loaded,
        "oom_errors": state.oom_errors,
        "gpu1_rejects": state.gpu1_rejects,
        "gpu0_peak": state.gpu0_peak,
    }


def create_app(delay_scale: float = 1.0) -> FastAPI:
    """Создаёт приложение с чистым состоянием (тесты поднимают свой экземпляр на каждый тест)."""
    app = FastAPI(title="mock-engines")
    app.state.mock = MockState(delay_scale=delay_scale)
    app.include_router(router)
    return app


app = create_app()
