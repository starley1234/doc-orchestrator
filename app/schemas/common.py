"""Общие типы: перечисления, реквизиты сторон, строки табличной части и модели ответов API."""

import datetime as dt
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, PlainSerializer, WithJsonSchema

# Деньги и количества считаем в Decimal (без погрешностей float),
# а в JSON-ответе и в JSON Schema для VLM это обычное число.
Number = Annotated[
    Decimal,
    PlainSerializer(float, return_type=float, when_used="json"),
    WithJsonSchema({"type": "number"}),
]


class DocType(StrEnum):
    """Тип документа для структурированного извлечения."""

    UPD = "upd"
    INVOICE = "invoice"
    CONTRACT = "contract"
    DRAWING = "drawing"


class TaskStatus(StrEnum):
    """Состояние задачи извлечения."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# --- Части документов ---------------------------------------------------------------------


class Counterparty(BaseModel):
    """Сторона документа: юрлицо или ИП."""

    name: str
    inn: str
    kpp: str | None = None  # у ИП КПП нет
    address: str | None = None


class LineItem(BaseModel):
    """Строка табличной части: товар или услуга."""

    name: str
    unit: str | None = None  # единица измерения
    qty: Number
    price: Number  # цена за единицу без НДС
    amount_without_vat: Number
    vat_rate: str | None = None  # «20%», «10%», «без НДС»
    vat_amount: Number
    row_total: Number  # сумма по строке с НДС


class TradeDocument(BaseModel):
    """Общая часть УПД и счёта на оплату: стороны, табличная часть, итог."""

    number: str
    date: dt.date
    seller: Counterparty
    buyer: Counterparty
    items: list[LineItem]
    document_total: Number  # всего к оплате с НДС


# --- Ответы API ---------------------------------------------------------------------------


class ProcessMetadata(BaseModel):
    filename: str
    engine_used: str  # движки через «+», например «mineru+qwen3vl»
    dlp_masked: bool  # замаскировал ли DLP что-нибудь в тексте


class ProcessResponse(BaseModel):
    """Ответ PUT /process в формате Open WebUI External Document Loader."""

    page_content: str
    metadata: ProcessMetadata


class ValidationResult(BaseModel):
    """Итог пост-валидации. При is_valid=False документ уходит на ручную верификацию."""

    is_valid: bool
    warnings: list[str]


class StructuredResult(BaseModel):
    """Ответ POST /api/extract-structured и GET /api/tasks/{task_id}. Незаполненные поля не выводятся."""

    task_id: str | None = None
    status: TaskStatus
    doc_type: DocType | None = None
    data: dict[str, Any] | None = None  # JSON по схеме документа
    validation: ValidationResult | None = None
    error: str | None = None  # причина для status=failed
