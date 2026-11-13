"""Пайплайн Б: документ -> строгий JSON с валидацией для 1С:ДО и Teamcenter."""

import json
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from app.core.aio import run_cpu
from app.core.classifier import RouteType
from app.core.image_ops import crop_drawing_zones, first_page_image, smart_resize
from app.pipelines.base import Engines
from app.schemas.common import Counterparty, DocType, StructuredResult, TaskStatus, ValidationResult
from app.schemas.drawing_stamp import DrawingStampCore, DrawingStampSchema, TechRequirementsList
from app.schemas.invoice import InvoiceSchema
from app.schemas.upd import UpdSchema
from app.validators.checksums import validate_bik, validate_inn, validate_kpp
from app.validators.math_checks import validate_invoice_math

ModelT = TypeVar("ModelT", bound=BaseModel)
VlmRequest = tuple[type[BaseModel], str, bytes | None]  # схема ответа (поле format), промпт, картинка

# Тип документа -> (схема JSON, название для промпта)
DOCUMENTS: dict[DocType, tuple[type[UpdSchema] | type[InvoiceSchema], str]] = {
    DocType.UPD: (UpdSchema, "УПД / счёт-фактура"),
    DocType.INVOICE: (InvoiceSchema, "счёт на оплату"),
}

EXTRACT_PROMPT = (
    "Извлеки данные документа «{filename}» ({doc_name}) и верни JSON строго по заданной схеме.\n"
    "Бери значения только из документа и ничего не выдумывай: если необязательного поля нет, ставь null.\n"
    "Числа пиши без пробелов, с точкой; даты — в формате ГГГГ-ММ-ДД.\n\n"
    "Распознанный текст документа:\n{text}"
)
STAMP_PROMPT = (
    "На изображении — основная надпись (штамп) чертежа «{filename}» по ГОСТ 2.104. "
    "Извлеки её поля и верни JSON строго по заданной схеме. Чего в штампе нет — ставь null."
)
TECH_REQ_PROMPT = (
    "На изображении — зона технических требований чертежа «{filename}». "
    "Перепиши все пункты по порядку и верни JSON строго по заданной схеме. Нет требований — пустой список."
)


class StructuredPipeline:
    def __init__(self, engines: Engines) -> None:
        self._engines = engines

    async def run(
        self, file_bytes: bytes, filename: str, doc_type: DocType, has_seals: bool = False
    ) -> StructuredResult:
        """Извлекает JSON документа и проверяет его. Ошибки в данных — не сбой, а warnings.

        has_seals=True отправляет скан в PaddleOCR-VL: он читает оттиски печатей и сложные таблицы.
        """
        if doc_type is DocType.DRAWING:
            data, warnings = await self._extract_drawing(file_bytes, filename)
        else:
            data, warnings = await self._extract_document(file_bytes, filename, doc_type, has_seals)
        validation = ValidationResult(is_valid=not warnings, warnings=warnings)
        return StructuredResult(
            status=TaskStatus.COMPLETED, doc_type=doc_type, data=data, validation=validation
        )

    async def _extract_drawing(self, file_bytes: bytes, filename: str) -> tuple[dict[str, Any], list[str]]:
        """Чертёж: штамп и техтребования читаются с кропов двумя запросами к Qwen3-VL."""
        settings = self._engines.settings
        page = await run_cpu(first_page_image, file_bytes, settings.drawing_dpi)
        zones = await run_cpu(crop_drawing_zones, page, settings.max_pixels)
        requests: list[VlmRequest] = [
            (DrawingStampCore, STAMP_PROMPT.format(filename=filename), zones["stamp"]),
            (TechRequirementsList, TECH_REQ_PROMPT.format(filename=filename), zones["tech_req"]),
        ]
        _, data, warnings = await self._ask_json(DrawingStampSchema, requests)
        return data, warnings

    async def _extract_document(
        self, file_bytes: bytes, filename: str, doc_type: DocType, has_seals: bool
    ) -> tuple[dict[str, Any], list[str]]:
        """УПД или счёт: OCR -> сборка JSON в Qwen3-VL -> валидаторы."""
        engines, settings = self._engines, self._engines.settings
        schema, doc_name = DOCUMENTS[doc_type]

        # Шаг 1. Точный текст и таблицы. Скан УПД и документы с печатями читает PaddleOCR-VL
        ocr = await engines.ocr(file_bytes, filename, use_paddle=has_seals or doc_type is DocType.UPD)

        # Шаг 2. Сборка JSON: текст и обзорная картинка первой страницы -> Qwen3-VL
        image = None  # офисный или текстовый файл в картинку не превращаем: модели хватит текста
        if ocr.route in (RouteType.DIGITAL_PDF, RouteType.SCAN_PDF, RouteType.SCAN_IMAGE):
            page = await run_cpu(first_page_image, file_bytes, settings.overview_dpi)
            image = await run_cpu(smart_resize, page, settings.max_pixels)
        text = ocr.markdown[: settings.vlm_max_text_chars]
        prompt = EXTRACT_PROMPT.format(filename=filename, doc_name=doc_name, text=text)
        document, data, warnings = await self._ask_json(schema, [(schema, prompt, image)])

        # Шаг 3. Пост-валидация: контрольные суммы реквизитов и арифметика
        if document:
            warnings += _validate(document)
        if len(ocr.markdown) > len(text):
            warnings.append(
                f"Документ слишком длинный: модель получила только первые {len(text)} "
                f"из {len(ocr.markdown)} символов текста"
            )
        return data, warnings

    async def _ask_json(
        self, schema: type[ModelT], requests: list[VlmRequest]
    ) -> tuple[ModelT | None, dict[str, Any], list[str]]:
        """Запросы к Qwen3-VL с JSON Schema ответа; ответы складываются в один объект schema.

        Возвращает документ (None, если ответ не подошёл под схему), данные для ответа и предупреждения.
        """
        raw: dict[str, Any] = {}
        for part_schema, prompt, image in requests:
            answer = await self._engines.ask_vlm(prompt, image, part_schema.model_json_schema())
            try:
                part = json.loads(answer)
            except ValueError:
                part = None
            if not isinstance(part, dict):
                return None, raw, ["Модель вернула не JSON-объект: данные извлечь не удалось"]
            raw |= part
        try:
            document = schema.model_validate(raw)
        except ValidationError as exc:
            # Ответ не подошёл под схему: отдаём данные как есть, причины — в предупреждения
            return None, raw, _schema_warnings(exc)
        return document, document.model_dump(mode="json"), []


# --- Валидация ----------------------------------------------------------------------------


def _validate(document: UpdSchema | InvoiceSchema) -> list[str]:
    """Пост-валидация УПД и счёта: реквизиты сторон, БИК и сходимость сумм."""
    warnings = _check_party(document.seller, "продавца") + _check_party(document.buyer, "покупателя")
    if isinstance(document, InvoiceSchema) and not validate_bik(document.bank.bik):
        warnings.append(f"БИК банка {document.bank.bik} имеет неверный формат")
    return warnings + validate_invoice_math(document)


def _check_party(party: Counterparty, whose: str) -> list[str]:
    """Реквизиты одной стороны, например: «ИНН продавца 7701234569 не прошёл контрольную сумму»."""
    warnings = []
    if not validate_inn(party.inn):
        warnings.append(f"ИНН {whose} {party.inn} не прошёл контрольную сумму")
    if party.kpp and not validate_kpp(party.kpp):
        warnings.append(f"КПП {whose} {party.kpp} имеет неверный формат")
    return warnings


def _schema_warnings(error: ValidationError) -> list[str]:
    """Ошибки разбора по схеме понятным языком: какое поле и что с ним не так."""
    warnings = []
    for item in error.errors():
        field = ".".join(str(part) for part in item["loc"]) or "документ"
        reason = (
            "обязательное поле не заполнено"
            if item["type"] == "missing"
            else f"некорректное значение ({item['msg']})"
        )
        warnings.append(f"Поле {field}: {reason}")
    return warnings
