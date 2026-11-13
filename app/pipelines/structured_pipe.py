"""Пайплайн Б: документ -> строгий JSON с валидацией для 1С и Teamcenter."""

import json
from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel, ValidationError

from app.core.aio import run_cpu
from app.core.classifier import RouteType, classify_file
from app.core.errors import UnsupportedFileError
from app.core.image_ops import crop_drawing_zones, first_page_image
from app.pipelines.base import Engines
from app.schemas.common import Counterparty, DocType, StructuredResult, TaskStatus, ValidationResult
from app.schemas.contract import ContractSchema
from app.schemas.drawing_stamp import DrawingStampSchema, TechRequirementsSchema
from app.schemas.invoice import InvoiceSchema
from app.schemas.upd import UpdSchema
from app.validators.checksums import validate_bik, validate_inn, validate_kpp
from app.validators.math_checks import validate_invoice_math

# Тип документа -> (схема JSON, название для промпта)
DOCUMENTS: dict[DocType, tuple[type[BaseModel], str]] = {
    DocType.UPD: (UpdSchema, "УПД / счёт-фактура"),
    DocType.INVOICE: (InvoiceSchema, "счёт на оплату"),
    DocType.CONTRACT: (ContractSchema, "договор"),
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

    async def run(self, file_bytes: bytes, filename: str, doc_type: DocType) -> StructuredResult:
        """Извлекает JSON документа и проверяет его. Ошибки в данных — не сбой, а warnings."""
        if doc_type is DocType.DRAWING:
            data, warnings = await self._extract_drawing(file_bytes, filename)
        else:
            data, warnings = await self._extract_document(file_bytes, filename, doc_type)

        # DLP post-check всего, что уходит наружу: предупреждения тоже цитируют значения из документа
        checked, masked = await self._dlp_post_check({"data": data, "warnings": warnings})
        data, warnings = checked["data"], checked["warnings"]
        if masked:
            warnings.append("DLP замаскировал часть данных: документ нужно проверить вручную")
        validation = ValidationResult(is_valid=not warnings, warnings=warnings)
        return StructuredResult(
            status=TaskStatus.COMPLETED, doc_type=doc_type, data=data, validation=validation
        )

    async def _extract_drawing(self, file_bytes: bytes, filename: str) -> tuple[dict[str, Any], list[str]]:
        """Чертёж: штамп и техтребования читаются с кропов двумя запросами к Qwen3-VL."""
        settings = self._engines.settings
        if classify_file(file_bytes, filename, DocType.DRAWING) is not RouteType.DRAWING:
            raise UnsupportedFileError("Чертёж принимается только как PDF или изображение")
        page = await run_cpu(first_page_image, file_bytes, settings.drawing_dpi)
        zones = await run_cpu(crop_drawing_zones, page, settings.max_pixels)

        prompt = STAMP_PROMPT.format(filename=filename)
        _, stamp, stamp_warnings = await self._ask_json(DrawingStampSchema, prompt, zones["stamp"])
        prompt = TECH_REQ_PROMPT.format(filename=filename)
        _, tech_req, tech_warnings = await self._ask_json(TechRequirementsSchema, prompt, zones["tech_req"])
        return {**stamp, **tech_req}, stamp_warnings + tech_warnings

    async def _extract_document(
        self, file_bytes: bytes, filename: str, doc_type: DocType
    ) -> tuple[dict[str, Any], list[str]]:
        """УПД, счёт или договор: OCR -> структурирование в Qwen3-VL -> валидаторы."""
        engines = self._engines
        schema, doc_name = DOCUMENTS[doc_type]

        # Шаг 1. OCR: точный текст и таблицы. Цифровой документ читает Docling, скан — PaddleOCR-VL
        route = await run_cpu(classify_file, file_bytes, filename, doc_type)
        if route in (RouteType.DIGITAL_OFFICE, RouteType.DIGITAL_PDF):
            text = await engines.docling.convert(file_bytes, filename)
        else:
            text = await engines.paddle_ocr(file_bytes, filename)

        # Шаг 2. Структурирование: текст и картинка первой страницы -> JSON по схеме
        image = None  # офисный файл в картинку не превращаем: модели хватит текста
        if route is not RouteType.DIGITAL_OFFICE:
            image = await run_cpu(first_page_image, file_bytes, engines.settings.render_dpi)
        text = text[: engines.settings.vlm_max_text_chars]
        prompt = EXTRACT_PROMPT.format(filename=filename, doc_name=doc_name, text=text)
        document, data, warnings = await self._ask_json(schema, prompt, image)

        # Шаг 3. Валидация: контрольные суммы реквизитов и арифметика
        if document:
            warnings += _validate(document)
        return data, warnings

    async def _ask_json(
        self, schema: type[BaseModel], prompt: str, image: bytes | None
    ) -> tuple[BaseModel | None, dict[str, Any], list[str]]:
        """Запрос к Qwen3-VL с JSON Schema ответа.

        Возвращает документ (None, если ответ не подошёл под схему), данные для ответа и предупреждения.
        """
        answer = await self._engines.ask_vlm(prompt, image, schema.model_json_schema())
        try:
            raw = json.loads(answer)
        except ValueError:
            return None, {}, ["Модель вернула не JSON: данные извлечь не удалось"]
        try:
            document = schema.model_validate_json(answer)
        except ValidationError as exc:
            # Ответ не подошёл под схему: отдаём данные как есть, причины — в предупреждения
            return None, raw if isinstance(raw, dict) else {}, _schema_warnings(exc)
        return document, document.model_dump(mode="json"), []

    async def _dlp_post_check(self, payload: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """Маскирует секреты в JSON. Каждая строка проверяется отдельно, поэтому структура не страдает."""
        checked = {text: await self._engines.dlp.inspect(text) for text in set(_strings(payload))}
        if not any(result.masked for result in checked.values()):
            return payload, False
        return _replace_strings(payload, {text: result.text for text, result in checked.items()}), True


# --- Валидация ----------------------------------------------------------------------------


def _validate(document: BaseModel) -> list[str]:
    """Пост-валидация документа: реквизиты сторон, БИК и сходимость сумм."""
    warnings: list[str] = []
    if isinstance(document, ContractSchema):
        for number, party in enumerate(document.parties, start=1):
            warnings += _check_party(party, f"стороны {number}")
    if isinstance(document, UpdSchema | InvoiceSchema):
        warnings += _check_party(document.seller, "продавца") + _check_party(document.buyer, "покупателя")
        if isinstance(document, InvoiceSchema) and not validate_bik(document.bank.bik):
            warnings.append(f"БИК банка {document.bank.bik} имеет неверный формат")
        warnings += validate_invoice_math(document)
    return warnings


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


# --- Обход JSON ---------------------------------------------------------------------------


def _strings(value: Any) -> Iterator[str]:
    """Все непустые строковые значения JSON (ключи словарей не считаются)."""
    if isinstance(value, str) and value:
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _replace_strings(value: Any, replacements: dict[str, str]) -> Any:
    """Копия JSON, в которой строковые значения заменены по словарю replacements."""
    if isinstance(value, str):
        return replacements.get(value, value)
    if isinstance(value, dict):
        return {key: _replace_strings(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_strings(item, replacements) for item in value]
    return value
