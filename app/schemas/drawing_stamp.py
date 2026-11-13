"""Схемы чертежа: основная надпись (штамп) по ГОСТ 2.104 и технические требования (ТТ)."""

from pydantic import BaseModel


class DrawingStampCore(BaseModel):
    """Основная надпись чертежа (ЕСКД, ГОСТ 2.104): её Qwen3-VL читает с кропа штампа."""

    designation: str  # обозначение документа, например «АБВГ.711141.001»
    title: str  # наименование изделия
    material: str | None = None
    mass: str | None = None  # как в штампе: бывает с единицами или «см. табл.»
    scale: str | None = None  # масштаб, например «1:2»
    organization: str | None = None  # предприятие-разработчик


class TechRequirementsList(BaseModel):
    """Технические требования: Qwen3-VL читает их с кропа зоны над основной надписью."""

    requirements: list[str]  # пункты ТТ по порядку; пустой список, если требований нет


class DrawingStampSchema(TechRequirementsList, DrawingStampCore):
    """Итог по чертежу для Teamcenter: основная надпись вместе с техническими требованиями."""
