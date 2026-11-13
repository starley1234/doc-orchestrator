"""Схемы чертежа: основная надпись (штамп) по ГОСТ 2.104 и технические требования."""

from pydantic import BaseModel


class DrawingStampSchema(BaseModel):
    """Основная надпись чертежа (ЕСКД, ГОСТ 2.104)."""

    designation: str  # обозначение документа, например «АБВГ.123456.001»
    title: str  # наименование изделия
    material: str | None = None
    mass: str | None = None  # как в штампе: бывает с единицами или «см. табл.»
    scale: str | None = None  # масштаб, например «1:2»
    organization: str | None = None  # предприятие-разработчик


class TechRequirementsSchema(BaseModel):
    """Технические требования — текстовая зона над основной надписью."""

    tech_requirements: list[str]
