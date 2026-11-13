"""Схема УПД / счёта-фактуры."""

from typing import Literal

from app.schemas.common import TradeDocument


class UpdSchema(TradeDocument):
    """Универсальный передаточный документ (УПД) или счёт-фактура."""

    # Статус УПД: 1 — счёт-фактура и передаточный документ, 2 — только передаточный документ
    upd_status: Literal[1, 2] | None = None
    basis: str | None = None  # основание передачи: договор, заказ
