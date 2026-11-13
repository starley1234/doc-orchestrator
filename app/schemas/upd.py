"""Схема УПД / счёта-фактуры."""

from typing import Literal

from app.schemas.common import TradeDocument


class UpdSchema(TradeDocument):
    """Универсальный передаточный документ (УПД) или счёт-фактура."""

    upd_status: Literal[1, 2] | None = (
        None  # 1 — счёт-фактура и передаточный документ, 2 — только передаточный
    )
    basis: str | None = None  # основание передачи: договор, заказ
