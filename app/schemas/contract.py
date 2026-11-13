"""Схема договора."""

import datetime as dt

from pydantic import BaseModel

from app.schemas.common import Counterparty, Number


class ContractParty(Counterparty):
    """Сторона договора."""

    role: str | None = None  # роль по договору: «Поставщик», «Покупатель», «Заказчик»...


class ContractSchema(BaseModel):
    """Договор: реквизиты, стороны и предмет."""

    number: str
    date: dt.date
    parties: list[ContractParty]
    subject: str  # предмет договора
    amount: Number | None = None  # сумма договора, если указана
    valid_until: dt.date | None = None  # срок действия
