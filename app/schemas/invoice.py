"""Схема счёта на оплату."""

import datetime as dt

from pydantic import BaseModel

from app.schemas.common import TradeDocument


class BankDetails(BaseModel):
    """Платёжные реквизиты получателя."""

    bank_name: str | None = None
    bik: str
    account: str  # расчётный счёт
    corr_account: str | None = None  # корреспондентский счёт


class InvoiceSchema(TradeDocument):
    """Счёт на оплату."""

    bank: BankDetails
    payment_due: dt.date | None = None  # оплатить до
