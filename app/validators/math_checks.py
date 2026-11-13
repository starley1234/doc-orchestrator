"""Арифметическая сверка табличной части: строки и итог документа должны сходиться."""

from decimal import Decimal

from app.schemas.invoice import InvoiceSchema
from app.schemas.upd import UpdSchema

TOLERANCE = Decimal("0.05")  # допуск на копейки округления


def validate_invoice_math(data: UpdSchema | InvoiceSchema) -> list[str]:
    """Возвращает список расхождений понятным языком (пустой, если всё сошлось)."""
    errors: list[str] = []
    for n, row in enumerate(data.items, start=1):
        amount = row.qty * row.price
        if abs(amount - row.amount_without_vat) > TOLERANCE:
            errors.append(
                f"Строка {n}: кол-во × цена ({amount:.2f}) != сумма без НДС ({row.amount_without_vat:.2f})"
            )
        total = row.amount_without_vat + row.vat_amount
        if abs(total - row.row_total) > TOLERANCE:
            errors.append(
                f"Строка {n}: сумма без НДС + НДС ({total:.2f}) != сумма по строке ({row.row_total:.2f})"
            )

    rows_total = sum((row.row_total for row in data.items), Decimal(0))
    if abs(rows_total - data.document_total) > TOLERANCE:
        errors.append(f"Сумма по строкам ({rows_total:.2f}) != Итого ({data.document_total:.2f})")
    return errors
