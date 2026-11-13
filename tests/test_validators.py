"""Пост-валидаторы: контрольные суммы реквизитов и арифметика табличной части."""

import json

import pytest

from app.schemas.invoice import InvoiceSchema
from app.schemas.upd import UpdSchema
from app.validators.checksums import validate_bik, validate_inn, validate_kpp
from app.validators.math_checks import validate_invoice_math
from mocks.main import DOCUMENT

# --- ИНН ----------------------------------------------------------------------------------


@pytest.mark.parametrize("inn", ["7707083893", "7701234560", "5024156782"])
def test_valid_inn_of_legal_entity(inn: str):
    assert validate_inn(inn)


@pytest.mark.parametrize("inn", ["500100732259", "771234567859"])
def test_valid_inn_of_individual(inn: str):
    assert validate_inn(inn)


@pytest.mark.parametrize(
    "inn",
    [
        "7701234569",  # 10 цифр: неверная контрольная цифра (пример из ТЗ)
        "7707083983",  # 10 цифр: переставлены соседние цифры
        "500100732249",  # 12 цифр: неверна первая контрольная цифра
        "500100732258",  # 12 цифр: неверна вторая контрольная цифра
        "770123456",  # 9 цифр
        "77012345600",  # 11 цифр
        "77012345AB",  # буквы
        " 7701234560",  # пробел
        "0000000000",  # контрольная сумма сходится, но кода региона 00 нет
        "７７０１２３４５６０",  # полноширинные цифры Юникода
        "",
        None,
    ],
)
def test_invalid_inn(inn: str | None):
    assert not validate_inn(inn)


# --- КПП и БИК ----------------------------------------------------------------------------


@pytest.mark.parametrize("kpp", ["770701001", "7701AB001", "502401001"])
def test_valid_kpp(kpp: str):
    assert validate_kpp(kpp)


@pytest.mark.parametrize("kpp", ["77010100", "7701010010", "7701ab001", "АБ0101001", "770101 01", "", None])
def test_invalid_kpp(kpp: str | None):
    assert not validate_kpp(kpp)


@pytest.mark.parametrize("bik", ["044525225", "044525999"])
def test_valid_bik(bik: str):
    assert validate_bik(bik)


@pytest.mark.parametrize("bik", ["144525225", "04452522", "0445252250", "04452522A", "", None])
def test_invalid_bik(bik: str | None):
    assert not validate_bik(bik)


# --- Арифметика табличной части ------------------------------------------------------------


def upd(rows: list[tuple[float, float, float, float, float]], total: float) -> UpdSchema:
    """УПД из строк (кол-во, цена, сумма без НДС, НДС, всего). Разбирается из JSON — как ответ VLM."""
    keys = ("qty", "price", "amount_without_vat", "vat_amount", "row_total")
    document = {
        "number": "128",
        "date": "2025-03-14",
        "seller": {"name": "ООО «Ромашка»", "inn": "7701234560"},
        "buyer": {"name": "АО «Вектор»", "inn": "5024156782"},
        "items": [
            {"name": f"Товар {n}", **dict(zip(keys, row, strict=True))} for n, row in enumerate(rows, 1)
        ],
        "total_with_vat": total,
    }
    return UpdSchema.model_validate_json(json.dumps(document))


def test_math_ok():
    document = upd(
        [(10, 250.00, 2500.00, 500.00, 3000.00), (5, 320.50, 1602.50, 320.50, 1923.00)], total=4923.00
    )
    assert validate_invoice_math(document) == []


def test_math_ok_for_invoice():
    assert validate_invoice_math(InvoiceSchema.model_validate(DOCUMENT)) == []


def test_math_tolerates_kopeck_rounding():
    # 3 × 33.33 = 99.99 при сумме 100.00, итог расходится на 3 копейки — всё в пределах допуска
    document = upd([(3, 33.33, 100.00, 20.00, 120.00)], total=120.03)
    assert validate_invoice_math(document) == []


def test_math_tolerance_boundary_is_exact():
    # Расхождение ровно 0.05 допустимо. На float это была бы ошибка: 8.05 - 8.0 = 0.05000000000000071
    assert validate_invoice_math(upd([(1, 8.05, 8.00, 1.60, 9.60)], total=9.60)) == []
    # 0.06 — уже нет
    assert len(validate_invoice_math(upd([(1, 8.06, 8.00, 1.60, 9.60)], total=9.60))) == 1


def test_math_detects_qty_price_mismatch():
    document = upd([(10, 250.00, 2600.00, 520.00, 3120.00)], total=3120.00)
    assert validate_invoice_math(document) == ["Строка 1: кол-во × цена (2500.00) != сумма без НДС (2600.00)"]


def test_math_detects_vat_mismatch():
    document = upd([(10, 250.00, 2500.00, 500.00, 3100.00)], total=3100.00)
    assert validate_invoice_math(document) == [
        "Строка 1: сумма без НДС + НДС (3000.00) != сумма по строке (3100.00)"
    ]


def test_math_detects_total_mismatch():
    document = upd([(4, 200.00, 800.00, 200.00, 1000.00)], total=1200.00)
    assert validate_invoice_math(document) == ["Сумма по строкам (1000.00) != Итого по документу (1200.00)"]


def test_math_reports_all_mismatches_at_once():
    rows = [(1, 100.00, 100.00, 20.00, 120.00), (2, 50.00, 90.00, 18.00, 200.00)]
    errors = validate_invoice_math(upd(rows, total=500.00))

    assert len(errors) == 3
    assert errors[0].startswith("Строка 2: кол-во × цена")
    assert errors[1].startswith("Строка 2: сумма без НДС + НДС")
    assert errors[2].startswith("Сумма по строкам (320.00)")
