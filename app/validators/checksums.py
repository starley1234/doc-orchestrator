"""Проверка реквизитов: контрольные цифры ИНН, формат КПП и БИК (защита от галлюцинаций VLM)."""

import re

# Весовые коэффициенты ФНС по длине ИНН. Набор из N весов проверяет цифру с индексом N:
# у 10-значного ИНН контрольная цифра одна (последняя), у 12-значного — две.
_INN_WEIGHTS = {
    10: [(2, 4, 10, 3, 5, 9, 4, 6, 8)],
    12: [(7, 2, 4, 10, 3, 5, 9, 4, 6, 8), (3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8)],
}

_KPP_RE = re.compile(r"\d{4}[\dA-Z]{2}\d{3}", re.ASCII)
_BIK_RE = re.compile(r"04\d{7}", re.ASCII)


def _control_digit(digits: list[int], weights: tuple[int, ...]) -> int:
    """Контрольная цифра ИНН: взвешенная сумма первых цифр по модулю 11, затем по модулю 10."""
    return sum(weight * digits[i] for i, weight in enumerate(weights)) % 11 % 10


def validate_inn(inn: str) -> bool:
    """ИНН: 10 цифр (юрлицо) или 12 цифр (ИП, физлицо) с верными контрольными цифрами."""
    if not (inn.isascii() and inn.isdigit()) or inn.startswith("00"):  # кода региона 00 не существует
        return False
    digits = [int(char) for char in inn]
    weight_sets = _INN_WEIGHTS.get(len(digits))
    if weight_sets is None:
        return False
    return all(digits[len(weights)] == _control_digit(digits, weights) for weights in weight_sets)


def validate_kpp(kpp: str) -> bool:
    """КПП: 9 символов — 4 цифры, 2 цифры или заглавные латинские буквы, 3 цифры."""
    return _KPP_RE.fullmatch(kpp) is not None


def validate_bik(bik: str) -> bool:
    """БИК: 9 цифр, начинается с 04 (код России)."""
    return _BIK_RE.fullmatch(bik) is not None
