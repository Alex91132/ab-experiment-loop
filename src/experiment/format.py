"""Форматирование чисел для русскоязычного текста: запятая в дробях, пробел в разрядах."""

from __future__ import annotations


def n(value: float, nd: int = 2) -> str:
    """Число с запятой: 0.82 → «0,82»."""
    return f"{value:.{nd}f}".replace(".", ",")


def signed(value: float, nd: int = 2) -> str:
    """Число со знаком: +0.19 → «+0,19»."""
    return f"{value:+.{nd}f}".replace(".", ",")


def ci(pair, nd: int = 2) -> str:
    """Интервал: [-1.239, 0.058] → «[-1,24; +0,06]»."""
    lo, hi = pair
    return f"[{signed(lo, nd)}; {signed(hi, nd)}]"


def thousands(value: int) -> str:
    """Разряды через пробел: 49854 → «49 854»."""
    return f"{value:,}".replace(",", " ")
