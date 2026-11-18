"""Продвинутые оценки: CUPED и дельта-метод для ratio-метрик.

Оба метода покрыты тестами на синтетике. В этом эксперименте CUPED применить
не к чему (нет предпериода), а дельта-метод применён к реальной ratio-метрике —
доле раундов, сыгранных тяжёлыми игроками.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def cuped_theta(pre, post) -> float:
    """Коэффициент CUPED: cov(до, после) / var(до)."""
    x = np.asarray(pre, dtype=float)
    y = np.asarray(post, dtype=float)
    if len(x) != len(y):
        raise ValueError("предпериод и целевая метрика должны быть одинаковой длины")
    var_x = float(np.var(x, ddof=1))
    if var_x == 0:
        return 0.0
    return float(np.cov(x, y, ddof=1)[0, 1] / var_x)


def cuped_adjust(pre, post, theta: float | None = None) -> np.ndarray:
    """Скорректированная метрика: y − θ(x − x̄).

    При θ = cov/var дисперсия оценки минимальна, а среднее остаётся несмещённым:
    поправка центрирована около нуля.
    """
    x = np.asarray(pre, dtype=float)
    y = np.asarray(post, dtype=float)
    if theta is None:
        theta = cuped_theta(x, y)
    return y - theta * (x - x.mean())


def cuped_variance_reduction(pre, post) -> float:
    """Какая доля дисперсии уйдёт после CUPED: 1 − var(adjusted)/var(raw)."""
    x = np.asarray(pre, dtype=float)
    y = np.asarray(post, dtype=float)
    adjusted = cuped_adjust(x, y)
    var_raw = float(np.var(y, ddof=1))
    if var_raw == 0:
        return 0.0
    return float(1 - np.var(adjusted, ddof=1) / var_raw)


def delta_ratio(numerator, denominator) -> dict:
    """Оценка отношения сумм через дельта-метод (линеаризация).

    Если r = Σnum / Σden, то var(r) ≈ n·var(num − r·den) / (Σden)^2.
    Метод нужен, когда метрика — отношение, и обычный тест долей неприменим.
    """
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    if len(num) != len(den):
        raise ValueError("числитель и знаменатель должны быть одной длины")
    n = len(num)
    total_den = den.sum()
    if total_den == 0:
        raise ValueError("знаменатель нулевой")
    ratio = num.sum() / total_den
    residual = num - ratio * den
    var = n * float(np.var(residual, ddof=1)) / total_den**2
    return {"ratio": float(ratio), "se": float(np.sqrt(var)), "n": n}


def delta_ratio_diff(num_a, den_a, num_b, den_b, alpha: float = 0.05) -> dict:
    """Разница двух отношений b − a с интервалом по дельта-методу."""
    a = delta_ratio(num_a, den_a)
    b = delta_ratio(num_b, den_b)
    z = stats.norm.ppf(1 - alpha / 2)
    diff = b["ratio"] - a["ratio"]
    se = float(np.sqrt(a["se"] ** 2 + b["se"] ** 2))
    return {
        "control": a["ratio"], "treatment": b["ratio"],
        "diff": diff, "se": se,
        "ci": (diff - z * se, diff + z * se),
        "significant": bool(diff - z * se > 0 or diff + z * se < 0),
    }
