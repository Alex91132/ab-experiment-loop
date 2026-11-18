"""Статистическое ядро: доли, интервалы, bootstrap, Холм, последовательный мониторинг.

Всё реализовано явно, без внешних библиотек кроме numpy/scipy: числа должны
воспроизводиться и проверяться тестами, а не быть результатом «чёрного ящика».

Соглашение: p1 — контроль, p2 — тест. Эффект считаем как p2 − p1
(отрицательный — тест хуже контроля).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

Z = {0.05: 1.959963984540054, 0.10: 1.6448536269514722, 0.01: 2.5758293035489004}


# ----------------------------------------------------------------------------- доли

def wilson_interval(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Интервал Уилсона для доли — честнее нормального при малых долях."""
    if n == 0:
        return (float("nan"), float("nan"))
    z = Z.get(alpha) or stats.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def two_proportion_z(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """z-статистика и p-value (двусторонний) для разницы долей, объединённая дисперсия."""
    p1, p2 = k1 / n1, k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return float("nan"), float("nan")
    z = (p2 - p1) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p


def diff_interval(k1: int, n1: int, k2: int, n2: int, alpha: float = 0.05) -> dict:
    """Разница долей: абсолютная (п.п.), интервал Уолда и интервал относительного прироста.

    Относительный прирост считаем через логарифм отношения — так интервал
    не вылезает за пределы и остаётся симметричным по шкале «во сколько раз».
    """
    z = Z.get(alpha) or stats.norm.ppf(1 - alpha / 2)
    p1, p2 = k1 / n1, k2 / n2
    diff = p2 - p1
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    lo, hi = diff - z * se, diff + z * se

    # относительный прирост: exp(ln(p2/p1)) ± z * se_log
    if p1 > 0 and p2 > 0:
        se_log = math.sqrt((1 - p1) / (p1 * n1) + (1 - p2) / (p2 * n2))
        rel = p2 / p1 - 1
        rel_lo = math.exp(math.log(p2 / p1) - z * se_log) - 1
        rel_hi = math.exp(math.log(p2 / p1) + z * se_log) - 1
    else:
        rel = rel_lo = rel_hi = float("nan")

    return {
        "p1": p1, "p2": p2,
        "diff_pp": diff * 100,
        "diff_ci_pp": (lo * 100, hi * 100),
        "rel_lift_pct": rel * 100,
        "rel_ci_pct": (rel_lo * 100, rel_hi * 100),
        "se_pp": se * 100,
        "significant": bool(lo > 0 or hi < 0),
    }


def bootstrap_diff(k1: int, n1: int, k2: int, n2: int, n_boot: int = 10_000,
                   seed: int = 42, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile-bootstrap разницы долей — проверка нормального интервала."""
    rng = np.random.default_rng(seed)
    p1 = rng.binomial(n1, k1 / n1, n_boot) / n1
    p2 = rng.binomial(n2, k2 / n2, n_boot) / n2
    d = (p2 - p1) * 100
    return (float(np.percentile(d, 100 * alpha / 2)), float(np.percentile(d, 100 * (1 - alpha / 2))))


# ----------------------------------------------------------------------------- непрерывные

def bootstrap_stat(values, stat: str = "median", n_boot: int = 10_000, seed: int = 42,
                   alpha: float = 0.05) -> dict:
    """Bootstrap среднего/медианы: тяжёлый хвост делает t-тест неинформативным."""
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), (n_boot, len(arr)))
    samples = arr[idx]
    point = np.median(samples, axis=1) if stat == "median" else samples.mean(axis=1)
    lo, hi = np.percentile(point, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    typical = float(np.median(arr)) if stat == "median" else float(arr.mean())
    return {"stat": stat, "value": typical, "ci": (float(lo), float(hi))}


def bootstrap_stat_diff(a, b, stat: str = "median", n_boot: int = 10_000,
                        seed: int = 42, alpha: float = 0.05) -> dict:
    """Bootstrap разницы b − a (тест минус контроль) для среднего или медианы."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    xs = x[rng.integers(0, len(x), (n_boot, len(x)))]
    ys = y[rng.integers(0, len(y), (n_boot, len(y)))]
    f = (lambda m: np.median(m, axis=1)) if stat == "median" else (lambda m: m.mean(axis=1))
    d = f(ys) - f(xs)
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    point = (float(np.median(y)) if stat == "median" else float(y.mean())) - \
            (float(np.median(x)) if stat == "median" else float(x.mean()))
    return {"stat": stat, "diff": point, "ci": (float(lo), float(hi)),
            "significant": bool(lo > 0 or hi < 0)}


# ----------------------------------------------------------------------------- множественность

def holm(pvalues: dict[str, float]) -> dict[str, dict]:
    """Поправка Холма: возвращает скорректированные p и факт отклонения."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted = {}
    running_max = 0.0
    for i, (name, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running_max = max(running_max, adj)      # монотонность
        adjusted[name] = {"p": p, "p_holm": running_max, "reject": bool(running_max < 0.05),
                          "rank": i + 1}
    return adjusted


# ----------------------------------------------------------------------------- последовательный мониторинг

def obf_boundaries(times: list[float], alpha: float = 0.05, tol: float = 1e-5) -> list[float]:
    """Точные границы O'Brien–Fleming: b_i = c / sqrt(t_i), константа из совместной вероятности.

    Подбираем c так, чтобы при верной H0 вероятность хотя бы раз пересечь границу
    равнялась alpha. Совместное распределение Z — многомерное нормальное с
    корреляцией sqrt(t_i / t_j), вероятность считаем через его CDF.
    """
    ts = np.asarray(times, dtype=float)
    if not np.all(np.diff(ts) > 0) or ts[-1] > 1.0 or ts[0] <= 0:
        raise ValueError("доли информации должны возрастать, быть в (0; 1] и заканчиваться единицей")

    corr = np.sqrt(np.minimum(ts[:, None], ts[None, :]) / np.maximum(ts[:, None], ts[None, :]))
    mean = np.zeros(len(ts))

    def not_crossed(c: float) -> float:
        b = c / np.sqrt(ts)
        return float(stats.multivariate_normal.cdf(b, mean=mean, cov=corr, lower_limit=-b))

    lo, hi = 0.5, 8.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if not_crossed(mid) > 1 - alpha:      # пересекаем реже, чем нужно → границу можно сузить
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    c = (lo + hi) / 2
    return [float(c / math.sqrt(t)) for t in ts]


def sequential_decision(z_values: list[float], times: list[float], alpha: float = 0.05) -> dict:
    """Первое пересечение границы OBF; если не пересекли — «данных мало»."""
    bounds = obf_boundaries(times, alpha)
    for i, (z, b) in enumerate(zip(z_values, bounds)):
        if abs(z) >= b:
            return {"stopped_at": i + 1, "z": float(z), "boundary": b, "decision": "эффект есть"}
    return {"stopped_at": None, "z": float(z_values[-1]), "boundary": bounds[-1],
            "decision": "граница не пересечена"}
