"""Тесты CUPED и дельта-метода: проверяем свойства, а не только отсутствие ошибок."""

from __future__ import annotations

import numpy as np
import pytest

from experiment.advanced import (
    cuped_adjust, cuped_theta, cuped_variance_reduction, delta_ratio, delta_ratio_diff,
)


def make_panel(n: int = 4000, rho: float = 0.7, seed: int = 7):
    """Предпериод и целевая метрика, связанные корреляцией rho."""
    rng = np.random.default_rng(seed)
    pre = rng.normal(50, 10, n)
    noise = rng.normal(0, 10 * np.sqrt(1 - rho**2), n)
    post = 50 + rho * (pre - 50) + noise
    return pre, post


def test_cuped_theta_matches_formula():
    pre, post = make_panel()
    theta = cuped_theta(pre, post)
    expected = np.cov(pre, post, ddof=1)[0, 1] / np.var(pre, ddof=1)
    assert theta == pytest.approx(expected, rel=1e-9)


def test_cuped_keeps_mean_unbiased():
    pre, post = make_panel()
    adjusted = cuped_adjust(pre, post)
    assert adjusted.mean() == pytest.approx(post.mean(), abs=1e-9)


def test_cuped_reduces_variance_when_correlated():
    pre, post = make_panel(rho=0.7)
    reduction = cuped_variance_reduction(pre, post)
    assert 0.3 < reduction < 0.6          # теоретически ожидаем ≈ rho² = 0,49


def test_cuped_does_nothing_without_correlation():
    rng = np.random.default_rng(3)
    pre = rng.normal(0, 1, 3000)
    post = rng.normal(0, 1, 3000)
    assert abs(cuped_variance_reduction(pre, post)) < 0.02


def test_cuped_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        cuped_theta(np.arange(10), np.arange(9))


def test_delta_ratio_is_exact_for_constant_denominator():
    """Если знаменатель постоянный, отношение — это среднее, дельта-метод совпадает с ним."""
    num = np.array([10.0, 20.0, 30.0, 40.0])
    den = np.ones(4)
    r = delta_ratio(num, den)
    assert r["ratio"] == pytest.approx(num.mean())
    assert r["se"] == pytest.approx(num.std(ddof=1) / np.sqrt(4))


def test_delta_ratio_close_to_bootstrap():
    rng = np.random.default_rng(11)
    num = rng.gamma(2.0, 5.0, 2000)
    den = rng.gamma(3.0, 2.0, 2000)
    analytic = delta_ratio(num, den)

    boots = []
    for _ in range(600):
        idx = rng.integers(0, len(num), len(num))
        boots.append(num[idx].sum() / den[idx].sum())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    analytic_ci = (analytic["ratio"] - 1.96 * analytic["se"], analytic["ratio"] + 1.96 * analytic["se"])
    assert abs(analytic_ci[0] - lo) / (hi - lo) < 0.15
    assert abs(analytic_ci[1] - hi) / (hi - lo) < 0.15


def test_delta_ratio_diff_detects_shift():
    rng = np.random.default_rng(5)
    den_a = rng.uniform(1, 5, 3000)
    den_b = rng.uniform(1, 5, 3000)
    num_a = den_a * rng.normal(10, 1, 3000)
    num_b = den_b * rng.normal(11, 1, 3000)
    result = delta_ratio_diff(num_a, den_a, num_b, den_b)
    assert result["significant"] is True
    assert result["ci"][0] > 0
