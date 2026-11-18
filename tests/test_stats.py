"""Тесты статистического ядра: сверка с известными значениями и свойствами."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats as sps

from experiment.stats import (
    bootstrap_diff, bootstrap_stat, bootstrap_stat_diff, diff_interval, holm,
    obf_boundaries, sequential_decision, two_proportion_z, wilson_interval,
)


def test_wilson_interval_known_value():
    lo, hi = wilson_interval(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-4)
    assert hi == pytest.approx(0.5962, abs=1e-4)


def test_wilson_interval_narrows_with_n():
    small = wilson_interval(50, 100)
    large = wilson_interval(5000, 10000)
    assert (large[1] - large[0]) < (small[1] - small[0]) / 5


def test_z_test_matches_manual_calculation():
    z, p = two_proportion_z(k1=100, n1=1000, k2=130, n2=1000)
    # объединённая доля 0.115, se = sqrt(0.115*0.885*2/1000)
    se = math.sqrt(0.115 * 0.885 * 2 / 1000)
    assert z == pytest.approx(0.03 / se, rel=1e-9)
    assert p == pytest.approx(2 * (1 - sps.norm.cdf(z)), rel=1e-9)


def test_diff_interval_reports_absolute_and_relative():
    r = diff_interval(k1=4482, n1=10000, k2=4423, n2=10000)
    assert r["diff_pp"] == pytest.approx(-0.59, abs=1e-9)
    assert r["diff_ci_pp"][0] < r["diff_pp"] < r["diff_ci_pp"][1]
    assert r["rel_lift_pct"] == pytest.approx(-1.32, abs=0.05)
    assert r["significant"] is False        # интервал накрывает ноль


def test_bootstrap_diff_close_to_wald_and_reproducible():
    a = bootstrap_diff(4482, 10000, 4423, 10000, n_boot=4000, seed=7)
    b = bootstrap_diff(4482, 10000, 4423, 10000, n_boot=4000, seed=7)
    assert a == b                            # воспроизводимость по seed
    wald = diff_interval(4482, 10000, 4423, 10000)["diff_ci_pp"]
    assert abs(a[0] - wald[0]) < 0.05 and abs(a[1] - wald[1]) < 0.05


def test_bootstrap_median_interval_contains_median_on_skewed_data():
    rng = np.random.default_rng(0)
    data = rng.lognormal(mean=2.0, sigma=1.0, size=2000)   # тяжёлый хвост
    r = bootstrap_stat(data, stat="median", n_boot=2000, seed=1)
    assert r["ci"][0] <= r["value"] <= r["ci"][1]
    assert r["ci"][1] - r["ci"][0] < np.median(data)        # интервал узкий


def test_bootstrap_median_robust_to_outlier():
    rng = np.random.default_rng(3)
    data = rng.gamma(2.0, 5.0, size=1000)
    clean = bootstrap_stat_diff(data, data * 1.0, stat="median", n_boot=1000, seed=2)
    assert abs(clean["diff"]) < 1e-9
    outlier = np.concatenate([data, [49_854.0]])
    mean_shift = bootstrap_stat(outlier, stat="mean")["value"] - bootstrap_stat(data, stat="mean")["value"]
    median_shift = bootstrap_stat(outlier, stat="median")["value"] - bootstrap_stat(data, stat="median")["value"]
    assert mean_shift > 20 * max(median_shift, 1e-9)        # среднее ломается, медиана нет


def test_holm_known_example():
    r = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert r["a"]["p_holm"] == pytest.approx(0.03)
    assert r["c"]["p_holm"] == pytest.approx(0.06)
    assert r["b"]["p_holm"] == pytest.approx(0.06)
    assert r["a"]["reject"] is True
    assert r["b"]["reject"] is False and r["c"]["reject"] is False


def test_obf_boundaries_shape():
    times = [0.25, 0.5, 0.75, 1.0]
    b = obf_boundaries(times)
    assert all(b[i] > b[i + 1] for i in range(3))     # границы падают с информацией
    assert b[0] > 3.5 and b[-1] < 2.3                 # ранний взгляд строже, финал близок к 1,96


def test_obf_boundaries_control_type_one_error():
    """Симуляция: при верной H0 граница пересекается примерно в alpha случаев."""
    times = [0.5, 1.0]
    bounds = obf_boundaries(times)
    rng = np.random.default_rng(11)
    n = 40_000
    # Z на двух срезах: корреляция sqrt(0.5)
    z1 = rng.standard_normal(n)
    z2 = math.sqrt(0.5) * z1 + math.sqrt(0.5) * rng.standard_normal(n)
    crossed = (np.abs(z1) >= bounds[0]) | (np.abs(z2) >= bounds[1])
    assert crossed.mean() == pytest.approx(0.05, abs=0.008)


def test_sequential_decision_stops_only_on_crossing():
    assert sequential_decision([0.5], [1.0])["decision"] == "граница не пересечена"
    hit = sequential_decision([3.2], [1.0])
    assert hit["decision"] == "эффект есть" and hit["stopped_at"] == 1
