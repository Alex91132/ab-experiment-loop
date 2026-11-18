"""Тесты калькулятора MDE: формула, обратимость, свойства."""

from __future__ import annotations

import pytest

from experiment.power import (
    mde_proportions, relative_to_absolute, required_sample, z_alpha, z_beta,
)


def test_z_values():
    assert z_alpha(0.05) == pytest.approx(1.959964, abs=1e-5)
    assert z_beta(0.8) == pytest.approx(0.841621, abs=1e-5)


def test_mde_hand_computed():
    p1, n = 0.448, 45000
    expected = (1.959964 + 0.841621) * (2 * p1 * (1 - p1) / n) ** 0.5
    assert mde_proportions(p1, n) == pytest.approx(expected, rel=1e-6)
    assert 0.008 < mde_proportions(p1, n) < 0.010      # ≈ 0,9 п.п., как в плане


def test_sample_round_trip():
    p1, mde = 0.19, 0.007
    n = required_sample(p1, mde)
    assert mde_proportions(p1, n) <= mde * 1.01         # найденная выборка покрывает MDE


def test_mde_shrinks_as_sqrt_n():
    small = mde_proportions(0.3, 10_000)
    large = mde_proportions(0.3, 40_000)
    assert small / large == pytest.approx(2.0, rel=1e-6)


def test_relative_to_absolute():
    assert relative_to_absolute(0.5, 0.02) == pytest.approx(0.01)


def test_power_increase_shrinks_mde():
    assert mde_proportions(0.4, 20_000, power=0.9) > mde_proportions(0.4, 20_000, power=0.8)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        mde_proportions(0.0, 1000)
    with pytest.raises(ValueError):
        required_sample(0.4, 0.0)
