"""Тесты алертов и сегментов."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiment.alerts import check_alerts
from experiment.segments import add_segments, segment_effects, summary


def validation(p: float, shift: float) -> dict:
    return {"srm_p": p, "srm_shift_pp": shift}


def effects(**kwargs) -> dict:
    base = {"retention_1": {"diff_pp": -0.5, "ci_pp": (-1.2, 0.1), "significant": False},
            "zero_rounds": {"diff_pp": 0.19, "ci_pp": (-0.08, 0.46), "significant": False},
            "heavy_user": {"diff_pp": 0.02, "ci_pp": (-0.02, 0.07), "significant": False}}
    base.update(kwargs)
    return base


def test_no_alerts_on_healthy_data():
    assert check_alerts(validation(0.0086, 0.44), effects()) == []


def test_srm_alert_requires_both_significance_and_shift():
    assert check_alerts(validation(0.0001, 0.44), effects()) == []      # p мал, сдвиг мал
    assert check_alerts(validation(0.5, 3.0), effects()) == []          # сдвиг велик, p велик
    alerts = check_alerts(validation(0.0001, 3.0), effects())
    assert len(alerts) == 1 and "SRM" in alerts[0]


def test_guardrail_increase_is_alert_when_significant():
    broken = effects(zero_rounds={"diff_pp": 0.9, "ci_pp": (0.2, 1.6), "significant": True})
    alerts = check_alerts(validation(0.5, 0.1), broken)
    assert len(alerts) == 1 and "zero_rounds" in alerts[0]


def test_guardrail_drop_is_alert_for_core_audience():
    broken = effects(heavy_user={"diff_pp": -0.05, "ci_pp": (-0.08, -0.01), "significant": True})
    alerts = check_alerts(validation(0.5, 0.1), broken)
    assert len(alerts) == 1 and "heavy_user" in alerts[0]


def test_guardrail_improvement_is_not_alert():
    good = effects(zero_rounds={"diff_pp": -0.5, "ci_pp": (-0.9, -0.1), "significant": True})
    assert check_alerts(validation(0.5, 0.1), good) == []


def make_frame(seed: int = 1, n: int = 6000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    versions = rng.choice(["gate_30", "gate_40"], n)
    rounds = rng.integers(0, 120, n)
    base = 0.45 - 0.0005 * (versions == "gate_40")
    retention = rng.random(n) < base
    return pd.DataFrame({"userid": np.arange(n), "version": versions,
                         "sum_gamerounds": rounds, "retention_1": retention,
                         "retention_7": retention, "install_day": rng.integers(1, 31, n)})


def test_add_segments_covers_everyone():
    df = add_segments(make_frame())
    assert set(df["activity"]) == {"0 раундов", "1–10", "11–50", "51+"}
    assert df["install_decade"].nunique() == 3


def test_segment_effects_returns_rows_and_holm():
    table = segment_effects(add_segments(make_frame()), by="activity")
    assert len(table) == 4
    assert "p_holm" in table.columns
    assert table["n_control"].sum() + table["n_treatment"].sum() == 6000


def test_segment_summary_shape():
    table = segment_effects(add_segments(make_frame()), by="install_decade")
    info = summary(table)
    assert info["segments"] == 3
    assert isinstance(info["names"], list)
