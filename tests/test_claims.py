"""Числа в меморандуме и results/ сверяются с данными.

Если пайплайн или форматирование изменят цифры, тест упадёт — меморандум
не сможет разъехаться с расчётами.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "cookie_cats.csv"
RESULTS = ROOT / "results"

pytestmark = pytest.mark.skipif(not DATA.exists(), reason="нет data/cookie_cats.csv (см. scripts/fetch_data.py)")


@pytest.fixture(scope="module")
def df():
    import pandas as pd

    frame = pd.read_csv(DATA)
    frame["retention_1"] = frame["retention_1"].astype(bool)
    frame["retention_7"] = frame["retention_7"].astype(bool)
    return frame


@pytest.fixture(scope="module")
def effects():
    return json.loads((RESULTS / "effects.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def memo():
    return (RESULTS / "memo.md").read_text(encoding="utf-8")


def test_group_sizes_match(effects, df):
    counts = df["version"].value_counts()
    assert effects["primary"]["retention_1"]["control"]["n"] == counts["gate_30"]
    assert effects["primary"]["retention_1"]["treatment"]["n"] == counts["gate_40"]


def test_effects_match_recomputation(effects, df):
    for name in ("retention_1", "retention_7"):
        c = df[df.version == "gate_30"][name]
        t = df[df.version == "gate_40"][name]
        diff_pp = (t.mean() - c.mean()) * 100
        assert effects["primary"][name]["diff_pp"] == pytest.approx(diff_pp, abs=1e-3)
        assert effects["primary"][name]["control"]["rate_pct"] == pytest.approx(c.mean() * 100, abs=0.01)
        assert effects["primary"][name]["treatment"]["rate_pct"] == pytest.approx(t.mean() * 100, abs=0.01)


def test_retention_7_is_significant_and_negative(effects):
    r7 = effects["primary"]["retention_7"]
    assert r7["significant"] is True
    assert r7["diff_pp"] < 0
    assert r7["diff_ci_pp"][1] < 0


def test_retention_1_crosses_zero(effects):
    r1 = effects["primary"]["retention_1"]
    assert r1["significant"] is False
    assert r1["diff_ci_pp"][0] < 0 < r1["diff_ci_pp"][1]


def test_guardrails_not_significant(effects):
    for name in ("zero_rounds_share", "heavy_users_share"):
        assert effects["guardrails"][name]["significant"] is False


def test_mean_difference_is_an_outlier(effects):
    assert effects["rounds"]["all"]["mean_diff"] < -1.0
    assert abs(effects["rounds"]["without_max"]["mean_diff"]) < 0.1


def test_memo_has_decision_and_numbers(memo):
    assert "правило 3" in memo
    assert "оставить как было" in memo
    assert "44,82 %" in memo and "44,23 %" in memo
    assert "19,02 %" in memo and "18,20 %" in memo
    assert "−0,82" not in memo  # используем обычный дефис в форматтере
    assert "-0,82" in memo
    assert "49 854" in memo                     # разряды через пробел
    assert "0,0032" in memo                     # p после поправки Холма


def test_memo_mentions_srm_and_limits(memo):
    assert "SRM" in memo
    assert "Нет выручки" in memo
    assert "CUPED" in memo


def test_eda_srm_numbers(eda_json):
    srm = eda_json["srm"]
    assert srm["counts"] == {"gate_40": 45_489, "gate_30": 44_700}
    assert srm["max_shift_pp"] == pytest.approx(0.44, abs=0.01)
    assert srm["practically_balanced"] is True


@pytest.fixture(scope="module")
def eda_json():
    return json.loads((RESULTS / "eda.json").read_text(encoding="utf-8"))
