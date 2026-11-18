"""Тесты правила принятия решения: каждое правило срабатывает на своих данных."""

from __future__ import annotations

import pytest

from experiment.decide import decide

MDE = {"retention_1": 0.9, "retention_7": 0.7}


def row(diff: float, lo: float, hi: float) -> dict:
    return {"diff_pp": diff, "diff_ci_pp": (lo, hi)}


def test_rule_5_guardrail_worse_blocks_transfer():
    primary = {"retention_1": row(0.6, 0.1, 1.1), "retention_7": row(0.5, 0.1, 0.9)}
    guardrails = {"zero_rounds_share": row(0.4, 0.05, 0.75)}
    v = decide(primary, guardrails, MDE)
    assert v.rule == 5 and v.decision == "оставить"
    assert "guardrail" in v.reasons[0]


def test_rule_3_significant_worsening_keeps_current():
    primary = {"retention_1": row(-0.6, -1.2, 0.06), "retention_7": row(-0.82, -1.33, -0.31)}
    v = decide(primary, {}, MDE)
    assert v.rule == 3 and v.decision == "оставить"
    assert "retention_7" in v.reasons[0]


def test_rule_3_notes_effect_above_mde():
    primary = {"retention_1": row(-1.2, -1.8, -0.6), "retention_7": row(-0.2, -0.6, 0.2)}
    v = decide(primary, {}, MDE)
    assert "выше MDE" in v.reasons[0]


def test_rule_2_both_better_and_guardrails_clean():
    primary = {"retention_1": row(0.8, 0.2, 1.4), "retention_7": row(0.9, 0.4, 1.4)}
    guardrails = {"zero_rounds_share": row(-0.1, -0.3, 0.1)}
    v = decide(primary, guardrails, MDE)
    assert v.rule == 2 and v.decision == "перенести"


def test_rule_1_all_intervals_cover_zero():
    primary = {"retention_1": row(-0.59, -1.24, 0.06), "retention_7": row(-0.1, -0.5, 0.3)}
    v = decide(primary, {}, MDE)
    assert v.rule == 1 and v.decision == "не менять"


def test_rule_4_improvement_inside_noise():
    primary = {"retention_1": row(0.3, -0.2, 0.8), "retention_7": row(0.2, -0.15, 0.55)}
    v = decide(primary, {}, MDE)
    assert v.rule == 4 and v.decision == "не менять"
    assert "улучшение" in v.reasons[-1].lower()


def test_real_results_trigger_rule_3():
    """На фактических числах эксперимента срабатывает правило 3, а не 1."""
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    effects = json.loads((root / "results" / "effects.json").read_text(encoding="utf-8"))
    v = decide(effects["primary"], effects["guardrails"], MDE)
    assert v.rule == 3
    assert v.decision == "оставить"
