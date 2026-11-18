"""Расчёт эффектов: основные метрики и guardrails, с интервалами и поправкой.

    python scripts/run_effects.py

Пишет results/effects.json — из него потом собирается меморандум и дашборд.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiment.stats import (  # noqa: E402
    bootstrap_diff, bootstrap_stat_diff, diff_interval, holm, two_proportion_z,
    wilson_interval,
)

DATA = ROOT / "data" / "cookie_cats.csv"
OUT = ROOT / "results" / "effects.json"
CONTROL, TREATMENT = "gate_30", "gate_40"


def ratio_metric(df: pd.DataFrame, column: str) -> dict:
    """Доля в каждой группе: разница, интервалы, z-тест, bootstrap-проверка."""
    c = df[df.version == CONTROL][column]
    t = df[df.version == TREATMENT][column]
    k1, n1 = int(c.sum()), int(len(c))
    k2, n2 = int(t.sum()), int(len(t))

    r = diff_interval(k1, n1, k2, n2)
    z, p = two_proportion_z(k1, n1, k2, n2)
    boot = bootstrap_diff(k1, n1, k2, n2)
    return {
        "control": {"k": k1, "n": n1, "rate_pct": round(r["p1"] * 100, 2),
                    "ci_pct": [round(v * 100, 2) for v in wilson_interval(k1, n1)]},
        "treatment": {"k": k2, "n": n2, "rate_pct": round(r["p2"] * 100, 2),
                      "ci_pct": [round(v * 100, 2) for v in wilson_interval(k2, n2)]},
        "diff_pp": round(r["diff_pp"], 3),
        "diff_ci_pp": [round(v, 3) for v in r["diff_ci_pp"]],
        "rel_lift_pct": round(r["rel_lift_pct"], 2),
        "rel_ci_pct": [round(v, 2) for v in r["rel_ci_pct"]],
        "z": round(z, 3),
        "p_value": round(p, 4),
        "bootstrap_ci_pp": [round(v, 3) for v in boot],
        "significant": bool(r["significant"]),
    }


def main() -> None:
    df = pd.read_csv(DATA)
    df["retention_1"] = df["retention_1"].astype(bool)
    df["retention_7"] = df["retention_7"].astype(bool)
    df["zero_rounds"] = df["sum_gamerounds"] == 0
    df["heavy_user"] = df["sum_gamerounds"] >= 1000

    effects = {
        "primary": {
            "retention_1": ratio_metric(df, "retention_1"),
            "retention_7": ratio_metric(df, "retention_7"),
        },
        "guardrails": {
            "zero_rounds_share": ratio_metric(df, "zero_rounds"),
            "heavy_users_share": ratio_metric(df, "heavy_user"),
        },
    }

    # поправка Холма только на две основные метрики (guardrails — отдельные правила)
    adj = holm({k: v["p_value"] for k, v in effects["primary"].items()})
    for name, row in adj.items():
        effects["primary"][name]["p_holm"] = round(row["p_holm"], 4)
        effects["primary"][name]["reject_holm"] = row["reject"]

    # раунды: медиана и среднее, с выбросом и без
    rounds = {}
    for label, frame in (("all", df), ("without_max", df[df.sum_gamerounds < df.sum_gamerounds.max()])):
        c = frame[frame.version == CONTROL]["sum_gamerounds"].values
        t = frame[frame.version == TREATMENT]["sum_gamerounds"].values
        med = bootstrap_stat_diff(c, t, stat="median")
        mean = bootstrap_stat_diff(c, t, stat="mean")
        rounds[label] = {
            "control_median": float(np.median(c)), "treatment_median": float(np.median(t)),
            "median_diff": round(med["diff"], 3), "median_ci": [round(v, 3) for v in med["ci"]],
            "control_mean": round(float(c.mean()), 2), "treatment_mean": round(float(t.mean()), 2),
            "mean_diff": round(mean["diff"], 2), "mean_ci": [round(v, 2) for v in mean["ci"]],
            "mean_significant": bool(mean["significant"]),
        }
    effects["rounds"] = rounds

    # ratio-метрика: какую долю всех раундов сыграли тяжёлые игроки (дельта-метод)
    from experiment.advanced import delta_ratio_diff

    num_c = np.where(df[df.version == CONTROL]["sum_gamerounds"] >= 1000,
                     df[df.version == CONTROL]["sum_gamerounds"], 0.0)
    den_c = df[df.version == CONTROL]["sum_gamerounds"].values.astype(float)
    num_t = np.where(df[df.version == TREATMENT]["sum_gamerounds"] >= 1000,
                     df[df.version == TREATMENT]["sum_gamerounds"], 0.0)
    den_t = df[df.version == TREATMENT]["sum_gamerounds"].values.astype(float)
    ratio = delta_ratio_diff(num_c, den_c, num_t, den_t)
    effects["ratio_metrics"] = {
        "heavy_rounds_share": {
            "control_pct": round(ratio["control"] * 100, 3),
            "treatment_pct": round(ratio["treatment"] * 100, 3),
            "diff_pp": round(ratio["diff"] * 100, 3),
            "ci_pp": [round(ratio["ci"][0] * 100, 3), round(ratio["ci"][1] * 100, 3)],
            "significant": ratio["significant"],
        }
    }

    # сегменты по активности: где эффект сильнее, где его нет
    from experiment.segments import add_segments, segment_effects, summary as segment_summary

    seg_df = add_segments(df)
    segments = segment_effects(seg_df, by="activity")
    segments.to_csv(ROOT / "results" / "segments.csv", index=False)
    effects["segments"] = segment_summary(segments)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(effects, ensure_ascii=False, indent=1), encoding="utf-8")

    print("основные метрики")
    for name, r in effects["primary"].items():
        print(f"  {name}: {r['control']['rate_pct']} % → {r['treatment']['rate_pct']} % | "
              f"разница {r['diff_pp']:+.3f} п.п. [{r['diff_ci_pp'][0]:+.3f}; {r['diff_ci_pp'][1]:+.3f}] | "
              f"p = {r['p_value']}, p_holm = {r['p_holm']} | значимо: {r['significant']}")
    print("\nguardrails")
    for name, r in effects["guardrails"].items():
        print(f"  {name}: {r['control']['rate_pct']} % → {r['treatment']['rate_pct']} % | "
              f"разница {r['diff_pp']:+.3f} п.п. [{r['diff_ci_pp'][0]:+.3f}; {r['diff_ci_pp'][1]:+.3f}]")
    print("\nраунды")
    for label, r in rounds.items():
        print(f"  {label}: медиана {r['control_median']:.0f} → {r['treatment_median']:.0f} "
              f"({r['median_diff']:+.2f}, [{r['median_ci'][0]:+.2f}; {r['median_ci'][1]:+.2f}]) | "
              f"среднее {r['control_mean']} → {r['treatment_mean']} ({r['mean_diff']:+.2f})")
    print("\nсегменты по активности")
    print(segments.to_string(index=False))
    print(f"\nнаписано: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
