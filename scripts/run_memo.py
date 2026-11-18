"""Сборка меморандума из results/: решение, числа, ограничения.

    python scripts/run_memo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiment.decide import decide  # noqa: E402
from experiment.memo import MDE, render  # noqa: E402


def main() -> None:
    effects = json.loads((ROOT / "results" / "effects.json").read_text(encoding="utf-8"))
    eda = json.loads((ROOT / "results" / "eda.json").read_text(encoding="utf-8"))

    monitor_path = ROOT / "results" / "monitor.json"
    looks_path = ROOT / "results" / "sequential_looks.csv"
    monitor = json.loads(monitor_path.read_text(encoding="utf-8")) if monitor_path.exists() else None
    looks = None
    if looks_path.exists():
        import csv

        with looks_path.open(encoding="utf-8") as fh:
            looks = list(csv.DictReader(fh))

    verdict = decide(effects["primary"], effects["guardrails"], MDE)
    memo = render(effects, eda, verdict, monitor=monitor, looks=looks)

    out = ROOT / "results" / "memo.md"
    out.write_text(memo, encoding="utf-8")
    print(memo)
    print(f"написано: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
