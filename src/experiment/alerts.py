"""Алерты контура: SRM и предохранители. Вынесены из DAG, чтобы тестировать.

DAG-задача `guard_check` вызывает `check_alerts` и падает, если список непуст.
"""

from __future__ import annotations

DEFAULT_DIRECTION = {"zero_rounds": +1, "heavy_user": -1}


def check_alerts(validation: dict, effects: dict,
                 direction: dict[str, int] | None = None,
                 srm_p_threshold: float = 0.001,
                 srm_shift_threshold: float = 1.0) -> list[str]:
    """Возвращает список проблем. Пустой список — всё в порядке.

    SRM: тревога только если и p очень мал, и отклонение от 50/50 больше 1 п.п.
    Иначе на больших n тревога горит постоянно — это уже проходили.
    """
    problems: list[str] = []
    if validation["srm_p"] < srm_p_threshold and validation["srm_shift_pp"] > srm_shift_threshold:
        problems.append(
            f"SRM: p={validation['srm_p']}, отклонение {validation['srm_shift_pp']} п.п. — "
            "проверить рандомизацию до публикации результатов"
        )

    dirs = direction or DEFAULT_DIRECTION
    for name, row in effects.items():
        if name not in dirs or not row.get("significant"):
            continue
        worse = (row["diff_pp"] > 0) if dirs[name] > 0 else (row["diff_pp"] < 0)
        if worse:
            problems.append(
                f"guardrail «{name}» значимо хуже: {row['diff_pp']:+.2f} п.п. "
                f"[{row['ci_pp'][0]:+.2f}; {row['ci_pp'][1]:+.2f}]"
            )
    return problems
