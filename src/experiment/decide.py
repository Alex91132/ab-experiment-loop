"""Правило принятия решения из PLAN.md, реализованное кодом.

Смысл: решение не должно зависеть от того, кто читает таблицу. Правила те же,
что зафиксированы до анализа; порядок проверки документирован и проверяется
тестами.

Порядок (первое сработавшее правило выигрывает):
  5. guardrail значимо хуже — вето, независимо от основных метрик;
  3. основная метрика значимо хуже — оставить как было;
  2. обе основные метрики улучшились и guardrails не хуже — переносить;
  1. интервал основной метрики накрывает ноль — эффект не доказан, не менять;
  4. улучшение есть, но меньше MDE — не менять, отметить нехватку чувствительности.

Направление guardrail задаётся словарём: +1 — хуже, когда метрика растёт
(доля игроков без раундов), −1 — хуже, когда падает (доля активного ядра).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from experiment.format import ci, n, signed

DEFAULT_GUARDRAIL_DIRECTION = {"zero_rounds_share": +1, "heavy_users_share": -1}


@dataclass
class Verdict:
    decision: str                 # «оставить», «перенести», «не менять»
    rule: int
    headline: str
    reasons: list[str] = field(default_factory=list)


def _worse_direction(row: dict, direction: int) -> bool:
    """Значимое ухудшение с учётом того, в какую сторону «плохо» для этой метрики."""
    lo, hi = row["diff_ci_pp"]
    if direction > 0:            # рост — плохо
        return row["diff_pp"] > 0 and lo > 0
    return row["diff_pp"] < 0 and hi < 0


def _significantly_better(row: dict) -> bool:
    lo, hi = row["diff_ci_pp"]
    return row["diff_pp"] > 0 and lo > 0


def _crosses_zero(row: dict) -> bool:
    lo, hi = row["diff_ci_pp"]
    return lo <= 0 <= hi


def decide(primary: dict[str, dict], guardrails: dict[str, dict], mde: dict[str, float],
           guardrail_direction: dict[str, int] | None = None) -> Verdict:
    """Применяет правила плана к результатам и возвращает вердикт с обоснованием."""
    directions = guardrail_direction or DEFAULT_GUARDRAIL_DIRECTION
    reasons: list[str] = []

    worse_guards = [name for name, row in guardrails.items()
                    if _worse_direction(row, directions.get(name, +1))]
    if worse_guards:
        for name in worse_guards:
            row = guardrails[name]
            reasons.append(f"guardrail «{name}» значимо хуже: {signed(row['diff_pp'])} п.п. "
                           f"{ci(row['diff_ci_pp'])}")
        return Verdict("оставить", 5, "Предохранитель упал — переносить нельзя", reasons)

    worse = [name for name, row in primary.items() if _worse_direction(row, -1)]
    if worse:
        for name in worse:
            row = primary[name]
            tail = ""
            limit = mde.get(name)
            if limit is not None:
                tail = (f"; ухудшение {n(abs(row['diff_pp']))} п.п. "
                        f"{'выше' if abs(row['diff_pp']) >= limit else 'ниже'} MDE {n(limit, 1)} п.п.")
            reasons.append(f"«{name}» значимо хуже: {signed(row['diff_pp'])} п.п. "
                           f"{ci(row['diff_ci_pp'])}{tail}")
        return Verdict("оставить", 3, "Основная метрика значимо ухудшилась — оставить как было", reasons)

    better = [name for name, row in primary.items() if _significantly_better(row)]
    if len(better) == len(primary):
        reasons.append("обе основные метрики значимо улучшились, guardrails не хуже")
        return Verdict("перенести", 2, "Эффект положительный и устойчивый — можно переносить", reasons)

    neutral = [name for name, row in primary.items() if _crosses_zero(row)]
    if neutral:
        for name in neutral:
            row = primary[name]
            reasons.append(f"«{name}»: {signed(row['diff_pp'])} п.п. "
                           f"{ci(row['diff_ci_pp'])} — интервал накрывает ноль")
        if any(row["diff_pp"] > 0 for row in primary.values()):
            reasons.append("улучшение не доказано — решение в пользу текущего варианта")
            return Verdict("не менять", 4, "Улучшение не доказано: эффект меньше чувствительности", reasons)
        return Verdict("не менять", 1, "Эффект не доказан — менять не на чем", reasons)

    reasons.append("ни ухудшения, ни улучшения не доказано")
    return Verdict("не менять", 1, "Эффект не доказан — менять не на чем", reasons)
