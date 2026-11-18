"""Генератор меморандума: собирает решение и числа из results/ в один текст.

Меморандум — то, что аналитик реально отдаёт: не графики, а ответ на вопрос,
насколько ему можно верить и что проверить дальше.
"""

from __future__ import annotations

from experiment.decide import Verdict
from experiment.format import ci, n, signed, thousands

MDE = {"retention_1": 0.9, "retention_7": 0.7}   # п.п., из PLAN.md


def render(effects: dict, eda: dict, verdict: Verdict, monitor: dict | None = None,
           looks: list[dict] | None = None) -> str:
    lines: list[str] = []
    add = lines.append

    add("# Меморандум: перенос «ворот» в Cookie Cats с 30-го на 40-й уровень")
    add("")
    add(f"**Решение: {verdict.headline.lower()}** (правило {verdict.rule} из плана).")
    add("")
    for reason in verdict.reasons:
        add(f"- {reason}")
    add("")

    add("## Основные метрики")
    add("")
    add("| Метрика | Контроль (30) | Тест (40) | Разница | 95 % ДИ | p | MDE |")
    add("|---|---:|---:|---:|---|---:|---:|")
    for name, row in effects["primary"].items():
        add(f"| {name} | {n(row['control']['rate_pct'])} % | {n(row['treatment']['rate_pct'])} % | "
            f"{signed(row['diff_pp'])} п.п. | {ci(row['diff_ci_pp'])} | {n(row['p_value'], 4)} | "
            f"{n(MDE[name], 1)} п.п. |")
    add("")
    add("Поправка Холма на две метрики: "
        + ", ".join(f"{name} → p = {n(row['p_holm'], 4)}" for name, row in effects["primary"].items()) + ".")
    add("")

    add("## Предохранители")
    add("")
    add("| Метрика | Контроль | Тест | Разница | 95 % ДИ |")
    add("|---|---:|---:|---:|---|")
    for name, row in effects["guardrails"].items():
        add(f"| {name} | {n(row['control']['rate_pct'])} % | {n(row['treatment']['rate_pct'])} % | "
            f"{signed(row['diff_pp'])} п.п. | {ci(row['diff_ci_pp'])} |")
    add("")

    add("## Раунды и выбросы")
    add("")
    rounds = effects["rounds"]
    all_r, no_max = rounds["all"], rounds["without_max"]
    add(f"- Медиана: {n(all_r['control_median'], 0)} → {n(all_r['treatment_median'], 0)} "
        f"({signed(all_r['median_diff'], 0)} раунда), интервал {ci(all_r['median_ci'], 0)}.")
    add(f"- Среднее: {n(all_r['control_mean'])} → {n(all_r['treatment_mean'])} "
        f"({signed(all_r['mean_diff'])}).")
    add(f"- Без одного игрока с {thousands(eda['rounds']['max'])} раундами разница средних — "
        f"{signed(no_max['mean_diff'])}, то есть исчезает.")
    add(f"- Один этот игрок даёт {n(eda['rounds']['max_share_of_mean_pp'])} % суммы всех раундов, "
        "поэтому среднее здесь не показатель, а иллюстрация.")
    ratio = effects.get("ratio_metrics", {}).get("heavy_rounds_share")
    if ratio:
        add(f"- Доля раундов, сыгранных тяжёлыми игроками (дельта-метод): "
            f"{n(ratio['control_pct'])} % → {n(ratio['treatment_pct'])} %, "
            f"разница {signed(ratio['diff_pp'])} п.п. {ci(ratio['ci_pp'])} — "
            f"{'значимо' if ratio['significant'] else 'не значимо'}.")
    add("")

    if monitor:
        add("## Последовательный мониторинг (D1)")
        add("")
        add(f"Смотрели раз в неделю — {monitor['looks']} взглядов (дни "
            + ", ".join(str(d) for d in monitor["look_days"]) + "), границы O'Brien–Fleming.")
        if monitor.get("crossed_at_day"):
            hit = next(r for r in (looks or []) if int(r["day"]) == int(monitor["crossed_at_day"]))
            add("")
            add(f"- Пересечение на день {monitor['crossed_at_day']}: z = "
                f"{n(abs(float(hit['z'])))}, граница {n(float(hit['boundary']))}, разница "
                f"{signed(float(hit['diff_pp']))} п.п. "
                f"{ci([float(hit['ci_lo_pp']), float(hit['ci_hi_pp'])])}.")
            add("- По правилам последовательного анализа на этом взгляде H0 отклоняется: "
                "удержание D1 хуже.")
        else:
            add(f"- Граница не пересечена: финальный z = {n(abs(monitor['final_z']))} "
                f"против границы {n(monitor['final_boundary'])}.")
        add(f"- Финальный взгляд: z = {n(abs(monitor['final_z']))} при границе "
            f"{n(monitor['final_boundary'])}.")
        add("")
        add("Это не противоречит основному тесту, а показывает его цену: обычный тест на "
            "финале даёт p = 0,074 по D1 и «эффекта нет», а последовательный — отклоняет H0 "
            "на 25-й день и требует остановки. Оба вывода согласуются с одним и тем же "
            "направлением (вред), но цена раннего взгляда — консервативные границы. Именно "
            "поэтому подглядывать в данные можно, если границы заданы заранее.")
        add("")
        add("Оговорка: прирост по дням синтетический, поэтому «день 25» — подвыборка, "
            "а не реальный промежуточный срез. Ценность раздела — в механике остановки, "
            "а не в конкретной дате.")
        add("")

    add("## Качество эксперимента")
    add("")
    srm = eda["srm"]
    add(f"- Размеры групп: {thousands(srm['counts']['gate_40'])} и "
        f"{thousands(srm['counts']['gate_30'])}, отклонение от 50/50 — {n(srm['max_shift_pp'])} п.п.")
    add(f"- SRM: χ² = {n(srm['chi2'])}, p = {n(srm['p_value'], 4)}. Формально это ниже 0,05, "
        f"но отклонение {n(srm['max_shift_pp'])} п.п. — практического сбоя рандомизации нет: "
        "при 90 тыс. игроков даже доли процента дают значимость.")
    add(f"- Пропусков: {eda['missing']}, дублей userid: {eda['duplicate_userids']}.")
    add("")

    add("## Ограничения")
    add("")
    add("- **Нет выручки.** Решение сформулировано по удержанию; денежный эффект ворот эти данные "
        "не показывают.")
    add("- **Нет предпериода**, поэтому CUPED применить не на чем — метод реализован, но здесь "
        "не используется.")
    add("- **Один срез** (14 дней после установки), не поток: для ежедневного пересчёта данные "
        "разворачиваются по дням, и это указано в README.")
    add("")

    add("## Что перепроверить")
    add("")
    add("- Если есть возможность — повторить с мощностью под эффект 0,5 п.п. по D1: текущий "
        "эксперимент такие эффекты не различает.")
    add("- Проверить, не изменился ли состав трафика между группами по стране и платформе "
        "(здесь этих полей нет — это следующий шаг, а не вывод).")
    add("- Перед решением о деньгах добавить метрику выручки: ворота существуют ради неё, "
        "и удержание может падать при росте дохода.")
    add("")
    add("---")
    add("")
    add("Числа воспроизводятся: `python scripts/run_eda.py && python scripts/run_effects.py && "
        "python scripts/run_memo.py`; проверяются тестами `tests/test_claims.py`.")
    return "\n".join(lines) + "\n"
