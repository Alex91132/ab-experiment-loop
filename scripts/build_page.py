"""Собирает docs/index.html — страницу результатов без внешних зависимостей.

Графики рисуются как SVG прямо при сборке: страница открывается как файл,
грузится мгновенно и не требует ни JS, ни CDN.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "docs" / "index.html"

def f2(x: float, nd: int = 2) -> str:
    """Число с запятой: -0.82 → «-0,82»."""
    return f"{x:.{nd}f}".replace(".", ",")


def fs(x: float, nd: int = 2) -> str:
    """Число со знаком: 0.19 → «+0,19»."""
    return f"{x:+.{nd}f}".replace(".", ",")


CSS = """
:root { --ink:#1c2530; --muted:#5b6b7c; --line:#d7e0e8; --accent:#2b6cb0; --good:#2f855a; --bad:#c53030; }
* { box-sizing: border-box; }
body { margin:0; font:16px/1.55 -apple-system,Segoe UI,Roboto,sans-serif; color:var(--ink); background:#fbfcfd; }
main { max-width: 920px; margin: 0 auto; padding: 48px 24px 80px; }
h1 { font-size: 30px; line-height:1.25; margin: 0 0 8px; }
h2 { font-size: 20px; margin: 40px 0 12px; }
p.lead { color: var(--muted); margin: 0 0 24px; }
.verdict { border-left:5px solid var(--bad); background:#fff; padding:16px 20px; border-radius:8px;
           box-shadow: 0 1px 2px rgba(16,24,40,.06); }
.verdict strong { display:block; font-size:18px; margin-bottom:6px; }
table { border-collapse: collapse; width:100%; font-size:15px; background:#fff; }
th, td { padding:10px 12px; border-bottom:1px solid var(--line); text-align:right; }
th:first-child, td:first-child { text-align:left; }
thead th { color:var(--muted); font-weight:600; font-size:13px; text-transform:uppercase; letter-spacing:.03em; }
tfoot td { font-weight:600; }
.card { background:#fff; border:1px solid var(--line); border-radius:10px; padding:16px 20px; margin-top:16px; }
.muted { color:var(--muted); font-size:14px; }
a { color: var(--accent); }
.grid { display:grid; grid-template-columns: 1fr 1fr; gap:16px; }
@media (max-width: 720px) { .grid { grid-template-columns: 1fr; } }
"""


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def load_csv(name: str) -> list[dict]:
    """Читает CSV и приводит числа к float — иначе графики получат строки."""
    import csv

    rows = []
    with (RESULTS / name).open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            parsed = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = value
            rows.append(parsed)
    return rows


def chart_daily(rows: list[dict], width: int = 860, height: int = 280) -> str:
    """Разница D1 по дням с интервалом: линия + полоса доверия."""
    days = [int(r["day"]) for r in rows]
    diff = [float(r["diff_pp"]) for r in rows]
    lo = [float(r["ci_lo_pp"]) for r in rows]
    hi = [float(r["ci_hi_pp"]) for r in rows]
    y_min = min(min(lo), -1.0) - 0.3
    y_max = max(max(hi), 1.0) + 0.3
    left, right, top, bottom = 48, 16, 16, 32
    w, h = width - left - right, height - top - bottom

    def x(i: int) -> float:
        return left + w * i / max(len(days) - 1, 1)

    def y(v: float) -> float:
        return top + h * (y_max - v) / (y_max - y_min)

    band = " ".join(f"{x(i):.1f},{y(hi[i]):.1f}" for i in range(len(days))) + " " + \
           " ".join(f"{x(i):.1f},{y(lo[i]):.1f}" for i in reversed(range(len(days))))
    line = " ".join(f"{x(i):.1f},{y(diff[i]):.1f}" for i in range(len(days)))
    zero = y(0)

    ticks = ""
    for v in (-4, -3, -2, -1, 0, 1, 2):
        if y_min < v < y_max:
            ticks += f'<line x1="{left}" y1="{y(v):.1f}" x2="{width-right}" y2="{y(v):.1f}" stroke="#eef2f6"/>'
            ticks += f'<text x="{left-8}" y="{y(v)+4:.1f}" font-size="11" fill="#5b6b7c" text-anchor="end">{v}</text>'
    labels = "".join(
        f'<text x="{x(i):.1f}" y="{height-10}" font-size="11" fill="#5b6b7c">{d}</text>'
        for i, d in enumerate(days) if d in (1, 5, 10, 15, 20, 25, 30)
    )
    return f"""<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="Разница D1 по дням">
  {ticks}
  <line x1="{left}" y1="{zero:.1f}" x2="{width-right}" y2="{zero:.1f}" stroke="#b9c6d3" stroke-dasharray="4 3"/>
  <polygon points="{band}" fill="#2b6cb0" opacity="0.12"/>
  <polyline points="{line}" fill="none" stroke="#2b6cb0" stroke-width="2"/>
  {labels}
  <text x="{left}" y="{top-2}" font-size="11" fill="#5b6b7c">разница, п.п. (95 % интервал — полоса)</text>
</svg>"""


def chart_monitor(rows: list[dict], width: int = 860, height: int = 260) -> str:
    """z-статистика против границы O'Brien–Fleming по взглядам."""
    days = [int(r["day"]) for r in rows]
    z = [abs(float(r["z"])) for r in rows]
    b = [float(r["boundary"]) for r in rows]
    y_max = max(max(z), max(b)) + 0.6
    left, right, top, bottom = 48, 16, 16, 32
    w, h = width - left - right, height - top - bottom

    def x(i: int) -> float:
        return left + w * i / max(len(days) - 1, 1)

    def y(v: float) -> float:
        return top + h * (y_max - v) / y_max

    zl = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(z))
    bl = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(b))
    ticks = "".join(
        f'<line x1="{left}" y1="{y(v):.1f}" x2="{width-right}" y2="{y(v):.1f}" stroke="#eef2f6"/>'
        f'<text x="{left-8}" y="{y(v)+4:.1f}" font-size="11" fill="#5b6b7c" text-anchor="end">{v:.0f}</text>'
        for v in range(0, int(y_max) + 1)
    )
    labels = "".join(
        f'<text x="{x(i):.1f}" y="{height-10}" font-size="11" fill="#5b6b7c">{d}</text>'
        for i, d in enumerate(days)
    )
    return f"""<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="Мониторинг">
  {ticks}
  <polyline points="{bl}" fill="none" stroke="#38a169" stroke-width="2" stroke-dasharray="5 3"/>
  <polyline points="{zl}" fill="none" stroke="#553c9a" stroke-width="2"/>
  {labels}
  <text x="{left}" y="{top-2}" font-size="11" fill="#5b6b7c">|z| (фиолетовая) и граница OBF (зелёная)</text>
</svg>"""


def main() -> None:
    effects = load("effects.json")
    eda = load("eda.json")
    monitor = load("monitor.json")
    daily = load_csv("daily_metrics.csv")
    looks = load_csv("sequential_looks.csv")

    primary = effects["primary"]
    guardrails = effects["guardrails"]
    ratio = effects.get("ratio_metrics", {}).get("heavy_rounds_share", {})
    rounds = effects["rounds"]
    srm = eda["srm"]

    rows_primary = "".join(
        f"<tr><td>{name}</td>"
        f"<td>{f2(r['control']['rate_pct'])} %</td><td>{f2(r['treatment']['rate_pct'])} %</td>"
        f"<td>{fs(r['diff_pp'])} п.п.</td>"
        f"<td>[{fs(r['diff_ci_pp'][0])}; {fs(r['diff_ci_pp'][1])}]</td>"
        f"<td>{f2(r['p_value'], 4)}</td><td>{f2(r['p_holm'], 4)}</td></tr>"
        for name, r in primary.items()
    )
    rows_guard = "".join(
        f"<tr><td>{name}</td><td>{f2(r['control']['rate_pct'])} %</td>"
        f"<td>{f2(r['treatment']['rate_pct'])} %</td><td>{fs(r['diff_pp'])} п.п.</td>"
        f"<td>[{fs(r['diff_ci_pp'][0])}; {fs(r['diff_ci_pp'][1])}]</td></tr>"
        for name, r in guardrails.items()
    )

    ratio_line = ""
    if ratio:
        ratio_line = (f"<p class='muted'>Доля раундов у тяжёлых игроков (дельта-метод): "
                      f"{f2(ratio['control_pct'])} % → {f2(ratio['treatment_pct'])} %, "
                      f"разница {fs(ratio['diff_pp'])} п.п. "
                      f"[{fs(ratio['ci_pp'][0])}; {fs(ratio['ci_pp'][1])}] — "
                      f"{'значимо' if ratio['significant'] else 'не значимо'}.</p>")

    html = f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cookie Cats: анализ A/B-эксперимента</title>
<style>{CSS}</style></head>
<body><main>
<h1>Перенос «ворот» в Cookie Cats: стоит ли двигать</h1>
<p class="lead">A/B-эксперимент на 90 189 игроках. Решение принято по правилам, зафиксированным до анализа.</p>

<div class="verdict">
  <strong>Решение: ворота оставить на 30-м уровне</strong>
  Удержание на 7-й день значимо хуже: −{f2(abs(primary['retention_7']['diff_pp']))} п.п.
  [{fs(primary['retention_7']['diff_ci_pp'][0])}; {fs(primary['retention_7']['diff_ci_pp'][1])}],
  p после поправки Холма — {f2(primary['retention_7']['p_holm'], 4)}. Ухудшение выше чувствительности
  эксперимента (MDE 0,7 п.п.), то есть это не «мало данных».
</div>

<h2>Основные метрики</h2>
<table><thead><tr><th>Метрика</th><th>Контроль</th><th>Тест</th><th>Разница</th><th>95 % ДИ</th><th>p</th><th>p (Холм)</th></tr></thead>
<tbody>{rows_primary}</tbody></table>
<p class="muted">Удержание D1 не значимо: эффект −{f2(abs(primary['retention_1']['diff_pp']))} п.п. ниже MDE 0,9 п.п.,
поэтому вывод — «недостаточно данных», а не «эффекта нет».</p>

<h2>Предохранители</h2>
<table><thead><tr><th>Метрика</th><th>Контроль</th><th>Тест</th><th>Разница</th><th>95 % ДИ</th></tr></thead>
<tbody>{rows_guard}</tbody></table>
{ratio_line}

<h2>Как шло накопление данных</h2>
<div class="card">{chart_daily(daily)}</div>

<h2>Последовательный мониторинг</h2>
<div class="card">{chart_monitor(looks)}
<p class="muted">Пересечение на день {monitor.get('crossed_at_day') or '—'} — так работает остановка
по границам O'Brien–Fleming: подглядывать можно, если границы заданы заранее.</p></div>

<h2>Качество эксперимента</h2>
<div class="card">
<p>Группы: {srm['counts']['gate_40']:,} и {srm['counts']['gate_30']:,}, отклонение от 50/50 — {f2(srm['max_shift_pp'])} п.п.<br>
SRM: χ² = {f2(srm['chi2'])}, p = {f2(srm['p_value'], 4)}. Формально ниже 0,05, но практического сбоя рандомизации нет.</p>
<p class="muted">Раунды: медиана {rounds['all']['control_median']:.0f} → {rounds['all']['treatment_median']:.0f};
разница средних {fs(rounds['all']['mean_diff'])} исчезает без одного игрока с 49 854 раундами
({fs(rounds['without_max']['mean_diff'])}).</p>
</div>

<h2>Что дальше</h2>
<p>Данных нет по выручке: ворота существуют ради неё. Решение о деньгах требует отдельной метрики.
Подробности и правила — в <a href="../PLAN.md">PLAN.md</a>, полный текст — в
<a href="../results/memo.md">results/memo.md</a>.</p>
</main></body></html>
"""
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"собрано: {OUT.relative_to(ROOT)} ({len(html) / 1024:.0f} КБ)")


if __name__ == "__main__":
    main()
