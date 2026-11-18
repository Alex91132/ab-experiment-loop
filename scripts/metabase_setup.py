"""Настройка Metabase через API: админ, подключение к Postgres, карточки, дашборд.

Запускать после `docker-compose up -d`, когда Metabase поднялся:

    python scripts/metabase_setup.py

Скрипт идемпотентен настолько, насколько это возможно: если уже настроен —
сообщит и выйдет, не ломая существующее.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://localhost:3000"
ADMIN = {"email": "analytics@example.com", "password": "analytics123", "first_name": "Analytics",
         "last_name": "Team", "site_name": "Cookie Cats Experiments"}

DB_DETAILS = {
    "host": "postgres", "port": 5432, "db": "airflow",
    "user": "airflow", "password": "airflow", "schema": "analytics",
    "ssl": False, "tunnel-enabled": False,
}

CARDS = [
    {
        "name": "Динамика разницы D1 по дням",
        "description": "Накопительная разница удержания D1 и её интервал",
        "display": "line",
        "sql": "SELECT day, diff_pp, ci_lo_pp, ci_hi_pp FROM analytics.exp_daily_metrics ORDER BY day",
        "viz": {"graph.dimensions": ["day"], "graph.metrics": ["diff_pp", "ci_lo_pp", "ci_hi_pp"]},
    },
    {
        "name": "Последовательный мониторинг: z и граница",
        "description": "Еженедельные взгляды, границы O'Brien–Fleming",
        "display": "line",
        "sql": "SELECT day, z, boundary FROM analytics.exp_sequential_looks ORDER BY day",
        "viz": {"graph.dimensions": ["day"], "graph.metrics": ["z", "boundary"]},
    },
    {
        "name": "Качество эксперимента (SRM)",
        "description": "Размеры групп, хи-квадрат и отклонение от 50/50",
        "display": "table",
        "sql": ("SELECT rows, gate_30, gate_40, srm_chi2, srm_p, srm_shift_pp "
                "FROM analytics.exp_quality"),
        "viz": {},
    },
    {
        "name": "Эффекты по метрикам",
        "description": "Контроль, тест, разница и значимость",
        "display": "table",
        "sql": ("SELECT metric, control_pct, treatment_pct, diff_pp, significant "
                "FROM analytics.exp_effects ORDER BY metric"),
        "viz": {},
    },
]


def call(path: str, payload: dict | None = None, method: str = "GET", session: str | None = None) -> dict:
    url = BASE + path
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Content-Type", "application/json")
    if session:
        request.add_header("X-Metabase-Session", session)
    with urllib.request.urlopen(request, timeout=120) as response:
        body = response.read().decode()
    return json.loads(body) if body else {}


def wait_for_metabase(timeout: int = 300) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            call("/api/health")
            return
        except (urllib.error.URLError, ConnectionError):
            time.sleep(5)
    sys.exit("Metabase не поднялся за отведённое время")


def ensure_setup() -> str:
    """Возвращает сессионный токен; при первом запуске настраивает админа и базу."""
    props = call("/api/session/properties")
    token = props.get("setup-token")
    if token:
        print("первый запуск: создаю админа и подключение к Postgres")
        payload = {
            "token": token,
            "user": {k: ADMIN[k] for k in ("email", "password", "first_name", "last_name")},
            "prefs": {"site_name": ADMIN["site_name"], "allow_tracking": False},
            "database": {"engine": "postgres", "name": "analytics", "details": DB_DETAILS,
                         "is_full_sync": True, "is_on_demand": False},
        }
        result = call("/api/setup", payload, method="POST")
        return result.get("id") or login()
    return login()


def login() -> str:
    result = call("/api/session", {"username": ADMIN["email"], "password": ADMIN["password"]},
                  method="POST")
    return result["id"]


def find_database(session: str) -> int:
    databases = call("/api/database", session=session)["data"]
    for db in databases:
        if db["name"] == "analytics":
            return db["id"]
    created = call("/api/database", {"engine": "postgres", "name": "analytics",
                                     "details": DB_DETAILS}, method="POST", session=session)
    return created["id"]


def ensure_card(session: str, db_id: int, spec: dict) -> int:
    existing = call("/api/card", session=session)
    for card in existing:
        if card["name"] == spec["name"]:
            return card["id"]

    payload = {
        "name": spec["name"],
        "description": spec["description"],
        "display": spec["display"],
        "dataset_query": {"database": db_id, "type": "native", "native": {"query": spec["sql"]}},
        "visualization_settings": spec["viz"],
    }
    created = call("/api/card", payload, method="POST", session=session)
    return created["id"]


def ensure_dashboard(session: str, card_ids: list[int]) -> int:
    """Создаёт дашборд и раскладывает карточки.

    В Metabase 0.6x эндпоинта `/api/dashboard/{id}/cards` больше нет: сетка
    задаётся через PUT `/api/dashboard/{id}` c массивом dashcards.
    """
    name = "Эксперимент Cookie Cats"
    dashboards = call("/api/dashboard", session=session)
    existing = [d for d in dashboards if d["name"] == name]
    if existing:
        return existing[0]["id"]

    dash = call("/api/dashboard", {"name": name, "description": "Удержание, мониторинг, эффекты"},
                method="POST", session=session)
    dash_id = dash["id"]

    layout = [
        ("Эффект", [card_ids[0], card_ids[2]], [8, 6]),
        ("Мониторинг", [card_ids[1]], [8]),
        ("Качество", [card_ids[3]], [6]),
    ]
    tabs = [{"id": -(i + 1), "name": name} for i, (name, _, _) in enumerate(layout)]

    dashcards, index = [], 0
    for tab_index, (_, ids, heights) in enumerate(layout):
        row = 0
        for card_id, height in zip(ids, heights):
            index += 1
            dashcards.append({
                "id": -(100 + index), "card_id": card_id, "row": row, "col": 0,
                "size_x": 24, "size_y": height, "dashboard_tab_id": tabs[tab_index]["id"],
                "parameter_mappings": [], "visualization_settings": {}, "dashboard_id": dash_id,
            })
            row += height

    # в Metabase 0.6x карточки живут во вкладке: без tabs они не отрисуются
    call(f"/api/dashboard/{dash_id}", {"tabs": tabs, "dashcards": dashcards},
         method="PUT", session=session)
    return dash_id


def main() -> None:
    wait_for_metabase()
    session = ensure_setup()
    db_id = find_database(session)
    print(f"база analytics: id {db_id}")

    card_ids = [ensure_card(session, db_id, spec) for spec in CARDS]
    print(f"карточки: {card_ids}")

    dash_id = ensure_dashboard(session, card_ids)
    print(f"дашборд: http://localhost:3000/dashboard/{dash_id}")
    print("готово")
    (Path(__file__).resolve().parents[1] / "results" / "metabase.txt").write_text(
        f"http://localhost:3000/dashboard/{dash_id}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
