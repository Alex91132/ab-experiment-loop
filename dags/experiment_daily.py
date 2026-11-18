"""Ежедневный контур эксперимента: пересчёт метрик, мониторинг и алерты.

DAG только связывает шаги из `experiment.pipeline`; сами расчёты проверяются
тестами без Airflow. Запуск по расписанию раз в день, догоняющих прогонов нет:
пересчитываем накопленный срез заново — операция идемпотентная.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.exceptions import AirflowException
from airflow.operators.python import PythonOperator

PROJECT = Path("/opt/airflow/project")
if str(PROJECT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT / "src"))

from experiment import pipeline  # noqa: E402

RESULTS = PROJECT / "results"

default_args = {
    "owner": "analytics",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def guard_check() -> None:
    """Падает, если данные или предохранители вышли за пределы — это и есть алерт."""
    validation = json.loads((RESULTS / "validation.json").read_text(encoding="utf-8"))
    if validation["srm_p"] < 0.001 and validation["srm_shift_pp"] > 1.0:
        raise AirflowException(
            f"SRM: p={validation['srm_p']}, отклонение {validation['srm_shift_pp']} п.п. — "
            "проверить рандомизацию до публикации результатов"
        )

    effects = json.loads((RESULTS / "dag_effects.json").read_text(encoding="utf-8"))
    broken = {name: row for name, row in effects.items()
              if name in ("zero_rounds", "heavy_user") and row["significant"]
              and ((name == "zero_rounds" and row["diff_pp"] > 0)
                   or (name == "heavy_user" and row["diff_pp"] < 0))}
    if broken:
        raise AirflowException(f"guardrail сломан: {json.dumps(broken, ensure_ascii=False)}")


with DAG(
    dag_id="experiment_daily",
    description="Cookie Cats: ежедневный пересчёт эффектов, последовательный мониторинг, алерты",
    default_args=default_args,
    start_date=datetime(2026, 11, 1),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    tags=["ab-test", "experiment"],
) as dag:
    ingest = PythonOperator(task_id="ingest", python_callable=pipeline.step_ingest)
    validate = PythonOperator(task_id="validate", python_callable=pipeline.step_validate)
    daily = PythonOperator(task_id="daily_metrics", python_callable=pipeline.step_daily_metrics)
    monitor = PythonOperator(task_id="sequential_monitor", python_callable=pipeline.step_monitor)
    effects = PythonOperator(task_id="effects", python_callable=pipeline.step_effects)
    publish = PythonOperator(task_id="publish_marts", python_callable=pipeline.step_publish)
    guard = PythonOperator(task_id="guard_check", python_callable=guard_check)

    ingest >> validate >> daily >> monitor >> effects >> publish >> guard
