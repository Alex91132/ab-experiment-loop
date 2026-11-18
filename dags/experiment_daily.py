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
from experiment.alerts import check_alerts  # noqa: E402

RESULTS = PROJECT / "results"

default_args = {
    "owner": "analytics",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def guard_check() -> None:
    """Падает, если данные или предохранители вышли за пределы — это и есть алерт."""
    validation = json.loads((RESULTS / "validation.json").read_text(encoding="utf-8"))
    effects = json.loads((RESULTS / "dag_effects.json").read_text(encoding="utf-8"))
    problems = check_alerts(validation, effects)
    if problems:
        raise AirflowException("; ".join(problems))


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
    memo = PythonOperator(task_id="render_memo", python_callable=pipeline.step_memo)
    guard = PythonOperator(task_id="guard_check", python_callable=guard_check)

    ingest >> validate >> daily >> monitor >> effects >> publish >> memo >> guard
