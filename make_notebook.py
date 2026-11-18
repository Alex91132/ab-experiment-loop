"""Собирает и исполняет ноутбук из analysis.py (формат jupytext-ячеек).

    python make_notebook.py

Результат — `notebooks/experiment_tour.ipynb` с выполненными ячейками и картинками.
Ноутбук не хранится руками: он всегда собирается из `analysis.py`, поэтому не
расходится с кодом.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Jupyter пишет служебные файлы и ставит на них ACL — в песочнице это падает.
os.environ.setdefault("JUPYTER_ALLOW_INSECURE_WRITES", "1")
os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(ROOT / ".jupyter" / "run"))
os.environ.setdefault("JUPYTER_DATA_DIR", str(ROOT / ".jupyter" / "data"))
os.environ.setdefault("IPYTHONDIR", str(ROOT / ".jupyter" / "ipython"))
for key in ("JUPYTER_RUNTIME_DIR", "JUPYTER_DATA_DIR", "IPYTHONDIR"):
    Path(os.environ[key]).mkdir(parents=True, exist_ok=True)

# Свой kernelspec: системный `python3` может смотреть в чужое окружение, и тогда
# ноутбук падает на несовместимом numpy. Ядро запускаем тем же интерпретатором.
KERNEL_NAME = "project-python"
KERNEL_DIR = Path(os.environ["JUPYTER_DATA_DIR"]) / "kernels" / KERNEL_NAME
KERNEL_DIR.mkdir(parents=True, exist_ok=True)
(KERNEL_DIR / "kernel.json").write_text(json.dumps({
    "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
    "display_name": "Project Python",
    "language": "python",
}, ensure_ascii=False), encoding="utf-8")

import nbformat as nbf                      # noqa: E402
from nbclient import NotebookClient          # noqa: E402

SRC = ROOT / "analysis.py"
OUT = ROOT / "notebooks" / "experiment_tour.ipynb"


def build() -> nbf.NotebookNode:
    cells: list[tuple[str, list[str]]] = []
    for line in SRC.read_text(encoding="utf-8").splitlines(keepends=True):
        if line.startswith("# %%"):
            kind = "markdown" if "[markdown]" in line else "code"
            cells.append((kind, []))
        elif cells:
            cells[-1][1].append(line)

    notebook = nbf.v4.new_notebook()
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python",
                                       "name": KERNEL_NAME}
    for kind, lines in cells:
        source = "".join(lines).strip("\n")
        if kind == "markdown":
            source = "\n".join(l[2:] if l.startswith("# ") else l for l in source.splitlines())
            notebook.cells.append(nbf.v4.new_markdown_cell(source))
        else:
            notebook.cells.append(nbf.v4.new_code_cell(source))
    return notebook


def main() -> None:
    notebook = build()
    print(f"собрано ячеек: {len(notebook.cells)}")

    client = NotebookClient(notebook, timeout=900, kernel_name=KERNEL_NAME,
                            resources={"metadata": {"path": str(ROOT)}})
    client.execute()

    OUT.parent.mkdir(exist_ok=True)
    nbf.write(notebook, str(OUT))

    errors = [o for c in notebook.cells for o in c.get("outputs", [])
              if o.get("output_type") == "error"]
    images = sum(1 for c in notebook.cells for o in c.get("outputs", [])
                 if o.get("output_type") == "display_data" and "image/png" in o.get("data", {}))
    print(f"исполнено: {OUT.relative_to(ROOT)} | ошибок: {len(errors)} | картинок: {images}")
    if errors:
        raise SystemExit("в ноутбуке есть ошибки — см. вывод ячеек")


if __name__ == "__main__":
    main()
