"""Корень src — в sys.path, чтобы тесты видели пакет experiment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
