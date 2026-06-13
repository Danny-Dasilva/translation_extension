"""Pytest configuration: put ``backend/scripts/data`` on sys.path so the CLI
modules (which import each other as siblings, not as a package) are importable
from tests."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _REPO_ROOT / "backend" / "scripts" / "data"
if str(_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_DIR))
