"""Stand-in for the model builder's edition_inputs module so that shell_coupled_limits.py runs on its own.

shell_coupled_limits.py imports load_edition and folder_path only to find its data file when load_limits() is called WITHOUT a path.
In this repository the data file is always given explicitly (see check_pose.py), so these two functions only explain what to do.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

HINT = "pass the data file explicitly: load_limits(Path('shell_coupled_limits_v8.json')) (this folder has no model-builder edition file)"


def load_edition(name: str | None = None) -> tuple[str, dict[str, Any]]:
    """Not available here; see HINT."""
    raise RuntimeError(HINT)


def folder_path(relative: str) -> Path:
    """Not available here; see HINT."""
    raise RuntimeError(HINT)
