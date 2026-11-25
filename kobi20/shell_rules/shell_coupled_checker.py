"""The model's coupled-limit checker: rules A to E of shell_coupled_limits.py plus rule F (arm against head) and rule G (arm against the back wing).

Use this module (not shell_coupled_limits.py) when arm and neck poses are judged. check_pose has the same signature as in shell_coupled_limits.
clip_leg_targets (legs only) is the one of shell_coupled_limits.py, unchanged. Rule F: shell_arm_head.py, rule G: shell_arm_wing.py.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import shell_arm_head as arm_head
import shell_arm_wing as arm_wing
import shell_coupled_limits as base
from shell_coupled_limits import Violation, clip_leg_targets, load_limits

__all__ = ["RULE_F", "RULE_G", "Violation", "arm_head_present", "arm_wing_present", "check_pose", "clip_leg_targets", "load_limits", "violates"]

RULE_F = arm_head.RULE_F
RULE_G = arm_wing.RULE_G


def arm_head_present(limits: Mapping[str, Any]) -> bool:
    """True when the converted data holds the arm_head rule."""
    return RULE_F in limits


def arm_wing_present(limits: Mapping[str, Any]) -> bool:
    """True when the converted data holds the arm_wing rule."""
    return RULE_G in limits


def check_pose(pose: Mapping[str, float], limits: Mapping[str, Any] | None = None, unit: str = "rad") -> list[Violation]:
    """Every coupled limit a pose breaks: rules A to E, then rule F and rule G (when the data holds them)."""
    table = limits if limits is not None else load_limits()
    found = base.check_pose(pose, table, unit)
    pose_deg = base._to_degrees(pose, unit)
    return found + arm_head.arm_head_violations(pose_deg, table) + arm_wing.arm_wing_violations(pose_deg, table)


def violates(pose: Mapping[str, float], limits: Mapping[str, Any] | None = None, unit: str = "rad") -> bool:
    """True when the pose breaks at least one coupled limit."""
    return bool(check_pose(pose, limits, unit))
