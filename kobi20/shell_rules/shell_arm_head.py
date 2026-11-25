"""Rule F, arm against head (shell rule ``arm_head``): a table of shoulder-roll caps over shoulder pitch x neck yaw toward the arm x neck nod.

Source: the shell folder's coupled-limits file (rule id ``arm_head`` inside coupled_limits_v7.json, or the separate file
coupled_limits_v7rc1_armhead.json). Reading rule (the shell's own ``evaluation_order``):
  1. the neck yaw is turned to "toward the arm": neck yaw for the left arm, minus neck yaw for the right arm;
  2. on each of the three axes (shoulder pitch, neck yaw toward the arm, neck nod) take the nearest node at or below and the nearest node at
     or above the value (a value ON a node: that node only; outside the node range: the end node);
  3. roll_cap is the SMALLEST table value among those nodes (up to 8);
  4. roll_cap below 0 (-1): this shoulder pitch is not allowed with this neck at any roll; otherwise 0 <= shoulder roll offset <= roll_cap.
Angles are offsets from the rest pose (shoulder roll rest +20). The rule is applied IN ADDITION to the arm boxes and the neck boxes (rules D, E);
it needs the joints shoulder pitch, shoulder roll, neck yaw and neck pitch of the pose, else it is skipped for that arm.
The data is read from limits["F_arm_head"] (made by convert_coupled_limits_v2.py). Pure Python, no packages.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from shell_coupled_limits import SIDES, TOLERANCE_DEG, Violation

RULE_F = "F_arm_head"
NECK_YAW_JOINT = "neck_yaw_joint"
NECK_PITCH_JOINT = "neck_pitch_joint"


def bracket_indices(nodes: Sequence[float], value: float) -> tuple[int, int]:
    """Indices of the nearest node at or below and at or above the value; on a node both are that node; outside the range the end node."""
    below = [i for i, node in enumerate(nodes) if node <= value]
    above = [i for i, node in enumerate(nodes) if node >= value]
    low = below[-1] if below else 0
    high = above[0] if above else len(nodes) - 1
    return low, high


def roll_cap_deg(rule: Mapping[str, Any], pitch_deg: float, yaw_toward_deg: float, nod_deg: float) -> float:
    """The smallest table value among the (up to 8) grid nodes around the three values; -1 means not allowed at any roll."""
    table = rule["table"]
    p_lo, p_hi = bracket_indices(table["shoulder_pitch_nodes"], pitch_deg)
    y_lo, y_hi = bracket_indices(table["neck_yaw_toward_arm_nodes"], yaw_toward_deg)
    n_lo, n_hi = bracket_indices(table["neck_pitch_nodes"], nod_deg)
    return min(table["roll_cap"][p][y][n] for p in {p_lo, p_hi} for y in {y_lo, y_hi} for n in {n_lo, n_hi})


def arm_head_violations(pose_deg: Mapping[str, float], limits: Mapping[str, Any]) -> list[Violation]:
    """Rule F violations of a pose given in degrees (empty when the limits hold no arm_head rule: see arm_head_present)."""
    rule = limits.get(RULE_F)
    if rule is None:
        return []
    yaw, nod = pose_deg.get(NECK_YAW_JOINT), pose_deg.get(NECK_PITCH_JOINT)
    if yaw is None or nod is None:
        return []
    rest_roll = limits["rest_pose_deg"]["shoulder_roll"]
    found: list[Violation] = []
    for side in SIDES:
        pitch, roll_joint = pose_deg.get(f"{side}_shoulder_pitch_joint"), f"{side}_shoulder_roll_joint"
        if pitch is None or roll_joint not in pose_deg:
            continue
        roll = pose_deg[roll_joint] - rest_roll
        toward = yaw * rule["neck_yaw_sign"][side]
        cap = roll_cap_deg(rule, pitch, toward, nod)
        if cap < 0.0:
            found.append(Violation(RULE_F, roll_joint, roll, cap, f"shoulder pitch {pitch:.3f} is not allowed with this neck (yaw toward the arm {toward:.3f}, nod {nod:.3f}) at any roll"))
        elif roll > cap + TOLERANCE_DEG:
            found.append(Violation(RULE_F, roll_joint, roll, cap, f"shoulder roll offset may be at most {cap:.3f} deg at shoulder pitch {pitch:.3f}, neck yaw toward the arm {toward:.3f}, nod {nod:.3f}"))
        elif roll < 0.0 - TOLERANCE_DEG:
            found.append(Violation(RULE_F, roll_joint, roll, 0.0, "shoulder roll offset must be at least 0 (arm_head inequality)"))
    return found
