"""Rule G, arm against the back wing (shell rule ``arm_wing``): a table of shoulder-pitch caps over shoulder roll x arm yaw x elbow.

Source: rule id ``arm_wing`` of the shell folder's coupled-limits file (coupled_limits_v7_armwing.json, given with --coupled-extra, or inside the
main file). Reading rule (the shell's own ``evaluation_order``):
  1. the pose must also be inside the old arm box of its shoulder roll and satisfy the other rules (rules D and F report those);
  2. on each of the three axes (shoulder roll, arm yaw, elbow) take the nearest node at or below and the nearest node at or above the value
     (a value ON a node: that node only; outside the node range: the end node);
  3. pitch_cap = the SMALLEST table value among those nodes (up to 8);
  4. shoulder pitch (positive = backward) must be <= pitch_cap. A table value equal to the box end (60) means no limit beyond the box.
Angles are offsets from the rest pose (shoulder roll rest +20, elbow rest -18, arm yaw rest 0). The right arm uses the same numbers (its joints
are mirrored in the kinematics file, so the model joint values are directly comparable). The rule is skipped for an arm when one of its four
joints is missing. Data: limits["G_arm_wing"] (made by convert_coupled_limits_v2.py). Pure Python, no packages.

v8 (20-joint edition, arm yaw frozen at 0): the table is the slice of the 25-joint table at the arm_yaw node 0.0, so it has two axes (shoulder roll, elbow); the
rule needs the three joints shoulder pitch, shoulder roll and elbow of an arm, and pitch_cap_deg ignores its yaw argument.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import product
from typing import Any

from shell_arm_head import bracket_indices
from shell_coupled_limits import SIDES, TOLERANCE_DEG, Violation

RULE_G = "G_arm_wing"


def pitch_cap_deg(rule: Mapping[str, Any], roll_deg: float, yaw_deg: float, elbow_deg: float) -> float:
    """The smallest table value among the (up to 8) grid nodes around the three values (offsets from the rest pose)."""
    table = rule["table"]
    r_lo, r_hi = bracket_indices(table["shoulder_roll_nodes"], roll_deg)
    e_lo, e_hi = bracket_indices(table["elbow_nodes"], elbow_deg)
    if "arm_yaw_nodes" not in table:  # arm yaw frozen: two axes
        return min(table["pitch_cap"][r][e] for r, e in product({r_lo, r_hi}, {e_lo, e_hi}))
    y_lo, y_hi = bracket_indices(table["arm_yaw_nodes"], yaw_deg)
    return min(table["pitch_cap"][r][y][e] for r, y, e in product({r_lo, r_hi}, {y_lo, y_hi}, {e_lo, e_hi}))


def arm_wing_violations(pose_deg: Mapping[str, float], limits: Mapping[str, Any]) -> list[Violation]:
    """Rule G violations of a pose given in degrees (empty when the data holds no arm_wing rule)."""
    rule = limits.get(RULE_G)
    if rule is None:
        return []
    rest = limits["rest_pose_deg"]
    found: list[Violation] = []
    for side in SIDES:
        frozen_yaw = "arm_yaw_nodes" not in rule["table"]
        keys = ("shoulder_pitch", "shoulder_roll", "elbow") if frozen_yaw else ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")
        names = {key: f"{side}_{key}_joint" for key in keys}
        if any(name not in pose_deg for name in names.values()):
            continue
        roll = pose_deg[names["shoulder_roll"]] - rest["shoulder_roll"]
        elbow = pose_deg[names["elbow"]] - rest["elbow"]
        yaw = 0.0 if frozen_yaw else pose_deg[names["shoulder_yaw"]]
        pitch = pose_deg[names["shoulder_pitch"]]
        cap = pitch_cap_deg(rule, roll, yaw, elbow)
        if pitch > cap + TOLERANCE_DEG:
            found.append(Violation(RULE_G, names["shoulder_pitch"], pitch, cap, f"shoulder pitch may be at most {cap:.3f} deg at roll offset {roll:.3f}, arm yaw {yaw:.3f}, elbow offset {elbow:.3f} (arm against the back wing)"))
    return found
