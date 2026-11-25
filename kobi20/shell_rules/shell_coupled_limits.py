"""Coupled joint limits of the costume shell (rules A to E of the shell folder): does a pose break one, and which?

The data file (config/shell_coupled_limits_<edition>.json) is made by convert_coupled_limits.py from the shell folder's
coupled-limits file; this module only reads it. It needs no MuJoCo and no third-party package (the array function at the
bottom takes numpy or jax.numpy through the parameter xp, so the walking environment can clip commands inside a jitted step).

Rules (ids as in the source file): A between_legs, B waist_with_legs, C knee_ankle_table, D arm_boxes_by_shoulder_roll, E neck.
Poses map model joint names (left_hip_roll_joint, ...) to angles (default radians, unit="deg" for degrees). Arm joints are
measured from the rest pose (the data file holds it). PROVISIONAL: every number may change; nothing is enforced by default.

v7 reads the strict-side ambiguity fixes of the shell's coupled-limits file (schema of coupled_limits_v7a.json) exactly:
  M5 (rule B)  omega <= 0.75 (also negative): hip yaw <= 15.0; the step table starts above 0.75 (first entry omega 1.0).
  M4 (rule C)  knee table for knee 0..60: ankle pitch <= table(knee) and >= -24; a knee outside 0..60 is allowed only when every other joint
               is at rest (a single-joint pose), otherwise it is a violation.
  M2 (rule E)  neck_nod only for |neck yaw| < 45 (open edge); at exactly 45 degrees the neck_turn box (pitch -14..7.0) applies.
  M3           the waist range [-4.5, 4.5] is a precondition of rules B and E; a waist outside it is reported once, as rule B.
Comparisons use a tolerance of 1e-9 degrees (the source's own tolerance for the waist range); v6g used 1e-6.

v8 (20-joint edition, data file made from coupled_limits_v8.json): five joints are FROZEN at 0 and are no joints of the model (waist yaw, both arm yaws,
both wrists). The data file lists them under frozen_joints_deg. Rules read the frozen value, not the pose: rule B has the form "frozen_waist_deg"
(omega = 0, so only the inactive limit hip yaw <= 15.0 is left, no table, no waist range); rule D has no arm_yaw range; the neck rule has no waist
precondition. A pose that still holds a frozen joint at another value than its frozen one breaks the extra rule Z_frozen_joint. A 25-joint data file
(no frozen_joints_deg) is read exactly as before.
"""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from edition_inputs import folder_path, load_edition

TOLERANCE_DEG = 1e-9
SIDES = ("left", "right")
RULE_A = "A_between_legs"
RULE_B = "B_waist_with_legs"
RULE_C = "C_knee_ankle"
RULE_D = "D_arm"
RULE_E = "E_neck"
RULE_FROZEN = "Z_frozen_joint"
WAIST_JOINT = "waist_yaw_joint"
NECK_YAW_JOINT = "neck_yaw_joint"
NECK_PITCH_JOINT = "neck_pitch_joint"


@dataclass(frozen=True)
class Violation:
    """One broken coupled limit."""

    rule: str
    joint: str
    value_deg: float
    limit_deg: float
    detail: str

    def __str__(self) -> str:
        return f"{self.rule}: {self.joint} = {self.value_deg:.3f} deg against the limit {self.limit_deg:.3f} deg ({self.detail})"


def load_limits(path: Path | None = None) -> dict[str, Any]:
    """Read a coupled-limits data file; by default the one of the selected edition (KOBI_EDITION)."""
    if path is None:
        _, block = load_edition()
        path = folder_path(block["coupled_limits_file"])
    return json.loads(Path(path).read_text(encoding="utf-8"))


def interpolate(x: float, points: list[list[float]]) -> float:
    """Straight-line value through (x, y) points; flat before the first and after the last point."""
    if x <= points[0][0]:
        return points[0][1]
    for (x0, y0), (x1, y1) in itertools.pairwise(points):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return points[-1][1]


def hip_roll_minimum_deg(hip_yaw_deg: float, limits: Mapping[str, Any]) -> float:
    """Rule A: smallest hip roll that keeps the two shoes apart."""
    rule = limits["A_between_legs"]
    return max(
        rule["hip_roll_minimum_floor"],
        -hip_yaw_deg,
        rule["toe_out_factor"] * hip_yaw_deg,
    )


def omega_deg(waist_deg: float, side: str, limits: Mapping[str, Any]) -> float:
    """Rule B: the waist angle as seen by one leg (waist for the right leg, minus waist for the left leg)."""
    return limits["B_waist_with_legs"]["omega_sign"][side] * waist_deg


def hip_yaw_maximum_deg(omega: float, limits: Mapping[str, Any]) -> float:
    """Rule B (M5): inactive at omega <= 0.75 (hip yaw 15.0), else the entry of the first breakpoint at or above omega (the last entry beyond)."""
    rule = limits["B_waist_with_legs"]
    if omega <= rule["inactive_at_or_below_omega"] + TOLERANCE_DEG:
        return rule["inactive_hip_yaw_limit_deg"]
    table = rule["step_breakpoints_omega_to_hip_yaw_max"]
    if not table:
        raise ValueError(f"omega {omega} is above the inactive limit but this data file has no step table (frozen waist: omega is always 0)")
    for breakpoint_omega, yaw_max in table:
        if omega <= breakpoint_omega + TOLERANCE_DEG:
            return yaw_max
    return table[-1][1]


def ankle_pitch_upper_limit_deg(knee_deg: float, limits: Mapping[str, Any]) -> float:
    """Rule C: upper limit of the ankle pitch for a knee angle."""
    return interpolate(knee_deg, limits["C_knee_ankle"]["points_knee_to_ankle_max"])


def knee_in_table_range(knee_deg: float, limits: Mapping[str, Any]) -> bool:
    """Rule C (M4): True when the knee lies in the range 0..60 in which the table and the lower ankle limit were measured (closed)."""
    low, high = limits["C_knee_ankle"]["knee_range_deg"]
    return low - TOLERANCE_DEG <= knee_deg <= high + TOLERANCE_DEG


def _others_at_rest(pose: Mapping[str, float], joint: str, limits: Mapping[str, Any]) -> bool:
    """True when every joint of the pose except ``joint`` is at its rest value (0, shoulder roll and elbow at the rest pose of the data file)."""
    rest = limits["rest_pose_deg"]
    for name, value in pose.items():
        if name == joint:
            continue
        if name.endswith("_shoulder_roll_joint"):
            target = rest["shoulder_roll"]
        elif name.endswith("_elbow_joint"):
            target = rest["elbow"]
        else:
            target = 0.0
        if abs(value - target) > TOLERANCE_DEG:
            return False
    return True


def _intersection(boxes: list[Mapping[str, Any]], joint: str) -> tuple[float, float]:
    """The overlap of one joint's range over several boxes (the stricter reading at a shared edge)."""
    return (
        max(box["ranges"][joint][0] for box in boxes),
        min(box["ranges"][joint][1] for box in boxes),
    )


def boxes_containing(
    boxes: list[Mapping[str, Any]], joint: str, value: float
) -> list[Mapping[str, Any]]:
    """Every box whose closed range of the selecting joint contains the value."""
    return [
        box
        for box in boxes
        if box["ranges"][joint][0] - TOLERANCE_DEG
        <= value
        <= box["ranges"][joint][1] + TOLERANCE_DEG
    ]


def _to_degrees(pose: Mapping[str, float], unit: str) -> dict[str, float]:
    if unit == "deg":
        return {name: float(value) for name, value in pose.items()}
    if unit == "rad":
        return {name: math.degrees(float(value)) for name, value in pose.items()}
    raise ValueError(f"unit must be 'rad' or 'deg', not {unit!r}")


def _leg_violations(
    pose: Mapping[str, float], limits: Mapping[str, Any]
) -> list[Violation]:
    found: list[Violation] = []
    rule_b = limits["B_waist_with_legs"]
    frozen_waist = rule_b.get("frozen_waist_deg")
    waist = frozen_waist if frozen_waist is not None else pose.get(WAIST_JOINT)  # a frozen waist is no pose entry: rule B reads its frozen value
    if "waist_range_deg" in rule_b:
        low, high = rule_b["waist_range_deg"]
        waist_tolerance = rule_b["waist_range_tolerance_deg"]
        waist_inside = waist is None or low - waist_tolerance <= waist <= high + waist_tolerance
    else:
        waist_inside = True  # frozen waist: the old range check cannot fail
    if not waist_inside:
        found.append(
            Violation(
                RULE_B,
                WAIST_JOINT,
                waist,
                high if waist > high else low,
                "waist outside the range in which the rule was measured",
            )
        )
    for side in SIDES:
        yaw = pose.get(f"{side}_hip_yaw_joint")
        roll = pose.get(f"{side}_hip_roll_joint")
        if yaw is not None and roll is not None:
            needed = hip_roll_minimum_deg(yaw, limits)
            if roll < needed - TOLERANCE_DEG:
                found.append(
                    Violation(
                        RULE_A,
                        f"{side}_hip_roll_joint",
                        roll,
                        needed,
                        f"hip roll must be at least {needed:.3f} deg at hip yaw {yaw:.3f} deg",
                    )
                )
        if yaw is not None and waist is not None and waist_inside:  # M3/A2: outside the waist range no table is read
            omega = omega_deg(waist, side, limits)
            top = hip_yaw_maximum_deg(omega, limits)
            if yaw > top + TOLERANCE_DEG:
                found.append(
                    Violation(
                        RULE_B,
                        f"{side}_hip_yaw_joint",
                        yaw,
                        top,
                        f"hip yaw may be at most {top:.3f} deg at omega {omega:.3f} deg (waist {waist:.3f})",
                    )
                )
        knee_joint = f"{side}_knee_joint"
        knee = pose.get(knee_joint)
        ankle = pose.get(f"{side}_ankle_pitch_joint")
        if knee is not None and not knee_in_table_range(knee, limits) and not _others_at_rest(pose, knee_joint, limits):
            low_knee, high_knee = limits["C_knee_ankle"]["knee_range_deg"]
            found.append(
                Violation(
                    RULE_C,
                    knee_joint,
                    knee,
                    high_knee if knee > high_knee else low_knee,
                    "knee outside 0..60 is outside the validated combined space: allowed only when every other joint is at rest (M4)",
                )
            )
        if knee is not None and ankle is not None and knee_in_table_range(knee, limits):
            upper = ankle_pitch_upper_limit_deg(knee, limits)
            lower = limits["C_knee_ankle"]["ankle_pitch_lower_limit_deg"]
            if ankle > upper + TOLERANCE_DEG:
                found.append(
                    Violation(
                        RULE_C,
                        f"{side}_ankle_pitch_joint",
                        ankle,
                        upper,
                        f"ankle pitch may be at most {upper:.3f} deg at knee {knee:.3f} deg",
                    )
                )
            elif ankle < lower - TOLERANCE_DEG:
                found.append(
                    Violation(
                        RULE_C,
                        f"{side}_ankle_pitch_joint",
                        ankle,
                        lower,
                        f"ankle pitch may be at least {lower:.3f} deg for a knee in 0..60 (M4), knee {knee:.3f} deg",
                    )
                )
    return found


def _arm_violations(
    pose: Mapping[str, float], limits: Mapping[str, Any]
) -> list[Violation]:
    found: list[Violation] = []
    rest = limits["rest_pose_deg"]
    boxes = limits["D_arm"]["boxes"]
    for side in SIDES:
        roll_joint = f"{side}_shoulder_roll_joint"
        if roll_joint not in pose:
            continue
        roll = pose[roll_joint] - rest["shoulder_roll"]
        chosen = boxes_containing(boxes, "shoulder_roll", roll)
        if not chosen:
            found.append(
                Violation(
                    RULE_D,
                    roll_joint,
                    roll,
                    boxes[0]["ranges"]["shoulder_roll"][0],
                    "shoulder roll offset outside every measured box",
                )
            )
            continue
        joints = {
            "shoulder_pitch": f"{side}_shoulder_pitch_joint",
            "arm_yaw": f"{side}_shoulder_yaw_joint",
            "elbow": f"{side}_elbow_joint",
        }
        for name, joint in joints.items():
            if joint not in pose or name not in chosen[0]["ranges"]:  # no arm_yaw range in the boxes of a frozen-yaw data file
                continue
            value = pose[joint] - (rest["elbow"] if name == "elbow" else 0.0)
            low, high = _intersection(chosen, name)
            if value < low - TOLERANCE_DEG:
                found.append(
                    Violation(
                        RULE_D,
                        joint,
                        value,
                        low,
                        f"{name} lower limit at shoulder roll offset {roll:.3f} deg",
                    )
                )
            elif value > high + TOLERANCE_DEG:
                found.append(
                    Violation(
                        RULE_D,
                        joint,
                        value,
                        high,
                        f"{name} upper limit at shoulder roll offset {roll:.3f} deg",
                    )
                )
    return found


def _neck_violations(
    pose: Mapping[str, float], limits: Mapping[str, Any]
) -> list[Violation]:
    found: list[Violation] = []
    rule = limits["E_neck"]
    # M3: the waist range is a precondition of the neck boxes; a waist outside it is reported once, by _leg_violations, as rule B
    yaw, pitch = pose.get(NECK_YAW_JOINT), pose.get(NECK_PITCH_JOINT)
    if yaw is None or pitch is None:
        return found
    # the neck boxes are NESTED by design (neck_nod inside neck_turn): the tightest box that contains the yaw applies, except that the inner
    # box's yaw edge is OPEN (M2): at |yaw| = 45 exactly (and within the tolerance of it) only the outer box neck_turn applies (strict side)
    chosen = boxes_containing(rule["boxes"], "neck_yaw", yaw)
    if rule["inner_box_yaw_edge_is_open"]:
        chosen = [
            box
            for box in chosen
            if box["name"] != rule["inner_box"]
            or box["ranges"]["neck_yaw"][0] + TOLERANCE_DEG < yaw < box["ranges"]["neck_yaw"][1] - TOLERANCE_DEG
        ]
    if chosen:
        chosen = [
            min(
                chosen,
                key=lambda box: (
                    box["ranges"]["neck_yaw"][1] - box["ranges"]["neck_yaw"][0]
                ),
            )
        ]
    if not chosen:
        found.append(
            Violation(
                RULE_E,
                NECK_YAW_JOINT,
                yaw,
                max(abs(b["ranges"]["neck_yaw"][1]) for b in rule["boxes"]),
                "neck yaw outside every measured box",
            )
        )
        return found
    low, high = _intersection(chosen, "neck_pitch")
    if pitch < low - TOLERANCE_DEG:
        found.append(
            Violation(
                RULE_E,
                NECK_PITCH_JOINT,
                pitch,
                low,
                f"nod lower limit at neck yaw {yaw:.3f} deg",
            )
        )
    elif pitch > high + TOLERANCE_DEG:
        found.append(
            Violation(
                RULE_E,
                NECK_PITCH_JOINT,
                pitch,
                high,
                f"nod upper limit at neck yaw {yaw:.3f} deg",
            )
        )
    return found


def _frozen_violations(
    pose: Mapping[str, float], limits: Mapping[str, Any]
) -> list[Violation]:
    """A frozen joint that appears in the pose with another value than its frozen one (empty for a 25-joint data file)."""
    return [
        Violation(RULE_FROZEN, joint, pose[joint], value, "this joint is frozen in the 20-joint edition")
        for joint, value in limits.get("frozen_joints_deg", {}).items()
        if joint in pose and abs(pose[joint] - value) > TOLERANCE_DEG
    ]


def check_pose(
    pose: Mapping[str, float],
    limits: Mapping[str, Any] | None = None,
    unit: str = "rad",
) -> list[Violation]:
    """Return every coupled limit that a pose breaks (an empty list means none is broken).

    pose maps model joint names to angles (all 25 joints, or the 20 joints of the v8 model, or fewer: a rule is skipped when one of its joints is missing).
    """
    table = limits if limits is not None else load_limits()
    pose_deg = _to_degrees(pose, unit)
    return (
        _frozen_violations(pose_deg, table)
        + _leg_violations(pose_deg, table)
        + _arm_violations(pose_deg, table)
        + _neck_violations(pose_deg, table)
    )


def violates(
    pose: Mapping[str, float],
    limits: Mapping[str, Any] | None = None,
    unit: str = "rad",
) -> bool:
    """Return True when the pose breaks at least one coupled limit."""
    return bool(check_pose(pose, limits, unit))


def clip_leg_targets(
    hip_yaw_deg: Any,
    hip_roll_deg: Any,
    knee_deg: Any,
    ankle_pitch_deg: Any,
    omega_deg_values: Any,
    limits: Mapping[str, Any],
    xp: Any,
) -> tuple[Any, Any, Any]:
    """Return (hip yaw, hip roll, ankle pitch) in degrees moved just far enough to obey rules B, A and C.

    Works with floats, numpy arrays or jax arrays (xp is numpy or jax.numpy). omega_deg_values is the rule-B waist angle seen
    by each leg (same shape as the hip yaw: waist for the right leg, minus waist for the left). Order: the hip yaw is lowered
    to the step table (B); the hip roll is then raised to the minimum for the new yaw (A); the ankle pitch is lowered to the
    knee-dependent limit (C). A projection for commands, not a proof that the pose is free.
    """
    waist_rule = limits["B_waist_with_legs"]
    table = waist_rule["step_breakpoints_omega_to_hip_yaw_max"]
    if table:
        breakpoints = xp.asarray([row[0] for row in table])
        yaw_maxima = xp.asarray([row[1] for row in table])
        index = xp.minimum(
            xp.searchsorted(breakpoints, omega_deg_values - TOLERANCE_DEG, side="left"),
            len(table) - 1,
        )
        # M5: at omega <= 0.75 (also negative) the rule is inactive and the hip yaw limit is 15.0; above it the step table applies
        inactive = omega_deg_values <= waist_rule["inactive_at_or_below_omega"] + TOLERANCE_DEG
        yaw_limit = xp.where(inactive, waist_rule["inactive_hip_yaw_limit_deg"], yaw_maxima[index])
    else:
        # frozen waist (v8): omega is always 0, the rule is always inactive and its whole content is the constant limit
        yaw_limit = waist_rule["inactive_hip_yaw_limit_deg"] + 0.0 * omega_deg_values
    yaw = xp.minimum(hip_yaw_deg, yaw_limit)
    rule = limits["A_between_legs"]
    needed = xp.maximum(
        xp.maximum(rule["hip_roll_minimum_floor"], -yaw), rule["toe_out_factor"] * yaw
    )
    points = limits["C_knee_ankle"]["points_knee_to_ankle_max"]
    upper = xp.interp(
        knee_deg, xp.asarray([p[0] for p in points]), xp.asarray([p[1] for p in points])
    )
    # M4: for a knee in 0..60 the ankle pitch is also at least the lower limit (-24); a knee outside 0..60 is not projected (see check_pose)
    knee_low, knee_high = limits["C_knee_ankle"]["knee_range_deg"]
    in_range = (knee_deg >= knee_low - TOLERANCE_DEG) & (knee_deg <= knee_high + TOLERANCE_DEG)
    lowered = xp.minimum(ankle_pitch_deg, upper)
    ankle = xp.where(in_range, xp.maximum(lowered, limits["C_knee_ankle"]["ankle_pitch_lower_limit_deg"]), lowered)
    return yaw, xp.maximum(hip_roll_deg, needed), ankle
