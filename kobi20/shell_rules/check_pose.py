"""Check a Kobi-20 pose against the coupled shell limits (rules A to G) without MuJoCo; only the Python standard library is needed.

Per-joint ranges (the limits in the URDF and MJCF) are NOT enough: two shell parts can collide although every joint is inside its own range
(for example a raised arm against the head shell, a hip yaw against the other leg). The coupled rules in shell_coupled_limits_v8.json
describe those combined limits. The model and the URDF do not enforce them; this script only judges a pose.

    python3 -I -B kobi20/shell_rules/check_pose.py --pose left_shoulder_roll_joint=40 left_shoulder_pitch_joint=-90 --unit deg
    python3 -I -B kobi20/shell_rules/check_pose.py --pose-json my_pose.json --unit rad
    python3 -I -B kobi20/shell_rules/check_pose.py --base zero

A pose is a dictionary joint name -> absolute joint angle, the same angles as the MJCF qpos (not offsets). Joints that you do not give are
taken from --base (default: the home pose of the MJCF; "zero" = all angles 0; "none" = nothing, and a rule whose joints are missing is then
skipped). Exit code 0 = no rule broken, 1 = at least one rule broken, 2 = bad input.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # python -I drops the script folder from sys.path

import shell_coupled_checker as checker  # noqa: E402

DEFAULT_LIMITS = HERE / "shell_coupled_limits_v8.json"
DEFAULT_MJCF = HERE.parent / "mujoco" / "kobi20_v8.xml"
HOME_KEY_NAME = "home"
ROOT_QPOS_COUNT = 7  # free joint: x y z + quaternion


def mjcf_joint_names(mjcf: Path) -> list[str]:
    """Names of the hinge joints in MJCF document order (= qpos order after the free joint)."""
    root = ET.parse(mjcf).getroot()
    return [j.get("name") for j in root.iter("joint") if j.get("name")]


def mjcf_home_pose(mjcf: Path) -> dict[str, float]:
    """Joint name -> angle (rad) of the MJCF keyframe 'home', read as text (no MuJoCo)."""
    text = mjcf.read_text(encoding="utf-8")
    match = re.search(rf'<key name="{HOME_KEY_NAME}" qpos="([^"]*)"', text)
    if match is None:
        raise ValueError(f"keyframe {HOME_KEY_NAME!r} not found in {mjcf.name}")
    values = [float(v) for v in match.group(1).split()][ROOT_QPOS_COUNT:]
    names = mjcf_joint_names(mjcf)
    if len(values) != len(names):
        raise ValueError(f"home qpos has {len(values)} joint values but the MJCF has {len(names)} hinge joints")
    return dict(zip(names, values, strict=True))


def parse_assignments(items: list[str]) -> dict[str, float]:
    """Turn ['name=1.5', ...] into a dictionary."""
    result: dict[str, float] = {}
    for item in items:
        name, separator, value = item.partition("=")
        if not separator:
            raise ValueError(f"{item!r} is not NAME=VALUE")
        result[name.strip()] = float(value)
    return result


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pose", nargs="*", default=[], metavar="NAME=VALUE", help="joint angles to set")
    parser.add_argument("--pose-json", type=Path, default=None, help="JSON file {joint name: angle}")
    parser.add_argument("--unit", choices=("rad", "deg"), default="rad", help="unit of --pose and --pose-json (default rad)")
    parser.add_argument("--base", choices=("home", "zero", "none"), default="home")
    parser.add_argument("--limits", type=Path, default=DEFAULT_LIMITS)
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF, help="only read for the joint names and the home pose")
    args = parser.parse_args(argv)

    try:
        home = mjcf_home_pose(args.mjcf)
        given = parse_assignments(args.pose)
        if args.pose_json is not None:
            given.update({k: float(v) for k, v in json.loads(args.pose_json.read_text(encoding="utf-8")).items()})
    except (ValueError, OSError, json.JSONDecodeError) as error:
        print(f"input error: {error}", file=sys.stderr)
        return 2
    unknown = sorted(set(given) - set(home))
    if unknown:
        print(f"input error: unknown joint names {unknown}; the 20 joints are {sorted(home)}", file=sys.stderr)
        return 2

    base = {"home": dict(home), "zero": {name: 0.0 for name in home}, "none": {}}[args.base]
    if args.unit == "deg":  # the base is in radians, the given values in degrees: convert the base to degrees as well
        base = {name: value * 180.0 / 3.141592653589793 for name, value in base.items()}
    pose = {**base, **given}
    limits = checker.load_limits(args.limits)
    violations = checker.check_pose(pose, limits, unit=args.unit)
    print(f"checked {len(pose)} joints (base: {args.base}, unit: {args.unit}); rules A to G of {args.limits.name}")
    if not violations:
        print("OK: no coupled limit is broken")
        return 0
    for violation in violations:
        print(f"VIOLATION {violation}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
