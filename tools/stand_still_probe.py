"""Measure how the policy behaves when it is commanded to stand still (zero walking command, upper body idle).

Reports how often each sole touches the floor and how many touchdowns happen. A policy that really stands has both soles on
the floor all the time and no touchdowns; the Kobi policy marches in place (see README, "알려진 문제").

Run from the repository root:
    uv run --no-project --python 3.12 --with onnxruntime --with mujoco==3.14.0 --with numpy python -I tools/stand_still_probe.py \
        [--policy policy/kobi_v2_final_151M.onnx] [--seconds 10] [--settle-seconds 2]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kobi_env_mirror as mirror  # noqa: E402
import record_policy  # noqa: E402

DEFAULT_POLICY = HERE.parent / "policy" / "kobi_v2_final_151M.onnx"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--seconds", type=float, default=10.0, help="measured time after the settling time")
    parser.add_argument("--settle-seconds", type=float, default=2.0, help="time before the measurement starts")
    parser.add_argument("--noise-level", type=float, default=0.0, help="1.0 = training observation noise")
    args = parser.parse_args()

    dt = mirror.EnvConfig().ctrl_dt
    settle_steps, measure_steps = int(round(args.settle_seconds / dt)), int(round(args.seconds / dt))
    env = mirror.KobiPlaybackEnv(noise_level=args.noise_level)
    policy = record_policy.load_policy(args.policy)
    walk, upper = np.zeros(mirror.NUM_MOVE), np.zeros(mirror.NUM_UPPER)
    env.reset(walk, upper)

    contacts = []
    for k in range(settle_steps + measure_steps):
        env.step(policy(env.obs), walk, upper)
        if k >= settle_steps:
            contacts.append(env.feet_contacts().copy())
    c = np.array(contacts)
    touchdowns = [int(((c[1:, i] == 1) & (c[:-1, i] == 0)).sum()) for i in (0, 1)]
    print(f"measured {len(c)} control steps ({len(c) * dt:.1f} s) of a stand-still command")
    print(f"fraction of steps with the sole on the floor: left {c[:, 0].mean():.2f}, right {c[:, 1].mean():.2f}")
    print(f"both soles down {np.mean(c.sum(axis=1) == 2):.2f}, one sole down {np.mean(c.sum(axis=1) == 1):.2f}, none {np.mean(c.sum(axis=1) == 0):.2f}")
    print(f"touchdowns (floor contact starts): left {touchdowns[0]}, right {touchdowns[1]}")


if __name__ == "__main__":
    main()
