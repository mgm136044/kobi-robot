"""Record a trained Kobi ONNX policy running in MuJoCo (C) into the motion .npz used by animate_motion.py.

The observation, the action-to-motor-target mapping, the rate limit, the torque limits and the imitation-phase counter are
NOT written here: they live in kobi_env_mirror.py, which copies the training environment (joystick.py + base.py of our fork) step by
step (see its docstring).
This file only owns the schedule (schedule.py: walk segments in command slots 0..2, gesture targets in slots 3..6), the
rollout loop and the recording.

Run (ephemeral environment, nothing installed):
    uv run --no-project --python 3.12 --with onnxruntime --with mujoco==3.14.0 --with numpy python -I tools/record_policy.py \
        --policy policy/kobi_v2_final_151M.onnx [--out out/policy_motion.npz]
Without a policy, --dry-run drives the same loop with zero actions (legs hold home, neck and shoulders follow the gestures).
Playback differs from training in: no domain randomisation, no pushes, action delay 0, clean observation (--noise-level 1.0
restores the training noise), deterministic reset from the home keyframe (training randomises yaw, xy and joint angles).
MuJoCo C 3.14 plays back a policy trained on MJX 3.3.4, so small contact differences are expected.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kobi_env_mirror as mirror  # noqa: E402
import motion_format  # noqa: E402
import schedule as sched  # noqa: E402

REPO_ROOT = HERE.parent
OUT_NPZ = REPO_ROOT / "out" / "policy_motion.npz"
POLICY_INPUT_SIZE = mirror.OBS_SIZE
POLICY_OUTPUT_SIZE = mirror.NUM_ACT
EXPLODE_ABS_LIMIT = 1.0e3        # an observation entry this large means the simulation blew up

Policy = Callable[[np.ndarray], np.ndarray]


def load_policy(onnx_path: Path) -> Policy:
    """observation (101,) -> action (14,) with onnxruntime (the export is tanh(mean) with the observation normaliser inside)."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp, out = session.get_inputs()[0], session.get_outputs()[0]
    if list(inp.shape)[-1] != POLICY_INPUT_SIZE or list(out.shape)[-1] != POLICY_OUTPUT_SIZE:
        raise ValueError(f"{onnx_path.name}: input {inp.shape}, output {out.shape}; expected (*, {POLICY_INPUT_SIZE}) and (*, {POLICY_OUTPUT_SIZE})")

    def run(obs: np.ndarray) -> np.ndarray:
        return np.asarray(session.run(None, {inp.name: obs[None].astype(np.float32)})[0])[0].astype(np.float64)

    return run


def zero_policy(_: np.ndarray) -> np.ndarray:
    return np.zeros(POLICY_OUTPUT_SIZE)


def record(env: mirror.KobiPlaybackEnv, expanded: sched.Expanded, policy: Policy) -> dict[str, np.ndarray]:
    """Run the schedule. Frame k is the state at the start of control step k (plus the state after a terminating step)."""
    model, data = env.model, env.data
    steps = len(expanded.commands)
    env.reset(expanded.commands[0], expanded.upper_targets[0])
    xpos, xquat, obs_log, act_log, target_log = [], [], [], [], []
    fell_at = -1.0
    exploded = False
    for k in range(steps):
        xpos.append(data.xpos.copy())
        xquat.append(data.xquat.copy())
        obs = env.obs
        if not np.all(np.isfinite(obs)) or np.abs(obs).max() > EXPLODE_ABS_LIMIT:
            exploded = True
            print(f"WARNING: observation blew up at step {k} (max |obs| {np.nanmax(np.abs(obs)):.3g})", flush=True)
            break
        action = policy(obs)
        nxt = min(k + 1, steps - 1)
        result = env.step(action, expanded.commands[nxt], expanded.upper_targets[nxt])
        obs_log.append(obs.astype(np.float32))
        act_log.append(action)
        target_log.append(result.motor_targets)
        if result.done:
            fell_at = (k + 1) * env.cfg.ctrl_dt
            xpos.append(data.xpos.copy())
            xquat.append(data.xquat.copy())
            print(f"robot fell (tilt > 60 deg) during step {k} at t = {fell_at:.2f} s; recording stops", flush=True)
            break
    return {"xpos": np.array(xpos), "xquat": np.array(xquat), "obs": np.array(obs_log), "actions": np.array(act_log),
            "motor_targets": np.array(target_log), "fell_at_s": np.float64(fell_at), "exploded": np.bool_(exploded)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--policy", type=Path, help="trained ONNX policy (needs onnxruntime)")
    parser.add_argument("--dry-run", action="store_true", help="no policy: zero actions through the same env mirror")
    parser.add_argument("--xml", type=Path, default=mirror.BACKLASH_XML)
    parser.add_argument("--out", type=Path, default=OUT_NPZ)
    parser.add_argument("--noise-level", type=float, default=0.0, help="1.0 = training observation noise")
    parser.add_argument("--action-delay", type=int, default=0, choices=(0, 1, 2), help="steps; training draws 0..2, playback 0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--schedule", choices=sorted(sched.SCHEDULES), default="preview")
    args = parser.parse_args()
    if not args.dry_run and args.policy is None:
        parser.error("give --policy x.onnx, or --dry-run")
    if abs(mirror.EnvConfig().ctrl_dt * motion_format.FPS - 1.0) > 1e-9:
        raise RuntimeError("control rate and motion_format.FPS disagree")

    env = mirror.KobiPlaybackEnv(args.xml, noise_level=args.noise_level, action_delay_steps=args.action_delay, seed=args.seed)
    names = [mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(env.model.nbody)]
    expanded = sched.expand(sched.SCHEDULES[args.schedule], motion_format.FPS)
    policy = zero_policy if args.dry_run else load_policy(args.policy)
    result = record(env, expanded, policy)
    n = len(result["xpos"])
    labels = (expanded.labels + [expanded.labels[-1]])[:n]
    source = "mujoco_dry_run_zero_action" if args.dry_run else motion_format.SOURCE_POLICY
    motion_format.save_motion(
        args.out, names, result["xpos"], result["xquat"], motion_format.FPS, labels, source=source, kinematic=False,
        commands=expanded.commands[:n], upper_targets=expanded.upper_targets[:n], obs=result["obs"], actions=result["actions"],
        motor_targets=result["motor_targets"], fell_at_s=result["fell_at_s"],
    )
    a = result["actions"]
    print(f"recorded {n} frames ({n / motion_format.FPS:.1f} s) of {len(expanded.commands)} scheduled from {args.xml.name}; "
          f"actions in [{a.min():.2f}, {a.max():.2f}], base z in [{result['xpos'][:, 1, 2].min():.3f}, {result['xpos'][:, 1, 2].max():.3f}] m; "
          f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
