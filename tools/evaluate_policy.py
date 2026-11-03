"""Evaluate a trained Kobi ONNX policy in MuJoCo C through the env mirror (backlash scene) and write out/eval_<tag>.json.

    uv run --no-project --python 3.12 --with onnxruntime --with mujoco==3.14.0 --with numpy python -I evaluate_policy.py \
        --policy policy/<name>.onnx [--tag name] [--quick] [--workers 8]

Measured (all with the training observation noise, --noise-level 1.0, action delay 0, deterministic home reset; seeds change
the observation noise and the push timing):
  tracking  walking/turning/standing commands, 1 s settling then 10 s measured: achieved mean body-frame velocity, falls,
            foot slip (horizontal speed of a foot site while its sole touches the floor), trunk tilt;
  push      a sudden base velocity change (like the env's push) of 0.25/0.5/0.75/1.0 m/s in 8 directions relative to the
            robot heading, standing and walking 0.1 m/s; success = no fall (tilt > 60 deg) within 3 s;
  gestures  every gesture of the library once, standing and walking 0.1 m/s: |q - target| of the 4 upper joints, falls, tilt;
  covers    fraction of control steps in which c_cover_L/R touch the floor, and cover/cover or cover/opposite-sole contact;
  torques   max |actuator force| per joint over every physics substep versus the 1.91 / 0.20 N*m limits.
Contacts and sensors are read once per control step, as in the env. Representative episodes are saved as motion .npz files
(--no-motions to skip) that animate_motion.py can render.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kobi_env_mirror as mirror  # noqa: E402
import motion_format  # noqa: E402
import record_policy  # noqa: E402
import schedule as sched  # noqa: E402

FPS = motion_format.FPS
DT = 1.0 / FPS
SETTLE_S, MEASURE_S = 1.0, 10.0
TRACKING_COMMANDS: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("forward 0.05", (0.05, 0.0, 0.0)), ("forward 0.10", (0.10, 0.0, 0.0)), ("forward 0.15", (0.15, 0.0, 0.0)),
    ("backward -0.10", (-0.10, 0.0, 0.0)), ("lateral +0.074", (0.0, 0.074, 0.0)), ("lateral -0.074", (0.0, -0.074, 0.0)),
    ("turn +0.5", (0.0, 0.0, 0.5)), ("turn -0.5", (0.0, 0.0, -0.5)), ("turn +1.0", (0.0, 0.0, 1.0)),
    ("turn -1.0", (0.0, 0.0, -1.0)), ("stand still", (0.0, 0.0, 0.0)),
)
PUSH_MAGNITUDES = (0.25, 0.5, 0.75, 1.0)               # m/s added to the base velocity
PUSH_DIRECTIONS_DEG = tuple(range(0, 360, 45))          # relative to the robot heading, 0 = pushed forward
PUSH_GROUPS = {"front": (0, 45, 315), "side": (90, 270), "back": (135, 180, 225)}
PUSH_MODES = (("standing", 0.0), ("walking 0.1", 0.1))
PUSH_SETTLE_S, PUSH_OBSERVE_S, PUSH_PHASE_JITTER_STEPS = 2.0, 3.0, 27
GESTURE_SETTLE_S, GESTURE_TAIL_S = 1.0, 0.5
GESTURE_MODES = (("standing", 0.0), ("walking 0.1", 0.1))
SLIP_MIN_CONTACT_STEPS = 5
FORCE_LIMIT_TOLERANCE = 1e-6
COVER_KEYS = ("cover_L_floor", "cover_R_floor", "cover_L_cover_R", "cover_L_sole_R", "cover_R_sole_L")
SOLE_KEYS = ("sole_L_floor", "sole_R_floor")
UPPER_NAMES = mirror.UPPER_JOINTS

ENV: mirror.KobiPlaybackEnv | None = None
POLICY: record_policy.Policy | None = None
PROBE: "Probe | None" = None


class Probe:
    """Per-substep torque tracking and per-control-step contact / foot-slip readings."""

    def __init__(self, env: mirror.KobiPlaybackEnv) -> None:
        m = env.model
        gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
        cl, cr, sl, sr, fl = gid("c_cover_L"), gid("c_cover_R"), env.feet_geoms[0], env.feet_geoms[1], env.floor_geom
        self.pairs = {frozenset((cl, fl)): "cover_L_floor", frozenset((cr, fl)): "cover_R_floor",
                      frozenset((cl, cr)): "cover_L_cover_R", frozenset((cl, sr)): "cover_L_sole_R",
                      frozenset((cr, sl)): "cover_R_sole_L", frozenset((sl, fl)): "sole_L_floor",
                      frozenset((sr, fl)): "sole_R_floor"}
        self.limit = m.actuator_forcerange[:, 1].copy()
        sadr = lambda n: int(m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, n)])
        self.foot_vel_adr = (sadr("left_foot_global_linvel"), sadr("right_foot_global_linvel"))
        self.reset()

    def reset(self) -> None:
        self.force_max = np.zeros(len(self.limit))
        self.at_limit = np.zeros(len(self.limit))
        self.substeps = 0

    def hook(self, data: mujoco.MjData) -> None:
        f = np.abs(data.actuator_force)
        self.force_max = np.maximum(self.force_max, f)
        self.at_limit += f >= self.limit - FORCE_LIMIT_TOLERANCE
        self.substeps += 1

    def scan(self, data: mujoco.MjData) -> tuple[dict[str, bool], dict[str, float]]:
        """Which monitored geom pairs touch (dist < 0) after this control step, and the deepest penetration per pair [m]."""
        touching = {k: False for k in COVER_KEYS + SOLE_KEYS}
        depth = {k: 0.0 for k in COVER_KEYS}
        for i in range(data.ncon):
            c = data.contact[i]
            if c.dist >= 0.0:
                continue
            key = self.pairs.get(frozenset((int(c.geom1), int(c.geom2))))
            if key is None:
                continue
            touching[key] = True
            if key in depth:
                depth[key] = min(depth[key], float(c.dist))
        return touching, depth

    def foot_speeds(self, data: mujoco.MjData) -> np.ndarray:
        return np.array([np.linalg.norm(data.sensordata[a:a + 2]) for a in self.foot_vel_adr])


def init_worker(policy_path: str, xml: str, noise_level: float, action_delay: int) -> None:
    global ENV, POLICY, PROBE
    ENV = mirror.KobiPlaybackEnv(Path(xml), noise_level=noise_level, action_delay_steps=action_delay)
    POLICY = record_policy.load_policy(Path(policy_path))
    PROBE = Probe(ENV)
    ENV.substep_hook = PROBE.hook


def yaw_of(env: mirror.KobiPlaybackEnv) -> float:
    w, x, y, z = env.data.qpos[env.free_qpos + 3:env.free_qpos + 7]
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def tilt_deg(env: mirror.KobiPlaybackEnv) -> float:
    return math.degrees(math.acos(float(np.clip(env.upright_z(), -1.0, 1.0))))


def build_schedule(job: dict[str, Any]) -> sched.Expanded:
    gestures = ((sched.GestureEvent(job["gesture_start_s"], job["gesture"]),) if job.get("gesture") else ())
    walk = sched.WalkSegment(job["total_s"], *job["command"])
    return sched.expand(sched.Schedule(walk=(walk,), gestures=gestures), FPS)


def run_episode(job: dict[str, Any]) -> dict[str, Any]:
    env, probe, policy = ENV, PROBE, POLICY
    assert env is not None and probe is not None and policy is not None
    exp = build_schedule(job)
    steps = len(exp.commands)
    env.rng = np.random.default_rng(job["seed"])
    env.reset(exp.commands[0], exp.upper_targets[0])
    probe.reset()
    jitter = int(np.random.default_rng(job["seed"] + 1).integers(0, PUSH_PHASE_JITTER_STEPS))
    push = job.get("push")
    push_step = int(round(PUSH_SETTLE_S * FPS)) + jitter if push else -1
    tilt, pos, yaw, slip, q_up, tgt_up = [], [], [], [], [], []
    touching: list[dict[str, bool]] = []
    depth: list[dict[str, float]] = []
    pos.append(env.data.qpos[env.free_qpos:env.free_qpos + 2].copy())
    yaw.append(yaw_of(env))
    xpos, xquat = ([env.data.xpos.copy()], [env.data.xquat.copy()]) if job.get("record") else (None, None)
    fell_step = -1
    for k in range(steps):
        if k == push_step:
            heading = yaw_of(env) + math.radians(push["theta_deg"])
            env.data.qvel[0:2] += push["magnitude"] * np.array([math.cos(heading), math.sin(heading)])
        target = env.command[mirror.NUM_MOVE:].copy()
        nxt = min(k + 1, steps - 1)
        result = env.step(policy(env.obs), exp.commands[nxt], exp.upper_targets[nxt])
        d = env.data
        t, dep = probe.scan(d)
        touching.append(t)
        depth.append(dep)
        slip.append(probe.foot_speeds(d))
        tilt.append(tilt_deg(env))
        pos.append(d.qpos[env.free_qpos:env.free_qpos + 2].copy())
        yaw.append(yaw_of(env))
        q_up.append(d.qpos[env.act_qpos[mirror.NUM_LEGS:]].copy())
        tgt_up.append(target)
        if xpos is not None:
            xpos.append(d.xpos.copy())
            xquat.append(d.xquat.copy())
        if result.done:
            fell_step = k
            break
    out = summarize(job, exp, np.array(pos), np.unwrap(yaw), np.array(tilt), np.array(slip), touching, depth,
                    np.array(q_up), np.array(tgt_up), fell_step, push_step, probe)
    if xpos is not None:
        names = [mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(env.model.nbody)]
        n = len(xpos)
        labels = [f"eval {job['label']} | {exp.labels[min(i, steps - 1)]}" for i in range(n)]
        motion_format.save_motion(Path(job["record"]), names, np.array(xpos), np.array(xquat), FPS, labels,
                                  source=motion_format.SOURCE_POLICY, kinematic=False)
        out["motion"] = job["record"]
    return out


def summarize(job: dict[str, Any], exp: sched.Expanded, pos: np.ndarray, yaw: np.ndarray, tilt: np.ndarray,
              slip: np.ndarray, touching: list[dict[str, bool]], depth: list[dict[str, float]], q_up: np.ndarray,
              tgt_up: np.ndarray, fell_step: int, push_step: int, probe: Probe) -> dict[str, Any]:
    n = len(tilt)
    fell = fell_step >= 0
    start = min(int(round(job.get("window_start_s", SETTLE_S) * FPS)), n)
    end = min(start + int(round(job["window_s"] * FPS)), n) if job.get("window_s") else n
    out: dict[str, Any] = {"kind": job["kind"], "label": job["label"], "mode": job.get("mode", ""), "seed": job["seed"],
                           "fell": fell, "fell_t": (fell_step + 1) * DT if fell else None, "steps": n,
                           "force_max": probe.force_max.tolist(), "force_at_limit": probe.at_limit.tolist(),
                           "substeps": probe.substeps}
    window = slice(start, end)
    out["window_complete"] = (not fell) and end - start >= (job["window_s"] * FPS - 1 if job.get("window_s") else 1)
    out["tilt_max"] = float(tilt[window].max()) if end > start else float(tilt.max())
    out["tilt_mean"] = float(tilt[window].mean()) if end > start else float(tilt.mean())
    if job["kind"] == "tracking" and end > start + 1:
        dp = pos[start + 1:end + 1] - pos[start:end]
        yw = 0.5 * (yaw[start + 1:end + 1] + yaw[start:end])
        vx = (np.cos(yw) * dp[:, 0] + np.sin(yw) * dp[:, 1]) / DT
        vy = (-np.sin(yw) * dp[:, 0] + np.cos(yw) * dp[:, 1]) / DT
        wz = np.diff(yaw[start:end + 1]) / DT
        out["velocity"] = [float(vx.mean()), float(vy.mean()), float(wz.mean())]
        contact = np.array([[t["sole_L_floor"], t["sole_R_floor"]] for t in touching[window]])
        sp = slip[window]
        out["slip"] = float((sp[contact]).mean()) if contact.sum() >= SLIP_MIN_CONTACT_STEPS else None
        out["sole_contact_fraction"] = contact.mean(axis=0).tolist()
        out["covers"] = {k: float(np.mean([t[k] for t in touching[window]])) for k in COVER_KEYS}
        out["cover_depth_mm"] = {k: float(1000 * min(d[k] for d in depth[window])) for k in COVER_KEYS}
    if job["kind"] == "push":
        out["push"] = job["push"]
        out["fell_before_push"] = bool(fell and fell_step < push_step)
        after = tilt[push_step:] if push_step < n else tilt[-1:]
        out["tilt_max_after_push"] = float(after.max())
        out["success"] = not fell
    if job["kind"] == "gesture":
        g0 = int(round(job["gesture_start_s"] * FPS))
        g1 = min(g0 + int(round(job["gesture_len_s"] * FPS)), len(q_up))
        err = np.abs(q_up[g0:g1] - tgt_up[g0:g1]) if g1 > g0 else np.zeros((1, 4))
        out["err_mean"], out["err_max"] = err.mean(axis=0).tolist(), err.max(axis=0).tolist()
        out["gesture_complete"] = g1 - g0 >= int(round(job["gesture_len_s"] * FPS)) - 1
    return out


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    return run_episode(job)


def build_jobs(args: argparse.Namespace, out_dir: Path, tag: str) -> list[dict[str, Any]]:
    import kobi_gestures

    jobs: list[dict[str, Any]] = []
    rec = lambda name: str(out_dir / f"eval_{tag}_{name}.npz") if not args.no_motions else None
    record_label = {"forward 0.15": "tracking_forward", "turn +1.0": "tracking_turn", "stand still": "tracking_stand"}
    for label, cmd in TRACKING_COMMANDS:
        for seed in range(args.seeds_track):
            jobs.append(dict(kind="tracking", label=label, command=cmd, seed=seed, total_s=SETTLE_S + MEASURE_S,
                             window_s=MEASURE_S, record=rec(record_label[label]) if label in record_label and seed == 0 else None))
    push_total = PUSH_SETTLE_S + PUSH_PHASE_JITTER_STEPS * DT + PUSH_OBSERVE_S
    for mode, speed in PUSH_MODES:
        for mag in PUSH_MAGNITUDES:
            for theta in PUSH_DIRECTIONS_DEG:
                for seed in range(args.seeds_push):
                    first = mode == "standing" and mag == 0.75 and theta == 0 and seed == 0
                    first_walk = mode != "standing" and mag == 0.5 and theta == 90 and seed == 0
                    jobs.append(dict(kind="push", label=f"{mode} push {mag}", mode=mode, command=(speed, 0.0, 0.0), seed=seed,
                                     total_s=push_total, push=dict(magnitude=mag, theta_deg=theta), window_start_s=0.0,
                                     record=rec("push_standing") if first else rec("push_walking") if first_walk else None))
    for mode, speed in GESTURE_MODES:
        for g in kobi_gestures.library():
            length = (len(kobi_gestures.sample(g)) - 1) * DT
            for seed in range(args.seeds_gesture):
                total = GESTURE_SETTLE_S + length + GESTURE_TAIL_S
                jobs.append(dict(kind="gesture", label=g.name, mode=mode, command=(speed, 0.0, 0.0), seed=seed, total_s=total,
                                 gesture=g.name, gesture_start_s=GESTURE_SETTLE_S, gesture_len_s=length, window_start_s=GESTURE_SETTLE_S,
                                 record=rec(f"gesture_{mode.split()[0]}") if g.name == "wave_right" and seed == 0 else None))
    return jobs


# ------------------------------------------------------------------ aggregation
def mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def aggregate_tracking(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    table = []
    for label, cmd in TRACKING_COMMANDS:
        r = [x for x in rows if x["label"] == label]
        ok = [x for x in r if x["window_complete"]]
        vel = np.array([x["velocity"] for x in ok]) if ok else np.zeros((0, 3))
        slips = [x["slip"] for x in ok if x.get("slip") is not None]
        table.append({
            "command": list(cmd), "label": label, "episodes": len(r), "falls": sum(x["fell"] for x in r),
            "fall_times_s": [x["fell_t"] for x in r if x["fell"]],
            "achieved_mean": vel.mean(axis=0).tolist() if len(vel) else None,
            "achieved_std": vel.std(axis=0).tolist() if len(vel) else None,
            "error_mean": (vel.mean(axis=0) - np.array(cmd)).tolist() if len(vel) else None,
            "slip_m_s": mean_or_none(slips), "tilt_max_deg": max((x["tilt_max"] for x in r), default=None),
            "tilt_mean_deg": mean_or_none([x["tilt_mean"] for x in r]),
        })
    return table


def aggregate_push(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for mode, _ in PUSH_MODES:
        r = [x for x in rows if x["mode"] == mode]
        by_mag = {str(m): mean_or_none([x["success"] for x in r if x["push"]["magnitude"] == m]) for m in PUSH_MAGNITUDES}
        by_dir = {str(t): mean_or_none([x["success"] for x in r if x["push"]["theta_deg"] == t]) for t in PUSH_DIRECTIONS_DEG}
        by_group = {g: mean_or_none([x["success"] for x in r if x["push"]["theta_deg"] in ds]) for g, ds in PUSH_GROUPS.items()}
        grid = {f"{m}": {str(t): mean_or_none([x["success"] for x in r if x["push"]["magnitude"] == m and x["push"]["theta_deg"] == t])
                         for t in PUSH_DIRECTIONS_DEG} for m in PUSH_MAGNITUDES}
        out[mode] = {"episodes": len(r), "fell_before_push": sum(x["fell_before_push"] for x in r), "success_by_magnitude": by_mag,
                     "success_by_direction_deg": by_dir, "success_by_group": by_group, "success_grid": grid}
    return out


def aggregate_gestures(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    table = []
    for mode, _ in GESTURE_MODES:
        for name in dict.fromkeys(x["label"] for x in rows):
            r = [x for x in rows if x["label"] == name and x["mode"] == mode]
            if not r:
                continue
            table.append({"gesture": name, "mode": mode, "episodes": len(r), "falls": sum(x["fell"] for x in r),
                          "err_mean_rad": np.mean([x["err_mean"] for x in r], axis=0).tolist(),
                          "err_max_rad": np.max([x["err_max"] for x in r], axis=0).tolist(),
                          "tilt_max_deg": max(x["tilt_max"] for x in r)})
    return table


def aggregate_covers(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = {"walking (forward/backward/lateral)": lambda c: c != (0, 0, 0) and c[2] == 0, "turning": lambda c: c[2] != 0,
              "standing": lambda c: c == (0, 0, 0)}
    cmds = {label: tuple(cmd) for label, cmd in TRACKING_COMMANDS}
    out: dict[str, Any] = {}
    for gname, test in groups.items():
        r = [x for x in rows if x.get("covers") and test(cmds[x["label"]])]
        if not r:
            continue
        out[gname] = {"episodes": len(r),
                      "fraction_of_steps": {k: float(np.mean([x["covers"][k] for x in r])) for k in COVER_KEYS},
                      "episodes_with_any_contact": {k: int(sum(x["covers"][k] > 0 for x in r)) for k in COVER_KEYS},
                      "deepest_penetration_mm": {k: float(min(x["cover_depth_mm"][k] for x in r)) for k in COVER_KEYS}}
    return out


def aggregate_torques(rows: list[dict[str, Any]], limits: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for kind in ("tracking", "push", "gesture"):
        r = [x for x in rows if x["kind"] == kind]
        if not r:
            continue
        fmax = np.max([x["force_max"] for x in r], axis=0)
        frac = np.sum([x["force_at_limit"] for x in r], axis=0) / max(1, sum(x["substeps"] for x in r))
        out[kind] = {n: {"max_abs_Nm": float(fmax[i]), "limit_Nm": float(limits[i]), "fraction_at_limit": float(frac[i])}
                     for i, n in enumerate(mirror.ACTUATOR_ORDER)}
    return out


def aggregate(results: list[dict[str, Any]], limits: np.ndarray) -> dict[str, Any]:
    by = lambda k: [x for x in results if x["kind"] == k]
    return {"tracking": aggregate_tracking(by("tracking")), "push": aggregate_push(by("push")),
            "gestures": aggregate_gestures(by("gesture")), "covers": aggregate_covers(by("tracking")),
            "torques": aggregate_torques(results, limits)}


# ------------------------------------------------------------------ printing
def fmt(v: float | None, spec: str = "+.3f") -> str:
    return "  n/a " if v is None else format(v, spec)


def print_report(rep: dict[str, Any]) -> None:
    print("\nTRACKING (body-frame mean over the 10 s window; falls = fallen episodes / episodes)")
    print(f"{'command':16s} {'vx':>7s} {'vy':>7s} {'wz':>7s} {'err vx':>7s} {'err vy':>7s} {'err wz':>7s} {'falls':>6s} {'slip m/s':>9s} {'tilt max/mean deg':>18s}")
    for r in rep["tracking"]:
        a, e = r["achieved_mean"], r["error_mean"]
        g = lambda v, i: fmt(v[i]) if v else "  n/a "
        print(f"{r['label']:16s} {g(a, 0):>7s} {g(a, 1):>7s} {g(a, 2):>7s} {g(e, 0):>7s} {g(e, 1):>7s} {g(e, 2):>7s} "
              f"{r['falls']:>2d}/{r['episodes']:<3d} {fmt(r['slip_m_s'], '.3f'):>9s} {fmt(r['tilt_max_deg'], '.1f'):>9s}/{fmt(r['tilt_mean_deg'], '.1f'):>7s}")
    print("\nPUSH RECOVERY (success = no fall within 3 s)")
    for mode, r in rep["push"].items():
        pct = lambda v: "  n/a" if v is None else f"{100 * v:4.0f}%"
        print(f"{mode:12s} episodes {r['episodes']}, fell before the push {r['fell_before_push']}")
        print("   by magnitude  " + "  ".join(f"{m} m/s: {pct(v)}" for m, v in r["success_by_magnitude"].items()))
        print("   by group      " + "  ".join(f"{g}: {pct(v)}" for g, v in r["success_by_group"].items()))
        print("   by direction  " + "  ".join(f"{d}deg: {pct(v)}" for d, v in r["success_by_direction_deg"].items()))
    print("\nGESTURES |q - target| mean/max over the gesture (rad): neck_yaw, neck_pitch, shoulder_left, shoulder_right")
    for r in rep["gestures"]:
        cells = " ".join(f"{m:.3f}/{x:.3f}" for m, x in zip(r["err_mean_rad"], r["err_max_rad"]))
        print(f"{r['mode']:12s} {r['gesture']:18s} {cells}  falls {r['falls']}/{r['episodes']}  tilt max {r['tilt_max_deg']:.1f}")
    print("\nFOOT COVERS (fraction of control steps; deepest penetration in mm)")
    for g, r in rep["covers"].items():
        cells = "  ".join(f"{k}: {100 * v:.1f}% ({r['deepest_penetration_mm'][k]:.1f} mm)" for k, v in r["fraction_of_steps"].items())
        print(f"{g:36s} n={r['episodes']}  {cells}")
    print("\nTORQUES max |force| N*m (limit) and fraction of substeps at the limit")
    names = list(mirror.ACTUATOR_ORDER)
    print(f"{'':10s} " + " ".join(f"{n[:9]:>9s}" for n in names))
    for kind, r in rep["torques"].items():
        print(f"{kind:10s} " + " ".join(f"{r[n]['max_abs_Nm']:9.2f}" for n in names))
        print(f"{'  at limit':10s} " + " ".join(f"{100 * r[n]['fraction_at_limit']:8.1f}%" for n in names))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--tag", default="")
    parser.add_argument("--out-dir", type=Path, default=HERE.parent / "out")
    parser.add_argument("--xml", type=Path, default=mirror.BACKLASH_XML)
    parser.add_argument("--seeds-track", type=int, default=6)
    parser.add_argument("--seeds-push", type=int, default=3)
    parser.add_argument("--seeds-gesture", type=int, default=2)
    parser.add_argument("--noise-level", type=float, default=1.0)
    parser.add_argument("--action-delay", type=int, default=0, choices=(0, 1, 2))
    parser.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 2))
    parser.add_argument("--quick", action="store_true", help="one seed per case")
    parser.add_argument("--no-motions", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.seeds_track = args.seeds_push = args.seeds_gesture = 1
    tag = args.tag or args.policy.stem
    args.out_dir.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args, args.out_dir, tag)
    print(f"{len(jobs)} episodes on {args.workers} workers (policy {args.policy.name}, noise {args.noise_level}, delay {args.action_delay})", flush=True)
    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=init_worker, initargs=(str(args.policy), str(args.xml), args.noise_level, args.action_delay)) as pool:
        results = pool.map(run_job, jobs, chunksize=4)
    runtime = time.perf_counter() - t0
    probe_env = mirror.KobiPlaybackEnv(args.xml)
    report = aggregate(results, probe_env.model.actuator_forcerange[:, 1])
    report["settings"] = {"policy": str(args.policy), "tag": tag, "noise_level": args.noise_level, "action_delay": args.action_delay,
                          "seeds": [args.seeds_track, args.seeds_push, args.seeds_gesture], "episodes": len(jobs),
                          "fall_rule": f"tilt > {mirror.FALL_TILT_DEG} deg", "runtime_s": runtime, "xml": str(args.xml)}
    report["motions"] = sorted({x["motion"] for x in results if "motion" in x})
    out_json = args.out_dir / f"eval_{tag}.json"
    out_json.write_text(json.dumps(report, indent=1))
    print_report(report)
    print(f"\n{len(jobs)} episodes in {runtime:.1f} s; wrote {out_json}; motions: {len(report['motions'])}", flush=True)


if __name__ == "__main__":
    main()
