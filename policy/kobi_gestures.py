"""Kobi upper-body gesture library: neck and shoulder targets for meaningful scenario motions.

Each gesture is a list of keyframes (time in s, neck_yaw, neck_pitch, shoulder_left, shoulder_right in degrees).
Signs were measured on kobi_scene.xml at the home pose (2026-11-02):
  neck_yaw  +  = head turns to the robot's left
  neck_pitch + = head nods down (looks down)
  shoulder  -  = arm swings forward and up, + = arm swings back (same sign on both sides)
Keyframes are joined with cosine easing, sampled at the control rate, checked against joint limits
with a margin and against a speed limit. Mirroring swaps the arms and flips the neck yaw.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

CONTROL_DT_S: float = 0.02                      # policy rate of the Playground env (50 Hz)
DEG: float = math.pi / 180.0
SLOT_NAMES: tuple[str, ...] = ("neck_yaw", "neck_pitch", "shoulder_left", "shoulder_right")
LIMITS_DEG: dict[str, tuple[float, float]] = {   # kobi_scene.xml joint ranges
    "neck_yaw": (-30.0, 30.0),
    "neck_pitch": (-10.0, 10.0),
    "shoulder_left": (-135.0, 25.0),
    "shoulder_right": (-135.0, 25.0),
}
LIMIT_MARGIN_DEG: float = 2.0                    # targets stay this far inside the joint range
MAX_SPEED_DEG_S: float = 200.0                   # peak target speed; env clips targets at 5.24 rad/s (300 deg/s)
COSINE_PEAK_FACTOR: float = math.pi / 2.0        # peak / mean speed of a cosine-eased segment

Keyframe = tuple[float, float, float, float, float]


@dataclass(frozen=True)
class Gesture:
    name: str
    meaning: str          # Korean label used in the scenario documents
    keys: tuple[Keyframe, ...]


def _g(name: str, meaning: str, *keys: Keyframe) -> Gesture:
    return Gesture(name, meaning, tuple(keys))


GESTURES: tuple[Gesture, ...] = (
    _g("idle", "대기", (0.0, 0, 0, 0, 0), (2.0, 0, 0, 0, 0)),
    _g("look_left_right", "고개 좌우 시범",
       (0.0, 0, 0, 0, 0), (0.7, 25, 0, 0, 0), (1.0, 25, 0, 0, 0), (2.0, -25, 0, 0, 0),
       (2.3, -25, 0, 0, 0), (3.0, 0, 0, 0, 0)),
    _g("nod", "끄덕임",
       (0.0, 0, 0, 0, 0), (0.3, 0, 8, 0, 0), (0.6, 0, -2, 0, 0), (0.9, 0, 8, 0, 0),
       (1.2, 0, -2, 0, 0), (1.5, 0, 8, 0, 0), (1.9, 0, 0, 0, 0)),
    _g("shake_no", "고개 젓기(아니야·다시)",
       (0.0, 0, 0, 0, 0), (0.3, 15, 0, 0, 0), (0.6, -15, 0, 0, 0), (0.9, 15, 0, 0, 0),
       (1.2, -15, 0, 0, 0), (1.5, 0, 0, 0, 0)),
    _g("wave_right", "인사(손 흔들기)",
       (0.0, 0, 0, 0, 0), (1.0, 0, 4, 0, -120), (1.3, 0, 4, 0, -95), (1.6, 0, 4, 0, -120),
       (1.9, 0, 4, 0, -95), (2.2, 0, 4, 0, -120), (3.2, 0, 0, 0, 0)),
    _g("bow", "꾸벅 인사",
       (0.0, 0, 0, 0, 0), (0.5, 0, 8, -15, -15), (1.3, 0, 8, -15, -15), (1.9, 0, 0, 0, 0)),
    _g("guide_left", "안내(왼쪽)",
       (0.0, 0, 0, 0, 0), (0.8, 28, 0, -90, 0), (2.3, 28, 0, -90, 0), (3.1, 0, 0, 0, 0)),
    _g("cheer", "응원·파이팅",
       (0.0, 0, 0, 0, 0), (1.1, 0, -5, -130, -95), (1.45, 0, -5, -95, -130), (1.8, 0, -5, -130, -95),
       (2.15, 0, -5, -95, -130), (2.5, 0, -5, -130, -95), (3.6, 0, 0, 0, 0)),
    _g("celebrate", "축하·만세",
       (0.0, 0, 0, 0, 0), (1.1, 0, -8, -130, -130), (1.4, 0, -8, -115, -115), (1.7, 0, -8, -130, -130),
       (2.0, 0, -8, -115, -115), (2.3, 0, -8, -130, -130), (3.4, 0, 0, 0, 0)),
    _g("photo_pose", "사진 포즈",
       (0.0, 0, 0, 0, 0), (0.9, 12, -3, 15, -100), (2.9, 12, -3, 15, -100), (3.8, 0, 0, 0, 0)),
    _g("surprise", "놀람",
       (0.0, 0, 0, 0, 0), (0.55, 0, -8, -70, -70), (1.2, 0, -8, -70, -70), (2.0, 0, 0, 0, 0)),
    _g("look_far", "멀리 보기",
       (0.0, 0, 0, 0, 0), (0.9, 0, -6, 0, -110), (1.7, 20, -6, 0, -110), (2.7, -20, -6, 0, -110),
       (3.3, 0, -6, 0, -110), (4.2, 0, 0, 0, 0)),
    _g("think", "생각하기·관찰",
       (0.0, 0, 0, 0, 0), (0.6, 18, 6, 0, -40), (2.0, 18, 6, 0, -40), (2.6, 0, 0, 0, 0)),
    _g("like", "좋아",
       (0.0, 0, 0, 0, 0), (0.5, 0, 0, -60, -60), (0.8, 0, 7, -60, -60), (1.1, 0, -1, -60, -60),
       (1.4, 0, 7, -60, -60), (1.7, 0, 0, -60, -60), (2.4, 0, 0, 0, 0)),
)


def mirror(g: Gesture) -> Gesture:
    """Left-right mirror: neck yaw changes sign, the two shoulders swap (their sign convention is shared)."""
    name = g.name.replace("right", "@").replace("left", "right").replace("@", "left")
    keys = tuple((t, -yaw, pitch, sh_r, sh_l) for t, yaw, pitch, sh_l, sh_r in g.keys)
    return Gesture(name if name != g.name else g.name + "_mirror", g.meaning, keys)


def sample(g: Gesture, dt: float = CONTROL_DT_S) -> np.ndarray:
    """Targets in radians, shape (steps, 4), cosine easing between keyframes."""
    times = np.array([k[0] for k in g.keys])
    values = np.array([k[1:] for k in g.keys], dtype=float)
    steps = int(round(times[-1] / dt)) + 1
    out = np.empty((steps, len(SLOT_NAMES)))
    for i in range(steps):
        t = min(i * dt, times[-1])
        seg = min(int(np.searchsorted(times, t, side="right")) - 1, len(times) - 2)
        span = times[seg + 1] - times[seg]
        u = 0.0 if span <= 0.0 else (t - times[seg]) / span
        w = 0.5 - 0.5 * math.cos(math.pi * u)
        out[i] = values[seg] + w * (values[seg + 1] - values[seg])
    return out * DEG


def check(g: Gesture) -> list[str]:
    """Return problems: targets outside limits (with margin), peak speed above the limit, non-increasing times."""
    problems: list[str] = []
    times = [k[0] for k in g.keys]
    if any(b <= a for a, b in zip(times, times[1:])):
        problems.append(f"{g.name}: keyframe times must increase")
    for slot, name in enumerate(SLOT_NAMES):
        lo, hi = LIMITS_DEG[name]
        for k in g.keys:
            v = k[slot + 1]
            if not lo + LIMIT_MARGIN_DEG <= v <= hi - LIMIT_MARGIN_DEG:
                problems.append(f"{g.name}: {name}={v} deg outside [{lo + LIMIT_MARGIN_DEG}, {hi - LIMIT_MARGIN_DEG}]")
        for a, b in zip(g.keys, g.keys[1:]):
            span = b[0] - a[0]
            if span > 0.0:
                peak = COSINE_PEAK_FACTOR * abs(b[slot + 1] - a[slot + 1]) / span
                if peak > MAX_SPEED_DEG_S:
                    problems.append(f"{g.name}: {name} peak {peak:.0f} deg/s at t={a[0]}-{b[0]}")
    return problems


def library(include_mirrors: bool = True) -> list[Gesture]:
    """All gestures plus mirrored copies of the asymmetric ones."""
    out = list(GESTURES)
    if include_mirrors:
        for g in GESTURES:
            m = mirror(g)
            if m.keys != g.keys:
                out.append(m)
    return out


def as_padded_array(gestures: list[Gesture], dt: float = CONTROL_DT_S) -> tuple[np.ndarray, np.ndarray]:
    """Stack sampled gestures into (n, max_steps, 4) radians, padded with the last value, plus lengths (n,)."""
    clips = [sample(g, dt) for g in gestures]
    max_len = max(len(c) for c in clips)
    arr = np.stack([np.concatenate([c, np.repeat(c[-1:], max_len - len(c), axis=0)]) for c in clips])
    return arr, np.array([len(c) for c in clips], dtype=np.int32)


if __name__ == "__main__":
    lib = library()
    issues = [p for g in lib for p in check(g)]
    arr, lengths = as_padded_array(lib)
    print(f"gestures={len(lib)} padded={arr.shape} max_len_s={lengths.max() * CONTROL_DT_S:.2f}")
    for g, n in zip(lib, lengths):
        print(f"  {g.name:22s} {g.meaning:16s} {n * CONTROL_DT_S:4.2f} s")
    print("PROBLEMS", len(issues))
    for p in issues:
        print("  ", p)
