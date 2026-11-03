"""Command schedule shared by record_preview.py (kinematic) and record_policy.py (trained policy).

A schedule has two independent tracks, like the training episodes:
  walk     consecutive segments with a base velocity command (vx m/s forward, vy m/s left, wz rad/s yaw left);
  gestures named gestures from kobi_gestures.py, started at absolute times; between gestures the neck and
           shoulders hold the idle pose (all zero). Gestures may play while walking but must not overlap.
expand() turns a schedule into per-control-step arrays at 50 Hz.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

KOBI_TRAIN_DIR = Path(__file__).resolve().parent.parent / "policy"   # contains kobi_gestures.py
if str(KOBI_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(KOBI_TRAIN_DIR))
import kobi_gestures  # noqa: E402

UPPER_SLOTS: tuple[str, ...] = kobi_gestures.SLOT_NAMES          # neck_yaw, neck_pitch, shoulder_left, shoulder_right
STAND_LABEL: str = "stand still"
NO_GESTURE_LABEL: str = "-"


@dataclass(frozen=True)
class WalkSegment:
    duration_s: float
    vx: float = 0.0       # m/s, forward
    vy: float = 0.0       # m/s, left
    wz: float = 0.0       # rad/s, counter-clockwise seen from above


@dataclass(frozen=True)
class GestureEvent:
    start_s: float
    name: str


@dataclass(frozen=True)
class Schedule:
    walk: tuple[WalkSegment, ...]
    gestures: tuple[GestureEvent, ...] = ()
    tail_s: float = 0.0   # extra standing time appended after the last walk segment


@dataclass(frozen=True)
class Expanded:
    """Per-control-step arrays; row k is the command during step k (time k / fps)."""
    fps: int
    commands: np.ndarray        # (T, 3) vx, vy, wz
    upper_targets: np.ndarray   # (T, 4) neck_yaw, neck_pitch, shoulder_left, shoulder_right in radians
    segment: np.ndarray         # (T,) index of the walk segment (the last one for the tail)
    labels: list[str]           # (T,) text such as "walk vx=+0.15 vy=+0.00 wz=+0.00 | gesture: nod"
    gesture_names: list[str]    # (T,) the active gesture or "-"


# Default demo: 3 s walk forward, 3 s turn, then standing still while the remaining gestures play.
PREVIEW_SCHEDULE = Schedule(
    walk=(
        WalkSegment(3.0, vx=0.15),
        WalkSegment(3.0, wz=0.5),
        WalkSegment(10.2),
    ),
    gestures=(
        GestureEvent(0.4, "look_left_right"),
        GestureEvent(3.5, "nod"),
        GestureEvent(5.5, "guide_left"),
        GestureEvent(8.8, "wave_right"),
        GestureEvent(12.2, "cheer"),
    ),
    tail_s=0.0,
)


# Walking demo for the trained policy (2026-11-02): every command inside the training ranges
# (vx +-0.15 m/s, vy +-0.074 m/s, wz +-1.0 rad/s); no gestures, so the upper body holds home.
WALK_DEMO_SCHEDULE = Schedule(
    walk=(
        WalkSegment(2.0),
        WalkSegment(6.0, vx=0.10),
        WalkSegment(4.0, vx=0.15),
        WalkSegment(4.0, wz=0.8),
        WalkSegment(4.0, vx=0.10, wz=0.4),
        WalkSegment(4.0, wz=-0.8),
        WalkSegment(3.0, vx=-0.10),
        WalkSegment(4.0, vy=0.07),
        WalkSegment(4.0, vy=-0.07),
        WalkSegment(3.0),
    ),
)

# Gesture demo: the 14 library gestures one after another while standing (0.6 s idle between clips),
# then walking 0.10 m/s while waving, nodding and cheering, then standing.
GESTURE_DEMO_SCHEDULE = Schedule(
    walk=(
        WalkSegment(49.5),
        WalkSegment(12.0, vx=0.10),
        WalkSegment(2.0),
    ),
    gestures=(
        GestureEvent(1.0, "look_left_right"),
        GestureEvent(4.6, "nod"),
        GestureEvent(7.1, "shake_no"),
        GestureEvent(9.2, "wave_right"),
        GestureEvent(13.0, "bow"),
        GestureEvent(15.5, "guide_left"),
        GestureEvent(19.2, "guide_right"),
        GestureEvent(22.9, "cheer"),
        GestureEvent(27.1, "celebrate"),
        GestureEvent(31.1, "photo_pose"),
        GestureEvent(35.5, "surprise"),
        GestureEvent(38.1, "think"),
        GestureEvent(41.3, "like"),
        GestureEvent(44.3, "look_far"),
        GestureEvent(50.5, "wave_right"),
        GestureEvent(54.3, "nod"),
        GestureEvent(56.8, "cheer"),
    ),
)

SCHEDULES: dict[str, Schedule] = {
    "preview": PREVIEW_SCHEDULE,
    "walk_demo": WALK_DEMO_SCHEDULE,
    "gesture_demo": GESTURE_DEMO_SCHEDULE,
}

def total_duration_s(schedule: Schedule) -> float:
    return sum(s.duration_s for s in schedule.walk) + schedule.tail_s


def _label(cmd: np.ndarray, gesture: str) -> str:
    moving = bool(np.any(np.abs(cmd) > 1e-9))
    walk = f"walk vx={cmd[0]:+.2f} vy={cmd[1]:+.2f} wz={cmd[2]:+.2f}" if moving else STAND_LABEL
    return f"{walk} | gesture: {gesture}"


def expand(schedule: Schedule, fps: int = 50) -> Expanded:
    """Sample the schedule at fps. Raises ValueError for unknown or overlapping gestures."""
    dt = 1.0 / fps
    if abs(dt - kobi_gestures.CONTROL_DT_S) > 1e-9:
        raise ValueError(f"gesture library is sampled at {kobi_gestures.CONTROL_DT_S} s, not {dt} s")
    steps = int(round(total_duration_s(schedule) * fps))
    commands = np.zeros((steps, 3))
    segment = np.zeros(steps, dtype=np.int64)
    edges = np.cumsum([0.0] + [s.duration_s for s in schedule.walk])
    for k in range(steps):
        t = k * dt
        idx = min(int(np.searchsorted(edges, t + 1e-9, side="right")) - 1, len(schedule.walk) - 1)
        seg = schedule.walk[idx]
        commands[k] = (seg.vx, seg.vy, seg.wz)
        segment[k] = idx

    library = {g.name: g for g in kobi_gestures.library()}
    upper = np.zeros((steps, len(UPPER_SLOTS)))
    names = [NO_GESTURE_LABEL] * steps
    for ev in sorted(schedule.gestures, key=lambda e: e.start_s):
        if ev.name not in library:
            raise ValueError(f"unknown gesture {ev.name!r}; available: {sorted(library)}")
        clip = kobi_gestures.sample(library[ev.name])
        first = int(round(ev.start_s * fps))
        last = min(first + len(clip), steps)
        if first < 0 or first >= steps:
            raise ValueError(f"gesture {ev.name} starts outside the schedule at {ev.start_s} s")
        if any(n != NO_GESTURE_LABEL for n in names[first:last]):
            raise ValueError(f"gesture {ev.name} at {ev.start_s} s overlaps an earlier gesture")
        upper[first:last] = clip[: last - first]
        names[first:last] = [ev.name] * (last - first)
    labels = [_label(commands[k], names[k]) for k in range(steps)]
    return Expanded(fps, commands, upper, segment, labels, names)
