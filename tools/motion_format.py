"""Motion file format shared by the recorders (MuJoCo side) and animate_motion.py (Blender side).

Only numpy is needed, so Blender's bundled Python can import this module too.
See README.md for the field list.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

FPS: int = 50                           # control rate of the Playground environment
BASE_BODY: str = "base"                 # free-floating root body of the Kobi MJCF
TRUNK_BODY: str = "trunk_assembly"      # the body that carries the jacket, used by the camera
REQUIRED_KEYS: tuple[str, ...] = ("body_names", "xpos", "xquat", "fps")
SOURCE_KINEMATIC: str = "kinematic_preview"
SOURCE_POLICY: str = "policy_mujoco"


def save_motion(
    path: Path,
    body_names: list[str],
    xpos: np.ndarray,
    xquat: np.ndarray,
    fps: int = FPS,
    labels: list[str] | None = None,
    source: str = SOURCE_POLICY,
    kinematic: bool = False,
    **extra: np.ndarray,
) -> None:
    """Write a motion .npz. xpos is (frames, nbody, 3), xquat (frames, nbody, 4) in MuJoCo wxyz order."""
    xpos = np.asarray(xpos, dtype=np.float64)
    xquat = np.asarray(xquat, dtype=np.float64)
    if xpos.ndim != 3 or xpos.shape[2] != 3 or xquat.shape != xpos.shape[:2] + (4,):
        raise ValueError(f"bad shapes: xpos {xpos.shape}, xquat {xquat.shape}")
    if xpos.shape[1] != len(body_names):
        raise ValueError(f"{xpos.shape[1]} bodies in xpos but {len(body_names)} names")
    arrays: dict[str, Any] = dict(
        body_names=np.array(body_names), xpos=xpos, xquat=xquat, fps=np.int64(fps),
        source=np.array(source), kinematic=np.bool_(kinematic), **extra,
    )
    if labels is not None:
        if len(labels) != len(xpos):
            raise ValueError(f"{len(labels)} labels for {len(xpos)} frames")
        arrays["labels"] = np.array(labels)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_motion(path: Path) -> dict[str, Any]:
    """Read a motion .npz into a dict with python types for the scalar fields."""
    with np.load(path, allow_pickle=False) as data:
        missing = [k for k in REQUIRED_KEYS if k not in data.files]
        if missing:
            raise KeyError(f"{path} lacks {missing}")
        out: dict[str, Any] = {k: data[k] for k in data.files}
    out["body_names"] = [str(n) for n in out["body_names"]]
    out["fps"] = int(out["fps"])
    out["kinematic"] = bool(out.get("kinematic", False))
    out["source"] = str(out.get("source", "unknown"))
    out["labels"] = [str(s) for s in out["labels"]] if "labels" in out else None
    return out


def quat_to_matrices(quat: np.ndarray) -> np.ndarray:
    """Rotation matrices (..., 3, 3) from MuJoCo wxyz quaternions (..., 4); input need not be normalised."""
    q = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    rot = np.empty(q.shape[:-1] + (3, 3))
    rot[..., 0, 0] = 1 - 2 * (y * y + z * z)
    rot[..., 0, 1] = 2 * (x * y - w * z)
    rot[..., 0, 2] = 2 * (x * z + w * y)
    rot[..., 1, 0] = 2 * (x * y + w * z)
    rot[..., 1, 1] = 1 - 2 * (x * x + z * z)
    rot[..., 1, 2] = 2 * (y * z - w * x)
    rot[..., 2, 0] = 2 * (x * z - w * y)
    rot[..., 2, 1] = 2 * (y * z + w * x)
    rot[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return rot


def pose_matrices(xpos: np.ndarray, xquat: np.ndarray) -> np.ndarray:
    """Homogeneous world poses (frames, nbody, 4, 4) of every body."""
    mats = np.zeros(xpos.shape[:-1] + (4, 4))
    mats[..., :3, :3] = quat_to_matrices(xquat)
    mats[..., :3, 3] = xpos
    mats[..., 3, 3] = 1.0
    return mats
