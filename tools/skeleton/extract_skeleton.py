"""Extract the Kobi kinematic skeleton from a MuJoCo MJCF into skeleton.json.

Run (MuJoCo only, no project install; run tools/fetch_upstream_meshes.py once beforehand so the scene compiles):

    uv run --no-project --python 3.12 --with mujoco==3.14.0 --with numpy python -I \
        tools/skeleton/extract_skeleton.py [--mjcf PATH] [--out PATH] [--fk-reference PATH]

What is written to skeleton.json
  * every body: parent, world pose at qpos0 (all hinge angles 0) and at the keyframe "home";
  * every hinge: child body, driving bone, world anchor and world axis at qpos0, range, policy action index,
    torque limit used in training, home angle;
  * the visual mesh geoms that we may redistribute (our own meshes), with the affine transform
    "world_from_file" that maps the vertices of the STL file (file units) to the world frame at qpos0;
  * the names of the visual geoms that were left out because their mesh comes from Open_Duck_Playground
    (no license). Only names are listed, never geometry.

With --fk-reference a second JSON is written: MuJoCo forward kinematics (joint anchors and axes) for random joint
configurations inside the joint ranges, used by verify_rig.py as ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MJCF = REPO_ROOT / "printed_face" / "mujoco" / "kobi_scene.xml"
DEFAULT_OUT = REPO_ROOT / "printed_face" / "skeleton" / "skeleton.json"
FETCH_SCRIPT = REPO_ROOT / "tools" / "fetch_upstream_meshes.py"

SCHEMA_VERSION = 1
HOME_KEYFRAME = "home"
ROOT_BONE_NAME = "trunk"
VISUAL_GEOM_GROUP = 2  # class "visual" of the MJCF
TORQUE_LIMIT_LEG_NECK_NM = 1.91  # tools/kobi_env_mirror.py: torque_limit_leg_neck
TORQUE_LIMIT_SHOULDER_NM = 0.20  # tools/kobi_env_mirror.py: torque_limit_shoulder
SHOULDER_PREFIX = "shoulder_"
STL_HEADER_BYTES = 80
STL_TRIANGLE_BYTES = 50
MESH_CHECK_TOLERANCE_M = 1e-5
DEFAULT_SAMPLES = 20
DEFAULT_SEED = 20261103

# Policy action order (README section 4). The MJCF actuator order must be identical.
POLICY_ACTION_ORDER: tuple[str, ...] = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
    "neck_yaw", "neck_pitch", "shoulder_left", "shoulder_right",
)


def read_binary_stl(path: Path) -> np.ndarray:
    """Return the triangle corners of a binary STL as a float64 array of shape (n_triangles, 3, 3)."""
    data = path.read_bytes()
    (count,) = struct.unpack_from("<I", data, STL_HEADER_BYTES)
    expected = STL_HEADER_BYTES + 4 + count * STL_TRIANGLE_BYTES
    if len(data) != expected:
        raise ValueError(f"{path.name}: not a binary STL (size {len(data)} != {expected})")
    dtype = np.dtype([("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    return np.frombuffer(data, dtype=dtype, count=count, offset=STL_HEADER_BYTES + 4)["v"].astype(np.float64)


def load_upstream_hashes(fetch_script: Path) -> dict[str, str]:
    """Parse the file name -> SHA-256 table of the 28 upstream meshes out of tools/fetch_upstream_meshes.py."""
    text = fetch_script.read_text(encoding="utf-8")
    table = dict(re.findall(r'"([A-Za-z0-9_]+\.stl)":\s*"([0-9a-f]{64})"', text))
    if len(table) != 28:
        raise RuntimeError(f"expected 28 upstream mesh hashes, found {len(table)}")
    return table


def mesh_files_from_xml(mjcf: Path) -> dict[str, tuple[str, Path]]:
    """Map mesh asset name -> (file name, absolute path) by reading the MJCF text (meshdir aware)."""
    root = ET.parse(mjcf).getroot()
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    result: dict[str, tuple[str, Path]] = {}
    for mesh in root.iter("mesh"):
        name, file = mesh.get("name"), mesh.get("file")
        if name and file:
            result[name] = (Path(file).name, (mjcf.parent / meshdir / file).resolve())
    return result


def pose_record(pos: np.ndarray, xmat: np.ndarray) -> dict[str, Any]:
    """World pose as position, quaternion (w, x, y, z) and a row-major 3x3 rotation matrix."""
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, xmat.reshape(9))
    return {"position": pos.tolist(), "quat_wxyz": quat.tolist(), "matrix": xmat.reshape(3, 3).tolist()}


def kinematics(model: mujoco.MjModel, qpos: np.ndarray) -> mujoco.MjData:
    """Run forward kinematics for the given generalized positions."""
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    return data


def torque_limit_for(joint_name: str) -> float:
    """Torque limit (N*m) applied during training: shoulders 0.20, legs and neck 1.91."""
    return TORQUE_LIMIT_SHOULDER_NM if joint_name.startswith(SHOULDER_PREFIX) else TORQUE_LIMIT_LEG_NECK_NM


def find_root_body(model: mujoco.MjModel) -> int:
    """Return the body that is the child of the free-joint body (the trunk)."""
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            free_body = int(model.jnt_bodyid[j])
            children = [b for b in range(model.nbody) if model.body_parentid[b] == free_body and b != free_body]
            if len(children) != 1:
                raise RuntimeError("expected exactly one child below the free-joint body")
            return children[0]
    raise RuntimeError("no free joint found")


def build_bone_map(model: mujoco.MjModel, root_body: int) -> tuple[dict[int, str | None], dict[str, str | None]]:
    """Return (body id -> bone name, bone name -> parent bone name). Bodies without a hinge share their parent's bone."""
    hinge_of_body: dict[int, str] = {}
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            hinge_of_body[int(model.jnt_bodyid[j])] = model.joint(j).name
    body_bone: dict[int, str | None] = {}
    bone_parent: dict[str, str | None] = {}
    for b in range(model.nbody):  # parents always have a smaller id than their children in MJCF order
        if b == root_body:
            body_bone[b] = ROOT_BONE_NAME
            bone_parent[ROOT_BONE_NAME] = None
        elif b in hinge_of_body:
            name = hinge_of_body[b]
            parent_bone = body_bone.get(int(model.body_parentid[b]))
            if parent_bone is None:
                raise RuntimeError(f"hinge {name} has no skinned ancestor")
            body_bone[b] = name
            bone_parent[name] = parent_bone
        else:
            body_bone[b] = body_bone.get(int(model.body_parentid[b]))  # None above the trunk (world, base, floor)
    return body_bone, bone_parent


def check_action_order(model: mujoco.MjModel) -> dict[str, int]:
    """Check that the actuator order equals the policy action order and return joint name -> action index."""
    names = tuple(model.joint(int(model.actuator_trnid[a][0])).name for a in range(model.nu))
    if names != POLICY_ACTION_ORDER:
        raise RuntimeError(f"actuator order {names} differs from the policy action order")
    return {name: i for i, name in enumerate(names)}


def check_mesh_transform(model: mujoco.MjModel, data: mujoco.MjData, g: int, world_from_file: np.ndarray,
                         triangles: np.ndarray) -> float:
    """Compare the STL placed with world_from_file against MuJoCo's own compiled mesh. Return the max deviation (m)."""
    mid = int(model.geom_dataid[g])
    start, count = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
    verts = model.mesh_vert[start:start + count].astype(np.float64)
    mj_world = data.geom_xpos[g] + verts @ data.geom_xmat[g].reshape(3, 3).T
    flat = triangles.reshape(-1, 3)
    ours = flat @ world_from_file[:3, :3].T + world_from_file[:3, 3]
    deviation = max(
        float(np.abs(ours.min(axis=0) - mj_world.min(axis=0)).max()),
        float(np.abs(ours.max(axis=0) - mj_world.max(axis=0)).max()),
    )
    # centroid of the triangle centres is order independent; MuJoCo keeps the STL face order
    face_start, face_count = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
    faces = model.mesh_face[face_start:face_start + face_count]
    mj_centres = mj_world[faces].mean(axis=1).mean(axis=0)
    our_centres = ours.reshape(-1, 3, 3).mean(axis=1).mean(axis=0)
    return max(deviation, float(np.abs(mj_centres - our_centres).max()))


def collect_geoms(model: mujoco.MjModel, data: mujoco.MjData, body_bone: dict[int, str | None], mjcf: Path,
                  upstream: dict[str, str]) -> tuple[list[dict[str, Any]], list[dict[str, str]], float]:
    """Collect the redistributable visual mesh geoms and the names of the excluded upstream ones."""
    files = mesh_files_from_xml(mjcf)
    upstream_hashes = set(upstream.values())
    kept: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    worst = 0.0
    for g in range(model.ngeom):
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or model.geom_group[g] != VISUAL_GEOM_GROUP:
            continue
        mid = int(model.geom_dataid[g])
        mesh_name = model.mesh(mid).name
        file_name, file_path = files[mesh_name]
        body_name = model.body(int(model.geom_bodyid[g])).name
        digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
        if file_name in upstream or digest in upstream_hashes:
            excluded.append({"body": body_name, "mesh": mesh_name, "file": file_name, "reason": "upstream mesh without license"})
            continue
        scale = model.mesh_scale[mid].astype(np.float64)
        rot_mesh = np.zeros(9)
        mujoco.mju_quat2Mat(rot_mesh, model.mesh_quat[mid])
        rot_mesh = rot_mesh.reshape(3, 3)
        xmat = data.geom_xmat[g].reshape(3, 3)
        # stored = R_mesh^T (scale * file - mesh_pos);  world = xpos + xmat @ stored
        linear = xmat @ rot_mesh.T
        world_from_file = np.eye(4)
        world_from_file[:3, :3] = linear * scale[None, :]
        world_from_file[:3, 3] = data.geom_xpos[g] - linear @ model.mesh_pos[mid]
        deviation = check_mesh_transform(model, data, g, world_from_file, read_binary_stl(file_path))
        if deviation > MESH_CHECK_TOLERANCE_M:
            raise RuntimeError(f"mesh {mesh_name}: transform check failed ({deviation:.3e} m)")
        worst = max(worst, deviation)
        matid = int(model.geom_matid[g])
        rgba = model.mat_rgba[matid] if matid >= 0 else model.geom_rgba[g]
        geom_name = model.geom(g).name or f"{body_name}__{mesh_name}"
        kept.append({
            "geom": geom_name,
            "body": body_name,
            "bone": body_bone[int(model.geom_bodyid[g])],
            "mesh": mesh_name,
            "mesh_file": file_path.relative_to(REPO_ROOT).as_posix(),
            "mesh_file_sha256": digest,
            "rgba": [float(x) for x in rgba],
            "world_from_file": world_from_file.tolist(),
        })
    return kept, excluded, worst


def extract(mjcf: Path) -> tuple[dict[str, Any], mujoco.MjModel]:
    """Build the skeleton dictionary for one MJCF."""
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    action_index = check_action_order(model)
    root_body = find_root_body(model)
    body_bone, bone_parent = build_bone_map(model, root_body)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, HOME_KEYFRAME)
    if key_id < 0:
        raise RuntimeError(f"keyframe {HOME_KEYFRAME!r} not found")
    data0 = kinematics(model, model.qpos0)
    data_home = kinematics(model, model.key_qpos[key_id])

    bodies = []
    for b in range(model.nbody):
        bodies.append({
            "name": model.body(b).name,
            "parent": None if b == 0 else model.body(int(model.body_parentid[b])).name,
            "bone": body_bone[b],
            "mass_kg": float(model.body_mass[b]),
            "com_qpos0": data0.xipos[b].tolist(),
            "qpos0": pose_record(data0.xpos[b], data0.xmat[b]),
            "home": pose_record(data_home.xpos[b], data_home.xmat[b]),
        })

    joints = []
    home_angles: dict[str, float] = {}
    for j in range(model.njnt):
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        name = model.joint(j).name
        body_id = int(model.jnt_bodyid[j])
        angle = float(model.key_qpos[key_id][model.jnt_qposadr[j]])
        home_angles[name] = angle
        joints.append({
            "name": name,
            "child_body": model.body(body_id).name,
            "parent_bone": bone_parent[name],
            "anchor_world": data0.xanchor[j].tolist(),
            "axis_world": data0.xaxis[j].tolist(),
            "range": model.jnt_range[j].tolist(),
            "action_index": action_index[name],
            "torque_limit_nm": torque_limit_for(name),
            "home_angle": angle,
        })
    joints.sort(key=lambda item: item["action_index"])

    upstream = load_upstream_hashes(FETCH_SCRIPT)
    geoms, excluded, worst = collect_geoms(model, data0, body_bone, mjcf, upstream)
    free = next(j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)
    adr = int(model.jnt_qposadr[free])
    result = {
        "schema_version": SCHEMA_VERSION,
        "source_mjcf": mjcf.relative_to(REPO_ROOT).as_posix(),
        "mujoco_version": mujoco.__version__,
        "units": {"length": "metre", "angle": "radian", "torque": "N*m"},
        "coordinate_system": "MuJoCo world: right-handed, x forward, y left, z up",
        "root_bone": ROOT_BONE_NAME,
        "root_body": model.body(root_body).name,
        "action_order": list(POLICY_ACTION_ORDER),
        "home": {
            "keyframe": HOME_KEYFRAME,
            "root_position": model.key_qpos[key_id][adr:adr + 3].tolist(),
            "root_quat_wxyz": model.key_qpos[key_id][adr + 3:adr + 7].tolist(),
            "qpos0_root_position": model.qpos0[adr:adr + 3].tolist(),
            "joint_angles": home_angles,
        },
        "bodies": bodies,
        "joints": joints,
        "mesh_geoms": geoms,
        "excluded_upstream_geoms": excluded,
        "checks": {"mesh_transform_max_deviation_m": worst},
    }
    return result, model


def fk_reference(model: mujoco.MjModel, skeleton: dict[str, Any], samples: int, seed: int) -> dict[str, Any]:
    """MuJoCo ground truth: joint anchors and axes for random in-range angles, plus the home pose."""
    rng = np.random.default_rng(seed)
    hinges = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    root_body = find_root_body(model)

    def record(label: str, qpos: np.ndarray) -> dict[str, Any]:
        data = kinematics(model, qpos)
        return {
            "label": label,
            "angles": {model.joint(j).name: float(qpos[model.jnt_qposadr[j]]) for j in hinges},
            "root_position": data.xpos[root_body].tolist(),
            "bodies": {skeleton_body["name"]: {"position": data.xpos[i].tolist(), "matrix": data.xmat[i].reshape(3, 3).tolist()}
                       for i, skeleton_body in enumerate(skeleton["bodies"]) if skeleton_body["bone"] is not None},
            "anchors": {model.joint(j).name: data.xanchor[j].tolist() for j in hinges},
            "axes": {model.joint(j).name: data.xaxis[j].tolist() for j in hinges},
        }

    cases = []
    for i in range(samples):
        qpos = model.qpos0.copy()
        for j in hinges:
            low, high = model.jnt_range[j]
            qpos[model.jnt_qposadr[j]] = rng.uniform(low, high)
        cases.append(record(f"random_{i:02d}", qpos))
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, HOME_KEYFRAME)
    home = record("home", model.key_qpos[key_id])
    rest = record("rest", model.qpos0)
    return {"seed": seed, "random_cases": cases, "home": home, "rest": rest}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--fk-reference", type=Path, default=None, help="also write MuJoCo FK ground truth JSON here")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    skeleton, model = extract(args.mjcf.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(skeleton, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out.name}: {len(skeleton['bodies'])} bodies, {len(skeleton['joints'])} hinges, "
          f"{len(skeleton['mesh_geoms'])} mesh geoms kept, {len(skeleton['excluded_upstream_geoms'])} upstream geoms excluded, "
          f"mesh check max {skeleton['checks']['mesh_transform_max_deviation_m']:.2e} m")
    if args.fk_reference is not None:
        args.fk_reference.parent.mkdir(parents=True, exist_ok=True)
        args.fk_reference.write_text(json.dumps(fk_reference(model, skeleton, args.samples, args.seed)) + "\n", encoding="utf-8")
        print(f"wrote FK reference with {args.samples} random cases + home + rest")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
