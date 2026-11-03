"""Round-trip check of printed_face/urdf/kobi.urdf against printed_face/mujoco/kobi_scene.xml.

Loads the URDF with MuJoCo and compares, link by link:
  - mass and principal inertia,
  - pose of every link and site frame relative to the root link at random joint angles inside the limits,
  - world position of every visual mesh vertex (so mesh frames and scales are right),
  - joint ranges and axes.
Prints one URDFCHECK line with the worst errors and exits 1 if any exceeds its tolerance.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
MJCF_PATH = HERE.parent / "printed_face" / "mujoco" / "kobi_scene.xml"
URDF_PATH = HERE.parent / "printed_face" / "urdf" / "kobi.urdf"
ROOT = "trunk_assembly"
SAMPLES = 200
SEED = 20261102
POSE_TOL_M = 1e-6
ROT_TOL = 1e-6
MASS_TOL_KG = 1e-9
INERTIA_REL_TOL = 1e-6
VERTEX_TOL_M = 1e-6
INERTIA_FLOOR = 1e-12


def body_names(model: mujoco.MjModel) -> list[str]:
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(model.nbody)]


def inertia_tensor(model: mujoco.MjModel, body: int) -> np.ndarray:
    """Full inertia tensor in the body frame; principal axes may come back in another order, the tensor may not."""
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, model.body_iquat[body])
    rot = rot.reshape(3, 3)
    return rot @ np.diag(model.body_inertia[body]) @ rot.T


def relative_pose(data: mujoco.MjData, root: int, body: int) -> tuple[np.ndarray, np.ndarray]:
    r_root = data.xmat[root].reshape(3, 3)
    pos = r_root.T @ (data.xpos[body] - data.xpos[root])
    rot = r_root.T @ data.xmat[body].reshape(3, 3)
    return pos, rot


def set_joints(model: mujoco.MjModel, data: mujoco.MjData, angles: dict[str, float]) -> None:
    for name, value in angles.items():
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[j]] = value
    mujoco.mj_fwdPosition(model, data)   # kinematics plus subtree centres of mass


def mesh_vertices_world(model: mujoco.MjModel, data: mujoco.MjData, body: int) -> list[np.ndarray]:
    out = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != body or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        m = model.geom_dataid[g]
        verts = model.mesh_vert[model.mesh_vertadr[m]:model.mesh_vertadr[m] + model.mesh_vertnum[m]]
        out.append(data.geom_xpos[g] + verts @ data.geom_xmat[g].reshape(3, 3).T)
    return out


def worst_vertex_error(ref: list[np.ndarray], got: list[np.ndarray]) -> float:
    """Every reference mesh must match some URDF mesh of the same body vertex by vertex."""
    worst = 0.0
    for r in ref:
        best = min((float(np.abs(g - r).max()) for g in got if g.shape == r.shape), default=float("inf"))
        worst = max(worst, best)
    return worst


def main() -> int:
    ref = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    urdf = mujoco.MjModel.from_xml_path(str(URDF_PATH))
    rd, ud = mujoco.MjData(ref), mujoco.MjData(urdf)
    links = [n for n in body_names(urdf) if n and n != "world"]
    shared = [n for n in links if n in body_names(ref)]
    frames = [n for n in links if n not in shared]
    site_ids = {mujoco.mj_id2name(ref, mujoco.mjtObj.mjOBJ_SITE, s): s for s in range(ref.nsite)}
    hinge = [mujoco.mj_id2name(ref, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(ref.njnt)
             if ref.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    res = {"links": len(links), "bodies_matched": len(shared), "frames": len(frames), "joints": len(hinge)}

    mass_err = max(abs(ref.body_mass[ref.body(n).id] - urdf.body_mass[urdf.body(n).id]) for n in shared)
    inertia_err = max(float(np.abs(inertia_tensor(ref, ref.body(n).id) - inertia_tensor(urdf, urdf.body(n).id)).max()
                            / max(float(ref.body_inertia[ref.body(n).id].max()), INERTIA_FLOOR)) for n in shared)
    range_err = max(float(np.abs(ref.jnt_range[ref.joint(n).id] - urdf.jnt_range[urdf.joint(n).id]).max()) for n in hinge)
    res.update(total_mass_ref=float(ref.body_mass.sum()), total_mass_urdf=float(urdf.body_mass.sum()),
               mass_err=float(mass_err), inertia_rel_err=inertia_err, range_err=range_err)

    rng = np.random.default_rng(SEED)
    pose_err, rot_err, vert_err, frame_err, com_err = 0.0, 0.0, 0.0, 0.0, 0.0
    r_root, u_root = ref.body(ROOT).id, urdf.body(ROOT).id
    for k in range(SAMPLES + 1):
        angles = {n: (0.0 if k == 0 else float(rng.uniform(*ref.jnt_range[ref.joint(n).id]))) for n in hinge}
        set_joints(ref, rd, angles)
        set_joints(urdf, ud, angles)
        for n in shared:
            p1, m1 = relative_pose(rd, r_root, ref.body(n).id)
            p2, m2 = relative_pose(ud, u_root, urdf.body(n).id)
            pose_err = max(pose_err, float(np.abs(p1 - p2).max()))
            rot_err = max(rot_err, float(np.abs(m1 - m2).max()))
            if k % 20 == 0:
                vert_err = max(vert_err, worst_vertex_error(
                    [(v - rd.xpos[r_root]) @ rd.xmat[r_root].reshape(3, 3) for v in mesh_vertices_world(ref, rd, ref.body(n).id)],
                    [(v - ud.xpos[u_root]) @ ud.xmat[u_root].reshape(3, 3) for v in mesh_vertices_world(urdf, ud, urdf.body(n).id)]))
        for n in frames:
            s = site_ids[n]
            rr = rd.xmat[r_root].reshape(3, 3)
            p1 = rr.T @ (rd.site_xpos[s] - rd.xpos[r_root])
            m1 = rr.T @ rd.site_xmat[s].reshape(3, 3)
            p2, m2 = relative_pose(ud, u_root, urdf.body(n).id)
            frame_err = max(frame_err, float(np.abs(p1 - p2).max()), float(np.abs(m1 - m2).max()))
        rr = rd.xmat[r_root].reshape(3, 3)
        c1 = rr.T @ (rd.subtree_com[0] - rd.xpos[r_root])
        c2 = ud.xmat[u_root].reshape(3, 3).T @ (ud.subtree_com[0] - ud.xpos[u_root])
        com_err = max(com_err, float(np.abs(c1 - c2).max()))
    res.update(pose_err_m=pose_err, rot_err=rot_err, vertex_err_m=vert_err, frame_err=frame_err, com_err_m=com_err,
               samples=SAMPLES)
    ok = (mass_err <= MASS_TOL_KG and inertia_err <= INERTIA_REL_TOL and range_err <= ROT_TOL and pose_err <= POSE_TOL_M
          and rot_err <= ROT_TOL and vert_err <= VERTEX_TOL_M and frame_err <= POSE_TOL_M and com_err <= POSE_TOL_M
          and len(frames) == ref.nsite)
    res["ok"] = bool(ok)
    print("URDFCHECK", json.dumps(res))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
