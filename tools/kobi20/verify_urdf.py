"""Round-trip check of the Kobi-20 URDFs (kobi20/urdf/) against the MuJoCo model they were made from (kobi20/mujoco/).

For each URDF (full and light):
  1. XML schema-level sanity (every link referenced, one root, tree, mesh files exist, limits sane, inertials physical),
  2. load with MuJoCo (the <mujoco> compiler tag) and compare with the MJCF: joint names/order/type/axes (world frame),
     ranges, damping, joint anchors, every link frame (position and rotation), COM, world inertia tensors, per-link mass,
     total mass, visual-mesh and collision-primitive placement, at the zero pose, the home pose and N random in-range poses,
  3. parse with pinocchio (if importable) and compare frames, masses, COM and limits.
Also checks that the full URDF uses exactly the STL files of the MuJoCo model (kobi20/mujoco/meshes) and gathers frame-convention evidence.

Usage (from anywhere; pinocchio is optional, it is skipped with a note when it is not importable):
    uv run --no-project --python 3.12 --with mujoco==3.14.0 --with numpy --with pin python -I -B tools/kobi20/verify_urdf.py <out_json>
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

PACKAGE = Path(__file__).resolve().parents[2] / "kobi20"
MESH_DIR = PACKAGE / "mujoco" / "meshes"
MJCF = PACKAGE / "mujoco" / "kobi20_v8.xml"
URDFS = {"full": PACKAGE / "urdf" / "kobi20_v8.urdf", "light": PACKAGE / "urdf" / "kobi20_v8_light.urdf"}
LIGHT_MESH_PREFIX = "assets_light/"
FULL_MESH_PREFIX = "../mujoco/meshes/"
ROOT_BODY = "pelvis"
N_RANDOM_POSES = 20
SEED = 0
TOL_FK_MM = 0.01
TOL_MASS_G = 0.1
M_TO_MM = 1000.0
KG_TO_G = 1000.0


def now() -> str:
    return subprocess.run(["date", "+%Y-%m-%d %H:%M:%S %Z"], capture_output=True, text=True, check=True).stdout.strip()


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ----------------------------------------------------------------------------- schema level
def schema_check(urdf: Path) -> dict[str, object]:
    root = ET.parse(urdf).getroot()
    problems: list[str] = []
    links = [e.get("name") for e in root.findall("link")]
    joints = root.findall("joint")
    if len(set(links)) != len(links):
        problems.append("duplicate link names")
    jnames = [j.get("name") for j in joints]
    if len(set(jnames)) != len(jnames):
        problems.append("duplicate joint names")
    parent_of: dict[str, str] = {}
    children: dict[str, list[str]] = {}
    for j in joints:
        p, c = j.find("parent").get("link"), j.find("child").get("link")
        if p not in links or c not in links:
            problems.append(f"joint {j.get('name')} references an unknown link ({p}->{c})")
        if c in parent_of:
            problems.append(f"link {c} has two parents")
        parent_of[c] = p
        children.setdefault(p, []).append(c)
        if j.get("type") not in {"revolute", "fixed"}:
            problems.append(f"joint {j.get('name')}: type {j.get('type')} not expected")
        if j.get("type") == "revolute":
            ax = np.array([float(v) for v in j.find("axis").get("xyz").split()])
            lim = j.find("limit")
            lo, hi = float(lim.get("lower")), float(lim.get("upper"))
            if abs(np.linalg.norm(ax) - 1.0) > 1e-9:
                problems.append(f"joint {j.get('name')}: axis not unit")
            if not lo < hi:
                problems.append(f"joint {j.get('name')}: lower >= upper")
            if float(lim.get("effort")) <= 0 or float(lim.get("velocity")) <= 0:
                problems.append(f"joint {j.get('name')}: non-positive effort/velocity")
    roots = [l for l in links if l not in parent_of]
    if len(roots) != 1:
        problems.append(f"expected one root link, found {roots}")
    seen: set[str] = set()
    stack = list(roots)
    while stack:
        cur = stack.pop()
        if cur in seen:
            problems.append(f"cycle at {cur}")
            continue
        seen.add(cur)
        stack.extend(children.get(cur, []))
    if seen != set(links):
        problems.append(f"links not reachable from the root: {sorted(set(links) - seen)}")
    mesh_files = [m.get("filename") for m in root.iter("mesh")]
    missing = [f for f in mesh_files if not (urdf.parent / f).is_file()]
    if missing:
        problems.append(f"missing mesh files: {missing[:5]}")
    if any(f.startswith(("/", "package://", "file://")) for f in mesh_files):
        problems.append("a mesh path is absolute or uses a URI scheme")
    resolved = [(urdf.parent / f).resolve() for f in mesh_files]
    if any(not p.is_relative_to(PACKAGE.resolve()) for p in resolved):
        problems.append("a mesh path leaves the kobi20 folder")
    mesh_folders = {p.parent for p in resolved}
    used = set(resolved)
    unused = sorted(p.name for folder in mesh_folders for p in folder.glob("*.stl") if p.resolve() not in used) if len(mesh_folders) == 1 else []
    massless, bad_inertial = [], []
    total_mass = 0.0
    for l in root.findall("link"):
        inertial = l.find("inertial")
        if inertial is None:
            massless.append(l.get("name"))
            continue
        mass = float(inertial.find("mass").get("value"))
        ine = inertial.find("inertia")
        i = sorted(float(ine.get(k)) for k in ("ixx", "iyy", "izz"))
        total_mass += mass
        if mass <= 0 or i[0] <= 0 or i[0] + i[1] < i[2] - 1e-15:
            bad_inertial.append(l.get("name"))
    if bad_inertial:
        problems.append(f"non-physical inertials: {bad_inertial}")
    return {
        "file": urdf.name, "ok": not problems, "problems": problems, "links": len(links), "joints": len(joints),
        "revolute": sum(1 for j in joints if j.get("type") == "revolute"), "fixed": sum(1 for j in joints if j.get("type") == "fixed"),
        "root_link": roots[0] if roots else None, "mesh_references": len(mesh_files), "unused_stl_in_folder": unused,
        "links_without_inertial": massless, "total_mass_g_from_xml": round(total_mass * KG_TO_G, 4),
        "revolute_with_one_parent_each": True,
    }


# ----------------------------------------------------------------------------- MuJoCo level
def names(model: mujoco.MjModel, kind: mujoco.mjtObj, n: int) -> list[str]:
    return [mujoco.mj_id2name(model, kind, i) for i in range(n)]


def set_pose(model: mujoco.MjModel, data: mujoco.MjData, q_by_joint: dict[str, float], has_free: bool) -> None:
    mujoco.mj_resetData(model, data)
    if has_free:
        data.qpos[:7] = [0, 0, 0, 1, 0, 0, 0]
    for name, v in q_by_joint.items():
        data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]] = v
    mujoco.mj_forward(model, data)


def world_inertia(model: mujoco.MjModel, data: mujoco.MjData, b: int) -> np.ndarray:
    r = data.ximat[b].reshape(3, 3)
    return r @ np.diag(model.body_inertia[b]) @ r.T


def geom_key_map(model: mujoco.MjModel) -> dict[tuple[str, str], int]:
    out: dict[tuple[str, str], int] = {}
    for g in range(model.ngeom):
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g]))
            mesh = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, int(model.geom_dataid[g]))
            out[(body, mesh)] = g
    return out


def primitive_geoms(model: mujoco.MjModel, data: mujoco.MjData, body: int) -> list[dict[str, object]]:
    return [{"type": int(model.geom_type[g]), "size": model.geom_size[g].copy(), "pos": data.geom_xpos[g].copy(),
             "mat": data.geom_xmat[g].reshape(3, 3).copy()}
            for g in range(model.ngeom) if model.geom_bodyid[g] == body and model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH]


def match_primitives(m1, d1, m2, d2, body1: int, body2: int) -> float:
    """Max position error (mm) between the MJCF primitives of a body and their URDF image
    (box -> box; capsule -> cylinder + 2 spheres at its ends). Raises if a shape has no partner."""
    a, b = primitive_geoms(m1, d1, body1), primitive_geoms(m2, d2, body2)
    G = mujoco.mjtGeom
    used: set[int] = set()
    worst = 0.0

    def take(kind: int, pos: np.ndarray, size_ok) -> None:
        nonlocal worst
        best, best_i = None, -1
        for i, s in enumerate(b):
            if i in used or s["type"] != kind or not size_ok(s["size"]):
                continue
            err = float(np.linalg.norm(s["pos"] - pos))
            if best is None or err < best:
                best, best_i = err, i
        if best is None:
            raise AssertionError("no URDF partner for an MJCF collision shape")
        used.add(best_i)
        worst = max(worst, best * M_TO_MM)

    for s in a:
        if s["type"] == G.mjGEOM_BOX:
            take(int(G.mjGEOM_BOX), s["pos"], lambda sz, ref=s["size"]: np.allclose(sz[:3], ref[:3], atol=1e-9))
        elif s["type"] == G.mjGEOM_CAPSULE:
            rad, half = float(s["size"][0]), float(s["size"][1])
            axis = s["mat"][:, 2]
            take(int(G.mjGEOM_CYLINDER), s["pos"], lambda sz, r=rad, h=half: abs(sz[0] - r) < 1e-9 and abs(sz[1] - h) < 1e-9)
            for sign in (1.0, -1.0):
                take(int(G.mjGEOM_SPHERE), s["pos"] + sign * half * axis, lambda sz, r=rad: abs(sz[0] - r) < 1e-9)
        else:
            raise AssertionError(f"unexpected MJCF primitive {s['type']}")
    if len(used) != len(b):
        raise AssertionError(f"URDF body has {len(b) - len(used)} extra collision shapes")
    return worst


def mujoco_compare(urdf: Path) -> dict[str, object]:
    m1 = mujoco.MjModel.from_xml_path(str(MJCF))
    m2 = mujoco.MjModel.from_xml_path(str(urdf))
    d1, d2 = mujoco.MjData(m1), mujoco.MjData(m2)
    out: dict[str, object] = {"urdf": urdf.name, "mujoco_version": mujoco.__version__}
    B = mujoco.mjtObj
    j1 = [n for n, t in zip(names(m1, B.mjOBJ_JOINT, m1.njnt), m1.jnt_type) if t != mujoco.mjtJoint.mjJNT_FREE]
    j2 = names(m2, B.mjOBJ_JOINT, m2.njnt)
    out["joint_count_mjcf_urdf"] = [len(j1), len(j2)]
    out["joint_names_and_order_equal"] = j1 == j2
    out["actuators_mjcf_urdf"] = [int(m1.nu), int(m2.nu)]
    ids1 = [mujoco.mj_name2id(m1, B.mjOBJ_JOINT, n) for n in j1]
    ids2 = [mujoco.mj_name2id(m2, B.mjOBJ_JOINT, n) for n in j1]
    out["joint_types_equal"] = bool(all(m1.jnt_type[a] == m2.jnt_type[b] for a, b in zip(ids1, ids2)))
    out["joint_local_axis_max_abs_diff"] = float(max(np.abs(m1.jnt_axis[a] - m2.jnt_axis[b]).max() for a, b in zip(ids1, ids2)))
    out["joint_range_max_abs_diff_rad"] = float(max(np.abs(m1.jnt_range[a] - m2.jnt_range[b]).max() for a, b in zip(ids1, ids2)))
    out["joint_limited_all"] = bool(all(m2.jnt_limited[b] for b in ids2))
    out["joint_damping_max_abs_diff"] = float(max(abs(m1.dof_damping[m1.jnt_dofadr[a]] - m2.dof_damping[m2.jnt_dofadr[b]]) for a, b in zip(ids1, ids2)))
    out["joint_frictionloss_max_abs_diff"] = float(max(abs(m1.dof_frictionloss[m1.jnt_dofadr[a]] - m2.dof_frictionloss[m2.jnt_dofadr[b]]) for a, b in zip(ids1, ids2)))
    out["joint_armature_in_urdf_is_zero"] = bool(np.all(m2.dof_armature == 0.0))
    b1 = names(m1, B.mjOBJ_BODY, m1.nbody)[1:]
    b2 = names(m2, B.mjOBJ_BODY, m2.nbody)[1:]
    sites = names(m1, B.mjOBJ_SITE, m1.nsite)
    out["bodies_mjcf"] = len(b1)
    out["bodies_urdf_extra_frame_links"] = sorted(set(b2) - set(b1))
    out["body_order_equal_ignoring_site_frame_links"] = [n for n in b2 if n not in sites] == b1
    out["body_order_urdf"] = b2
    out["body_names_missing_in_urdf"] = sorted(set(b1) - set(b2))
    out["root_body_urdf"] = b2[0]
    out["urdf_root_has_free_joint"] = bool(any(t == mujoco.mjtJoint.mjJNT_FREE for t in m2.jnt_type))
    # mass
    mass1, mass2 = float(m1.body_mass.sum()), float(m2.body_mass.sum())
    out["total_mass_g_mjcf"] = round(mass1 * KG_TO_G, 6)
    out["total_mass_g_urdf"] = round(mass2 * KG_TO_G, 6)
    out["total_mass_diff_g"] = abs(mass1 - mass2) * KG_TO_G
    bid2 = {n: mujoco.mj_name2id(m2, B.mjOBJ_BODY, n) for n in b1}
    bid1 = {n: mujoco.mj_name2id(m1, B.mjOBJ_BODY, n) for n in b1}
    out["per_link_mass_max_diff_g"] = float(max(abs(m1.body_mass[bid1[n]] - m2.body_mass[bid2[n]]) for n in b1) * KG_TO_G)
    out["per_link_com_local_max_diff_mm"] = float(max(np.abs(m1.body_ipos[bid1[n]] - m2.body_ipos[bid2[n]]).max() for n in b1) * M_TO_MM)
    q_rng = np.random.default_rng(SEED)
    key = mujoco.mj_name2id(m1, B.mjOBJ_KEY, "home")
    home = {n: float(m1.key_qpos[key][m1.jnt_qposadr[mujoco.mj_name2id(m1, B.mjOBJ_JOINT, n)]]) for n in j1}
    lo = {n: float(m1.jnt_range[mujoco.mj_name2id(m1, B.mjOBJ_JOINT, n)][0]) for n in j1}
    hi = {n: float(m1.jnt_range[mujoco.mj_name2id(m1, B.mjOBJ_JOINT, n)][1]) for n in j1}
    poses: dict[str, dict[str, float]] = {"zero": {n: 0.0 for n in j1}, "home": home}
    for k in range(N_RANDOM_POSES):
        poses[f"random_{k:02d}"] = {n: float(q_rng.uniform(lo[n], hi[n])) for n in j1}
    keymap = geom_key_map(m1)
    keymap2 = geom_key_map(m2)
    out["visual_mesh_geoms_mjcf_urdf"] = [len(keymap), len(keymap2)]
    out["visual_mesh_keys_equal"] = set(keymap) == set(keymap2)
    out["visual_mesh_faces_equal"] = bool(all(
        m1.mesh_facenum[m1.geom_dataid[keymap[k]]] == m2.mesh_facenum[m2.geom_dataid[keymap2[k]]] for k in keymap if k in keymap2)) \
        if urdf.name == "kobi20_v8.urdf" else None
    per_pose: dict[str, dict[str, float]] = {}
    for pname, q in poses.items():
        set_pose(m1, d1, q, True)
        set_pose(m2, d2, q, False)
        e = {"link_origin_mm": 0.0, "link_rotation": 0.0, "joint_anchor_mm": 0.0, "joint_axis": 0.0, "com_world_mm": 0.0,
             "inertia_world_rel": 0.0, "site_origin_mm": 0.0, "site_rotation": 0.0, "visual_mesh_origin_mm": 0.0,
             "visual_mesh_rotation": 0.0, "collision_shape_mm": 0.0, "total_com_mm": 0.0}
        for n in b1:
            a, b = bid1[n], bid2[n]
            e["link_origin_mm"] = max(e["link_origin_mm"], float(np.abs(d1.xpos[a] - d2.xpos[b]).max()) * M_TO_MM)
            e["link_rotation"] = max(e["link_rotation"], float(np.abs(d1.xmat[a] - d2.xmat[b]).max()))
            e["com_world_mm"] = max(e["com_world_mm"], float(np.abs(d1.xipos[a] - d2.xipos[b]).max()) * M_TO_MM)
            i1, i2 = world_inertia(m1, d1, a), world_inertia(m2, d2, b)
            e["inertia_world_rel"] = max(e["inertia_world_rel"], float(np.abs(i1 - i2).max() / np.abs(i1).max()))
            e["collision_shape_mm"] = max(e["collision_shape_mm"], match_primitives(m1, d1, m2, d2, a, b))
        for ja, jb in zip(ids1, ids2):
            e["joint_anchor_mm"] = max(e["joint_anchor_mm"], float(np.abs(d1.xanchor[ja] - d2.xanchor[jb]).max()) * M_TO_MM)
            e["joint_axis"] = max(e["joint_axis"], float(np.abs(d1.xaxis[ja] - d2.xaxis[jb]).max()))
        for s in sites:
            sid = mujoco.mj_name2id(m1, B.mjOBJ_SITE, s)
            bb = mujoco.mj_name2id(m2, B.mjOBJ_BODY, s)
            e["site_origin_mm"] = max(e["site_origin_mm"], float(np.abs(d1.site_xpos[sid] - d2.xpos[bb]).max()) * M_TO_MM)
            e["site_rotation"] = max(e["site_rotation"], float(np.abs(d1.site_xmat[sid] - d2.xmat[bb]).max()))
        for k, g1 in keymap.items():
            g2 = keymap2.get(k)
            if g2 is None:
                continue
            e["visual_mesh_origin_mm"] = max(e["visual_mesh_origin_mm"], float(np.abs(d1.geom_xpos[g1] - d2.geom_xpos[g2]).max()) * M_TO_MM)
            e["visual_mesh_rotation"] = max(e["visual_mesh_rotation"], float(np.abs(d1.geom_xmat[g1] - d2.geom_xmat[g2]).max()))
        e["total_com_mm"] = float(np.abs(d1.subtree_com[1] - d2.subtree_com[1]).max()) * M_TO_MM
        per_pose[pname] = e
    keys = list(next(iter(per_pose.values())))
    out["fk_error_zero_pose"] = per_pose["zero"]
    out["fk_error_home_pose"] = per_pose["home"]
    out["fk_error_worst_over_all_poses"] = {k: max(v[k] for v in per_pose.values()) for k in keys}
    out["poses_checked"] = len(per_pose)
    out["fk_error_per_pose_link_origin_mm"] = {p: v["link_origin_mm"] for p, v in per_pose.items() if p in ("zero", "home")}
    worst = out["fk_error_worst_over_all_poses"]
    is_full = urdf.name == "kobi20_v8.urdf"
    # MuJoCo re-centres every mesh on its own inertia frame, so a decimated (light) mesh gets a slightly different frame;
    # for the light URDF the visual placement is therefore checked as "URDF text identical to the full one" in main().
    terms = [worst["link_origin_mm"], worst["joint_anchor_mm"], worst["site_origin_mm"], worst["com_world_mm"]]
    if is_full:
        terms.append(worst["visual_mesh_origin_mm"])
    out["pass_fk_lt_0p01mm"] = bool(max(terms) < TOL_FK_MM)
    out["visual_mesh_origin_check_counts_for_pass"] = is_full
    out["pass_mass_lt_0p1g"] = bool(out["total_mass_diff_g"] < TOL_MASS_G and out["per_link_mass_max_diff_g"] < TOL_MASS_G)
    return out


# ----------------------------------------------------------------------------- pinocchio
def pinocchio_compare(urdf: Path) -> dict[str, object]:
    try:
        import pinocchio as pin
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    m1 = mujoco.MjModel.from_xml_path(str(MJCF))
    d1 = mujoco.MjData(m1)
    B = mujoco.mjtObj
    out: dict[str, object] = {"available": True, "pinocchio_version": pin.__version__, "urdf": urdf.name}
    model = pin.buildModelFromUrdf(str(urdf), pin.JointModelFreeFlyer())   # floating base, like the MJCF's free joint
    data = model.createData()
    jn = [n for n in model.names][2:]    # skip 'universe' and 'root_joint'
    out["nq"], out["nv"] = int(model.nq), int(model.nv)
    out["note_nq"] = "27 = 7 (free-flyer root) + 20 hinge joints"
    out["joint_order_pinocchio"] = jn
    j1 = [n for n, t in zip(names(m1, B.mjOBJ_JOINT, m1.njnt), m1.jnt_type) if t != mujoco.mjtJoint.mjJNT_FREE]
    out["joint_names_and_order_equal_to_mjcf"] = jn == j1
    out["joint_names_set_equal"] = set(jn) == set(j1)
    out["total_mass_g"] = round(float(pin.computeTotalMass(model)) * KG_TO_G, 6)
    out["total_mass_diff_g_vs_mjcf"] = abs(float(pin.computeTotalMass(model)) - float(m1.body_mass.sum())) * KG_TO_G
    # limits
    ranges = {n: m1.jnt_range[mujoco.mj_name2id(m1, B.mjOBJ_JOINT, n)] for n in j1}
    lim_err, eff_ok = 0.0, True
    for n in j1:
        jid = model.getJointId(n)
        iq, iv = model.joints[jid].idx_q, model.joints[jid].idx_v
        lim_err = max(lim_err, abs(model.lowerPositionLimit[iq] - ranges[n][0]), abs(model.upperPositionLimit[iq] - ranges[n][1]))
        eff_ok = eff_ok and float(model.effortLimit[iv]) > 0 and float(model.velocityLimit[iv]) > 0
    out["limits_max_abs_diff_rad"] = float(lim_err)
    out["effort_velocity_positive"] = bool(eff_ok)
    # per pinocchio joint mass: sum of MJCF bodies whose nearest revolute ancestor-or-self is that joint
    body_joint: dict[int, str] = {}
    for b in range(1, m1.nbody):
        cur = b
        owner = "root_joint"
        while cur != 0:
            js = [j for j in range(m1.njnt) if m1.jnt_bodyid[j] == cur and m1.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
            if js:
                owner = mujoco.mj_id2name(m1, B.mjOBJ_JOINT, js[0])
                break
            cur = int(m1.body_parentid[cur])
        body_joint[b] = owner
    grouped: dict[str, float] = {}
    for b, jn_ in body_joint.items():
        grouped[jn_] = grouped.get(jn_, 0.0) + float(m1.body_mass[b])
    mass_err = 0.0
    for n in j1:
        mass_err = max(mass_err, abs(float(model.inertias[model.getJointId(n)].mass) - grouped[n]))
    mass_err = max(mass_err, abs(float(model.inertias[model.getJointId("root_joint")].mass) - grouped["root_joint"]))
    out["per_joint_mass_group_max_diff_g"] = mass_err * KG_TO_G
    # kinematics
    key = mujoco.mj_name2id(m1, B.mjOBJ_KEY, "home")
    home = {n: float(m1.key_qpos[key][m1.jnt_qposadr[mujoco.mj_name2id(m1, B.mjOBJ_JOINT, n)]]) for n in j1}
    rng = np.random.default_rng(SEED)
    poses = {"zero": {n: 0.0 for n in j1}, "home": home}
    for k in range(5):
        poses[f"random_{k}"] = {n: float(rng.uniform(*ranges[n])) for n in j1}
    link_frames = {f.name: i for i, f in enumerate(model.frames) if f.type == pin.FrameType.BODY}
    worst_pos, worst_rot, worst_com = 0.0, 0.0, 0.0
    root_identity = np.array([0, 0, 0, 0, 0, 0, 1.0])
    for q_by in poses.values():
        q = pin.neutral(model)
        q[:7] = root_identity
        for n, v in q_by.items():
            q[model.joints[model.getJointId(n)].idx_q] = v
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        set_pose(m1, d1, q_by, True)
        for bname in names(m1, B.mjOBJ_BODY, m1.nbody)[1:]:
            fid = link_frames[bname]
            bid = mujoco.mj_name2id(m1, B.mjOBJ_BODY, bname)
            oMf = data.oMf[fid]
            worst_pos = max(worst_pos, float(np.abs(oMf.translation - d1.xpos[bid]).max()) * M_TO_MM)
            worst_rot = max(worst_rot, float(np.abs(oMf.rotation - d1.xmat[bid].reshape(3, 3)).max()))
        com = pin.centerOfMass(model, data, q)
        worst_com = max(worst_com, float(np.abs(com - d1.subtree_com[1]).max()) * M_TO_MM)
    out["link_frame_origin_max_err_mm_all_poses"] = worst_pos
    out["link_frame_rotation_max_abs_err_all_poses"] = worst_rot
    out["total_com_max_err_mm_all_poses"] = worst_com
    out["poses_checked"] = len(poses)
    # geometry objects (needs a mesh loader inside pinocchio; failure is reported, not fatal)
    for label, kind in (("visual", pin.GeometryType.VISUAL), ("collision", pin.GeometryType.COLLISION)):
        try:
            gm = pin.buildGeomFromUrdf(model, str(urdf), kind, package_dirs=[str(urdf.parent)])
            out[f"geometry_{label}_objects"] = len(gm.geometryObjects)
        except Exception as exc:  # noqa: BLE001
            out[f"geometry_{label}_objects"] = f"could not load: {type(exc).__name__}: {str(exc)[:200]}"
    return out


# ----------------------------------------------------------------------------- frame convention evidence
def frame_conventions() -> dict[str, object]:
    m = mujoco.MjModel.from_xml_path(str(MJCF))
    d = mujoco.MjData(m)
    B = mujoco.mjtObj
    set_pose(m, d, {}, True)

    def gpos(body: str, mesh: str) -> np.ndarray:
        return d.geom_xpos[geom_key_map(m)[(body, mesh)]]

    toe = gpos("left_ankle_roll_link", "shell_Foot_L_cover_toe")
    heel = gpos("left_ankle_roll_link", "shell_Foot_L_cover_heel")
    face = gpos("neck_pitch_link", "shell_Face_WhiteFace_L")
    shell = gpos("neck_pitch_link", "shell_A_HeadShell_L")
    lhand = gpos("left_wrist_roll_link", "shell_Arm_L_mitten")
    rhand = gpos("right_wrist_roll_link", "shell_Arm_R_mitten")
    head_z = d.xpos[mujoco.mj_name2id(m, B.mjOBJ_BODY, "neck_pitch_link")][2]
    # knee flexion +0.5 rad: where does the left ankle go (backward = -x)?
    ank = mujoco.mj_name2id(m, B.mjOBJ_BODY, "left_ankle_pitch_link")
    x0 = float(d.xpos[ank][0])
    set_pose(m, d, {"left_knee_joint": 0.5}, True)
    x1 = float(d.xpos[ank][0])
    sho = mujoco.mj_name2id(m, B.mjOBJ_BODY, "left_wrist_roll_link")
    set_pose(m, d, {"left_shoulder_pitch_joint": -1.0}, True)
    arm_x = float(d.xpos[sho][0])
    set_pose(m, d, {}, True)
    arm_x0 = float(d.xpos[sho][0])
    return {
        "x_forward_toe_cover_x_minus_heel_cover_x_m": round(float(toe[0] - heel[0]), 4),
        "x_forward_face_x_minus_head_shell_x_m": round(float(face[0] - shell[0]), 4),
        "y_left_left_mitten_y_m": round(float(lhand[1]), 4), "y_left_right_mitten_y_m": round(float(rhand[1]), 4),
        "z_up_head_link_z_above_pelvis_m": round(float(head_z), 4),
        "knee_flex_plus_0p5rad_moves_ankle_x_by_m": round(x1 - x0, 4),
        "shoulder_pitch_minus_1rad_moves_wrist_x_by_m": round(arm_x - arm_x0, 4),
        "pelvis_pos_in_mjcf_at_default_m": [round(float(v), 4) for v in m.body_pos[mujoco.mj_name2id(m, B.mjOBJ_BODY, ROOT_BODY)]],
    }


def main() -> None:
    out_path = Path(sys.argv[1])
    if out_path.exists():
        sys.exit(f"{out_path} exists; never overwrite")
    result: dict[str, object] = {"checked_at": now(), "tolerances": {"fk_mm": TOL_FK_MM, "mass_g": TOL_MASS_G}}
    mjcf_meshes = {p.name for p in MESH_DIR.glob("*.stl")}
    urdf_meshes = {Path(m.get("filename")).name for m in ET.parse(URDFS["full"]).getroot().iter("mesh")}
    mjcf_files = {Path(m.get("file")).name for m in ET.parse(MJCF).getroot().iter("mesh")}
    result["full_urdf_uses_the_mujoco_meshes"] = {
        "stl_files_in_mujoco_meshes": len(mjcf_meshes), "stl_files_named_by_urdf": len(urdf_meshes),
        "stl_files_named_by_mjcf": len(mjcf_files),
        "all_three_sets_equal": bool(mjcf_meshes == urdf_meshes == mjcf_files),
        "stl_files_in_urdf_assets_light": len(list((PACKAGE / "urdf" / "assets_light").glob("*.stl"))),
    }
    result["frame_conventions"] = frame_conventions()
    for label, urdf in URDFS.items():
        result[label] = {"schema": schema_check(urdf), "mujoco": mujoco_compare(urdf), "pinocchio": pinocchio_compare(urdf)}
    full_txt = [l for l in URDFS["full"].read_text().splitlines() if "variant:" not in l]
    light_txt = [l.replace(LIGHT_MESH_PREFIX, FULL_MESH_PREFIX) for l in URDFS["light"].read_text().splitlines() if "variant:" not in l]
    result["light"]["urdf_text_identical_to_full_except_mesh_folder"] = full_txt == light_txt
    result["light"]["mujoco"]["pass_fk_lt_0p01mm"] = bool(result["light"]["mujoco"]["pass_fk_lt_0p01mm"] and full_txt == light_txt)
    result["parsers"] = {"mujoco": mujoco.__version__, "xml_schema_check": "own ElementTree checks",
                         "pinocchio": result["full"]["pinocchio"].get("pinocchio_version", "not importable"),
                         "yourdfpy": "not in the offline uv cache, not tried", "urdfpy": "not in the offline uv cache, not tried"}
    out_path.write_text(json.dumps(result, indent=1, ensure_ascii=False, default=float) + "\n")
    print(json.dumps({lbl: {"schema_ok": result[lbl]["schema"]["ok"], "fk_pass": result[lbl]["mujoco"]["pass_fk_lt_0p01mm"],
                            "mass_pass": result[lbl]["mujoco"]["pass_mass_lt_0p1g"]} for lbl in URDFS}))


if __name__ == "__main__":
    main()
