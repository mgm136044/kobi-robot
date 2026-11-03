"""Build a URDF of Kobi from the generated MJCF (kobi_scene.xml), for placo / pinocchio tools.

Source of truth is the MJCF:
  - links and joint frames come from the compiled model (body_pos / body_quat, every joint sits at its body origin),
  - mass, centre of mass and principal inertia come from the compiled model (what MuJoCo simulates),
  - visual meshes come from the uncompiled spec (geom pose as written, before MuJoCo re-centres meshes),
  - MJCF sites become fixed frame links (imu, trunk, left_foot, right_foot, head).
The massless MJCF body `base` (it only carries the free joint) is folded into the root link; this is exact because
`trunk_assembly` sits at the identity inside `base`.
Collision: only the two TPU sole meshes. The other MJCF colliders are capsules and ellipsoids that URDF cannot express.

Run:  uv run --no-project --python 3.12 --with mujoco==3.14.0 --with numpy python -I tools/build_urdf.py
Output: printed_face/urdf/kobi.urdf and printed_face/urdf/assets/*.stl (the layout of a reference-generator robot folder), and
printed_face/urdf/kobi_gait.urdf: the same robot with the neck and shoulder joints fixed at 0 (their home angle) and without
meshes, so the gait generator cannot swing the head or arms to move the centre of mass and the head and arm mass
stays where it is at the home pose.
"""

from __future__ import annotations

import copy
import math
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
MJCF_PATH = HERE.parent / "printed_face" / "mujoco" / "kobi_scene.xml"
OUT_DIR = HERE.parent / "printed_face" / "urdf"
URDF_NAME = "kobi.urdf"
GAIT_URDF_NAME = "kobi_gait.urdf"        # for the reference gait generator (placo): upper body fixed, no meshes
GAIT_FIXED_JOINTS = ("neck_yaw", "neck_pitch", "shoulder_left", "shoulder_right")   # held at 0 = home
MESH_SUBDIR = "assets"
MUJOCO_MESHDIR = "."                    # mesh filenames already start with MESH_SUBDIR/, relative to the URDF
ROBOT_NAME = "kobi"

SKIP_BODIES = {"world", "floor"}
FREE_BASE_BODY = "base"                 # massless MJCF wrapper of the free joint, folded into the root link
ROOT_BODY = "trunk_assembly"
COLLISION_GEOMS = {"left_foot_bottom_tpu", "right_foot_bottom_tpu"}

STS3215_EFFORT_NM = 1.91                # datasheet stall torque at 7.4 V; user chose it as the training limit (2026-11-02)
MG90S_JOINTS = {"shoulder_left", "shoulder_right"}   # these keep the MJCF actuator limit (0.20 N*m, source unconfirmed)
JOINT_VELOCITY_RAD_S = 5.24             # motor-target speed limit of the Open Duck Playground env
POS_TOL_M = 1e-12
QUAT_TOL = 1e-9
GIMBAL_EPS = 1e-9
FMT = "{:.9g}"


def fmt(values: np.ndarray | list[float]) -> str:
    return " ".join(FMT.format(float(v)) for v in values)


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, dtype=float))
    return m.reshape(3, 3)


def mat_to_rpy(r: np.ndarray) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    if abs(r[2, 0]) < 1.0 - GIMBAL_EPS:
        pitch = math.asin(-r[2, 0])
        roll = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(r[1, 0], r[0, 0])
    else:
        pitch = math.copysign(math.pi / 2.0, -r[2, 0])
        roll = 0.0
        yaw = math.atan2(-r[0, 1], r[1, 1])
    return np.array([roll, pitch, yaw])


def origin(parent: ET.Element, pos: np.ndarray, quat: np.ndarray) -> None:
    ET.SubElement(parent, "origin", xyz=fmt(pos), rpy=fmt(mat_to_rpy(quat_to_mat(quat))))


def name_of(model: mujoco.MjModel, kind: mujoco.mjtObj, i: int) -> str:
    name = mujoco.mj_id2name(model, kind, i)
    if not name:
        raise ValueError(f"unnamed {kind} {i}: every element must be named to map it into the URDF")
    return name


def link_name(model: mujoco.MjModel, body: int) -> str:
    name = name_of(model, mujoco.mjtObj.mjOBJ_BODY, body)
    return ROOT_BODY if name == FREE_BASE_BODY else name


def check_base_fold(model: mujoco.MjModel) -> None:
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, FREE_BASE_BODY)
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ROOT_BODY)
    if model.body_parentid[root] != base:
        raise ValueError(f"{ROOT_BODY} must be the child of {FREE_BASE_BODY}")
    if np.abs(model.body_pos[root]).max() > POS_TOL_M or np.abs(model.body_quat[root] - [1, 0, 0, 0]).max() > QUAT_TOL:
        raise ValueError(f"{ROOT_BODY} is not at the identity of {FREE_BASE_BODY}; folding would move it")
    if model.body_mass[base] != 0.0:
        raise ValueError(f"{FREE_BASE_BODY} has mass; folding it would lose that mass")


def add_inertial(link: ET.Element, model: mujoco.MjModel, body: int) -> None:
    inertial = ET.SubElement(link, "inertial")
    origin(inertial, model.body_ipos[body], model.body_iquat[body])
    ET.SubElement(inertial, "mass", value=FMT.format(float(model.body_mass[body])))
    i1, i2, i3 = (float(v) for v in model.body_inertia[body])
    ET.SubElement(inertial, "inertia", ixx=FMT.format(i1), iyy=FMT.format(i2), izz=FMT.format(i3),
                  ixy="0", ixz="0", iyz="0")


def geom_rgba(spec: mujoco.MjSpec, geom: mujoco.MjsGeom) -> np.ndarray:
    if geom.material:
        return np.asarray(spec.material(geom.material).rgba, dtype=float)
    return np.asarray(geom.rgba, dtype=float)


def add_meshes(link: ET.Element, spec: mujoco.MjSpec, sbody: mujoco.MjsBody, used: dict[str, Path]) -> None:
    mesh_root = MJCF_PATH.parent / spec.meshdir
    for geom in sbody.geoms:
        if geom.type != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh = spec.mesh(geom.meshname)
        if np.abs(np.asarray(mesh.refpos)).max() > 0.0 or np.abs(np.asarray(mesh.refquat) - [1, 0, 0, 0]).max() > 0.0:
            raise ValueError(f"mesh {mesh.name} uses refpos/refquat, which this builder does not convert")
        used[mesh.file] = mesh_root / mesh.file
        filename = f"{MESH_SUBDIR}/{mesh.file}"
        tag = "collision" if geom.name in COLLISION_GEOMS else "visual"
        for kind in ("visual", "collision") if tag == "collision" else ("visual",):
            base_name = geom.name or f"{sbody.name}_{mesh.name}"
            el = ET.SubElement(link, kind, name=base_name if kind == "visual" else f"{base_name}_collision")
            origin(el, np.asarray(geom.pos), np.asarray(geom.quat))
            g = ET.SubElement(el, "geometry")
            ET.SubElement(g, "mesh", filename=filename, scale=fmt(mesh.scale))
            if kind == "visual":
                mat = ET.SubElement(el, "material", name=f"{el.get('name')}_material")
                ET.SubElement(mat, "color", rgba=fmt(geom_rgba(spec, geom)))


def joint_effort(model: mujoco.MjModel, jname: str) -> float:
    if jname not in MG90S_JOINTS:
        return STS3215_EFFORT_NM
    act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, jname)
    return float(model.actuator_forcerange[act, 1])


def add_joint(robot: ET.Element, model: mujoco.MjModel, body: int) -> None:
    joints = [j for j in range(model.njnt) if model.jnt_bodyid[j] == body]
    parent = link_name(model, int(model.body_parentid[body]))
    child = link_name(model, body)
    if not joints:
        j = ET.SubElement(robot, "joint", name=f"{child}_fixed", type="fixed")
    else:
        if len(joints) != 1 or model.jnt_type[joints[0]] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"body {child}: only one hinge joint per body is supported")
        jid = joints[0]
        if np.abs(model.jnt_pos[jid]).max() > POS_TOL_M:
            raise ValueError(f"joint of {child} is not at the body origin")
        jname = name_of(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        j = ET.SubElement(robot, "joint", name=jname, type="revolute")
        ET.SubElement(j, "axis", xyz=fmt(model.jnt_axis[jid]))
        lo, hi = model.jnt_range[jid]
        ET.SubElement(j, "limit", lower=FMT.format(lo), upper=FMT.format(hi),
                      effort=FMT.format(joint_effort(model, jname)), velocity=FMT.format(JOINT_VELOCITY_RAD_S))
    origin(j, model.body_pos[body], model.body_quat[body])
    ET.SubElement(j, "parent", link=parent)
    ET.SubElement(j, "child", link=child)


def add_site_frames(robot: ET.Element, model: mujoco.MjModel) -> None:
    for s in range(model.nsite):
        sname = name_of(model, mujoco.mjtObj.mjOBJ_SITE, s)
        ET.SubElement(robot, "link", name=sname)
        j = ET.SubElement(robot, "joint", name=f"{sname}_frame", type="fixed")
        origin(j, model.site_pos[s], model.site_quat[s])
        ET.SubElement(j, "parent", link=link_name(model, int(model.site_bodyid[s])))
        ET.SubElement(j, "child", link=sname)


def gait_variant(robot: ET.Element) -> ET.Element:
    """Copy with GAIT_FIXED_JOINTS turned into fixed joints and every visual and collision removed."""
    out = copy.deepcopy(robot)
    for joint in out.findall("joint"):
        if joint.get("name") in GAIT_FIXED_JOINTS:
            joint.set("type", "fixed")
            for child in [c for c in joint if c.tag in ("axis", "limit")]:
                joint.remove(child)
    for link in out.findall("link"):
        for child in [c for c in link if c.tag in ("visual", "collision")]:
            link.remove(child)
    return out


def write_urdf(robot: ET.Element, path: Path) -> None:
    ET.indent(robot)
    path.write_text('<?xml version="1.0"?>\n<!-- generated by tools/build_urdf.py from printed_face/mujoco/kobi_scene.xml; '
                    'do not edit by hand -->\n' + ET.tostring(robot, encoding="unicode") + "\n")


def build() -> Path:
    model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    spec = mujoco.MjSpec.from_file(str(MJCF_PATH))
    check_base_fold(model)
    robot = ET.Element("robot", name=ROBOT_NAME)
    mj = ET.SubElement(robot, "mujoco")   # lets MuJoCo load this file for the round-trip check; other parsers ignore it
    ET.SubElement(mj, "compiler", meshdir=MUJOCO_MESHDIR, discardvisual="false", fusestatic="false", strippath="false")
    used: dict[str, Path] = {}
    for body in range(model.nbody):
        bname = name_of(model, mujoco.mjtObj.mjOBJ_BODY, body)
        if bname in SKIP_BODIES or bname == FREE_BASE_BODY:
            continue
        link = ET.SubElement(robot, "link", name=bname)
        add_inertial(link, model, body)
        add_meshes(link, spec, spec.body(bname), used)
    for body in range(model.nbody):
        bname = name_of(model, mujoco.mjtObj.mjOBJ_BODY, body)
        if bname in SKIP_BODIES or bname in (FREE_BASE_BODY, ROOT_BODY):
            continue
        add_joint(robot, model, body)
    add_site_frames(robot, model)
    mesh_dir = OUT_DIR / MESH_SUBDIR
    mesh_dir.mkdir(parents=True, exist_ok=True)
    for file, src in sorted(used.items()):
        shutil.copy2(src, mesh_dir / file)
    out = OUT_DIR / URDF_NAME
    write_urdf(robot, out)
    gait = gait_variant(robot)
    write_urdf(gait, OUT_DIR / GAIT_URDF_NAME)
    moving = sum(1 for j in gait.findall("joint") if j.get("type") == "revolute")
    print(f"URDF {out} links={len(robot.findall('link'))} joints={len(robot.findall('joint'))} meshes={len(used)}")
    print(f"GAITURDF {OUT_DIR / GAIT_URDF_NAME} revolute={moving} fixed_upper={len(GAIT_FIXED_JOINTS)}")
    return out


if __name__ == "__main__":
    build()
    sys.exit(0)
