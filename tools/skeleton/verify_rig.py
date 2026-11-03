"""Verify the Kobi rig files against MuJoCo forward kinematics (runs in headless Blender).

    /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup --python tools/skeleton/verify_rig.py -- \
        --skeleton printed_face/skeleton/skeleton.json --fk-reference fk_reference.json \
        --rig-dir printed_face/skeleton --work-dir out/skeleton_verify

fk_reference.json comes from: extract_skeleton.py --fk-reference (MuJoCo mj_kinematics for 20 random in-range
joint configurations, plus "home" and "rest").

Checks, for the original .blend and for fresh re-imports of the FBX, GLB and BVH
  * bone count, names and parent hierarchy equal skeleton.json;
  * for every random configuration the pose is set by rotating each bone about its local hinge axis by the joint angle
    (.blend: the local Z axis, as promised; re-imports: the local axis derived from the imported rest pose, which is also
    reported), then every bone head in world space is compared to the MuJoCo joint anchor (tolerance 1e-4 m);
  * the same for the actions "rest" and "home" stored in the files;
  * skinned mesh vertices follow the MuJoCo body motion (rigid skinning) and every vertex has exactly one bone weight 1.0;
  * the GLB is also parsed without Blender (JSON + binary): Y-up node tree, hinge axis in the node frame, and the animation
    keyframes, compared with MuJoCo after the explicit axis change (x, y, z) -> (x, z, -y).
BVH has no bone roll, so only head positions are compared (delivered 2-frame file plus a 22-frame test file).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import bpy
import numpy as np
from mathutils import Quaternion, Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
import raw_formats  # noqa: E402  (same folder; Blender does not add it to sys.path)

TOLERANCE_M = 1e-4
LOCAL_Z = Vector((0.0, 0.0, 1.0))
BVH_TEST_NAME = "kobi_verify_test.bvh"
WEIGHT_TOLERANCE = 1e-5


def parse_args() -> argparse.Namespace:
    """Parse the arguments after '--'."""
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--skeleton", type=Path, required=True)
    parser.add_argument("--fk-reference", type=Path, required=True)
    parser.add_argument("--rig-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True, help="empty scratch folder for the 22-frame test BVH")
    parser.add_argument("--report", type=Path, default=None, help="optional JSON report path")
    return parser.parse_args(argv)


# ------------------------------------------------------------------ scene helpers

def fresh_scene() -> None:
    """Empty Blender scene (nothing carried over between imports)."""
    bpy.ops.wm.read_factory_settings(use_empty=True)


def find_objects(kind: str) -> list[bpy.types.Object]:
    """All objects of one type in the current file."""
    return [o for o in bpy.data.objects if o.type == kind]


def bone_parents(arm_obj: bpy.types.Object) -> dict[str, str | None]:
    """Bone name -> parent bone name."""
    return {b.name: (b.parent.name if b.parent else None) for b in arm_obj.data.bones}


def expected_parents(skeleton: dict[str, Any]) -> dict[str, str | None]:
    """Bone name -> parent bone name from skeleton.json."""
    table: dict[str, str | None] = {skeleton["root_bone"]: None}
    table.update({j["name"]: j["parent_bone"] for j in skeleton["joints"]})
    return table


def structure_report(parents: dict[str, str | None], skeleton: dict[str, Any]) -> dict[str, Any]:
    """Compare bone count, names and hierarchy with the expectation."""
    expected = expected_parents(skeleton)
    return {
        "bone_count": len(parents),
        "count_ok": len(parents) == len(expected),
        "names_ok": set(parents) == set(expected),
        "hierarchy_ok": parents == expected,
    }


def clear_animation(arm_obj: bpy.types.Object) -> None:
    """Detach any action so manual pose edits are not overwritten on frame change."""
    if arm_obj.animation_data is not None:
        arm_obj.animation_data.action = None


def assign_action(arm_obj: bpy.types.Object, suffix: str) -> bool:
    """Assign the action whose name ends with `suffix` (importers may prefix the armature name)."""
    candidates = [a for a in bpy.data.actions if a.name == suffix or a.name.endswith(suffix) or a.name.endswith("|" + suffix)]
    if not candidates:
        return False
    if arm_obj.animation_data is None:
        arm_obj.animation_data_create()
    action = candidates[0]
    arm_obj.animation_data.action = action
    slots = getattr(action, "slots", None)
    if slots is not None and len(slots) > 0 and getattr(arm_obj.animation_data, "action_slot", None) is None:
        arm_obj.animation_data.action_slot = slots[0]
    return True


def world_rotation(arm_obj: bpy.types.Object):
    """Rotation part of the armature object's world matrix."""
    return arm_obj.matrix_world.to_3x3().normalized()


def local_axes_from_rest(arm_obj: bpy.types.Object, skeleton: dict[str, Any]) -> dict[str, Vector]:
    """Hinge axis expressed in each bone's own rest frame (what the file really stores)."""
    rot_inv = world_rotation(arm_obj).inverted()
    axes: dict[str, Vector] = {}
    for joint in skeleton["joints"]:
        bone = arm_obj.data.bones[joint["name"]]
        axis_arm = rot_inv @ Vector(joint["axis_world"])
        axes[joint["name"]] = (bone.matrix_local.to_3x3().normalized().transposed() @ axis_arm).normalized()
    return axes


def apply_case(arm_obj: bpy.types.Object, case: dict[str, Any], axes: dict[str, Vector], euler_z: bool) -> None:
    """Pose every hinge bone by its joint angle about the local hinge axis; root stays at rest."""
    for pose_bone in arm_obj.pose.bones:
        pose_bone.location = (0.0, 0.0, 0.0)
        pose_bone.rotation_mode = "XYZ" if euler_z else "QUATERNION"
        pose_bone.rotation_euler = (0.0, 0.0, 0.0)
        pose_bone.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
    for name, angle in case["angles"].items():
        pose_bone = arm_obj.pose.bones[name]
        if euler_z:
            pose_bone.rotation_euler = (0.0, 0.0, angle)
        else:
            pose_bone.rotation_quaternion = Quaternion(axes[name], angle)
    bpy.context.view_layer.update()


def head_errors(arm_obj: bpy.types.Object, case: dict[str, Any], axes: dict[str, Vector] | None) -> tuple[float, float]:
    """Max bone-head position error (m) and max hinge-axis error (unit vector) against the MuJoCo case."""
    matrix_world = arm_obj.matrix_world
    rot = world_rotation(arm_obj)
    pos_err, axis_err = 0.0, 0.0
    for pose_bone in arm_obj.pose.bones:
        name = pose_bone.name
        target = case["anchors"].get(name, case["root_position"])
        pos_err = max(pos_err, float((matrix_world @ pose_bone.head - Vector(target)).length))
        if axes is not None and name in case["axes"]:
            world_axis = rot @ (pose_bone.matrix.to_3x3().normalized() @ axes[name])
            axis_err = max(axis_err, float((world_axis - Vector(case["axes"][name])).length))
    return pos_err, axis_err


# ------------------------------------------------------------------ mesh checks

def vertex_bone_names(mesh_obj: bpy.types.Object, strict: bool) -> tuple[list[str], int]:
    """Bone name per vertex and the number of vertices that do not have exactly one weight-1.0 bone."""
    names = [g.name for g in mesh_obj.vertex_groups]
    result: list[str] = []
    bad = 0
    for vertex in mesh_obj.data.vertices:
        groups = [g for g in vertex.groups if g.weight > WEIGHT_TOLERANCE] if not strict else list(vertex.groups)
        ok = len(groups) == 1 and abs(groups[0].weight - 1.0) <= (0.0 if strict else WEIGHT_TOLERANCE)
        bad += 0 if ok else 1
        result.append(names[groups[0].group] if groups else "")
    return result, bad


def evaluated_world_coords(mesh_obj: bpy.types.Object) -> np.ndarray:
    """World-space vertex positions after the armature deformation."""
    evaluated = mesh_obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    coords = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", coords)
    evaluated.to_mesh_clear()
    matrix = np.array(mesh_obj.matrix_world)
    return coords.reshape(-1, 3).astype(np.float64) @ matrix[:3, :3].T + matrix[:3, 3]


def skin_error(mesh_obj: bpy.types.Object, bones: list[str], rest_world: np.ndarray, case: dict[str, Any],
               skeleton: dict[str, Any], bone_body: dict[str, str]) -> float:
    """Max distance between the deformed vertices and the MuJoCo rigid-body motion of their body."""
    posed = evaluated_world_coords(mesh_obj)
    bodies0 = {b["name"]: b["qpos0"] for b in skeleton["bodies"]}
    names = np.array(bones)
    worst = 0.0
    for bone in sorted(set(bones)):
        if not bone:
            continue
        body = bone_body[bone]
        q_pos, q_mat = np.array(bodies0[body]["position"]), np.array(bodies0[body]["matrix"])
        h_pos, h_mat = np.array(case["bodies"][body]["position"]), np.array(case["bodies"][body]["matrix"])
        rot = h_mat @ q_mat.T
        index = np.nonzero(names == bone)[0]
        expected = h_pos + (rest_world[index] - q_pos) @ rot.T
        worst = max(worst, float(np.abs(posed[index] - expected).max()))
    return worst


# ------------------------------------------------------------------ per-format armature check

def check_armature_file(label: str, skeleton: dict[str, Any], fk: dict[str, Any], euler_z: bool,
                        mesh_check: bool) -> dict[str, Any]:
    """Run all Blender-side checks on the armature (and mesh) that is currently loaded."""
    arms = find_objects("ARMATURE")
    report: dict[str, Any] = {"label": label, "armature_objects": len(arms)}
    if len(arms) != 1:
        report["error"] = "expected exactly one armature object"
        return report
    arm_obj = arms[0]
    report.update(structure_report(bone_parents(arm_obj), skeleton))
    axes = local_axes_from_rest(arm_obj, skeleton)
    deviation = max((v - LOCAL_Z).length for v in axes.values())
    deviation_neg = max((v + LOCAL_Z).length for v in axes.values())
    report["local_axis_is_z_max_deviation"] = float(min(deviation, deviation_neg)) if (deviation < 1e-3 or deviation_neg < 1e-3) \
        else float(deviation)
    report["local_axes"] = {name: [round(c, 4) for c in vec] for name, vec in axes.items()}
    if euler_z:
        axes = {name: LOCAL_Z.copy() for name in axes}

    bone_body = {j["name"]: j["child_body"] for j in skeleton["joints"]}
    bone_body[skeleton["root_bone"]] = skeleton["root_body"]
    meshes = [o for o in find_objects("MESH") if o.vertex_groups]
    mesh_info: list[tuple[bpy.types.Object, list[str], int]] = []
    if mesh_check:
        for mesh_obj in meshes:
            names, bad = vertex_bone_names(mesh_obj, strict=euler_z)
            mesh_info.append((mesh_obj, names, bad))
        report["mesh_objects"] = len(meshes)
        report["vertices"] = sum(len(m.data.vertices) for m in meshes)
        report["vertices_not_single_bone_weight_1"] = sum(item[2] for item in mesh_info)

    clear_animation(arm_obj)
    apply_case(arm_obj, {"angles": {n: 0.0 for n in axes}}, axes, euler_z)
    rest_world = [evaluated_world_coords(m) for m, _, _ in mesh_info]

    cases = list(fk["random_cases"])
    pos_max, axis_max, skin_max = 0.0, 0.0, 0.0
    for case in cases:
        apply_case(arm_obj, case, axes, euler_z)
        pos, axis = head_errors(arm_obj, case, axes)
        pos_max, axis_max = max(pos_max, pos), max(axis_max, axis)
        for (mesh_obj, names, _), rest in zip(mesh_info, rest_world, strict=True):
            skin_max = max(skin_max, skin_error(mesh_obj, names, rest, case, skeleton, bone_body))
    report["random_cases"] = len(cases)
    report["max_head_error_m_random"] = pos_max
    report["max_axis_error_random"] = axis_max
    report["max_skin_error_m_random"] = skin_max if mesh_info else None

    for action_name, case in (("rest", fk["rest"]), ("home", fk["home"])):
        for pose_bone in arm_obj.pose.bones:
            pose_bone.rotation_mode = "XYZ" if euler_z else "QUATERNION"
        if not assign_action(arm_obj, action_name):
            report[f"action_{action_name}_found"] = False
            continue
        report[f"action_{action_name}_found"] = True
        bpy.context.scene.frame_set(1)
        bpy.context.view_layer.update()
        pos, _ = head_errors(arm_obj, case, None)
        report[f"max_head_error_m_action_{action_name}"] = pos
        skin = None
        for (mesh_obj, names, _), rest in zip(mesh_info, rest_world, strict=True):
            skin = max(skin or 0.0, skin_error(mesh_obj, names, rest, case, skeleton, bone_body))
        report[f"max_skin_error_m_action_{action_name}"] = skin
        clear_animation(arm_obj)
    return report


# ------------------------------------------------------------------ BVH

def check_bvh(path: Path, skeleton: dict[str, Any], cases: list[dict[str, Any]], label: str) -> dict[str, Any]:
    """Import a BVH (Blender world axes) and compare bone heads on frames 1..len(cases)."""
    fresh_scene()
    bpy.ops.import_anim.bvh(filepath=str(path), axis_forward="Y", axis_up="Z", global_scale=1.0, frame_start=1,
                            rotate_mode="NATIVE", update_scene_fps=False, update_scene_duration=True)
    arms = find_objects("ARMATURE")
    report: dict[str, Any] = {"label": label, "armature_objects": len(arms)}
    if len(arms) != 1:
        report["error"] = "expected exactly one armature object"
        return report
    arm_obj = arms[0]
    report.update(structure_report(bone_parents(arm_obj), skeleton))
    errors = []
    for frame, case in enumerate(cases, start=1):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        errors.append(head_errors(arm_obj, case, None)[0])
    report["frames_checked"] = len(cases)
    report["max_head_error_m"] = max(errors)
    return report


def write_test_bvh(skeleton: dict[str, Any], fk: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    """From the loaded .blend rig, bake the 20 random cases + home + rest into a BVH (frames 1..22)."""
    arm_obj = find_objects("ARMATURE")[0]
    cases = list(fk["random_cases"]) + [fk["home"], fk["rest"]]
    clip = bpy.data.actions.new("verify_clip")
    if arm_obj.animation_data is None:
        arm_obj.animation_data_create()
    arm_obj.animation_data.action = clip
    root = arm_obj.pose.bones[skeleton["root_bone"]]
    rest_root = Vector(skeleton["bodies"][_root_index(skeleton)]["qpos0"]["position"])
    for frame, case in enumerate(cases, start=1):
        root.location = Vector(case["root_position"]) - rest_root
        root.keyframe_insert("location", frame=frame)
        for name, angle in case["angles"].items():
            pose_bone = arm_obj.pose.bones[name]
            pose_bone.rotation_euler = (0.0, 0.0, angle)
            pose_bone.keyframe_insert("rotation_euler", index=2, frame=frame)
    bpy.ops.object.select_all(action="DESELECT")
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.context.scene.frame_set(1)
    bpy.ops.export_anim.bvh(filepath=str(path), frame_start=1, frame_end=len(cases), global_scale=1.0,
                            rotate_mode="NATIVE", root_transform_only=True)
    return cases


def _root_index(skeleton: dict[str, Any]) -> int:
    """Index of the root body in skeleton['bodies']."""
    return next(i for i, b in enumerate(skeleton["bodies"]) if b["name"] == skeleton["root_body"])


# ------------------------------------------------------------------ main

def import_fbx(path: Path) -> None:
    """Fresh scene + FBX import with the same axis and bone-axis settings used for export."""
    fresh_scene()
    bpy.ops.import_scene.fbx(filepath=str(path), axis_forward="-Z", axis_up="Y", global_scale=1.0,
                             automatic_bone_orientation=False, primary_bone_axis="Y", secondary_bone_axis="X",
                             ignore_leaf_bones=False, use_anim=True)


def import_glb(path: Path) -> None:
    """Fresh scene + glTF import (Blender converts Y-up back to Z-up)."""
    fresh_scene()
    bpy.ops.import_scene.gltf(filepath=str(path))


def worst_of(report: dict[str, Any]) -> float:
    """Largest position error (m) found in a report."""
    values = [v for k, v in report.items() if k.startswith("max_head_error_m") and isinstance(v, (int, float))]
    values += [v for k, v in report.items() if k.startswith("max_skin_error_m") and isinstance(v, (int, float))]
    values += [report["max_rest_head_error_m"]] if "max_rest_head_error_m" in report else []
    return max(values) if values else float("nan")


def main() -> int:
    args = parse_args()
    skeleton = json.loads(args.skeleton.read_text(encoding="utf-8"))
    fk = json.loads(args.fk_reference.read_text(encoding="utf-8"))
    args.work_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []

    bpy.ops.wm.open_mainfile(filepath=str(args.rig_dir / "kobi_rig.blend"))
    test_bvh = args.work_dir / BVH_TEST_NAME
    # test BVH first: it keys temporary data into the in-memory file, so check the .blend before that
    reports.append(check_armature_file("blend", skeleton, fk, euler_z=True, mesh_check=True))
    bvh_cases = write_test_bvh(skeleton, fk, test_bvh)

    import_fbx(args.rig_dir / "kobi_rig.fbx")
    reports.append(check_armature_file("fbx_reimport", skeleton, fk, euler_z=False, mesh_check=True))
    import_glb(args.rig_dir / "kobi_rig.glb")
    reports.append(check_armature_file("glb_reimport", skeleton, fk, euler_z=False, mesh_check=True))
    reports.append(raw_formats.check_glb_raw((args.rig_dir / "kobi_rig.glb").read_bytes(), skeleton, fk))
    reports.append(raw_formats.check_fbx_raw((args.rig_dir / "kobi_rig.fbx").read_bytes(), skeleton, fk))
    reports.append(check_bvh(args.rig_dir / "kobi_skeleton.bvh", skeleton, [fk["rest"], fk["home"]], "bvh_delivered_2_frames"))
    reports.append(check_bvh(test_bvh, skeleton, bvh_cases, "bvh_test_22_frames"))

    ok = True
    for report in reports:
        structure = [report.get(k) for k in ("count_ok", "names_ok", "hierarchy_ok")]
        worst = worst_of(report)
        bad_vertices = report.get("vertices_not_single_bone_weight_1", 0)
        passed = all(structure) and worst <= TOLERANCE_M and bad_vertices == 0 and "error" not in report
        ok = ok and passed
        print(f"VERIFY {report['label']}: structure={structure} worst_position_error_m={worst:.3e} "
              f"bad_weight_vertices={bad_vertices} -> {'PASS' if passed else 'FAIL'}")
    if args.report:
        args.report.write_text(json.dumps(reports, indent=1) + "\n", encoding="utf-8")
    print("VERIFY RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


sys.exit(main())
