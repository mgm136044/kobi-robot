"""Build the Kobi animation rig (armature + skinned mesh + actions) in headless Blender from skeleton.json.

Run (Blender 5.2, no GUI):

    /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup --python tools/skeleton/build_rig.py -- \
        --skeleton printed_face/skeleton/skeleton.json --repo-root . --out-dir printed_face/skeleton

Outputs in --out-dir: kobi_rig.blend, kobi_rig.fbx, kobi_rig.glb, kobi_skeleton.bvh.

Rig conventions
  * one bone per moving body, named after the joint that drives it; the root bone is "trunk";
  * bone head = joint anchor; the bone tail is perpendicular to the hinge axis and points to the child anchor;
  * EditBone.align_roll makes the hinge axis exactly the bone's local +Z axis, so rotation about local Z == joint angle;
  * rest pose = every joint angle 0 (MuJoCo qpos0). The trunk bone's local axes equal the world axes.
Only the redistributable visual meshes listed in skeleton.json ("mesh_geoms") are imported; the upstream meshes never are.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Any

import bpy
import numpy as np
from mathutils import Vector

ROOT_BONE_TAIL_M = 0.05  # trunk bone points along +Y so that its local axes equal the world axes
MIN_BONE_LENGTH_M = 0.01
FALLBACK_BONE_LENGTH_RANGE_M = (0.02, 0.08)
COM_DIRECTION_MIN_M = 1e-3
VERTEX_MERGE_QUANTUM_M = 1e-7
ARMATURE_NAME = "Kobi_Armature"
MESH_NAME = "Kobi_Mesh"
ACTION_REST = "rest"
ACTION_HOME = "home"
BVH_CLIP_NAME = "bvh_clip"
FRAME_FIRST, FRAME_SECOND = 1, 2
FPS = 30
STL_HEADER_BYTES = 80
STL_TRIANGLE_BYTES = 50
DEFAULT_MAX_EXPORT_TRIANGLES = 0  # 0 = never decimate
PATH_BUFFER_FILL = 1000  # shorter than Blender's 1024-byte path buffer, long enough to cover any old path

# Export settings (documented in printed_face/skeleton/README.md)
FBX_AXIS_FORWARD = "-Z"
FBX_AXIS_UP = "Y"


def parse_args() -> argparse.Namespace:
    """Parse the arguments that follow the '--' separator of the Blender command line."""
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--skeleton", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-export-triangles", type=int, default=DEFAULT_MAX_EXPORT_TRIANGLES,
                        help="if the mesh has more triangles, decimate a COPY for fbx/glb (0 = off)")
    parser.add_argument("--no-export", action="store_true", help="only build and save the .blend")
    return parser.parse_args(argv)


def read_binary_stl(path: Path) -> np.ndarray:
    """Return the triangle corners of a binary STL as float64 (n_triangles, 3, 3)."""
    data = path.read_bytes()
    (count,) = struct.unpack_from("<I", data, STL_HEADER_BYTES)
    if len(data) != STL_HEADER_BYTES + 4 + count * STL_TRIANGLE_BYTES:
        raise ValueError(f"{path.name}: not a binary STL")
    dtype = np.dtype([("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    return np.frombuffer(data, dtype=dtype, count=count, offset=STL_HEADER_BYTES + 4)["v"].astype(np.float64)


# ---------------------------------------------------------------- armature

def perpendicular_to(axis: Vector, vector: Vector) -> Vector:
    """Component of `vector` that is perpendicular to the unit `axis`."""
    return vector - axis * vector.dot(axis)


def choose_tail_direction(joint: dict[str, Any], children: list[dict[str, Any]], com: Vector) -> tuple[Vector, float]:
    """Pick the bone direction (unit, perpendicular to the hinge axis) and its length.

    Preferred: toward the first child anchor. Fallback: toward the body's centre of mass; last resort: world +Z/+X.
    """
    axis = Vector(joint["axis_world"]).normalized()
    head = Vector(joint["anchor_world"])
    for child in children:
        offset = perpendicular_to(axis, Vector(child["anchor_world"]) - head)
        if offset.length >= MIN_BONE_LENGTH_M:
            return offset.normalized(), offset.length
    low, high = FALLBACK_BONE_LENGTH_RANGE_M
    offset = perpendicular_to(axis, com - head)
    if offset.length >= COM_DIRECTION_MIN_M:
        return offset.normalized(), min(max(offset.length, low), high)
    for world in (Vector((0, 0, 1)), Vector((1, 0, 0))):
        offset = perpendicular_to(axis, world)
        if offset.length > COM_DIRECTION_MIN_M:
            return offset.normalized(), low
    raise RuntimeError(f"cannot choose a bone direction for {joint['name']}")


def build_armature(skeleton: dict[str, Any]) -> bpy.types.Object:
    """Create the armature object with one bone per joint plus the root bone."""
    arm_data = bpy.data.armatures.new(ARMATURE_NAME)
    arm_obj = bpy.data.objects.new(ARMATURE_NAME, arm_data)
    bpy.context.scene.collection.objects.link(arm_obj)
    bpy.context.view_layer.objects.active = arm_obj
    arm_obj.select_set(True)

    bodies = {b["name"]: b for b in skeleton["bodies"]}
    joints = skeleton["joints"]
    root_body = bodies[skeleton["root_body"]]
    root_name = skeleton["root_bone"]
    children_of: dict[str, list[dict[str, Any]]] = {}
    for joint in joints:
        children_of.setdefault(joint["parent_bone"], []).append(joint)

    bpy.ops.object.mode_set(mode="EDIT")
    edit = arm_data.edit_bones
    root_head = Vector(root_body["qpos0"]["position"])
    root = edit.new(root_name)
    root.head = root_head
    root.tail = root_head + Vector((0.0, ROOT_BONE_TAIL_M, 0.0))
    root.align_roll(Vector((0.0, 0.0, 1.0)))  # local axes == world axes
    for joint in joints:  # action order: a parent always precedes its children
        axis = Vector(joint["axis_world"]).normalized()
        head = Vector(joint["anchor_world"])
        com = Vector(bodies[joint["child_body"]]["com_qpos0"])
        direction, length = choose_tail_direction(joint, children_of.get(joint["name"], []), com)
        bone = edit.new(joint["name"])
        bone.head = head
        bone.tail = head + direction * length
        bone.align_roll(axis)  # bone local +Z == hinge axis
        bone.parent = edit[joint["parent_bone"]]
        bone.use_connect = False
    bpy.ops.object.mode_set(mode="OBJECT")

    for pose_bone in arm_obj.pose.bones:
        pose_bone.rotation_mode = "XYZ"
    for joint in joints:
        pose_bone = arm_obj.pose.bones[joint["name"]]
        pose_bone.lock_location = (True, True, True)
        pose_bone.lock_rotation = (True, True, False)  # only local Z turns
        pose_bone.lock_scale = (True, True, True)
        pose_bone["action_index"] = joint["action_index"]
        pose_bone["range_min_rad"] = joint["range"][0]
        pose_bone["range_max_rad"] = joint["range"][1]
        pose_bone["torque_limit_nm"] = joint["torque_limit_nm"]
    arm_obj.show_in_front = True
    return arm_obj


# ---------------------------------------------------------------- mesh

def color_material(cache: dict[str, bpy.types.Material], rgba: list[float]) -> bpy.types.Material:
    """Return (and cache) a simple Principled material for an RGBA colour."""
    key = "kobi_" + "".join(f"{round(c * 255):02x}" for c in rgba[:3])
    if key in cache:
        return cache[key]
    mat = bpy.data.materials.new(key)
    mat.diffuse_color = tuple(rgba)
    mat.use_nodes = True
    principled = mat.node_tree.nodes.get("Principled BSDF")
    if principled is not None:
        principled.inputs["Base Color"].default_value = tuple(rgba)
        principled.inputs["Roughness"].default_value = 0.6
    cache[key] = mat
    return mat


def load_geom_arrays(skeleton: dict[str, Any], repo_root: Path) -> list[dict[str, Any]]:
    """Read every allowed STL, place it in the world at qpos0 and merge duplicate corners inside the geom."""
    result = []
    for geom in skeleton["mesh_geoms"]:
        tri = read_binary_stl(repo_root / geom["mesh_file"])
        matrix = np.array(geom["world_from_file"])
        world = tri.reshape(-1, 3) @ matrix[:3, :3].T + matrix[:3, 3]
        keys = np.round(world / VERTEX_MERGE_QUANTUM_M).astype(np.int64)
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)
        faces = inverse.reshape(-1, 3)
        keep = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
        result.append({"geom": geom, "verts": unique * VERTEX_MERGE_QUANTUM_M, "faces": faces[keep]})
    return result


def build_mesh(skeleton: dict[str, Any], repo_root: Path, arm_obj: bpy.types.Object) -> bpy.types.Object:
    """Join all allowed visual meshes into one object with one rigid vertex group per bone."""
    arrays = load_geom_arrays(skeleton, repo_root)
    materials: dict[str, bpy.types.Material] = {}
    all_verts, all_faces, face_mat, vert_bone = [], [], [], []
    offset = 0
    slot_of: dict[str, int] = {}
    for item in arrays:
        geom = item["geom"]
        mat = color_material(materials, geom["rgba"])
        slot = slot_of.setdefault(mat.name, len(slot_of))
        all_verts.append(item["verts"])
        all_faces.append(item["faces"] + offset)
        face_mat.append(np.full(len(item["faces"]), slot, dtype=np.int32))
        vert_bone.extend([geom["bone"]] * len(item["verts"]))
        offset += len(item["verts"])
    verts = np.concatenate(all_verts)
    faces = np.concatenate(all_faces)
    mat_idx = np.concatenate(face_mat)

    mesh = bpy.data.meshes.new(MESH_NAME)
    mesh.from_pydata(verts.tolist(), [], faces.tolist())
    for name in slot_of:
        mesh.materials.append(materials[name])
    mesh.polygons.foreach_set("material_index", mat_idx)
    mesh.update()
    mesh.validate()
    obj = bpy.data.objects.new(MESH_NAME, mesh)
    bpy.context.scene.collection.objects.link(obj)

    indices_of: dict[str, list[int]] = {}
    for index, bone in enumerate(vert_bone):
        indices_of.setdefault(bone, []).append(index)
    for bone in arm_obj.data.bones:  # one group per bone, even when no allowed mesh is attached to it
        group = obj.vertex_groups.new(name=bone.name)
        if bone.name in indices_of:
            group.add(indices_of[bone.name], 1.0, "REPLACE")
    obj.parent = arm_obj
    modifier = obj.modifiers.new("Armature", "ARMATURE")
    modifier.object = arm_obj
    return obj


# ---------------------------------------------------------------- actions

def key_pose(arm_obj: bpy.types.Object, skeleton: dict[str, Any], angles: dict[str, float], root_delta: Vector,
             frames: tuple[int, ...]) -> None:
    """Set the pose (local Z rotations, root location) and insert keyframes on the given frames."""
    root_bone = arm_obj.pose.bones[skeleton["root_bone"]]
    root_bone.location = root_delta
    root_bone.rotation_euler = (0.0, 0.0, 0.0)
    for joint in skeleton["joints"]:
        arm_obj.pose.bones[joint["name"]].rotation_euler = (0.0, 0.0, angles[joint["name"]])
    for frame in frames:
        root_bone.keyframe_insert("location", frame=frame)
        root_bone.keyframe_insert("rotation_euler", frame=frame)
        for joint in skeleton["joints"]:
            arm_obj.pose.bones[joint["name"]].keyframe_insert("rotation_euler", index=2, frame=frame)


def make_pose_action(arm_obj: bpy.types.Object, skeleton: dict[str, Any], name: str, angles: dict[str, float],
                     root_delta: Vector) -> bpy.types.Action:
    """Create an action holding one pose on frames 1 and 2 (so that the clip has a non-zero length)."""
    action = bpy.data.actions.new(name)
    action.use_fake_user = True
    if arm_obj.animation_data is None:
        arm_obj.animation_data_create()
    arm_obj.animation_data.action = action
    key_pose(arm_obj, skeleton, angles, root_delta, (FRAME_FIRST, FRAME_SECOND))
    return action


def home_root_delta(skeleton: dict[str, Any]) -> Vector:
    """Root translation from qpos0 to the home keyframe (metres)."""
    home, rest = skeleton["home"]["root_position"], skeleton["home"]["qpos0_root_position"]
    return Vector([h - r for h, r in zip(home, rest, strict=True)])


def make_actions(arm_obj: bpy.types.Object, skeleton: dict[str, Any]) -> dict[str, bpy.types.Action]:
    """Add the actions "rest" and "home"."""
    zeros = {j["name"]: 0.0 for j in skeleton["joints"]}
    rest = make_pose_action(arm_obj, skeleton, ACTION_REST, zeros, Vector((0, 0, 0)))
    home = make_pose_action(arm_obj, skeleton, ACTION_HOME, skeleton["home"]["joint_angles"], home_root_delta(skeleton))
    arm_obj.animation_data.action = rest
    for pose_bone in arm_obj.pose.bones:
        pose_bone.location = (0, 0, 0)
        pose_bone.rotation_euler = (0, 0, 0)
    return {ACTION_REST: rest, ACTION_HOME: home}


# ---------------------------------------------------------------- scene helpers

def reset_scene() -> None:
    """Start from an empty scene with metric units and a fixed frame range."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    scene.unit_settings.length_unit = "METERS"
    scene.render.fps = FPS
    scene.frame_start, scene.frame_end = FRAME_FIRST, FRAME_SECOND


def overwrite_path_buffer(params: Any, attribute: str, final: bytes) -> None:
    """Overwrite a fixed-size path buffer completely (long filler first), so no old bytes survive behind the new value."""
    value = getattr(params, attribute)
    as_bytes = isinstance(value, bytes)
    for text in (b" " * PATH_BUFFER_FILL, final):
        setattr(params, attribute, text if as_bytes else text.decode())


def scrub_paths() -> None:
    """Remove absolute path strings that Blender would otherwise save (file browser buffers, render path)."""
    for scene in bpy.data.scenes:
        scene.render.filepath = "//"
    for screen in bpy.data.screens:
        for area in screen.areas:
            for space in area.spaces:
                params = getattr(space, "params", None)
                if params is None:
                    continue
                overwrite_path_buffer(params, "directory", b"//")
                overwrite_path_buffer(params, "filename", b"")


def supported(operator: Any, options: dict[str, Any]) -> dict[str, Any]:
    """Keep only the options that this Blender version's operator knows (and say which ones were dropped)."""
    known = {p.identifier for p in operator.get_rna_type().properties}
    dropped = sorted(set(options) - known)
    if dropped:
        print(f"WARNING: operator options not available in this Blender, skipped: {dropped}")
    return {k: v for k, v in options.items() if k in known}


def select_only(objects: list[bpy.types.Object]) -> None:
    """Select exactly the given objects and make the first one active."""
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]


def decimated_copy(mesh_obj: bpy.types.Object, arm_obj: bpy.types.Object, max_triangles: int) -> bpy.types.Object:
    """Return a decimated, skinned COPY of the mesh object (the original stays untouched)."""
    ratio = max_triangles / len(mesh_obj.data.polygons)
    copy = mesh_obj.copy()
    copy.data = mesh_obj.data.copy()
    copy.name = MESH_NAME + "_export"
    bpy.context.scene.collection.objects.link(copy)
    for existing in list(copy.modifiers):
        copy.modifiers.remove(existing)  # decimate the bind-pose geometry first, then skin it again
    modifier = copy.modifiers.new("Decimate", "DECIMATE")
    modifier.ratio = ratio
    modifier.use_collapse_triangulate = True
    select_only([copy])
    bpy.ops.object.modifier_apply(modifier="Decimate")
    skin = copy.modifiers.new("Armature", "ARMATURE")
    skin.object = arm_obj
    copy.parent = arm_obj
    return copy


# ---------------------------------------------------------------- exports

def export_fbx(path: Path, arm_obj: bpy.types.Object, mesh_obj: bpy.types.Object) -> None:
    """Export armature + skinned mesh + actions with Blender's default FBX axis settings (-Z forward, Y up)."""
    select_only([arm_obj, mesh_obj])
    options = {
        "filepath": str(path), "use_selection": True, "object_types": {"ARMATURE", "MESH"},
        "axis_forward": FBX_AXIS_FORWARD, "axis_up": FBX_AXIS_UP, "global_scale": 1.0, "apply_unit_scale": True,
        "apply_scale_options": "FBX_SCALE_NONE", "bake_space_transform": False, "use_mesh_modifiers": True,
        "add_leaf_bones": False, "primary_bone_axis": "Y", "secondary_bone_axis": "X", "armature_nodetype": "NULL",
        "bake_anim": True, "bake_anim_use_all_actions": True, "bake_anim_use_nla_strips": False,
        "bake_anim_use_all_bones": True, "bake_anim_force_startend_keying": True, "bake_anim_step": 1.0,
        "bake_anim_simplify_factor": 0.0, "path_mode": "STRIP", "use_metadata": False,
    }
    bpy.ops.export_scene.fbx(**supported(bpy.ops.export_scene.fbx, options))


def export_glb(path: Path, arm_obj: bpy.types.Object, mesh_obj: bpy.types.Object) -> None:
    """Export a glTF binary with skin and one animation per action (glTF is Y-up)."""
    select_only([arm_obj, mesh_obj])
    options = {
        "filepath": str(path), "export_format": "GLB", "use_selection": True, "export_yup": True,
        "export_apply": False, "export_skins": True, "export_animations": True,
        "export_animation_mode": "ACTIONS", "export_materials": "EXPORT", "export_cameras": False,
        "export_lights": False, "export_extras": False, "export_def_bones": False,
        "export_force_sampling": True, "export_optimize_animation_size": False,
        "export_rest_position_armature": True, "export_reset_pose_bones": False,
    }
    bpy.ops.export_scene.gltf(**supported(bpy.ops.export_scene.gltf, options))


def export_bvh(path: Path, skeleton: dict[str, Any], arm_obj: bpy.types.Object) -> None:
    """Export the armature only: frame 1 = rest, frame 2 = home (temporary clip, removed again)."""
    clip = bpy.data.actions.new(BVH_CLIP_NAME)
    arm_obj.animation_data.action = clip
    zeros = {j["name"]: 0.0 for j in skeleton["joints"]}
    key_pose(arm_obj, skeleton, zeros, Vector((0, 0, 0)), (FRAME_FIRST,))
    key_pose(arm_obj, skeleton, skeleton["home"]["joint_angles"], home_root_delta(skeleton), (FRAME_SECOND,))
    select_only([arm_obj])
    bpy.context.scene.frame_set(FRAME_FIRST)
    options = {"filepath": str(path), "frame_start": FRAME_FIRST, "frame_end": FRAME_SECOND, "global_scale": 1.0,
               "rotate_mode": "NATIVE", "root_transform_only": True}
    bpy.ops.export_anim.bvh(**supported(bpy.ops.export_anim.bvh, options))
    arm_obj.animation_data.action = bpy.data.actions[ACTION_REST]
    bpy.data.actions.remove(clip)


def main() -> None:
    args = parse_args()
    skeleton = json.loads(args.skeleton.read_text(encoding="utf-8"))
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    reset_scene()
    arm_obj = build_armature(skeleton)
    mesh_obj = build_mesh(skeleton, args.repo_root.resolve(), arm_obj)
    make_actions(arm_obj, skeleton)
    bpy.context.scene.frame_set(FRAME_FIRST)
    triangles = len(mesh_obj.data.polygons)
    print(f"RIG bones={len(arm_obj.data.bones)} vertices={len(mesh_obj.data.vertices)} triangles={triangles}")

    scrub_paths()
    blend_path = out_dir / "kobi_rig.blend"
    # copy=True keeps bpy.data.filepath empty, so the FBX exporter does not write the .blend path into its header
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path), compress=True, relative_remap=True, copy=True)
    print(f"SAVED {blend_path.name}")
    if args.no_export:
        return

    export_mesh = mesh_obj
    if args.max_export_triangles and triangles > args.max_export_triangles:
        export_mesh = decimated_copy(mesh_obj, arm_obj, args.max_export_triangles)
        mesh_obj.hide_set(True)
        print(f"EXPORT mesh decimated copy: {len(export_mesh.data.polygons)} triangles")
    export_fbx(out_dir / "kobi_rig.fbx", arm_obj, export_mesh)
    export_glb(out_dir / "kobi_rig.glb", arm_obj, export_mesh)
    export_bvh(out_dir / "kobi_skeleton.bvh", skeleton, arm_obj)
    print("EXPORTS done")


main()
