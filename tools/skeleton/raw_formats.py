"""Read the exported GLB and FBX files directly (no Blender importer) and check them against MuJoCo.

Used by verify_rig.py. Only numpy and the standard library are needed. The point of reading the raw files is to prove what an
animation tool sees in the file itself: node tree, rest orientation, hinge axis in the node frame, and the animation keys.

Both files are Y-up. The Blender/MuJoCo Z-up coordinates map to them with (x, y, z) -> (x, z, -y).
The FBX additionally carries centimetres: its armature node has scale 100, so file positions are 100 x metres.
"""

from __future__ import annotations

import json
import struct
import zlib
from typing import Any

import numpy as np

FBX_MAGIC = b"Kaydara FBX Binary  "
FBX_HEADER_BYTES = 27
FBX_NULL_RECORD_BYTES = 13
FBX_NODE_HEADER_BYTES = 13
GLTF_FLOAT = 5126
COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
EXPECTED_JOINT_COUNT = 14


# ------------------------------------------------------------------ shared math

def to_y_up(vec: np.ndarray) -> np.ndarray:
    """Z-up (x, y, z) -> Y-up (x, z, -y)."""
    return np.array([vec[0], vec[2], -vec[1]], dtype=float)


def axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation matrix about a (not necessarily unit) axis."""
    a = axis / np.linalg.norm(axis)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Quaternion (x, y, z, w) -> 3x3 rotation matrix."""
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def euler_xyz_degrees_to_matrix(euler_deg: np.ndarray) -> np.ndarray:
    """FBX rotation order XYZ (X applied first): R = Rz Ry Rx."""
    a, b, c = np.radians(euler_deg)
    rx = axis_angle_matrix(np.array([1.0, 0, 0]), a)
    ry = axis_angle_matrix(np.array([0, 1.0, 0]), b)
    rz = axis_angle_matrix(np.array([0, 0, 1.0]), c)
    return rz @ ry @ rx


def unit_columns(lin: np.ndarray) -> np.ndarray:
    """Remove the (positive) scale of a rotation*scale matrix."""
    return lin / np.linalg.norm(lin, axis=0)


def structure(parents: dict[str, str | None], expected: dict[str, str | None]) -> dict[str, Any]:
    """Bone count, names and hierarchy comparison."""
    return {"bone_count": len(parents), "count_ok": len(parents) == len(expected),
            "names_ok": set(parents) == set(expected), "hierarchy_ok": parents == expected}


def expected_parents(skeleton: dict[str, Any]) -> dict[str, str | None]:
    """Bone name -> parent bone name from skeleton.json."""
    table: dict[str, str | None] = {skeleton["root_bone"]: None}
    table.update({j["name"]: j["parent_bone"] for j in skeleton["joints"]})
    return table


def root_rest_position(skeleton: dict[str, Any]) -> np.ndarray:
    """World position of the trunk body at qpos0."""
    body = next(b for b in skeleton["bodies"] if b["name"] == skeleton["root_body"])
    return np.array(body["qpos0"]["position"])


# ------------------------------------------------------------------ GLB

def read_glb(data: bytes) -> tuple[dict[str, Any], bytes]:
    """Return (JSON chunk, BIN chunk) of a .glb file."""
    magic, version, _length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF" or version != 2:
        raise ValueError("not a glTF 2.0 binary")
    offset, gltf, binary = 12, {}, b""
    while offset < len(data):
        chunk_length, chunk_type = struct.unpack_from("<I4s", data, offset)
        body = data[offset + 8: offset + 8 + chunk_length]
        if chunk_type == b"JSON":
            gltf = json.loads(body.decode("utf-8"))
        elif chunk_type == b"BIN\x00":
            binary = bytes(body)
        offset += 8 + chunk_length
    return gltf, binary


def read_accessor(gltf: dict[str, Any], binary: bytes, index: int) -> np.ndarray:
    """Read a float32 accessor as a (count, components) array."""
    accessor = gltf["accessors"][index]
    if accessor["componentType"] != GLTF_FLOAT:
        raise ValueError("only float accessors are read here")
    view = gltf["bufferViews"][accessor["bufferView"]]
    width = COMPONENTS[accessor["type"]]
    stride = view.get("byteStride", width * 4)
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    rows = [np.frombuffer(binary, dtype="<f4", count=width, offset=start + i * stride) for i in range(accessor["count"])]
    return np.array(rows, dtype=np.float64)


def gltf_local(node: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(rotation matrix, translation, scale) of a glTF node."""
    if "matrix" in node:
        m = np.array(node["matrix"]).reshape(4, 4).T
        scale = np.linalg.norm(m[:3, :3], axis=0)
        return m[:3, :3] / scale, m[:3, 3], scale
    rot = quat_to_matrix(np.array(node.get("rotation", [0, 0, 0, 1]), dtype=float))
    return rot, np.array(node.get("translation", [0, 0, 0]), dtype=float), np.array(node.get("scale", [1, 1, 1]), dtype=float)


def gltf_world(gltf: dict[str, Any], rot_override: dict[int, np.ndarray],
               trans_override: dict[int, np.ndarray]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """World (rotation*scale, translation) of every node reachable from the first scene."""
    nodes = gltf["nodes"]
    world: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def visit(index: int, parent_lin: np.ndarray, parent_pos: np.ndarray) -> None:
        rot, pos, scale = gltf_local(nodes[index])
        rot = rot_override.get(index, rot)
        pos = trans_override.get(index, pos)
        lin = parent_lin @ rot @ np.diag(scale)
        position = parent_lin @ pos + parent_pos
        world[index] = (lin, position)
        for child in nodes[index].get("children", []):
            visit(child, lin, position)

    for root in gltf["scenes"][gltf.get("scene", 0)]["nodes"]:
        visit(root, np.eye(3), np.zeros(3))
    return world


def check_glb_raw(data: bytes, skeleton: dict[str, Any], fk: dict[str, Any]) -> dict[str, Any]:
    """Compare the glTF node tree, hinge axes and animations with MuJoCo (explicit Y-up conversion)."""
    gltf, binary = read_glb(data)
    nodes = gltf["nodes"]
    by_name = {n.get("name"): i for i, n in enumerate(nodes)}
    joint_names = [nodes[i]["name"] for i in gltf["skins"][0]["joints"]]
    parent_of = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}
    parents: dict[str, str | None] = {}
    for name in joint_names:
        parent = parent_of.get(by_name[name])
        parents[name] = nodes[parent]["name"] if parent is not None and nodes[parent]["name"] in joint_names else None
    report: dict[str, Any] = {"label": "glb_raw", **structure(parents, expected_parents(skeleton))}
    report["skins"] = len(gltf["skins"])
    report["animation_names"] = [a.get("name") for a in gltf.get("animations", [])]

    rest = gltf_world(gltf, {}, {})
    local_axes: dict[str, np.ndarray] = {}
    rest_err = float(np.abs(rest[by_name[skeleton["root_bone"]]][1] - to_y_up(root_rest_position(skeleton))).max())
    for joint in skeleton["joints"]:
        lin, pos = rest[by_name[joint["name"]]]
        rest_err = max(rest_err, float(np.abs(pos - to_y_up(np.array(joint["anchor_world"]))).max()))
        local_axes[joint["name"]] = unit_columns(lin).T @ to_y_up(np.array(joint["axis_world"]))
    report["max_rest_head_error_m"] = rest_err
    report["node_local_axes"] = {n: [round(float(c), 4) for c in v] for n, v in local_axes.items()}

    def evaluate(case: dict[str, Any], rotations: dict[int, np.ndarray] | None = None,
                 translations: dict[int, np.ndarray] | None = None) -> tuple[float, float]:
        rot_override = dict(rotations or {})
        if rotations is None:
            for name, angle in case["angles"].items():
                index = by_name[name]
                rot_override[index] = gltf_local(nodes[index])[0] @ axis_angle_matrix(local_axes[name], angle)
        world = gltf_world(gltf, rot_override, translations or {})
        pos_err, axis_err = 0.0, 0.0
        for name in joint_names:
            lin, pos = world[by_name[name]]
            target = case["anchors"].get(name, case["root_position"])
            pos_err = max(pos_err, float(np.abs(pos - to_y_up(np.array(target))).max()))
            if name in local_axes:
                expected_axis = to_y_up(np.array(case["axes"][name]))
                axis_err = max(axis_err, float(np.linalg.norm(unit_columns(lin) @ local_axes[name] - expected_axis)))
        return pos_err, axis_err

    pos_max, axis_max = 0.0, 0.0
    for case in fk["random_cases"]:
        pos, axis = evaluate(case)
        pos_max, axis_max = max(pos_max, pos), max(axis_max, axis)
    report["max_head_error_m_random"] = pos_max
    report["max_axis_error_random"] = axis_max

    for animation in gltf.get("animations", []):
        name = animation.get("name", "")
        case = fk.get(name)
        if case is None:
            continue
        rotations: dict[int, np.ndarray] = {}
        translations: dict[int, np.ndarray] = {}
        for channel in animation["channels"]:
            values = read_accessor(gltf, binary, animation["samplers"][channel["sampler"]]["output"])
            target, path = channel["target"]["node"], channel["target"]["path"]
            if path == "rotation":
                rotations[target] = quat_to_matrix(values[0])
            elif path == "translation":
                translations[target] = values[0]
        report[f"max_head_error_m_animation_{name}"] = evaluate(case, rotations, translations)[0]
        animated = {nodes[i]["name"] for i in rotations}
        report[f"animation_{name}_has_all_joints"] = animated >= {j["name"] for j in skeleton["joints"]}
    return report


# ------------------------------------------------------------------ FBX (binary 7.x, versions below 7500)

def _fbx_property(data: bytes, pos: int) -> tuple[Any, int]:
    """Read one FBX property value; return (value, next position)."""
    kind = chr(data[pos])
    pos += 1
    fixed = {"Y": ("<h", 2), "C": ("<B", 1), "I": ("<i", 4), "F": ("<f", 4), "D": ("<d", 8), "L": ("<q", 8)}
    if kind in fixed:
        fmt, size = fixed[kind]
        return struct.unpack_from(fmt, data, pos)[0], pos + size
    if kind in "SR":
        (length,) = struct.unpack_from("<I", data, pos)
        raw = data[pos + 4: pos + 4 + length]
        return (raw.decode("latin-1") if kind == "S" else raw), pos + 4 + length
    dtype = {"f": "<f4", "d": "<f8", "l": "<i8", "i": "<i4", "b": "u1"}[kind]
    count, encoding, compressed = struct.unpack_from("<III", data, pos)
    pos += 12
    raw = data[pos: pos + compressed]
    if encoding == 1:
        raw = zlib.decompress(raw)
    return np.frombuffer(raw, dtype=dtype, count=count), pos + compressed


def _fbx_node(data: bytes, pos: int) -> tuple[tuple[str, list[Any], list[Any]] | None, int]:
    """Read one FBX node record: (name, properties, children), or None for a null record."""
    end, count, _list_len, name_len = struct.unpack_from("<IIIB", data, pos)
    if end == 0:
        return None, pos + FBX_NULL_RECORD_BYTES
    name = data[pos + FBX_NODE_HEADER_BYTES: pos + FBX_NODE_HEADER_BYTES + name_len].decode()
    cursor = pos + FBX_NODE_HEADER_BYTES + name_len
    props = []
    for _ in range(count):
        value, cursor = _fbx_property(data, cursor)
        props.append(value)
    children = []
    while cursor < end:
        child, cursor = _fbx_node(data, cursor)
        if child is not None:
            children.append(child)
    return (name, props, children), end


def parse_fbx(data: bytes) -> list[tuple[str, list[Any], list[Any]]]:
    """Parse the top-level node records of a binary FBX."""
    if not data.startswith(FBX_MAGIC):
        raise ValueError("not a binary FBX")
    version = struct.unpack_from("<I", data, 23)[0]
    if version >= 7500:
        raise ValueError("FBX 7.5+ record layout is not supported by this reader")
    pos, nodes = FBX_HEADER_BYTES, []
    while pos < len(data):
        node, pos = _fbx_node(data, pos)
        if node is None:
            break
        nodes.append(node)
    return nodes


def _fbx_props70(node: tuple[str, list[Any], list[Any]]) -> dict[str, np.ndarray]:
    """Properties70 of a node as name -> numeric vector (only numeric P entries)."""
    result: dict[str, np.ndarray] = {}
    for child in node[2]:
        if child[0] != "Properties70":
            continue
        for entry in child[2]:
            values = entry[1]
            numbers = [v for v in values[4:] if isinstance(v, (int, float))]
            if numbers:
                result[values[0]] = np.array(numbers, dtype=float)
    return result


def _fbx_name(raw: str) -> str:
    """'name\\x00\\x01Class' -> 'name'."""
    return raw.split("\x00")[0]


def check_fbx_raw(data: bytes, skeleton: dict[str, Any], fk: dict[str, Any]) -> dict[str, Any]:
    """Compare the FBX node tree, rest orientation, hinge axes and animation takes with MuJoCo."""
    top = {n[0]: n for n in parse_fbx(data)}
    settings = _fbx_props70(top["GlobalSettings"])
    objects = top["Objects"][2]
    models = {n[1][0]: n for n in objects if n[0] == "Model"}
    names = {i: _fbx_name(n[1][1]) for i, n in models.items()}
    props = {i: _fbx_props70(n) for i, n in models.items()}
    kinds = {i: n[1][2] for i, n in models.items()}
    parent: dict[int, int] = {}
    curve_node_target: dict[int, tuple[int, str]] = {}
    curve_to_node: dict[int, tuple[int, str]] = {}
    node_to_layer: dict[int, int] = {}
    layer_to_stack: dict[int, int] = {}
    layers = {n[1][0]: n for n in objects if n[0] == "AnimationLayer"}
    stacks = {n[1][0]: _fbx_name(n[1][1]) for n in objects if n[0] == "AnimationStack"}
    curve_nodes = {n[1][0] for n in objects if n[0] == "AnimationCurveNode"}
    curves = {n[1][0]: n for n in objects if n[0] == "AnimationCurve"}
    for conn in top["Connections"][2]:
        kind, child, owner = conn[1][0], conn[1][1], conn[1][2]
        if child in models and owner in models:
            parent[child] = owner
        elif kind == "OP" and child in curve_nodes and owner in models:
            curve_node_target[child] = (owner, conn[1][3])
        elif kind == "OP" and child in curves and owner in curve_nodes:
            curve_to_node[child] = (owner, conn[1][3])
        elif child in curve_nodes and owner in layers:
            node_to_layer[child] = owner
        elif child in layers and owner in stacks:
            layer_to_stack[child] = owner

    limb = {i for i, k in kinds.items() if k == "LimbNode"}
    parents = {names[i]: (names[parent[i]] if parent.get(i) in limb else None) for i in limb}
    report: dict[str, Any] = {"label": "fbx_raw", **structure(parents, expected_parents(skeleton))}
    report["up_axis"] = [int(settings["UpAxis"][0]), int(settings["UpAxisSign"][0])]
    report["front_axis"] = [int(settings["FrontAxis"][0]), int(settings["FrontAxisSign"][0])]
    report["unit_scale_factor_cm_per_unit"] = float(settings["UnitScaleFactor"][0])
    report["takes"] = sorted(stacks.values())
    by_name = {names[i]: i for i in models}

    def local(i: int, rot_override: np.ndarray | None, trans_override: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        p = props[i]
        if "PreRotation" in p and np.abs(p["PreRotation"]).max() > 1e-9:
            raise ValueError("PreRotation is not expected in this export")
        rot = rot_override if rot_override is not None else euler_xyz_degrees_to_matrix(p.get("Lcl Rotation", np.zeros(3)))
        scale = p.get("Lcl Scaling", np.ones(3))
        trans = trans_override if trans_override is not None else p.get("Lcl Translation", np.zeros(3))
        return rot @ np.diag(scale), trans

    def world(rot_overrides: dict[int, np.ndarray],
              trans_overrides: dict[int, np.ndarray]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        result: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        def resolve(i: int) -> tuple[np.ndarray, np.ndarray]:
            if i in result:
                return result[i]
            lin_l, pos_l = local(i, rot_overrides.get(i), trans_overrides.get(i))
            if i in parent:
                lin_p, pos_p = resolve(parent[i])
                result[i] = (lin_p @ lin_l, lin_p @ pos_l + pos_p)
            else:
                result[i] = (lin_l, pos_l)
            return result[i]

        for i in models:
            resolve(i)
        return result

    cm_per_m = 100.0 / report["unit_scale_factor_cm_per_unit"]  # file units per metre
    rest = world({}, {})
    local_axes: dict[str, np.ndarray] = {}
    rest_err = float(np.abs(rest[by_name[skeleton["root_bone"]]][1] / cm_per_m - to_y_up(root_rest_position(skeleton))).max())
    for joint in skeleton["joints"]:
        lin, pos = rest[by_name[joint["name"]]]
        rest_err = max(rest_err, float(np.abs(pos / cm_per_m - to_y_up(np.array(joint["anchor_world"]))).max()))
        local_axes[joint["name"]] = unit_columns(lin).T @ to_y_up(np.array(joint["axis_world"]))
    report["max_rest_head_error_m"] = rest_err
    report["node_local_axes"] = {n: [round(float(c), 4) for c in v] for n, v in local_axes.items()}
    rest_rotation = {i: euler_xyz_degrees_to_matrix(props[i].get("Lcl Rotation", np.zeros(3))) for i in models}
    report["rest_lcl_rotation_nonzero_bones"] = sum(
        1 for i in limb if np.abs(props[i].get("Lcl Rotation", np.zeros(3))).max() > 1e-6)

    def evaluate(case: dict[str, Any], rot_overrides: dict[int, np.ndarray],
                 trans_overrides: dict[int, np.ndarray]) -> tuple[float, float]:
        world_now = world(rot_overrides, trans_overrides)
        pos_err, axis_err = 0.0, 0.0
        for name in parents:
            lin, pos = world_now[by_name[name]]
            target = case["anchors"].get(name, case["root_position"])
            pos_err = max(pos_err, float(np.abs(pos / cm_per_m - to_y_up(np.array(target))).max()))
            if name in local_axes:
                expected_axis = to_y_up(np.array(case["axes"][name]))
                axis_err = max(axis_err, float(np.linalg.norm(unit_columns(lin) @ local_axes[name] - expected_axis)))
        return pos_err, axis_err

    pos_max, axis_max = 0.0, 0.0
    for case in fk["random_cases"]:
        overrides = {by_name[n]: rest_rotation[by_name[n]] @ axis_angle_matrix(local_axes[n], angle)
                     for n, angle in case["angles"].items()}
        pos, axis = evaluate(case, overrides, {})
        pos_max, axis_max = max(pos_max, pos), max(axis_max, axis)
    report["max_head_error_m_random"] = pos_max
    report["max_axis_error_random"] = axis_max

    # animation takes: per stack, the first key of every Lcl Rotation / Lcl Translation curve
    for stack_id, stack_name in stacks.items():
        take = stack_name.split("|")[-1]
        case = fk.get(take)
        if case is None:
            continue
        values: dict[tuple[int, str], dict[str, float]] = {}
        for curve_id, (node_id, channel) in curve_to_node.items():
            layer = node_to_layer.get(node_id)
            if layer is None or layer_to_stack.get(layer) != stack_id or node_id not in curve_node_target:
                continue
            model_id, prop = curve_node_target[node_id]
            key_values = next(c for c in curves[curve_id][2] if c[0] == "KeyValueFloat")[1][0]
            values.setdefault((model_id, prop), {})[channel] = float(key_values[0])
        rot_overrides, trans_overrides = {}, {}
        for (model_id, prop), channels in values.items():
            base = props[model_id].get(prop, np.zeros(3)).copy()
            for axis_index, axis_name in enumerate(("d|X", "d|Y", "d|Z")):
                if axis_name in channels:
                    base[axis_index] = channels[axis_name]
            if prop == "Lcl Rotation":
                rot_overrides[model_id] = euler_xyz_degrees_to_matrix(base)
            elif prop == "Lcl Translation":
                trans_overrides[model_id] = base
        report[f"max_head_error_m_animation_{take}"] = evaluate(case, rot_overrides, trans_overrides)[0]
        report[f"animation_{take}_rotation_nodes"] = len(rot_overrides)
    return report
