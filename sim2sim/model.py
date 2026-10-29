"""MuJoCo model of the G1 with the collision shapes Isaac Lab uses.

Bodies, joints and inertia come from ``g1.xml`` of unitree_rl_mjlab (identical to the Isaac asset, see
``tools/compare_models.py``). Its collision capsules are replaced by the shapes of the Isaac asset, read from the JSON
of ``tools/dump_isaac_model.py``: the same meshes (the MJCF meshes match the USD meshes in the body frame) and the
same primitives at the same poses. A dump with ``--forearm_colliders`` gives the model with solid forearms.

PhysX cooks every mesh collider into a convex hull of at most 64 vertices (default of ``PhysxConvexHullCollisionAPI``,
which the USD does not set); ``hull_vertex_limit`` does the same in MuJoCo.
"""

from __future__ import annotations

import json
import os

import mujoco
import numpy as np

G = mujoco.mjtGeom
# rotation that turns MuJoCo's cylinder axis (local z) onto the USD axis token
_AXIS_ROT = {
    "X": np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float),
    "Y": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float),
    "Z": np.eye(3),
}


def _pose(to_body, extra_rot: np.ndarray | None = None):
    """USD row-vector matrix (p_body = p_local @ M) -> position, quaternion (w, x, y, z), scale per local axis."""
    m = np.asarray(to_body, dtype=float)
    scale = np.linalg.norm(m[:3, :3], axis=1)
    rot = (m[:3, :3] / scale[:, None]).T  # columns: local axes in the body frame
    if extra_rot is not None:
        rot = rot @ extra_rot
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot.flatten())
    return m[3, :3].copy(), quat, scale


def load_isaac_dump(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


PHYSX_HULL_VERTEX_LIMIT = 64


def isaac_collision_spec(
    mjcf_dir: str, isaac: dict, floor: bool = True, hull_vertex_limit: int | None = PHYSX_HULL_VERTEX_LIMIT
) -> mujoco.MjSpec:
    """``g1.xml`` with the collision shapes of the Isaac asset (and a floor plane)."""
    spec = mujoco.MjSpec.from_file(os.path.join(mjcf_dir, "g1.xml"))
    mesh_by_stem = {os.path.splitext(os.path.basename(m.file))[0]: m.name for m in spec.meshes}
    if hull_vertex_limit is not None:
        for mesh in spec.meshes:
            mesh.maxhullvert = hull_vertex_limit
    for geom in [g for g in spec.geoms if g.contype or g.conaffinity]:
        spec.delete(geom)
    for body_name, info in isaac["bodies"].items():
        body = spec.body(body_name)
        for i, shape in enumerate(info["shapes"]):
            if not shape["enabled"] or "to_body" not in shape:
                continue
            kind = shape["type"]
            extra = _AXIS_ROT[shape["axis"]] if kind in ("Cylinder", "Capsule") else None
            pos, quat, scale = _pose(shape["to_body"], extra)
            geom = body.add_geom(name=f"{body_name}_col{i}", pos=pos, quat=quat, contype=1, conaffinity=1, group=3)
            if kind == "Mesh":
                stem = shape["path"].rsplit("/", 2)[-2]  # .../<mesh stem>/mesh
                geom.type, geom.meshname = G.mjGEOM_MESH, mesh_by_stem[stem]
            elif kind == "Sphere":
                geom.type, geom.size = G.mjGEOM_SPHERE, [shape["radius"] * scale.max(), 0, 0]
            elif kind in ("Cylinder", "Capsule"):
                axis = "XYZ".index(shape["axis"])
                radial = max(s for k, s in enumerate(scale) if k != axis)
                geom.type = G.mjGEOM_CYLINDER if kind == "Cylinder" else G.mjGEOM_CAPSULE
                geom.size = [shape["radius"] * radial, shape["half_length"] * scale[axis], 0]
            elif kind == "Cube":
                geom.type, geom.size = G.mjGEOM_BOX, 0.5 * shape["size"] * scale
            else:
                raise ValueError(f"{body_name}: unsupported Isaac collision shape {kind}")
    if floor:
        spec.worldbody.add_geom(name="floor", type=G.mjGEOM_PLANE, size=[0, 0, 0.05])
    return spec


def collision_geoms(m: mujoco.MjModel) -> list[int]:
    return [g for g in range(m.ngeom) if m.geom_bodyid[g] > 0 and (m.geom_contype[g] or m.geom_conaffinity[g])]


def self_collision_pairs(m: mujoco.MjModel) -> list[tuple[int, int]]:
    """Collision geom pairs on different bodies that are not parent and child (PhysX filters jointed links)."""
    geoms = collision_geoms(m)
    pairs = []
    for i, a in enumerate(geoms):
        for b in geoms[i + 1 :]:
            ba, bb = m.geom_bodyid[a], m.geom_bodyid[b]
            if ba != bb and m.body_parentid[ba] != bb and m.body_parentid[bb] != ba:
                pairs.append((a, b))
    return pairs
