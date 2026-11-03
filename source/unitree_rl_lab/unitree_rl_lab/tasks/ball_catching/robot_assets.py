"""Robot asset of the ball tasks: the G1 29-DoF mimic asset with solid forearms and open palms.

Two changes to the G1 USD of ``unitree_model`` (``G1/29dof/usd/g1_29dof_rev_1_0``), authored on the spawned prim before
it is cloned (the USD file stays untouched):

- ``FOREARM_COLLIDERS``: the USD has no collision shape on the forearm: the collider prototypes of ``elbow_link`` and
  ``wrist_pitch_link`` in its base layer are empty and its physics layer leaves the links' ``collisions`` prims empty,
  although Unitree's URDF gives both links their mesh as collider. In Isaac the ball passes through the forearms (spec
  pitfall 26); the real forearms are solid. The link's own visual mesh becomes its collider, a convex hull as for the
  upper arm, the wrist roll piece and the hand, with the collision schemas and attributes of the existing wrist roll
  collider.
- ``CONCAVE_PALMS``: the USD makes the whole rubber hand one convex-hull collider, which closes the cup of the palm
  (1.6 cm above the palm centre; a ∅15 cm ball rests 1.8 cm farther out than on the real hand). Its vertices are split
  into the convex parts of ``rubber_hand.PARTS`` (palm, finger bases, fingertips, thumb; as in Phillip's g1-catch), each
  a convex-hull collider with the settings of the original, which is disabled.
"""

from __future__ import annotations

import os

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, Vt
from scipy.spatial import ConvexHull

from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file
from isaaclab.sim.utils import clone
from isaaclab.utils import configclass

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_MIMIC_CFG, UnitreeUsdFileCfg
from unitree_rl_lab.tasks.ball_catching import rubber_hand

# Switches for all three ball tasks (the Unified actor is fine-tuned in the Advanced tasks, so they share the robot)
FOREARM_COLLIDERS = True
CONCAVE_PALMS = True

FOREARM_LINKS = ("left_elbow_link", "left_wrist_pitch_link", "right_elbow_link", "right_wrist_pitch_link")
# existing collider whose physics settings the new forearm colliders copy, per side
TEMPLATE_COLLIDER = "{side}_wrist_roll_link/collisions/{side}_wrist_roll_link/mesh"
COPIED_ATTRIBUTE_PREFIXES = ("physics:", "physxCollision:", "physxConvexHullCollision:")


def _base_layer(usd_path: str) -> str:
    """Geometry layer of a USD made by Isaac Sim's URDF importer (``configuration/<name>_base.usd``)."""
    folder, name = os.path.split(usd_path)
    return os.path.join(folder, "configuration", f"{os.path.splitext(name)[0]}_base.usd")


def _copy_collision_settings(template: Usd.Prim, mesh: Usd.Prim):
    """Give ``mesh`` the physics schemas, authored physics attributes and physics material of ``template``."""
    for schema in template.GetAppliedSchemas():
        if schema.startswith(("Physics", "Physx")):
            mesh.AddAppliedSchema(schema)
    for attr in template.GetAttributes():
        if attr.GetName().startswith(COPIED_ATTRIBUTE_PREFIXES) and attr.HasAuthoredValue():
            mesh.CreateAttribute(attr.GetName(), attr.GetTypeName(), custom=False).Set(attr.Get())
    for rel in template.GetRelationships():
        if rel.GetName().startswith("material:binding:physics"):
            mesh.CreateRelationship(rel.GetName(), custom=False).SetTargets(rel.GetTargets())


def add_forearm_colliders(prim: Usd.Prim, usd_path: str, links=FOREARM_LINKS):
    """Add each link's visual mesh as collider below ``<link>/collisions`` of a spawned G1 prim."""
    stage, root = prim.GetStage(), prim.GetPath().pathString
    for link in links:
        template = stage.GetPrimAtPath(f"{root}/{TEMPLATE_COLLIDER.format(side=link.split('_')[0])}")
        if not template.IsValid():
            raise RuntimeError(f"Template collider missing: {template.GetPath()}")
        holder = stage.DefinePrim(f"{root}/{link}/collisions/{link}", "Xform")
        holder.GetReferences().AddReference(_base_layer(usd_path), f"/visuals/{link}")
        holder.SetInstanceable(False)
        UsdGeom.Imageable(holder).CreatePurposeAttr().Set(UsdGeom.Tokens.guide)  # not rendered, like the others
        meshes = [p for p in Usd.PrimRange(holder) if p.IsA(UsdGeom.Mesh)]
        if not meshes:
            raise RuntimeError(f"No visual mesh below {holder.GetPath()}")
        for mesh in meshes:
            _copy_collision_settings(template, mesh)


def split_rubber_hands(prim: Usd.Prim, sides=("left", "right")):
    """Replace the one-hull rubber-hand collider of each ``<side>_wrist_yaw_link`` by its convex parts."""
    stage, root = prim.GetStage(), prim.GetPath().pathString
    cache = UsdGeom.XformCache()
    for side in sides:
        link = stage.GetPrimAtPath(f"{root}/{side}_wrist_yaw_link")
        collisions = stage.GetPrimAtPath(f"{root}/{side}_wrist_yaw_link/collisions")
        collisions.SetInstanceable(False)  # an instanced prototype cannot be edited
        holder = stage.GetPrimAtPath(f"{collisions.GetPath()}/{side}_rubber_hand")
        original = stage.GetPrimAtPath(f"{holder.GetPath()}/mesh")
        if not original.IsValid() or not original.HasAPI(UsdPhysics.CollisionAPI):
            raise RuntimeError(f"Rubber hand collider missing: {original.GetPath()}")
        points = np.array(UsdGeom.Mesh(original).GetPointsAttr().Get(), dtype=float)
        # USD matrices act on row vectors: p_link = p_local @ M (rows 0-2 rotation and scale, row 3 translation)
        to_link = np.array(cache.GetLocalToWorldTransform(original) * cache.GetLocalToWorldTransform(link).GetInverse())
        in_link = points @ to_link[:3, :3] + to_link[3, :3]
        for part, mask in rubber_hand.part_masks(in_link).items():
            hull = ConvexHull(points[mask])
            index = {old: new for new, old in enumerate(hull.vertices)}
            vertices = hull.points[hull.vertices].astype(np.float32)
            part_holder = stage.DefinePrim(f"{collisions.GetPath()}/{side}_rubber_hand_{part}", "Xform")
            UsdGeom.Xformable(part_holder).AddTransformOp().Set(UsdGeom.Xformable(holder).GetLocalTransformation())
            UsdGeom.Imageable(part_holder).CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
            mesh = UsdGeom.Mesh.Define(stage, f"{part_holder.GetPath()}/mesh")
            mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices))
            mesh.CreateFaceVertexCountsAttr([3] * len(hull.simplices))
            mesh.CreateFaceVertexIndicesAttr([index[int(i)] for i in hull.simplices.flatten()])
            mesh.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.stack([vertices.min(axis=0), vertices.max(axis=0)])))
            _copy_collision_settings(original, mesh.GetPrim())
        UsdPhysics.CollisionAPI(original).CreateCollisionEnabledAttr().Set(False)


@clone
def spawn_g1_ball_asset(prim_path, cfg, translation=None, orientation=None, **kwargs) -> Usd.Prim:
    """``spawn_from_usd`` plus the collider changes of ``cfg``, authored on the source prim before it is cloned."""
    prim = _spawn_from_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation)
    if cfg.forearm_colliders:
        add_forearm_colliders(prim, cfg.usd_path)
    if cfg.concave_palms:
        split_rubber_hands(prim)
    return prim


@configclass
class G1BallUsdFileCfg(UnitreeUsdFileCfg):
    func = spawn_g1_ball_asset
    forearm_colliders: bool = True
    concave_palms: bool = True


ROBOT_CFG = UNITREE_G1_29DOF_MIMIC_CFG.replace(
    spawn=G1BallUsdFileCfg(
        usd_path=UNITREE_G1_29DOF_MIMIC_CFG.spawn.usd_path,
        forearm_colliders=FOREARM_COLLIDERS,
        concave_palms=CONCAVE_PALMS,
    )
)


# ── Joint load (since 2026-11-03): Advanced-A pressed its hands against each other with 20-30 N while waiting and
#    holding, and its left wrist yaw (5 Nm) worked at its torque limit. Advanced-A2 (and the Advanced stages built on
#    it) and Unified penalize both: the share of the physics steps in which a hand presses on the other arm above
#    HAND_CONTACT_THRESHOLD (``mdp.self_contact`` with the sensors below), and the commanded torque of the arm joints
#    above ARM_TORQUE_THRESHOLD of the limit (``mdp.torque_limit_excess``). ──
HAND_CONTACT_THRESHOLD = 5.0
"""Force (N) above which a hand presses on the other arm: light touch allowed, A pressed with 20-30 N."""
HAND_CONTACT_WEIGHT = -5.0
"""Weight per hand sensor (as `self_contact` of the arms on the torso)."""
ARM_JOINT_NAMES = [".*_shoulder_.*", ".*_elbow_joint", ".*_wrist_.*"]
"""Joints of the torque-limit penalty. The leg joints stay out: steps and push recovery are not affected."""
ARM_TORQUE_THRESHOLD = 0.6
"""Share of the effort limit above which the commanded torque is penalized (A: no joint but the left wrist yaw above
0.6 in more than 1 % of the steps)."""
ARM_TORQUE_WEIGHT = -20.0
"""Weight of the squared excess."""
OTHER_ARM_LINKS = ("wrist_yaw_link", "wrist_pitch_link", "wrist_roll_link", "elbow_link")
"""Links of the other arm a hand may press on (the hand first)."""


def hand_contact_sensor(side: str, other: str, links=OTHER_ARM_LINKS) -> ContactSensorCfg:
    """The ``side`` hand as the single sensor body, filtered against ``links`` of the ``other`` arm (Isaac Lab reports
    filtered contacts only one-to-many). Use ("left", "right") with all links and ("right", "left") without the hand,
    so the hand-hand contact is counted once."""
    return ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{side}_wrist_yaw_link",
        filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/Robot/{other}_{link}" for link in links],
        history_length=4,
    )
