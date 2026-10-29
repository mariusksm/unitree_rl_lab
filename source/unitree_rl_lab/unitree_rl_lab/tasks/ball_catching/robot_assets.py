"""Robot asset of the ball tasks: the G1 29-DoF mimic asset, optionally with forearm colliders.

The G1 USD of ``unitree_model`` (``G1/29dof/usd/g1_29dof_rev_1_0``) has no collision shape on the forearm: the collider
prototypes of ``elbow_link`` and ``wrist_pitch_link`` in its base layer are empty and its physics layer leaves the
links' ``collisions`` prims empty, although Unitree's URDF gives both links their mesh as collider. In Isaac the ball
passes through the forearms (spec pitfall 26); the real forearms are solid. With ``FOREARM_COLLIDERS`` the asset is
spawned with the link's own visual mesh below these links as collider, a convex hull as for the upper arm, the wrist
roll piece and the hand, with the collision schemas and attributes of the existing wrist roll collider.
"""

from __future__ import annotations

import os

from pxr import Usd, UsdGeom

from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file
from isaaclab.sim.utils import clone
from isaaclab.utils import configclass

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_MIMIC_CFG, UnitreeUsdFileCfg

# Switch for all three ball tasks (the Unified actor is fine-tuned in Phase 2, so they share the robot)
FOREARM_COLLIDERS = False

FOREARM_LINKS = ("left_elbow_link", "left_wrist_pitch_link", "right_elbow_link", "right_wrist_pitch_link")
# existing collider whose physics settings the new ones copy, per side
TEMPLATE_COLLIDER = "{side}_wrist_roll_link/collisions/{side}_wrist_roll_link/mesh"
COPIED_ATTRIBUTE_PREFIXES = ("physics:", "physxCollision:", "physxConvexHullCollision:")


def _base_layer(usd_path: str) -> str:
    """Geometry layer of a USD made by Isaac Sim's URDF importer (``configuration/<name>_base.usd``)."""
    folder, name = os.path.split(usd_path)
    return os.path.join(folder, "configuration", f"{os.path.splitext(name)[0]}_base.usd")


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
            for schema in template.GetAppliedSchemas():
                if schema.startswith(("Physics", "Physx")):
                    mesh.AddAppliedSchema(schema)
            for attr in template.GetAttributes():
                if attr.GetName().startswith(COPIED_ATTRIBUTE_PREFIXES) and attr.HasAuthoredValue():
                    mesh.CreateAttribute(attr.GetName(), attr.GetTypeName(), custom=False).Set(attr.Get())
            for rel in template.GetRelationships():
                if rel.GetName().startswith("material:binding:physics"):
                    mesh.CreateRelationship(rel.GetName(), custom=False).SetTargets(rel.GetTargets())


@clone
def spawn_usd_with_forearm_colliders(prim_path, cfg, translation=None, orientation=None, **kwargs) -> Usd.Prim:
    """``spawn_from_usd`` plus forearm colliders, authored on the source prim before it is cloned."""
    prim = _spawn_from_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation)
    add_forearm_colliders(prim, cfg.usd_path)
    return prim


@configclass
class G1ForearmUsdFileCfg(UnitreeUsdFileCfg):
    func = spawn_usd_with_forearm_colliders


G1_FOREARM_CFG = UNITREE_G1_29DOF_MIMIC_CFG.replace(
    spawn=G1ForearmUsdFileCfg(usd_path=UNITREE_G1_29DOF_MIMIC_CFG.spawn.usd_path)
)
ROBOT_CFG = G1_FOREARM_CFG if FOREARM_COLLIDERS else UNITREE_G1_29DOF_MIMIC_CFG
