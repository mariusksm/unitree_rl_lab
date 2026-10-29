"""Dump the G1 as Isaac Lab simulates it to JSON, for the comparison with the MuJoCo models (compare_models.py).

Spawns ``UNITREE_G1_29DOF_MIMIC_CFG`` (the robot of Phase 1, Phase 2 and Unified) alone on a ground plane and records
what PhysX uses:

- per body: mass, centre of mass, principal moments of inertia, the collision shapes of the USD (type, size, mesh
  approximation, pose and bounding box in the body frame) and the body position after forward kinematics at the zero
  and at the default joint pose (root 1 m above the origin, not rotated)
- per joint: range, armature, drive stiffness and damping, effort and velocity limit, friction, default position

Run in the Isaac Lab environment (CPU physics is enough, nothing is stepped):

    python sim2sim/tools/dump_isaac_model.py --headless --device cpu --out isaac_g1.json
"""

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--out", type=str, required=True, help="Output JSON file.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
import torch  # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_MIMIC_CFG  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shapes  # noqa: E402  # isort: skip


@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    robot = UNITREE_G1_29DOF_MIMIC_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def collision_shapes(stage, link_path: str) -> list[dict]:
    """Collision prims below a link (instance proxies included, child rigid bodies excluded), in the link frame."""
    cache = UsdGeom.XformCache()
    link = stage.GetPrimAtPath(link_path)
    to_link = cache.GetLocalToWorldTransform(link).GetInverse()
    out = []
    it = iter(Usd.PrimRange(link, Usd.TraverseInstanceProxies()))
    for prim in it:
        if prim != link and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            it.PruneChildren()
            continue
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        kind = prim.GetTypeName()
        enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        entry = {"path": str(prim.GetPath())[len(link_path) :], "type": kind, "enabled": enabled is not False}
        if kind == "Mesh":
            pts = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
            entry["points"] = len(pts)
            if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                entry["approximation"] = str(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get())
        elif kind == "Sphere":
            radius = UsdGeom.Sphere(prim).GetRadiusAttr().Get()
            pts = shapes.sphere(radius)
            entry["radius"] = radius
        elif kind in ("Capsule", "Cylinder"):
            geom = UsdGeom.Capsule(prim) if kind == "Capsule" else UsdGeom.Cylinder(prim)
            radius, height, axis = geom.GetRadiusAttr().Get(), geom.GetHeightAttr().Get(), geom.GetAxisAttr().Get()
            local = shapes.capsule(radius, height / 2) if kind == "Capsule" else shapes.cylinder(radius, height / 2)
            pts = shapes.along(local, axis)
            entry.update(radius=radius, half_length=height / 2, axis=axis)
        elif kind == "Cube":
            size = UsdGeom.Cube(prim).GetSizeAttr().Get()
            pts = shapes.box([size / 2] * 3)
            entry["size"] = size
        else:
            entry["unsupported"] = True
            out.append(entry)
            continue
        # USD matrices act on row vectors: p_link = p_local @ M (rows 0-2 rotation and scale, row 3 translation)
        to_body = np.array(cache.GetLocalToWorldTransform(prim) * to_link)
        entry["to_body"] = to_body.tolist()
        entry["aabb"] = shapes.aabb((np.c_[pts, np.ones(len(pts))] @ to_body)[:, :3])
        out.append(entry)
    return out


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args_cli.device))
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    robot = scene["robot"]
    view = robot.root_physx_view
    body_names, joint_names = list(robot.body_names), list(robot.joint_names)

    def first(x):
        return x.detach().cpu().numpy()[0]

    def fk(joint_pos: torch.Tensor) -> list[list[float]]:
        root = torch.zeros(1, 13, device=sim.device)
        root[0, 2], root[0, 3] = 1.0, 1.0  # position (0, 0, 1), quaternion (w, x, y, z) = identity
        robot.write_root_state_to_sim(root)
        robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))
        sim.forward()
        return first(view.get_link_transforms())[:, :3].tolist()

    fk_zero = fk(torch.zeros_like(robot.data.default_joint_pos))
    fk_default = fk(robot.data.default_joint_pos.clone())

    masses, coms, inertias = first(view.get_masses()), first(view.get_coms()), first(view.get_inertias())
    stage = omni.usd.get_context().get_stage()
    bodies = {}
    for i, name in enumerate(body_names):
        bodies[name] = {
            "mass": float(masses[i]),
            "com": coms[i, :3].tolist(),
            "inertia": sorted(np.linalg.eigvalsh(inertias[i].reshape(3, 3)).tolist()),
            "fk_zero": fk_zero[i],
            "fk_default": fk_default[i],
            "shapes": collision_shapes(stage, f"/World/envs/env_0/Robot/{name}"),
        }

    def dof(getter):
        try:
            return first(getattr(view, getter)()).tolist()
        except Exception:  # API differs between Isaac Sim versions
            return None

    limits = first(view.get_dof_limits())
    columns = {
        "armature": dof("get_dof_armatures"),
        "stiffness": dof("get_dof_stiffnesses"),
        "damping": dof("get_dof_dampings"),
        "effort_limit": dof("get_dof_max_forces"),
        "velocity_limit": dof("get_dof_max_velocities"),
        "friction": dof("get_dof_friction_coefficients") or dof("get_dof_friction_properties"),
    }
    default = first(robot.data.default_joint_pos)
    joints = {}
    for j, name in enumerate(joint_names):
        joints[name] = {"range": limits[j].tolist(), "default": float(default[j])}
        joints[name].update({k: (v[j] if v is not None else None) for k, v in columns.items()})

    cfg = UNITREE_G1_29DOF_MIMIC_CFG
    out = {
        "usd_path": cfg.spawn.usd_path,
        "init_pos": list(cfg.init_state.pos),
        "self_collisions": cfg.spawn.articulation_props.enabled_self_collisions,
        "bodies": bodies,
        "joints": joints,
    }
    with open(args_cli.out, "w") as f:
        json.dump(out, f, indent=1)
    total = sum(b["mass"] for b in bodies.values())
    print(f"[dump] {len(bodies)} bodies ({total:.3f} kg), {len(joints)} joints -> {args_cli.out}", flush=True)


if __name__ == "__main__":
    main()
    # Kit's shutdown hangs after this CPU-physics run (2026-10-29); there is nothing to clean up
    os._exit(0)
