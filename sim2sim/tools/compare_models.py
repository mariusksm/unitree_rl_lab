"""Compare MuJoCo models of the G1 with the robot Isaac Lab simulates (JSON of dump_isaac_model.py).

Per body: mass, centre of mass, principal moments of inertia, body position after forward kinematics at the zero and
the default joint pose, and the collision shapes with their bounding box in the body frame. Per joint: range,
armature, passive damping and friction, effort limit. Run in the sim2sim environment:

    python sim2sim/tools/compare_models.py isaac_g1.json ~/unitree_mjcf/g1/g1.xml ~/unitree_mjcf/g1/scene_g1.xml
"""

import argparse
import json
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shapes  # noqa: E402  # isort: skip

BODY, JOINT = mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_JOINT
G = mujoco.mjtGeom


def mjcf_info(xml: str, isaac_joints: dict) -> dict:
    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    bodies = {}
    for b in range(1, m.nbody):
        bodies[mujoco.mj_id2name(m, BODY, b)] = {
            "mass": float(m.body_mass[b]),
            "com": m.body_ipos[b].tolist(),
            "inertia": sorted(m.body_inertia[b].tolist()),
            "shapes": [],
        }
    for g in range(m.ngeom):
        b = m.geom_bodyid[g]
        if b == 0 or (m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0):
            continue  # world geoms and visual-only geoms
        t, size = m.geom_type[g], m.geom_size[g]
        if t == G.mjGEOM_SPHERE:
            kind, pts = f"sphere r={size[0] * 100:.1f}", shapes.sphere(size[0])
        elif t == G.mjGEOM_CAPSULE:
            kind, pts = f"capsule r={size[0] * 100:.1f} l={2 * size[1] * 100:.1f}", shapes.capsule(size[0], size[1])
        elif t == G.mjGEOM_CYLINDER:
            kind, pts = f"cylinder r={size[0] * 100:.1f} h={2 * size[1] * 100:.1f}", shapes.cylinder(size[0], size[1])
        elif t == G.mjGEOM_BOX:
            kind, pts = "box", shapes.box(size)
        elif t == G.mjGEOM_ELLIPSOID:
            kind, pts = "ellipsoid", shapes.sphere(1.0) * size
        elif t == G.mjGEOM_MESH:
            mesh = m.geom_dataid[g]
            start = m.mesh_vertadr[mesh]
            kind, pts = "mesh", m.mesh_vert[start : start + m.mesh_vertnum[mesh]].astype(float)
        else:
            continue
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, m.geom_quat[g])
        in_body = pts @ rot.reshape(3, 3).T + m.geom_pos[g]
        bodies[mujoco.mj_id2name(m, BODY, b)]["shapes"].append({"type": kind, "aabb": shapes.aabb(in_body)})

    joints = {}
    for j in range(m.njnt):
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            free = j
            continue
        dof = m.jnt_dofadr[j]
        joints[mujoco.mj_id2name(m, JOINT, j)] = {
            "range": m.jnt_range[j].tolist() if m.jnt_limited[j] else None,
            "armature": float(m.dof_armature[dof]),
            "damping": float(m.dof_damping[dof]),
            "frictionloss": float(m.dof_frictionloss[dof]),
            "effort_limit": float(m.jnt_actfrcrange[j][1]) if m.jnt_actfrclimited[j] else None,
        }

    def fk(joint_pos: dict) -> dict:
        d.qpos[:] = 0.0
        a = m.jnt_qposadr[free]
        d.qpos[a : a + 7] = [0, 0, 1, 1, 0, 0, 0]
        for name, q in joint_pos.items():
            j = mujoco.mj_name2id(m, JOINT, name)
            if j >= 0:
                d.qpos[m.jnt_qposadr[j]] = q
        mujoco.mj_kinematics(m, d)
        return {mujoco.mj_id2name(m, BODY, b): d.xpos[b].copy() for b in range(1, m.nbody)}

    zero = fk({n: 0.0 for n in isaac_joints})
    default = fk({n: v["default"] for n, v in isaac_joints.items()})
    for name, body in bodies.items():
        body["fk_zero"], body["fk_default"] = zero[name], default[name]
    return {"bodies": bodies, "joints": joints, "nu": m.nu}


def bbox_text(aabbs: list) -> str:
    """Bounding box of several shapes in cm."""
    if not aabbs:
        return "no collision shape"
    lo = np.min([a[0] for a in aabbs], axis=0) * 100
    hi = np.max([a[1] for a in aabbs], axis=0) * 100
    return " ".join(f"{axis}[{lo[i]:6.1f},{hi[i]:6.1f}]" for i, axis in enumerate("xyz"))


def kinds_text(kinds: list[str]) -> str:
    if len(kinds) <= 3:
        return ", ".join(kinds)
    return f"{len(kinds)} shapes: " + ", ".join(sorted(set(kinds)))


def isaac_kind(s: dict) -> str:
    if s["type"] == "Mesh":
        return f"mesh({s.get('approximation', '?')}, {s['points']} pts)"
    if s["type"] == "Sphere":
        return f"sphere r={s['radius'] * 100:.1f}"
    if s["type"] in ("Capsule", "Cylinder"):
        return f"{s['type'].lower()} r={s['radius'] * 100:.1f} l={2 * s['half_length'] * 100:.1f} {s['axis']}"
    return s["type"].lower()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("isaac_json")
    parser.add_argument("mjcf", nargs="+")
    args = parser.parse_args()
    with open(args.isaac_json) as f:
        isaac = json.load(f)
    models = {os.path.basename(x): mjcf_info(x, isaac["joints"]) for x in args.mjcf}
    names = list(models)

    print(f"Isaac: {isaac['usd_path']}, start height {isaac['init_pos'][2]} m, "
          f"self-collisions {isaac['self_collisions']}")
    totals = [sum(b["mass"] for b in isaac["bodies"].values())]
    totals += [sum(b["mass"] for b in models[n]["bodies"].values()) for n in names]
    print("total mass (kg): " + " | ".join(f"{n} {t:.3f}" for n, t in zip(["isaac"] + names, totals)))
    for n in names:
        only_mj = sorted(set(models[n]["bodies"]) - set(isaac["bodies"]))
        only_is = sorted(set(isaac["bodies"]) - set(models[n]["bodies"]))
        print(f"{n}: {models[n]['nu']} actuators; bodies only in MuJoCo {only_mj}, only in Isaac {only_is}")

    print("\nBODIES  mass kg | per MJCF: mass, |COM diff| mm, max principal inertia diff %, "
          "FK diff zero/default pose mm")
    for body, ib in isaac["bodies"].items():
        cells = []
        for n in names:
            mb = models[n]["bodies"].get(body)
            if mb is None:
                cells.append(f"{'-':>34}")
                continue
            com = np.linalg.norm(np.subtract(mb["com"], ib["com"])) * 1000
            inertia = np.max(np.abs(np.subtract(mb["inertia"], ib["inertia"])) / np.maximum(ib["inertia"], 1e-9)) * 100
            fk0 = np.linalg.norm(np.subtract(mb["fk_zero"], ib["fk_zero"])) * 1000
            fk1 = np.linalg.norm(np.subtract(mb["fk_default"], ib["fk_default"])) * 1000
            cells.append(f"{mb['mass']:7.3f} {com:5.1f} {inertia:5.0f}% {fk0:5.1f}/{fk1:5.1f}")
        print(f"{body:28s} {ib['mass']:7.3f} | " + " | ".join(cells))

    print("\nCOLLISION SHAPES  bounding box of all shapes of a body in the body frame (cm)")
    for body, ib in isaac["bodies"].items():
        enabled = [s for s in ib["shapes"] if s["enabled"] and "aabb" in s]
        skipped = len(ib["shapes"]) - len(enabled)
        print(body + (f"  ({skipped} disabled or unsupported Isaac shapes)" if skipped else ""))
        isaac_kinds = kinds_text([isaac_kind(s) for s in enabled])
        print(f"  {'isaac':14s} {bbox_text([s['aabb'] for s in enabled])}  {isaac_kinds}")
        for n in names:
            mb = models[n]["bodies"].get(body)
            if mb is None:
                print(f"  {n:14s} missing")
                continue
            aabbs, kinds = [s["aabb"] for s in mb["shapes"]], [s["type"] for s in mb["shapes"]]
            print(f"  {n:14s} {bbox_text(aabbs)}  {kinds_text(kinds)}")

    print("\nJOINTS  range rad | armature | effort limit Nm "
          "(Isaac: also drive stiffness and damping; MuJoCo: passive damping, friction loss)")
    for joint, ij in isaac["joints"].items():
        print(f"{joint:28s} isaac  [{ij['range'][0]:7.3f},{ij['range'][1]:7.3f}] arm {ij['armature'] or 0:.4f} "
              f"eff {ij['effort_limit'] or 0:6.1f} kp {ij['stiffness'] or 0:6.1f} kd {ij['damping'] or 0:5.2f}")
        for n in names:
            mj = models[n]["joints"].get(joint)
            if mj is None:
                print(f"{'':28s} {n}: missing")
                continue
            rng = f"[{mj['range'][0]:7.3f},{mj['range'][1]:7.3f}]" if mj["range"] else "unlimited"
            eff = f"{mj['effort_limit']:6.1f}" if mj["effort_limit"] is not None else "     -"
            print(f"{'':28s} {n[:12]:12s} {rng} arm {mj['armature']:.4f} eff {eff} "
                  f"damp {mj['damping']:.2f} fric {mj['frictionloss']:.2f}")


if __name__ == "__main__":
    main()
