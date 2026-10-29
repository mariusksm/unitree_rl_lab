"""Do the robot's collision shapes overlap each other in the poses of the reference clips?

Kinematic check in MuJoCo with the collision shapes of the Isaac asset (``model.isaac_collision_spec``): every clip
frame's joint angles are set and every pair of shapes on bodies that are not parent and child (PhysX filters those)
is measured. A negative distance is an overlap: with self-collisions on, PhysX pushes such shapes apart, so the robot
cannot take that pose. ``--focus`` limits the pairs to shapes on bodies whose name contains one of the given parts.

    python sim2sim/tools/check_self_collision.py isaac_g1_forearm.json ~/unitree_mjcf/g1 ~/imitation-data/isaaclab/g1 \\
        --focus elbow,wrist_pitch
"""

import argparse
import glob
import os
import sys
from collections import defaultdict

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import (  # noqa: E402  # isort: skip
    PHYSX_HULL_VERTEX_LIMIT,
    isaac_collision_spec,
    load_isaac_dump,
    self_collision_pairs,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("isaac_json")
    parser.add_argument("mjcf_dir")
    parser.add_argument("clip_dir")
    parser.add_argument("--focus", type=str, default=None, help="Comma-separated body name parts (elbow,wrist_pitch)")
    parser.add_argument("--full_hulls", action="store_true", help="Exact convex hulls instead of PhysX's 64 vertices.")
    parser.add_argument("--stride", type=int, default=1, help="Use every n-th clip frame.")
    parser.add_argument("--overlap_mm", type=float, default=2.0, help="Count an overlap from this depth on.")
    args = parser.parse_args()

    isaac = load_isaac_dump(args.isaac_json)
    limit = None if args.full_hulls else PHYSX_HULL_VERTEX_LIMIT
    m = isaac_collision_spec(args.mjcf_dir, isaac, floor=False, hull_vertex_limit=limit).compile()
    d = mujoco.MjData(m)
    body = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])  # noqa: E731
    pairs = self_collision_pairs(m)
    if args.focus:
        parts = args.focus.split(",")
        pairs = [(a, b) for a, b in pairs if any(p in body(a) or p in body(b) for p in parts)]
    qadr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in isaac["joints"]]
    print(f"{len(pairs)} shape pairs, forearm colliders: {isaac.get('forearm_colliders')}, hull vertices: {limit}")
    fromto = np.zeros(6)
    for label, pose in (("zero pose", {}), ("default pose", {j: v["default"] for j, v in isaac["joints"].items()})):
        d.qpos[:] = 0
        d.qpos[:7] = [0, 0, 1, 1, 0, 0, 0]
        for j, q in pose.items():
            d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]] = q
        mujoco.mj_kinematics(m, d)
        hits = [(body(a), body(b), mujoco.mj_geomDistance(m, d, a, b, 0.01, fromto)) for a, b in pairs]
        hits = [f"{x} x {y} {-z * 1000:.1f} mm" for x, y, z in hits if z < -args.overlap_mm / 1000]
        print(f"{label}: {hits if hits else 'no overlap'}")

    frames, deepest = 0, defaultdict(lambda: (0.0, ""))
    overlapping = defaultdict(int)
    for path in sorted(glob.glob(os.path.join(args.clip_dir, "*.npz"))):
        clip = os.path.basename(path).split("_")[0]
        joint_pos = np.load(path)["joint_pos"][:: args.stride]
        for t, q in enumerate(joint_pos):
            d.qpos[:7] = [0, 0, 1, 1, 0, 0, 0]
            d.qpos[qadr] = q
            mujoco.mj_kinematics(m, d)
            frames += 1
            for a, b in pairs:
                dist = mujoco.mj_geomDistance(m, d, a, b, 0.01, fromto)
                if dist < -args.overlap_mm / 1000:
                    key = (body(a), body(b))
                    overlapping[key] += 1
                    if -dist > deepest[key][0]:
                        deepest[key] = (-dist, f"clip {clip} frame {t * args.stride}")

    print(f"{frames} frames of {len(glob.glob(os.path.join(args.clip_dir, '*.npz')))} clips")
    if not overlapping:
        print(f"no overlap deeper than {args.overlap_mm} mm")
    for key, n in sorted(overlapping.items(), key=lambda kv: -kv[1]):
        depth, where = deepest[key]
        share = 100 * n / frames
        print(f"{key[0]:24s} x {key[1]:24s} {share:5.1f} % of frames, deepest {depth * 1000:5.1f} mm ({where})")


if __name__ == "__main__":
    main()
