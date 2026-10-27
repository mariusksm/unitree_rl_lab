"""Print the catch timing and catch point that the Unified task derives from the paired ball recordings.

Numpy only (no Isaac Sim). Uses the same discovery and alignment as the task
(``tasks/ball_catching/motion_data.py``):

    python scripts/mimic/catch_info.py                 # auto-discovered dataset
    python scripts/mimic/catch_info.py --json out.json # also write the per-clip results
    HREACT_MOTION_DIR=/path/to/isaaclab/g1 python scripts/mimic/catch_info.py
"""

import argparse
import importlib.util
import json
import os
import statistics

import numpy as np

_MODULE = os.path.join(
    os.path.dirname(__file__), "..", "..", "source", "unitree_rl_lab", "unitree_rl_lab", "tasks", "ball_catching",
    "motion_data.py",
)
_spec = importlib.util.spec_from_file_location("motion_data", _MODULE)
motion_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(motion_data)

# Isaac articulation order of the G1 29-DoF asset (UNITREE_G1_29DOF_MIMIC_CFG)
G1_JOINT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint", "left_hip_roll_joint", "right_hip_roll_joint",
    "waist_roll_joint", "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint", "left_knee_joint",
    "right_knee_joint", "left_shoulder_pitch_joint", "right_shoulder_pitch_joint", "left_ankle_pitch_joint",
    "right_ankle_pitch_joint", "left_shoulder_roll_joint", "right_shoulder_roll_joint", "left_ankle_roll_joint",
    "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint", "left_elbow_joint",
    "right_elbow_joint", "left_wrist_roll_joint", "right_wrist_roll_joint", "left_wrist_pitch_joint",
    "right_wrist_pitch_joint", "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]
G1_HAND_BODY_INDEXES = [28, 29]  # left_wrist_yaw_link, right_wrist_yaw_link in the asset's body order


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", type=str, default=None, help="Write the per-clip results to this file.")
    args = parser.parse_args()

    files, ball_dir, source = motion_data.find_motion_data()
    print(f"clips: {len(files)} from {source}; ball recordings: {ball_dir}")
    fps = {float(np.asarray(np.load(f)["fps"]).reshape(-1)[0]) for f in files}
    print(f"clip frame rates: {sorted(fps)} (the tasks need 50 Hz)")
    infos = motion_data.catch_infos(files, ball_dir, G1_JOINT_NAMES, G1_HAND_BODY_INDEXES)

    print(f"\n{'clip':28s} {'ok':3s} {'catch s':>7s} {'flight s':>8s} {'clip s':>6s} {'hands m':>7s} {'rmse':>5s}  reason")
    for f, info in zip(files, infos):
        def fmt(key, spec):
            return format(info[key], spec) if key in info else "-"
        print(
            f"{os.path.basename(f):28s} {'yes' if info['valid'] else 'no':3s} {fmt('catch_time', '7.2f')} "
            f"{fmt('flight_time', '8.2f')} {fmt('clip_duration', '6.2f')} {fmt('ball_hands_distance', '7.2f')} "
            f"{fmt('joint_rmse', '5.2f')}  {info['reason']}"
        )

    valid = [i for i in infos if i["valid"]]
    print(f"\nvalid catches: {len(valid)} of {len(files)} clips")
    for key in ("catch_time", "launch_time", "flight_time", "ball_hands_distance", "joint_rmse"):
        values = [i[key] for i in valid]
        if values:
            print(f"  {key:20s} min {min(values):6.2f}  median {statistics.median(values):6.2f}  max {max(values):6.2f}")
    if valid:
        heights = [i["catch_point"][2] for i in valid]
        print(f"  {'catch height m':20s} min {min(heights):6.2f}  median {statistics.median(heights):6.2f}  max {max(heights):6.2f}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({os.path.basename(p): i for p, i in zip(files, infos)}, f, indent=1)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
