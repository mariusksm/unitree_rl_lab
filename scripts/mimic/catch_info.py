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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", type=str, default=None, help="Write the per-clip results to this file.")
    args = parser.parse_args()

    files, ball_dir, source = motion_data.find_motion_data()
    print(f"clips: {len(files)} from {source}; ball recordings: {ball_dir}")
    fps = {float(np.asarray(np.load(f)["fps"]).reshape(-1)[0]) for f in files}
    print(f"clip frame rates: {sorted(fps)} (the tasks need 50 Hz)")
    infos = motion_data.catch_infos(files, ball_dir, motion_data.G1_JOINT_NAMES, motion_data.G1_HAND_BODY_INDEXES)

    print(f"\n{'clip':28s} {'ok':3s} {'catch s':>7s} {'flight s':>8s} {'clip s':>6s} {'hands m':>7s} {'rmse':>5s} "
          f"{'arms m':>6s}  reason")
    for f, info in zip(files, infos):
        def fmt(key, spec):
            return format(info[key], spec) if key in info else "-"
        print(
            f"{os.path.basename(f):28s} {'yes' if info['valid'] else 'no':3s} {fmt('catch_time', '7.2f')} "
            f"{fmt('flight_time', '8.2f')} {fmt('clip_duration', '6.2f')} {fmt('ball_hands_distance', '7.2f')} "
            f"{fmt('joint_rmse', '5.2f')} {fmt('arm_rmse', '6.2f')}  {info['reason']}"
        )

    valid = [i for i in infos if i["valid"]]
    print(f"\nvalid catches: {len(valid)} of {len(files)} clips")
    for key in ("catch_time", "launch_time", "flight_time", "ball_hands_distance", "joint_rmse", "arm_rmse"):
        values = [i[key] for i in valid]
        if values:
            print(f"  {key:20s} min {min(values):6.2f}  median {statistics.median(values):6.2f}  max {max(values):6.2f}")
    if valid:
        heights = [i["catch_point"][2] for i in valid]
        print(f"  {'catch height m':20s} min {min(heights):6.2f}  median {statistics.median(heights):6.2f}  max {max(heights):6.2f}")
        print(f"  (clip frame; Phase 1 and Unified lift clips and catch points by {motion_data.TEMP_CLIP_LIFT} m, see "
              "motion_data.TEMP_CLIP_LIFT)")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({os.path.basename(p): i for p, i in zip(files, infos)}, f, indent=1)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
