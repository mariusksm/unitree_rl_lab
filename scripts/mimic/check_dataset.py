"""Check that the dataset clips carry the right arm motion, on the right side (no hand joints in an arm).

Numpy only. Counterpart of ``check_motion.py`` (which needs the GMR ``.pkl``) for the group's dataset: every
VICON recording (``ball/<take>.npz``) holds the human's body points of the take (``human_body_pos``: hips, elbows,
palms, ...). After the time alignment of ``motion_data.align_recording`` the robot's elbows and hands of the clip
(``body_pos_w``, relative to the pelvis in its yaw frame) must follow the human's elbows and palms (relative to the
hips, scaled to the robot by one least-squares factor per clip) and stay on the same side. Finger data in an arm or
swapped arms put a hand tens of centimetres off or on the wrong side. Joint angles are no good test here: the
recording's own retarget distributes the arm motion differently over shoulder, elbow and wrist.

Also flags clips with an almost static arm (all clips, with or without recording), as ``check_motion.py`` does.
The comparison itself (``motion_data.compare_with_human``) is part of the validity check of the Unified task's ball
recordings.

    python scripts/mimic/check_dataset.py               # auto-discovered dataset
    HREACT_MOTION_DIR=/path/to/isaaclab/g1 python scripts/mimic/check_dataset.py
"""

import importlib.util
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "motion_data",
    os.path.join(_HERE, "..", "..", "source", "unitree_rl_lab", "unitree_rl_lab", "tasks", "ball_catching", "motion_data.py"),
)
motion_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(motion_data)
JOINTS = motion_data.G1_JOINT_NAMES
POINTS = motion_data.G1_ARM_POINTS  # robot body index (asset order) and the matching human body point

MAX_RMSE = motion_data.MAX_ARM_RMSE  # m, robot point vs scaled human point over the overlap
MIN_SAME_SIDE = 0.9  # share of the frames in which robot and human point lie on the same side of the body
STATIC_ARM_RANGE = 0.15  # rad: an arm whose joints all move less than this is flagged
ARMS = {side: [i for i, n in enumerate(JOINTS) if n.startswith(side) and any(k in n for k in ("shoulder", "elbow", "wrist"))]
        for side in ("left", "right")}


def main():
    files, ball_dir, source = motion_data.find_motion_data()
    print(f"clips: {len(files)} from {source}; recordings: {ball_dir}\n")
    static, results = [], {}
    for f in files:
        clip = np.load(f)
        q = clip["joint_pos"]
        for side, idx in ARMS.items():
            if np.ptp(q[:, idx], axis=0).max() < STATIC_ARM_RANGE:
                static.append(f"{os.path.basename(f)} ({side} arm)")
        rec_file = motion_data.ball_file_for(f, ball_dir)
        if rec_file is None:
            continue
        rec = np.load(rec_file)
        alignment = motion_data.align_recording(clip, rec, JOINTS)
        if alignment is not None:
            results[os.path.basename(f)] = motion_data.compare_with_human(clip, rec, alignment)

    print(f"compared with the human in their recording: {len(results)} clips "
          f"(robot/human scale median {np.median([r['scale'] for r in results.values()]):.2f})\n")
    print(f"{'point':12s} {'RMSE med':>8s} {'p90':>6s} {'max':>6s} {'same side':>9s} {'swapped fits better':>20s}")
    bad = []
    for k in POINTS:
        rmse = np.array([r[k]["rmse"] for r in results.values()])
        side = np.array([r[k]["same_side"] for r in results.values()])
        swapped = sum(r[k]["other_side_fits_better"] for r in results.values())
        print(f"{k:12s} {np.median(rmse):8.3f} {np.percentile(rmse, 90):6.3f} {rmse.max():6.3f} {100 * side.mean():8.0f}% "
              f"{swapped:13d} clips")
        for name, r in results.items():
            if r[k]["rmse"] > MAX_RMSE or r[k]["same_side"] < MIN_SAME_SIDE or r[k]["other_side_fits_better"]:
                bad.append(f"{name}: {k} (RMSE {r[k]['rmse']:.2f} m, same side {100 * r[k]['same_side']:.0f}%)")
    if bad:
        print(f"\nclips to look at ({len(bad)}):")
        for line in bad:
            print("  " + line)
    if static:
        print(f"\nWARNING: almost static arm in {len(static)} clips: {static[:10]}")
    clean = len(results) - len({b.split(':')[0] for b in bad})
    print(f"\nRESULT: {'OK' if not static and clean == len(results) else 'CHECK'} — {clean} of {len(results)} clips follow "
          f"the human with both arms on the right side; {len(files) - len(results)} clips have no recording "
          f"(static-arm check only{', none static' if not static else ''})")


if __name__ == "__main__":
    main()
