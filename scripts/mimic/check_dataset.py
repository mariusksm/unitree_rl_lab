"""Check that the dataset clips carry the right arm motion, on the right side (no hand joints in an arm).

Numpy only. Counterpart of ``check_motion.py`` (which needs the GMR ``.pkl``) for the group's dataset: every
VICON recording (``ball/<take>.npz``) holds the human's body points of the take (``human_body_pos``: hips, elbows,
palms, ...). After the time alignment of ``motion_data.align_recording`` the robot's elbows and hands of the clip
(``body_pos_w``, relative to the pelvis in its yaw frame) must follow the human's elbows and palms (relative to the
hips, scaled to the robot by one least-squares factor per clip) and stay on the same side. Finger data in an arm or
swapped arms put a hand tens of centimetres off or on the wrong side. Joint angles are no good test here: the
recording's own retarget distributes the arm motion differently over shoulder, elbow and wrist.

Also flags clips with an almost static arm (all clips, with or without recording), as ``check_motion.py`` does.

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
_spec = importlib.util.spec_from_file_location("catch_info", os.path.join(_HERE, "catch_info.py"))
catch_info = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(catch_info)
JOINTS = catch_info.G1_JOINT_NAMES

# robot body index (asset order) and the matching human body point of the recording
POINTS = {
    "left elbow": (22, "LeftForeArm"),
    "right elbow": (23, "RightForeArm"),
    "left hand": (28, "LeftPalm"),
    "right hand": (29, "RightPalm"),
}
MAX_RMSE = 0.12  # m, robot point vs scaled human point over the overlap
MIN_SAME_SIDE = 0.9  # share of the frames in which robot and human point lie on the same side of the body
MIDLINE = 0.05  # m: frames with a point closer than this to the body's midline do not count for the side check
STATIC_ARM_RANGE = 0.15  # rad: an arm whose joints all move less than this is flagged
ARMS = {side: [i for i, n in enumerate(JOINTS) if n.startswith(side) and any(k in n for k in ("shoulder", "elbow", "wrist"))]
        for side in ("left", "right")}


def _yaw(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _rot_z(p: np.ndarray, angle: np.ndarray) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.stack([c * p[..., 0] - s * p[..., 1], s * p[..., 0] + c * p[..., 1], p[..., 2]], axis=-1)


def compare_with_human(clip, rec, alignment) -> dict:
    """Per point: RMSE robot vs scaled human, share of frames on the same side, and whether the other side fits better."""
    idx = np.arange(alignment["shift"] + alignment["j0"], alignment["shift"] + alignment["j1"])
    t = alignment["window_times"][alignment["j0"] : alignment["j1"]]
    rec_t = alignment["rec_times"]
    names = [str(n) for n in rec["human_body_names"]]

    def human(name):
        return np.stack([np.interp(t, rec_t, rec["human_body_pos"][:, names.index(name), k]) for k in range(3)], -1)

    human_yaw = np.interp(t, rec_t, np.unwrap(_yaw(rec["root_rot_xyzw"][:, [3, 0, 1, 2]])))
    hips = human("Hips")
    pelvis = clip["body_pos_w"][idx, 0]
    pelvis_yaw = np.unwrap(_yaw(clip["body_quat_w"][idx, 0]))
    robot = {k: _rot_z(clip["body_pos_w"][idx, i] - pelvis, -pelvis_yaw) for k, (i, _) in POINTS.items()}
    hum = {k: _rot_z(human(n) - hips, -human_yaw) for k, (_, n) in POINTS.items()}
    scale = sum((robot[k] * hum[k]).sum() for k in POINTS) / sum((hum[k] * hum[k]).sum() for k in POINTS)
    out = {"scale": float(scale)}
    for k in POINTS:
        other = k.replace("left", "#").replace("right", "left").replace("#", "right")
        rmse = float(np.sqrt(np.mean(np.sum((robot[k] - scale * hum[k]) ** 2, -1))))
        rmse_other = float(np.sqrt(np.mean(np.sum((robot[k] - scale * hum[other]) ** 2, -1))))
        # sides only count where the point is clearly off the midline (hands meet in the middle at the catch)
        off_mid = (np.abs(robot[k][:, 1]) > MIDLINE) & (np.abs(scale * hum[k][:, 1]) > MIDLINE)
        same_side = float(np.mean(np.sign(robot[k][off_mid, 1]) == np.sign(hum[k][off_mid, 1]))) if off_mid.any() else 1.0
        out[k] = {"rmse": rmse, "same_side": same_side, "other_side_fits_better": rmse_other < rmse}
    return out


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
            results[os.path.basename(f)] = compare_with_human(clip, rec, alignment)

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
