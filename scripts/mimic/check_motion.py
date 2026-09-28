#!/usr/bin/env python3
"""Check that a converted .npz contains the right joint data from its GMR .pkl.

Pure numpy, no Isaac Sim needed. For every G1 body joint it compares the .pkl column
(after picking the body joints) with the matching .npz column (Isaac joint order),
resampled to the same time base, and flags joints that do not match or barely move.

Usage:
    python scripts/mimic/check_motion.py --pkl <file>.pkl --npz <file>.npz
    python scripts/mimic/check_motion.py --pkl <file>.pkl            # only inspect the .pkl
"""

import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pkl_to_csv_without_hands import GMR_BODY_COLUMNS, select_body_joints  # noqa: E402

SDK_JOINT_NAMES = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
    "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
    "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]  # fmt: skip

# joint order of the G1 29dof articulation in Isaac Lab (g1_29dof_rev_1_0.usd), i.e. the .npz column order
ISAAC_JOINT_NAMES = [
    "left_hip_pitch", "right_hip_pitch", "waist_yaw", "left_hip_roll", "right_hip_roll", "waist_roll",
    "left_hip_yaw", "right_hip_yaw", "waist_pitch", "left_knee", "right_knee",
    "left_shoulder_pitch", "right_shoulder_pitch", "left_ankle_pitch", "right_ankle_pitch",
    "left_shoulder_roll", "right_shoulder_roll", "left_ankle_roll", "right_ankle_roll",
    "left_shoulder_yaw", "right_shoulder_yaw", "left_elbow", "right_elbow",
    "left_wrist_roll", "right_wrist_roll", "left_wrist_pitch", "right_wrist_pitch",
    "left_wrist_yaw", "right_wrist_yaw",
]  # fmt: skip


def load_pkl(path):
    # pickles written with numpy >= 2 reference numpy._core
    if not hasattr(np, "_core"):
        import numpy.core

        sys.modules["numpy._core"] = numpy.core
        for sub in ("multiarray", "numeric", "umath", "_multiarray_umath"):
            try:
                sys.modules[f"numpy._core.{sub}"] = __import__(f"numpy.core.{sub}", fromlist=[""])
            except ImportError:
                pass
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--npz", default=None)
    parser.add_argument("--tol", type=float, default=0.05, help="Max allowed joint difference (rad).")
    args = parser.parse_args()

    data = load_pkl(args.pkl)
    raw = np.asarray(data["dof_pos"])
    body = select_body_joints(raw, args.pkl)
    fps = float(data["fps"])
    print(f".pkl: {raw.shape[0]} frames @ {fps:.0f} fps, {raw.shape[1]} dof columns -> body columns {GMR_BODY_COLUMNS}")

    npz_sdk = None
    if args.npz:
        npz = np.load(args.npz)
        npz_joint = npz["joint_pos"]
        npz_fps = float(np.asarray(npz["fps"]).reshape(-1)[0])
        print(f".npz: {npz_joint.shape[0]} frames @ {npz_fps:.0f} fps, {npz_joint.shape[1]} joints")
        # reorder npz (Isaac order) -> SDK order and resample the pkl onto the npz time base
        npz_sdk = npz_joint[:, [ISAAC_JOINT_NAMES.index(n) for n in SDK_JOINT_NAMES]]
        t_npz = np.arange(npz_joint.shape[0]) / npz_fps
        t_pkl = np.arange(body.shape[0]) / fps
        body = np.stack([np.interp(t_npz, t_pkl, body[:, j]) for j in range(29)], axis=1)

    print(f"\n{'joint':22s} {'pkl min':>8s} {'pkl max':>8s} {'range':>6s}", end="")
    print(f" {'npz min':>8s} {'npz max':>8s} {'maxdiff':>8s}  status" if npz_sdk is not None else "  status")
    problems = 0
    for j, name in enumerate(SDK_JOINT_NAMES):
        lo, hi = body[:, j].min(), body[:, j].max()
        line = f"{name:22s} {lo:8.3f} {hi:8.3f} {hi - lo:6.3f}"
        status = []
        if npz_sdk is not None:
            diff = np.abs(npz_sdk[:, j] - body[:, j]).max()
            line += f" {npz_sdk[:, j].min():8.3f} {npz_sdk[:, j].max():8.3f} {diff:8.3f}"
            if diff > args.tol:
                status.append("MISMATCH")
        if hi - lo < 0.02:
            status.append("static")
        problems += "MISMATCH" in status
        print(f"{line}  {' '.join(status)}")

    # left/right arm plausibility: both arms should move comparably in a catching clip
    left = body[:, 15:22].max(0) - body[:, 15:22].min(0)
    right = body[:, 22:29].max(0) - body[:, 22:29].min(0)
    print(f"\narm motion range (sum over 7 joints): left {left.sum():.2f} rad, right {right.sum():.2f} rad")
    if right.sum() < 0.1 * left.sum() or left.sum() < 0.1 * right.sum():
        print("WARNING: one arm is almost static — check the column selection / the recording.")
    if npz_sdk is not None:
        print("RESULT:", "OK — npz matches pkl" if problems == 0 else f"{problems} joint(s) differ from the pkl")


if __name__ == "__main__":
    main()
