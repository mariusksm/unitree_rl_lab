#!/usr/bin/env python3
"""Convert GMR retargeted .pkl files to CSV format for Phase 1 training.

GMR outputs .pkl files with 53 joint columns. The hand joints are NOT at the end:

    0-11 legs | 12-14 waist | 15-21 left arm | 22-33 left hand | 34-40 right arm | 41-52 right hand

This script extracts the 29 G1 body joints (in SDK order) and writes them in the format
expected by csv_to_npz.py:

    base_x, base_y, base_z, qx, qy, qz, qw, joint_1, ..., joint_29

Usage:
    python pkl_to_csv_without_hands.py --input_folder pkl_result/ --output_folder phase1/csv/

Then convert to .npz:
    python scripts/mimic/csv_to_npz.py -f phase1/csv/<file>.csv --input_fps 30 --output_fps 50
"""

import argparse
import os
import pickle

import numpy as np

# GMR "unitree_g1 with hands" (53 dof): legs+waist+left arm = 0-21, right arm = 34-40
GMR_BODY_COLUMNS = list(range(0, 22)) + list(range(34, 41))


def select_body_joints(dof_pos: np.ndarray, source: str) -> np.ndarray:
    """Return the 29 G1 body joints in SDK order from a GMR dof_pos array."""
    dof_pos = np.asarray(dof_pos)
    if dof_pos.shape[1] == 53:
        return dof_pos[:, GMR_BODY_COLUMNS]
    if dof_pos.shape[1] == 29:
        return dof_pos
    raise ValueError(f"Unexpected dof_pos width {dof_pos.shape[1]} in {source} (expected 53 or 29).")


def convert_pkl_to_csv(pkl_path: str, csv_path: str):
    """Read a GMR .pkl file and write a Phase 1 compatible .csv file.

    The .pkl contains:
        fps: int
        root_pos: (N, 3)  — base position in world frame
        root_rot: (N, 4)  — base orientation quaternion (xyzw format)
        dof_pos: (N, 53)  — joint positions, layout see module docstring

    The output .csv has 36 columns:
        base_x, base_y, base_z, qx, qy, qz, qw, joint_0, ..., joint_28
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    root_pos = data["root_pos"]          # (N, 3)
    root_rot = data["root_rot"]          # (N, 4)  xyzw
    dof_pos = select_body_joints(data["dof_pos"], pkl_path)  # (N, 29) G1 body joints, SDK order

    num_frames = root_pos.shape[0]
    motion = np.zeros((num_frames, 36), dtype=np.float64)
    motion[:, :3] = root_pos
    motion[:, 3:7] = root_rot
    motion[:, 7:] = dof_pos

    np.savetxt(csv_path, motion, delimiter=",")


def main():
    parser = argparse.ArgumentParser(
        description="Convert GMR .pkl retargeting results to Phase 1 CSV format."
    )
    parser.add_argument(
        "--input_folder", "-i", type=str, required=True,
        help="Folder containing .pkl files from GMR retargeting.",
    )
    parser.add_argument(
        "--output_folder", "-o", type=str, default=None,
        help="Folder to write .csv files. Defaults to <input_folder>/csv/.",
    )
    args = parser.parse_args()

    if args.output_folder is None:
        args.output_folder = os.path.join(args.input_folder, "csv")
    os.makedirs(args.output_folder, exist_ok=True)

    pkl_files = sorted(f for f in os.listdir(args.input_folder) if f.endswith(".pkl"))
    if not pkl_files:
        print(f"No .pkl files found in {args.input_folder}")
        return

    print(f"Converting {len(pkl_files)} .pkl files from {args.input_folder}")
    for i, filename in enumerate(pkl_files):
        pkl_path = os.path.join(args.input_folder, filename)
        csv_name = filename.replace(".pkl", ".csv")
        csv_path = os.path.join(args.output_folder, csv_name)
        convert_pkl_to_csv(pkl_path, csv_path)
        print(f"  [{i+1}/{len(pkl_files)}] {filename} → {csv_name}")

    print(f"\nDone. {len(pkl_files)} CSV files written to {args.output_folder}")
    print(f"\nNext: convert to .npz with:")
    print(f"  python scripts/mimic/csv_to_npz.py -f <csv_file> --input_fps 30 --output_fps 50")


if __name__ == "__main__":
    main()
