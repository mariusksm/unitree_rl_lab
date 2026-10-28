"""Reference motion clips and their recorded ball throws (Phase 1 and Unified).

Clips are BeyondMimic-style ``.npz`` files (``fps``, ``joint_pos``, ``joint_vel``, ``body_pos_w``, ``body_quat_w``,
``body_lin_vel_w``, ``body_ang_vel_w``) with joints and bodies in the Isaac articulation order of the G1 29-DoF asset.
The group's dataset (GitLab ``pg-681-h-react/data/imitation-data``, folder ``isaaclab/``) pairs every clip
``g1/<name>.npz`` with a VICON recording ``ball/<name>.npz`` of the same take. The recording carries its own
retarget (``dof_pos``, ``root_pos``, ``root_rot_xyzw`` at 120 Hz) in the VICON world frame, the ball trajectory and
the catch moment (``ball_catch_time_s``), but neither its time window nor its frame matches the clip.
:func:`compute_catch_info` aligns the two and checks the alignment against the human body points of the recording
(:func:`compare_with_human`).

Clip directory search order (first hit wins):

1. ``$HREACT_MOTION_DIR`` (ball recordings: ``$HREACT_BALL_DIR``, default ``<motion dir>/../ball``)
2. ``<repo>/data/imitation/g1`` (local copy, ``*.npz`` is gitignored)
3. ``<repo>/../imitation-data/isaaclab/g1`` (the dataset repository cloned next to this one)
4. ``phase1/motions/*.npz`` (drop-in folder of the single-clip pipeline, no ball recordings)

Only numpy is imported here, so ``scripts/mimic/catch_info.py`` can use this module without Isaac Sim.
"""

from __future__ import annotations

import glob
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[5]
PHASE1_MOTION_DIR = Path(__file__).resolve().parent / "robots" / "g1_29dof" / "phase1" / "motions"

# validity limits of an aligned recording (see compute_catch_info)
MAX_JOINT_RMSE = 0.5  # rad, joint-angle mismatch between the two retargets of the take
MAX_TRANSLATION_STD = 0.2  # m, spread of the recording-to-clip offset over the overlap
MAX_YAW_STD = 5.0  # deg
MAX_BALL_HANDS_DISTANCE = 0.3  # m, ball centre to the wrist midpoint of the clip at the catch moment
FLIGHT_TIME_RANGE = (0.3, 1.5)  # s
MIN_CATCH_TIME = 0.3  # s into the clip
MAX_ARM_RMSE = 0.10  # m, worst clip elbow/hand vs the scaled human point (clean clips <= 0.07 m, rejected 0.11-0.22 m)
ARM_MIDLINE = 0.05  # m: points closer than this to the body's midline do not count for the side check

# G1 29-DoF asset (UNITREE_G1_29DOF_MIMIC_CFG): joint order and body indices of the clips, confirmed against the asset
# by scripts/mimic/check_npz_fk.py; here for the numpy-only scripts and the task configs
G1_JOINT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint", "left_hip_roll_joint", "right_hip_roll_joint",
    "waist_roll_joint", "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint", "left_knee_joint",
    "right_knee_joint", "left_shoulder_pitch_joint", "right_shoulder_pitch_joint", "left_ankle_pitch_joint",
    "right_ankle_pitch_joint", "left_shoulder_roll_joint", "right_shoulder_roll_joint", "left_ankle_roll_joint",
    "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint", "left_elbow_joint",
    "right_elbow_joint", "left_wrist_roll_joint", "right_wrist_roll_joint", "left_wrist_pitch_joint",
    "right_wrist_pitch_joint", "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]
G1_HAND_BODY_INDEXES = [28, 29]  # left_wrist_yaw_link, right_wrist_yaw_link
# clip body index and the human body point of the recording it is compared with (compare_with_human)
G1_ARM_POINTS = {
    "left elbow": (22, "LeftForeArm"),
    "right elbow": (23, "RightForeArm"),
    "left hand": (28, "LeftPalm"),
    "right hand": (29, "RightPalm"),
}


def is_lfs_pointer(path: str | os.PathLike) -> bool:
    """Whether ``path`` is a Git-LFS pointer file instead of the real content."""
    try:
        if os.path.getsize(path) > 1024:
            return False
        with open(path, "rb") as f:
            return f.read(40).startswith(b"version https://git-lfs")
    except OSError:
        return False


def _npz_files(directory: str | os.PathLike) -> list[str]:
    files = sorted(glob.glob(os.path.join(directory, "*.npz")))
    pointers = [f for f in files if is_lfs_pointer(f)]
    if pointers:
        raise RuntimeError(
            f"{len(pointers)} of {len(files)} .npz files in {directory} are Git-LFS pointer files, not data. "
            "Run `git lfs pull` in the dataset repository (or copy the real files)."
        )
    return files


def find_motion_data() -> tuple[list[str], str | None, str]:
    """Locate the reference clips.

    Returns:
        (clip files, directory of the paired ball recordings or ``None``, human-readable source).
        The clip list is empty if nothing was found.
    """
    candidates: list[tuple[str, str | None, str]] = []
    if os.environ.get("HREACT_MOTION_DIR"):
        candidates.append((os.environ["HREACT_MOTION_DIR"], os.environ.get("HREACT_BALL_DIR"), "$HREACT_MOTION_DIR"))
    candidates.append((str(REPO_ROOT / "data" / "imitation" / "g1"), None, "data/imitation"))
    candidates.append((str(REPO_ROOT.parent / "imitation-data" / "isaaclab" / "g1"), None, "../imitation-data"))
    for directory, ball_dir, label in candidates:
        if not os.path.isdir(directory):
            continue
        files = _npz_files(directory)
        if files:
            ball_dir = ball_dir or os.path.join(os.path.dirname(os.path.normpath(directory)), "ball")
            return files, (ball_dir if os.path.isdir(ball_dir) else None), f"{label} ({directory})"
    if PHASE1_MOTION_DIR.is_dir():
        files = _npz_files(PHASE1_MOTION_DIR)
        if files:
            return files, None, f"phase1/motions ({PHASE1_MOTION_DIR})"
    return [], None, "none"


def ball_file_for(clip_file: str, ball_dir: str | None) -> str | None:
    """The recording that shares the clip's file name, if any."""
    if ball_dir is None:
        return None
    path = os.path.join(ball_dir, os.path.basename(clip_file))
    return path if os.path.isfile(path) else None


def _yaw_wxyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _rot_z(p: np.ndarray, angle: np.ndarray | float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.stack([c * p[..., 0] - s * p[..., 1], s * p[..., 0] + c * p[..., 1], p[..., 2]], axis=-1)


def align_recording(clip: np.lib.npyio.NpzFile, rec: np.lib.npyio.NpzFile, joint_names: Sequence[str]) -> dict | None:
    """Time alignment of a recording with its clip by the best joint-angle match.

    Returns ``None`` if the recording lacks clip joints or does not overlap the clip, else a dict with ``shift``
    (clip frame of the recording's first resampled frame), ``fps``, ``window_times`` / ``window_q`` (the recording's
    joints resampled to the clip's frame period, in the clip's joint order), the overlap ``j0:j1`` in window frames
    and the ``rmse`` over it.
    """
    fps = float(np.asarray(clip["fps"]).reshape(-1)[0])
    joint_pos = clip["joint_pos"].astype(np.float64)
    num_frames = joint_pos.shape[0]
    rec_fps = float(np.asarray(rec["fps"]).reshape(-1)[0])
    rec_names = [str(n) for n in rec["joint_names"]]
    if any(n not in rec_names for n in joint_names):
        return None
    rec_q = rec["dof_pos"][:, [rec_names.index(n) for n in joint_names]].astype(np.float64)
    rec_times = np.arange(rec_q.shape[0]) / rec_fps
    window_times = np.arange(0.0, rec_times[-1] + 1e-9, 1.0 / fps)
    window_q = np.stack([np.interp(window_times, rec_times, rec_q[:, k]) for k in range(rec_q.shape[1])], axis=-1)
    num_window = len(window_times)

    best_rmse, best_shift = np.inf, None
    for shift in range(-num_window + 1, num_frames):  # window frame j sits at clip frame shift + j
        j0, j1 = max(0, -shift), min(num_window, num_frames - shift)
        if j1 - j0 < num_window // 2:
            continue
        err = window_q[j0:j1] - joint_pos[shift + j0 : shift + j1]
        rmse = float(np.sqrt(np.mean(err**2)))
        if rmse < best_rmse:
            best_rmse, best_shift = rmse, shift
    if best_shift is None:
        return None
    j0, j1 = max(0, -best_shift), min(num_window, num_frames - best_shift)
    return {"shift": best_shift, "fps": fps, "rec_fps": rec_fps, "rec_times": rec_times, "window_times": window_times,
            "window_q": window_q, "j0": j0, "j1": j1, "rmse": best_rmse}


def compare_with_human(
    clip: np.lib.npyio.NpzFile, rec: np.lib.npyio.NpzFile, alignment: dict, points: dict = G1_ARM_POINTS
) -> dict:
    """Compare the clip's elbows and hands with the human elbows and palms of its recording over the aligned overlap.

    Robot points relative to the clip's pelvis in its yaw frame, human points relative to the hips in the recording
    root's yaw frame, scaled to the robot by one least-squares factor per clip. A wrong time alignment (or arm data in
    the wrong joints) moves the robot points away from the human ones.

    Returns:
        Dict with ``scale`` and per point (keys of ``points``) the ``rmse`` (m), the share of frames on the same side
        of the body (``same_side``, frames near the midline excluded) and whether the mirrored human point fits better
        (``other_side_fits_better``).
    """
    idx = np.arange(alignment["shift"] + alignment["j0"], alignment["shift"] + alignment["j1"])
    t = alignment["window_times"][alignment["j0"] : alignment["j1"]]
    rec_t = alignment["rec_times"]
    names = [str(n) for n in rec["human_body_names"]]

    def human(name):
        return np.stack([np.interp(t, rec_t, rec["human_body_pos"][:, names.index(name), k]) for k in range(3)], -1)

    human_yaw = np.interp(t, rec_t, np.unwrap(_yaw_wxyz(rec["root_rot_xyzw"][:, [3, 0, 1, 2]])))
    hips = human("Hips")
    pelvis = clip["body_pos_w"][idx, 0]
    pelvis_yaw = np.unwrap(_yaw_wxyz(clip["body_quat_w"][idx, 0]))
    robot = {k: _rot_z(clip["body_pos_w"][idx, i] - pelvis, -pelvis_yaw) for k, (i, _) in points.items()}
    hum = {k: _rot_z(human(n) - hips, -human_yaw) for k, (_, n) in points.items()}
    scale = sum((robot[k] * hum[k]).sum() for k in points) / sum((hum[k] * hum[k]).sum() for k in points)
    out = {"scale": float(scale)}
    for k in points:
        other = k.replace("left", "#").replace("right", "left").replace("#", "right")
        rmse = float(np.sqrt(np.mean(np.sum((robot[k] - scale * hum[k]) ** 2, -1))))
        rmse_other = float(np.sqrt(np.mean(np.sum((robot[k] - scale * hum[other]) ** 2, -1)))) if other in hum else np.inf
        # sides only count where the point is clearly off the midline (hands meet in the middle at the catch)
        off_mid = (np.abs(robot[k][:, 1]) > ARM_MIDLINE) & (np.abs(scale * hum[k][:, 1]) > ARM_MIDLINE)
        same_side = float(np.mean(np.sign(robot[k][off_mid, 1]) == np.sign(hum[k][off_mid, 1]))) if off_mid.any() else 1.0
        out[k] = {"rmse": rmse, "same_side": same_side, "other_side_fits_better": rmse_other < rmse}
    return out


def compute_catch_info(
    clip_file: str, ball_file: str, joint_names: Sequence[str], hand_body_indexes: Sequence[int]
) -> dict:
    """Align a recording with its clip; return the catch moment and point in the clip's time and frame.

    Time: the recording's window is slid over the clip in clip frames; the offset with the smallest joint-angle
    RMSE wins (both retargets come from the same take). Space: the yaw rotation and translation that map the
    recording's root onto the clip's pelvis, averaged over the overlap. Check: the clip's elbows and hands must
    follow the human ones of the recording (:func:`compare_with_human`, worst point within ``MAX_ARM_RMSE``), else
    the recorded catch point does not belong to the clip's hands. The joint-angle match alone accepted unusual time
    offsets that put the hands 0.12-0.22 m away from the human's (clips 51, 52), and clip 82's right hand drifts
    0.11 m away from the human's although its offset is normal.

    Args:
        clip_file: clip ``.npz`` (Isaac order).
        ball_file: recording ``.npz`` of the same take.
        joint_names: joint names in the clip's (Isaac) order.
        hand_body_indexes: left and right hand body indices in the clip's body order.

    Returns:
        Dict with ``valid``, ``reason`` and, when the alignment succeeded, ``catch_time`` / ``launch_time`` /
        ``flight_time`` (s into the clip), ``catch_point`` (clip frame, m), ``ball_hands_distance`` (m) and the
        alignment diagnostics ``joint_rmse``, ``arm_rmse``, ``yaw_std_deg``, ``translation_std``, ``offset``.
    """
    clip = np.load(clip_file)
    rec = np.load(ball_file)
    alignment = align_recording(clip, rec, joint_names)
    if alignment is None:
        return {"valid": False, "reason": "recording lacks clip joints or does not overlap the clip"}
    fps, rec_fps, rec_times = alignment["fps"], alignment["rec_fps"], alignment["rec_times"]
    best_shift, best_rmse, j0, j1 = alignment["shift"], alignment["rmse"], alignment["j0"], alignment["j1"]
    num_frames = clip["joint_pos"].shape[0]
    offset = best_shift / fps  # clip time of the recording's first frame
    arms = compare_with_human(clip, rec, alignment)

    clip_idx = np.arange(best_shift + j0, best_shift + j1)
    rec_t = alignment["window_times"][j0:j1]
    rec_root = np.stack([np.interp(rec_t, rec_times, rec["root_pos"][:, k]) for k in range(3)], axis=-1)
    rec_yaw = np.interp(rec_t, rec_times, np.unwrap(_yaw_wxyz(rec["root_rot_xyzw"][:, [3, 0, 1, 2]])))
    clip_yaw = np.unwrap(_yaw_wxyz(clip["body_quat_w"][clip_idx, 0]))
    dyaw = np.angle(np.exp(1j * (clip_yaw - rec_yaw)))
    yaw = float(np.angle(np.mean(np.exp(1j * dyaw))))
    yaw_std = float(np.degrees(np.std(np.angle(np.exp(1j * (dyaw - yaw))))))

    def to_clip(p: np.ndarray) -> np.ndarray:
        return _rot_z(p, yaw)

    offsets = clip["body_pos_w"][clip_idx, 0] - to_clip(rec_root)
    translation = offsets.mean(axis=0)
    translation_std = float(np.linalg.norm(offsets.std(axis=0)))

    visible = np.nonzero(rec["ball_visible"])[0]
    if len(visible) == 0:
        return {"valid": False, "reason": "ball never visible"}
    rec_catch = float(np.asarray(rec["ball_catch_time_s"]).reshape(-1)[0])
    catch_time = rec_catch + offset
    launch_time = visible[0] / rec_fps + offset
    catch_frame = int(round(rec_catch * rec_fps))
    catch_point = to_clip(rec["ball_pos"][min(catch_frame, len(rec["ball_pos"]) - 1)]) + translation
    info = {
        "catch_time": catch_time,
        "launch_time": launch_time,
        "flight_time": catch_time - launch_time,
        "catch_point": catch_point.tolist(),
        "joint_rmse": best_rmse,
        "arm_rmse": max(arms[k]["rmse"] for k in G1_ARM_POINTS),
        "yaw_std_deg": yaw_std,
        "translation_std": translation_std,
        "offset": offset,
        "clip_duration": num_frames / fps,
    }
    clip_catch_frame = int(round(catch_time * fps))
    if 0 <= clip_catch_frame < num_frames:
        hands = clip["body_pos_w"][clip_catch_frame, list(hand_body_indexes)].mean(axis=0)
        info["ball_hands_distance"] = float(np.linalg.norm(catch_point - hands))

    reason = ""
    if best_rmse > MAX_JOINT_RMSE:
        reason = f"joint RMSE {best_rmse:.2f} rad"
    elif translation_std > MAX_TRANSLATION_STD or yaw_std > MAX_YAW_STD:
        reason = f"inconsistent frame offset ({translation_std:.2f} m, {yaw_std:.1f} deg)"
    elif not MIN_CATCH_TIME <= catch_time < info["clip_duration"]:
        reason = f"catch at {catch_time:.2f} s outside the clip ({info['clip_duration']:.2f} s)"
    elif not FLIGHT_TIME_RANGE[0] <= info["flight_time"] <= FLIGHT_TIME_RANGE[1]:
        reason = f"flight time {info['flight_time']:.2f} s"
    elif info.get("ball_hands_distance", np.inf) > MAX_BALL_HANDS_DISTANCE:
        reason = f"ball {info.get('ball_hands_distance', np.inf):.2f} m from the hands at the catch"
    elif info["arm_rmse"] > MAX_ARM_RMSE:
        reason = f"arms {info['arm_rmse']:.2f} m off the human of the recording"
    info["valid"] = not reason
    info["reason"] = reason
    return info


def catch_infos(
    clip_files: Sequence[str], ball_dir: str | None, joint_names: Sequence[str], hand_body_indexes: Sequence[int]
) -> list[dict]:
    """:func:`compute_catch_info` for every clip; clips without a recording get ``valid=False``."""
    infos = []
    for clip_file in clip_files:
        ball_file = ball_file_for(clip_file, ball_dir)
        if ball_file is None:
            infos.append({"valid": False, "reason": "no ball recording"})
        else:
            infos.append(compute_catch_info(clip_file, ball_file, joint_names, hand_body_indexes))
    return infos
