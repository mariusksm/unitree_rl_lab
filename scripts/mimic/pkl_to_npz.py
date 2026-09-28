#!/usr/bin/env python3
"""Convert GMR retargeted .pkl files DIRECTLY to training-ready .npz motion files.

Replaces the two-step pipeline (pkl_to_csv_without_hands.py → csv_to_npz.py) with
one script and one Isaac Sim session for a whole folder.

What happens per file:
1. Load the .pkl (root_pos, root_rot in xyzw, dof_pos with 53 columns).
   The 29 G1 body joints (SDK order) are picked out — columns 0-21 and 34-40; the
   24 hand columns (22-33 left, 41-52 right) are dropped (no hands on the robot model).
2. Interpolate from the recording fps (read from the pkl, usually 30) to the
   training fps (default 50): lerp for positions, slerp for the base quaternion.
3. Differentiate to get base/joint velocities (finite differences, SO3-aware for
   the base rotation).
4. Replay the frames kinematically through the G1 model in Isaac Sim (states are
   written, no physics stepping) and record the resulting forward kinematics:
   world pose + velocity of every body.
5. Save the .npz with the keys the task's MotionLoader expects:
   fps, joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w.

Usage (inside the Isaac Lab python env):

    # whole folder, no GUI
    python scripts/mimic/pkl_to_npz.py -i pkl_isaac_lab_fixed_27_07_26 --headless

    # single file, watch the replay
    python scripts/mimic/pkl_to_npz.py -i pkl_isaac_lab_fixed_27_07_26/42_merged_filtered.pkl

Output goes to <input_folder>/npz/ by default. Pick ONE good take and copy its
.npz into source/.../tasks/ball_catching/robots/g1_29dof/phase1/motions/.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Convert GMR .pkl motions directly to .npz motion files.")
parser.add_argument(
    "--input", "-i", type=str, required=True,
    help="A .pkl file or a folder containing .pkl files.",
)
parser.add_argument(
    "--output_folder", "-o", type=str, default=None,
    help="Folder to write .npz files. Defaults to <input_folder>/npz/.",
)
parser.add_argument(
    "--input_fps", type=int, default=None,
    help="Override the recording fps. Defaults to the 'fps' stored in each .pkl.",
)
parser.add_argument("--output_fps", type=int, default=50, help="Output fps (training runs at 50 Hz).")
parser.add_argument("--overwrite", action="store_true", help="Re-convert files whose .npz already exists.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import os
import pickle
import sys
import torch

# Compatibility shim: pickles written with numpy >= 2.0 reference `numpy._core`,
# which does not exist in numpy 1.x (the Isaac Lab envs run 1.26). Alias it so
# those files load; a no-op when running under numpy >= 2.0.
if not hasattr(np, "_core"):
    import numpy.core

    sys.modules["numpy._core"] = numpy.core
    for _sub in ("multiarray", "numeric", "umath", "_multiarray_umath"):
        try:
            sys.modules[f"numpy._core.{_sub}"] = __import__(f"numpy.core.{_sub}", fromlist=[""])
        except ImportError:
            pass

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG  # only G1-29dof is supported

# GMR "unitree_g1 with hands" (53 dof). The hand joints are NOT at the end:
#   0-11 legs | 12-14 waist | 15-21 left arm | 22-33 left hand | 34-40 right arm | 41-52 right hand
GMR_BODY_COLUMNS = list(range(0, 22)) + list(range(34, 41))  # -> 29 G1 body joints in SDK order


def select_body_joints(dof_pos: np.ndarray, source: str) -> np.ndarray:
    """Return the 29 G1 body joints in SDK order from a GMR dof_pos array."""
    dof_pos = np.asarray(dof_pos)
    if dof_pos.shape[1] == 53:
        return dof_pos[:, GMR_BODY_COLUMNS]
    if dof_pos.shape[1] == 29:
        return dof_pos
    raise ValueError(f"Unexpected dof_pos width {dof_pos.shape[1]} in {source} (expected 53 or 29).")


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Minimal scene: ground, light, robot."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


class PklMotion:
    """Loads a GMR .pkl, resamples to the output fps and computes velocities."""

    def __init__(self, pkl_path: str, output_fps: int, device: torch.device, input_fps: int | None = None):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        self.input_fps = int(input_fps if input_fps is not None else data["fps"])
        self.output_fps = output_fps
        self.output_dt = 1.0 / output_fps
        self.device = device

        base_pos = torch.tensor(np.asarray(data["root_pos"]), dtype=torch.float32, device=device)
        base_rot_xyzw = torch.tensor(np.asarray(data["root_rot"]), dtype=torch.float32, device=device)
        base_rot = base_rot_xyzw[:, [3, 0, 1, 2]]  # xyzw (GMR) -> wxyz (Isaac Lab)
        dof_pos = torch.tensor(select_body_joints(data["dof_pos"], pkl_path), dtype=torch.float32, device=device)

        self.input_frames = base_pos.shape[0]
        self.duration = (self.input_frames - 1) / self.input_fps

        # -- interpolate to output fps
        times = torch.arange(0, self.duration, self.output_dt, device=device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._frame_blend(times)
        self.base_pos = self._lerp(base_pos[index_0], base_pos[index_1], blend.unsqueeze(1))
        self.base_rot = self._slerp(base_rot[index_0], base_rot[index_1], blend)
        self.dof_pos = self._lerp(dof_pos[index_0], dof_pos[index_1], blend.unsqueeze(1))

        # -- finite-difference velocities
        self.base_lin_vel = torch.gradient(self.base_pos, spacing=self.output_dt, dim=0)[0]
        self.dof_vel = torch.gradient(self.dof_pos, spacing=self.output_dt, dim=0)[0]
        self.base_ang_vel = self._so3_derivative(self.base_rot, self.output_dt)

    def _frame_blend(self, times: torch.Tensor):
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1, device=self.device))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    @staticmethod
    def _lerp(a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        return a * (1 - blend) + b * blend

    @staticmethod
    def _slerp(a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(a)
        for i in range(a.shape[0]):
            out[i] = quat_slerp(a[i], b[i], blend[i])
        return out

    @staticmethod
    def _so3_derivative(rotations: torch.Tensor, dt: float) -> torch.Tensor:
        q_prev, q_next = rotations[:-2], rotations[2:]
        omega = axis_angle_from_quat(quat_mul(q_next, quat_conjugate(q_prev))) / (2.0 * dt)
        return torch.cat([omega[:1], omega, omega[-1:]], dim=0)


def replay_and_record(sim: SimulationContext, scene: InteractiveScene, motion: PklMotion) -> dict:
    """Kinematically replay the motion and record forward-kinematics body states."""
    robot = scene["robot"]
    joint_indexes = robot.find_joints(scene.cfg.robot.joint_sdk_names, preserve_order=True)[0]

    log = {
        "fps": [motion.output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }

    for idx in range(motion.output_frames):
        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion.base_pos[idx]
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion.base_rot[idx]
        root_states[:, 7:10] = motion.base_lin_vel[idx]
        root_states[:, 10:] = motion.base_ang_vel[idx]
        robot.write_root_state_to_sim(root_states)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, joint_indexes] = motion.dof_pos[idx]
        joint_vel[:, joint_indexes] = motion.dof_vel[idx]
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        sim.render()  # kinematic replay — no physics stepping
        scene.update(sim.get_physics_dt())

        if not args_cli.headless:
            pos_lookat = root_states[0, :3].cpu().numpy()
            sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

        log["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy().copy())
        log["joint_vel"].append(robot.data.joint_vel[0].cpu().numpy().copy())
        log["body_pos_w"].append(robot.data.body_pos_w[0].cpu().numpy().copy())
        log["body_quat_w"].append(robot.data.body_quat_w[0].cpu().numpy().copy())
        log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0].cpu().numpy().copy())
        log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0].cpu().numpy().copy())

    for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        log[k] = np.stack(log[k], axis=0)
    return log


def main():
    # resolve inputs
    if os.path.isdir(args_cli.input):
        pkl_files = sorted(
            os.path.join(args_cli.input, f) for f in os.listdir(args_cli.input) if f.endswith(".pkl")
        )
        default_out = os.path.join(args_cli.input, "npz")
    else:
        pkl_files = [args_cli.input]
        default_out = os.path.join(os.path.dirname(args_cli.input) or ".", "npz")
    if not pkl_files:
        print(f"No .pkl files found in {args_cli.input}")
        return

    out_dir = args_cli.output_folder or default_out
    os.makedirs(out_dir, exist_ok=True)

    # one sim session for all files
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    print(f"[INFO] Setup complete. Converting {len(pkl_files)} file(s) → {out_dir}")

    for i, pkl_path in enumerate(pkl_files):
        npz_path = os.path.join(out_dir, os.path.basename(pkl_path).replace(".pkl", ".npz"))
        if os.path.exists(npz_path) and not args_cli.overwrite:
            print(f"  [{i + 1}/{len(pkl_files)}] SKIP (exists): {os.path.basename(npz_path)}")
            continue
        if not simulation_app.is_running():
            print("[WARN] Simulation app closed — stopping early.")
            break

        motion = PklMotion(pkl_path, args_cli.output_fps, sim.device, input_fps=args_cli.input_fps)
        log = replay_and_record(sim, scene, motion)
        np.savez(npz_path, **log)
        print(
            f"  [{i + 1}/{len(pkl_files)}] {os.path.basename(pkl_path)} → {os.path.basename(npz_path)}  "
            f"({motion.input_frames} frames @{motion.input_fps}fps → "
            f"{motion.output_frames} frames @{motion.output_fps}fps, {motion.duration:.2f}s)"
        )

    print(f"\n[DONE] npz files in: {out_dir}")
    print(
        "Next: pick ONE good take, verify it with scripts/mimic/replay_npz.py, then copy it to\n"
        "  source/unitree_rl_lab/unitree_rl_lab/tasks/ball_catching/robots/g1_29dof/phase1/motions/\n"
        "and set CATCH_MOTION_TIME in the unified task config to the ball-contact moment of that clip."
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
