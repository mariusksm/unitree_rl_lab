"""Check clips against the training asset: joint order and body order of the .npz must match our G1.

Puts sampled frames of every clip kinematically onto UNITREE_G1_29DOF_MIMIC_CFG (root pose and ``joint_pos`` as
they are fed to the MotionCommand) and compares the asset's forward kinematics with the clip's ``body_pos_w``.
A joint column in the wrong place or another body order moves the bodies by tens of centimetres; a clip made with
our asset matches to about a millimetre. Also reports how far the clips go beyond the asset's joint limits.

    python scripts/mimic/check_npz_fk.py --headless                    # auto-discovered dataset
    python scripts/mimic/check_npz_fk.py --headless --files a.npz b.npz
"""

import argparse
import importlib.util
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--files", nargs="*", default=None, help="Clips to check (default: the auto-discovered dataset).")
parser.add_argument("--frames_per_clip", type=int, default=12)
parser.add_argument("--batch", type=int, default=256, help="Number of robots posed at once.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_MIMIC_CFG as ROBOT_CFG

_spec = importlib.util.spec_from_file_location(
    "motion_data",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "source", "unitree_rl_lab", "unitree_rl_lab",
                 "tasks", "ball_catching", "motion_data.py"),
)
motion_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(motion_data)

MAX_OK_ERROR = 0.02  # m


@configclass
class SceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def main():
    files = args_cli.files or motion_data.find_motion_data()[0]
    assert files, "no clips found"
    clips = [np.load(f) for f in files]
    samples = [(c, int(t)) for c, clip in enumerate(clips)
               for t in np.linspace(0, clip["joint_pos"].shape[0] - 1, args_cli.frames_per_clip).round()]

    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.02, device=args_cli.device))
    scene = InteractiveScene(SceneCfg(num_envs=args_cli.batch, env_spacing=3.0))
    sim.reset()
    robot = scene["robot"]
    device = sim.device
    print(f"[check] asset: {robot.num_joints} joints, {robot.num_bodies} bodies; clips: {len(files)}, samples: {len(samples)}")
    n_bodies = clips[0]["body_pos_w"].shape[1]
    if n_bodies != robot.num_bodies:
        print(f"RESULT: body count differs (clip {n_bodies}, asset {robot.num_bodies})")
        return

    body_err = torch.zeros(robot.num_bodies, device=device)
    clip_err = np.zeros(len(files))
    limits = robot.data.joint_pos_limits[0]
    soft_limits = robot.data.soft_joint_pos_limits[0]
    beyond = torch.zeros(robot.num_joints, device=device)
    beyond_soft = torch.zeros(robot.num_joints, device=device)
    for start in range(0, len(samples), args_cli.batch):
        batch = samples[start : start + args_cli.batch]
        batch = batch + [batch[-1]] * (args_cli.batch - len(batch))
        jp = torch.tensor(np.stack([clips[c]["joint_pos"][t] for c, t in batch]), dtype=torch.float32, device=device)
        bp = torch.tensor(np.stack([clips[c]["body_pos_w"][t] for c, t in batch]), dtype=torch.float32, device=device)
        bq = torch.tensor(np.stack([clips[c]["body_quat_w"][t, 0] for c, t in batch]), dtype=torch.float32, device=device)
        root = torch.zeros(args_cli.batch, 13, device=device)
        root[:, :3] = bp[:, 0] + scene.env_origins
        root[:, 3:7] = bq
        robot.write_root_state_to_sim(root)
        robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
        scene.write_data_to_sim()
        sim.forward()
        scene.update(0.0)
        fk = robot.data.body_pos_w - scene.env_origins[:, None, :]
        err = torch.linalg.norm(fk - bp, dim=-1)  # (batch, bodies)
        body_err = torch.maximum(body_err, err.amax(dim=0))
        for i, (c, _) in enumerate(batch[: len(samples) - start]):
            clip_err[c] = max(clip_err[c], float(err[i].max()))
        beyond = torch.maximum(beyond, torch.clamp(torch.maximum(limits[:, 0] - jp, jp - limits[:, 1]), min=0).amax(dim=0))
        real = jp[: len(samples) - start]
        beyond_soft += ((real < soft_limits[:, 0]) | (real > soft_limits[:, 1])).float().sum(dim=0)

    print("\nlargest forward-kinematics error per body (m):")
    for i in torch.argsort(body_err, descending=True)[:8].tolist():
        print(f"  {robot.body_names[i]:28s} {float(body_err[i]):.4f}")
    worst = np.argsort(-clip_err)[:5]
    print("worst clips:", [(os.path.basename(files[i]), round(float(clip_err[i]), 4)) for i in worst])
    print("\njoints beyond the asset limits (largest excess over all samples, rad):")
    for i in torch.argsort(beyond, descending=True)[:6].tolist():
        if float(beyond[i]) > 0:
            print(f"  {robot.joint_names[i]:28s} {float(beyond[i]):.3f}")
    share_soft = beyond_soft / len(samples)
    print("samples beyond the soft limits (the joint_limit penalty of Phase 1 / Unified starts there):")
    for i in torch.argsort(share_soft, descending=True)[:6].tolist():
        if float(share_soft[i]) > 0:
            print(f"  {robot.joint_names[i]:28s} {100 * float(share_soft[i]):5.1f} %")
    ok = float(body_err.max()) < MAX_OK_ERROR
    print("\nRESULT:", f"OK — clips match the asset (max body error {float(body_err.max()) * 1000:.1f} mm)" if ok
          else f"MISMATCH — max body error {float(body_err.max()):.3f} m: joint or body order differs from the asset")


if __name__ == "__main__":
    main()
    simulation_app.close()
