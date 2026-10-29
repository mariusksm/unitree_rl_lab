"""Ball against the arm in Isaac: how close does the ball come to the real arm surface?

The robot hangs fixed at 1 m with stiff arms, upper arms 46 deg forward and forearms horizontal (clear of the upper
arm above them). The ball of the ball tasks (Phase 2 scene: r 7.5 cm, 112 g, same material and simulation step) is
dropped onto the forearm (elbow link), the wrist (wrist pitch link) and the hand of each arm: from 2 cm (impact
≈ 0.6 m/s) and from 30 cm with 0, 2, 4 and 6 m/s start speed (impact ≈ 2.4-6.4 m/s). Every physics step the ball
centre and the poses of the arm links are recorded; ``analyze_arm_contacts.py`` measures the distance to the visual
meshes, i.e. the real surface. Run in the Isaac Lab environment:

    python sim2sim/tools/probe_arm_contacts.py --headless [--forearm_colliders] --out probe.npz
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--out", type=str, required=True, help="Output .npz file.")
parser.add_argument("--forearm_colliders", action="store_true", help="Probe the asset with forearm colliders.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import os  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import ArticulationCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_MIMIC_CFG  # noqa: E402
from unitree_rl_lab.tasks.ball_catching.robot_assets import G1_FOREARM_CFG  # noqa: E402
from unitree_rl_lab.tasks.ball_catching.robots.g1_29dof.phase2.ball_catch_env_cfg import (  # noqa: E402
    RobotSceneCfg as Phase2SceneCfg,
)

PHASE2 = Phase2SceneCfg(num_envs=1, env_spacing=2.5)
ROBOT = G1_FOREARM_CFG if args_cli.forearm_colliders else UNITREE_G1_29DOF_MIMIC_CFG
ARM_POSE = {  # every joint matched once: upper arms down, forearms forward, slightly apart
    ".*_hip_.*": 0.0,
    ".*_knee_joint": 0.0,
    ".*_ankle_.*": 0.0,
    "waist_.*": 0.0,
    ".*_shoulder_pitch_joint": -0.8,
    "left_shoulder_roll_joint": 0.15,
    "right_shoulder_roll_joint": -0.15,
    ".*_shoulder_yaw_joint": 0.0,
    ".*_elbow_joint": 0.8,  # G1: 0 = forearm at 90 deg to the upper arm, +1.57 = arm straight
    ".*_wrist_.*": 0.0,
}
LINKS = ("elbow_link", "wrist_roll_link", "wrist_pitch_link", "wrist_yaw_link")
TARGETS = {"elbow_link": (0.07, 0.0, 0.0), "wrist_pitch_link": (0.03, 0.0, 0.0), "wrist_yaw_link": (0.10, 0.0, 0.01)}
# (height above the forearm top in m, start speed downward in m/s)
DROPS = ((0.02, 0.0), (0.3, 0.0), (0.3, 2.0), (0.3, 4.0), (0.3, 6.0))
ARM_TOP = 0.035  # m from the link axis to the top of the arm
STEPS = 100  # 0.5 s per test


@configclass
class ProbeSceneCfg(InteractiveSceneCfg):
    terrain = PHASE2.terrain
    robot = ROBOT.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=ROBOT.spawn.replace(articulation_props=ROBOT.spawn.articulation_props.replace(fix_root_link=True)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0), joint_pos=ARM_POSE),
    )
    ball = PHASE2.ball


def main():
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=0.005, device=args_cli.device, physics_material=PHASE2.terrain.physics_material)
    )
    scene = InteractiveScene(ProbeSceneCfg(num_envs=1, env_spacing=2.5))
    sim.reset()
    robot, ball = scene["robot"], scene["ball"]
    robot.write_joint_stiffness_to_sim(1000.0)  # stiff arms: the ball cannot push them away (strictest case)
    robot.write_joint_damping_to_sim(50.0)
    dt = sim.get_physics_dt()

    def step():
        robot.set_joint_position_target(robot.data.default_joint_pos)
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(dt)

    for _ in range(100):  # settle
        step()
    elbows = [robot.joint_names.index(f"{s}_elbow_joint") for s in ("left", "right")]
    print(f"[probe] elbow default {robot.data.default_joint_pos[0, elbows].tolist()}, "
          f"after settling {robot.data.joint_pos[0, elbows].tolist()}", flush=True)

    records = {"ball_pos": [], "link_pos": [], "link_quat": [], "side": [], "target": [], "height": [], "speed": []}
    for side in ("left", "right"):
        ids = [robot.body_names.index(f"{side}_{link}") for link in LINKS]
        for target, offset in TARGETS.items():
            body = robot.body_names.index(f"{side}_{target}")
            point = robot.data.body_link_pos_w[0, body] + quat_apply(
                robot.data.body_link_quat_w[0, body], torch.tensor(offset, device=sim.device)
            )
            for height, speed in DROPS:
                state = torch.zeros(1, 13, device=sim.device)
                lift = ARM_TOP + PHASE2.ball.spawn.radius + height
                state[0, :3] = point + torch.tensor([0.0, 0.0, lift], device=sim.device)
                state[0, 3] = 1.0
                state[0, 9] = -speed
                ball.write_root_state_to_sim(state)
                ball_pos, link_pos, link_quat = [], [], []
                for _ in range(STEPS):
                    step()
                    ball_pos.append(ball.data.root_pos_w[0].cpu().numpy().copy())
                    link_pos.append(robot.data.body_link_pos_w[0, ids].cpu().numpy().copy())
                    link_quat.append(robot.data.body_link_quat_w[0, ids].cpu().numpy().copy())
                records["ball_pos"].append(ball_pos)
                records["link_pos"].append(link_pos)
                records["link_quat"].append(link_quat)
                records["side"].append(side)
                records["target"].append(target)
                records["height"].append(height)
                records["speed"].append(speed)
                park = torch.zeros(1, 13, device=sim.device)
                park[0, :3] = torch.tensor([0.0, 0.0, 5.0], device=sim.device)
                park[0, 3] = 1.0
                ball.write_root_state_to_sim(park)
                for _ in range(10):
                    step()

    np.savez(
        args_cli.out,
        **{k: np.array(v) for k, v in records.items()},
        links=np.array(LINKS),
        radius=PHASE2.ball.spawn.radius,
        forearm_colliders=args_cli.forearm_colliders,
    )
    print(f"[probe] {len(records['side'])} drops -> {args_cli.out}", flush=True)


if __name__ == "__main__":
    main()
    os._exit(0)  # Kit's shutdown can hang; nothing to clean up
