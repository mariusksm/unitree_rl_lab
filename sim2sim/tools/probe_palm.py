"""Ball on the palm in Isaac: where does the hand collider end above the palm?

The robot hangs fixed at 1 m with stiff arms (as in ``probe_arm_contacts.py``), forearms forward and palms turned up.
Without gravity, the ball of the ball tasks is placed on each palm's normal through its palm point
(``rubber_hand.PALM_POINTS`` / ``PALM_NORMALS``) at centre offsets of 7.6–10 cm and left for 0.2 s. A collider that
reaches above that point pushes the ball out: the final offset is where the collider ends. MuJoCo, same rubber hand
mesh: one convex hull over the whole hand (the USD's own collider) ends at 9.2 cm, the split hand at the ball radius
(7.5 cm). Run in the Isaac Lab environment:

    python sim2sim/tools/probe_palm.py --headless [--ball_asset]
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--ball_asset", action="store_true", help="Probe robot_assets.ROBOT_CFG of the ball tasks.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import os  # noqa: E402

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import ArticulationCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_MIMIC_CFG  # noqa: E402
from unitree_rl_lab.tasks.ball_catching import rubber_hand  # noqa: E402
from unitree_rl_lab.tasks.ball_catching.robot_assets import ROBOT_CFG  # noqa: E402
from unitree_rl_lab.tasks.ball_catching.robots.g1_29dof.phase2.ball_catch_env_cfg import (  # noqa: E402
    RobotSceneCfg as Phase2SceneCfg,
)

PHASE2 = Phase2SceneCfg(num_envs=1, env_spacing=2.5)
ROBOT = ROBOT_CFG if args_cli.ball_asset else UNITREE_G1_29DOF_MIMIC_CFG
ARM_POSE = {  # every joint matched once: upper arms forward, forearms horizontal, palms up
    ".*_hip_.*": 0.0,
    ".*_knee_joint": 0.0,
    ".*_ankle_.*": 0.0,
    "waist_.*": 0.0,
    ".*_shoulder_pitch_joint": -0.8,
    "left_shoulder_roll_joint": 0.15,
    "right_shoulder_roll_joint": -0.15,
    ".*_shoulder_yaw_joint": 0.0,
    ".*_elbow_joint": 0.8,
    "left_wrist_roll_joint": -1.57,
    "right_wrist_roll_joint": 1.57,
    ".*_wrist_pitch_joint": 0.0,
    ".*_wrist_yaw_joint": 0.0,
}
OFFSETS = (0.076, 0.080, 0.085, 0.090, 0.095, 0.100)  # ball centre above the palm point along the normal (m)
STEPS = 40  # 0.2 s per placement


@configclass
class ProbeSceneCfg(InteractiveSceneCfg):
    terrain = PHASE2.terrain
    robot = ROBOT.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=ROBOT.spawn.replace(articulation_props=ROBOT.spawn.articulation_props.replace(fix_root_link=True)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0), joint_pos=ARM_POSE),
    )
    ball = PHASE2.ball.replace(
        spawn=PHASE2.ball.spawn.replace(
            rigid_props=PHASE2.ball.spawn.rigid_props.replace(disable_gravity=True)
        )
    )


def main():
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=0.005, device=args_cli.device, physics_material=PHASE2.terrain.physics_material)
    )
    scene = InteractiveScene(ProbeSceneCfg(num_envs=1, env_spacing=2.5))
    sim.reset()
    robot, ball = scene["robot"], scene["ball"]
    robot.write_joint_stiffness_to_sim(1000.0)
    robot.write_joint_damping_to_sim(50.0)
    dt = sim.get_physics_dt()

    def step():
        robot.set_joint_position_target(robot.data.default_joint_pos)
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(dt)

    for _ in range(100):  # settle
        step()
    asset = "ball tasks" if args_cli.ball_asset else "plain USD"
    print(f"[palm] asset: {asset}, spawn {type(ROBOT.spawn).__name__}", flush=True)
    for i, side in enumerate(("left", "right")):
        body = robot.body_names.index(f"{side}_wrist_yaw_link")
        pos, quat = robot.data.body_link_pos_w[0, body], robot.data.body_link_quat_w[0, body]
        palm = pos + quat_apply(quat, torch.tensor(rubber_hand.PALM_POINTS[i], device=sim.device))
        normal = quat_apply(quat, torch.tensor(rubber_hand.PALM_NORMALS[i], device=sim.device))
        print(f"[palm] {side}: palm point {palm.tolist()}, normal {[round(v, 3) for v in normal.tolist()]}", flush=True)
        for offset in OFFSETS:
            state = torch.zeros(1, 13, device=sim.device)
            state[0, :3] = palm + offset * normal
            state[0, 3] = 1.0
            ball.write_root_state_to_sim(state)
            for _ in range(STEPS):
                step()
            rel = ball.data.root_pos_w[0] - palm
            along = float(torch.dot(rel, normal))
            across = float(torch.norm(rel - along * normal))
            print(f"[palm]   placed {offset * 100:5.1f} cm -> {along * 100:5.1f} cm along the normal, "
                  f"{across * 100:4.1f} cm sideways", flush=True)
            park = torch.zeros(1, 13, device=sim.device)
            park[0, :3] = torch.tensor([0.0, 0.0, 5.0], device=sim.device)
            park[0, 3] = 1.0
            ball.write_root_state_to_sim(park)
            for _ in range(10):
                step()


if __name__ == "__main__":
    main()
    os._exit(0)  # Kit's shutdown hangs after CPU-physics runs
