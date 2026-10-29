"""Record the actor's ball terms in Isaac together with the true states they come from (for check_ball_obs.py).

Phase 2 env (the perception code is shared with Unified), zero actions, the final throw distribution without the
curriculum, throws after 0.5-1.5 s. Unless ``--random``, the VICON model is made deterministic so that the numpy port
must reproduce it exactly: fixed latency (3 steps) and detection delay (70 ms), no noise, bias or dropouts, and the
ball hidden in the catch box on every throw, so that coasting and losing the ball happen. Run in the Isaac Lab
environment:

    python sim2sim/tools/record_ball_obs.py --headless --num_envs 16 --steps 1000 --out ball_obs.npz
"""

import argparse
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--random", action="store_true", help="Keep the random VICON model (statistics only).")
parser.add_argument("--wide_zone", action="store_true", help="Larger catch box: the ball is hidden earlier in flight.")
parser.add_argument("--out", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import unitree_rl_lab.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

TASK = "Unitree-G1-29dof-BallCatch-Phase2"
DETERMINISTIC = {
    "obs_delay_steps_range": (3, 3),
    "detection_delay_range": (0.07, 0.07),  # not a multiple of the 20 ms step: no float32/float64 tie
    "obs_pos_noise_std": 0.0,
    "obs_vel_noise_std": 0.0,
    "obs_pos_bias": 0.0,
    "obs_dropout_prob": 0.0,
    "hide_in_catch_zone_prob": 1.0,
}
PERCEPTION_KEYS = (
    "obs_delay_steps_range", "detection_delay_range", "obs_pos_noise_std", "obs_vel_noise_std", "obs_pos_bias",
    "obs_dropout_prob", "obs_dropout_max_steps", "hide_in_catch_zone_prob", "max_coast_steps",
    "catch_zone_x", "catch_zone_y", "catch_zone_z",
)
BALL_TERMS = ("ball_state", "ball_visible", "ball_intercept")


def main():
    cfg = parse_env_cfg(TASK, num_envs=args_cli.num_envs)
    cfg.curriculum.ball_throws = None  # keep the command cfg (= final throw distribution)
    cfg.events.push_robot = None
    cfg.observations.policy.enable_corruption = False
    cmd = cfg.commands.ball_throw
    cmd.throw_delay_range, cmd.throw_delay_log_uniform = (0.5, 1.5), False
    if not args_cli.random:
        for key, value in DETERMINISTIC.items():
            setattr(cmd, key, value)
    if args_cli.wide_zone:  # more coasting and losing in the deterministic test
        cmd.catch_zone_x, cmd.catch_zone_y, cmd.catch_zone_z = (0.0, 1.5), (-0.8, 0.8), (-0.3, 1.2)
    env = gym.make(TASK, cfg=cfg)
    env.reset()
    ue = env.unwrapped
    command, robot, ball = ue.command_manager.get_term("ball_throw"), ue.scene["robot"], ue.scene["ball"]

    # newest frame of each ball term inside the flattened policy observation (term blocks hold the whole history)
    names = ue.observation_manager.active_terms["policy"]
    dims = [int(np.prod(d)) for d in ue.observation_manager.group_obs_term_dim["policy"]]
    history = cfg.observations.policy.history_length
    starts = np.cumsum([0] + dims)
    frames = {
        n: slice(starts[i + 1] - dims[i] // history, starts[i + 1]) for i, n in enumerate(names) if n in BALL_TERMS
    }

    zero = torch.zeros(ue.num_envs, ue.action_manager.total_action_dim, device=ue.device)
    rec = {k: [] for k in ("reset", "root_pos", "root_quat", "ball_pos", "ball_vel", "launched", "time_since_throw",
                           "in_zone", "obs_delay", *BALL_TERMS)}
    for _ in range(args_cli.steps):
        obs, _, terminated, truncated, _ = env.step(zero)
        policy = obs["policy"].cpu().numpy()
        rec["reset"].append((terminated | truncated).cpu().numpy())
        rec["root_pos"].append(robot.data.root_pos_w.cpu().numpy())
        rec["root_quat"].append(robot.data.root_quat_w.cpu().numpy())
        rec["ball_pos"].append(ball.data.root_pos_w.cpu().numpy())
        rec["ball_vel"].append(ball.data.root_lin_vel_w.cpu().numpy())
        rec["launched"].append(command.launched.cpu().numpy())
        rec["time_since_throw"].append(command.time_since_throw.cpu().numpy())
        rec["in_zone"].append(command._in_zone.cpu().numpy())
        rec["obs_delay"].append(command._obs_delay.cpu().numpy())
        for name in BALL_TERMS:
            rec[name].append(policy[:, frames[name]])
    perception = {k: getattr(cmd, k) for k in PERCEPTION_KEYS}
    np.savez(
        args_cli.out,
        **{k: np.array(v) for k, v in rec.items()},
        step_dt=ue.step_dt,
        deterministic=not args_cli.random,
        perception=json.dumps(perception),
    )
    visible = np.array(rec["ball_visible"])[..., 0]
    print(f"[record] {args_cli.steps} steps x {ue.num_envs} envs, ball visible in {visible.mean():.1%} of them, "
          f"{int(np.array(rec['reset']).sum())} resets -> {args_cli.out}", flush=True)


if __name__ == "__main__":
    main()
    os._exit(0)  # Kit's shutdown can hang
