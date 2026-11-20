from __future__ import annotations

import torch
from typing import TYPE_CHECKING

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from unitree_rl_lab.tasks.ball_catching.mdp.commands import MotionCommand
from unitree_rl_lab.tasks.ball_catching.mdp.rewards import _get_body_indexes


def bad_anchor_pos(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold


def bad_anchor_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold


def bad_anchor_ori(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_projected_gravity_b = quat_apply_inverse(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)

    robot_projected_gravity_b = quat_apply_inverse(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)

    return (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold


def bad_motion_body_pos(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.norm(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes], dim=-1)
    return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.abs(command.body_pos_relative_w[:, body_indexes, -1] - command.robot_body_pos_w[:, body_indexes, -1])
    return torch.any(error > threshold, dim=-1)


# ── Ball catching terminations ─────────────────────────────────────────


def ball_caught(env: ManagerBasedRLEnv, command_name: str = "ball_throw") -> torch.Tensor:
    """Success: the confirmed catch has lasted the hold phase (``BallCommand.catch_complete``; without a hold phase
    this is the confirmation itself). Use it as a truncation (``time_out=True``), see the task configs."""
    command = env.command_manager.get_term(command_name)
    return command.catch_complete


def ball_below_height(env: ManagerBasedRLEnv, ball_name: str, min_height: float) -> torch.Tensor:
    """Terminate when the ball falls below a minimum height (hit the ground or unrecoverable)."""
    ball: RigidObject = env.scene[ball_name]
    return ball.data.root_pos_w[:, 2] < min_height


def ball_far_from_robot(env: ManagerBasedRLEnv, ball_name: str, max_distance: float) -> torch.Tensor:
    """Terminate when the ball moves too far from the robot horizontally (missed)."""
    ball: RigidObject = env.scene[ball_name]
    robot: Articulation = env.scene["robot"]
    ball_xy = ball.data.root_pos_w[:, :2]
    robot_xy = robot.data.root_pos_w[:, :2]
    return torch.norm(ball_xy - robot_xy, dim=1) > max_distance


def ball_lost_since(
    env: ManagerBasedRLEnv, wait_s: float, command_name: str = "ball_throw", motion_command_name: str | None = None
) -> torch.Tensor:
    """The throw was lost at least ``wait_s`` seconds ago (``BallCommandCfg.track_loss``: below ``loss_height`` or
    farther than ``loss_distance``). Ends the episode a while after a loss instead of at once, so that the robot first
    has to stand next to the lost ball, visible or hidden (seventh Unified run on).

    With ``motion_command_name`` (Unified) a lost throw also ends the episode on the last step of its reference clip:
    the next cycle would start there and clear the loss (``SyncedBallCommand._start_cycle``), so a ball dropped late in
    the hold phase would go unpunished and uncounted by the adaptive sampling."""
    command = env.command_manager.get_term(command_name)
    done = command.time_since_loss >= wait_s
    if motion_command_name is not None:
        motion: MotionCommand = env.command_manager.get_term(motion_command_name)
        # the motion command resamples in the command update after this check, once time_steps reaches the clip length
        done = done | (motion.time_steps + 1 >= motion.motion.time_step_total[motion.clip_ids])
    return command.throw_failed & done
