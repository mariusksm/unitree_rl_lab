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
    """Terminate (successfully) once the ball has been held at a hand for the configured duration."""
    command = env.command_manager.get_term(command_name)
    return command.secured_steps >= command.cfg.secure_steps


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
