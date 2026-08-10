from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude

from unitree_rl_lab.tasks.ball_catching.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]
    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward


# ── Ball catching rewards ──────────────────────────────────────────────


def hand_to_ball_distance_exp(
    env: ManagerBasedRLEnv,
    std: float,
    ball_name: str = "ball",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Gaussian reward based on the closest hand's distance to the ball.

    Pass the hand bodies via ``asset_cfg`` (body indices are resolved once by the
    reward manager). Returns exp(-min_distance² / std²) — saturates at 1.0 when a
    hand is at the ball.
    """
    ball: RigidObject = env.scene[ball_name]
    robot: Articulation = env.scene[asset_cfg.name]
    ball_pos = ball.data.root_pos_w.unsqueeze(1)                # (N, 1, 3)
    hand_pos = robot.data.body_pos_w[:, asset_cfg.body_ids]     # (N, H, 3)
    dist = torch.norm(hand_pos - ball_pos, dim=-1)              # (N, H)
    min_dist = dist.min(dim=-1).values                          # (N,)
    return torch.exp(-(min_dist**2) / std**2)


def ball_caught_bonus(env: ManagerBasedRLEnv, command_name: str = "ball_throw") -> torch.Tensor:
    """One-time bonus on the step the catch is confirmed.

    Fires exactly once per catch: the secured-steps counter (tracked by the
    BallCommand term) equals ``secure_steps`` on the same step the ``ball_caught``
    termination triggers, after which the counter is reset. Note the reward manager
    multiplies by ``weight * dt`` — size the weight as bonus_value / dt.
    """
    command = env.command_manager.get_term(command_name)
    return (command.secured_steps == command.cfg.secure_steps).float()


def ball_secured(env: ManagerBasedRLEnv, command_name: str = "ball_throw") -> torch.Tensor:
    """Per-step reward while the ball is held at a hand (bounded by the catch termination)."""
    command = env.command_manager.get_term(command_name)
    return (command.secured_steps > 0).float()


def ball_height_penalty(
    env: ManagerBasedRLEnv,
    ball_name: str,
    min_height: float,
) -> torch.Tensor:
    """Penalty when the ball falls below the catch zone."""
    ball: RigidObject = env.scene[ball_name]
    return (ball.data.root_pos_w[:, 2] < min_height).float()
