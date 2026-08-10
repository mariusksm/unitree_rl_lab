from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat, quat_apply_inverse, subtract_frame_transforms, yaw_quat

from unitree_rl_lab.tasks.ball_catching.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def robot_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(command.robot_anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, :3].view(env.num_envs, -1)


def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, 3:6].view(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)


def ball_pos_vel_b(
    env: ManagerBasedEnv,
    ball_name: str = "ball",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Ball position (relative to robot root) and linear velocity, in the robot's yaw frame.

    This is the frame reproducible on hardware from VICON data (ball and robot-base
    tracking), independent of the robot's heading in the world.

    Returns (num_envs, 6): [pos_b (3), vel_b (3)].
    """
    ball: RigidObject = env.scene[ball_name]
    robot: Articulation = env.scene[asset_cfg.name]
    heading = yaw_quat(robot.data.root_quat_w)
    pos_b = quat_apply_inverse(heading, ball.data.root_pos_w - robot.data.root_pos_w)
    vel_b = quat_apply_inverse(heading, ball.data.root_lin_vel_w)
    return torch.cat([pos_b, vel_b], dim=1)


def ball_to_hands_b(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    ball_name: str = "ball",
) -> torch.Tensor:
    """Vector from the midpoint of the hand bodies to the ball, in the robot's yaw frame.

    Pass the hand bodies via ``asset_cfg`` (e.g. ``SceneEntityCfg("robot",
    body_names=["left_wrist_yaw_link", "right_wrist_yaw_link"])``) so the body
    indices are resolved once by the observation manager.

    Returns (num_envs, 3).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ball: RigidObject = env.scene[ball_name]
    hands_mid = asset.data.body_pos_w[:, asset_cfg.body_ids].mean(dim=1)
    heading = yaw_quat(asset.data.root_quat_w)
    return quat_apply_inverse(heading, ball.data.root_pos_w - hands_mid)


def hand_pos_b(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Positions of the bodies in ``asset_cfg`` relative to the robot root, in the robot's yaw frame.

    Returns (num_envs, N_bodies * 3): concatenated relative positions.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    rel = asset.data.body_pos_w[:, asset_cfg.body_ids] - asset.data.root_pos_w.unsqueeze(1)
    heading = yaw_quat(asset.data.root_quat_w).unsqueeze(1).expand(-1, rel.shape[1], -1)
    return quat_apply_inverse(heading, rel).reshape(env.num_envs, -1)


def dummy_zeros(env: ManagerBasedEnv, dim: int) -> torch.Tensor:
    """Returns a zero tensor. Used to pad observation space for transfer learning."""
    return torch.zeros(env.num_envs, dim, device=env.device)
