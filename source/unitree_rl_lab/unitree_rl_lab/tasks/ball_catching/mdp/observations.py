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


def _ballistic_intercept(pos_b: torch.Tensor, vel_b: torch.Tensor, catch_height: float, max_time: float):
    """Descending crossing of a drag-free ball trajectory with the plane ``catch_height`` above the root.

    ``pos_b`` / ``vel_b``: ball position relative to the root and velocity, both in the yaw frame (gravity
    along -z). Returns (num_envs, 3): [intercept_x_b, intercept_y_b, time_to_go]; if the trajectory never
    reaches the plane, time-to-go is 0 and the current position is returned.
    """
    g = 9.81
    # descending root of  z0 + vz*t - g*t²/2 = z_plane
    vz = vel_b[:, 2]
    disc = vz * vz + 2.0 * g * (pos_b[:, 2] - catch_height)
    t_go = (vz + torch.sqrt(torch.clamp(disc, min=0.0))) / g
    t_go = torch.where(disc > 0.0, t_go.clamp(0.0, max_time), torch.zeros_like(t_go))
    intercept_xy = pos_b[:, :2] + vel_b[:, :2] * t_go.unsqueeze(1)
    return torch.cat([intercept_xy, t_go.unsqueeze(1)], dim=1)


def ball_intercept_b(
    env: ManagerBasedEnv,
    ball_name: str = "ball",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    catch_height: float = 0.15,
    max_time: float = 3.0,
) -> torch.Tensor:
    """Predicted ballistic interception point and time-to-go from the TRUE ball state (critic).

    Solves for the (descending) crossing of the ball's ballistic trajectory with the
    horizontal catch plane ``catch_height`` above the robot root (default: belly height
    of the underhand two-arm catch) and returns the predicted crossing point relative to
    the root, rotated into the robot's yaw frame. If the trajectory never reaches the
    plane, time-to-go is 0 and the current ball position is returned instead.

    This is the timing signal catching lives on, and the same ballistic prediction
    the deployment pipeline computes from the filtered VICON stream. It ignores air
    drag on purpose (simple to reproduce on hardware): for the light 15 cm ball the
    early prediction lands up to ~0.3 m too far, the error shrinks as time-to-go drops.
    While the ball is parked before the throw, the output is the "if it fell now"
    point — a smooth, informative signal.

    Returns (num_envs, 3): [intercept_x_b, intercept_y_b, time_to_go].
    """
    state = ball_pos_vel_b(env, ball_name, asset_cfg)
    return _ballistic_intercept(state[:, :3], state[:, 3:], catch_height, max_time)


def ball_perceived_pos_vel_b(env: ManagerBasedEnv, command_name: str = "ball_throw") -> torch.Tensor:
    """Ball position (relative to the root) and velocity in the yaw frame as the robot perceives it (actor).

    Comes from the VICON model of the ball command (detection delay, latency, noise, bursts of dropouts,
    occlusion in the catch box, ballistic coasting); zeros while the ball is not visible. Corruption lives
    in the command so that all ball terms of the actor see the same estimate; give these terms no extra
    observation noise.

    Returns (num_envs, 6): [pos_b (3), vel_b (3)].
    """
    command = env.command_manager.get_term(command_name)
    return torch.cat([command.perceived_pos_b, command.perceived_vel_b], dim=1)


def ball_visible(env: ManagerBasedEnv, command_name: str = "ball_throw") -> torch.Tensor:
    """1 while the robot has a ball estimate (fresh measurement or coasting), else 0. Returns (num_envs, 1)."""
    command = env.command_manager.get_term(command_name)
    return command.ball_visible.float().unsqueeze(1)


def ball_perceived_intercept_b(
    env: ManagerBasedEnv, command_name: str = "ball_throw", catch_height: float = 0.15, max_time: float = 3.0
) -> torch.Tensor:
    """:func:`ball_intercept_b` computed from the perceived ball state (actor); zeros while not visible.

    Returns (num_envs, 3): [intercept_x_b, intercept_y_b, time_to_go].
    """
    command = env.command_manager.get_term(command_name)
    intercept = _ballistic_intercept(command.perceived_pos_b, command.perceived_vel_b, catch_height, max_time)
    return intercept * command.ball_visible.float().unsqueeze(1)


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
