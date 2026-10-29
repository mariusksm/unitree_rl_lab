from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, quat_error_magnitude

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


def joint_target_excess(env: ManagerBasedRLEnv, action_name: str = "JointPositionAction") -> torch.Tensor:
    """How far (rad, summed over the joints) the policy's joint targets lie beyond the clamp of a
    :class:`ClampedJointPositionAction` (the joint limits). Beyond the clamp the reward has no other gradient."""
    return env.action_manager.get_term(action_name).target_excess


def feet_flat(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    force_threshold: float = 50.0,
) -> torch.Tensor:
    """Tilt of the feet that carry weight: sum over the feet in firm ground contact of |gravity_xy|² in the foot frame.

    That is sin² of the tilt angle per foot (15° -> 0.067). The ``*_ankle_roll_link`` frame is level when the sole is
    flat on the ground. Contact counts above ``force_threshold`` N (as in the evaluation suite), so a foot that only
    touches down or lifts off is not penalized for rolling over its edge.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).amax(dim=1)
    asset: Articulation = env.scene[asset_cfg.name]
    feet_quat = asset.data.body_quat_w[:, asset_cfg.body_ids]
    gravity = asset.data.GRAVITY_VEC_W.unsqueeze(1).expand(-1, feet_quat.shape[1], -1)
    gravity_feet = quat_apply_inverse(feet_quat, gravity)
    return torch.sum(torch.sum(torch.square(gravity_feet[..., :2]), dim=-1) * (forces > force_threshold), dim=-1)


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward


# ── Ball catching rewards ──────────────────────────────────────────────


def ball_interception_exp(
    env: ManagerBasedRLEnv,
    std_dist: float,
    std_vel: float | None = None,
    command_name: str = "ball_throw",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward the hands for meeting the thrown ball (interim report, eq. 9.1).

    d is the distance between the ball and the midpoint of the bodies in ``asset_cfg`` (both hands, so a
    two-arm catch is preferred over reaching with one hand), v the ball speed relative to the mean hand
    velocity. Returns exp(-d²/std_dist² - v²/std_vel²); without ``std_vel`` only the distance counts.
    Zero until the ball is launched, so the parked ball at the release point does not attract the hands.
    """
    command = env.command_manager.get_term(command_name)
    ball: RigidObject = command.ball
    robot: Articulation = env.scene[asset_cfg.name]
    hands_pos = robot.data.body_pos_w[:, asset_cfg.body_ids].mean(dim=1)
    error = torch.sum(torch.square(ball.data.root_pos_w - hands_pos), dim=-1) / std_dist**2
    if std_vel is not None:
        hands_vel = robot.data.body_lin_vel_w[:, asset_cfg.body_ids].mean(dim=1)
        error += torch.sum(torch.square(ball.data.root_lin_vel_w - hands_vel), dim=-1) / std_vel**2
    return torch.exp(-error) * command.launched.float()


def ball_arm_contact(env: ManagerBasedRLEnv, command_name: str = "ball_throw") -> torch.Tensor:
    """Fraction of arms (left, right) the launched ball touches in this step (interim report, eq. 9.2); with the
    command's ``palm_hold`` only palm contacts count."""
    command = env.command_manager.get_term(command_name)
    return command.arm_contact.float().mean(dim=-1) * command.launched.float()


def ball_caught_bonus(env: ManagerBasedRLEnv, command_name: str = "ball_throw") -> torch.Tensor:
    """One-time bonus on the step the catch is confirmed (``secured_steps`` reaches ``secure_steps``).

    Fires exactly once per throw (``BallCommand.confirmed_now``), also if the secured streak breaks and restarts
    during the hold phase. Note the reward manager multiplies by ``weight * dt`` — size the weight as
    bonus_value / dt.
    """
    command = env.command_manager.get_term(command_name)
    return command.confirmed_now.float()


def ball_secured(env: ManagerBasedRLEnv, command_name: str = "ball_throw") -> torch.Tensor:
    """Per-step reward while the ball is held at a hand (bounded by the catch termination)."""
    command = env.command_manager.get_term(command_name)
    return (command.secured_steps > 0).float()


def _should_stand_still(command) -> torch.Tensor:
    """Before the launch (waiting for the throw) and during the hold phase after the confirmed catch."""
    return ~command.launched | command.confirmed


def stand_still_root(
    env: ManagerBasedRLEnv,
    command_name: str = "ball_throw",
    yaw_rate_weight: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Root motion while the robot should stand still (waiting, holding): |v_xy|² + yaw_rate_weight·ω_z².

    The Phase 2 baseline drifted 0.16 m/s and turned 15°/s while waiting, and walked off with the ball while holding.
    """
    command = env.command_manager.get_term(command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    motion = torch.sum(torch.square(asset.data.root_lin_vel_w[:, :2]), dim=-1)
    motion += yaw_rate_weight * torch.square(asset.data.root_ang_vel_w[:, 2])
    return motion * _should_stand_still(command).float()


def stand_still_feet(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "ball_throw",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    force_threshold: float = 50.0,
) -> torch.Tensor:
    """Horizontal speed (m/s) of the feet in firm ground contact (> ``force_threshold`` N), summed, while the robot
    should stand still (waiting, holding): sliding and pivoting on the foot edges."""
    command = env.command_manager.get_term(command_name)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).amax(dim=1)
    asset: Articulation = env.scene[asset_cfg.name]
    speed = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2].norm(dim=-1)
    return torch.sum(speed * (forces > force_threshold), dim=-1) * _should_stand_still(command).float()


def ball_height_penalty(
    env: ManagerBasedRLEnv,
    ball_name: str,
    min_height: float,
) -> torch.Tensor:
    """Penalty when the ball falls below the catch zone."""
    ball: RigidObject = env.scene[ball_name]
    return (ball.data.root_pos_w[:, 2] < min_height).float()
