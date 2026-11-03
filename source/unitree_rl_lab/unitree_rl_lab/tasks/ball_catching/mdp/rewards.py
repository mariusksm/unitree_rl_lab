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


def posture_quality(
    env: ManagerBasedRLEnv,
    tilt_std: float = 0.35,
    height_tolerance: float = 0.03,
    height_std: float = 0.05,
    torso_body_name: str = "torso_link",
    ref_command_name: str | None = None,
    ref_body_name: str = "pelvis",
    ref_height: float = 0.78,
) -> torch.Tensor:
    """Posture factor P in [0, 1]: torso upright and pelvis at standing height (since 2026-11-01).

    P = exp(-|g_xy|² / tilt_std²) · exp(-max(0, h_ref - height_tolerance - h)² / height_std²), with g the gravity in
    the torso frame (|g_xy| = sin of its tilt in any direction; the waist lets the torso lean although the pelvis is
    level, so the torso is measured) and h the pelvis height. The height part is one-sided: within
    ``height_tolerance`` below the reference and anything higher costs nothing (bent knees of up to about 35-40°), a
    deeper squat falls off fast (fifth Unified run: pelvis 0.71 m while holding -> 0.43, 0.67 m while waiting -> 0.07).
    h_ref is the height of ``ref_body_name`` of the motion command ``ref_command_name`` in the current frame (so a clip
    in which the human squats is not punished), else ``ref_height``. As Phillip's ``standing`` in g1-catch, which also
    measures the torso and the pelvis height. Used as the reward ``stand`` and as a factor of the hold reward and the
    catch bonus, so that a hold in a bad posture is worth little instead of costing a little."""
    asset: Articulation = env.scene["robot"]
    torso_quat = asset.data.body_quat_w[:, asset.body_names.index(torso_body_name)]
    gravity_torso = quat_apply_inverse(torso_quat, asset.data.GRAVITY_VEC_W)
    upright = torch.exp(-torch.sum(torch.square(gravity_torso[:, :2]), dim=-1) / tilt_std**2)
    ground = env.scene.env_origins[:, 2]
    height = asset.data.root_pos_w[:, 2] - ground
    if ref_command_name is not None:
        command: MotionCommand = env.command_manager.get_term(ref_command_name)
        reference = command.body_pos_w[:, command.cfg.body_names.index(ref_body_name), 2] - ground
    else:
        reference = torch.full_like(height, ref_height)
    sag = torch.clamp(reference - height_tolerance - height, min=0.0)
    return upright * torch.exp(-torch.square(sag) / height_std**2)


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
    until_confirmed: bool = False,
) -> torch.Tensor:
    """Reward the hands for meeting the thrown ball (interim report, eq. 9.1).

    d is the distance between the ball and the midpoint of the bodies in ``asset_cfg`` (both hands, so a
    two-arm catch is preferred over reaching with one hand), v the ball speed relative to the mean hand
    velocity. Returns exp(-d²/std_dist² - v²/std_vel²); without ``std_vel`` only the distance counts.
    Zero until the ball is launched, so the parked ball at the release point does not attract the hands.
    With ``until_confirmed`` also zero after the catch is confirmed: then ``hold_quality`` alone says how well it is
    held, and hands near the ball pay nothing by themselves.
    """
    command = env.command_manager.get_term(command_name)
    ball: RigidObject = command.ball
    robot: Articulation = env.scene[asset_cfg.name]
    hands_pos = robot.data.body_pos_w[:, asset_cfg.body_ids].mean(dim=1)
    error = torch.sum(torch.square(ball.data.root_pos_w - hands_pos), dim=-1) / std_dist**2
    if std_vel is not None:
        hands_vel = robot.data.body_lin_vel_w[:, asset_cfg.body_ids].mean(dim=1)
        error += torch.sum(torch.square(ball.data.root_lin_vel_w - hands_vel), dim=-1) / std_vel**2
    active = command.launched & ~command.confirmed if until_confirmed else command.launched
    return torch.exp(-error) * active.float()


def hold_quality(env: ManagerBasedRLEnv, command_name: str = "ball_throw", posture: dict | None = None) -> torch.Tensor:
    """How well the ball is held (``BallCommand.hold_quality``: both palms carry it, at the hold point, only allowed
    contacts), times the posture factor :func:`posture_quality` with the parameters ``posture`` (if given).

    Replaces the flat terms ``ball_arm_contact`` and ``ball_secured``, which paid the same for every hold inside loose
    limits, so the policy picked the most robust one (wedged, clamped to the chest; Unified runs 1-5)."""
    command = env.command_manager.get_term(command_name)
    quality = command.hold_quality
    if posture is not None:
        quality = quality * posture_quality(env, **posture)
    return quality


def ball_body_contact(env: ManagerBasedRLEnv, command_name: str = "ball_throw") -> torch.Tensor:
    """1 while the launched ball touches a forbidden body part or presses on the torso above the limit
    (``BallCommand.forbidden_contact``, ``contact_map``). Kept small: the lost hold reward is the main cost, so keeping
    the ball always pays more than dropping it."""
    command = env.command_manager.get_term(command_name)
    return command.forbidden_contact.float()


def self_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 10.0) -> torch.Tensor:
    """Share of the physics steps of this control step in which a filter body of the sensor presses on the sensor body
    with more than ``threshold`` N (arms against the torso). The fifth Unified run pressed its arms against the torso
    with more than 140 N to stiffen its grip; ``undesired_contacts`` excludes arms and torso and never saw it."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    data = sensor.data
    forces = data.force_matrix_w_history
    if forces is None:
        forces = data.force_matrix_w.unsqueeze(1)
    pressing = (torch.norm(forces, dim=-1) > threshold).flatten(start_dim=2).any(dim=-1)  # (N, T)
    return pressing.float().mean(dim=-1)


def torque_limit_excess(
    env: ManagerBasedRLEnv, threshold: float = 0.6, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Sum over the joints of ``asset_cfg`` of the squared excess of the commanded torque over ``threshold`` times the
    joint's effort limit: max(0, |tau| / tau_max - threshold)^2. Uses the commanded (not the clipped) torque, so the
    penalty keeps growing beyond the limit. Normalized per joint, unlike ``joint_torques_l2``, where the 5 Nm wrist
    motors vanish next to the 100 Nm leg joints (Advanced-A pressed its hands together with the left wrist yaw at its
    limit; since 2026-11-03)."""
    asset: Articulation = env.scene[asset_cfg.name]
    tau = asset.data.computed_torque[:, asset_cfg.joint_ids]
    limit = asset.data.joint_effort_limits[:, asset_cfg.joint_ids]
    excess = (tau.abs() / limit - threshold).clamp(min=0.0)
    return torch.sum(excess**2, dim=1)


def ball_arm_contact(env: ManagerBasedRLEnv, command_name: str = "ball_throw") -> torch.Tensor:
    """Fraction of arms (left, right) the launched ball touches in this step (interim report, eq. 9.2); with the
    command's ``palm_hold`` only palm contacts count."""
    command = env.command_manager.get_term(command_name)
    return command.arm_contact.float().mean(dim=-1) * command.launched.float()


def ball_caught_bonus(
    env: ManagerBasedRLEnv, command_name: str = "ball_throw", posture: dict | None = None
) -> torch.Tensor:
    """One-time bonus on the step the catch is confirmed (``secured_steps`` reaches ``secure_steps``).

    Fires exactly once per throw (``BallCommand.confirmed_now``), also if the secured streak breaks and restarts
    during the hold phase. With ``posture`` scaled by :func:`posture_quality` at that step. Note the reward manager
    multiplies by ``weight * dt`` — size the weight as bonus_value / dt.
    """
    command = env.command_manager.get_term(command_name)
    bonus = command.confirmed_now.float()
    if posture is not None:
        bonus = bonus * posture_quality(env, **posture)
    return bonus


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
