from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import _randomize_prop_by_op, push_by_setting_velocity
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def push_by_setting_velocity_phased(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    command_name: str = "ball_throw",
    ball_phase_scale: float = 1.0,
    ball_phase_scale_final: float | None = None,
    difficulty: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """``push_by_setting_velocity`` with the velocity range scaled by ``ball_phase_scale`` once the ball is launched
    (flight, catch and hold phase); full strength while waiting for the throw.

    Unified (since 2026-11-01) pushes at 20 % in the ball phase: a push changes the velocity of the whole robot by up
    to 0.5 m/s, the free ball does not follow, and a clean grip with two palms loses it, so the full pushes rewarded
    clamping the ball against the chest.

    With ``ball_phase_scale_final`` (Advanced-D) the share in the ball phase is drawn per push: ``ball_phase_scale``
    plus a level uniform in [0, ``difficulty``] times the way to ``ball_phase_scale_final``. The success-gated
    curriculum (``mdp.success_gated_difficulty``) raises ``difficulty``; weaker pushes stay in the mix.
    """
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)
    launched = env.command_manager.get_term(command_name).launched[env_ids]
    waiting_ids, ball_ids = env_ids[~launched], env_ids[launched]
    if len(waiting_ids) > 0:
        push_by_setting_velocity(env, waiting_ids, velocity_range, asset_cfg)
    if len(ball_ids) == 0:
        return
    if ball_phase_scale_final is None:
        if ball_phase_scale > 0.0:
            s = ball_phase_scale
            scaled = {key: (low * s, high * s) for key, (low, high) in velocity_range.items()}
            push_by_setting_velocity(env, ball_ids, scaled, asset_cfg)
        return
    level = torch.rand(len(ball_ids), device=env.device) * difficulty
    scale = ball_phase_scale + level * (ball_phase_scale_final - ball_phase_scale)
    _push_scaled(env, ball_ids, velocity_range, scale, asset_cfg)


def _push_scaled(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    scale: torch.Tensor,
    asset_cfg: SceneEntityCfg,
):
    """``push_by_setting_velocity`` with a velocity change scaled per env (``scale``, shape (len(env_ids),))."""
    asset: Articulation = env.scene[asset_cfg.name]
    vel_w = asset.data.root_vel_w[env_ids]
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    push = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=asset.device)
    vel_w += push * scale.unsqueeze(1)
    asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)


def reset_joints_to_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    joint_pos: dict[str, float],
    position_noise: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the joints to a fixed pose plus uniform noise of ±``position_noise`` rad per joint, at zero velocity and
    within the soft joint limits. ``joint_pos`` maps joint-name regexes to angles (rad); every other joint gets 0.

    Advanced-C starts each episode in the pose in which the deploy FSM hands the robot to the policy (FixStand), not in
    the default pose. The default pose stays the offset of the actions and of ``joint_pos_rel``, so the actor's
    observations and actions keep their meaning.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    pose = torch.zeros(asset.num_joints, device=asset.device)
    for pattern, value in joint_pos.items():
        ids, _ = asset.find_joints(pattern)  # raises if the pattern matches no joint
        pose[ids] = value
    pos = pose.unsqueeze(0).repeat(len(env_ids), 1)
    if position_noise > 0.0:
        pos += math_utils.sample_uniform(-position_noise, position_noise, pos.shape, asset.device)
    limits = asset.data.soft_joint_pos_limits[env_ids]
    pos = pos.clamp(limits[..., 0], limits[..., 1])
    asset.write_joint_state_to_sim(pos, torch.zeros_like(pos), env_ids=env_ids)


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # save nominal value for export
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos
        # update the offset in action since it is not updated automatically
        env.action_manager.get_term("JointPositionAction")._offset[env_ids, joint_ids] = pos


def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Randomize the center of mass (CoM) of rigid bodies by adding a random value sampled from the given ranges.

    .. note::
        This function uses CPU tensors to assign the CoM. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # sample random CoM values
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu").unsqueeze(1)

    # get the current com of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms().clone()

    # Randomize the com in range
    coms[:, body_ids, :3] += rand_samples

    # Set the new coms
    asset.root_physx_view.set_coms(coms, env_ids)
