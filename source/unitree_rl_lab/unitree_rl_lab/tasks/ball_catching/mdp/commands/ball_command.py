from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, sample_uniform, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class BallCommand(CommandTerm):
    """Single-throw ball command.

    On reset (per env): parks the ball at a spawn point sampled in the robot's
    heading (yaw) frame and starts a launch timer. While the timer runs, the ball is
    held at the spawn point so the policy can settle into a stable stance and observe
    the incoming throw origin. When the timer expires, the ball is launched toward a
    catch zone in front of the robot's chest with a gravity-compensated ballistic
    velocity. Physics (gravity, collision) is handled by PhysX after the launch.

    The command does NOT re-throw within an episode. Episode outcomes are owned by
    the termination terms (ball dropped / missed / caught / time out), which read the
    catch state tracked here:

    - ``secured_steps``: consecutive control steps for which the ball has been close
      to a hand, moving slowly relative to that hand, and above the ground. Reaching
      ``cfg.secure_steps`` counts as a confirmed catch.
    """

    cfg: BallCommandCfg

    def __init__(self, cfg: BallCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.ball: RigidObject = env.scene[cfg.asset_name]
        self.robot: Articulation = env.scene["robot"]

        # resolve hand body indices once
        self.hand_body_ids, _ = self.robot.find_bodies(cfg.hand_body_names, preserve_order=True)

        # per-environment state
        self.time_until_throw = torch.zeros(self.num_envs, device=self.device)
        self.time_since_throw = torch.zeros(self.num_envs, device=self.device)
        self.secured_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._spawn_pos = torch.zeros(self.num_envs, 3, device=self.device)

        # identity quaternion (w,x,y,z) — ball orientation doesn't matter
        self._default_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        # gravity vector for the ballistic velocity computation
        self._gravity = torch.tensor([0.0, 0.0, -9.81], device=self.device)

        self.metrics["ball_height"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["ball_secured_time"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["hand_ball_distance"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """Ball position (3) + linear velocity (3) in world frame, shape (num_envs, 6).

        Note: not used as a policy observation (world coordinates include per-env
        origins). Policy observations use the yaw-frame terms in ``observations.py``.
        """
        return torch.cat([self.ball.data.root_pos_w, self.ball.data.root_lin_vel_w], dim=1)

    def _update_metrics(self):
        self.metrics["ball_height"] = self.ball.data.root_pos_w[:, 2].clone()
        self.metrics["ball_secured_time"] = self.secured_steps.float() * self._env.step_dt

    def _resample_command(self, env_ids: Sequence[int]):
        """Park the ball at a new spawn point and arm the launch timer."""
        if len(env_ids) == 0:
            return

        n = len(env_ids)
        device = self.device

        robot_pos = self.robot.data.root_pos_w[env_ids]
        heading = yaw_quat(self.robot.data.root_quat_w[env_ids])

        # spawn offset in the robot's heading frame
        offset = torch.zeros(n, 3, device=device)
        offset[:, 0] = sample_uniform(*self.cfg.throw_distance_range, (n,), device)
        offset[:, 1] = sample_uniform(*self.cfg.throw_lateral_range, (n,), device)
        offset[:, 2] = sample_uniform(*self.cfg.throw_height_range, (n,), device)
        self._spawn_pos[env_ids] = robot_pos + quat_apply(heading, offset)

        self._write_parked_state(env_ids)

        self.time_until_throw[env_ids] = sample_uniform(*self.cfg.throw_delay_range, (n,), device)
        self.time_since_throw[env_ids] = 0.0
        self.secured_steps[env_ids] = 0

    def _write_parked_state(self, env_ids: Sequence[int]):
        """Hold the ball at its spawn point with zero velocity."""
        n = len(env_ids)
        root_state = torch.cat(
            [
                self._spawn_pos[env_ids],
                self._default_quat.unsqueeze(0).expand(n, -1),
                torch.zeros(n, 6, device=self.device),
            ],
            dim=1,
        )
        self.ball.write_root_state_to_sim(root_state, env_ids=env_ids)

    def _launch(self, env_ids: torch.Tensor):
        """Throw the ball from its spawn point toward the catch zone."""
        n = len(env_ids)
        device = self.device

        robot_pos = self.robot.data.root_pos_w[env_ids]
        heading = yaw_quat(self.robot.data.root_quat_w[env_ids])

        # catch zone in the robot's heading frame (chest height, slightly in front)
        catch_local = torch.zeros(n, 3, device=device)
        catch_local[:, 0] = self.cfg.catch_forward
        catch_local[:, 1] = torch.randn(n, device=device) * self.cfg.catch_lateral_std
        catch_local[:, 2] = sample_uniform(*self.cfg.catch_height_range, (n,), device)
        catch_zone = robot_pos + quat_apply(heading, catch_local)

        flight_time = sample_uniform(*self.cfg.flight_time_range, (n,), device)

        # gravity-compensated ballistic velocity
        spawn_pos = self._spawn_pos[env_ids]
        direction = catch_zone - spawn_pos
        dt_sq = flight_time.unsqueeze(1) ** 2
        throw_vel = (direction - 0.5 * self._gravity.unsqueeze(0) * dt_sq) / flight_time.unsqueeze(1)
        throw_vel += torch.randn(n, 3, device=device) * self.cfg.velocity_noise

        root_state = torch.cat(
            [
                spawn_pos,
                self._default_quat.unsqueeze(0).expand(n, -1),
                throw_vel,
                torch.zeros(n, 3, device=device),
            ],
            dim=1,
        )
        self.ball.write_root_state_to_sim(root_state, env_ids=env_ids)

        self.time_since_throw[env_ids] = 0.0

    def _update_command(self):
        step_dt = self._env.step_dt

        # launch timer
        pre_throw = self.time_until_throw > 0.0
        self.time_until_throw -= step_dt
        launch_ids = (pre_throw & (self.time_until_throw <= 0.0)).nonzero(as_tuple=False).flatten()
        hold_ids = (self.time_until_throw > 0.0).nonzero(as_tuple=False).flatten()

        if len(hold_ids) > 0:
            self._write_parked_state(hold_ids)
        if len(launch_ids) > 0:
            self._launch(launch_ids)

        in_flight = self.time_until_throw <= 0.0
        self.time_since_throw += in_flight.float() * step_dt

        # catch state: ball close to a hand, slow relative to that hand, above ground
        ball_pos = self.ball.data.root_pos_w
        ball_vel = self.ball.data.root_lin_vel_w
        hand_pos = self.robot.data.body_pos_w[:, self.hand_body_ids]
        hand_vel = self.robot.data.body_lin_vel_w[:, self.hand_body_ids]

        dist = torch.norm(hand_pos - ball_pos.unsqueeze(1), dim=-1)
        min_dist, min_idx = dist.min(dim=-1)
        closest_hand_vel = hand_vel.gather(1, min_idx.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)
        rel_speed = torch.norm(ball_vel - closest_hand_vel, dim=-1)

        secured = (
            in_flight
            & (min_dist < self.cfg.catch_radius)
            & (rel_speed < self.cfg.secure_rel_vel)
            & (ball_pos[:, 2] > self.cfg.secure_min_height)
        )
        self.secured_steps = torch.where(secured, self.secured_steps + 1, torch.zeros_like(self.secured_steps))

        self.metrics["hand_ball_distance"] = min_dist

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass


@configclass
class BallCommandCfg(CommandTermCfg):
    """Configuration for the single-throw ball command."""

    class_type: type = BallCommand

    asset_name: str = MISSING

    hand_body_names: list[str] = ["left_wrist_yaw_link", "right_wrist_yaw_link"]
    """Bodies used for catch detection (later: palm links of the dexterous hands)."""

    # ── throw generation (all offsets in the robot's heading frame) ──
    throw_delay_range: tuple[float, float] = (0.5, 1.5)
    """Time (s) after reset before the ball is launched. The ball is parked at the
    spawn point in the meantime so the robot can settle and see the throw origin."""

    throw_distance_range: tuple[float, float] = (1.0, 2.5)
    """Spawn distance (m) in front of the robot."""

    throw_lateral_range: tuple[float, float] = (-0.75, 0.75)
    """Spawn lateral offset (m)."""

    throw_height_range: tuple[float, float] = (0.8, 1.5)
    """Spawn height (m) above the robot root."""

    catch_forward: float = 0.4
    """Aim point distance (m) in front of the robot root."""

    catch_lateral_std: float = 0.05
    """Std (m) of the lateral aim scatter."""

    catch_height_range: tuple[float, float] = (0.25, 0.45)
    """Aim point height (m) above the robot root (root ≈ 0.76 m → aim ≈ 1.0–1.2 m,
    chest height of the G1). Curriculum variable."""

    flight_time_range: tuple[float, float] = (0.3, 0.6)
    """Ballistic flight time (s) to the aim point."""

    velocity_noise: float = 0.3
    """Std (m/s) of Gaussian noise added per axis to the throw velocity."""

    # ── catch detection ──
    catch_radius: float = 0.2
    """Max hand-to-ball distance (m) for the ball to count as held."""

    secure_rel_vel: float = 0.5
    """Max ball speed (m/s) relative to the closest hand for the ball to count as held."""

    secure_min_height: float = 0.5
    """Min ball height (m) for the ball to count as held (excludes balls on the ground)."""

    secure_steps: int = 25
    """Consecutive held steps for a confirmed catch (25 steps = 0.5 s at 50 Hz)."""
