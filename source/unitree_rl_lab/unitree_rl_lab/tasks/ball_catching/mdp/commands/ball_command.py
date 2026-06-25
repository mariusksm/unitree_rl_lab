from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class BallCommand(CommandTerm):
    """Command term that spawns and throws a ball toward the robot using PhysX physics.

    On resample: spawns the ball at a random position in front of the robot with
    a velocity computed to bring it near the robot's catch zone (chest height).
    Physics (gravity, collision) is handled by PhysX after the throw.

    Resampling is triggered when:
    - The ball falls below min_height_throw (dropped)
    - The ball moves too far from the robot (missed)
    - The throw has been active longer than max_flight_time (timed out)
    """

    cfg: BallCommandCfg

    def __init__(self, cfg: BallCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.ball: RigidObject = env.scene[cfg.asset_name]
        self.robot: Articulation = env.scene["robot"]

        # [BALL THROW PARAMS] per-environment counters
        self.time_since_throw = torch.zeros(self.num_envs, device=self.device)

        # identity quaternion (w,x,y,z) — ball orientation doesn't matter
        self._default_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        # gravity vector for velocity computation
        self._gravity = torch.tensor([0.0, 0.0, -9.81], device=self.device)

    def __del__(self):
        pass

    @property
    def command(self) -> torch.Tensor:
        """Returns (num_envs, 6): ball position (3) + ball velocity (3) in world frame."""
        return torch.cat(
            [self.ball.data.root_pos_w, self.ball.data.root_lin_vel_w],
            dim=1,
        )

    def _update_metrics(self):
        self.metrics["ball_height"] = self.ball.data.root_pos_w[:, 2].clone()

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return

        n = len(env_ids)
        device = self.device

        # ── spawn position ──────────────────────────────────
        robot_pos = self.robot.data.root_pos_w[env_ids]

        offset_x = 1.0 + torch.rand(n, device=device) * 1.5        # [THROW DISTANCE] 1.0–2.5m in front
        offset_y = (torch.rand(n, device=device) - 0.5) * 1.5      # [THROW LATERAL] ±0.75m sideways
        offset_z = 0.8 + torch.rand(n, device=device) * 0.7        # [THROW HEIGHT] 0.8–1.5m above robot

        spawn_pos = robot_pos.clone()
        spawn_pos[:, 0] += offset_x
        spawn_pos[:, 1] += offset_y
        spawn_pos[:, 2] += offset_z

        # ── catch target ────────────────────────────────────
        catch_zone = robot_pos.clone()
        catch_zone[:, 0] += 0.4                                     # [CATCH DISTANCE] 0.4m in front
        catch_zone[:, 1] += torch.randn(n, device=device) * 0.05   # small lateral noise
        catch_zone[:, 2] += 1.0 + torch.randn(n, device=device) * 0.05  # [CATCH HEIGHT] ~1.0m above base (chest)

        # ── flight time ─────────────────────────────────────
        flight_time = 0.3 + torch.rand(n, device=device) * 0.3     # [FLIGHT TIME] 0.3–0.6s

        # ── throw velocity (gravity-compensated) ────────────
        direction = catch_zone - spawn_pos
        dt_sq = flight_time.unsqueeze(1) ** 2
        throw_vel = (direction - 0.5 * self._gravity.unsqueeze(0) * dt_sq) / flight_time.unsqueeze(1)
        throw_vel += torch.randn(n, 3, device=device) * 0.3         # [VELOCITY NOISE] ±0.3 m/s per axis

        # ── write to simulation ─────────────────────────────
        zero_ang_vel = torch.zeros(n, 3, device=device)
        root_state = torch.cat(
            [
                spawn_pos,
                self._default_quat.unsqueeze(0).expand(n, -1),
                throw_vel,
                zero_ang_vel,
            ],
            dim=1,
        )
        self.ball.write_root_state_to_sim(root_state, env_ids=env_ids)

        self.time_since_throw[env_ids] = 0.0

    def _update_command(self):
        self.time_since_throw += self._env.step_dt

        ball_pos = self.ball.data.root_pos_w
        robot_pos = self.robot.data.root_pos_w

        dropped = ball_pos[:, 2] < self.cfg.min_height_throw
        too_far = torch.norm(ball_pos[:, :2] - robot_pos[:, :2], dim=1) > self.cfg.max_distance
        timed_out = self.time_since_throw > self.cfg.max_flight_time

        resample_ids = (dropped | too_far | timed_out).nonzero(as_tuple=False).flatten()
        if len(resample_ids) > 0:
            self._resample(resample_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass


@configclass
class BallCommandCfg(CommandTermCfg):
    """Configuration for the ball throw command."""

    class_type: type = BallCommand

    asset_name: str = MISSING

    # [RESAMPLE TRIGGERS] — when to re-throw
    min_height_throw: float = 0.2
    """Minimum ball Z height (m). Ball below this is considered dropped and re-thrown."""

    max_distance: float = 5.0
    """Maximum horizontal distance (m) from robot base. Ball beyond this is considered missed."""

    max_flight_time: float = 5.0
    """Maximum time (s) a ball can be in flight before re-throwing."""
