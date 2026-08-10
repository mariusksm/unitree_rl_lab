from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

from .ball_command import BallCommand, BallCommandCfg
from .motion_command import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class SyncedBallCommand(BallCommand):
    """Ball command whose launch is synchronized to the mimic motion phase.

    For the unified (single-task) ball-catching environment: instead of a
    wall-clock delay after reset, the ball is launched so that it *arrives* at the
    catch zone when the reference motion reaches its catch frame
    (``cfg.catch_motion_time`` seconds into the clip):

        launch_time = catch_motion_time + jitter − flight_time

    Behavior per motion cycle:
    - While the motion time is before ``launch_time``, the ball is parked at its
      spawn point (visible "thrower" position).
    - At ``launch_time`` the ball is thrown with the pre-sampled flight time, so
      ball arrival and the mocap catch pose coincide.
    - If an episode starts *past* the launch window (adaptive sampling can start
      anywhere in the clip), no throw happens this cycle — the episode trains pure
      motion tracking (e.g. the retract phase) until the motion wraps.
    - When the motion wraps or is resampled (detected as a backward jump of the
      motion time; the robot is teleported by the MotionCommand at that moment),
      the ball is re-parked at a fresh spawn point and re-armed.

    Catch-state tracking (``secured_steps``) is inherited from :class:`BallCommand`.

    Note: the motion command must be declared *before* this command in the
    environment's ``CommandsCfg`` so that on env reset the motion time is already
    resampled when this term arms the throw.
    """

    cfg: SyncedBallCommandCfg

    def __init__(self, cfg: SyncedBallCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self._armed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._launched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._flight_time = torch.zeros(self.num_envs, device=self.device)
        self._launch_time = torch.zeros(self.num_envs, device=self.device)
        self._prev_motion_time = torch.zeros(self.num_envs, device=self.device)
        self._validated = False

    @property
    def motion_command(self) -> MotionCommand:
        return self._env.command_manager.get_term(self.cfg.motion_command_name)

    def _motion_time(self) -> torch.Tensor:
        """Current time (s) into the reference motion clip, per env."""
        motion = self.motion_command
        return motion.time_steps.float() / float(motion.motion.fps)

    def _validate_catch_time(self):
        motion = self.motion_command
        clip_duration = motion.motion.time_step_total / float(motion.motion.fps)
        latest_launch = self.cfg.catch_motion_time + self.cfg.arrival_jitter - self.cfg.flight_time_range[0]
        if latest_launch >= clip_duration:
            print(
                f"[SyncedBallCommand] WARNING: catch_motion_time ({self.cfg.catch_motion_time:.2f}s) is at/after "
                f"the end of the motion clip ({clip_duration:.2f}s) — the ball will never launch. "
                "Set catch_motion_time to the ball-contact moment of YOUR catching mocap."
            )
        self._validated = True

    def _arm(self, env_ids: Sequence[int]):
        """Sample flight time and launch moment; arm envs whose launch is still ahead."""
        n = len(env_ids)
        self._flight_time[env_ids] = sample_uniform(*self.cfg.flight_time_range, (n,), self.device)
        jitter = sample_uniform(-self.cfg.arrival_jitter, self.cfg.arrival_jitter, (n,), self.device)
        self._launch_time[env_ids] = self.cfg.catch_motion_time + jitter - self._flight_time[env_ids]
        self._armed[env_ids] = self._motion_time()[env_ids] <= self._launch_time[env_ids]
        self._launched[env_ids] = False

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        if not self._validated:
            self._validate_catch_time()

        self._sample_spawn(env_ids)
        self._write_parked_state(env_ids)
        self.time_since_throw[env_ids] = 0.0
        self.secured_steps[env_ids] = 0
        self._arm(env_ids)
        # refresh so the next update does not misread the reset as a motion wrap
        self._prev_motion_time[env_ids] = self._motion_time()[env_ids]

    def _update_command(self):
        motion_time = self._motion_time()

        # motion wrapped or was resampled mid-episode (robot teleported): re-park + re-arm
        restart_ids = (motion_time < self._prev_motion_time).nonzero(as_tuple=False).flatten()
        if len(restart_ids) > 0:
            self._sample_spawn(restart_ids)
            self._write_parked_state(restart_ids)
            self.time_since_throw[restart_ids] = 0.0
            self.secured_steps[restart_ids] = 0
            self._arm(restart_ids)
        self._prev_motion_time = motion_time.clone()

        # launch when the motion reaches the launch moment
        launch_ids = (self._armed & (motion_time >= self._launch_time)).nonzero(as_tuple=False).flatten()
        if len(launch_ids) > 0:
            self._launch(launch_ids, flight_time=self._flight_time[launch_ids])
            self._armed[launch_ids] = False
            self._launched[launch_ids] = True

        # keep un-launched balls parked (both armed and past-window envs)
        hold_ids = (~self._launched).nonzero(as_tuple=False).flatten()
        if len(hold_ids) > 0:
            self._write_parked_state(hold_ids)

        self.time_since_throw += self._launched.float() * self._env.step_dt
        self._update_catch_state(in_flight=self._launched)


@configclass
class SyncedBallCommandCfg(BallCommandCfg):
    """Configuration for the motion-synchronized ball command."""

    class_type: type = SyncedBallCommand

    motion_command_name: str = "motion"
    """Name of the MotionCommand term to synchronize with (must be declared before
    this command in the CommandsCfg)."""

    catch_motion_time: float = MISSING
    """Time (s) into the motion clip at which the mocap catch (ball-hand contact)
    happens. Read it off the retargeted clip (e.g. with scripts/mimic/replay_npz.py)."""

    arrival_jitter: float = 0.05
    """Uniform jitter (±s) on the ball arrival time relative to the mocap catch
    moment, so the policy cannot blindly overfit the exact timing."""

    # throw_delay_range is unused in the synced variant (launch is motion-driven)
