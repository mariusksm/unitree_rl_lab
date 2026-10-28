from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_inv, quat_mul, sample_uniform, yaw_quat

from ...motion_data import G1_HAND_BODY_INDEXES, G1_JOINT_NAMES
from .ball_command import BallCommand, BallCommandCfg
from .motion_command import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class SyncedBallCommand(BallCommand):
    """Ball command whose launch is synchronized to the mimic motion phase.

    For the unified (single-task) ball-catching environment: instead of a wall-clock delay after reset, the ball is
    launched so that it *arrives* where and when the reference catches it:

        launch_time = catch_time[clip] + jitter − flight_time

    Recorded mode (``catch_data`` set, computed by :func:`motion_data.catch_data` in the env config): every clip with
    a usable VICON recording of its take brings its own catch time, flight time and catch point. The release point is
    sampled as in :class:`BallCommand`, the flight time is the recorded one (the human reacts to the real release),
    and at the launch the target is the reference catch point re-anchored to the robot exactly like the tracked
    bodies of the :class:`MotionCommand` (robot anchor xy and yaw, reference anchor height), plus
    ``catch_point_jitter``. Clips without a usable recording get no throw and train pure tracking. Catch times are
    shifted by the motion command's lead-in (``start_hold_s``), during which the robot waits for the throw.

    Fixed mode (``catch_motion_time`` set, no recordings): the catch time is the same for all clips, and target and
    flight time are sampled from the :class:`BallCommandCfg` ranges.

    Behavior per motion cycle:
    - While the motion time is before ``launch_time``, the ball is parked at its release point.
    - If an episode starts *past* the launch window (adaptive sampling can start anywhere in the clip), no throw
      happens this cycle; the episode trains pure motion tracking until the clip ends.
    - When the motion command resamples (reset or clip end, detected via its ``resample_count``), a fresh throw is
      sampled, the ball is re-parked and re-armed.

    Throw geometry, drag, perception and catch-state tracking (``secured_steps``) are inherited from
    :class:`BallCommand`.

    Note: the motion command must be declared *before* this command in the environment's ``CommandsCfg`` so that on
    env reset the motion time is already resampled when this term arms the throw.
    """

    cfg: SyncedBallCommandCfg

    def __init__(self, cfg: SyncedBallCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self._armed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._launched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._launch_time = torch.zeros(self.num_envs, device=self.device)
        self._cycle = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        # per-clip catch data, built at the first resample (the motion command does not exist yet in __init__)
        self._clip_has_catch: torch.Tensor | None = None
        self._clip_catch_time: torch.Tensor | None = None
        self._clip_flight_time: torch.Tensor | None = None
        self._clip_catch_point: torch.Tensor | None = None
        self._recorded = False

    @property
    def motion_command(self) -> MotionCommand:
        return self._env.command_manager.get_term(self.cfg.motion_command_name)

    @property
    def launched(self) -> torch.Tensor:
        return self._launched

    def _motion_time(self) -> torch.Tensor:
        """Current time (s) into the reference motion clip, per env."""
        motion = self.motion_command
        return motion.time_steps.float() / motion.motion.fps

    def _check_asset_constants(self):
        """The catch data was computed with the G1 constants of ``motion_data``; they must match the asset."""
        hand_ids = self.robot.find_bodies(self.cfg.hand_body_names, preserve_order=True)[0]
        if list(self.robot.joint_names) != G1_JOINT_NAMES or list(hand_ids) != G1_HAND_BODY_INDEXES:
            raise ValueError(
                "SyncedBallCommand: the catch data was computed with motion_data.G1_JOINT_NAMES / G1_HAND_BODY_INDEXES, "
                "which do not match the robot asset; update them (scripts/mimic/check_npz_fk.py)"
            )

    def _build_clip_catch_data(self):
        motion = self.motion_command
        num_clips = motion.motion.num_clips
        lead_in = motion.motion.start_hold_frames / motion.motion.fps  # the clip starts this late in the cycle
        has_catch = torch.zeros(num_clips, dtype=torch.bool)
        catch_time = torch.zeros(num_clips)
        flight_time = torch.ones(num_clips)
        catch_point = torch.zeros(num_clips, 3)

        if self.cfg.catch_data is not None:
            if len(self.cfg.catch_data) != num_clips:
                raise ValueError(
                    f"SyncedBallCommand: catch_data has {len(self.cfg.catch_data)} entries for {num_clips} motion clips"
                )
            self._check_asset_constants()
            for i, data in enumerate(self.cfg.catch_data):
                if data["valid"]:
                    has_catch[i] = True
                    catch_time[i] = data["catch_time"] + lead_in
                    # launch at the cycle start at the latest (without a lead-in, a few clips start after the release)
                    flight_time[i] = min(data["flight_time"], float(catch_time[i]) - self.cfg.arrival_jitter)
                    catch_point[i] = torch.tensor(data["catch_point"])
            self._recorded = bool(has_catch.any())
            reasons: dict[str, int] = {}
            for info in self.cfg.catch_data:
                if not info["valid"]:
                    key = next(
                        (k for k in ("no ball recording", "joint RMSE", "frame offset", "outside the clip",
                                     "flight time", "from the hands", "off the human") if k in info["reason"]),
                        info["reason"],
                    )
                    reasons[key] = reasons.get(key, 0) + 1
            print(
                f"[SyncedBallCommand] recorded catches for {int(has_catch.sum())} of {num_clips} clips"
                + (f"; no throw for the rest ({reasons})" if reasons else "")
            )
        if not self._recorded:
            if self.cfg.catch_motion_time is None:
                raise ValueError(
                    "SyncedBallCommand: no usable ball recordings and no catch_motion_time; set catch_data "
                    "(motion_data.catch_data of the clips' recordings) or a fixed catch_motion_time"
                )
            durations = motion.motion.time_step_total.cpu().float() / motion.motion.fps
            has_catch[:] = durations > self.cfg.catch_motion_time + lead_in
            catch_time[:] = self.cfg.catch_motion_time + lead_in
            print(
                f"[SyncedBallCommand] fixed catch time {self.cfg.catch_motion_time:.2f} s for "
                f"{int(has_catch.sum())} of {num_clips} clips (the others are shorter)"
            )

        self._clip_has_catch = has_catch.to(self.device)
        self._clip_catch_time = catch_time.to(self.device)
        self._clip_flight_time = flight_time.to(self.device)
        self._clip_catch_point = catch_point.to(self.device)

    def _reference_catch_point_w(self, env_ids: torch.Tensor) -> torch.Tensor:
        """The env's reference catch point, re-anchored to the robot like MotionCommand.body_pos_relative_w."""
        motion = self.motion_command
        catch_ref_w = self._clip_catch_point[motion.clip_ids[env_ids]] + self._env.scene.env_origins[env_ids]
        anchor_pos_w = motion.anchor_pos_w[env_ids]
        robot_anchor_pos_w = motion.robot_anchor_pos_w[env_ids]
        delta_pos_w = robot_anchor_pos_w.clone()
        delta_pos_w[:, 2] = anchor_pos_w[:, 2]
        delta_ori_w = yaw_quat(quat_mul(motion.robot_anchor_quat_w[env_ids], quat_inv(motion.anchor_quat_w[env_ids])))
        return delta_pos_w + quat_apply(delta_ori_w, catch_ref_w - anchor_pos_w)

    def _sample_throw(self, env_ids: Sequence[int]):
        super()._sample_throw(env_ids)
        if self._recorded:
            self._flight_time[env_ids] = self._clip_flight_time[self.motion_command.clip_ids[env_ids]]

    def _launch(self, env_ids: torch.Tensor):
        if self._recorded:
            jitter = torch.tensor(self.cfg.catch_point_jitter, device=self.device)
            noise = sample_uniform(-jitter, jitter, (len(env_ids), 3), self.device)
            self._target_pos[env_ids] = self._reference_catch_point_w(env_ids) + noise
        super()._launch(env_ids)

    def _arm(self, env_ids: Sequence[int]):
        """Derive the launch moment from the catch time and flight time; arm envs whose launch is still ahead."""
        n = len(env_ids)
        clip_ids = self.motion_command.clip_ids[env_ids]
        jitter = sample_uniform(-self.cfg.arrival_jitter, self.cfg.arrival_jitter, (n,), self.device)
        self._launch_time[env_ids] = self._clip_catch_time[clip_ids] + jitter - self._flight_time[env_ids]
        self._armed[env_ids] = self._clip_has_catch[clip_ids] & (self._motion_time()[env_ids] <= self._launch_time[env_ids])
        self._launched[env_ids] = False

    def _start_cycle(self, env_ids: Sequence[int]):
        """Fresh throw for a new motion cycle: sample it, park the ball, reset the catch state, arm."""
        self._sample_throw(env_ids)
        self._write_parked_state(env_ids)
        self.time_since_throw[env_ids] = 0.0
        self._reset_catch_state(env_ids)
        self._reset_perception(env_ids)
        self._arm(env_ids)
        self._cycle[env_ids] = self.motion_command.resample_count[env_ids]

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        if self._clip_has_catch is None:
            self._build_clip_catch_data()
        self._start_cycle(env_ids)

    def _update_command(self):
        # the motion command resampled clip or time (reset or clip end; robot teleported): new throw
        restart_ids = (self.motion_command.resample_count != self._cycle).nonzero(as_tuple=False).flatten()
        if len(restart_ids) > 0:
            self._start_cycle(restart_ids)

        # launch when the motion reaches the launch moment
        motion_time = self._motion_time()
        launch_ids = (self._armed & (motion_time >= self._launch_time)).nonzero(as_tuple=False).flatten()
        if len(launch_ids) > 0:
            self._launch(launch_ids)
            self._armed[launch_ids] = False
            self._launched[launch_ids] = True

        # keep un-launched balls parked (both armed and past-window envs)
        hold_ids = (~self._launched).nonzero(as_tuple=False).flatten()
        if len(hold_ids) > 0:
            self._write_parked_state(hold_ids)

        self.time_since_throw += self._launched.float() * self._env.step_dt
        self._apply_drag(self._launched)
        self._update_catch_state(in_flight=self._launched)
        self._update_perception()


@configclass
class SyncedBallCommandCfg(BallCommandCfg):
    """Configuration for the motion-synchronized ball command."""

    class_type: type = SyncedBallCommand

    motion_command_name: str = "motion"
    """Name of the MotionCommand term to synchronize with (must be declared before
    this command in the CommandsCfg)."""

    catch_data: list[dict] | None = None
    """Recorded mode: per motion clip (order of ``motion_files``) the output of :func:`motion_data.catch_data`
    (``valid``, ``reason``; ``catch_time``, ``flight_time`` in s into the clip and ``catch_point`` in the clip frame)."""

    catch_motion_time: float | None = None
    """Fixed mode only (no usable recordings): time (s) into every clip at which the catch happens."""

    arrival_jitter: float = 0.05
    """Uniform jitter (±s) on the ball arrival time relative to the reference catch, so the policy cannot blindly
    overfit the exact timing."""

    catch_point_jitter: tuple[float, float, float] = (0.05, 0.05, 0.05)
    """Recorded mode: uniform jitter (±m per world axis) on the re-anchored reference catch point."""

    # throw_delay_range is unused in the synced variant (launch is motion-driven); in recorded mode the
    # target_*_range and apex_height_range fields are unused as well (target and flight time come from the clip)
