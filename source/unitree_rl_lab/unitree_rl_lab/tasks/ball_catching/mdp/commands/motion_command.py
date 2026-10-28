from __future__ import annotations

import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MotionLoader:
    """One or more clips as padded tensors indexed (clip, frame, ...).

    Shorter clips are padded with their last frame; ``time_step_total`` holds each clip's real length, so callers
    never read the padding. All clips must share one frame rate.

    Optionally every clip becomes a wait-and-hold cycle: its first frame held for ``start_hold_s`` in front (standing
    before the action), the clip up to ``hold_from_s[clip]`` (``None``: to its end) and that frame held for
    ``end_hold_s`` (keeping the pose). The held frames repeat positions and orientations with zero velocities.
    ``start_hold_frames`` is the length of the lead-in, the same for all clips.
    """

    def __init__(
        self,
        motion_files: Sequence[str],
        body_indexes: Sequence[int],
        device: str = "cpu",
        start_hold_s: float = 0.0,
        hold_from_s: Sequence[float | None] | None = None,
        end_hold_s: float = 0.0,
    ):
        assert len(motion_files) > 0, "no motion files given"
        clips = []
        for motion_file in motion_files:
            assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
            clips.append(np.load(motion_file))
        rates = sorted({float(np.asarray(clip["fps"]).reshape(-1)[0]) for clip in clips})
        assert len(rates) == 1, f"motion clips have different frame rates: {rates}"
        self.fps = rates[0]
        self.files = list(motion_files)
        self.num_clips = len(clips)
        if hold_from_s is None:
            hold_from_s = [None] * len(clips)
        assert len(hold_from_s) == len(clips), f"hold_from_s has {len(hold_from_s)} entries for {len(clips)} clips"
        self.start_hold_frames = int(round(start_hold_s * self.fps))
        end_hold_frames = int(round(end_hold_s * self.fps))
        # last clip frame used (then held), per clip
        last = [
            clip["joint_pos"].shape[0] - 1 if t is None else min(int(round(t * self.fps)), clip["joint_pos"].shape[0] - 1)
            for clip, t in zip(clips, hold_from_s)
        ]
        lengths = [self.start_hold_frames + k + 1 + end_hold_frames for k in last]
        self.time_step_total = torch.tensor(lengths, dtype=torch.long, device=device)

        def stacked(key: str, bodies: Sequence[int] | None = None, velocity: bool = False) -> torch.Tensor:
            padded = []
            for clip, k in zip(clips, last):
                data = (clip[key] if bodies is None else clip[key][:, bodies])[: k + 1]
                first, held = (np.zeros_like(data[:1]), np.zeros_like(data[-1:])) if velocity else (data[:1], data[-1:])
                padded.append(np.concatenate([
                    np.repeat(first, self.start_hold_frames, axis=0),
                    data,
                    np.repeat(held, max(lengths) - self.start_hold_frames - len(data), axis=0),  # end hold + padding
                ]))
            return torch.tensor(np.stack(padded), dtype=torch.float32, device=device)

        # bodies are selected once here (the command only tracks cfg.body_names), not on every access
        body_indexes = [int(i) for i in body_indexes]
        self.joint_pos = stacked("joint_pos")
        self.joint_vel = stacked("joint_vel", velocity=True)
        self.body_pos_w = stacked("body_pos_w", body_indexes)
        self.body_quat_w = stacked("body_quat_w", body_indexes)
        self.body_lin_vel_w = stacked("body_lin_vel_w", body_indexes, velocity=True)
        self.body_ang_vel_w = stacked("body_ang_vel_w", body_indexes, velocity=True)


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        self.motion = MotionLoader(
            self.cfg.motion_files,
            self.body_indexes,
            device=self.device,
            start_hold_s=self.cfg.start_hold_s,
            hold_from_s=self.cfg.hold_from_s,
            end_hold_s=self.cfg.end_hold_s,
        )
        control_rate = 1.0 / (env.cfg.decimation * env.cfg.sim.dt)
        if abs(self.motion.fps - control_rate) > 1e-3:
            raise ValueError(
                f"motion clips run at {self.motion.fps} Hz, but the command advances one frame per control step "
                f"({control_rate} Hz); resample the clips"
            )
        self.clip_ids = torch.randint(0, self.motion.num_clips, (self.num_envs,), device=self.device)
        """Clip each env tracks; a new clip is drawn whenever the env is reset or its clip ends."""
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.resample_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        """Incremented whenever clip and time are resampled (reset or clip end), so that other terms can detect a
        new cycle even when the new start time lies after the old one."""
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        # adaptive sampling: one bin per second of the longest clip, applied to each clip's normalized time
        self.bin_count = int(int(self.motion.time_step_total.max()) // control_rate) + 1
        self.bin_failed_count = torch.zeros(self.motion.num_clips, self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(
            self.motion.num_clips, self.bin_count, dtype=torch.float, device=self.device
        )
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.clip_ids, self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.clip_ids, self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.clip_ids, self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.clip_ids, self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.clip_ids, self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.clip_ids, self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return (
            self.motion.body_pos_w[self.clip_ids, self.time_steps, self.motion_anchor_body_index]
            + self._env.scene.env_origins
        )

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.clip_ids, self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.clip_ids, self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.clip_ids, self.time_steps, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        num_clips = self.motion.num_clips
        # failures are counted in the (clip, bin) the env was in, before it gets a new clip
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            failed = env_ids[episode_failed]
            lengths = self.motion.time_step_total[self.clip_ids[failed]].clamp(min=1)
            bins = torch.clamp((self.time_steps[failed] * self.bin_count) // lengths, 0, self.bin_count - 1)
            flat = torch.bincount(self.clip_ids[failed] * self.bin_count + bins, minlength=num_clips * self.bin_count)
            self._current_bin_failed[:] = flat.view(num_clips, self.bin_count).float()

        # new clip (uniform), then a start bin within it from that clip's failure statistics
        self.clip_ids[env_ids] = torch.randint(0, num_clips, (len(env_ids),), device=self.device)
        sampling_probabilities = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(1), (0, self.cfg.adaptive_kernel_size - 1), mode="replicate"
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1))
        sampling_probabilities = sampling_probabilities.squeeze(1)
        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum(dim=1, keepdim=True)

        clip_probabilities = sampling_probabilities[self.clip_ids[env_ids]]
        sampled_bins = torch.multinomial(clip_probabilities, 1).squeeze(-1)
        lengths = self.motion.time_step_total[self.clip_ids[env_ids]]
        self.time_steps[env_ids] = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count
            * (lengths - 1)
        ).long()
        # part of the cycles starts inside the static lead-in: nothing fails there, so the adaptive sampling alone
        # would pick it ever more rarely and long waits would vanish from training
        lead_in = self.motion.start_hold_frames
        if lead_in > 0:
            if self.cfg.start_hold_from_begin:
                self.time_steps[env_ids] = 0
            else:
                waiting = env_ids[torch.rand(len(env_ids), device=self.device) < self.cfg.start_hold_prob]
                self.time_steps[waiting] = torch.randint(0, lead_in, (len(waiting),), device=self.device)

        # Metrics (of the clip each resampled env now tracks)
        H = -(clip_probabilities * (clip_probabilities + 1e-12).log()).sum(dim=1)
        pmax, imax = clip_probabilities.max(dim=1)
        self.metrics["sampling_entropy"][env_ids] = H / math.log(self.bin_count) if self.bin_count > 1 else H
        self.metrics["sampling_top1_prob"][env_ids] = pmax
        self.metrics["sampling_top1_bin"][env_ids] = imax.float() / self.bin_count

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        self._adaptive_sampling(env_ids)
        self.resample_count[env_ids] += 1

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

    def _update_command(self):
        self.time_steps += 1
        env_ids = torch.where(self.time_steps >= self.motion.time_step_total[self.clip_ids])[0]
        self._resample_command(env_ids)

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_files: list[str] = MISSING
    """Reference clips (``.npz``, all at the control rate). Each env tracks one, drawn uniformly at every
    resample."""
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    # ── wait-and-hold cycle (Unified; all zero/None: the plain clips, as in Phase 1) ──
    start_hold_s: float = 0.0
    """Seconds the first frame of every clip is held in front of it (standing still, zero velocities)."""
    start_hold_prob: float = 0.0
    """Share of the cycles that start uniformly inside that lead-in; the others use the adaptive sampling over the
    whole extended clip."""
    start_hold_from_begin: bool = False
    """Start every cycle at the beginning of the lead-in (evaluation: the longest wait)."""
    hold_from_s: list[float | None] | None = None
    """Per clip (order of ``motion_files``): time (s) into the clip from which its frame is held and the rest cut
    off; ``None`` keeps the clip whole."""
    end_hold_s: float = 0.0
    """Seconds the last (or cut) frame is held after the clip (keeping the pose, zero velocities)."""

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
