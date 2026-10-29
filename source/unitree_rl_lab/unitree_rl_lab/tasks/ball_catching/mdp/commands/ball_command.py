from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse, sample_uniform, yaw_quat

from unitree_rl_lab.tasks.ball_catching import rubber_hand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

GRAVITY = 9.81


def flight_time_from_apex(release_z, target_z, apex_z):
    """Flight time (s) of a drag-free lob from ``release_z`` over ``apex_z`` down to ``target_z`` (heights in m).

    Works on floats and tensors; the apex must not be below the release or the target height.
    """
    return ((2.0 * (apex_z - release_z) / GRAVITY) ** 0.5) + ((2.0 * (apex_z - target_z) / GRAVITY) ** 0.5)


class BallCommand(CommandTerm):
    """Single-throw ball command: an underhand lob into a two-arm catch in front of the belly.

    The throw follows the interim report (LocoMuJoCo ch. 7.6.2, mjlab ch. 9.4) and the VICON ball recordings
    of the mocap sessions. On reset (per env) it is sampled in the robot's heading (yaw) frame and frozen,
    so moving toward the thrower gains nothing:

    - release point: ``throw_distance_range`` m from the root at azimuth ``throw_direction_range``
      around the heading, ``release_height_range`` m above the ground;
    - target point: ``target_forward_range`` / ``target_lateral_range`` in front of the root,
      ``target_height_range`` m above the ground;
    - apex height from ``apex_height_range``, which fixes the flight time.

    The ball is parked at the release point until the launch timer expires (``throw_delay_range``, optionally
    log-uniform so that long waits occur without making every episode long), so the policy has to stand and wait as
    in front of a real thrower. Then it is launched with the velocity that reaches the target after the flight time
    under gravity *and* quadratic air drag; the drag force is applied every control step during the flight.
    PhysX handles everything else (gravity, collisions).

    The command does NOT re-throw within an episode. Episode outcomes are owned by the termination terms
    (ball dropped / missed / caught / time out), which read the catch state tracked here:

    - ``secured_steps``: consecutive control steps for which the launched ball is inside the catch box in
      front of the torso, slow relative to the root, above ``secure_min_height``, in contact with the
      arms (both arms if ``require_both_arms``; with ``palm_hold`` both palms, holding it from below) and, with
      ``max_torso_force``, not squeezed against the torso. Reaching ``cfg.secure_steps`` confirms the catch
      (``confirmed``, ``confirmed_now`` on that one step).
    - ``catch_complete``: the confirmed catch has lasted ``hold_after_catch_s`` (hold phase: the robot keeps the
      ball and stands still, as after a real catch). With 0 the confirmation itself completes the catch.

    It also models how the robot *perceives* the ball through the VICON pipeline (actor observations only;
    rewards, catch check and critic use the true state). Every control step the tracker produces a
    measurement of the ball relative to the root in the yaw frame. The measurement is valid once the ball is
    in flight for ``detection_delay_range``, unless it drops out (``obs_dropout_prob`` per step, or the whole
    stay in the catch box with probability ``hide_in_catch_zone_prob`` per throw: the arms occlude it).
    The measurement reaches the robot ``obs_delay_steps_range`` steps late with noise and a per-throw bias.
    Missing measurements are bridged by ballistic prediction for ``max_coast_steps``; after that the ball
    counts as lost. Result: ``perceived_pos_b`` / ``perceived_vel_b`` (zero while not visible) and
    ``ball_visible``.
    """

    cfg: BallCommandCfg

    _NO_CONTACT = 1_000_000
    """Value of the steps-since-contact counter before the first contact of a throw."""

    def __init__(self, cfg: BallCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.ball: RigidObject = env.scene[cfg.asset_name]
        self.robot: Articulation = env.scene["robot"]

        # resolve hand body indices once
        self.hand_body_ids, _ = self.robot.find_bodies(cfg.hand_body_names, preserve_order=True)

        # ball contact sensor: the ball is the single sensor body, filtered against the arm links
        # (Isaac Lab only reports filtered contacts one-to-many). Columns are split into left/right arm.
        self.contact_sensor: ContactSensor | None = None
        self._torso_column: int | None = None
        self._hand_columns: torch.Tensor | None = None
        if cfg.contact_sensor_name is not None:
            self.contact_sensor = env.scene.sensors[cfg.contact_sensor_name]
            filter_names = [path.rsplit("/", 1)[-1] for path in self.contact_sensor.cfg.filter_prim_paths_expr]
            self._contact_columns = []
            for side in ("left_", "right_"):
                columns = [i for i, name in enumerate(filter_names) if name.startswith(side)]
                if not columns:
                    raise ValueError(
                        f"Contact sensor '{cfg.contact_sensor_name}' has no '{side}*' filter body: {filter_names}"
                    )
                self._contact_columns.append(torch.tensor(columns, dtype=torch.long, device=self.device))
            if cfg.torso_body_name is not None:
                if cfg.torso_body_name not in filter_names:
                    raise ValueError(
                        f"Contact sensor '{cfg.contact_sensor_name}' has no '{cfg.torso_body_name}' filter body: "
                        f"{filter_names}"
                    )
                self._torso_column = filter_names.index(cfg.torso_body_name)
            if cfg.palm_hold:
                missing = [name for name in cfg.hand_body_names if name not in filter_names]
                if missing:
                    raise ValueError(f"Contact sensor '{cfg.contact_sensor_name}' has no hand filter body {missing}")
                self._hand_columns = torch.tensor(
                    [filter_names.index(name) for name in cfg.hand_body_names], dtype=torch.long, device=self.device
                )
        # palm centre and normal of the [left, right] hand in its body frame (palm hold)
        self._palm_points = torch.tensor(cfg.palm_points, dtype=torch.float, device=self.device)
        self._palm_normals = torch.tensor(cfg.palm_normals, dtype=torch.float, device=self.device)

        # per-environment state
        self.time_until_throw = torch.zeros(self.num_envs, device=self.device)
        self.time_since_throw = torch.zeros(self.num_envs, device=self.device)
        self.secured_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.confirmed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        """The catch of the current throw is confirmed (``secured_steps`` reached ``secure_steps`` once)."""
        self.confirmed_now = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        """The catch was confirmed in the last update (true on exactly one step per throw)."""
        self.steps_since_confirm = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        """Control steps since the confirmation (0 on the confirming step and before)."""
        self.arm_contact = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        """Ball touches the [left, right] arm in the current control step (with ``palm_hold``: that hand's palm)."""
        self.torso_force = torch.zeros(self.num_envs, device=self.device)
        """Force (N) between the ball and ``cfg.torso_body_name`` in the current control step (0 without it)."""
        self._steps_since_contact = torch.full(
            (self.num_envs, 2), self._NO_CONTACT, dtype=torch.long, device=self.device
        )
        self._spawn_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._flight_time = torch.ones(self.num_envs, device=self.device)
        self._drag_coeff = torch.zeros(self.num_envs, device=self.device)
        """k = 0.5 * air density * drag coefficient * cross-section [kg/m]; drag force = -k |v| v."""
        self._ball_mass: torch.Tensor | None = None  # read at the first launch (after the startup mass DR)
        self._ball_area = math.pi * float(self.ball.cfg.spawn.radius) ** 2
        self._in_zone = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # perception (VICON model): ring buffer of the true measurements, read with a per-throw delay
        buffer_len = cfg.obs_delay_steps_range[1] + 1
        self._meas_pos_b = torch.zeros(self.num_envs, buffer_len, 3, device=self.device)
        self._meas_vel_b = torch.zeros(self.num_envs, buffer_len, 3, device=self.device)
        self._meas_valid = torch.zeros(self.num_envs, buffer_len, dtype=torch.bool, device=self.device)
        self._meas_index = 0
        self._obs_delay = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._detection_delay = torch.zeros(self.num_envs, device=self.device)
        self._obs_bias = torch.zeros(self.num_envs, 3, device=self.device)
        self._hide_in_zone = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._steps_since_measurement = torch.full(
            (self.num_envs,), self._NO_CONTACT, dtype=torch.long, device=self.device
        )
        self._dropout_left = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._last_pos_b = torch.zeros(self.num_envs, 3, device=self.device)
        self._last_vel_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.perceived_pos_b = torch.zeros(self.num_envs, 3, device=self.device)
        """Ball position relative to the root in the yaw frame as the robot perceives it (zero if not visible)."""
        self.perceived_vel_b = torch.zeros(self.num_envs, 3, device=self.device)
        """Ball velocity in the yaw frame as the robot perceives it (zero if not visible)."""
        self.ball_visible = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        """The robot has a ball estimate (fresh measurement or ballistic coasting)."""

        # identity quaternion (w,x,y,z) — ball orientation doesn't matter
        self._default_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        # gravity vector for the ballistic velocity computation
        self._gravity = torch.tensor([0.0, 0.0, -9.81], device=self.device)
        self._catch_zone_lo = torch.tensor(
            [cfg.catch_zone_x[0], cfg.catch_zone_y[0], cfg.catch_zone_z[0]], device=self.device
        )
        self._catch_zone_hi = torch.tensor(
            [cfg.catch_zone_x[1], cfg.catch_zone_y[1], cfg.catch_zone_z[1]], device=self.device
        )

        self.metrics["ball_height"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["ball_secured_time"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["hold_time_after_confirm"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["hand_ball_distance"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["ball_in_catch_zone"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["ball_arm_contact"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["ball_torso_force"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """Ball position (3) + linear velocity (3) in world frame, shape (num_envs, 6).

        Note: not used as a policy observation (world coordinates include per-env
        origins). Policy observations use the yaw-frame terms in ``observations.py``.
        """
        return torch.cat([self.ball.data.root_pos_w, self.ball.data.root_lin_vel_w], dim=1)

    @property
    def launched(self) -> torch.Tensor:
        """Whether the ball of the current throw is in the air (or caught/dropped), shape (num_envs,)."""
        return self.time_until_throw <= 0.0

    @property
    def catch_complete(self) -> torch.Tensor:
        """The confirmed catch has lasted ``hold_after_catch_s``: the throw succeeded, shape (num_envs,)."""
        held = self.steps_since_confirm.float() * self._env.step_dt >= self.cfg.hold_after_catch_s - 1e-6
        return self.confirmed & held

    def _update_metrics(self):
        self.metrics["ball_height"] = self.ball.data.root_pos_w[:, 2].clone()
        self.metrics["ball_secured_time"] = self.secured_steps.float() * self._env.step_dt
        self.metrics["hold_time_after_confirm"] = self.steps_since_confirm.float() * self._env.step_dt

    def _sample_throw(self, env_ids: Sequence[int]):
        """Sample release point, target point, flight time and drag of a new throw (heading frame)."""
        n = len(env_ids)
        device = self.device

        root_pos = self.robot.data.root_pos_w[env_ids]
        heading = yaw_quat(self.robot.data.root_quat_w[env_ids])
        ground_z = self._env.scene.env_origins[env_ids, 2]

        distance = sample_uniform(*self.cfg.throw_distance_range, (n,), device)
        azimuth = sample_uniform(*self.cfg.throw_direction_range, (n,), device)
        release = torch.stack([distance * torch.cos(azimuth), distance * torch.sin(azimuth), torch.zeros_like(distance)], dim=1)
        spawn = root_pos + quat_apply(heading, release)
        spawn[:, 2] = ground_z + sample_uniform(*self.cfg.release_height_range, (n,), device)
        self._spawn_pos[env_ids] = spawn

        aim = torch.zeros(n, 3, device=device)
        aim[:, 0] = sample_uniform(*self.cfg.target_forward_range, (n,), device)
        aim[:, 1] = sample_uniform(*self.cfg.target_lateral_range, (n,), device)
        target = root_pos + quat_apply(heading, aim)
        target[:, 2] = ground_z + sample_uniform(*self.cfg.target_height_range, (n,), device)
        self._target_pos[env_ids] = target

        # the lob's apex fixes the flight time; farther throws are then faster, as with a human thrower
        apex = ground_z + sample_uniform(*self.cfg.apex_height_range, (n,), device)
        apex = torch.maximum(apex, torch.maximum(spawn[:, 2], target[:, 2]) + 0.05)
        self._flight_time[env_ids] = flight_time_from_apex(spawn[:, 2], target[:, 2], apex)
        drag_coefficient = sample_uniform(*self.cfg.drag_coefficient_range, (n,), device)
        self._drag_coeff[env_ids] = 0.5 * self.cfg.air_density * drag_coefficient * self._ball_area

    def _resample_command(self, env_ids: Sequence[int]):
        """Sample a new throw, park the ball at its release point and arm the launch timer."""
        if len(env_ids) == 0:
            return

        self._sample_throw(env_ids)
        self._write_parked_state(env_ids)

        low, high = self.cfg.throw_delay_range
        if self.cfg.throw_delay_log_uniform:
            self.time_until_throw[env_ids] = low * (high / low) ** torch.rand(len(env_ids), device=self.device)
        else:
            self.time_until_throw[env_ids] = sample_uniform(low, high, (len(env_ids),), self.device)
        self.time_since_throw[env_ids] = 0.0
        self._reset_catch_state(env_ids)
        self._reset_perception(env_ids)

    def _reset_catch_state(self, env_ids: Sequence[int]):
        self.secured_steps[env_ids] = 0
        self.confirmed[env_ids] = False
        self.confirmed_now[env_ids] = False
        self.steps_since_confirm[env_ids] = 0
        self.arm_contact[env_ids] = False
        self._steps_since_contact[env_ids] = self._NO_CONTACT

    def _reset_perception(self, env_ids: Sequence[int]):
        """Forget the previous throw and sample the tracker imperfections of the new one."""
        n = len(env_ids)
        cfg = self.cfg
        self._meas_valid[env_ids] = False
        self._steps_since_measurement[env_ids] = self._NO_CONTACT
        self._dropout_left[env_ids] = 0
        self.perceived_pos_b[env_ids] = 0.0
        self.perceived_vel_b[env_ids] = 0.0
        self.ball_visible[env_ids] = False
        # the buffer is sized for the default (= largest) delay range; the curriculum only shrinks it
        self._obs_delay[env_ids] = torch.randint(
            cfg.obs_delay_steps_range[0], cfg.obs_delay_steps_range[1] + 1, (n,), device=self.device
        ).clamp(max=self._meas_valid.shape[1] - 1)
        self._detection_delay[env_ids] = sample_uniform(*cfg.detection_delay_range, (n,), self.device)
        self._obs_bias[env_ids] = sample_uniform(-cfg.obs_pos_bias, cfg.obs_pos_bias, (n, 3), self.device)
        self._hide_in_zone[env_ids] = torch.rand(n, device=self.device) < cfg.hide_in_catch_zone_prob

    def _write_parked_state(self, env_ids: Sequence[int]):
        """Hold the ball at its release point with zero velocity."""
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
        """Throw the ball from its release point so that it reaches the target after the flight time."""
        n = len(env_ids)
        if self._ball_mass is None:
            # ball mass randomization is a startup event, so the per-env masses are final here
            self._ball_mass = self.ball.root_physx_view.get_masses()[:, 0].to(self.device)

        throw_vel = self._solve_launch_velocity(
            self._spawn_pos[env_ids],
            self._target_pos[env_ids],
            self._flight_time[env_ids],
            self._drag_coeff[env_ids] / self._ball_mass[env_ids],
        )
        throw_vel += torch.randn(n, 3, device=self.device) * self.cfg.velocity_noise

        root_state = torch.cat(
            [
                self._spawn_pos[env_ids],
                self._default_quat.unsqueeze(0).expand(n, -1),
                throw_vel,
                torch.zeros(n, 3, device=self.device),
            ],
            dim=1,
        )
        self.ball.write_root_state_to_sim(root_state, env_ids=env_ids)

        self.time_since_throw[env_ids] = 0.0

    def _solve_launch_velocity(
        self, start: torch.Tensor, target: torch.Tensor, flight_time: torch.Tensor, drag_per_mass: torch.Tensor
    ) -> torch.Tensor:
        """Initial velocity that reaches ``target`` from ``start`` after ``flight_time`` with gravity and drag.

        Fixed-point iteration on  p(T) = p0 + v0 T + g T²/2 + D(v0),  where D is the displacement caused by
        the drag (see :meth:`_drag_displacement`). Without drag this is the plain ballistic solution.
        """
        t = flight_time.unsqueeze(1)
        ballistic = target - start - 0.5 * self._gravity * t**2
        throw_vel = ballistic / t
        for _ in range(self.cfg.drag_solver_iterations):
            throw_vel = (ballistic - self._drag_displacement(throw_vel, flight_time, drag_per_mass)) / t
        return throw_vel

    def _drag_displacement(
        self, throw_vel: torch.Tensor, flight_time: torch.Tensor, drag_per_mass: torch.Tensor
    ) -> torch.Tensor:
        """Displacement after ``flight_time`` caused by the drag acceleration a = -(k/m) |v| v.

        D = ∫0^T (T - s) a(s) ds, evaluated with the trapezoid rule on ``drag_solver_nodes`` intervals. The
        velocity along the flight is refined by two Picard passes (gravity-only velocity, then corrected by
        the integrated drag), which is accurate to a few mm for the ball speeds here.
        """
        nodes = self.cfg.drag_solver_nodes
        times = flight_time.unsqueeze(1) * torch.linspace(0.0, 1.0, nodes + 1, device=self.device)  # (n, K+1)
        dt = (flight_time / nodes).view(-1, 1, 1)
        c = drag_per_mass.view(-1, 1, 1)
        vel_free = throw_vel.unsqueeze(1) + self._gravity * times.unsqueeze(-1)  # (n, K+1, 3)
        vel = vel_free
        for _ in range(2):
            acc = -c * torch.norm(vel, dim=-1, keepdim=True) * vel
            dv = torch.cumsum(0.5 * (acc[:, 1:] + acc[:, :-1]) * dt, dim=1)
            vel = vel_free + torch.cat([torch.zeros_like(dv[:, :1]), dv], dim=1)
        acc = -c * torch.norm(vel, dim=-1, keepdim=True) * vel
        integrand = (flight_time.unsqueeze(1) - times).unsqueeze(-1) * acc
        return (0.5 * (integrand[:, 1:] + integrand[:, :-1]) * dt).sum(dim=1)

    def _apply_drag(self, in_flight: torch.Tensor):
        """Quadratic air drag on the flying ball, held constant over the physics steps of a control step."""
        vel = self.ball.data.root_lin_vel_w
        force = -(self._drag_coeff * in_flight.float()).unsqueeze(1) * torch.norm(vel, dim=1, keepdim=True) * vel
        composer = self.ball.permanent_wrench_composer
        # reset first: it refreshes the ball orientation used to convert the world-frame force to the body frame
        composer.reset()
        composer.set_forces_and_torques(forces=force.unsqueeze(1).contiguous(), is_global=True)

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

        in_flight = self.launched
        self.time_since_throw += in_flight.float() * step_dt

        self._apply_drag(in_flight)
        self._update_catch_state(in_flight)
        self._update_perception()

    def _palm_geometry(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Per hand whether the ball centre lies on its palm side (``min_palm_alignment``, ``max_palm_distance``),
        shape (N, 2), and the height (m) of the ball centre above the mean height of the two palm points, shape (N,)."""
        num = self.num_envs
        hand_quat = self.robot.data.body_quat_w[:, self.hand_body_ids].reshape(-1, 4)
        palm_pos = self.robot.data.body_pos_w[:, self.hand_body_ids] + quat_apply(
            hand_quat, self._palm_points.expand(num, 2, 3).reshape(-1, 3)
        ).view(num, 2, 3)
        normal = quat_apply(hand_quat, self._palm_normals.expand(num, 2, 3).reshape(-1, 3)).view(num, 2, 3)
        ball_pos = self.ball.data.root_pos_w
        to_ball = ball_pos[:, None, :] - palm_pos
        distance = torch.norm(to_ball, dim=-1)
        alignment = torch.sum(normal * to_ball, dim=-1) / distance.clamp(min=1e-6)
        on_palm = (alignment >= self.cfg.min_palm_alignment) & (distance <= self.cfg.max_palm_distance)
        return on_palm, ball_pos[:, 2] - palm_pos[..., 2].mean(dim=-1)

    def _update_arm_contact(self, on_palm: torch.Tensor | None = None) -> torch.Tensor:
        """Update ``arm_contact`` and return per arm whether the ball touched it within the grace window.

        With ``on_palm`` (palm hold, per hand) only the hand bodies count, and only with the ball on their palm side.
        """
        if self.contact_sensor is None:
            return torch.ones(self.num_envs, 2, dtype=torch.bool, device=self.device)
        data = self.contact_sensor.data
        # (N, T, 1, M, 3) -> strongest contact per filter body over the physics steps of this control step
        forces = data.force_matrix_w_history if data.force_matrix_w_history is not None else data.force_matrix_w.unsqueeze(1)
        magnitude = torch.norm(forces, dim=-1).amax(dim=1)[:, 0]
        if on_palm is not None:
            self.arm_contact = (magnitude[:, self._hand_columns] > self.cfg.contact_force_threshold) & on_palm
        else:
            self.arm_contact = torch.stack(
                [
                    magnitude[:, columns].amax(dim=-1) > self.cfg.contact_force_threshold
                    for columns in self._contact_columns
                ],
                dim=-1,
            )
        self._steps_since_contact = torch.where(
            self.arm_contact, torch.zeros_like(self._steps_since_contact), self._steps_since_contact + 1
        )
        if self._torso_column is not None:
            self.torso_force = magnitude[:, self._torso_column]
        return self._steps_since_contact <= self.cfg.contact_grace_steps

    def _update_catch_state(self, in_flight: torch.Tensor):
        """Track consecutive steps where the ball counts as held in the arms.

        Held := ball in the catch box (root yaw frame), slow relative to the root, above ground and touching
        the arms (with ``palm_hold``: both palms, ball above them) — gated by ``in_flight`` so a parked (pre-throw)
        ball never counts.
        """
        ball_pos = self.ball.data.root_pos_w
        ball_vel = self.ball.data.root_lin_vel_w
        root_pos = self.robot.data.root_pos_w

        ball_b = quat_apply_inverse(yaw_quat(self.robot.data.root_quat_w), ball_pos - root_pos)
        in_zone = torch.all((ball_b >= self._catch_zone_lo) & (ball_b <= self._catch_zone_hi), dim=-1)
        self._in_zone = in_zone
        rel_speed = torch.norm(ball_vel - self.robot.data.root_lin_vel_w, dim=-1)
        if self.cfg.palm_hold:
            on_palm, above_palms = self._palm_geometry()
            recent_contact = self._update_arm_contact(on_palm)
            touching = recent_contact.all(dim=-1) & (above_palms >= self.cfg.min_ball_above_palms)
        else:
            recent_contact = self._update_arm_contact()
            touching = recent_contact.all(dim=-1) if self.cfg.require_both_arms else recent_contact.any(dim=-1)

        secured = (
            in_flight
            & in_zone
            & (rel_speed < self.cfg.secure_rel_vel)
            & (ball_pos[:, 2] > self.cfg.secure_min_height)
            & touching
        )
        if self.cfg.max_torso_force is not None:
            secured &= self.torso_force <= self.cfg.max_torso_force  # held, not squeezed against the body
        self.secured_steps = torch.where(secured, self.secured_steps + 1, torch.zeros_like(self.secured_steps))
        # confirmation once per throw; the hold phase runs on even if the secured streak breaks (a drop ends it)
        self.confirmed_now = ~self.confirmed & (self.secured_steps >= self.cfg.secure_steps)
        self.steps_since_confirm += self.confirmed.long()
        self.confirmed |= self.confirmed_now

        hands_mid = self.robot.data.body_pos_w[:, self.hand_body_ids].mean(dim=1)
        self.metrics["hand_ball_distance"] = torch.norm(ball_pos - hands_mid, dim=-1)
        self.metrics["ball_in_catch_zone"] = in_zone.float()
        self.metrics["ball_arm_contact"] = self.arm_contact.float().mean(dim=-1)
        self.metrics["ball_torso_force"] = self.torso_force

    def _update_perception(self):
        """Update the ball estimate the robot gets from the VICON pipeline (see the class docstring)."""
        cfg = self.cfg
        n = self.num_envs
        heading = yaw_quat(self.robot.data.root_quat_w)
        pos_b = quat_apply_inverse(heading, self.ball.data.root_pos_w - self.robot.data.root_pos_w)
        vel_b = quat_apply_inverse(heading, self.ball.data.root_lin_vel_w)

        # what the tracker measures now; dropouts come in bursts of 1..obs_dropout_max_steps control steps
        detected = self.launched & (self.time_since_throw >= self._detection_delay)
        burst_start = (self._dropout_left == 0) & (torch.rand(n, device=self.device) < cfg.obs_dropout_prob)
        burst_length = torch.randint(1, cfg.obs_dropout_max_steps + 1, (n,), device=self.device)
        self._dropout_left = torch.where(burst_start, burst_length, self._dropout_left)
        occluded = (self._dropout_left > 0) | (self._hide_in_zone & self._in_zone)
        self._dropout_left = (self._dropout_left - 1).clamp(min=0)
        self._meas_index = (self._meas_index + 1) % self._meas_valid.shape[1]
        self._meas_pos_b[:, self._meas_index] = pos_b
        self._meas_vel_b[:, self._meas_index] = vel_b
        self._meas_valid[:, self._meas_index] = detected & ~occluded

        # what reaches the robot: the measurement of ``delay`` steps ago, noisy and biased
        env_ids = torch.arange(n, device=self.device)
        read = (self._meas_index - self._obs_delay) % self._meas_valid.shape[1]
        valid = self._meas_valid[env_ids, read]
        meas_pos = self._meas_pos_b[env_ids, read] + self._obs_bias + torch.randn(n, 3, device=self.device) * cfg.obs_pos_noise_std
        meas_vel = self._meas_vel_b[env_ids, read] + torch.randn(n, 3, device=self.device) * cfg.obs_vel_noise_std

        # missing measurement: coast on the last estimate with the ballistic model
        self._steps_since_measurement = torch.where(
            valid, torch.zeros_like(self._steps_since_measurement), self._steps_since_measurement + 1
        )
        coast_t = (self._steps_since_measurement.float() * self._env.step_dt).unsqueeze(1)
        coast_pos = self._last_pos_b + self._last_vel_b * coast_t + 0.5 * self._gravity * coast_t**2
        coast_vel = self._last_vel_b + self._gravity * coast_t
        self._last_pos_b = torch.where(valid.unsqueeze(1), meas_pos, self._last_pos_b)
        self._last_vel_b = torch.where(valid.unsqueeze(1), meas_vel, self._last_vel_b)

        self.ball_visible = self._steps_since_measurement <= cfg.max_coast_steps
        visible = self.ball_visible.unsqueeze(1).float()
        self.perceived_pos_b = torch.where(valid.unsqueeze(1), meas_pos, coast_pos) * visible
        self.perceived_vel_b = torch.where(valid.unsqueeze(1), meas_vel, coast_vel) * visible

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass


@configclass
class BallCommandCfg(CommandTermCfg):
    """Configuration for the single-throw ball command.

    Defaults are the FINAL throw distribution (the curriculum starts from easier values).
    """

    class_type: type = BallCommand

    asset_name: str = MISSING

    hand_body_names: list[str] = ["left_wrist_yaw_link", "right_wrist_yaw_link"]
    """Bodies whose midpoint is the reach target (shaping metric; later: palm links of the dexterous hands)."""

    contact_sensor_name: str | None = "ball_contact"
    """Contact sensor on the ball, filtered against the arm links whose names start with ``left_`` / ``right_``.
    ``None`` drops the contact condition from the catch check."""

    # ── throw generation: underhand lob, heading frame of the robot. Calibrated on 12 VICON throws of the
    #    2026-07-06 mocap session (release ≈3.1 m from the catcher at 0.93–1.10 m, apex 1.76–2.03 m, flight
    #    0.69–0.86 s); distance and direction ranges from mjlab (interim report ch. 9.4) ──
    throw_delay_range: tuple[float, float] = (0.5, 1.5)
    """Time (s) after reset before the ball is launched. The ball is parked at the release point in the
    meantime (invisible to the actor, see the perception model): the robot has to stand and wait."""

    throw_delay_log_uniform: bool = False
    """Sample the throw delay log-uniformly instead of uniformly (lower bound > 0). For (0.5, 8) s the median is 2 s
    and a quarter of the throws come after more than 4 s: long waits are trained without halving the throws per
    training step."""

    throw_distance_range: tuple[float, float] = (2.0, 5.0)
    """Horizontal distance (m) of the release point from the robot root (mjlab: 2 m -> 5 m; VICON throws ≈3.1 m)."""

    throw_direction_range: tuple[float, float] = (-math.radians(75.0), math.radians(75.0))
    """Azimuth (rad) of the release point around the robot heading, positive = robot's left
    (mjlab: ±15° -> ±75°)."""

    release_height_range: tuple[float, float] = (0.85, 1.2)
    """Release height (m) above the ground (VICON throws: 0.93–1.10 m)."""

    target_forward_range: tuple[float, float] = (0.3, 0.4)
    """Aim point distance (m) in front of the robot root (LocoMuJoCo: 0.35 m)."""

    target_lateral_range: tuple[float, float] = (-0.1, 0.1)
    """Aim point lateral offset (m). Curriculum variable."""

    target_height_range: tuple[float, float] = (0.8, 1.05)
    """Aim point height (m) above the ground, in front of the belly (LocoMuJoCo: 0.8–1.05 m)."""

    apex_height_range: tuple[float, float] = (1.75, 2.1)
    """Apex height (m) above the ground of the lob (VICON throws: 1.76–2.03 m). Sets the flight time
    (here 0.71–1.02 s) and thereby the throw speed for the sampled distance."""

    velocity_noise: float = 0.1
    """Std (m/s) of Gaussian noise added per axis to the launch velocity."""

    # ── air drag (matters for a light 15 cm ball: ~0.3 m shorter over a real 3 m throw) ──
    air_density: float = 1.2
    """Air density (kg/m³)."""

    drag_coefficient_range: tuple[float, float] = (0.3, 0.45)
    """Drag coefficient of the ball, sampled per throw. The VICON throws fit k/m = 0.030–0.040 1/m, i.e.
    c_d ≈ 0.32–0.42 for the ∅15 cm / 112 g ball. (0, 0) disables drag."""

    drag_solver_iterations: int = 3
    """Fixed-point iterations for the drag-compensated launch velocity."""

    drag_solver_nodes: int = 16
    """Trapezoid intervals used to integrate the drag over the flight (16: aim error < 1 cm vs. RK4)."""

    # ── catch detection ──
    catch_zone_x: tuple[float, float] = (0.0, 0.6)
    catch_zone_y: tuple[float, float] = (-0.35, 0.35)
    catch_zone_z: tuple[float, float] = (-0.3, 0.5)
    """Catch box (m) in the robot's yaw frame relative to the root: in front of the torso, from below the
    belly up to the chest (excludes balls held to the side or above the head)."""

    secure_rel_vel: float = 0.5
    """Max ball speed (m/s) relative to the robot root for the ball to count as held."""

    secure_min_height: float = 0.5
    """Min ball height (m) for the ball to count as held (excludes balls on the ground)."""

    require_both_arms: bool = True
    """Two-arm catch: the ball must touch the left AND the right arm."""

    palm_hold: bool = False
    """The ball counts as held only in both palms, from below (as in Phillip's g1-catch): each hand of
    ``hand_body_names`` ([left, right], bodies of the contact sensor) touches it with the ball centre on its palm side
    (``min_palm_alignment``) within ``max_palm_distance`` of its palm point, and the ball centre is at least
    ``min_ball_above_palms`` above the mean height of the palm points. Contacts of the other arm links (forearm, wrist)
    count for nothing, also in ``arm_contact`` (reward ``ball_arm_contact``); ``require_both_arms`` is implied.
    False: any contact of the sensor's arm links counts."""

    palm_points: tuple = rubber_hand.PALM_POINTS
    """Palm centre (m) of the [left, right] hand in its body frame (rubber hand of the G1)."""

    palm_normals: tuple = rubber_hand.PALM_NORMALS
    """Normal out of the [left, right] palm in its body frame."""

    min_palm_alignment: float = 0.5
    """Min cosine between the palm normal and the direction from the palm point to the ball centre (0.5: 60°). At the
    recorded catches of the clips the worse hand reaches 0.78 (median)."""

    max_palm_distance: float = 0.18
    """Max distance (m) from each palm point to the ball centre (ball radius 0.075)."""

    min_ball_above_palms: float = 0.02
    """Min height (m) of the ball centre above the mean height of the two palm points. At the recorded catches of the
    clips the catch point is 0.049 m above them (median); 79 % of the clips meet this and ``min_palm_alignment``."""

    contact_force_threshold: float = 0.1
    """Min normal force (N) between ball and an arm link that counts as contact (ball weight ≈ 1.1 N)."""

    torso_body_name: str | None = None
    """Filter body of the contact sensor that is the torso; needed for ``max_torso_force``."""

    max_torso_force: float | None = None
    """A ball pressed against the torso harder than this (N) does not count as held. Resting against the belly is
    allowed; clamping the ball to the chest (Phase 2 baseline 236 N, Unified runs 196-233 N) is not a catch with the
    arms. ``None``: no limit."""

    contact_grace_steps: int = 5
    """A contact within the last N control steps still counts (rolling or bouncing ball)."""

    secure_steps: int = 50
    """Consecutive held steps for a confirmed catch (50 steps = 1.0 s at 50 Hz; 0.5 s counted balls
    that rolled off afterwards). Curriculum variable."""

    hold_after_catch_s: float = 0.0
    """Hold phase (s) after the confirmed catch before the throw counts as completed (``catch_complete``, the
    ``ball_caught`` termination). A drop during it ends the episode as a failure. 0 = completed at the confirmation."""

    # ── perception: how the robot gets the ball from the VICON pipeline (actor observations only).
    #    Dropout and noise values are calibrated on the same 12 VICON throws ──
    detection_delay_range: tuple[float, float] = (0.04, 0.12)
    """Time (s) after the release until the pipeline reports a ball in flight (ballistic-fit window). Before
    that, e.g. during the thrower's wind-up, the ball is not visible."""

    obs_delay_steps_range: tuple[int, int] = (1, 5)
    """Latency (control steps of 20 ms) of the ball observation, sampled per throw (VICON + MQTT + filter;
    not measured yet, so up to 100 ms: 80-120 ms cost the 2026-10-27 baseline, trained on 1-3 steps, 6 points of
    catch rate). The default sizes the perception buffer and must hold the largest delay. If the deploy filter
    predicts the latency away, set this to the residual latency."""

    obs_pos_noise_std: float = 0.005
    """Std (m) of the position noise per axis (VICON throws: 4–5 mm around a drag-fitted parabola)."""

    obs_vel_noise_std: float = 0.1
    """Std (m/s) of the velocity noise per axis (estimated by the deploy filter; assumption)."""

    obs_pos_bias: float = 0.01
    """Half range (m) of a per-throw constant position offset per axis (marker centroid, calibration)."""

    obs_dropout_prob: float = 0.05
    """Probability per control step that a dropout burst starts (VICON throws: 0.047)."""

    obs_dropout_max_steps: int = 3
    """Burst length is uniform in 1..N control steps (VICON throws: 11×1, 7×2, 2×3; 6.8 % of the steps)."""

    hide_in_catch_zone_prob: float = 0.5
    """Probability per throw that the arms occlude the ball for its whole stay in the catch box (VICON throws:
    about a third lost the ball at the catch, the rest tracked it for another 0.25–1.2 s)."""

    max_coast_steps: int = 3
    """Missing measurements bridged by ballistic prediction (the tracker gives up after 8 frames ≈ 67 ms);
    after that the ball counts as not visible."""
