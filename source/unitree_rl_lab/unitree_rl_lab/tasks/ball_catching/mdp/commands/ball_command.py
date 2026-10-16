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

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class BallCommand(CommandTerm):
    """Single-throw ball command: an underhand lob into a two-arm catch in front of the belly.

    The throw follows the interim report (LocoMuJoCo ch. 7.6.2, mjlab ch. 9.4). On reset (per env) it is
    sampled in the robot's heading (yaw) frame and frozen, so moving toward the thrower gains nothing:

    - release point: ``throw_distance_range`` m from the root at azimuth ``throw_direction_range``
      around the heading, ``release_height_range`` m above the ground;
    - target point: ``target_forward_range`` / ``target_lateral_range`` in front of the root,
      ``target_height_range`` m above the ground;
    - flight time from ``flight_time_range``.

    The ball is parked at the release point until the launch timer expires, so the policy can settle into
    a stable stance. Then it is launched with the velocity that reaches the target after the flight time
    under gravity *and* quadratic air drag; the drag force is applied every control step during the flight.
    PhysX handles everything else (gravity, collisions).

    The command does NOT re-throw within an episode. Episode outcomes are owned by the termination terms
    (ball dropped / missed / caught / time out), which read the catch state tracked here:

    - ``secured_steps``: consecutive control steps for which the launched ball is inside the catch box in
      front of the torso, slow relative to the root, above ``secure_min_height`` and in contact with the
      arms (both arms if ``require_both_arms``). Reaching ``cfg.secure_steps`` counts as a confirmed catch.
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

        # per-environment state
        self.time_until_throw = torch.zeros(self.num_envs, device=self.device)
        self.time_since_throw = torch.zeros(self.num_envs, device=self.device)
        self.secured_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.arm_contact = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        """Ball touches the [left, right] arm in the current control step."""
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
        self.metrics["hand_ball_distance"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["ball_in_catch_zone"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["ball_arm_contact"] = torch.zeros(self.num_envs, device=self.device)

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

    def _update_metrics(self):
        self.metrics["ball_height"] = self.ball.data.root_pos_w[:, 2].clone()
        self.metrics["ball_secured_time"] = self.secured_steps.float() * self._env.step_dt

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

        self._flight_time[env_ids] = sample_uniform(*self.cfg.flight_time_range, (n,), device)
        drag_coefficient = sample_uniform(*self.cfg.drag_coefficient_range, (n,), device)
        self._drag_coeff[env_ids] = 0.5 * self.cfg.air_density * drag_coefficient * self._ball_area

    def _resample_command(self, env_ids: Sequence[int]):
        """Sample a new throw, park the ball at its release point and arm the launch timer."""
        if len(env_ids) == 0:
            return

        self._sample_throw(env_ids)
        self._write_parked_state(env_ids)

        self.time_until_throw[env_ids] = sample_uniform(*self.cfg.throw_delay_range, (len(env_ids),), self.device)
        self.time_since_throw[env_ids] = 0.0
        self._reset_catch_state(env_ids)

    def _reset_catch_state(self, env_ids: Sequence[int]):
        self.secured_steps[env_ids] = 0
        self.arm_contact[env_ids] = False
        self._steps_since_contact[env_ids] = self._NO_CONTACT

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

    def _update_arm_contact(self) -> torch.Tensor:
        """Update ``arm_contact`` and return per arm whether the ball touched it within the grace window."""
        if self.contact_sensor is None:
            return torch.ones(self.num_envs, 2, dtype=torch.bool, device=self.device)
        data = self.contact_sensor.data
        # (N, T, 1, M, 3) -> strongest contact per filter body over the physics steps of this control step
        forces = data.force_matrix_w_history if data.force_matrix_w_history is not None else data.force_matrix_w.unsqueeze(1)
        magnitude = torch.norm(forces, dim=-1).amax(dim=1)[:, 0]
        self.arm_contact = torch.stack(
            [magnitude[:, columns].amax(dim=-1) > self.cfg.contact_force_threshold for columns in self._contact_columns],
            dim=-1,
        )
        self._steps_since_contact = torch.where(
            self.arm_contact, torch.zeros_like(self._steps_since_contact), self._steps_since_contact + 1
        )
        return self._steps_since_contact <= self.cfg.contact_grace_steps

    def _update_catch_state(self, in_flight: torch.Tensor):
        """Track consecutive steps where the ball counts as held in the arms.

        Held := ball in the catch box (root yaw frame), slow relative to the root, above ground and touching
        the arms — gated by ``in_flight`` so a parked (pre-throw) ball never counts.
        """
        ball_pos = self.ball.data.root_pos_w
        ball_vel = self.ball.data.root_lin_vel_w
        root_pos = self.robot.data.root_pos_w

        ball_b = quat_apply_inverse(yaw_quat(self.robot.data.root_quat_w), ball_pos - root_pos)
        in_zone = torch.all((ball_b >= self._catch_zone_lo) & (ball_b <= self._catch_zone_hi), dim=-1)
        rel_speed = torch.norm(ball_vel - self.robot.data.root_lin_vel_w, dim=-1)
        recent_contact = self._update_arm_contact()
        touching = recent_contact.all(dim=-1) if self.cfg.require_both_arms else recent_contact.any(dim=-1)

        secured = (
            in_flight
            & in_zone
            & (rel_speed < self.cfg.secure_rel_vel)
            & (ball_pos[:, 2] > self.cfg.secure_min_height)
            & touching
        )
        self.secured_steps = torch.where(secured, self.secured_steps + 1, torch.zeros_like(self.secured_steps))

        hands_mid = self.robot.data.body_pos_w[:, self.hand_body_ids].mean(dim=1)
        self.metrics["hand_ball_distance"] = torch.norm(ball_pos - hands_mid, dim=-1)
        self.metrics["ball_in_catch_zone"] = in_zone.float()
        self.metrics["ball_arm_contact"] = self.arm_contact.float().mean(dim=-1)

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

    # ── throw generation: underhand lob (interim report ch. 7.6.2 / 9.4), heading frame of the robot ──
    throw_delay_range: tuple[float, float] = (0.5, 1.5)
    """Time (s) after reset before the ball is launched. The ball is parked at the release point in the
    meantime so the robot can settle and see the throw origin."""

    throw_distance_range: tuple[float, float] = (2.0, 5.0)
    """Horizontal distance (m) of the release point from the robot root (mjlab: 2 m -> 5 m)."""

    throw_direction_range: tuple[float, float] = (-math.radians(75.0), math.radians(75.0))
    """Azimuth (rad) of the release point around the robot heading, positive = robot's left
    (mjlab: ±15° -> ±75°)."""

    release_height_range: tuple[float, float] = (1.1, 1.7)
    """Release height (m) above the ground (LocoMuJoCo)."""

    target_forward_range: tuple[float, float] = (0.3, 0.4)
    """Aim point distance (m) in front of the robot root (LocoMuJoCo: 0.35 m)."""

    target_lateral_range: tuple[float, float] = (-0.1, 0.1)
    """Aim point lateral offset (m). Curriculum variable."""

    target_height_range: tuple[float, float] = (0.8, 1.05)
    """Aim point height (m) above the ground, in front of the belly (LocoMuJoCo: 0.8–1.05 m)."""

    flight_time_range: tuple[float, float] = (0.8, 1.4)
    """Flight time (s) from release to the aim point (LocoMuJoCo: 0.8–1.4 s)."""

    velocity_noise: float = 0.1
    """Std (m/s) of Gaussian noise added per axis to the launch velocity."""

    # ── air drag (matters for a light 15 cm ball: ~0.4 m shorter over a 4 m throw) ──
    air_density: float = 1.2
    """Air density (kg/m³)."""

    drag_coefficient_range: tuple[float, float] = (0.4, 0.6)
    """Drag coefficient of the ball, sampled per throw (smooth sphere ≈ 0.47). (0, 0) disables drag."""

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

    contact_force_threshold: float = 0.1
    """Min normal force (N) between ball and an arm link that counts as contact (ball weight ≈ 1.1 N)."""

    contact_grace_steps: int = 5
    """A contact within the last N control steps still counts (rolling or bouncing ball)."""

    secure_steps: int = 50
    """Consecutive held steps for a confirmed catch (50 steps = 1.0 s at 50 Hz; 0.5 s counted balls
    that rolled off afterwards). Curriculum variable."""
