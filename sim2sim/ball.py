"""The ball as the actor perceives it, in numpy: port of the VICON model of ``BallCommand`` (Isaac).

Same logic as ``BallCommand._update_perception`` and ``observations.ball_perceived_*`` for one robot: every control
step the tracker measures the ball relative to the root in the yaw frame (position relative to the root, world
velocity rotated); the measurement is valid once the ball has flown for the detection delay, unless a dropout burst
or the occlusion in the catch box hides it; it reaches the robot ``obs_delay`` steps late with a per-throw bias and
noise; missing measurements are bridged ballistically for ``max_coast_steps``, then the ball is lost. The parameters
are those of ``commands.ball_throw`` in the run's ``params/env.yaml`` (the final values of the perception
curriculum). ``tools/check_ball_obs.py`` checks this port against observations recorded in Isaac.
"""

from __future__ import annotations

import numpy as np

GRAVITY = np.array([0.0, 0.0, -9.81])
NOT_SEEN = 1_000_000  # steps since the last measurement before the first one of a throw


def yaw_of(quat_wxyz: np.ndarray) -> float:
    w, x, y, z = quat_wxyz
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def to_yaw_frame(vec: np.ndarray, yaw: float) -> np.ndarray:
    """Rotate a world vector into the frame turned by ``yaw`` about z (Isaac: quat_apply_inverse(yaw_quat(q), v))."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([c * vec[0] + s * vec[1], -s * vec[0] + c * vec[1], vec[2]])


def ballistic_intercept(pos_b: np.ndarray, vel_b: np.ndarray, catch_height: float, max_time: float):
    """Descending crossing of a drag-free trajectory with the plane ``catch_height`` above the root.

    Same as ``observations._ballistic_intercept``: [x, y, time to go], time 0 if the plane is never reached.
    """
    disc = vel_b[2] ** 2 + 2.0 * 9.81 * (pos_b[2] - catch_height)
    t_go = np.clip((vel_b[2] + np.sqrt(max(disc, 0.0))) / 9.81, 0.0, max_time) if disc > 0.0 else 0.0
    return np.array([pos_b[0] + vel_b[0] * t_go, pos_b[1] + vel_b[1] * t_go, t_go])


class Perception:
    """Perceived ball of one robot; ``reset()`` at every new throw, ``update()`` every control step."""

    def __init__(self, cfg: dict, step_dt: float, rng: np.random.Generator, catch_height: float = 0.15):
        self.cfg, self.step_dt, self.rng, self.catch_height = cfg, step_dt, rng, catch_height
        self.zone_lo = np.array([cfg["catch_zone_x"][0], cfg["catch_zone_y"][0], cfg["catch_zone_z"][0]])
        self.zone_hi = np.array([cfg["catch_zone_x"][1], cfg["catch_zone_y"][1], cfg["catch_zone_z"][1]])
        n = int(cfg["obs_delay_steps_range"][1]) + 1  # ring buffer sized for the largest delay
        self.meas_pos, self.meas_vel, self.meas_valid = np.zeros((n, 3)), np.zeros((n, 3)), np.zeros(n, dtype=bool)
        self.index = 0
        self.last_pos, self.last_vel = np.zeros(3), np.zeros(3)
        self.reset()

    def reset(self):
        """New throw: forget the old one and sample this throw's tracker imperfections."""
        cfg = self.cfg
        self.meas_valid[:] = False
        self.steps_since_measurement = NOT_SEEN
        self.dropout_left = 0
        self.pos_b, self.vel_b, self.visible = np.zeros(3), np.zeros(3), False
        lo, hi = cfg["obs_delay_steps_range"]
        self.delay = min(int(self.rng.integers(lo, hi + 1)), len(self.meas_valid) - 1)
        self.detection_delay = float(self.rng.uniform(*cfg["detection_delay_range"]))
        self.bias = self.rng.uniform(-cfg["obs_pos_bias"], cfg["obs_pos_bias"], 3)
        self.hide_in_zone = bool(self.rng.random() < cfg["hide_in_catch_zone_prob"])

    def in_zone(self, ball_pos_w, root_pos_w, root_quat_w) -> bool:
        """Ball inside the catch box (root yaw frame), as ``BallCommand._update_catch_state``."""
        ball_b = to_yaw_frame(np.asarray(ball_pos_w) - root_pos_w, yaw_of(root_quat_w))
        return bool(np.all((ball_b >= self.zone_lo) & (ball_b <= self.zone_hi)))

    def update(self, launched: bool, time_since_throw: float, ball_pos_w, ball_vel_w, root_pos_w, root_quat_w,
               in_zone: bool | None = None):
        cfg = self.cfg
        yaw = yaw_of(root_quat_w)
        pos_b = to_yaw_frame(np.asarray(ball_pos_w) - root_pos_w, yaw)
        vel_b = to_yaw_frame(np.asarray(ball_vel_w), yaw)
        if in_zone is None:
            in_zone = self.in_zone(ball_pos_w, root_pos_w, root_quat_w)

        # what the tracker measures now; dropouts come in bursts of 1..obs_dropout_max_steps steps
        detected = launched and time_since_throw >= self.detection_delay
        burst_start = self.dropout_left == 0 and self.rng.random() < cfg["obs_dropout_prob"]
        burst_length = int(self.rng.integers(1, cfg["obs_dropout_max_steps"] + 1))
        if burst_start:
            self.dropout_left = burst_length
        occluded = self.dropout_left > 0 or (self.hide_in_zone and in_zone)
        self.dropout_left = max(self.dropout_left - 1, 0)
        self.index = (self.index + 1) % len(self.meas_valid)
        self.meas_pos[self.index], self.meas_vel[self.index] = pos_b, vel_b
        self.meas_valid[self.index] = detected and not occluded

        # what reaches the robot: the measurement of ``delay`` steps ago, noisy and biased
        read = (self.index - self.delay) % len(self.meas_valid)
        valid = bool(self.meas_valid[read])
        meas_pos = self.meas_pos[read] + self.bias + self.rng.normal(0.0, 1.0, 3) * cfg["obs_pos_noise_std"]
        meas_vel = self.meas_vel[read] + self.rng.normal(0.0, 1.0, 3) * cfg["obs_vel_noise_std"]

        # missing measurement: coast on the last estimate with the ballistic model
        self.steps_since_measurement = 0 if valid else self.steps_since_measurement + 1
        coast_t = self.steps_since_measurement * self.step_dt
        coast_pos = self.last_pos + self.last_vel * coast_t + 0.5 * GRAVITY * coast_t**2
        coast_vel = self.last_vel + GRAVITY * coast_t
        if valid:
            self.last_pos, self.last_vel = meas_pos, meas_vel

        self.visible = self.steps_since_measurement <= cfg["max_coast_steps"]
        self.pos_b = (meas_pos if valid else coast_pos) * self.visible
        self.vel_b = (meas_vel if valid else coast_vel) * self.visible

    def actor_terms(self) -> dict[str, np.ndarray]:
        """``ball_state``, ``ball_visible``, ``ball_intercept`` of the actor observation (zeros while not visible)."""
        intercept = ballistic_intercept(self.pos_b, self.vel_b, self.catch_height, 3.0) * float(self.visible)
        return {
            "ball_state": np.concatenate([self.pos_b, self.vel_b]),
            "ball_visible": np.array([float(self.visible)]),
            "ball_intercept": intercept,
        }


# ── throws (port of BallCommand._sample_throw, _solve_launch_velocity, _drag_displacement) ─────────────────────────


def flight_time_from_apex(release_z: float, target_z: float, apex_z: float) -> float:
    return float(np.sqrt(2.0 * (apex_z - release_z) / 9.81) + np.sqrt(2.0 * (apex_z - target_z) / 9.81))


def drag_displacement(throw_vel: np.ndarray, flight_time: float, drag_per_mass: float, nodes: int) -> np.ndarray:
    """Displacement after ``flight_time`` caused by the drag acceleration -(k/m)|v|v.

    Trapezoid rule on ``nodes`` intervals, velocity refined by two Picard passes, as in Isaac.
    """
    times = flight_time * np.linspace(0.0, 1.0, nodes + 1)
    dt = flight_time / nodes
    vel_free = throw_vel + GRAVITY * times[:, None]
    vel = vel_free
    for _ in range(2):
        acc = -drag_per_mass * np.linalg.norm(vel, axis=1, keepdims=True) * vel
        dv = np.cumsum(0.5 * (acc[1:] + acc[:-1]) * dt, axis=0)
        vel = vel_free + np.vstack([np.zeros(3), dv])
    acc = -drag_per_mass * np.linalg.norm(vel, axis=1, keepdims=True) * vel
    integrand = (flight_time - times)[:, None] * acc
    return (0.5 * (integrand[1:] + integrand[:-1]) * dt).sum(axis=0)


class Throw:
    """One throw of ``BallCommand``, sampled in the robot's heading frame at the reset and frozen."""

    def __init__(self, cfg: dict, root_pos: np.ndarray, root_yaw: float, ball_area: float, rng: np.random.Generator,
                 wait: tuple[float, float], log_wait: bool):
        u = lambda r: float(rng.uniform(*r))  # noqa: E731
        c, s = np.cos(root_yaw), np.sin(root_yaw)
        heading = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        distance, azimuth = u(cfg["throw_distance_range"]), u(cfg["throw_direction_range"])
        self.release = root_pos + heading @ np.array([distance * np.cos(azimuth), distance * np.sin(azimuth), 0.0])
        self.release[2] = u(cfg["release_height_range"])
        aim = np.array([u(cfg["target_forward_range"]), u(cfg["target_lateral_range"]), 0.0])
        self.target = root_pos + heading @ aim
        self.target[2] = u(cfg["target_height_range"])
        apex = max(u(cfg["apex_height_range"]), max(self.release[2], self.target[2]) + 0.05)
        self.flight_time = flight_time_from_apex(self.release[2], self.target[2], apex)
        self.drag_coeff = 0.5 * cfg["air_density"] * u(cfg["drag_coefficient_range"]) * ball_area  # F = -k |v| v
        self.delay = wait[0] * (wait[1] / wait[0]) ** rng.random() if log_wait else u(wait)
        self.velocity_noise = rng.normal(0.0, 1.0, 3) * cfg["velocity_noise"]
        self.iterations, self.nodes = int(cfg["drag_solver_iterations"]), int(cfg["drag_solver_nodes"])

    def launch_velocity(self, ball_mass: float) -> np.ndarray:
        """Velocity that reaches the target after the flight time with gravity and drag, plus the velocity noise."""
        t, c = self.flight_time, self.drag_coeff / ball_mass
        ballistic = self.target - self.release - 0.5 * GRAVITY * t**2
        vel = ballistic / t
        for _ in range(self.iterations):
            vel = (ballistic - drag_displacement(vel, t, c, self.nodes)) / t
        return vel + self.velocity_noise


# ── catch check (port of BallCommand._update_catch_state and the ball terminations) ────────────────────────────────


def palm_geometry(ball_pos, hand_pos, hand_rot, cfg: dict) -> tuple[np.ndarray, float]:
    """Per hand whether the ball centre lies on its palm side, and its height above the mean palm height
    (``BallCommand._palm_geometry``; palm points and normals of ``cfg`` in the hand body frames)."""
    points, normals = np.asarray(cfg["palm_points"], dtype=float), np.asarray(cfg["palm_normals"], dtype=float)
    palm = np.asarray(hand_pos) + np.einsum("hij,hj->hi", hand_rot, points)
    normal = np.einsum("hij,hj->hi", hand_rot, normals)
    to_ball = np.asarray(ball_pos) - palm
    distance = np.linalg.norm(to_ball, axis=1)
    alignment = np.sum(normal * to_ball, axis=1) / np.maximum(distance, 1e-6)
    on_palm = (alignment >= cfg["min_palm_alignment"]) & (distance <= cfg["max_palm_distance"])
    return on_palm, float(ball_pos[2] - palm[:, 2].mean())


class CatchState:
    """Secured steps, confirmation and hold phase of one throw, as ``BallCommand`` tracks them."""

    def __init__(self, cfg: dict, step_dt: float):
        self.cfg, self.step_dt = cfg, step_dt
        self.zone_lo = np.array([cfg["catch_zone_x"][0], cfg["catch_zone_y"][0], cfg["catch_zone_z"][0]])
        self.zone_hi = np.array([cfg["catch_zone_x"][1], cfg["catch_zone_y"][1], cfg["catch_zone_z"][1]])
        self.reset()

    def reset(self):
        self.secured_steps, self.steps_since_confirm, self.confirmed = 0, 0, False
        self.steps_since_contact = np.full(2, NOT_SEEN)
        self.in_zone = False

    @property
    def catch_complete(self) -> bool:
        held = self.steps_since_confirm * self.step_dt >= self.cfg["hold_after_catch_s"] - 1e-6
        return self.confirmed and held

    def update(
        self, in_flight: bool, ball_pos, ball_vel, root_pos, root_quat, root_lin_vel, arm_force, torso_force,
        hand_force=None, hand_pos=None, hand_rot=None,
    ):
        """``arm_force``: strongest ball force per arm (any arm link); ``hand_force``, ``hand_pos``, ``hand_rot``:
        ball force, position and rotation matrix of the [left, right] hand body, needed with ``palm_hold``."""
        cfg = self.cfg
        ball_b = to_yaw_frame(np.asarray(ball_pos) - root_pos, yaw_of(root_quat))
        self.in_zone = bool(np.all((ball_b >= self.zone_lo) & (ball_b <= self.zone_hi)))
        palm_hold = bool(cfg.get("palm_hold", False))  # runs before 2026-10-29 have no such key
        if palm_hold:
            on_palm, above_palms = palm_geometry(ball_pos, hand_pos, hand_rot, cfg)
            contact = (np.asarray(hand_force) > cfg["contact_force_threshold"]) & on_palm
        else:
            contact = np.asarray(arm_force) > cfg["contact_force_threshold"]
        self.steps_since_contact = np.where(contact, 0, self.steps_since_contact + 1)
        recent = self.steps_since_contact <= cfg["contact_grace_steps"]
        if palm_hold:
            touching = bool(recent.all()) and above_palms >= cfg["min_ball_above_palms"]
        else:
            touching = recent.all() if cfg["require_both_arms"] else recent.any()
        secured = (
            in_flight
            and self.in_zone
            and np.linalg.norm(np.asarray(ball_vel) - root_lin_vel) < cfg["secure_rel_vel"]
            and ball_pos[2] > cfg["secure_min_height"]
            and touching
        )
        if cfg.get("max_torso_force") is not None:
            secured = secured and torso_force <= cfg["max_torso_force"]
        self.secured_steps = self.secured_steps + 1 if secured else 0
        confirmed_now = not self.confirmed and self.secured_steps >= cfg["secure_steps"]
        self.steps_since_confirm += int(self.confirmed)
        self.confirmed = self.confirmed or confirmed_now
