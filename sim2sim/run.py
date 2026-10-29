"""Run a ball-catching policy trained in Isaac Lab in MuJoCo (Sim2Sim).

Without ``--throws`` the robot only stands (step 2 of the plan in spec Section 11, item 9). With ``--throws N`` it
plays N episodes the way Phase 2 does in Isaac (step 4): reset to the start pose; a throw sampled in the robot's
heading frame (``ball.Throw``: release point, aim point, apex, drag; parameters of ``commands.ball_throw`` in the run's
``params/env.yaml``); the ball parked at the release point until the wait is over, then launched with the
drag-compensated velocity, air drag as an external force every control step; the actor sees the ball through the
VICON model (``ball.Perception``); the catch is checked as in Isaac (``ball.CatchState``: both arms or, with
``palm_hold``, both palms from below, catch box, slow, torso force at most 50 N, 1 s secured, then the hold phase); an
episode ends as with the Phase 2 terminations (catch completed, ball below 10 cm, ball 6 m away, pelvis below 0.2 m or
tilted more than 0.8 rad, 20 s).

- Robot: ``model.py`` (``g1.xml`` with the collision shapes of the Isaac asset from the dump), the armature of the
  Isaac asset per joint, no passive damping or friction, position actuators with the Isaac drive gains and effort
  limits (an implicit PD like PhysX's drives: MuJoCo's implicitfast integrator), 5 ms physics step, floor friction 1.
- Ball: sphere with radius, mass and friction from ``env.yaml``. MuJoCo has no restitution coefficient: the bounce
  comes from the contact damping ratio (``--ball_dampratio``, ``--drop_test`` measures it), and it acts at every speed,
  while PhysX applies restitution only above its bounce threshold, so that a ball squeezed slowly between forearms and
  belly stays put. Matching the free bounce of Isaac (0.4: e = 0.36) shoots a wedged ball out; the default 0.7
  (e = 0.14) keeps it, as in Isaac (2026-10-29: 0 % against 23 % catches).
- Control: every 20 ms the actor observation is built with the term order, scales and history of the run's
  ``params/deploy.yaml``, the ONNX policy is evaluated, and the targets ``default + scale * action``, clamped to the
  hard joint limits, go to the actuators. Default pose, gains, limits and action scale (0.25 * effort / stiffness)
  come exactly from the dump; ``deploy.yaml`` rounds them to three digits.

    python sim2sim/run.py --onnx policy.onnx --run <run dir> --isaac isaac_g1.json [--throws 50] [--video out.mp4]
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import deque

import numpy as np
import yaml

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--onnx", required=True, help="Policy exported with tools/export_onnx.py.")
parser.add_argument("--run", required=True, help="Run directory (params/deploy.yaml and params/env.yaml).")
parser.add_argument("--isaac", required=True, help="JSON of tools/dump_isaac_model.py.")
parser.add_argument("--mjcf_dir", default=os.path.expanduser("~/unitree_mjcf/g1"))
parser.add_argument("--seconds", type=float, default=30.0, help="Length of the standing test (without --throws).")
parser.add_argument("--throws", type=int, default=0, help="Number of throw episodes.")
parser.add_argument("--wait", type=str, default="0.5,8", help="Wait before the throw, min,max s (Phase 2: 0.5,8).")
parser.add_argument("--uniform_wait", action="store_true", help="Uniform instead of log-uniform wait.")
parser.add_argument("--ball_dampratio", type=float, default=0.7, help="Ball contact damping ratio (0.7: e = 0.14).")
parser.add_argument("--start_clip", default=None, help="Start in frame 0 of this clip instead of the default pose.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--video", default=None, help="Write an mp4 (offscreen rendering, EGL without a display).")
parser.add_argument("--video_throws", type=int, default=4, help="Episodes in the video.")
parser.add_argument("--drop_test", action="store_true", help="Only drop the ball on the floor and report its bounce.")
parser.add_argument("--debug_ball", action="store_true", help="Print every step of the first throw's flight.")
parser.add_argument("--easy", action="store_true", help="Throws and perception of the curriculum's easy stage.")
parser.add_argument("--ghost_ball", action="store_true", help="Debug: the ball collides with nothing (only seen).")
args = parser.parse_args()
if args.video and "DISPLAY" not in os.environ:
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import onnxruntime as ort  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ball import CatchState, Perception, Throw, yaw_of  # noqa: E402  # isort: skip
from model import isaac_collision_spec, load_isaac_dump  # noqa: E402  # isort: skip

PHYSICS_DT, DECIMATION = 0.005, 4
STEP_DT = PHYSICS_DT * DECIMATION
EPISODE_S = 20.0
BALL_TIMECONST = 0.01  # s, contact stiffness of the ball (at least 2 physics steps)
# camera of the Isaac videos (eval_catch.py --eye/--lookat, relative to the root)
CAM_EYE, CAM_LOOKAT = np.array([2.1, -1.7, 0.4]), np.array([0.1, 0.0, -0.08])
ARM_LINKS = ("elbow_link", "wrist_roll_link", "wrist_pitch_link", "wrist_yaw_link")
OUTCOMES = ("caught", "dropped", "missed", "fell", "time_out")


class _Loader(yaml.SafeLoader):
    """env.yaml holds python tags (tuples, slices); read them as plain YAML."""


def _python_tag(loader, suffix, node):
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    return loader.construct_scalar(node)


_Loader.add_multi_constructor("tag:yaml.org,2002:python/", _python_tag)


class Policy:
    def __init__(self, path: str):
        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        check = path[: -len(".onnx")] + "_check.npz"
        if os.path.exists(check):  # same outputs as the torch actor?
            c = np.load(check)
            out = np.concatenate([self(x) for x in c["obs"]]).reshape(c["actions"].shape)
            err = float(np.abs(out - c["actions"]).max())
            print(f"[sim2sim] {os.path.basename(path)}: onnxruntime vs torch max |diff| {err:.1e}")
            assert err < 1e-4, "ONNX export does not match the checkpoint"

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self.session.run(None, {"obs": obs[None].astype(np.float32)})[0][0]


def build(isaac: dict, ball: dict) -> mujoco.MjModel:
    spec = isaac_collision_spec(args.mjcf_dir, isaac, floor=True)
    spec.option.timestep = PHYSICS_DT
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 720
    for name, j in isaac["joints"].items():
        joint = spec.joint(name)
        joint.armature, joint.frictionloss = j["armature"], 0.0
        joint.damping = np.zeros(3)  # MuJoCo 3.10: damping coefficients as a 3-vector
        act = spec.add_actuator(name=name, target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.gaintype, act.biastype = mujoco.mjtGain.mjGAIN_FIXED, mujoco.mjtBias.mjBIAS_AFFINE
        act.gainprm = [j["stiffness"]] + [0.0] * 9
        act.biasprm = [0.0, -j["stiffness"], -j["damping"]] + [0.0] * 7
        act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.forcerange = [-j["effort_limit"], j["effort_limit"]]
    body = spec.worldbody.add_body(name="ball", pos=[0.0, 0.0, 5.0])
    body.add_freejoint(name="ball")
    geom = body.add_geom(name="ball", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[ball["radius"], 0, 0])
    geom.mass, geom.rgba, geom.priority = ball["mass"], [1.0, 0.3, 0.0, 1.0], 2  # its contact parameters win
    geom.friction = [ball["friction"], 0.005, 0.0001]
    geom.solref = [BALL_TIMECONST, args.ball_dampratio]
    if args.ghost_ball:
        geom.contype = geom.conaffinity = 0
    tex = spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D)
    tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    tex.rgb1, tex.rgb2, tex.width, tex.height = [0.22, 0.24, 0.26], [0.30, 0.32, 0.34], 512, 512
    mat = spec.add_material(name="grid", texrepeat=[2, 2], reflectance=0.1)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    spec.geom("floor").material = "grid"
    spec.worldbody.add_light(pos=[0, 0, 4], dir=[0, 0, -1], diffuse=[0.5, 0.5, 0.5])
    sky = spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX)
    sky.builtin, sky.width, sky.height = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, 512, 3072
    sky.rgb1, sky.rgb2 = [0.62, 0.63, 0.66], [0.38, 0.39, 0.42]
    spec.visual.headlight.ambient, spec.visual.headlight.diffuse = [0.35, 0.35, 0.35], [0.55, 0.55, 0.55]
    return spec.compile()


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, q)
    return mat.reshape(3, 3)


class Sim:
    """The MuJoCo scene with robot, ball, policy and observation history."""

    def __init__(self):
        self.isaac = load_isaac_dump(args.isaac)
        with open(os.path.join(args.run, "params", "deploy.yaml")) as f:
            self.deploy = yaml.safe_load(f)
        with open(os.path.join(args.run, "params", "env.yaml")) as f:
            env_cfg = yaml.load(f, Loader=_Loader)
        self.cmd_cfg = env_cfg["commands"]["ball_throw"]
        if args.easy:
            self.cmd_cfg.update(EASY)
        spawn = env_cfg["scene"]["ball"]["spawn"]
        self.ball = {
            "radius": float(spawn["radius"]),
            "mass": float(spawn["mass_props"]["mass"]),
            "friction": float(spawn["physics_material"]["static_friction"]),
        }
        joints = self.joint_names = list(self.isaac["joints"])  # Isaac (policy) order
        self.default = np.array([self.isaac["joints"][j]["default"] for j in joints])
        stiffness = np.array([self.isaac["joints"][j]["stiffness"] for j in joints])
        self.effort = np.array([self.isaac["joints"][j]["effort_limit"] for j in joints])
        self.lower, self.upper = np.array([self.isaac["joints"][j]["range"] for j in joints]).T
        self.scale = 0.25 * self.effort / stiffness  # UNITREE_G1_29DOF_MIMIC_ACTION_SCALE
        scale_diff = np.max(np.abs(np.array(self.deploy["actions"]["JointPositionAction"]["scale"]) / self.scale - 1))
        pose_diff = np.max(np.abs(np.array(self.deploy["default_joint_pos"]) - self.default))
        print(f"[sim2sim] deploy.yaml vs dump (max diff): action scale {scale_diff:.1%}, default pose {pose_diff:.4f}")

        m = self.m = build(self.isaac, self.ball)
        self.d = mujoco.MjData(m)
        jid = lambda name: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)  # noqa: E731
        bid = lambda name: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)  # noqa: E731
        self.qadr = np.array([m.jnt_qposadr[jid(j)] for j in joints])
        self.dofadr = np.array([m.jnt_dofadr[jid(j)] for j in joints])
        self.acts = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in joints])
        self.floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.ball_geom, self.ball_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ball"), bid("ball")
        self.ball_q, self.ball_v = m.jnt_qposadr[jid("ball")], m.jnt_dofadr[jid("ball")]
        self.feet = [bid(f"{s}_ankle_roll_link") for s in ("left", "right")]
        self.torso = bid("torso_link")
        self.arm_side = {bid(f"{s}_{link}"): i for i, s in enumerate(("left", "right")) for link in ARM_LINKS}
        self.hands = [bid(f"{s}_wrist_yaw_link") for s in ("left", "right")]  # rubber hands (palm hold)
        part_of = lambda names: np.array([i for i, j in enumerate(joints) if any(k in j for k in names)])  # noqa: E731
        self.arm_joints, self.leg_joints = part_of(("shoulder", "elbow", "wrist")), part_of(("hip", "knee", "ankle"))
        self.robot_geoms = [g for g in range(m.ngeom) if 0 < m.geom_bodyid[g] != self.ball_body and m.geom_contype[g]]
        self.history = {name: deque(maxlen=c["history_length"]) for name, c in self.deploy["observations"].items()}
        self.obs_scale = {name: np.array(c["scale"]) for name, c in self.deploy["observations"].items()}
        self.clip = np.load(args.start_clip) if args.start_clip else None
        self.policy = None
        self.last_action = np.zeros(len(joints))
        self.renderer = self.writer = self.cam = None

    # ── state ──
    @property
    def ball_pos(self) -> np.ndarray:
        return self.d.qpos[self.ball_q : self.ball_q + 3]

    @property
    def ball_vel(self) -> np.ndarray:
        return self.d.qvel[self.ball_v : self.ball_v + 3]

    def set_ball(self, pos, vel=(0.0, 0.0, 0.0)):
        self.d.qpos[self.ball_q : self.ball_q + 7] = [*pos, 1, 0, 0, 0]
        self.d.qvel[self.ball_v : self.ball_v + 6] = [*vel, 0, 0, 0]

    def pelvis_tilt(self) -> float:
        return float(np.arccos(np.clip(quat_to_mat(self.d.qpos[3:7])[2, 2], -1, 1)))

    def reset_robot(self):
        """Start pose (default or frame 0 of the clip, root yaw removed), feet on the floor, at the origin."""
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        d.qpos[3:7] = [1, 0, 0, 0]
        d.qpos[self.qadr] = self.default
        if self.clip is not None:
            d.qpos[self.qadr] = self.clip["joint_pos"][0]
            w, x, y, z = self.clip["body_quat_w"][0, 0]
            yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            unyaw = np.array([np.cos(yaw / 2), 0, 0, -np.sin(yaw / 2)])
            mujoco.mju_mulQuat(d.qpos[3:7], unyaw, np.array([w, x, y, z]))
        self.set_ball((0.0, 0.0, 10.0))
        d.qpos[2] = 1.0
        mujoco.mj_kinematics(m, d)
        fromto = np.zeros(6)
        lowest = min(mujoco.mj_geomDistance(m, d, self.floor, g, 2.0, fromto) for g in self.robot_geoms)
        d.qpos[2] -= lowest - 0.001
        mujoco.mj_forward(m, d)
        for buf in self.history.values():
            buf.clear()
        self.last_action[:] = 0.0

    def observation(self, ball_terms: dict) -> np.ndarray:
        d = self.d
        terms = {
            "base_ang_vel": d.qvel[3:6],  # free joint: angular velocity in the body frame
            "projected_gravity": quat_to_mat(d.qpos[3:7]).T @ np.array([0.0, 0.0, -1.0]),
            "joint_pos_rel": d.qpos[self.qadr] - self.default,
            "joint_vel_rel": d.qvel[self.dofadr],
            "last_action": self.last_action,
            **ball_terms,
        }
        parts = []
        for name, buf in self.history.items():
            value = terms[name] * self.obs_scale[name]
            if not buf:  # Isaac fills the whole history with the first observation after a reset
                buf.extend([value] * buf.maxlen)
            buf.append(value)
            parts.append(np.concatenate(buf))  # oldest first, per term
        return np.concatenate(parts)

    def ball_contacts(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Force (N) between the ball and each arm (strongest link), each hand and the torso, as the ball_contact
        sensor."""
        m, d = self.m, self.d
        per_body: dict[int, np.ndarray] = {}
        pairs = np.stack([d.contact.geom1, d.contact.geom2], axis=1)[: d.ncon]
        for i in np.nonzero((pairs == self.ball_geom).any(axis=1))[0]:
            body = m.geom_bodyid[pairs[i, 1] if pairs[i, 0] == self.ball_geom else pairs[i, 0]]
            if body in self.arm_side or body == self.torso:
                f = np.zeros(6)
                mujoco.mj_contactForce(m, d, i, f)
                per_body[body] = per_body.get(body, 0.0) + d.contact.frame[i].reshape(3, 3).T @ f[:3]
        arm, hand, torso_force = np.zeros(2), np.zeros(2), 0.0
        for body, force in per_body.items():
            if body == self.torso:
                torso_force = float(np.linalg.norm(force))
            else:
                side = self.arm_side[body]
                arm[side] = max(arm[side], float(np.linalg.norm(force)))
                if body == self.hands[side]:
                    hand[side] = float(np.linalg.norm(force))
        return arm, hand, torso_force

    def hand_poses(self) -> tuple[np.ndarray, np.ndarray]:
        """Position (2, 3) and rotation matrix (2, 3, 3) of the [left, right] hand body."""
        return self.d.xpos[self.hands].copy(), self.d.xmat[self.hands].reshape(2, 3, 3).copy()

    def feet_on_floor(self) -> np.ndarray:
        pairs = np.stack([self.d.contact.geom1, self.d.contact.geom2], axis=1)[: self.d.ncon]
        on_floor = (pairs == self.floor).any(axis=1)
        return np.array([np.any(on_floor & (self.m.geom_bodyid[pairs] == foot).any(axis=1)) for foot in self.feet])

    def control_step(self, obs: np.ndarray):
        """Policy, clamped targets, 4 physics steps. Returns ball forces (strongest over the steps) and counts."""
        action = self.policy(obs)
        target = self.default + self.scale * action
        clamped = int(np.sum((target < self.lower) | (target > self.upper)))
        self.d.ctrl[self.acts] = np.clip(target, self.lower, self.upper)
        arm_force, hand_force, torso_force = np.zeros(2), np.zeros(2), 0.0
        for _ in range(DECIMATION):
            mujoco.mj_step(self.m, self.d)
            arm, hand, tf = self.ball_contacts()
            arm_force, hand_force = np.maximum(arm_force, arm), np.maximum(hand_force, hand)
            torso_force = max(torso_force, tf)
        saturated = int(np.sum(np.abs(self.d.actuator_force[self.acts]) >= 0.99 * self.effort))
        change = float(np.abs(action - self.last_action).mean())
        self.last_action = action
        return arm_force, hand_force, torso_force, clamped, saturated, change

    # ── video ──
    def open_video(self, path: str):
        import imageio

        self.renderer = mujoco.Renderer(self.m, height=720, width=1280)
        self.writer = imageio.get_writer(path, fps=25, codec="libx264", quality=8)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        view = CAM_LOOKAT - CAM_EYE
        self.cam.distance = float(np.linalg.norm(view))
        self.cam.azimuth = float(np.degrees(np.arctan2(view[1], view[0])))
        self.cam.elevation = float(np.degrees(np.arcsin(view[2] / self.cam.distance)))

    def frame(self, k: int):
        if self.writer is not None and k % 2 == 0:
            self.cam.lookat[:] = self.d.qpos[:3] + CAM_LOOKAT
            self.renderer.update_scene(self.d, camera=self.cam)
            self.writer.append_data(self.renderer.render())


EASY = {  # easy stage of the ball_throws curriculum (Phase 2 and Unified)
    "throw_distance_range": (2.0, 2.5), "throw_direction_range": (-0.2618, 0.2618),
    "target_lateral_range": (-0.03, 0.03),
    "target_height_range": (0.85, 0.95), "apex_height_range": (1.85, 1.95), "velocity_noise": 0.0, "secure_steps": 25,
    "obs_delay_steps_range": (1, 1), "obs_dropout_prob": 0.0, "hide_in_catch_zone_prob": 0.0,
}
NO_BALL = {"ball_state": np.zeros(6), "ball_visible": np.zeros(1), "ball_intercept": np.zeros(3)}


def drop_test(sim: Sim):
    """Bounce of the ball on the floor: e = sqrt(rebound height / drop height)."""
    m, d = sim.m, sim.d
    mujoco.mj_resetData(m, d)
    sim.set_ball((3.0, 0.0, 1.0 + sim.ball["radius"]))  # 3 m beside the robot
    apex, rising = 0.0, False
    for _ in range(2000):
        mujoco.mj_step(m, d)
        z, vz = sim.ball_pos[2] - sim.ball["radius"], sim.ball_vel[2]
        rising = rising or vz > 0
        if rising:
            apex = max(apex, z)
            if vz < 0:
                break
    print(f"[sim2sim] drop test: dampratio {args.ball_dampratio}, rebound {apex:.3f} m from 1 m, "
          f"e = {np.sqrt(apex):.2f} (Isaac: ball 0.4 averaged with the robot's 0-0.5)")


def stand(sim: Sim):
    """Standing test: no ball, ``--seconds`` long."""
    sim.reset_robot()
    d = sim.d
    start_xy, steps = d.qpos[:2].copy(), int(args.seconds / STEP_DT)
    liftoffs, in_contact = np.zeros(2, dtype=int), np.ones(2, dtype=bool)
    motion, fallen_at, clamped, saturated = [], None, 0, 0
    obs = sim.observation(NO_BALL)
    for k in range(steps):
        _, _, _, c, s, change = sim.control_step(obs)
        qd = d.qvel[sim.dofadr]
        motion.append((np.sqrt(np.mean(qd[sim.arm_joints] ** 2)), np.sqrt(np.mean(qd[sim.leg_joints] ** 2)), change,
                       np.linalg.norm(d.qvel[:2])))
        clamped, saturated = clamped + c, saturated + s
        touching = sim.feet_on_floor()
        liftoffs += in_contact & ~touching
        in_contact = touching
        if fallen_at is None and (d.qpos[2] < 0.45 or sim.pelvis_tilt() > 1.0):
            fallen_at = (k + 1) * STEP_DT
        sim.frame(k)
        obs = sim.observation(NO_BALL)
    arm_v, leg_v, da, speed = np.mean(motion[25:], axis=0)
    status = f"FELL at {fallen_at:.1f} s" if fallen_at is not None else "stood"
    print(f"[sim2sim] {status}, {args.seconds:.0f} s: pelvis z {d.qpos[2]:.3f} m, drift "
          f"{np.linalg.norm(d.qpos[:2] - start_xy) * 100:.1f} cm, lift-offs L/R {liftoffs[0]}/{liftoffs[1]}; arm joint "
          f"vel RMS {arm_v:.3f} rad/s, leg {leg_v:.3f}, action change {da:.3f}, root speed {speed:.3f} m/s; clamped "
          f"targets {clamped}, saturated torques {saturated}")


def throws(sim: Sim):
    """``--throws`` episodes with the Phase 2 lifecycle (see the module docstring)."""
    d, rng = sim.d, np.random.default_rng(args.seed)
    perception, catch = Perception(sim.cmd_cfg, STEP_DT, rng), CatchState(sim.cmd_cfg, STEP_DT)
    wait = tuple(float(x) for x in args.wait.split(","))
    area = np.pi * sim.ball["radius"] ** 2
    outcomes, confirm_delays, held_torso = [], [], []
    for episode in range(args.throws):
        sim.reset_robot()
        throw = Throw(sim.cmd_cfg, d.qpos[:3].copy(), yaw_of(d.qpos[3:7]), area, rng, wait, not args.uniform_wait)
        perception.reset()
        catch.reset()
        sim.set_ball(throw.release)
        mujoco.mj_forward(sim.m, d)
        obs = sim.observation(NO_BALL)
        time_until_throw, time_since_throw, outcome, confirm_t = throw.delay, 0.0, "time_out", None
        arrival_miss = None
        recording = sim.writer if episode < args.video_throws else None
        for k in range(int(EPISODE_S / STEP_DT)):
            arm_force, hand_force, torso_force, _, _, _ = sim.control_step(obs)
            if recording is not None:
                sim.frame(k)
            # terminations: state after the physics steps, catch state of the previous update
            if catch.catch_complete:
                outcome = "caught"
            elif sim.ball_pos[2] < 0.1:
                outcome = "dropped"
            elif np.linalg.norm(sim.ball_pos[:2] - d.qpos[:2]) > 6.0:
                outcome = "missed"
            elif d.qpos[2] < 0.2 or sim.pelvis_tilt() > 0.8:
                outcome = "fell"
            if outcome != "time_out":
                break
            # command update: launch timer, park or launch, drag, catch state, perception
            pre_throw = time_until_throw > 0.0
            time_until_throw -= STEP_DT
            if pre_throw and time_until_throw <= 0.0:
                sim.set_ball(throw.release, throw.launch_velocity(sim.ball["mass"]))
                time_since_throw = 0.0
            elif time_until_throw > 0.0:
                sim.set_ball(throw.release)
            in_flight = time_until_throw <= 0.0
            time_since_throw += STEP_DT if in_flight else 0.0
            if in_flight and arrival_miss is None and time_since_throw >= throw.flight_time:
                arrival_miss = np.linalg.norm(sim.ball_pos - throw.target)  # touched or not: a check of the flight
            ball_pos, ball_vel = sim.ball_pos.copy(), sim.ball_vel.copy()
            drag = -throw.drag_coeff * np.linalg.norm(ball_vel) * ball_vel if in_flight else np.zeros(3)
            d.xfrc_applied[sim.ball_body, :3] = drag
            was_confirmed = catch.confirmed
            hand_pos, hand_rot = sim.hand_poses()
            catch.update(
                in_flight, ball_pos, ball_vel, d.qpos[:3], d.qpos[3:7], d.qvel[:3], arm_force, torso_force,
                hand_force, hand_pos, hand_rot,
            )
            if catch.confirmed and not was_confirmed:
                confirm_t = time_since_throw
            if catch.secured_steps > 0:
                held_torso.append(torso_force)
            perception.update(in_flight, time_since_throw, ball_pos, ball_vel, d.qpos[:3], d.qpos[3:7], catch.in_zone)
            obs = sim.observation(perception.actor_terms())
            if args.debug_ball and episode == 0 and in_flight:
                rel, tilt = np.round(ball_pos - d.qpos[:3], 2), np.degrees(sim.pelvis_tilt())
                print(f"  t {time_since_throw:4.2f} s: ball {rel} from the pelvis (world), seen {perception.visible}, "
                      f"force arms {np.round(arm_force)} N torso {torso_force:.0f} N, pelvis tilt {tilt:4.1f} deg, "
                      f"max |action| {np.abs(sim.last_action).max():.1f}")
        d.xfrc_applied[sim.ball_body] = 0.0
        outcomes.append(outcome)
        if confirm_t is not None:
            confirm_delays.append(confirm_t)
        print(f"[sim2sim] throw {episode + 1:3d}: wait {throw.delay:4.1f} s, release "
              f"{np.linalg.norm(throw.release[:2] - d.qpos[:2]):.1f} m, aim y {throw.target[1]:+.2f} m z "
              f"{throw.target[2]:.2f} m, ball {100 * (arrival_miss or 0):.0f} cm from the aim at the arrival time"
              f" -> {outcome}", flush=True)
    n = len(outcomes)
    summary = ", ".join(f"{o} {100.0 * outcomes.count(o) / n:.0f} %" for o in OUTCOMES)
    delay = np.median(confirm_delays) if confirm_delays else float("nan")
    force = np.mean(held_torso) if held_torso else 0.0
    print(f"[sim2sim] {summary} of {n} throws; confirmations {len(confirm_delays)}, {delay:.2f} s after the launch "
          f"(median); torso force while secured {force:.0f} N (mean)")


def main():
    sim = Sim()
    if args.drop_test:
        drop_test(sim)
        return
    sim.policy = Policy(args.onnx)
    if args.video:
        sim.open_video(args.video)
    if args.throws > 0:
        throws(sim)
    else:
        stand(sim)
    if sim.writer is not None:
        sim.writer.close()
        print(f"[sim2sim] video {args.video}")


if __name__ == "__main__":
    main()
