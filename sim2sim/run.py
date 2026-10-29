"""Run a ball-catching policy trained in Isaac Lab in MuJoCo (Sim2Sim).

Robot and control loop, so far without a ball (step 2 of the plan in spec Section 11, item 9): the robot stands for
``--seconds`` from the default pose or from the first frame of a clip (as a Unified cycle starts its lead-in), and the
policy sees the ball as not visible.

- Model: ``model.py`` (``g1.xml`` with the collision shapes of the Isaac asset from the dump), the armature of the
  Isaac asset per joint, no passive damping or friction, position actuators with the Isaac drive gains and effort
  limits (an implicit PD like PhysX's drives: MuJoCo's implicitfast integrator), 5 ms physics step, floor friction 1.
- Control: every 20 ms the actor observation is built with the term order, scales and history of the run's
  ``params/deploy.yaml``, the ONNX policy is evaluated, and the targets ``default + scale * action``, clamped to the
  hard joint limits, go to the actuators. Default pose, gains, limits and action scale (0.25 * effort / stiffness) come
  exactly from the dump; ``deploy.yaml`` rounds them to three digits and, until 2026-10-29, exported env 0's randomized
  default pose, so it is only compared.
- Output: a summary every 5 s (pelvis height and drift, torso tilt, foot lift-offs, clamped targets, saturated
  torques), a final line and optionally a video.

    python sim2sim/run.py --onnx policy.onnx --deploy <run>/params/deploy.yaml --isaac isaac_g1.json \\
        --mjcf_dir ~/unitree_mjcf/g1 --seconds 30 [--start_clip clip.npz] [--video stand.mp4]
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
parser.add_argument("--deploy", required=True, help="params/deploy.yaml of the run.")
parser.add_argument("--isaac", required=True, help="JSON of tools/dump_isaac_model.py.")
parser.add_argument("--mjcf_dir", default=os.path.expanduser("~/unitree_mjcf/g1"))
parser.add_argument("--seconds", type=float, default=30.0)
parser.add_argument("--start_clip", default=None, help="Start in frame 0 of this clip instead of the default pose.")
parser.add_argument("--video", default=None, help="Write an mp4 (offscreen rendering, EGL without a display).")
args = parser.parse_args()
if args.video and "DISPLAY" not in os.environ:
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import onnxruntime as ort  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import isaac_collision_spec, load_isaac_dump  # noqa: E402  # isort: skip

PHYSICS_DT, DECIMATION = 0.005, 4
BALL_TERMS = {"ball_state": 6, "ball_visible": 1, "ball_intercept": 3}  # "not visible": zeros
# camera of the Isaac videos (eval_catch.py --eye/--lookat, relative to the root)
CAM_EYE, CAM_LOOKAT = np.array([2.1, -1.7, 0.4]), np.array([0.1, 0.0, -0.08])


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


def build(isaac: dict) -> mujoco.MjModel:
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


def main():
    isaac = load_isaac_dump(args.isaac)
    with open(args.deploy) as f:
        deploy = yaml.safe_load(f)
    joints = list(isaac["joints"])  # Isaac (policy) order
    default = np.array([isaac["joints"][j]["default"] for j in joints])
    stiffness = np.array([isaac["joints"][j]["stiffness"] for j in joints])
    effort = np.array([isaac["joints"][j]["effort_limit"] for j in joints])
    lower, upper = np.array([isaac["joints"][j]["range"] for j in joints]).T
    scale = 0.25 * effort / stiffness  # UNITREE_G1_29DOF_MIMIC_ACTION_SCALE
    action_cfg = deploy["actions"]["JointPositionAction"]
    print(
        "[sim2sim] deploy.yaml vs dump: action scale max rel. diff "
        f"{np.max(np.abs(np.array(action_cfg['scale']) - scale) / scale):.1%}, default pose max diff "
        f"{np.max(np.abs(np.array(deploy['default_joint_pos']) - default)):.4f} rad"
    )

    m = build(isaac)
    d = mujoco.MjData(m)
    qadr = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in joints])
    dofadr = np.array([m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in joints])
    acts = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in joints])
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    feet = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_ankle_roll_link") for s in ("left", "right")]
    torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    policy = Policy(args.onnx)

    # start pose: default pose or frame 0 of a clip (root roll and pitch kept, yaw removed), feet on the floor
    mujoco.mj_resetData(m, d)
    d.qpos[3:7] = [1, 0, 0, 0]
    d.qpos[qadr] = default
    if args.start_clip:
        clip = np.load(args.start_clip)
        d.qpos[qadr] = clip["joint_pos"][0]
        w, x, y, z = clip["body_quat_w"][0, 0]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        unyaw = np.array([np.cos(yaw / 2), 0, 0, -np.sin(yaw / 2)])
        mujoco.mju_mulQuat(d.qpos[3:7], unyaw, np.array([w, x, y, z]))
    d.qpos[2] = 1.0
    mujoco.mj_kinematics(m, d)
    fromto = np.zeros(6)
    robot_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] > 0 and m.geom_contype[g]]
    lowest = min(mujoco.mj_geomDistance(m, d, floor, g, 2.0, fromto) for g in robot_geoms)
    d.qpos[2] -= lowest - 0.001
    mujoco.mj_forward(m, d)
    start_xy, start_yaw = d.qpos[:2].copy(), None

    history = {name: deque(maxlen=cfg["history_length"]) for name, cfg in deploy["observations"].items()}
    obs_scale = {name: np.array(cfg["scale"]) for name, cfg in deploy["observations"].items()}
    last_action = np.zeros(len(joints))

    def observation() -> np.ndarray:
        rot = quat_to_mat(d.qpos[3:7])
        terms = {
            "base_ang_vel": d.qvel[3:6],  # free joint: angular velocity in the body frame
            "projected_gravity": rot.T @ np.array([0.0, 0.0, -1.0]),
            "joint_pos_rel": d.qpos[qadr] - default,
            "joint_vel_rel": d.qvel[dofadr],
            "last_action": last_action,
            **{name: np.zeros(n) for name, n in BALL_TERMS.items()},
        }
        parts = []
        for name, buf in history.items():
            value = terms[name] * obs_scale[name]
            if not buf:  # Isaac fills the whole history with the first observation after a reset
                buf.extend([value] * buf.maxlen)
            buf.append(value)
            parts.append(np.concatenate(buf))  # oldest first, per term
        return np.concatenate(parts)

    renderer, writer, cam = None, None, None
    if args.video:
        import imageio

        renderer = mujoco.Renderer(m, height=720, width=1280)
        writer = imageio.get_writer(args.video, fps=25, codec="libx264", quality=8)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        view = CAM_LOOKAT - CAM_EYE
        cam.distance = float(np.linalg.norm(view))
        cam.azimuth = float(np.degrees(np.arctan2(view[1], view[0])))
        cam.elevation = float(np.degrees(np.arcsin(view[2] / cam.distance)))

    steps = int(args.seconds / (PHYSICS_DT * DECIMATION))
    in_contact, liftoffs = np.ones(2, dtype=bool), np.zeros(2, dtype=int)
    clamped, saturated, fallen_at = 0, 0, None
    window = []
    for k in range(steps):
        action = policy(observation())
        target = default + scale * action
        clamped += int(np.sum((target < lower) | (target > upper)))
        d.ctrl[acts] = np.clip(target, lower, upper)
        last_action = action
        for _ in range(DECIMATION):
            mujoco.mj_step(m, d)
        saturated += int(np.sum(np.abs(d.actuator_force[acts]) >= 0.99 * effort))

        pairs = np.stack([d.contact.geom1, d.contact.geom2], axis=1)[: d.ncon]
        on_floor = (pairs == floor).any(axis=1)
        touching = np.array([np.any(on_floor & (m.geom_bodyid[pairs] == foot).any(axis=1)) for foot in feet])
        liftoffs += in_contact & ~touching
        in_contact = touching

        torso_up = d.xmat[torso].reshape(3, 3)[:, 2]
        tilt = float(np.degrees(np.arccos(np.clip(torso_up[2], -1, 1))))
        w, x, y, z = d.qpos[3:7]
        yaw = float(np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))))
        start_yaw = yaw if start_yaw is None else start_yaw
        window.append((d.qpos[2], tilt))
        if fallen_at is None and (d.qpos[2] < 0.45 or tilt > 60):
            fallen_at = (k + 1) * PHYSICS_DT * DECIMATION
        t = (k + 1) * PHYSICS_DT * DECIMATION
        if (k + 1) % 250 == 0 or k == steps - 1:
            h, tl = np.array(window).T
            drift = np.linalg.norm(d.qpos[:2] - start_xy) * 100
            print(
                f"[sim2sim] t {t:5.1f} s  pelvis z {h.mean():.3f} m (min {h.min():.3f})  tilt {tl.mean():4.1f} deg"
                f"  drift {drift:5.1f} cm  yaw {yaw - start_yaw:+6.1f} deg  lift-offs L/R {liftoffs[0]}/{liftoffs[1]}"
            )
            window = []
        if writer is not None and k % 2 == 0:
            cam.lookat[:] = d.qpos[:3] + CAM_LOOKAT
            renderer.update_scene(d, camera=cam)
            writer.append_data(renderer.render())

    if writer is not None:
        writer.close()
    status = f"FELL at {fallen_at:.1f} s" if fallen_at is not None else "stood"
    rate = liftoffs / args.seconds
    print(
        f"[sim2sim] {status}; lift-offs per s L/R {rate[0]:.2f}/{rate[1]:.2f}; clamped targets {clamped / steps:.2f} "
        f"per step; saturated torques {saturated / steps:.2f} joints per step"
        + (f"; video {args.video}" if args.video else "")
    )


if __name__ == "__main__":
    main()
