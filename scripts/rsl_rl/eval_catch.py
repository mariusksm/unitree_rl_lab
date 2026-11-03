"""Evaluate a ball-catching checkpoint under one stress-test scenario (Unified and the Advanced tasks).

Runs the task's play config (final throw distribution) with the deterministic policy and measures what the
training statistics do not show: outcomes per throw (= per episode, except in tasks with several throws per episode,
Advanced-C on: there a completed catch counts as caught and arms the next throw, and a throw cut off by the episode's
time-out counts as unfinished, not as an outcome), foot behavior (gliding vs. pivoting, drift of each stance,
lift-offs, tilt of the feet in contact), posture while waiting and while holding (incl. the posture factor P of the
reward and the pelvis height against the reference), where the ball sits while held (distance to the hold point,
contact with the torso, forbidden and support contacts) and while it lies in both palms (whether or not it counts as
held, to compare runs trained under other hold definitions), arms pressing on the torso, joints at their limits and
torque saturation, and whether confirmed catches last through the hold phase. Only the actor of the checkpoint is
loaded, so checkpoints from before a change of the critic observations still load. Scenarios change the env config
before the env is built:

  base       play config as is
  fric_high  robot friction fixed to static 1.0 / dynamic 0.8 (rubber on a lab floor)
  fric_low   robot friction fixed to static 0.4 / dynamic 0.3
  hold       no ball_caught termination, 12 s episodes: does the robot keep holding after the confirmation?
             (Advanced: throws after 0.5-1.5 s, so ~9 s of holding, one throw per episode also in Advanced-C;
             Unified: until its clip ends)
  wait       throw after 6-9 s, as on the real robot (Advanced trains 0.5-8 s, log-uniform; Advanced-C also waits 6-9 s
             before its later throws);
             Unified: every cycle starts at the beginning of its 6 s lead-in (throw after ~6.9 s)
  push       velocity pushes of ±0.5 m/s every 1-2 s
  lateral    throws aimed ±0.45 m sideways (Advanced only: Unified aims at the clip; Advanced-A trains -0.10 to
             +0.17 m, the former Phase 2 trained ±0.3 m)
  delay      perception latency 4-6 control steps (training until 2026-10-27: 1-3, since then 1-5; kept for comparison)
  arcs       flat and high lobs: apex 1.55-2.6 m above the ground (A-D train 1.75-2.1 m, Advanced-E widens to this),
             mean horizontal speed capped at 8.5 m/s (Advanced only)

All scenarios of a checkpoint plus the comparison table: ``scripts/rsl_rl/eval_catch_all.sh``.

The play config throws the final distribution of the training curriculum. ``--curriculum_progress p`` uses the
curriculum's values at progress p instead (0 = easy, 1 = final, linear as in training), so checkpoints of one run can
be compared at the same difficulty: does a falling catch rate during training come from harder throws or from a
worse policy? ``--hold_rules_progress q`` sets the hold rules alone (``secure_steps``, ``hold_radius``) to progress q,
to tell the throws from the stricter hold definition.

    python scripts/rsl_rl/eval_catch.py --headless --task Unitree-G1-29dof-BallCatch-Advanced-A \\
        --checkpoint /abs/path/model_29999.pt --scenario base --out eval/base
    # full-body video of one robot (on the workstations add the two extras below the example)
    python scripts/rsl_rl/eval_catch.py --headless --checkpoint /abs/path/model.pt --scenario hold --out eval/hold \\
        --video --num_envs 1 --video_length 900

On the group workstations camera rendering needs ``export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6`` and
``--kit_args="--portable-root $HOME/tmp/kitroot"`` (the shared environment is read-only). ``--checkpoint`` must be an
absolute path. Output: ``<out>.json`` (all metrics) and a short summary on stdout.
"""

import argparse
import json
import math
import os
import sys
import traceback

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cli_args  # isort: skip

SCENARIOS = ("base", "fric_high", "fric_low", "hold", "wait", "push", "lateral", "delay", "arcs")

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-BallCatch-Advanced-A")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--steps", type=int, default=3000, help="Control steps (50 Hz) to simulate.")
parser.add_argument("--scenario", type=str, default="base", choices=SCENARIOS)
parser.add_argument("--out", type=str, required=True, help="Output path without extension.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video instead of many robots.")
parser.add_argument("--video_length", type=int, default=1000)
parser.add_argument("--eye", type=str, default="2.6,-2.6,1.3", help="Camera position relative to the robot root.")
parser.add_argument("--lookat", type=str, default="0.0,0.0,0.55", help="Camera target relative to the robot root.")
parser.add_argument("--no_push", action="store_true", default=False,
                    help="Switch off the training pushes (to tell the policy's own unsteadiness from the recovery).")
parser.add_argument("--curriculum_progress", type=float, default=None,
                    help="Throws and hold rules of the ball_throws curriculum at this progress (0 easy .. 1 final).")
parser.add_argument("--hold_rules_progress", type=float, default=None,
                    help="Hold rules alone (secure_steps, hold_radius) at this curriculum progress.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True
# checked before Isaac Sim starts: once it runs, closing it ends the process without printing an error
if "Unified" in args_cli.task and args_cli.scenario in ("lateral", "arcs"):
    print(f"[eval] scenario '{args_cli.scenario}' does not apply to {args_cli.task}: its throws follow the reference "
          "clip (launch time and aim from the recording)", flush=True)
    sys.exit(2)

cli_args.add_setup_workaround(args_cli)  # one tasking thread: random heap corruption while the scene is built
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from collections import defaultdict
from importlib.metadata import version

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_apply_inverse, yaw_quat
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.tasks.ball_catching.mdp.curriculums import _lerp
from unitree_rl_lab.tasks.ball_catching.mdp.rewards import posture_quality
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

SECURE_STEPS_FINAL = 50  # 1 s at 50 Hz: a hold this long counts as a confirmed catch
# columns of the ball_contact sensor reported per link (label, filter body); the forbidden upper arms exist since
# 2026-11-01
BALL_CONTACT_LINKS = [
    (f"{s[0].upper()}_{label}", f"{s}_{link}")
    for s in ("left", "right")
    for label, link in (("elbow", "elbow_link"), ("wroll", "wrist_roll_link"), ("wpitch", "wrist_pitch_link"),
                        ("wyaw", "wrist_yaw_link"), ("shpitch", "shoulder_pitch_link"), ("shyaw", "shoulder_yaw_link"))
]


def apply_scenario(env_cfg, name: str):
    synced = type(env_cfg.commands.ball_throw).__name__ == "SyncedBallCommandCfg"
    if synced and name in ("lateral", "arcs"):
        raise ValueError(f"scenario '{name}' does not apply to a task whose throws follow the reference clip")
    if name == "base":
        return
    if name == "fric_high":
        env_cfg.events.physics_material.params.update(static_friction_range=(1.0, 1.0), dynamic_friction_range=(0.8, 0.8))
    elif name == "fric_low":
        env_cfg.events.physics_material.params.update(static_friction_range=(0.4, 0.4), dynamic_friction_range=(0.3, 0.3))
    elif name == "hold":
        env_cfg.terminations.ball_caught = None
        env_cfg.episode_length_s = 12.0
        if not synced:
            env_cfg.commands.ball_throw.throw_delay_range = (0.5, 1.5)
            env_cfg.commands.ball_throw.rethrow_after_catch = False
    elif name == "wait":
        if synced:
            env_cfg.commands.motion.start_hold_from_begin = True
        else:
            env_cfg.commands.ball_throw.throw_delay_range = (6.0, 9.0)
            env_cfg.commands.ball_throw.rethrow_delay_range = (6.0, 9.0)  # later throws of an episode (Advanced-C)
    elif name == "push":
        env_cfg.events.push_robot.interval_range_s = (1.0, 2.0)
    elif name == "lateral":
        env_cfg.commands.ball_throw.target_lateral_range = (-0.45, 0.45)
    elif name == "delay":
        # the perception ring buffer is sized from this range, so it must be set before the env is built
        env_cfg.commands.ball_throw.obs_delay_steps_range = (4, 6)
    elif name == "arcs":
        env_cfg.commands.ball_throw.apex_height_range = (1.55, 2.6)
        env_cfg.commands.ball_throw.apex_height_range_hard = None
        env_cfg.commands.ball_throw.max_horizontal_speed = 8.5


HOLD_RULE_KEYS = ("secure_steps", "hold_radius")


def apply_curriculum_progress(env_cfg, progress: float | None, hold_rules_progress: float | None) -> dict:
    """Set the ball command to the values of the training's ``ball_throws`` curriculum at ``progress`` (throws and
    hold rules) and ``hold_rules_progress`` (hold rules only); None keeps the play config's final values. The play
    config drops the curriculum, so its easy/final values come from the training config of the same task. Returns
    the values set."""
    if progress is None and hold_rules_progress is None:
        return {}
    train_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    term = getattr(train_cfg.curriculum, "ball_throws", None) if train_cfg.curriculum is not None else None
    if term is None:
        raise ValueError(f"{args_cli.task} has no ball_throws curriculum")
    easy, final = term.params["easy"], term.params["final"]
    applied = {}
    for key, easy_value in easy.items():
        p = hold_rules_progress if key in HOLD_RULE_KEYS and hold_rules_progress is not None else progress
        if p is None:
            continue
        applied[key] = _lerp(easy_value, final[key], min(max(p, 0.0), 1.0))
        setattr(env_cfg.commands.ball_throw, key, applied[key])
    return applied


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def yaw_of(q):
    return euler_xyz_from_quat(q.reshape(-1, 4))[2].reshape(q.shape[:-1])


def tilt_deg(q):
    """Angle between the body's z axis and the world z axis."""
    z = torch.zeros(q.reshape(-1, 4).shape[0], 3, device=q.device)
    z[:, 2] = 1.0
    up = quat_apply(q.reshape(-1, 4), z)
    return torch.rad2deg(torch.acos(up[:, 2].clamp(-1, 1))).reshape(q.shape[:-1])


def lean_fwd_deg(q):
    """Signed lean of the body's z axis in its own heading frame: + forward, - backward (tilt_deg has no sign)."""
    q = q.reshape(-1, 4)
    z = torch.zeros(q.shape[0], 3, device=q.device)
    z[:, 2] = 1.0
    up = quat_apply_inverse(yaw_quat(q), quat_apply(q, z))
    return torch.rad2deg(torch.atan2(up[:, 0], up[:, 2]))


class Stats:
    """Accumulates the metrics over all envs and steps; ``report`` turns them into a dict."""

    def __init__(self, ue):
        self.ue = ue
        self.r = r = ue.scene["robot"]
        self.cmd = ue.command_manager.get_term("ball_throw")
        self.cf = ue.scene["contact_forces"]
        self.bc = ue.scene["ball_contact"]
        self.action_term = ue.action_manager.get_term(ue.action_manager.active_terms[0])
        self.dt = ue.step_dt
        self.feet_b = r.find_bodies(["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True)[0]
        self.feet_c = self.cf.find_bodies(["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True)[0]
        self.torso_b = r.find_bodies("torso_link")[0]
        self.torso_c = self.cf.find_bodies("torso_link")[0]
        self.pelvis_c = self.cf.find_bodies("pelvis")[0]
        self.hand_b = r.find_bodies(["left_wrist_yaw_link", "right_wrist_yaw_link"], preserve_order=True)[0]
        self.elbow_b = r.find_bodies(["left_elbow_link", "right_elbow_link"], preserve_order=True)[0]
        bc_names = [path.rsplit("/", 1)[-1] for path in self.bc.cfg.filter_prim_paths_expr]
        self.bc_torso = bc_names.index("torso_link") if "torso_link" in bc_names else None
        self.bc_links = [(label, bc_names.index(link)) for label, link in BALL_CONTACT_LINKS if link in bc_names]
        self.atc = ue.scene.sensors.get("arm_torso_contact")  # arms pressing on the torso (since 2026-11-01)
        self.motion = (
            ue.command_manager.get_term("motion") if "motion" in ue.command_manager.active_terms else None
        )
        # posture factor P with the parameters of the task's `stand` reward (fixed reference height without one)
        rm = ue.reward_manager
        self.posture_params = dict(rm.get_term_cfg("stand").params) if "stand" in rm.active_terms else {}
        self.knee_j = r.find_joints(".*_knee_joint")[0]
        self.shroll_j = r.find_joints(["left_shoulder_roll_joint", "right_shoulder_roll_joint"], preserve_order=True)[0]
        self.arm_j = r.find_joints([".*_shoulder_.*", ".*_elbow_joint", ".*_wrist_.*"])[0]
        self.leg_j = r.find_joints([".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"])[0]
        self.joint_names = r.joint_names
        N, dev = ue.num_envs, ue.device
        self.acc = defaultdict(float)
        self.cnt = defaultdict(int)
        self.terms = defaultdict(int)
        self.lists = defaultdict(list)
        self.in_stance = torch.zeros(N, 2, dtype=torch.bool, device=dev)
        self.st_xy = torch.zeros(N, 2, 2, device=dev)
        self.st_yaw = torch.zeros(N, 2, device=dev)
        self.st_launched = torch.zeros(N, 2, dtype=torch.bool, device=dev)
        self.prev_foot_xy = r.data.body_pos_w[:, self.feet_b, :2].clone()
        self.prev_foot_yaw = yaw_of(r.data.body_quat_w[:, self.feet_b])
        # tasks that re-throw after a completed catch (Advanced-C): several throws per episode, counted one by one
        self.multi = bool(getattr(self.cmd.cfg, "rethrow_after_catch", False))
        self.episode_ends = 0
        self.unfinished = 0  # launched throws cut off by the time-out (several throws per episode only)
        self.ep_xy = r.data.root_pos_w[:, :2].clone()  # where the current throw's wait started
        self.ep_yaw = yaw_of(r.data.root_quat_w)
        self.throw_start = torch.zeros(N, dtype=torch.long, device=dev)  # episode step at which that wait started
        self.was_launched = self.cmd.launched.clone()
        self.launch_xy = self.ep_xy.clone()
        self.prev_root_xy = self.ep_xy.clone()
        self.max_sec = torch.zeros(N, dtype=torch.long, device=dev)
        self.ep_confirmed = torch.zeros(N, dtype=torch.bool, device=dev)
        self.ep_hold_steps = torch.zeros(N, dtype=torch.long, device=dev)
        J = len(self.joint_names)
        self.tau_max = torch.zeros(J, device=dev)
        self.tau_hi = torch.zeros(J, device=dev)
        self.tau_hi_phase = {"wait": torch.zeros(J, device=dev), "held": torch.zeros(J, device=dev)}
        self.near_lim = torch.zeros(J, device=dev)
        self.trk_err = torch.zeros(J, device=dev)
        self.tgt_out = torch.zeros(J, device=dev)
        self.tgt_beyond = torch.zeros(J, device=dev)
        self.tau_lim = r.data.joint_effort_limits[0].clone()
        self.valid_steps = 0

    def step(self):
        ue, r, cmd, acc, cnt, L = self.ue, self.r, self.cmd, self.acc, self.cnt, self.lists
        tm = ue.termination_manager
        reset = ue.episode_length_buf == 0  # reset during this step: the state is already the new episode's
        valid = ~reset
        launched = cmd.launched
        settled = valid & (ue.episode_length_buf > 15)  # skip the first 0.3 s after a reset

        # ---- throws completed within an episode (Advanced-C): the command already armed the next throw
        if self.multi:
            done = self.cmd.completed_now & valid
            if done.any():
                ids = done.nonzero().flatten()
                for i in ids.tolist():
                    self._end_throw(i, "ball_caught")
                self._start_throw(ids)
        # ---- episode ends: outcome, longest hold, drift after the launch; stances cut by the reset
        if reset.any():
            ids = reset.nonzero().flatten()
            reasons = {name: tm.get_term(name)[ids] for name in tm.active_terms}
            for k, i in enumerate(ids.tolist()):
                why = next((n for n, v in reasons.items() if bool(v[k])), "unknown")
                self.episode_ends += 1
                # with several throws per episode the time-out only cuts the current throw off (waiting, in flight or
                # held): no outcome, so the catch rate compares with the single-throw tasks
                if self.multi and why == "time_out":
                    self.unfinished += int(bool(self.was_launched[i]))
                    continue
                self._end_throw(i, why)
            cut = self.in_stance & reset[:, None]
            if cut.any():
                L["stance_disp"] += (self.prev_foot_xy - self.st_xy).norm(dim=-1)[cut].tolist()
                L["stance_dyaw"] += wrap(self.prev_foot_yaw - self.st_yaw)[cut].abs().tolist()
                L["stance_launched"] += self.st_launched[cut].tolist()
            self._start_throw(ids)
            self.in_stance[ids] = False
        self.max_sec = torch.maximum(self.max_sec, cmd.secured_steps)
        self.ep_confirmed |= cmd.confirmed
        self.ep_hold_steps = torch.maximum(self.ep_hold_steps, cmd.steps_since_confirm)

        # ---- launch: drift and turning while waiting
        new = launched & ~self.was_launched & valid
        if new.any():
            ids = new.nonzero().flatten()
            L["wait_drift"] += (r.data.root_pos_w[ids, :2] - self.ep_xy[ids]).norm(dim=-1).tolist()
            L["wait_dyaw"] += wrap(yaw_of(r.data.root_quat_w[ids]) - self.ep_yaw[ids]).abs().tolist()
            L["wait_time"] += ((ue.episode_length_buf[ids] - self.throw_start[ids]).float() * self.dt).tolist()
            self.launch_xy[ids] = r.data.root_pos_w[ids, :2]
        self.was_launched = launched.clone()

        # ---- feet: speed in firm contact, gliding (fast without turning), lift-offs, double support, stances
        fz = self.cf.data.net_forces_w[:, self.feet_c, :].norm(dim=-1)
        firm, touching = fz > 50.0, fz > 5.0
        v = r.data.body_lin_vel_w[:, self.feet_b, :2].norm(dim=-1)
        w = r.data.body_ang_vel_w[:, self.feet_b, :].norm(dim=-1)
        phase = launched[:, None].expand(-1, 2)
        for tag, pm in (("wait", ~phase), ("ball", phase)):
            m = firm & pm & settled[:, None]
            cnt[f"firm_{tag}"] += int(m.sum())
            acc[f"footv_{tag}"] += float(v[m].sum())
            cnt[f"slip_{tag}"] += int((m & (v > 0.05)).sum())
            cnt[f"glide_{tag}"] += int((m & (v > 0.05) & (w < 0.5)).sum())
            cnt[f"envsteps_{tag}"] += int((pm[:, 0] & settled).sum())
            cnt[f"double_{tag}"] += int((touching[:, 0] & touching[:, 1] & pm[:, 0] & settled).sum())
        foot_xy = r.data.body_pos_w[:, self.feet_b, :2]
        foot_yaw = yaw_of(r.data.body_quat_w[:, self.feet_b])
        start = firm & ~self.in_stance & valid[:, None]
        end = ~touching & self.in_stance & valid[:, None]
        if end.any():
            L["stance_disp"] += (foot_xy - self.st_xy).norm(dim=-1)[end].tolist()
            L["stance_dyaw"] += wrap(foot_yaw - self.st_yaw)[end].abs().tolist()
            L["stance_launched"] += self.st_launched[end].tolist()
            for tag, pm in (("wait", ~phase), ("ball", phase)):
                cnt[f"liftoff_{tag}"] += int((end & pm).sum())
        self.st_xy[start] = foot_xy[start]
        self.st_yaw[start] = foot_yaw[start]
        self.st_launched[start] = phase[start]
        self.in_stance = (self.in_stance | start) & ~end & valid[:, None]
        self.prev_foot_xy, self.prev_foot_yaw = foot_xy.clone(), foot_yaw.clone()
        self.prev_root_xy = r.data.root_pos_w[:, :2].clone()
        ftilt = tilt_deg(r.data.body_quat_w[:, self.feet_b])
        for j, side in enumerate(("L", "R")):
            m = firm[:, j] & valid
            cnt[f"ftilt_n_{side}"] += int(m.sum())
            acc[f"ftilt_{side}"] += float(ftilt[m, j].sum())
            cnt[f"ftilt_gt5_{side}"] += int((ftilt[m, j] > 5.0).sum())

        # ---- posture and smoothness while waiting / holding
        jp, jv = r.data.joint_pos, r.data.joint_vel
        da = (ue.action_manager.action - ue.action_manager.prev_action).abs().mean(dim=-1)
        sec = (cmd.secured_steps > 0) & valid
        posture = posture_quality(ue, **self.posture_params)
        ground = ue.scene.env_origins[:, 2]
        if self.motion is not None:
            ref_pelvis = self.motion.body_pos_w[:, self.motion.cfg.body_names.index("pelvis"), 2] - ground
            below_ref = ref_pelvis - (r.data.root_pos_w[:, 2] - ground)
        if self.atc is not None:
            arm_torso = self.atc.data.force_matrix_w.norm(dim=-1).flatten(start_dim=1).sum(dim=-1)  # all arm links
        for tag, m in (("wait", ~launched & settled), ("held", sec)):
            k = int(m.sum())
            if k == 0:
                continue
            cnt[f"n_{tag}"] += k
            acc[f"posture_P_{tag}"] += float(posture[m].sum())
            if self.motion is not None:
                acc[f"pelvis_below_ref_m_{tag}"] += float(below_ref[m].sum())
            if self.atc is not None:
                acc[f"arm_torso_force_N_{tag}"] += float(arm_torso[m].sum())
                cnt[f"arm_torso_over_10N_{tag}"] += int((arm_torso[m] > 10.0).sum())
            acc[f"pelvis_z_{tag}"] += float(r.data.root_pos_w[m, 2].sum())
            acc[f"pelvis_tilt_{tag}"] += float(tilt_deg(r.data.root_quat_w[m]).sum())
            acc[f"torso_tilt_{tag}"] += float(tilt_deg(r.data.body_quat_w[m][:, self.torso_b[0]]).sum())
            acc[f"pelvis_lean_fwd_{tag}"] += float(lean_fwd_deg(r.data.root_quat_w[m]).sum())
            acc[f"torso_lean_fwd_{tag}"] += float(lean_fwd_deg(r.data.body_quat_w[m][:, self.torso_b[0]]).sum())
            acc[f"knee_deg_{tag}"] += float(torch.rad2deg(jp[m][:, self.knee_j]).mean(dim=-1).sum())
            acc[f"shoulder_roll_abs_deg_{tag}"] += float(torch.rad2deg(jp[m][:, self.shroll_j].abs()).mean(dim=-1).sum())
            acc[f"arm_joint_vel_rms_{tag}"] += float(jv[m][:, self.arm_j].pow(2).mean(dim=-1).sqrt().sum())
            acc[f"leg_joint_vel_rms_{tag}"] += float(jv[m][:, self.leg_j].pow(2).mean(dim=-1).sqrt().sum())
            acc[f"action_change_{tag}"] += float(da[m].sum())
            acc[f"root_speed_{tag}"] += float(r.data.root_lin_vel_w[m, :2].norm(dim=-1).sum())

        # ---- where the ball sits while it is held
        if sec.any():
            cnt["sec"] += int(sec.sum())
            heading = yaw_quat(r.data.root_quat_w[sec])
            root = r.data.root_pos_w[sec]
            ball = cmd.ball.data.root_pos_w[sec]
            b = quat_apply_inverse(heading, ball - root)
            hands = r.data.body_pos_w[sec][:, self.hand_b]
            elbows = r.data.body_pos_w[sec][:, self.elbow_b]
            hl = quat_apply_inverse(heading, hands[:, 0] - root)
            hr = quat_apply_inverse(heading, hands[:, 1] - root)
            for i, ax in enumerate("xyz"):
                acc[f"ball_{ax}"] += float(b[:, i].sum())
                acc[f"handL_{ax}"] += float(hl[:, i].sum())
                acc[f"handR_{ax}"] += float(hr[:, i].sum())
            cnt["hands_around_ball"] += int(((hl[:, 1] > b[:, 1]) & (hr[:, 1] < b[:, 1])).sum())
            cnt["hands_below"] += int(((hl[:, 2] < b[:, 2]) & (hr[:, 2] < b[:, 2])).sum())
            acc["hand_gap"] += float((hands[:, 0] - hands[:, 1]).norm(dim=-1).sum())
            seg = hands - elbows
            t = ((ball[:, None] - elbows) * seg).sum(-1) / (seg * seg).sum(-1)
            acc["t_forearm"] += float(t.mean(-1).sum())
            torso_f = self.cf.data.net_forces_w[sec][:, self.torso_c[0]].norm(dim=-1)
            cnt["torso_contact"] += int((torso_f > 1.0).sum())
            cnt["pelvis_contact"] += int((self.cf.data.net_forces_w[sec][:, self.pelvis_c[0]].norm(dim=-1) > 1.0).sum())
            acc["torso_force"] += float(torso_f.sum())
            fm = self.bc.data.force_matrix_w[sec].norm(dim=-1).amax(dim=1)
            if self.bc_torso is not None:  # torso column of ball_contact (since 2026-10-29): ball against the torso only
                acc["ball_torso_force"] += float(fm[:, self.bc_torso].sum())
                cnt["ball_torso_over_50"] += int((fm[:, self.bc_torso] > 50.0).sum())
            for nm, j in self.bc_links:
                cnt[f"c_{nm}"] += int((fm[:, j] > 0.1).sum())
            # hold definition of 2026-11-01 (BallCommand.contact_map): distance to the hold point, graded quality,
            # forbidden and support contacts
            acc["hold_distance"] += float(cmd.hold_distance[sec].sum())
            acc["hold_quality"] += float(cmd.hold_quality[sec].sum())
            cnt["held_forbidden"] += int(cmd.forbidden_contact[sec].sum())
            cnt["held_support"] += int(cmd.support_contact[sec].sum())

        # ---- where the ball sits while it lies in both palms (counted as held or not): compares runs trained under
        #      different hold definitions (the fifth Unified run clamped it against the chest, which is not held now)
        in_hands = launched & cmd.arm_contact.all(dim=-1) & valid
        if in_hands.any():
            cnt["hands"] += int(in_hands.sum())
            b = quat_apply_inverse(yaw_quat(r.data.root_quat_w[in_hands]), cmd.ball.data.root_pos_w[in_hands]
                                   - r.data.root_pos_w[in_hands])
            for i, ax in enumerate("xyz"):
                acc[f"hands_ball_{ax}"] += float(b[:, i].sum())
            acc["hands_hold_distance"] += float(cmd.hold_distance[in_hands].sum())
            cnt["hands_counted"] += int(sec[in_hands].sum())
            cnt["hands_forbidden"] += int(cmd.forbidden_contact[in_hands].sum())
            if self.bc_torso is not None:
                fh = self.bc.data.force_matrix_w[in_hands].norm(dim=-1).amax(dim=1)
                acc["hands_ball_torso_force"] += float(fh[:, self.bc_torso].sum())

        # ---- joint limits, target outside the range, tracking error, torques
        if valid.any():
            lim = r.data.soft_joint_pos_limits[valid]
            q = jp[valid]
            span = (lim[..., 1] - lim[..., 0]).clamp(min=1e-6)
            self.near_lim += (((q - lim[..., 0]) < 0.05 * span) | ((lim[..., 1] - q) < 0.05 * span)).float().sum(dim=0)
            target = self.action_term.processed_actions[valid]
            self.trk_err += (target - q).abs().sum(dim=0)
            self.tgt_out += ((target < lim[..., 0]) | (target > lim[..., 1])).float().sum(dim=0)
            hard = r.data.joint_pos_limits[valid]  # the clamp of fix 2 keeps targets within these
            self.tgt_beyond += ((target < hard[..., 0] - 1e-4) | (target > hard[..., 1] + 1e-4)).float().sum(dim=0)
            tau = r.data.applied_torque.abs()
            self.tau_max = torch.maximum(self.tau_max, tau[valid].amax(dim=0))
            self.tau_hi += (tau[valid] > 0.8 * self.tau_lim).float().sum(dim=0)
            for tag, m in (("wait", ~launched & settled), ("held", sec)):
                if m.any():
                    self.tau_hi_phase[tag] += (tau[m] > 0.8 * self.tau_lim).float().sum(dim=0)
                    cnt[f"tau_n_{tag}"] += int(m.sum())
            self.valid_steps += int(valid.sum())

    def _end_throw(self, i: int, why: str):
        """Record the outcome of env ``i``'s current throw: ``why`` is the termination term or ``ball_caught``."""
        L = self.lists
        self.terms[why] += 1
        L["ep_max_sec"].append(int(self.max_sec[i]))
        L["ep_reason"].append(why)
        if bool(self.ep_confirmed[i]):
            L["confirmed_reason"].append(why)
            L["hold_after_confirm"].append(int(self.ep_hold_steps[i]) * self.dt)
        if bool(self.was_launched[i]):
            L["drift_after_launch"].append(float((self.prev_root_xy[i] - self.launch_xy[i]).norm()))

    def _start_throw(self, ids: torch.Tensor):
        """Reset the per-throw trackers of ``ids`` (new episode, or the next throw in a task with several)."""
        r = self.r
        self.max_sec[ids] = 0
        self.ep_confirmed[ids] = False
        self.ep_hold_steps[ids] = 0
        self.ep_xy[ids] = r.data.root_pos_w[ids, :2]
        self.ep_yaw[ids] = yaw_of(r.data.root_quat_w[ids])
        self.throw_start[ids] = self.ue.episode_length_buf[ids]
        self.was_launched[ids] = self.cmd.launched[ids]

    def report(self, scenario: str) -> dict:
        acc, cnt, L = self.acc, self.cnt, self.lists
        R = {"scenario": scenario}
        ends = sum(self.terms.values())  # throws with an outcome
        R["episodes"] = self.episode_ends
        R["throws"] = ends
        R["throws_per_episode"] = bool(self.multi)
        if self.multi:
            R["unfinished_throws"] = self.unfinished
        R["outcomes_pct"] = {k: round(100 * v / max(1, ends), 2) for k, v in sorted(self.terms.items(), key=lambda x: -x[1])}
        mx = torch.tensor(L["ep_max_sec"], dtype=torch.float) if L["ep_max_sec"] else torch.zeros(1)
        reached = mx >= SECURE_STEPS_FINAL
        R["held_1s_pct"] = round(100 * float(reached.float().mean()), 2)
        if L["confirmed_reason"]:
            reasons = L["confirmed_reason"]
            R["confirmed_pct"] = round(100 * len(reasons) / max(1, ends), 2)
            R["after_confirmation_pct"] = {k: round(100 * reasons.count(k) / len(reasons), 2) for k in set(reasons)}
            R["hold_after_confirm_s_median"] = round(float(torch.tensor(L["hold_after_confirm"]).median()), 2)
        if scenario == "hold" and reached.any():
            reasons = [L["ep_reason"][i] for i in range(len(L["ep_reason"])) if bool(reached[i])]
            R["after_confirmed_catch_pct"] = {k: round(100 * reasons.count(k) / len(reasons), 2) for k in set(reasons)}
            R["longest_hold_s_median"] = round(float(mx[reached].median()) * self.dt, 2)
        for tag in ("wait", "ball"):
            f, es = max(1, cnt[f"firm_{tag}"]), max(1, cnt[f"envsteps_{tag}"])
            R[f"feet_{tag}"] = {
                "speed_in_firm_contact_mps": round(acc[f"footv_{tag}"] / f, 3),
                "pct_contact_steps_faster_5cmps": round(100 * cnt[f"slip_{tag}"] / f, 1),
                "pct_contact_steps_gliding": round(100 * cnt[f"glide_{tag}"] / f, 1),
                "liftoffs_per_s": round(cnt[f"liftoff_{tag}"] / (es * self.dt), 2),
                "pct_time_double_support": round(100 * cnt[f"double_{tag}"] / es, 1),
            }
        if L["stance_disp"]:
            d = torch.tensor(L["stance_disp"])
            dy = torch.rad2deg(torch.tensor(L["stance_dyaw"]))
            lau = torch.tensor(L["stance_launched"], dtype=torch.bool)
            for tag, m in (("wait", ~lau), ("ball", lau)):
                if m.any():
                    R[f"stance_{tag}"] = {
                        "n": int(m.sum()),
                        "drift_cm_median": round(100 * float(d[m].median()), 1),
                        "drift_cm_p90": round(100 * float(d[m].quantile(0.9)), 1),
                        "turn_deg_median": round(float(dy[m].median()), 1),
                    }
        for side in ("L", "R"):
            n = max(1, cnt[f"ftilt_n_{side}"])
            R[f"foot_tilt_{side}"] = {"mean_deg": round(acc[f"ftilt_{side}"] / n, 1),
                                      "pct_over_5deg": round(100 * cnt[f"ftilt_gt5_{side}"] / n, 1)}
        for key in ("wait_drift", "drift_after_launch"):
            if L[key]:
                x = torch.tensor(L[key])
                R[key + "_cm"] = {"median": round(100 * float(x.median()), 1), "p90": round(100 * float(x.quantile(0.9)), 1)}
        if L["wait_dyaw"]:
            R["wait_turn_deg_median"] = round(float(torch.rad2deg(torch.tensor(L["wait_dyaw"])).median()), 1)
            R["wait_time_s_median"] = round(float(torch.tensor(L["wait_time"]).median()), 2)
        for tag in ("wait", "held"):
            n = max(1, cnt[f"n_{tag}"])
            keys = ["pelvis_z", "pelvis_tilt", "torso_tilt", "pelvis_lean_fwd", "torso_lean_fwd", "knee_deg",
                    "shoulder_roll_abs_deg", "arm_joint_vel_rms", "leg_joint_vel_rms", "action_change", "root_speed",
                    "posture_P"]
            if self.motion is not None:
                keys.append("pelvis_below_ref_m")
            if self.atc is not None:
                keys.append("arm_torso_force_N")
            R[f"posture_{tag}"] = {k: round(acc[f"{k}_{tag}"] / n, 3) for k in keys}
            if self.atc is not None:
                R[f"posture_{tag}"]["pct_arm_torso_over_10N"] = round(100 * cnt[f"arm_torso_over_10N_{tag}"] / n, 1)
        s = max(1, cnt["sec"])
        R["held"] = {
            "steps": cnt["sec"],
            "ball_from_pelvis_xyz_m": [round(acc[f"ball_{a}"] / s, 3) for a in "xyz"],
            "left_hand_xyz_m": [round(acc[f"handL_{a}"] / s, 3) for a in "xyz"],
            "right_hand_xyz_m": [round(acc[f"handR_{a}"] / s, 3) for a in "xyz"],
            "hand_gap_m": round(acc["hand_gap"] / s, 3),
            "pct_hands_left_and_right_of_ball": round(100 * cnt["hands_around_ball"] / s, 1),
            "pct_both_hands_below_ball": round(100 * cnt["hands_below"] / s, 1),
            "position_along_forearm": round(acc["t_forearm"] / s, 2),
            "pct_torso_contact": round(100 * cnt["torso_contact"] / s, 1),
            "pct_pelvis_contact": round(100 * cnt["pelvis_contact"] / s, 1),
            "torso_force_N_mean": round(acc["torso_force"] / s, 1),  # all contacts on the torso, own hands included
            "ball_torso_force_N_mean": round(acc["ball_torso_force"] / s, 1),
            "pct_ball_torso_over_50N": round(100 * cnt["ball_torso_over_50"] / s, 1),
            "arm_link_contact_pct": {k[2:]: round(100 * v / s, 1) for k, v in cnt.items() if k.startswith("c_")},
            "hold_distance_m_mean": round(acc["hold_distance"] / s, 3),
            "hold_quality_mean": round(acc["hold_quality"] / s, 3),
            "pct_forbidden_contact": round(100 * cnt["held_forbidden"] / s, 1),
            "pct_support_contact": round(100 * cnt["held_support"] / s, 1),
        }
        h = max(1, cnt["hands"])
        R["in_hands"] = {
            "steps": cnt["hands"],
            "ball_from_pelvis_xyz_m": [round(acc[f"hands_ball_{a}"] / h, 3) for a in "xyz"],
            "hold_distance_m_mean": round(acc["hands_hold_distance"] / h, 3),
            "ball_torso_force_N_mean": round(acc["hands_ball_torso_force"] / h, 1),
            "pct_counted_as_held": round(100 * cnt["hands_counted"] / h, 1),
            "pct_forbidden_contact": round(100 * cnt["hands_forbidden"] / h, 1),
        }
        vs = max(1, self.valid_steps)
        near, err, out, beyond = (x.cpu() / vs for x in (self.near_lim, self.trk_err, self.tgt_out, self.tgt_beyond))
        # limits and "outside the range": soft limits (90 % of the range); "beyond the hard limit": the joint stops
        R["joints_at_limit"] = [
            {"joint": self.joint_names[i], "pct_time_within_5pct_of_limit": round(100 * float(near[i]), 1),
             "pct_time_target_outside_range": round(100 * float(out[i]), 1),
             "pct_time_target_beyond_hard_limit": round(100 * float(beyond[i]), 1),
             "mean_abs_target_error_rad": round(float(err[i]), 3)}
            for i in torch.argsort(near, descending=True)[:6].tolist()
        ]
        worst = int(torch.argmax(beyond))
        R["target_beyond_hard_limit_max"] = {"joint": self.joint_names[worst], "pct_time": round(100 * float(beyond[worst]), 1)}
        lim, tmax, hi = self.tau_lim.cpu(), self.tau_max.cpu(), self.tau_hi.cpu() / vs
        hw = self.tau_hi_phase["wait"].cpu() / max(1, cnt["tau_n_wait"])
        hh = self.tau_hi_phase["held"].cpu() / max(1, cnt["tau_n_held"])
        R["torque_saturation"] = [
            {"joint": self.joint_names[i], "max_Nm": round(float(tmax[i]), 1), "limit_Nm": round(float(lim[i]), 1),
             "pct_steps_over_80pct": round(100 * float(hi[i]), 1), "pct_wait_over_80pct": round(100 * float(hw[i]), 1),
             "pct_held_over_80pct": round(100 * float(hh[i]), 1)}
            for i in torch.argsort(hi, descending=True)[:8].tolist()
        ]
        return R


def summary(R: dict) -> str:
    o = R["outcomes_pct"]
    fell = o.get("bad_orientation", 0) + o.get("base_height", 0)
    top = R["torque_saturation"][0]
    lines = [
        f"[eval] {R['scenario']}: {R['throws']} throws in {R['episodes']} episodes, caught {o.get('ball_caught', 0)} "
        f"%, dropped {o.get('ball_dropped', 0)} %, fell {round(fell, 2)} %, outcomes {o}"
        + (f"; {R['unfinished_throws']} throws cut off by the time-out" if "unfinished_throws" in R else ""),
        f"[eval] feet: tilt in contact L {R['foot_tilt_L']['mean_deg']}° / R {R['foot_tilt_R']['mean_deg']}°, "
        f"gliding {R['feet_wait']['pct_contact_steps_gliding']} % (wait) / {R['feet_ball']['pct_contact_steps_gliding']} % "
        f"(ball), lift-offs {R['feet_wait']['liftoffs_per_s']}/s (wait) / {R['feet_ball']['liftoffs_per_s']}/s (ball)",
        f"[eval] held: ball {R['held']['ball_from_pelvis_xyz_m']} m from the pelvis, torso contact "
        f"{R['held']['pct_torso_contact']} %, mean torso force {R['held']['torso_force_N_mean']} N (all contacts), "
        f"ball against the torso {R['held']['ball_torso_force_N_mean']} N, over 50 N in "
        f"{R['held']['pct_ball_torso_over_50N']} % of the held steps",
        f"[eval] most saturated joint: {top['joint']} above 80 % of its torque limit in {top['pct_steps_over_80pct']} % "
        f"of the steps; joint targets beyond the hard limit: at most {R['target_beyond_hard_limit_max']['pct_time']} % "
        f"of the steps ({R['target_beyond_hard_limit_max']['joint']})",
    ]
    lines.append(
        f"[eval] hold definition: held {R['held']['steps']} steps, {R['held']['hold_distance_m_mean']} m from the hold "
        f"point, quality {R['held']['hold_quality_mean']}, support contact {R['held']['pct_support_contact']} %; ball in "
        f"both palms {R['in_hands']['steps']} steps at {R['in_hands']['ball_from_pelvis_xyz_m']} m from the pelvis, "
        f"{R['in_hands']['hold_distance_m_mean']} m from the hold point, counted as held {R['in_hands']['pct_counted_as_held']} %"
    )
    lines.append(
        f"[eval] posture P waiting / holding: {R['posture_wait']['posture_P']} / {R['posture_held']['posture_P']}, pelvis "
        f"{R['posture_wait']['pelvis_z']} / {R['posture_held']['pelvis_z']} m, knees {R['posture_wait']['knee_deg']}° / "
        f"{R['posture_held']['knee_deg']}°"
        + (f", arms on the torso {R['posture_wait']['arm_torso_force_N']} / {R['posture_held']['arm_torso_force_N']} N"
           if "arm_torso_force_N" in R["posture_wait"] else "")
    )
    if "after_confirmation_pct" in R:
        lines.append(f"[eval] confirmed catches: {R['confirmed_pct']} % of the throws; how they ended: "
                     f"{R['after_confirmation_pct']} (ball_caught = hold phase completed), held "
                     f"{R['hold_after_confirm_s_median']} s after the confirmation (median)")
    if "after_confirmed_catch_pct" in R:
        lines.append(f"[eval] after a confirmed catch: {R['after_confirmed_catch_pct']}")
    return "\n".join(lines)


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True,
        entry_point_key="play_env_cfg_entry_point",
    )
    apply_scenario(env_cfg, args_cli.scenario)
    curriculum = apply_curriculum_progress(env_cfg, args_cli.curriculum_progress, args_cli.hold_rules_progress)
    if curriculum:
        print(f"[eval] curriculum progress {args_cli.curriculum_progress} (hold rules {args_cli.hold_rules_progress}): "
              f"{curriculum}", flush=True)
    if args_cli.no_push:
        env_cfg.events.push_robot = None
    if args_cli.video:
        # the reference markers of the motion command would hide the robot
        for term in vars(env_cfg.commands).values():
            if hasattr(term, "debug_vis"):
                term.debug_vis = False
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.eye = tuple(float(x) for x in args_cli.eye.split(","))
        env_cfg.viewer.lookat = tuple(float(x) for x in args_cli.lookat.split(","))
        env_cfg.viewer.resolution = (1280, 720)
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, version("rsl-rl-lib"))

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    try:  # an open env makes closing the app hang, also after an error
        steps = args_cli.steps
        if args_cli.video:
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=os.path.join(os.path.dirname(os.path.abspath(args_cli.out)), "videos", args_cli.scenario),
                step_trigger=lambda s: s == 0,
                video_length=args_cli.video_length,
                disable_logger=True,
            )
            steps = args_cli.video_length
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        # the actor only: the critic's observations changed with the tasks (2026-11-01: catch_phase), the actor's did not
        runner.load(args_cli.checkpoint, load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False})
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        with torch.inference_mode():
            obs = env.get_observations()
            stats = Stats(env.unwrapped)
            for _ in range(steps):
                obs, _, _, _ = env.step(policy(obs))
                stats.step()
            R = stats.report(args_cli.scenario)
        R.update({"task": args_cli.task, "checkpoint": args_cli.checkpoint, "num_envs": args_cli.num_envs,
                  "steps": steps, "curriculum_progress": args_cli.curriculum_progress,
                  "hold_rules_progress": args_cli.hold_rules_progress, "curriculum_values": curriculum})
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
        with open(args_cli.out + ".json", "w") as f:
            json.dump(R, f, indent=1)
        # flush before closing: closing the app ends the process without flushing a redirected stdout
        print(summary(R), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # print now: closing the app ends the process before Python would print the traceback
        traceback.print_exc()
        sys.stderr.flush()
    finally:
        simulation_app.close()
