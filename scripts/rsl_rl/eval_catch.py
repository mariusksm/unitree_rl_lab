"""Evaluate a ball-catching checkpoint under one stress-test scenario (Phase 2 and Unified).

Runs the task's play config (final throw distribution) with the deterministic policy and measures what the
training statistics do not show: outcomes per episode, foot behavior (gliding vs. pivoting, drift of each stance,
lift-offs, tilt of the feet in contact), posture while waiting and while holding, where the ball sits while held
(incl. contact with the torso), joints at their limits and torque saturation, and whether confirmed catches last
through the hold phase. Scenarios change the env config before the env is built:

  base       play config as is
  fric_high  robot friction fixed to static 1.0 / dynamic 0.8 (rubber on a lab floor)
  fric_low   robot friction fixed to static 0.4 / dynamic 0.3
  hold       no ball_caught termination, 12 s episodes: does the robot keep holding after the confirmation?
             (Phase 2: throws after 0.5-1.5 s as until 2026-10-27, so ~9 s of holding; Unified: until its clip ends)
  wait       throw after 6-9 s, as on the real robot (Phase 2 trains 0.5-8 s since 2026-10-28, 0.5-1.5 s before);
             Unified: every cycle starts at the beginning of its 6 s lead-in (throw after ~6.9 s)
  push       velocity pushes of ±0.5 m/s every 1-2 s
  lateral    throws aimed ±0.45 m sideways, beyond the trained ±0.3 m           (Phase 2 only: Unified aims at the clip;
             until 2026-10-29 ±0.3 m against a trained ±0.1 m)
  delay      perception latency 4-6 control steps (training until 2026-10-27: 1-3, since then 1-5; kept for comparison)

All scenarios of a checkpoint plus the comparison table: ``scripts/rsl_rl/eval_catch_all.sh``.

    python scripts/rsl_rl/eval_catch.py --headless --task Unitree-G1-29dof-BallCatch-Phase2 \\
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

SCENARIOS = ("base", "fric_high", "fric_low", "hold", "wait", "push", "lateral", "delay")

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-BallCatch-Phase2")
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
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True
# checked before Isaac Sim starts: once it runs, closing it ends the process without printing an error
if "Unified" in args_cli.task and args_cli.scenario == "lateral":
    print(f"[eval] scenario '{args_cli.scenario}' does not apply to {args_cli.task}: its throws follow the reference "
          "clip (launch time and aim from the recording)", flush=True)
    sys.exit(2)

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
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

SECURE_STEPS_FINAL = 50  # 1 s at 50 Hz: a hold this long counts as a confirmed catch


def apply_scenario(env_cfg, name: str):
    synced = type(env_cfg.commands.ball_throw).__name__ == "SyncedBallCommandCfg"
    if synced and name == "lateral":
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
    elif name == "wait":
        if synced:
            env_cfg.commands.motion.start_hold_from_begin = True
        else:
            env_cfg.commands.ball_throw.throw_delay_range = (6.0, 9.0)
    elif name == "push":
        env_cfg.events.push_robot.interval_range_s = (1.0, 2.0)
    elif name == "lateral":
        env_cfg.commands.ball_throw.target_lateral_range = (-0.45, 0.45)
    elif name == "delay":
        # the perception ring buffer is sized from this range, so it must be set before the env is built
        env_cfg.commands.ball_throw.obs_delay_steps_range = (4, 6)


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
        self.ep_xy = r.data.root_pos_w[:, :2].clone()
        self.ep_yaw = yaw_of(r.data.root_quat_w)
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

        # ---- episode ends: outcome, longest hold, drift after the launch; stances cut by the reset
        if reset.any():
            ids = reset.nonzero().flatten()
            reasons = {name: tm.get_term(name)[ids] for name in tm.active_terms}
            for k, i in enumerate(ids.tolist()):
                why = next((n for n, v in reasons.items() if bool(v[k])), "unknown")
                self.terms[why] += 1
                L["ep_max_sec"].append(int(self.max_sec[i]))
                L["ep_reason"].append(why)
                if bool(self.ep_confirmed[i]):
                    L["confirmed_reason"].append(why)
                    L["hold_after_confirm"].append(int(self.ep_hold_steps[i]) * self.dt)
                if bool(self.was_launched[i]):
                    L["drift_after_launch"].append(float((self.prev_root_xy[i] - self.launch_xy[i]).norm()))
            cut = self.in_stance & reset[:, None]
            if cut.any():
                L["stance_disp"] += (self.prev_foot_xy - self.st_xy).norm(dim=-1)[cut].tolist()
                L["stance_dyaw"] += wrap(self.prev_foot_yaw - self.st_yaw)[cut].abs().tolist()
                L["stance_launched"] += self.st_launched[cut].tolist()
            self.max_sec[ids] = 0
            self.ep_confirmed[ids] = False
            self.ep_hold_steps[ids] = 0
            self.ep_xy[ids] = r.data.root_pos_w[ids, :2]
            self.ep_yaw[ids] = yaw_of(r.data.root_quat_w[ids])
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
            L["wait_time"] += (ue.episode_length_buf[ids].float() * self.dt).tolist()
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
        for tag, m in (("wait", ~launched & settled), ("held", sec)):
            k = int(m.sum())
            if k == 0:
                continue
            cnt[f"n_{tag}"] += k
            acc[f"pelvis_z_{tag}"] += float(r.data.root_pos_w[m, 2].sum())
            acc[f"pelvis_tilt_{tag}"] += float(tilt_deg(r.data.root_quat_w[m]).sum())
            acc[f"torso_tilt_{tag}"] += float(tilt_deg(r.data.body_quat_w[m][:, self.torso_b[0]]).sum())
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
            for j, nm in enumerate(["L_elbow", "L_wroll", "L_wpitch", "L_wyaw", "R_elbow", "R_wroll", "R_wpitch", "R_wyaw"]):
                cnt[f"c_{nm}"] += int((fm[:, j] > 0.1).sum())

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

    def report(self, scenario: str) -> dict:
        acc, cnt, L = self.acc, self.cnt, self.lists
        R = {"scenario": scenario}
        ends = sum(self.terms.values())
        R["episodes"] = ends
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
            R[f"posture_{tag}"] = {k: round(acc[f"{k}_{tag}"] / n, 3) for k in (
                "pelvis_z", "pelvis_tilt", "torso_tilt", "knee_deg", "shoulder_roll_abs_deg", "arm_joint_vel_rms",
                "leg_joint_vel_rms", "action_change", "root_speed")}
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
            "torso_force_N_mean": round(acc["torso_force"] / s, 1),
            "arm_link_contact_pct": {k[2:]: round(100 * v / s, 1) for k, v in cnt.items() if k.startswith("c_")},
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
        f"[eval] {R['scenario']}: {R['episodes']} episodes, caught {o.get('ball_caught', 0)} %, dropped "
        f"{o.get('ball_dropped', 0)} %, fell {round(fell, 2)} %, outcomes {o}",
        f"[eval] feet: tilt in contact L {R['foot_tilt_L']['mean_deg']}° / R {R['foot_tilt_R']['mean_deg']}°, "
        f"gliding {R['feet_wait']['pct_contact_steps_gliding']} % (wait) / {R['feet_ball']['pct_contact_steps_gliding']} % "
        f"(ball), lift-offs {R['feet_wait']['liftoffs_per_s']}/s (wait) / {R['feet_ball']['liftoffs_per_s']}/s (ball)",
        f"[eval] held: ball {R['held']['ball_from_pelvis_xyz_m']} m from the pelvis, torso contact "
        f"{R['held']['pct_torso_contact']} %, mean torso force {R['held']['torso_force_N_mean']} N",
        f"[eval] most saturated joint: {top['joint']} above 80 % of its torque limit in {top['pct_steps_over_80pct']} % "
        f"of the steps; joint targets beyond the hard limit: at most {R['target_beyond_hard_limit_max']['pct_time']} % "
        f"of the steps ({R['target_beyond_hard_limit_max']['joint']})",
    ]
    if "after_confirmation_pct" in R:
        lines.append(f"[eval] confirmed catches: {R['confirmed_pct']} % of the episodes; how they ended: "
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
        runner.load(args_cli.checkpoint)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        with torch.inference_mode():
            obs = env.get_observations()
            stats = Stats(env.unwrapped)
            for _ in range(steps):
                obs, _, _, _ = env.step(policy(obs))
                stats.step()
            R = stats.report(args_cli.scenario)
        R.update({"task": args_cli.task, "checkpoint": args_cli.checkpoint, "num_envs": args_cli.num_envs, "steps": steps})
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
