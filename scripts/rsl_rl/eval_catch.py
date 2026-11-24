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
  unreachable throws aimed up to ±1.0 m sideways, most of them beyond the arms' reach (E trains up to ±0.45 m): does the
             robot stay upright when it cannot catch? (Advanced only; best with --after_loss_s)

A lost ball (dropped below 0.1 m or 6 m away) ends the episode in every task, so nothing after it is trained or
measured. ``--after_loss_s T`` keeps the episode running for T seconds after the loss and records what the robot does
in that window: falls (and when), the lowest pelvis and hand, the largest torso tilt, lift-offs, how far it moves, how
close the ball on the floor comes to a foot, and the posture factor at the end. The throw still counts as dropped or
missed. ``--hide_lost_ball`` makes the tracker stop reporting a ball that fell below 0.4 m or is more than 3 m away
after its planned arrival (``BallCommandCfg.hide_lost_ball_below`` / ``_beyond``), as a live pipeline could; without
it the lost ball stays visible on the floor, as raw VICON reports it (Advanced-S's play config, too).
``--show_ball_before_throw`` reports the ball already while it waits at the release point, held and swaying in the
thrower's hand (``BallCommandCfg.show_parked_ball_prob``), as VICON sees it before the throw.
``--gate_deviation D`` adds the catchability gate (``BallCommandCfg.gate_max_deviation``): a throw predicted from its
first measurement to come no closer than D m to the catch point while at a catchable height (0.75-1.4 m above the
ground) is never reported, so the robot keeps standing; the per-throw records mark it (``gated``).
``--hide_lost_ball SHARE`` and ``--show_ball_before_throw SHARE`` apply the option to that share of the throws only (the
flag alone: all throws). The per-throw columns ``lost_hidden`` and ``shown_before`` say which throws got it, and the
report splits the window after a loss (``after_loss_by_lost_ball``) and the time before the launch
(``ball_before_throw``) by them, so one run compares both cases on the same robots.

Unified (stage A) has no fall terminations, but the robot leaves its reference motion (tracking terminations
``anchor_pos``, ``anchor_ori``, ``ee_body_pos``), and at the end of a reference clip the next cycle starts within the
episode: the robot is moved to the new clip and a new throw is armed. A throw whose clip ends before it has an outcome
is recorded as ``cycle_end`` (until 2026-11-24 it was merged with the next cycle's throw into one record). With
``--after_loss_s`` the window after a loss ends at the clip end at the latest, as ``mdp.ball_lost_since`` does in
training (the next cycle would move the robot and clear the loss), and the column ``post_loss_end`` says how it ended:
``complete`` (ran its full length), ``fell``, ``left_reference`` (a tracking termination: in Unified, going after the
ball shows up here), ``clip_end`` or ``time_out``. Use 3 s for Unified, the training's window.

``--fixed_throws SEED`` evaluates on a fixed throw list: the startup randomization (``env_cfg.seed``) and the random
draws of every throw (release point, aim, wait, the perception parameters of the throw, launch noise; in Unified also
the clip, start point and reset state of every reference cycle, whose start is then drawn uniformly instead of by the
adaptive sampling) come from SEED, robot by robot and throw by throw. The k-th throw of robot i (columns ``env``,
``throw_index``) is then the same for every checkpoint evaluated with the same SEED, task, scenario and ``--num_envs``,
whatever the policy did before. A seed alone does not do that: when the robots reset depends on the policy, and so does
the order of the random stream. Pushes, tracker dropouts, measurement noise and the reset states of the Advanced tasks
stay random (``--no_push`` gives the tightest pairing). ``scripts/rsl_rl/eval_catch_paired.py`` compares two such runs
throw by throw.

More per-throw columns (since 2026-11-24): ``both_palms`` (both palms on the ball at once); the first palm contact:
``contact_rel_speed_mps`` (speed of the ball relative to the mean of both wrists) and ``contact_ball_speed_mps``, both in
the control step before the contact, and ``contact_hold_dist_m`` (the ball's distance to the hold point at it), to tell
a bounce off the palms from an inaccurate interception; ``wait_hand_h_m`` (wrists above the pelvis, mean over the wait)
and in Unified ``wait_hand_above_ref_m`` (the largest height of a wrist above the reference wrist while waiting, the
quantity ``ee_body_pos`` ends at 0.3 m); the startup randomization of the robot (``ball_restitution``,
``ball_friction``, ``ball_mass_kg``; ``hand_restitution`` and ``hand_friction``, means over the shapes of both hands;
``foot_friction``). The ball-hand contact uses the mean of the ball's and the hand's restitution (combine mode average).
The report summarizes them as ``first_palm_contact`` (caught against lost) and ``waiting_hands``.

The runner is built for inference only (``cli_args.for_inference``): AnchoredPPO's masked anchor, which needs ball
indices that only ``train.py`` fills in, is switched off, so Advanced-S and the chain-v2 tasks can be evaluated too.

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
absolute path. Output: ``<out>.json`` (all metrics), a short summary on stdout, and ``<out>_throws.csv`` with one row
per throw that has an outcome: where the throw went relative to the robot at its launch (aim point, apex, release
direction; the command samples them relative to the robot when the wait starts, and the robot may move or turn while it
waits), the perception delay, how it ended (lost without touching arms or torso, after touching them, after lying on a
palm, after being held, after the confirmation) and the pushes while the ball was in play: during the flight (launch to
the planned arrival, the same window for every throw), from the arrival to the confirmation, and while holding.
``scripts/rsl_rl/eval_catch_throws.py`` turns these rows into catch rates per throw type.
"""

import argparse
import csv
import functools
import hashlib
import json
import math
import os
import sys
import traceback

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cli_args  # isort: skip

SCENARIOS = ("base", "fric_high", "fric_low", "hold", "wait", "push", "lateral", "delay", "arcs", "unreachable")

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
parser.add_argument("--after_loss_s", type=float, default=None,
                    help="Keep the episode running this long after the ball is lost and measure what the robot does.")
parser.add_argument("--hide_lost_ball", type=float, nargs="?", const=1.0, default=None, metavar="SHARE",
                    help="The tracker stops reporting a lost ball (below 0.4 m, or 3 m away after the planned arrival); "
                         "with SHARE only in that share of the throws (column lost_hidden).")
parser.add_argument("--show_ball_before_throw", type=float, nargs="?", const=1.0, default=None, metavar="SHARE",
                    help="Report the ball already while it waits at the release point (thrower's hand), swaying; with "
                         "SHARE only in that share of the throws (column shown_before).")
parser.add_argument("--fixed_throws", type=int, default=None, metavar="SEED",
                    help="Fixed throw list: startup randomization and the random draws of every throw come from SEED, "
                         "robot by robot and throw by throw (same throws for every checkpoint; see the docstring).")
parser.add_argument("--gate_deviation", type=float, default=None,
                    help="Catchability gate: a throw predicted from its first measurement to come no closer than this "
                         "(m) to the catch point at a catchable height is never reported to the robot.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True
# checked before Isaac Sim starts: once it runs, closing it ends the process without printing an error
if "Unified" in args_cli.task and args_cli.scenario in ("lateral", "arcs", "unreachable"):
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
from isaaclab.managers import TerminationTermCfg as DoneTerm
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
    if synced and name in ("lateral", "arcs", "unreachable"):
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
    elif name == "unreachable":
        env_cfg.commands.ball_throw.target_lateral_range = (-1.0, 1.0)
        env_cfg.commands.ball_throw.target_lateral_range_hard = None


def record_pushes(func):
    """Wrap the push event term so that the evaluation sees which robots were pushed and how hard: the largest change
    of the root's horizontal velocity (m/s) per env since the last control step, kept in ``env._eval_push``."""

    @functools.wraps(func)  # the event manager checks the term's parameters against the wrapped signature
    def pushed(env, env_ids, *args, **kwargs):
        robot = env.scene["robot"]
        ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
        before = robot.data.root_lin_vel_w[ids, :2].clone()
        result = func(env, env_ids, *args, **kwargs)
        change = (robot.data.root_lin_vel_w[ids, :2] - before).norm(dim=-1)
        if not hasattr(env, "_eval_push"):
            env._eval_push = torch.zeros(env.num_envs, device=env.device)
        env._eval_push[ids] = torch.maximum(env._eval_push[ids], change)
        return result

    return pushed


LOST_HEIGHT, LOST_DISTANCE = 0.1, 6.0  # thresholds of the tasks' ball_dropped / ball_missed terminations


def ball_lost_now(env, cmd) -> torch.Tensor:
    """The ball is lost: below LOST_HEIGHT (as ``ball_dropped``) or farther than LOST_DISTANCE from the robot (as
    ``ball_missed``)."""
    ball, root = cmd.ball.data.root_pos_w, env.scene["robot"].data.root_pos_w
    return (ball[:, 2] < LOST_HEIGHT) | ((ball[:, :2] - root[:, :2]).norm(dim=-1) > LOST_DISTANCE)


def ball_lost_for(env, wait_s: float) -> torch.Tensor:
    """Termination of ``--after_loss_s``: True once the launched ball has been lost for ``wait_s`` seconds (the
    counter keeps running when a lost ball bounces back up)."""
    cmd = env.command_manager.get_term("ball_throw")
    steps = getattr(env, "_eval_lost_steps", None)
    if steps is None:
        steps = env._eval_lost_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    steps[env.episode_length_buf <= 1] = 0  # a new episode (the step counter was reset to 0 and has run once)
    lost = cmd.launched & ball_lost_now(env, cmd)
    steps[:] = torch.where(lost | (steps > 0), steps + 1, steps)
    return steps.float() * env.step_dt >= wait_s


def loss_at_clip_end(env) -> torch.Tensor:
    """Termination of ``--after_loss_s`` in Unified: a lost ball also ends the episode on the last step of its reference
    clip, as ``mdp.ball_lost_since`` does in training: the next cycle would start there, move the robot to the new clip
    and clear the loss. Reads the loss counter of ``ball_lost_for`` (term ``ball_dropped``, computed before this one)."""
    steps = getattr(env, "_eval_lost_steps", None)
    if steps is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    motion = env.command_manager.get_term("motion")
    # the motion command moves to the next cycle in the command update after this check, once time_steps reaches the
    # clip length
    last = motion.time_steps + 1 >= motion.motion.time_step_total[motion.clip_ids]
    return (steps > 0) & last & (env.episode_length_buf > 1)


def _seed_of(*parts) -> int:
    """A 63-bit seed from the parts, the same in every process (unlike Python's hash())."""
    digest = hashlib.blake2b(":".join(str(p) for p in parts).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") >> 1


def fix_throws(ue, seed: int):
    """``--fixed_throws``: every throw draws its random numbers from a stream of its own, seeded by (seed, robot, throw
    index), so the k-th throw of robot i is the same whatever the policy did before.

    Wraps, on the instances, the methods that draw a throw: ``BallCommand._arm_throw`` (Advanced: throw, wait,
    perception parameters of the throw) or ``SyncedBallCommand._start_cycle`` (Unified), the launch (velocity noise;
    Unified: catch point jitter) and in Unified ``MotionCommand._resample_command`` (clip, start point, reset state of a
    cycle; its start bins must not adapt to the policy, ``adaptive_alpha`` 0). Each wrapped call runs robot by robot
    inside ``torch.random.fork_rng``, so the global stream, which everything else draws from (pushes, dropouts,
    measurement noise, resets of the Advanced tasks), continues as if nothing had been drawn. The command's
    ``_eval_throw_index`` holds the index of each robot's current throw for the per-throw records (1 for the first)."""
    cmd = ue.command_manager.get_term("ball_throw")
    device = torch.device(ue.device)
    devices = [device.index or 0] if device.type == "cuda" else []
    cmd._eval_throw_index = torch.zeros(ue.num_envs, dtype=torch.long, device=device)

    def per_robot(method, site: str, index):
        """``method`` called robot by robot, each call seeded by (seed, site, robot, ``index(robot)``)."""

        def call(env_ids, *args, **kwargs):
            if isinstance(env_ids, slice):
                env_ids = torch.arange(ue.num_envs, device=device)[env_ids]
            ids = torch.as_tensor(env_ids, dtype=torch.long, device=device).flatten()
            for k in range(len(ids)):
                one = ids[k:k + 1]
                i = int(one)
                with torch.random.fork_rng(devices=devices):
                    torch.manual_seed(_seed_of(seed, site, i, index(i)))
                    method(one, *args, **kwargs)

        return call

    def current(i: int) -> int:
        return int(cmd._eval_throw_index[i])

    if hasattr(cmd, "_start_cycle"):  # Unified: one throw per reference cycle, indexed by the cycle
        motion = ue.command_manager.get_term(cmd.cfg.motion_command_name)

        def next_cycle(i: int) -> int:  # the motion command counts the cycle after drawing it
            return int(motion.resample_count[i]) + 1

        def cycle_throw(i: int) -> int:
            cmd._eval_throw_index[i] = motion.resample_count[i]
            return current(i)

        motion._resample_command = per_robot(motion._resample_command, "cycle", next_cycle)
        cmd._start_cycle = per_robot(cmd._start_cycle, "throw", cycle_throw)
    else:

        def next_throw(i: int) -> int:
            cmd._eval_throw_index[i] += 1
            return current(i)

        cmd._arm_throw = per_robot(cmd._arm_throw, "throw", next_throw)
    cmd._launch = per_robot(cmd._launch, "launch", current)


GRAVITY = 9.81
# per-throw records (<out>_throws.csv): parameters taken at the launch, then how the throw ended
THROW_FIELDS = ("wait_s", "turn_wait_deg", "aim_fwd_m", "aim_lat_m", "aim_h_m", "release_dist_m", "release_az_deg",
                "release_h_m", "apex_h_m", "flight_s", "speed_h_mps", "obs_delay_steps", "hidden_in_zone")
# with --after_loss_s: what the robot did between the loss and the episode end (empty otherwise); post_loss_end: how the
# window ended (window_end)
POST_LOSS_COLUMNS = ("lost_as", "post_loss_s", "post_loss_fell", "post_loss_min_pelvis_m", "post_loss_max_tilt_deg",
                     "post_loss_min_hand_m", "post_loss_liftoffs", "post_loss_disp_m", "post_loss_ball_foot_m",
                     "post_loss_end_P", "post_loss_end")
# which robot, its throw index (--fixed_throws: the index into the fixed list), whether the ball was reported before the
# launch, whether a lost ball would be hidden from the robot in this throw
ID_COLUMNS = ("env", "throw_index", "shown_before", "lost_hidden")
# both palms on the ball at once; the first palm contact: speed of the ball relative to the hands (mean of both wrists)
# and the ball's speed, both in the control step before it, and the ball's distance to the hold point at it
CONTACT_COLUMNS = ("both_palms", "contact_rel_speed_mps", "contact_ball_speed_mps", "contact_hold_dist_m")
# the hands while waiting for the launch: wrists above the pelvis (mean over the wait); Unified: the largest height of a
# wrist above the reference wrist (what ee_body_pos ends at 0.3 m)
WAIT_COLUMNS = ("wait_hand_h_m", "wait_hand_above_ref_m")
# startup randomization of the robot (fixed for the run): the ball, the shapes of both hands (mean), both feet (mean);
# the ball-hand contact uses the mean of the ball's and the hand's restitution (combine mode average)
MATERIAL_COLUMNS = ("ball_restitution", "ball_friction", "ball_mass_kg", "hand_restitution", "hand_friction",
                    "foot_friction")
THROW_COLUMNS = ("outcome", "launched") + THROW_FIELDS + (
    "touched", "palm", "max_secured_s", "confirmed", "pushes_flight", "push_max_flight_mps", "pushes_catch",
    "pushes_hold", "gated") + POST_LOSS_COLUMNS + ID_COLUMNS + CONTACT_COLUMNS + WAIT_COLUMNS + MATERIAL_COLUMNS
F = {name: k for k, name in enumerate(THROW_FIELDS)}
LOSSES = ("ball_dropped", "ball_missed")  # the ball is lost (a time-out is no loss: in `hold` it ends a held catch)
FALLS = ("bad_orientation", "base_height")
TRACKING = ("anchor_pos", "anchor_ori", "ee_body_pos")  # Unified: the robot left its reference motion


def window_end(why: str) -> str:
    """How the window after a loss ended, from the termination that ended the episode: ``complete`` (it ran its full
    length), ``fell``, ``left_reference`` (a tracking termination of Unified), ``clip_end`` (Unified: the reference clip
    ended), else the termination's name (``time_out``; ``ball_caught``: a confirmed catch whose hold phase completed
    after the drop)."""
    if why == "ball_dropped":  # the term ball_lost_for of --after_loss_s
        return "complete"
    if why in FALLS:
        return "fell"
    if why in TRACKING:
        return "left_reference"
    if why == "loss_clip_end":
        return "clip_end"
    return why


def quantile_of(rows: list[dict], key: str, q: float):
    """Quantile ``q`` of ``key`` over the records that have it (None if none has)."""
    values = sorted(x[key] for x in rows if x[key] != "")
    return round(values[min(len(values) - 1, int(q * len(values)))], 3) if values else None


def before_launch_stats(rows: list[dict]) -> dict:
    """For one group of throws (e.g. ball reported before the throw or not): how many were caught, and how many ended
    before the launch, by a fall or by leaving the reference (Unified)."""
    n = len(rows)
    if n == 0:
        return {"throws": 0}
    early = [x for x in rows if not x["launched"]]

    def pct(k: int) -> float:
        return round(100 * k / n, 2)

    return {
        "throws": n,
        "pct_caught": pct(sum(x["outcome"] == "ball_caught" for x in rows)),
        "pct_ended_before_launch": pct(len(early)),
        "pct_fell_before_launch": pct(sum(x["outcome"] in FALLS for x in early)),
        "pct_left_reference_before_launch": pct(sum(x["outcome"] in TRACKING for x in early)),
        "pct_fell_at_all": pct(sum(x["outcome"] in FALLS or x["post_loss_end"] == "fell" for x in rows)),
    }


def loss_stage(row: dict) -> str:
    """How far a lost ball got: never touched arms or torso, touched them, lay on a palm, was held, or was confirmed."""
    if row["confirmed"]:
        return "confirmed"
    if row["max_secured_s"] > 0:
        return "held"
    if row["palm"]:
        return "palm"
    return "touched" if row["touched"] else "never_touched"


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

    def __init__(self, ue, after_loss_s: float | None = None):
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
        # a hand pressing on the other arm (sensors of Advanced-A2 on and Unified, since 2026-11-03)
        self.hand_sensors = [ue.scene.sensors[n] for n in ("hand_contact_left", "hand_contact_right")
                             if n in ue.scene.sensors]
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
        # per throw: parameters at the launch (THROW_FIELDS), contact with the robot and pushes while the ball is in
        # play
        self.th = torch.full((N, len(THROW_FIELDS)), float("nan"), device=dev)
        self.th_touch = torch.zeros(N, dtype=torch.bool, device=dev)
        self.th_palm = torch.zeros(N, dtype=torch.bool, device=dev)
        self.th_gated = torch.zeros(N, dtype=torch.bool, device=dev)  # --gate_deviation: never reported to the robot
        # pushes during the flight (launch to the planned arrival: a window of the same length for caught and lost
        # balls), from the arrival to the confirmation (ends early when the ball is lost), after the confirmation
        self.th_push_flight = torch.zeros(N, dtype=torch.long, device=dev)
        self.th_push_max = torch.zeros(N, device=dev)  # largest push during the flight
        self.th_push_catch = torch.zeros(N, dtype=torch.long, device=dev)
        self.th_push_hold = torch.zeros(N, dtype=torch.long, device=dev)
        self.throw_rows = []
        # after a lost ball (--after_loss_s): the window from the loss to the episode end
        self.after_loss_s = after_loss_s
        self.pl_step = torch.full((N,), -1, dtype=torch.long, device=dev)  # episode step of the loss, -1: not lost
        self.pl_dropped = torch.zeros(N, dtype=torch.bool, device=dev)  # lost below LOST_HEIGHT (else: too far)
        self.pl_xy = torch.zeros(N, 2, device=dev)
        self.pl_steps = torch.zeros(N, dtype=torch.long, device=dev)
        self.pl_min_pelvis = torch.full((N,), float("inf"), device=dev)
        self.pl_max_tilt = torch.zeros(N, device=dev)
        self.pl_min_hand = torch.full((N,), float("inf"), device=dev)
        self.pl_liftoffs = torch.zeros(N, dtype=torch.long, device=dev)
        self.pl_disp = torch.zeros(N, device=dev)
        self.pl_ball_foot = torch.full((N,), float("inf"), device=dev)
        self.pl_end_P = torch.zeros(N, device=dev)
        # per throw: its index on the robot (--fixed_throws: the command's index into the list, else counted here, from
        # 1), whether the ball was reported before the launch, whether a lost ball would be hidden from the robot
        self.th_index = torch.zeros(N, dtype=torch.long, device=dev)
        self.th_shown = torch.zeros(N, dtype=torch.bool, device=dev)
        self.th_lost_hidden = torch.zeros(N, dtype=torch.bool, device=dev)
        ccfg = self.cmd.cfg
        self.hide_rule = ccfg.hide_lost_ball_below is not None or ccfg.hide_lost_ball_beyond is not None
        # Unified: the reference cycle of each robot; a new cycle within the episode starts a new throw
        self.cycle = self.motion.resample_count.clone() if self.motion is not None else None
        self.since_start = torch.zeros(N, dtype=torch.long, device=dev)  # steps since the reset or the cycle start
        # per throw (CONTACT_COLUMNS, WAIT_COLUMNS): both palms, the first palm contact (relative speed, ball speed,
        # distance to the hold point), the velocities of the last step (before the contact), the hands while waiting
        self.th_both = torch.zeros(N, dtype=torch.bool, device=dev)
        self.th_contact = torch.full((N, 3), float("nan"), device=dev)
        self.prev_rel_vel = torch.zeros(N, 3, device=dev)
        self.prev_ball_speed = torch.zeros(N, device=dev)
        self.th_wait_hand_sum = torch.zeros(N, device=dev)
        self.th_wait_n = torch.zeros(N, dtype=torch.long, device=dev)
        self.th_wait_above_ref = torch.full((N,), float("nan"), device=dev)
        self.ref_hand = []
        if self.motion is not None:
            names = list(self.motion.cfg.body_names)
            self.ref_hand = [names.index(n) for n in ("left_wrist_yaw_link", "right_wrist_yaw_link") if n in names]
        self.materials = self._env_materials()
        self._start_throw(torch.arange(N, device=dev))

    def _env_materials(self) -> dict:
        """Startup randomization of each robot, fixed for the run (MATERIAL_COLUMNS): friction and restitution of the
        ball and its mass, the mean friction and restitution over the shapes of both hands (``wrist_yaw_link`` with
        the rubber hand) and the mean friction of both feet. Columns stay empty where PhysX does not tell."""
        out = {}
        try:
            view = self.cmd.ball.root_physx_view
            mat = view.get_material_properties()  # (N, shapes, 3): static friction, dynamic friction, restitution
            out["ball_friction"], out["ball_restitution"] = mat[:, 0, 0], mat[:, 0, 2]
            out["ball_mass_kg"] = view.get_masses()[:, 0]
        except Exception as e:  # noqa: BLE001 - a missing column must not stop the evaluation
            print(f"[eval] ball material not read: {e}", flush=True)
        try:
            r = self.r
            view = r.root_physx_view
            mat = view.get_material_properties()
            # shapes of each link in link order, counted as Isaac Lab's randomize_rigid_body_material counts them
            counts = [r._physics_sim_view.create_rigid_body_view(p).max_shapes for p in view.link_paths[0]]
            starts = [sum(counts[:b]) for b in range(len(counts))]

            def mean_over(bodies: list[str], column: int) -> torch.Tensor:
                ids = r.find_bodies(bodies, preserve_order=True)[0]
                shapes = [k for b in ids for k in range(starts[b], starts[b] + counts[b])]
                return mat[:, shapes, column].mean(dim=1)

            hands = ["left_wrist_yaw_link", "right_wrist_yaw_link"]
            out["hand_friction"], out["hand_restitution"] = mean_over(hands, 0), mean_over(hands, 2)
            out["foot_friction"] = mean_over(["left_ankle_roll_link", "right_ankle_roll_link"], 0)
        except Exception as e:  # noqa: BLE001
            print(f"[eval] robot materials not read: {e}", flush=True)
        return {k: v.float().cpu() for k, v in out.items()}

    def _cut_stances(self, rows: torch.Tensor):
        """End the stances of the robots in ``rows`` without a lift-off: reset or move to a new reference cycle."""
        L = self.lists
        cut = self.in_stance & rows[:, None]
        if cut.any():
            L["stance_disp"] += (self.prev_foot_xy - self.st_xy).norm(dim=-1)[cut].tolist()
            L["stance_dyaw"] += wrap(self.prev_foot_yaw - self.st_yaw)[cut].abs().tolist()
            L["stance_launched"] += self.st_launched[cut].tolist()
        self.in_stance &= ~rows[:, None]

    def step(self):
        ue, r, cmd, acc, cnt, L = self.ue, self.r, self.cmd, self.acc, self.cnt, self.lists
        tm = ue.termination_manager
        reset = ue.episode_length_buf == 0  # reset during this step: the state is already the new episode's
        # Unified: the clip ended and the next cycle started within the episode (robot moved to the new clip, new throw)
        if self.motion is not None:
            restart = (self.motion.resample_count != self.cycle) & ~reset
        else:
            restart = torch.zeros_like(reset)
        valid = ~reset & ~restart
        self.since_start = torch.where(valid, self.since_start + 1, torch.zeros_like(self.since_start))
        launched = cmd.launched
        settled = valid & (self.since_start > 15)  # skip the first 0.3 s after a reset or a new cycle

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
                if self.multi and why == "time_out" and int(self.pl_step[i]) < 0:
                    self.unfinished += int(bool(self.was_launched[i]))
                    continue
                self._end_throw(i, why)
            self._cut_stances(reset)
            self._start_throw(ids)
        # ---- Unified: a new reference cycle started within the episode; the old cycle's throw had no outcome
        if restart.any():
            ids = restart.nonzero().flatten()
            for i in ids.tolist():
                self._end_throw(i, "cycle_end")
            self._cut_stances(restart)
            self._start_throw(ids)
        if self.motion is not None:
            self.cycle = self.motion.resample_count.clone()
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
            self._record_launch(ids)
        self.was_launched = launched.clone()

        # ---- per throw while the ball is in play: contact with the robot (any filter body of the ball sensor, over
        #      the physics steps of this control step), on a palm, pushes (recorded by the wrapped push term)
        live = launched & valid
        data = self.bc.data
        forces = data.force_matrix_w_history
        if forces is None:
            forces = data.force_matrix_w.unsqueeze(1)
        touching = forces.norm(dim=-1).amax(dim=1).amax(dim=1) > 1.0  # (N, filter bodies)
        self.th_touch |= live & touching.any(dim=-1)
        palm = live & cmd.arm_contact.any(dim=-1)
        first = palm & ~self.th_palm  # first palm contact of the throw: velocities of the step before, not after it
        if first.any():
            self.th_contact[first, 0] = self.prev_rel_vel[first].norm(dim=-1)
            self.th_contact[first, 1] = self.prev_ball_speed[first]
            self.th_contact[first, 2] = cmd.hold_distance[first]
        self.th_palm |= palm
        self.th_both |= live & cmd.arm_contact.all(dim=-1)
        ball_vel = cmd.ball.data.root_lin_vel_w
        self.prev_rel_vel = ball_vel - r.data.body_lin_vel_w[:, self.hand_b].mean(dim=1)
        self.prev_ball_speed = ball_vel.norm(dim=-1)
        self.th_gated |= live & cmd.ball_gated
        push = getattr(ue, "_eval_push", None)
        if push is not None:
            hit = live & (push > 0.01)
            flying = hit & (cmd.time_since_throw <= cmd._flight_time)
            self.th_push_flight += flying.long()
            self.th_push_max = torch.where(flying, torch.maximum(self.th_push_max, push), self.th_push_max)
            self.th_push_catch += (hit & ~flying & ~cmd.confirmed).long()
            self.th_push_hold += (hit & cmd.confirmed).long()
            push.zero_()

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
        if self.hand_sensors:
            hand_force = sum(  # strongest contact of each hand with the other arm
                sn.data.force_matrix_w.norm(dim=-1).flatten(start_dim=1).amax(dim=-1) for sn in self.hand_sensors
            )
        # the hands of each throw while waiting (WAIT_COLUMNS)
        waiting = ~launched & settled
        wrists_above_pelvis = r.data.body_pos_w[:, self.hand_b, 2].mean(dim=1) - r.data.root_pos_w[:, 2]
        self.th_wait_hand_sum += torch.where(waiting, wrists_above_pelvis, torch.zeros_like(wrists_above_pelvis))
        self.th_wait_n += waiting.long()
        if len(self.ref_hand) == 2:  # Unified: as ee_body_pos measures it (robot minus reference, height only)
            m = self.motion
            above = (m.robot_body_pos_w[:, self.ref_hand, 2] - m.body_pos_relative_w[:, self.ref_hand, 2]).amax(dim=1)
            self.th_wait_above_ref = torch.where(waiting, torch.fmax(self.th_wait_above_ref, above), self.th_wait_above_ref)
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
            if self.hand_sensors:
                acc[f"hand_contact_N_{tag}"] += float(hand_force[m].sum())
                cnt[f"hand_contact_over_5N_{tag}"] += int((hand_force[m] > 5.0).sum())
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

        # ---- after a lost ball (--after_loss_s): what the robot does until the episode ends
        if self.after_loss_s is not None:
            root, ball = r.data.root_pos_w, cmd.ball.data.root_pos_w
            new = launched & valid & ball_lost_now(ue, cmd) & (self.pl_step < 0)
            if new.any():
                self.pl_step[new] = ue.episode_length_buf[new]
                self.pl_dropped[new] = ball[new, 2] < LOST_HEIGHT
                self.pl_xy[new] = root[new, :2]
            w = valid & (self.pl_step >= 0)
            if w.any():
                self.pl_steps += w.long()
                keep = lambda new_value, old, better: torch.where(w, better(old, new_value), old)  # noqa: E731
                self.pl_min_pelvis = keep(root[:, 2] - ground, self.pl_min_pelvis, torch.minimum)
                self.pl_max_tilt = keep(tilt_deg(r.data.body_quat_w[:, self.torso_b[0]]), self.pl_max_tilt, torch.maximum)
                hand_z = r.data.body_pos_w[:, self.hand_b, 2].amin(dim=1) - ground
                self.pl_min_hand = keep(hand_z, self.pl_min_hand, torch.minimum)
                self.pl_liftoffs += (end.sum(dim=1) * w).long()
                self.pl_disp = keep((root[:, :2] - self.pl_xy).norm(dim=-1), self.pl_disp, torch.maximum)
                ball_foot = (r.data.body_pos_w[:, self.feet_b] - ball[:, None]).norm(dim=-1).amin(dim=1)
                self.pl_ball_foot = keep(ball_foot, self.pl_ball_foot, torch.minimum)
                self.pl_end_P = torch.where(w, posture, self.pl_end_P)

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
        """Record the outcome of env ``i``'s current throw: ``why`` is the termination term, ``ball_caught`` or
        ``cycle_end`` (Unified: the reference clip ended first). With ``--after_loss_s`` a lost ball stays the outcome;
        what ended the window after it (a fall, leaving the reference, ...) is recorded in the post-loss columns."""
        L = self.lists
        post = [""] * len(POST_LOSS_COLUMNS)
        if why == "loss_clip_end" and int(self.pl_step[i]) < 0:
            why = "ball_dropped"  # the loss counter saw a loss this record missed (a ball that bounced back at once)
        if self.after_loss_s is not None and int(self.pl_step[i]) >= 0:
            lost_as = "ball_dropped" if bool(self.pl_dropped[i]) else "ball_missed"
            post = [lost_as, round(int(self.pl_steps[i]) * self.dt, 2), int(why in FALLS),
                    round(float(self.pl_min_pelvis[i]), 3), round(float(self.pl_max_tilt[i]), 1),
                    round(float(self.pl_min_hand[i]), 3), int(self.pl_liftoffs[i]), round(float(self.pl_disp[i]), 3),
                    round(float(self.pl_ball_foot[i]), 3), round(float(self.pl_end_P[i]), 3), window_end(why)]
            why = lost_as
        self.terms[why] += 1
        L["ep_max_sec"].append(int(self.max_sec[i]))
        L["ep_reason"].append(why)
        if bool(self.ep_confirmed[i]):
            L["confirmed_reason"].append(why)
            L["hold_after_confirm"].append(int(self.ep_hold_steps[i]) * self.dt)
        if bool(self.was_launched[i]):
            L["drift_after_launch"].append(float((self.prev_root_xy[i] - self.launch_xy[i]).norm()))
        params = [round(x, 4) if x == x else "" for x in self.th[i].tolist()]  # NaN: not launched
        self.throw_rows.append(
            [why, int(bool(self.was_launched[i]))] + params
            + [int(bool(self.th_touch[i])), int(bool(self.th_palm[i])), round(int(self.max_sec[i]) * self.dt, 2),
               int(bool(self.ep_confirmed[i])), int(self.th_push_flight[i]), round(float(self.th_push_max[i]), 3),
               int(self.th_push_catch[i]), int(self.th_push_hold[i]), int(bool(self.th_gated[i]))] + post
            + [i, int(self.th_index[i]), int(bool(self.th_shown[i])), int(bool(self.th_lost_hidden[i]))]
            + self._contact_wait_material(i)
        )

    def _contact_wait_material(self, i: int) -> list:
        """CONTACT_COLUMNS, WAIT_COLUMNS and MATERIAL_COLUMNS of env ``i``'s current throw (empty where not known)."""
        def num(x: float, digits: int = 3):
            return round(x, digits) if x == x else ""  # NaN: not measured

        n_wait = int(self.th_wait_n[i])
        return (
            [int(bool(self.th_both[i]))] + [num(x) for x in self.th_contact[i].tolist()]
            + [num(float(self.th_wait_hand_sum[i]) / n_wait) if n_wait else "", num(float(self.th_wait_above_ref[i]))]
            + [num(float(self.materials[k][i]), 4) if k in self.materials else "" for k in MATERIAL_COLUMNS]
        )

    def _record_launch(self, ids: torch.Tensor):
        """Parameters of the launched throws of ``ids`` relative to the robot now: the command sampled them relative
        to the root when the wait started (aim forward / lateral of the root, heights above the ground), and the robot
        may have moved or turned since. The apex is that of the drag-free lob with the throw's flight time."""
        cmd, r, th = self.cmd, self.r, self.th
        th[ids, F["wait_s"]] = (self.ue.episode_length_buf[ids] - self.throw_start[ids]).float() * self.dt
        th[ids, F["turn_wait_deg"]] = torch.rad2deg(wrap(yaw_of(r.data.root_quat_w[ids]) - self.ep_yaw[ids]))
        if not hasattr(cmd, "_target_pos"):
            return
        root = r.data.root_pos_w[ids]
        heading = yaw_quat(r.data.root_quat_w[ids])
        ground = self.ue.scene.env_origins[ids, 2]
        spawn, target, flight = cmd._spawn_pos[ids], cmd._target_pos[ids], cmd._flight_time[ids]
        aim = quat_apply_inverse(heading, target - root)
        release = quat_apply_inverse(heading, spawn - root)
        v0z = (target[:, 2] - spawn[:, 2] + 0.5 * GRAVITY * flight**2) / flight
        th[ids, F["aim_fwd_m"]] = aim[:, 0]
        th[ids, F["aim_lat_m"]] = aim[:, 1]
        th[ids, F["aim_h_m"]] = target[:, 2] - ground
        th[ids, F["release_dist_m"]] = release[:, :2].norm(dim=-1)
        th[ids, F["release_az_deg"]] = torch.rad2deg(torch.atan2(release[:, 1], release[:, 0]))
        th[ids, F["release_h_m"]] = spawn[:, 2] - ground
        th[ids, F["apex_h_m"]] = spawn[:, 2] + v0z.clamp(min=0.0) ** 2 / (2 * GRAVITY) - ground
        th[ids, F["flight_s"]] = flight
        th[ids, F["speed_h_mps"]] = (target[:, :2] - spawn[:, :2]).norm(dim=-1) / flight
        th[ids, F["obs_delay_steps"]] = cmd._obs_delay[ids].float()
        th[ids, F["hidden_in_zone"]] = cmd._hide_in_zone[ids].float()

    def _start_throw(self, ids: torch.Tensor):
        """Reset the per-throw trackers of ``ids`` (new episode, the next throw in a task with several, or a new
        reference cycle in Unified). The command has already drawn the new throw, so its per-throw flags are read here:
        with several throws per episode they are overwritten before the old throw's record is written."""
        r, cmd = self.r, self.cmd
        if hasattr(cmd, "_eval_throw_index"):  # --fixed_throws: the index into the fixed list
            self.th_index[ids] = cmd._eval_throw_index[ids]
        else:
            self.th_index[ids] += 1
        self.th_shown[ids] = cmd._show_parked[ids]
        hidden = cmd._hide_lost[ids]
        self.th_lost_hidden[ids] = hidden if self.hide_rule else torch.zeros_like(hidden)
        self.th_both[ids] = False
        self.th_contact[ids] = float("nan")
        self.th_wait_hand_sum[ids] = 0.0
        self.th_wait_n[ids] = 0
        self.th_wait_above_ref[ids] = float("nan")
        self.max_sec[ids] = 0
        self.ep_confirmed[ids] = False
        self.ep_hold_steps[ids] = 0
        self.ep_xy[ids] = r.data.root_pos_w[ids, :2]
        self.ep_yaw[ids] = yaw_of(r.data.root_quat_w[ids])
        self.throw_start[ids] = self.ue.episode_length_buf[ids]
        self.was_launched[ids] = self.cmd.launched[ids]
        self.th[ids] = float("nan")
        self.th_touch[ids] = False
        self.th_palm[ids] = False
        self.th_gated[ids] = False
        self.th_push_flight[ids] = 0
        self.th_push_max[ids] = 0.0
        self.th_push_catch[ids] = 0
        self.th_push_hold[ids] = 0
        self.pl_step[ids] = -1
        self.pl_dropped[ids] = False
        self.pl_steps[ids] = 0
        self.pl_min_pelvis[ids] = float("inf")
        self.pl_max_tilt[ids] = 0.0
        self.pl_min_hand[ids] = float("inf")
        self.pl_liftoffs[ids] = 0
        self.pl_disp[ids] = 0.0
        self.pl_ball_foot[ids] = float("inf")
        self.pl_end_P[ids] = 0.0

    def write_throws(self, path: str):
        """One row per throw with an outcome (THROW_COLUMNS); empty parameters: the throw was not launched."""
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(THROW_COLUMNS)
            w.writerows(self.throw_rows)

    def throw_summary(self) -> dict:
        """How the lost balls ended (share of all throws with an outcome) and the catch rate with and without a push
        during the flight."""
        rows = [dict(zip(THROW_COLUMNS, x)) for x in self.throw_rows]
        n = max(1, len(rows))
        lost = [x for x in rows if x["outcome"] in LOSSES]
        stages = ("never_touched", "touched", "palm", "held", "confirmed")
        out = {"ball_lost_pct": {s: round(100 * sum(loss_stage(x) == s for x in lost) / n, 2) for s in stages}}
        launched = [x for x in rows if x["launched"]]
        pushed = [x for x in launched if x["pushes_flight"] > 0]
        calm = [x for x in launched if x["pushes_flight"] == 0]

        def caught(xs):
            return round(100 * sum(x["outcome"] == "ball_caught" for x in xs) / len(xs), 2) if xs else None

        out["pushes_during_flight"] = {"pct_throws_pushed": round(100 * len(pushed) / max(1, len(launched)), 1),
                                       "caught_pct_pushed": caught(pushed), "caught_pct_not_pushed": caught(calm)}
        if self.cmd.cfg.gate_max_deviation is not None:
            gated = [x for x in launched if x["gated"]]
            shown = [x for x in launched if not x["gated"]]
            out["gate"] = {"max_deviation_m": self.cmd.cfg.gate_max_deviation,
                           "pct_throws_gated": round(100 * len(gated) / max(1, len(launched)), 1),
                           "caught_pct_gated": caught(gated), "caught_pct_not_gated": caught(shown)}
        post = [x for x in rows if x["lost_as"] != ""]
        if self.after_loss_s is not None and post:
            out["after_loss"] = self._after_loss_stats(post)
            visible = [x for x in post if not x["lost_hidden"]]
            hidden = [x for x in post if x["lost_hidden"]]
            if visible and hidden:  # --hide_lost_ball with a share: both cases on the same robots
                out["after_loss_by_lost_ball"] = {"visible": self._after_loss_stats(visible),
                                                  "hidden": self._after_loss_stats(hidden)}
        shown = [x for x in rows if x["shown_before"]]
        if shown:  # --show_ball_before_throw
            out["ball_before_throw"] = {"shown": before_launch_stats(shown),
                                        "not_shown": before_launch_stats([x for x in rows if not x["shown_before"]])}
        # the first palm contact of caught and of lost balls: did the lost ones arrive faster relative to the hands
        # (a bounce off the palms) or farther from the hold point (an inaccurate interception)?
        palm = [x for x in launched if x["contact_rel_speed_mps"] != ""]
        if palm:
            out["first_palm_contact"] = {}
            for group, xs in (("caught", [x for x in palm if x["outcome"] == "ball_caught"]),
                              ("lost", [x for x in palm if x["outcome"] in LOSSES])):
                out["first_palm_contact"][group] = {
                    "throws": len(xs),
                    "rel_speed_mps_median": quantile_of(xs, "contact_rel_speed_mps", 0.5),
                    "ball_speed_mps_median": quantile_of(xs, "contact_ball_speed_mps", 0.5),
                    "hold_dist_m_median": quantile_of(xs, "contact_hold_dist_m", 0.5),
                    "pct_both_palms": round(100 * sum(x["both_palms"] for x in xs) / len(xs), 1) if xs else None,
                }
        waits = [x for x in rows if x["wait_hand_h_m"] != ""]
        if waits:
            out["waiting_hands"] = {"throws": len(waits),
                                    "wrists_above_pelvis_m_median": quantile_of(waits, "wait_hand_h_m", 0.5)}
            above = [x["wait_hand_above_ref_m"] for x in waits if x["wait_hand_above_ref_m"] != ""]
            if above:  # Unified
                out["waiting_hands"].update({
                    "above_reference_m_median": quantile_of(waits, "wait_hand_above_ref_m", 0.5),
                    "above_reference_m_p90": quantile_of(waits, "wait_hand_above_ref_m", 0.9),
                    "pct_above_reference_over_0.2m": round(100 * sum(v > 0.2 for v in above) / len(above), 1),
                    "pct_above_reference_over_0.3m": round(100 * sum(v > 0.3 for v in above) / len(above), 1),
                })
        return out

    def _after_loss_stats(self, post: list[dict]) -> dict:
        """What the robots did in the windows after the losses of ``post`` (records with post-loss columns)."""
        n = len(post)
        if n == 0:
            return {"window_s": self.after_loss_s, "lost_balls": 0}

        def pct(cond):
            return round(100 * sum(1 for x in post if cond(x)) / n, 1)

        def quantile(key, q, xs):
            v = sorted(x[key] for x in xs)
            return round(v[min(len(v) - 1, int(q * len(v)))], 3) if v else None

        fell = [x for x in post if x["post_loss_fell"]]
        left = [x for x in post if x["post_loss_end"] == "left_reference"]
        stood = [x for x in post if not x["post_loss_fell"]]
        ends = defaultdict(int)
        for x in post:
            ends[x["post_loss_end"]] += 1
        return {
            "window_s": self.after_loss_s,
            "lost_balls": n,
            # ended early without a fall (Advanced: time-out; Unified: also leaving the reference or the clip end)
            "pct_window_cut_by_time_out": pct(lambda x: not x["post_loss_fell"] and x["post_loss_s"] < self.after_loss_s - 0.1),
            "pct_fell": pct(lambda x: x["post_loss_fell"]),
            "fall_after_s_median": quantile("post_loss_s", 0.5, fell),
            "pct_left_reference": pct(lambda x: x["post_loss_end"] == "left_reference"),
            "left_reference_after_s_median": quantile("post_loss_s", 0.5, left),
            "window_end_pct": {k: round(100 * v / n, 1) for k, v in sorted(ends.items(), key=lambda kv: -kv[1])},
            "pct_hand_below_0.5m": pct(lambda x: x["post_loss_min_hand_m"] < 0.5),
            "pct_pelvis_below_0.65m": pct(lambda x: x["post_loss_min_pelvis_m"] < 0.65),
            "pct_torso_tilt_over_30deg": pct(lambda x: x["post_loss_max_tilt_deg"] > 30.0),
            "liftoffs_mean": round(sum(x["post_loss_liftoffs"] for x in post) / n, 2),
            "displacement_m_median": quantile("post_loss_disp_m", 0.5, post),
            "displacement_m_p90": quantile("post_loss_disp_m", 0.9, post),
            "pct_ball_within_0.15m_of_a_foot": pct(lambda x: x["post_loss_ball_foot_m"] < 0.15),
            "posture_P_at_end_mean_if_standing": round(sum(x["post_loss_end_P"] for x in stood) / max(1, len(stood)), 3),
        }

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
            if self.hand_sensors:
                keys.append("hand_contact_N")
            R[f"posture_{tag}"] = {k: round(acc[f"{k}_{tag}"] / n, 3) for k in keys}
            if self.atc is not None:
                R[f"posture_{tag}"]["pct_arm_torso_over_10N"] = round(100 * cnt[f"arm_torso_over_10N_{tag}"] / n, 1)
            if self.hand_sensors:
                R[f"posture_{tag}"]["pct_hand_contact_over_5N"] = round(100 * cnt[f"hand_contact_over_5N_{tag}"] / n, 1)
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
        R.update(self.throw_summary())
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
        + (f", hands on the other arm {R['posture_wait']['hand_contact_N']} / {R['posture_held']['hand_contact_N']} N"
           if "hand_contact_N" in R["posture_wait"] else "")
    )
    if "after_confirmation_pct" in R:
        lines.append(f"[eval] confirmed catches: {R['confirmed_pct']} % of the throws; how they ended: "
                     f"{R['after_confirmation_pct']} (ball_caught = hold phase completed), held "
                     f"{R['hold_after_confirm_s_median']} s after the confirmation (median)")
    if "after_confirmed_catch_pct" in R:
        lines.append(f"[eval] after a confirmed catch: {R['after_confirmed_catch_pct']}")
    lines.append(f"[eval] balls lost, by how far they got (% of the throws): {R['ball_lost_pct']}; pushes during the "
                 f"flight: {R['pushes_during_flight']}")
    if "after_loss" in R:
        lines.append(f"[eval] after a lost ball: {R['after_loss']}")
    if "after_loss_by_lost_ball" in R:
        v, h = R["after_loss_by_lost_ball"]["visible"], R["after_loss_by_lost_ball"]["hidden"]
        lines.append(
            f"[eval] after a lost ball, visible / hidden: {v['lost_balls']} / {h['lost_balls']} losses, fell "
            f"{v.get('pct_fell')} / {h.get('pct_fell')} %, left the reference {v.get('pct_left_reference')} / "
            f"{h.get('pct_left_reference')} %, how the windows ended {v.get('window_end_pct')} / {h.get('window_end_pct')}"
        )
    if "first_palm_contact" in R:
        c, lo = R["first_palm_contact"]["caught"], R["first_palm_contact"]["lost"]
        lines.append(
            f"[eval] first palm contact, caught / lost: {c['throws']} / {lo['throws']} throws, ball relative to the hands "
            f"{c['rel_speed_mps_median']} / {lo['rel_speed_mps_median']} m/s, ball {c['ball_speed_mps_median']} / "
            f"{lo['ball_speed_mps_median']} m/s, from the hold point {c['hold_dist_m_median']} / {lo['hold_dist_m_median']}"
            f" m (medians), both palms on the ball {c['pct_both_palms']} / {lo['pct_both_palms']} %"
        )
    if "waiting_hands" in R:
        lines.append(f"[eval] hands while waiting: {R['waiting_hands']}")
    if "ball_before_throw" in R:
        s, ns = R["ball_before_throw"]["shown"], R["ball_before_throw"]["not_shown"]
        lines.append(
            f"[eval] ball reported before the throw / not: {s['throws']} / {ns['throws']} throws, caught "
            f"{s.get('pct_caught')} / {ns.get('pct_caught')} %, ended before the launch {s.get('pct_ended_before_launch')}"
            f" / {ns.get('pct_ended_before_launch')} % (fell {s.get('pct_fell_before_launch')} / "
            f"{ns.get('pct_fell_before_launch')} %, left the reference {s.get('pct_left_reference_before_launch')} / "
            f"{ns.get('pct_left_reference_before_launch')} %)"
        )
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
    synced = type(env_cfg.commands.ball_throw).__name__ == "SyncedBallCommandCfg"  # Unified
    if args_cli.after_loss_s is not None:
        # a lost ball no longer ends the episode at once: it runs on for after_loss_s and then ends as the loss
        env_cfg.terminations.ball_dropped = DoneTerm(func=ball_lost_for, params={"wait_s": args_cli.after_loss_s})
        env_cfg.terminations.ball_missed = None
        if synced:
            # Unified: the window ends at the clip end at the latest (next cycle); the term comes after ball_dropped,
            # whose loss counter it reads, and counts for the adaptive sampling where the cycle started, as a drop
            env_cfg.terminations.loss_clip_end = DoneTerm(func=loss_at_clip_end)
            motion_cfg = env_cfg.commands.motion
            motion_cfg.start_bin_failure_terms = list(motion_cfg.start_bin_failure_terms) + ["loss_clip_end"]
    if args_cli.hide_lost_ball is not None:
        env_cfg.commands.ball_throw.hide_lost_ball_below = 0.4
        env_cfg.commands.ball_throw.hide_lost_ball_beyond = 3.0
        env_cfg.commands.ball_throw.hide_lost_ball_prob = args_cli.hide_lost_ball
    if args_cli.show_ball_before_throw is not None:
        env_cfg.commands.ball_throw.show_parked_ball_prob = args_cli.show_ball_before_throw
    if args_cli.fixed_throws is not None:
        env_cfg.seed = args_cli.fixed_throws  # the same startup randomization (materials, masses) for every checkpoint
        if synced:
            env_cfg.commands.motion.adaptive_alpha = 0.0  # cycle starts drawn uniformly, not from this policy's failures
    if args_cli.gate_deviation is not None:
        env_cfg.commands.ball_throw.gate_max_deviation = args_cli.gate_deviation
    if getattr(env_cfg.events, "push_robot", None) is not None:
        env_cfg.events.push_robot.func = record_pushes(env_cfg.events.push_robot.func)
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
    agent_cfg = cli_args.for_inference(cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli))
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, version("rsl-rl-lib"))

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    try:  # an open env makes closing the app hang, also after an error
        if args_cli.fixed_throws is not None:
            fix_throws(env.unwrapped, args_cli.fixed_throws)  # before the first reset, which draws the first throws
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
            stats = Stats(env.unwrapped, args_cli.after_loss_s)
            for _ in range(steps):
                obs, _, _, _ = env.step(policy(obs))
                stats.step()
            R = stats.report(args_cli.scenario)
        R.update({"task": args_cli.task, "checkpoint": args_cli.checkpoint, "num_envs": args_cli.num_envs,
                  "steps": steps, "curriculum_progress": args_cli.curriculum_progress,
                  "hold_rules_progress": args_cli.hold_rules_progress, "curriculum_values": curriculum,
                  "after_loss_s": args_cli.after_loss_s, "hide_lost_ball": args_cli.hide_lost_ball,
                  "show_ball_before_throw": args_cli.show_ball_before_throw, "fixed_throws": args_cli.fixed_throws,
                  "no_push": args_cli.no_push})
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
        with open(args_cli.out + ".json", "w") as f:
            json.dump(R, f, indent=1)
        stats.write_throws(args_cli.out + "_throws.csv")
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
