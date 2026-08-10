# P0 Changes — Fix Training-Breaking Defects

Implements exactly the **P0 steps (1–9)** of `g1_catching_optimization_plan.md`. The two-phase structure (Phase 1 mimic / Phase 2 RL) is **unchanged** — no phase merging. The cross-phase observation-shape contract is preserved: all slots keep their dimensions (`motion_command` 58, `ball_state` 6, `ball_relative` 3, `hand_pos` 6, critic placeholders 3/6/42/84).

Files touched:

| File | Change |
|---|---|
| `mdp/commands/ball_command.py` | Rewritten: yaw-frame throws, corrected aim height, single-throw contract, throw delay, catch-state tracking, debug code removed |
| `mdp/observations.py` | World-frame ball/hand obs replaced by robot-yaw-frame versions with cached body indices |
| `mdp/rewards.py` | `ball_caught` replaced by `ball_caught_bonus` + `ball_secured`; `hand_to_ball_distance_exp` now uses cached body ids |
| `mdp/terminations.py` | New success termination `ball_caught` |
| `mdp/__init__.py` | Removed self-import and duplicate `.commands` import |
| `phase2/ball_catch_env_cfg.py` | Action scale, obs terms, rewards, terminations, contact regex, push interval |
| `phase1/ball_catch_env_cfg.py` | Obs scales/noise unified with Phase 2 |

Validation done: all edited files pass `py_compile`; no stale references to removed functions remain. A full env-instantiation smoke test requires the Isaac Sim python environment (run `scripts/rsl_rl/train.py --task Unitree-G1-29dof-BallCatch-Phase2 --num_envs 16 --max_iterations 2` before a real run).

---

## 1. Robot-yaw-frame ball & hand observations (F-1, F-3)

**Problem:** `ball_state` observed `root_pos_w` — absolute world coordinates including the per-env origin (±80 m across the 4096-env grid), unnormalized. `ball_relative` and `hand_pos` were root-relative but in **world axes**, while the robot's yaw is randomized over ±π and unobservable.

**Change** (`observations.py`):
- `ball_pos_vel_b(env)` → 6 dims: ball position relative to the robot root **and** ball linear velocity, both rotated into the robot's yaw frame via `quat_apply_inverse(yaw_quat(root_quat_w), ·)`. Replaces `generated_commands("ball_throw")` in the Phase 2 policy **and** critic. This is exactly the frame you can reproduce on hardware from VICON (ball marker + robot-base tracking), so sim obs = deploy obs.
- `ball_to_hands_b(env, asset_cfg)` → 3 dims: vector from the **midpoint of the two wrists** to the ball, in the yaw frame. Replaces `ball_pos_relative` in the same slot. (A plain root-to-ball vector would have duplicated the first 3 dims of `ball_pos_vel_b`; the hand-midpoint vector fills the 3-dim slot with a genuinely useful reaching signal instead.)
- `hand_pos_b(env, asset_cfg)` → 6 dims: wrist positions relative to root, yaw frame. Replaces `hand_body_pos`.
- The old world-frame functions `ball_pos_relative` / `hand_body_pos` were **deleted** so they can't be reused by accident.
- All three take a `SceneEntityCfg` whose `body_names` are resolved **once** by the observation manager — no more per-step `find_bodies` regex matching (part of F-13).

`BallCommand.command` still exists (world-frame pos+vel, 6 dims) because a `CommandTerm` must expose a command; its docstring now states it is for logging/debug only and must not be used as an observation.

## 2. Throws in the robot's heading frame (F-4)

**Problem:** the ball always spawned toward world +x while `reset_base` randomizes yaw over ±π — robots frequently faced away from the throw.

**Change** (`ball_command.py`): spawn offset and catch-zone offset are sampled in a local frame and rotated by `yaw_quat(robot.root_quat_w)` before being added to the robot position. Every throw now comes from "in front of the robot" regardless of its heading. Yaw randomization stays — it is the defense against the VICON-frame/world-frame mismatch on hardware.

## 3. Catch-zone aim lowered to chest height (F-8)

**Problem:** the aim point was `root_z + 1.0` ≈ 1.76 m world — above the head of the ~1.32 m G1; many throws were physically uncatchable.

**Change:** `catch_height_range = (0.25, 0.45)` above the root (≈ 1.0–1.2 m world, chest height). All throw parameters are now `BallCommandCfg` fields (`throw_distance_range`, `throw_lateral_range`, `throw_height_range`, `catch_forward`, `catch_lateral_std`, `catch_height_range`, `flight_time_range`, `velocity_noise`) so the P1-roadmap curriculum can widen them without code changes.

## 4. Phase 2 action scale = Phase 1 action scale (F-2)

**Problem:** Phase 1 used the per-joint `UNITREE_G1_29DOF_MIMIC_ACTION_SCALE` dict, Phase 2 a uniform `0.25` — a transferred network commanded differently scaled joint targets, scrambling the mimic prior at load time.

**Change** (`phase2/ball_catch_env_cfg.py`): `JointPositionAction.scale = UNITREE_G1_29DOF_MIMIC_ACTION_SCALE`, with a comment marking it as part of the transfer contract. Any deploy config for Phase 2 must use the same per-joint scale.

## 5. Observation scales/noise unified across phases (F-10)

**Problem:** `base_ang_vel` (P1: unscaled / P2: ×0.2) and `joint_vel_rel` (P1: unscaled, noise ±0.5 / P2: ×0.05, noise ±1.5) had 5–20× different magnitudes for the same physical quantity — another silent transfer breaker.

**Change** (`phase1/ball_catch_env_cfg.py`): Phase 1 adopts the Phase 2 convention in both groups — policy: `base_ang_vel scale=0.2, noise ±0.2`, `joint_vel_rel scale=0.05, noise ±1.5`; critic: `base_ang_vel scale=0.2`, `joint_vel scale=0.05`. (Note Isaac Lab applies noise **before** scale, so the P2 noise settings were already effectively milder than P1's — the unified values are the P2 ones.)

## 6. `undesired_contacts` no longer punishes catching (F-5)

**Problem:** the Phase 2 regex excluded only ankles; ball-on-wrist contact — the goal of the task — cost −1 per step.

**Change:** Phase 2 now uses the same regex as Phase 1, excluding both ankle-roll and wrist-yaw links:
`^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$`

## 7. Single-throw episode contract (F-7)

**Problem:** `BallCommand` self-resampled (re-threw) at thresholds *tighter* than the termination thresholds (`z<0.2` vs. `z<0.1`; `dist>5` vs. `dist>6`) and the command update runs after the termination check in Isaac Lab's step loop — so `ball_missed` could never fire and `ball_dropped` was a per-drop coin flip. Episode semantics were effectively random.

**Change:** the ball is thrown **once per episode**; `BallCommand` no longer self-resamples (the drop/far/timeout re-throw logic and the `min_height_throw`/`max_distance`/`max_flight_time` cfg fields are gone). The termination manager now owns all outcomes — each episode ends in exactly one of:

| Outcome | Termination | Threshold |
|---|---|---|
| Caught | `ball_caught` (new) | ball held for `secure_steps` = 25 steps (0.5 s) |
| Dropped | `ball_dropped` | ball z < 0.1 m (ground contact: ball center rests at 0.05 m) |
| Missed | `ball_missed` | horizontal distance > **4.0 m** (was 6.0; max spawn distance is ~2.6 m, so 4.0 fires reliably after a fly-by instead of never) |
| Robot fell | `base_height` / `bad_orientation` | unchanged |
| Timeout | `time_out` | unchanged (20 s) |

**Added: launch delay.** With a single throw per episode, the ball previously would have launched at t≈0 — 0.3–0.6 s after reset, before the robot could stabilize. `BallCommand` now parks the ball at its spawn point (re-written with zero velocity each control step; residual observable sag/velocity from gravity between writes is ≤ 2 mm / 0.2 m/s) for `throw_delay_range = (0.5, 1.5)` s, then launches. The catch zone and ballistic velocity are computed **at launch time** from the robot's current pose, so the aim stays honest if the robot drifted during the delay. The policy sees the parked ball (zero velocity) and the transition to flight — a clean, learnable "throw incoming" signal that mirrors the real setup (thrower stands visible in front of the robot).

## 8. Catch detection redesigned (F-6)

**Problem:** `ball_caught` = `‖v_world‖ < 0.5 ∧ z > 0.5` — no proximity condition. A slow ball at its flight apex or resting on the head earned +10 **per step**; the dominant strategy was trap-and-farm.

**Change:** `BallCommand` now tracks a per-env `secured_steps` counter, incremented each step where **all** hold:
- min wrist-to-ball distance < `catch_radius` (0.2 m),
- ball speed **relative to the closest wrist** < `secure_rel_vel` (0.5 m/s) — relative, not absolute, so moving catches remain valid,
- ball z > `secure_min_height` (0.5 m), and the ball has been launched;

and reset to 0 otherwise. On top of it:
- **Reward `catch_success`** → `ball_caught_bonus`: fires exactly once, on the step `secured_steps == secure_steps` (the same step the success termination triggers, after which the counter resets). Weight 500 → effective one-time bonus of **10** (the reward manager multiplies by `weight × dt`, dt = 0.02 s).
- **Reward `ball_secured_hold`** (new): 0.1/step while the ball is held, hard-capped at 25 steps (≤ 2.5 total) by the success termination — no farming possible.
- **Termination `ball_caught`**: ends the episode on a confirmed catch, giving clean credit assignment (the bonus clearly dominates the ~0.3 discounted future `alive` reward at γ=0.99, so terminating-by-catching is strictly attractive).
- `hand_to_ball` shaping kept, with `std` widened 0.15 → **0.3** so the Gaussian gradient reaches ~0.8 m instead of dying beyond ~0.4 m from the ball (the hands start ~0.5–1.0 m from the ball's approach path; with std 0.15 the shaping signal was numerically zero exactly where it was needed).
- `catch_radius=0.2` is deliberately generous for the hand-less wrist-trap baseline; tighten it when the Dex3 palms exist (the `hand_body_names` cfg field is where palm links will be swapped in).

## 9. Hygiene (F-13)

- **Cached body indices**: `find_bodies` regex resolution now happens once (command `__init__`, manager-resolved `SceneEntityCfg`s) instead of 3× per step.
- **Debug block removed** from `BallCommand._update_metrics` (the ~30-line print block incl. its own `find_bodies` call and `termination_manager` poking); replaced by proper logged metrics: `ball_height`, `ball_secured_time`, `hand_ball_distance` — visible in TensorBoard via the command-manager metrics, so per-env catch behavior is observable without prints.
- **`mdp/__init__.py`**: removed the package importing *itself* and the duplicated `from .commands import *`.
- **`push_robot` interval** widened from the degenerate `(5.0, 5.0)` (all envs pushed in lockstep) to `(4.0, 8.0)`.
- **Removed** unused `import math` and the obsolete `__del__` stub.
- **Contact sensor narrowing — evaluated, deliberately not done**: `undesired_contacts` penalizes essentially every non-foot/non-wrist body, so the sensor must keep full-body coverage (`Robot/.*`). Narrowing it would silently disable that penalty. Revisit only if the sensor shows up in profiling.

---

## Not changed (explicitly out of P0 scope)

- **No phase merge** — Phase 1 and Phase 2 remain separate tasks with the `dummy_zeros` shape contract (per your instruction; the unified-task decision is roadmap step 11).
- No noise/latency/dropout on ball observations yet (roadmap steps 12/18) — note the policy still sees clean ball state.
- No curriculum, no posture-reward tapering (step 13), no mocap replacement (step 10), no hand integration (P2 steps), and Phase 1 still points at the Gangnam placeholder motion.
- Phase 1's mimic logic, rewards, and terminations are untouched apart from the obs scale unification.

## Retraining notes

- **Phase 1 checkpoints trained before this change are incompatible** with the new obs scaling (step 5) — retrain Phase 1 before transferring.
- Expected Phase 2 behavior change: episodes are now short (~1.5–4 s: delay + flight + outcome) instead of fixed 20 s multi-throw loops. Sanity-check in TensorBoard: `Episode_Termination/ball_dropped` and `ball_missed` should dominate early; `ball_caught` should appear and climb as training progresses; `Metrics/ball_throw/ball_secured_time` > 0 signals first successful holds. If `time_out` dominates, something is off (e.g., the ball resting on the robot's body without securing).
- When transferring Phase 1 → Phase 2, also lower `init_noise_std` (1.0 → ~0.2–0.3) for the fine-tuning run — not part of P0 (shared PPO cfg would affect Phase 1 training), flagged here so it isn't forgotten.
