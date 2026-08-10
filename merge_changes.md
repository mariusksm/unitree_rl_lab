# Merge Changes — Unified Ball-Catching Task

Implements the unified single-task architecture from §4.4 of `g1_catching_optimization_plan.md` (roadmap step 11, option b) as a **new, additional task**:

```bash
python scripts/rsl_rl/train.py --task Unitree-G1-29dof-BallCatch-Unified
```

The existing `...-BallCatch-Phase1` / `...-BallCatch-Phase2` tasks are untouched and still runnable — the unified task lives alongside them, so you can A/B the two approaches. Everything builds on the P0/P1 state (`p0_changes.md`, `p1_changes.md`).

New/changed files:

| File | Change |
|---|---|
| `robots/g1_29dof/unified/ball_catch_env_cfg.py` | **New** — the unified environment |
| `robots/g1_29dof/unified/__init__.py` | **New** — gym registration |
| `mdp/commands/synced_ball_command.py` | **New** — `SyncedBallCommand`: ball launch synchronized to the motion phase |
| `mdp/commands/ball_command.py` | Behavior-neutral refactor only: extracted `_sample_spawn()` / `_update_catch_state()`, optional `flight_time` arg on `_launch()` — so the synced variant reuses them instead of duplicating. Phase 2 behavior is unchanged. |
| `mdp/commands/__init__.py` | Export the new command (+ `BallCommand` itself) |

Validation done: all files pass `py_compile`; the phase tasks' code paths are untouched except the behavior-neutral refactor. Smoke-test both the unified and Phase 2 tasks in the Isaac Sim environment before long runs.

---

## 1. Why merge — and what the merge actually is

The two-phase design trains "how to move" (mimic) and "when/where to catch" (RL) **sequentially**, connected by a fragile transfer contract: identical observation layouts padded with `dummy_zeros`, identical action scaling, identical obs scales, plus a low-noise fine-tuning config so PPO doesn't erase the prior. P0/P1 repaired that contract, but it remains a contract — every future change must be mirrored twice, and nothing during Phase 2 *anchors* the motion prior: the tracking rewards are gone, so the policy is free to forget the catching form as long as the ball rewards are satisfied.

The unified task trains both **simultaneously**, DeepMimic-style:

> **Track the reference catching motion while a real physics ball is thrown such that it arrives exactly when the mocap catch happens.**

- The **mocap supplies the "how"**: whole-body form, weight shift, arm trajectory, retract — enforced every step by the tracking rewards.
- **RL supplies the "when/where"**: the ball never flies exactly like the mocap take (randomized spawn, aim scatter, velocity noise, ball physics DR), so the policy must deviate from the reference just enough to intercept — and the curriculum progressively widens that gap.
- **No transfer step exists at all**: no checkpoint surgery, no placeholder observations, no forgetting, one training run.

## 2. The synchronization mechanism (`SyncedBallCommand`)

This is the one genuinely new piece of machinery. It subclasses the P0 `BallCommand` (parked-ball spawning, ballistic launch, `secured_steps` catch tracking all reused) and replaces the wall-clock launch timer with a **motion-phase trigger**:

```
launch_time = catch_motion_time + jitter(±0.05 s) − flight_time
```

where `catch_motion_time` is the moment (seconds into the clip) the mocap hands contact the ball, and `flight_time` is pre-sampled per throw. Launching at `launch_time` makes ball arrival coincide with the mocap catch pose — the reference motion and the physics event line up by construction.

Per motion cycle:
- **Before `launch_time`**: ball parked at its spawn point (the visible "thrower" position, sampled in the robot's heading frame — all P0 geometry unchanged).
- **At `launch_time`**: ballistic launch toward the chest-height catch zone with the pre-sampled flight time.
- **Episode starts past the launch window**: the `MotionCommand`'s adaptive sampling can start an episode anywhere in the clip, including after the catch. Then no throw happens this cycle and the episode trains **pure tracking** — which is exactly how the *retract/recover* phase of the motion gets trained (see §4 on why catch success terminates the episode).
- **Motion wraps or is resampled mid-episode** (the `MotionCommand` teleports the robot at that moment): detected as a backward jump of the motion time → the ball is re-parked at a fresh spawn and re-armed, so a stale ball never flies at a teleported robot. Long episodes therefore contain **multiple throws**, one per motion cycle.

Two wiring requirements, both enforced/documented in the config:
1. **`motion` must be declared before `ball_throw`** in `CommandsCfg` — command terms reset in declaration order, so the ball arms against the freshly resampled motion time on env reset.
2. **`CATCH_MOTION_TIME` must be set from your actual mocap** (module constant in the env cfg, currently a `2.0 s` placeholder). Read it off the retargeted clip with `replay_npz.py`. The command validates it against the clip length at startup and warns loudly if the ball could never launch.

## 3. How the two phases map into the unified config

| Component | Taken from | Notes |
|---|---|---|
| Scene | Phase 1 + Phase 2 | Phase 1 scene ∪ the ball rigid object |
| `motion` command | Phase 1 | Identical (same motion file auto-discovery from `phase1/motions/`, same anchor/bodies/adaptive sampling) |
| `ball_throw` command | Phase 2 (P0) | `SyncedBallCommandCfg` — same throw geometry/catch detection, motion-driven launch |
| Actions | both | `MIMIC_ACTION_SCALE` (identical in both phases since P0) |
| Observations | **union, no placeholders** | Every former `dummy_zeros` slot now carries its real value on both the mimic and ball side |
| Rewards | Phase 1 tracking + Phase 2 ball + Phase 1 regularization | Phase 2's posture crutches **dropped** (see §4) |
| Events | P1-mirrored DR union | Robot DR identical to both phases; ball DR from Phase 2; **no robot reset events** — the `MotionCommand` owns resets (reference-state init) |
| Terminations | Phase 1 tracking + Phase 2 ball | `ee_body_pos` threshold loosened 0.25 → 0.3 (see §4) |
| Curriculum | P1's throw widening + **new tracking tapers** | Phase 2's posture tapers dropped with the posture rewards |
| PPO runner | `BasePPORunnerCfg` | Full exploration (std 1.0) — this is a from-scratch task; no fine-tuning config needed |

**Observation layout** is intentionally the *same 174-dim frame* as the P1-era phase contract, with real values in every slot: `motion_command(58) | ball_state(6) | ball_relative(3) | hand_pos(6) | ball_intercept(3) | motion_anchor_ori_b(6) | proprio(...)` — likewise for the critic. Consequence: **Phase 1 and Phase 2 checkpoints are shape-compatible with the unified task**, so a trained Phase 1 policy can warm-start unified training (`--resume` + `--agent.policy.init_noise_std=0.3`) even though nothing requires it.

## 4. Design decisions that make the merge work (the "optimize" part)

**a) Posture crutches removed.** Phase 2 needed `base_height` (−10), `flat_orientation_l2` (−5), `joint_deviation_*` and `alive` because, without a motion prior, PPO's first discovery is falling over. In the unified task the tracking rewards *are* the posture prior — and unlike the crutches, they encode the crouch/lean/reach of the actual catch instead of fighting it (finding F-11). Dropping them removes an entire class of reward conflicts; their taper curriculum goes with them.

**b) Catch success still terminates the episode.** Kept from P0 for two reasons. Credit assignment stays clean (one throw → one outcome → one-time bonus of 10). And without dexterous hands, the "held" ball drops the moment the reference motion retracts the arms — continuing the episode would immediately punish a successful catch with a drop penalty. The retract phase still gets trained: adaptive sampling starts episodes past the catch frame, where no throw happens and the policy purely tracks the retract motion. When the Dex3 hands land (roadmap P2), this termination can be revisited so catch-and-hold-through-retract becomes one continuous behavior.

**c) Failed catches steer the adaptive sampler.** `ball_dropped` is a non-timeout termination, so the `MotionCommand`'s failure-bin statistics count every drop at the motion phase where it happened — around the catch frame. The existing BeyondMimic machinery therefore **automatically oversamples the catch window** as long as catching keeps failing. The two halves of the task share one difficulty scheduler for free.

**d) Tracking leash loosens on a curriculum, not the throws alone.** P1's throw-widening curriculum is kept (easy 50k → final 250k steps). Added: `motion_body_pos/ori` taper 1.0 → 0.4 and body-velocity tapers 1.0 → 0.5 over steps 150k–500k, so early training clamps the policy to the reference (fast, safe skill acquisition) and late training frees it to deviate for off-nominal balls. The **anchor** tracking terms stay at full weight throughout — they keep the robot standing in place; only relative body tracking is released. The `ee_body_pos` termination is loosened 0.25 → 0.3 m for the same reason: reaching for a lateral ball legitimately pulls a wrist off the reference, and terminating that attempt would teach the policy to never reach.

**e) Pushes moderated.** Phase 1's aggressive 1–3 s pushes build recovery skills but a shove mid-ball-flight turns the catch reward into noise; Phase 2's 4–8 s xy-pushes are too tame for the dynamic motion. Compromise: full 6-DOF push range at 2–5 s intervals.

**f) Throw geometry and catch detection are unchanged from P0/P1.** Yaw-frame spawn/aim, chest-height catch zone, `secured_steps` (proximity + relative velocity + duration), yaw-frame ball observations, and the ballistic `ball_intercept` observation all carry over — the merge changes *when* the ball launches, not *how* it flies or how catching is detected.

## 5. What you must do before training this task seriously

1. **Record the catching mocap** and drop the `.npz` into `phase1/motions/` (shared by Phase 1 and the unified task — single source of truth; pipeline in `motions/README.md`).
2. **Set `CATCH_MOTION_TIME`** in `unified/ball_catch_env_cfg.py` to the ball-contact moment of that clip (via `replay_npz.py`). With the placeholder motion and the placeholder 2.0 s, the task runs but "catches" against a dance move — the startup warnings will remind you.
3. Check that `catch_height_range` (0.25–0.45 above root) matches the mocap's wrist height at the catch, per the README checklist.

## 6. Training notes

- From scratch: `--task Unitree-G1-29dof-BallCatch-Unified` (default runner, std 1.0). Optional warm start from a Phase 1 checkpoint: `--resume --load_run <phase1_run> --agent.policy.init_noise_std=0.3`.
- Watch in TensorBoard: the Phase-1-style tracking metrics (`Metrics/motion/error_*`, `sampling_*`) **and** the ball metrics (`Metrics/ball_throw/ball_secured_time`, `hand_ball_distance`, `Episode_Termination/ball_caught` vs `ball_dropped`, `Curriculum/*`). Healthy run: tracking errors fall first, catch rate climbs during the easy-throw stage, dips and recovers as throws widen and the tracking tapers release.
- `sampling_top1_bin` concentrating near the catch frame is the adaptive sampler doing its job (§4c), not a bug.
- If tracking collapses once the ball rewards kick in (policy "abandons the dance to chase the ball"), the intended tuning knobs are: raise `motion_body_*` weights or delay their taper (`start_step`), before touching the ball rewards.
- Episodes are up to 20 s with one throw per motion cycle; per-throw outcome statistics come from the termination counters, since a caught/dropped ball ends the episode.

## 7. Explicitly not done

- The phase tasks were **not** removed or modified (beyond the behavior-neutral `BallCommand` refactor) — decide after A/B-ing which line to keep.
- No hands (roadmap P2), no ball-obs noise/latency/dropout, no deployment work (P3).
- No new PPO hyperparameters — the unified task reuses `BasePPORunnerCfg`.
