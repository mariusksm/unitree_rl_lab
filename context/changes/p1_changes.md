# P1 Changes — Content + Transfer Integrity

Implements exactly the **P1 steps (10–13)** of `g1_catching_optimization_plan.md`, on top of the P0 fixes (`p0_changes.md`). The two-phase structure remains — per your instruction and the architecture decision below, the phases are **not** merged. Nothing from P2 (hands) or P3 (sim-to-real hardening) was touched.

Files touched:

| File | Change |
|---|---|
| `phase1/ball_catch_env_cfg.py` | Auto-discovering motion-file wiring, `ball_intercept` placeholder obs (×2), mirrored `add_base_mass` DR |
| `phase1/motions/README.md` | **New** — drop-in directory for the catching mocap + pipeline & quality checklist |
| `phase2/ball_catch_env_cfg.py` | `ball_intercept` obs (policy + critic), mirrored DR (`add_joint_default_pos`, `base_com`, material ranges), `CurriculumCfg` (throw widening + reward tapers), play-cfg curriculum override |
| `mdp/observations.py` | **New** `ball_intercept_b` — ballistic interception prediction |
| `mdp/curriculums.py` | **New** — `modify_reward_weight_linear`, `ball_throw_curriculum` |
| `mdp/__init__.py` | Export curriculums |
| `agents/rsl_rl_ppo_cfg.py` | **New** `Phase2PPORunnerCfg` (fine-tuning, `init_noise_std=0.2`) |
| `phase2/__init__.py` | Phase 2 registers `Phase2PPORunnerCfg` |

Validation done: all edited files pass `py_compile`; observation slot positions verified identical across both phases and both obs groups. As with P0, run a short smoke test in the Isaac Sim environment before a real run.

---

## Step 10 — Catching motion content (P1-1)

**What I could not do:** record the mocap. Capturing the catching motion with VICON Shogun and running the GMR retarget is a physical-world task that stays with you — **this is now the only blocker for meaningful Phase 1 training.**

**What is done code-side** so the recording drops in with zero code changes:

- New directory `phase1/motions/` — Phase 1 automatically picks up the first `*.npz` placed there (glob at import time). No more hardcoded path into the dance task's folder.
- Until the file exists, the config falls back to the Gangnam placeholder **with a loud console warning** ("the policy will imitate dancing, not catching!") so the placeholder can never silently make it into a serious run again. Multiple files also warn and use the alphabetically first.
- `motions/README.md` documents the exact pipeline (VICON → GMR `.pkl` → `pkl_to_csv_without_hands.py` → `csv_to_npz.py` → `replay_npz.py` verification) and a quality checklist. The most important item: **verify the mocap's wrist height during the catch against the Phase 2 aim point** (`catch_height_range = (0.25, 0.45)` above root ≈ 1.0–1.2 m world) — if they disagree, adjust the aim to the motion, not the motion to the aim.

## Step 11 — Phase architecture decision + transfer integrity

**Decision: keep the two-phase transfer setup** (option a). You explicitly ruled out merging for now; the unified-task option from §4.4 of the plan remains open as a future refactor. With that decision, the two fine-tuning safeguards named in the plan are implemented:

**a) Fine-tuning PPO config.** New `Phase2PPORunnerCfg` with `init_noise_std = 0.2` (was 1.0), registered as Phase 2's default runner config. With std 1.0, PPO's initial exploration noise commands near-full-range random joint offsets and destroys the transferred motor skills within the first iterations; 0.2 explores around the prior instead. The docstring and this note flag the flip side: **when training Phase 2 from scratch** (debugging, baselines), pass `--agent.policy.init_noise_std=1.0`, otherwise exploration is too timid. Phase 1 keeps `BasePPORunnerCfg` unchanged. Experiment names still auto-derive from the task name (`cli_args.py` fills empty `experiment_name`), so log directories are unaffected.

**b) DR sets mirrored (F-12).** Both phases now randomize the same robot properties:

| Event | Phase 1 | Phase 2 |
|---|---|---|
| `physics_material` (robot) | 0.3–1.6 / 0.3–1.2 / rest. 0–0.5 | **unified to the same (wider) ranges** — was 0.3–1.0 / 0.3–1.0 / rest. 0 |
| `add_joint_default_pos` (±0.01, calibration error) | already present | **added** |
| `base_com` (torso CoM shift) | already present | **added** |
| `add_base_mass` (torso −1…+3 kg) | **added** | already present |

Robot restitution randomization in Phase 2 also matters physically now: the ball bounces off arms/torso, so training across restitution 0–0.5 hardens the catch against the real shell's unknown bounciness. Deliberately *not* mirrored: the push events (P1's aggressive 6-DOF pushes at 1–3 s are part of the BeyondMimic recovery recipe; P2 keeps gentler xy pushes at 4–8 s) and the reset events (P1 resets via the motion command, P2 to standing) — these are task semantics, not dynamics DR.

## Step 12 — Flight-phase observation (F-9): predicted interception point + time-to-go

New observation `ball_intercept_b` (`mdp/observations.py`), added to the Phase 2 policy **and** critic, with matching `dummy_zeros(3)` placeholders in both Phase 1 groups (slot inserted directly after `hand_pos` everywhere — positions verified identical, so the cross-phase contract holds).

What it computes, per env: the descending crossing of the ball's ballistic trajectory `z(t) = z₀ + v_z t − ½gt²` with the horizontal catch plane 0.35 m above the robot root (the middle of the aim range), i.e. `t_go = (v_z + √(v_z² + 2g(z₀ − z_plane)))/g`, and the ball's xy position at that time relative to the root, rotated into the yaw frame. Output: `[intercept_x_b, intercept_y_b, t_go]` (3 dims).

- **Why this instead of raw `time_since_throw`:** "where will the ball be, and how long do I have" is the essence of catching; giving it explicitly saves the network from learning ballistics from 5 frames of history, and it is exactly what the deployment pipeline will compute from the filtered VICON stream (plan §5, S-2/S-3) — sim obs and real obs stay the same math. Air drag on the 0.15 kg ball at < 8 m/s is negligible.
- Edge cases are smooth: while the ball is parked pre-throw, the output is the "if it fell now" point (a meaningful, continuous signal); if the trajectory never reaches the plane, `t_go = 0` and the current position is returned; `t_go` is clamped to [0, 3] s.
- **Checkpoint compatibility:** this adds 3 dims per frame to both obs groups — **P0-era checkpoints are incompatible.** Since Phase 1 must be retrained anyway (P0 obs-scale change) and no Phase 2 logs exist, nothing of value is lost.

## Step 13 — Curriculum (throw widening + posture-crutch tapering, F-11)

New `mdp/curriculums.py` with two generic, logged curriculum functions, wired into a new Phase 2 `CurriculumCfg` (was `curriculum = None`):

**a) `ball_throw_curriculum`** — linearly interpolates `BallCommandCfg` fields from an easy to the final distribution over `common_step_counter` 50k → 250k (≈ iterations 2k → 10.4k at 24 steps/iter):

| Field | Easy (start) | Final (= P0 cfg defaults) |
|---|---|---|
| `throw_distance_range` | (1.2, 1.8) m | (1.0, 2.5) m |
| `throw_lateral_range` | ±0.3 m | ±0.75 m |
| `flight_time_range` | (0.5, 0.7) s — slower balls | (0.3, 0.6) s |
| `velocity_noise` | 0.1 m/s | 0.3 m/s |
| `catch_lateral_std` | 0.02 m | 0.05 m |

New throws sample the updated cfg immediately; progress is logged as `Curriculum/ball_throws` in TensorBoard.

**b) `modify_reward_weight_linear`** — tapers the posture crutches over steps 100k → 400k so the policy first learns stable standing under them, then gains the freedom to crouch/lean/reach that catching needs:

| Reward | Start | End |
|---|---|---|
| `base_height` | −10.0 | −2.5 |
| `flat_orientation_l2` | −5.0 | −1.0 |
| `joint_deviation_arms` | −0.1 | −0.02 |

Current weights are logged as `Curriculum/<term>_taper`. (Isaac Lab's built-in `modify_reward_weight` is a one-shot switch; these are linear anneals, hence the custom functions.)

Notes:
- Schedules assume roughly the default 30k-iteration run (720k steps); if you train much shorter/longer, scale `start_step`/`end_step` accordingly — they are plain params in `CurriculumCfg`.
- Curriculum terms update on env resets; with the short single-throw episodes that is effectively continuous.
- **`RobotPlayEnvCfgPhase2` sets `curriculum = None`** so play/evaluation always runs the *final* throw distribution and reward weights instead of the step-0 easy stage (a fresh play env has `common_step_counter = 0` and would otherwise silently evaluate on easy throws).
- Phase 1 keeps `curriculum = None` — the plan's step 13 targets the catching task; the mimic task's own difficulty scheduling is already handled by the adaptive failure-bin sampling in `MotionCommand`.

---

## Not changed (explicitly out of P1 scope)

- No phase merge (unified DeepMimic-style task stays a documented future option).
- No noise/latency/dropout on ball observations (P3, step 18) — `ball_intercept_b` is currently computed from clean sim state, like the other ball terms.
- No hands (P2 steps), no deployment pipeline work (P3 steps 19–20).
- Phase 2 reward *structure* is untouched — only weights are annealed by the curriculum.

## Training notes

- **Recommended flow now:** record catching mocap → drop `.npz` into `phase1/motions/` → train Phase 1 (`Unitree-G1-29dof-BallCatch-Phase1`) → fine-tune Phase 2 from that checkpoint (`--resume`, with the new low-noise runner active by default).
- For a Phase 2 **from-scratch** debug run: add `--agent.policy.init_noise_std=1.0`.
- New TensorBoard signals to watch: `Curriculum/ball_throws` (0→1), `Curriculum/*_taper` (weights annealing), plus the P0 metrics (`Metrics/ball_throw/ball_secured_time`, `Episode_Termination/ball_caught`). A healthy run shows the catch rate climbing during the easy stage and dipping-then-recovering as the throw distribution widens after step 50k.
- Until the real mocap exists, every training start prints the placeholder warning — that is intentional nagging.
