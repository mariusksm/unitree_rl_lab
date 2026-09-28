# unitree_rl_lab: Specification & Current State

> **Purpose of this document:** A fast, reliable entry point for humans and AI assistants.
> It describes *what* this repository does within the H-ReACT project group, *how* it is structured,
> and *where* it currently stands.
> **Maintenance rule:** For every substantial change, first update the affected sections, then append an
> entry to the [Changelog](#14-changelog) at the end (see `CLAUDE.md`).
>
> Last updated: **2026-09-28**. Paths are relative to the repo root. `BC/` is shorthand for
> `source/unitree_rl_lab/unitree_rl_lab/tasks/ball_catching/`.

---

## 1. Project Context

- **Project group H-ReACT** (Humanoid Reactive Action & Control Training), TU Dortmund and Fraunhofer IML,
  summer term 2026 to winter term 2026/27.
- **Goal:** A **Unitree G1 (29 DoF)** catches a ball thrown by a human. The whole-body controller is trained
  with RL in **Isaac Lab**; the reference comes from human mocap data (VICON Shogun, Manus gloves). The ball
  position is **ground truth** (from Isaac in simulation, from VICON on the real setup).
- **Minimum goals per the proposal:**
  1. A policy catches repeatedly in Isaac Sim using GT ball position.
  2. Deployment on the real G1 with a **success rate above 50 %**.
  3. Final report.
  4. Oral examination.
- **Hands:** Per the proposal, the 5-finger hand is replaced by a **static, 3D-printed mount**. The robot model
  in this repo has **no hands**; catching uses the `*_wrist_yaw_link` bodies.
- This repo is a fork of [unitreerobotics/unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab)
  (Isaac Lab 2.3.0 / Isaac Sim 5.1.0, RSL-RL PPO). Almost all project-specific work lives in `BC/`,
  `scripts/mimic/pkl_to_npz.py` and `scripts/pkl_to_csv_without_hands.py`.
- Other parts of the project group work in **other repos** (Fraunhofer GitLab), for example:
  - mocap recording
  - retargeting with GMR and SOMA
  - event camera
  - LocoMuJoCo and MuJoCo

  Background material (proposal, meeting minutes, plans) lives in `context/` (see [Section 12](#12-context-documents)).

## 2. Status at a Glance (2026-09-28)

| Area | Status |
|---|---|
| Motion conversion (.pkl → .npz) | ✅ Fixed on 2026-09-28: the right arm was filled with left-hand finger joints (wrong GMR column selection). **All `.npz` created before that date were corrupt** and have been deleted; verify with `scripts/mimic/check_motion.py`. The training clip in `phase1/motions/` has been regenerated (see Section 5). |
| Phase 1 (mimic) on real mocap data | ⚠️ Run `2026-08-31_15-06-41` (1657 iterations, clip `143_merged_filtered.npz`) converged, but on the **corrupt clip** (right arm frozen inside the torso). Must be retrained on a regenerated clip; do not use `model_1500.pt` for transfer. |
| Phase 2 (RL catching, transfer from Phase 1) | ⚠️ Code is in place (P0/P1 fixes), but **no training run** since the fixes (log folder is empty). |
| Unified task (mimic + catching simultaneously) | ⚠️ Code is in place, **never trained**. `CATCH_MOTION_TIME = 2.0` is a **placeholder**. |
| Catch success in simulation | ❌ not yet demonstrated |
| Hands (Dex3 / mount) in the model | ❌ not present (intentional, see proposal) |
| Noise / latency / dropout on ball observations | ❌ not implemented |
| Sim2Sim (MuJoCo), deployment, live VICON pipeline | ❌ not started for the ball task (`deploy/` is unmodified Unitree upstream) |

## 3. Repository Layout (relevant parts)

```
CLAUDE.md                              Instructions for AI assistants (incl. spec maintenance rule)
unitree_rl_lab.sh                      Launcher: -i install, -l list, -t train (headless), -p play
scripts/
  rsl_rl/train.py, play.py, cli_args.py   Upstream + additions: --experiment_name is honored; resume from
                                       another experiment (--resume_experiment), actor-only fine-tuning
                                       (--finetune), noise std override after loading (--finetune_std);
                                       play.py adapted to Isaac Lab 2.3 / rsl-rl 5 (exports policy.pt/.onnx
                                       to <run>/exported/)
  list_envs.py
  pkl_to_csv_without_hands.py          GMR .pkl → CSV (29 body joints: cols 0-21 + 34-40, hands dropped)
  mimic/csv_to_npz.py                  CSV → .npz (Isaac replay, upstream)
  mimic/pkl_to_npz.py                  GMR .pkl → .npz directly, whole folder in one Isaac session
  mimic/check_motion.py                Numpy-only check: .npz joints vs .pkl columns, static-arm warning
  mimic/replay_npz.py                  Visually inspect a .npz
source/unitree_rl_lab/unitree_rl_lab/
  assets/robots/unitree.py             UNITREE_G1_29DOF_MIMIC_CFG, UNITREE_G1_29DOF_MIMIC_ACTION_SCALE,
                                       local paths UNITREE_MODEL_DIR / UNITREE_ROS_DIR
  tasks/locomotion/, tasks/mimic/      Upstream tasks (Velocity, Gangnam, Dance)
  tasks/ball_catching/                 ← PROJECT TASK (= BC/)
    agents/rsl_rl_ppo_cfg.py           BasePPORunnerCfg (used by all three tasks)
    mdp/                               commands/, observations, rewards, terminations, events, curriculums
    robots/g1_29dof/phase1/            Env cfg + motions/ (drop-in folder for .npz, README)
    robots/g1_29dof/phase2/            Env cfg
    robots/g1_29dof/unified/           Env cfg
pkl_isaac_lab_fixed_27_07_26/          150 retargeted GMR .pkl (pkl_to_npz.py writes to npz/ inside), gitignored
pkl_isaac_lab_17_08_26/                7 newer .pkl (as of 17 Aug), gitignored
logs/rsl_rl/<experiment>/<timestamp>/  Checkpoints model_*.pt, params/, tfevents
deploy/                                C++ deployment (ONNX, unitree_sdk2), upstream, untouched for the ball task
context/                               Project background + this spec (see Section 12)
```

`.npz` files are excluded globally via `.gitignore`. The training clip in `phase1/motions/` is therefore
**not in git** and must be copied to each machine separately.

## 4. Environment & Commands

- Python env (Marius' machine): `/home/marius/miniconda3/envs/env_isaaclab_downgrade/bin/python`
  (Isaac Lab 2.3.0, Isaac Sim 5.1.0). The system Python has no numpy.
- Registered tasks: `Unitree-G1-29dof-BallCatch-Phase1`, `-Phase2`, `-Unified`
  (plus upstream: `Unitree-G1-29dof-Velocity`, `-Mimic-Gangnanm-Style`, `-Mimic-Dance-102`, Go2/H1 Velocity).

```bash
./unitree_rl_lab.sh -l                                           # list tasks
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-BallCatch-Phase1  # train headless
./unitree_rl_lab.sh -p --task Unitree-G1-29dof-BallCatch-Phase1  # play (loads latest run)
# Phase 2 from scratch (std 1.0)
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-BallCatch-Phase2
# Phase 2 fine-tuned from a Phase 1 checkpoint: loads the actor only, sets the noise std,
# logs into the Phase 2 folder (NOT into the Phase 1 folder)
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-BallCatch-Phase2 --resume \
  --resume_experiment unitree_g1_29dof_ballcatch_phase1 --load_run <timestamp> --checkpoint model_<N>.pt \
  --finetune --finetune_std 0.2
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-BallCatch-Unified # unified (from scratch, std 1.0)
# Smoke test: --num_envs 16 --max_iterations 2 --experiment_name /tmp/smoke  (absolute path keeps logs/ clean)
# Hydra overrides are passed without dashes, e.g. agent.policy.init_noise_std=1.0
tensorboard --logdir logs/rsl_rl/
```

## 5. Data Pipeline (mocap → training clip)

```
VICON Shogun (+Manus) ──(other repos: cleaning, GMR retargeting)──► *_merged_filtered.pkl
  .pkl: root_pos, root_rot (xyzw), fps (usually 30), dof_pos with 53 cols in this layout:
        0-11 legs | 12-14 waist | 15-21 left arm | 22-33 left hand | 34-40 right arm | 41-52 right hand
      │
      ├─ recommended: scripts/mimic/pkl_to_npz.py -i <folder|file> [--headless] [-o out] [--output_fps 50]
      │     29 body joints (cols 0-21 + 34-40, SDK order) → interpolation (lerp/slerp) to 50 fps
      │     → finite-difference velocities
      │     → kinematic replay in Isaac Sim → forward kinematics of all bodies → <input>/npz/*.npz
      └─ legacy: pkl_to_csv_without_hands.py → mimic/csv_to_npz.py
      ▼
.npz keys: fps, joint_pos (T,29), joint_vel (T,29), body_pos_w (T,30,3), body_quat_w (T,30,4),
           body_lin_vel_w (T,30,3), body_ang_vel_w (T,30,3)
      ▼
scripts/mimic/check_motion.py --pkl <file>.pkl --npz <file>.npz   → must print "RESULT: OK"
      ▼
BC/robots/g1_29dof/phase1/motions/*.npz   (the alphabetically first file is picked automatically)
```

- **Current clip:** `143_merged_filtered.npz`, 155 frames at 50 fps, i.e. **3.1 s**. Regenerated on 2026-09-28
  with the fixed `pkl_to_npz.py` from `pkl_isaac_lab_fixed_27_07_26/143_merged_filtered.pkl`; `check_motion.py`
  reports `RESULT: OK`, and a Phase 1 probe shows no more torso/arm self-contact (feet only).
- Known quirk: `pkl_to_npz.py` hangs while Isaac Sim shuts down after printing `[DONE]`. The files are
  complete at that point; stop it with Ctrl+C.
- The `.npz` column order is the Isaac articulation joint order (not SDK order);
  `check_motion.py` contains both orders.
- If no `.npz` is in `motions/`, Phase 1 (and therefore Unified) falls back to the Gangnam placeholder and
  prints a loud warning.
- Quality check before training (see `phase1/motions/README.md`):
  - full reach–catch–retract cycle
  - wrist height at the catch matches `catch_height_range`
  - no foot sliding
- **Known data issues** (from the meeting minutes):
  - jitter and wrist flipping
  - self-collisions
  - ball position offset or delayed relative to the robot
  - the ball is only usable in few recordings

  The mocap ball trajectory is **not used in training**; the ball is purely synthetic there.

## 6. Common Task Base (all three ball tasks)

| Parameter | Value |
|---|---|
| Robot | `UNITREE_G1_29DOF_MIMIC_CFG` (29 DoF, no hands) |
| Physics / control | `sim.dt = 0.005`, `decimation = 4`, i.e. **50 Hz policy** |
| Envs | 4096, spacing 2.5 m, flat plane |
| Actions | `JointPositionAction`, all joints, `scale = UNITREE_G1_29DOF_MIMIC_ACTION_SCALE` (per joint 0.25·effort/stiffness), `use_default_offset=True` |
| Ball | `RigidObject` sphere r = 0.05 m, 0.15 kg, friction 0.5, restitution 0.6. Startup DR: friction 0.3–0.8, restitution 0.4–0.8, mass ±0.05 kg |
| "Hands" | `HAND_BODY_NAMES = ["left_wrist_yaw_link", "right_wrist_yaw_link"]` |
| Contact sensor | `Robot/.*` (whole body, required by `undesired_contacts`) |
| Observation history | 5 frames (policy and critic) |
| PPO | Actor/critic `[512,256,128]` ELU, lr 1e-3 adaptive, γ 0.99, λ 0.95, 24 steps/env/iteration, max 30k iterations, save every 500, `empirical_normalization=False` |

### 6.1 Observation Layout (contract across tasks)

All three tasks share **the same slot order and dimensions**. Where a task has no value for a slot, it
uses `dummy_zeros`. This keeps checkpoints shape-compatible across tasks.

**Policy** (per frame, ×5 history):

| # | Slot | Dim | Phase 1 | Phase 2 | Unified |
|---|---|---|---|---|---|
| 1 | `motion_command` (reference joint_pos+vel) | 58 | real | 0 | real |
| 2 | `ball_state` = `ball_pos_vel_b` | 6 | 0 | real | real |
| 3 | `ball_relative` = `ball_to_hands_b` | 3 | 0 | real | real |
| 4 | `hand_pos` = `hand_pos_b` | 6 | 0 | real | real |
| 5 | `ball_intercept` = `ball_intercept_b` | 3 | 0 | real | real |
| 6 | `motion_anchor_ori_b` | 6 | real | 0 | real |
| 7 | `base_ang_vel` (×0.2, noise ±0.2) | 3 | real | real | real |
| 8 | `projected_gravity` | 3 | real | real | real |
| 9 | `joint_pos_rel` | 29 | real | real | real |
| 10 | `joint_vel_rel` (×0.05, noise ±1.5) | 29 | real | real | real |
| 11 | `last_action` | 29 | real | real | real |

By my own count this is **175 dims per frame**. `context/changes/merge_changes.md` states 174; verify the
actual size when loading (`Actor Model: ... input_dim`).

**Critic** (clean, no noise): `command`(58) · `ball_state`(6) · `ball_relative`(3) · `hand_pos`(6) ·
`ball_intercept`(3) · `motion_anchor_pos_b`(3) · `motion_anchor_ori_b`(6) · `body_pos`(42) · `body_ori`(84) ·
`base_lin_vel`(3) · `base_ang_vel`(3) · `projected_gravity`(3) · `joint_pos`(29) · `joint_vel`(29) ·
`actions`(29). That is **307 per frame**. The tasks fill the slots with the same pattern as the policy.

**Conventions:**
- All ball and hand observations are expressed in the **robot's yaw frame**, relative to the root.
- They contain no world coordinates; otherwise per-env origins would leak into the observations.
- This is the same frame that will be computed from VICON on the hardware.
- `BallCommand.command` (world coordinates) is **for logging only** and must not be used as an observation.

## 7. Task Specifications

### 7.1 Phase 1: `Unitree-G1-29dof-BallCatch-Phase1` (motion tracking)

- BeyondMimic style: `MotionCommand` with anchor `torso_link`, 14 tracked bodies, adaptive failure-bin
  sampling, teleport reset at clip end, episode length 30 s.
- Rewards:
  - Tracking exp kernels: anchor pos/ori 0.5 each; body pos/ori/lin_vel/ang_vel 1.0 each.
  - Regularization: joint_acc, joint_torque, action_rate −0.1, joint_limit −10, undesired_contacts −0.1
    (excluding ankles and wrists).
- Terminations: `anchor_pos` (z > 0.25), `anchor_ori` (0.8), `ee_body_pos` (z > 0.25 at ankles and wrists), timeout.
- DR: material 0.3–1.6 / 0.3–1.2 / restitution 0–0.5, default joint offset ±0.01, torso CoM, torso mass −1…+3 kg,
  aggressive 6-DoF pushes.
- No ball in the scene; ball slots are zeros. Runner: `BasePPORunnerCfg` (std 1.0).

### 7.2 Phase 2: `Unitree-G1-29dof-BallCatch-Phase2` (RL catching)

- Standing and catching without a motion reference. Reset: root randomized ±0.5 m, **yaw ±π**, random joint velocities.
- `BallCommand` (one throw per episode):
  1. **Park** the ball at a spawn point in the robot's heading frame: distance 1.0–2.5 m, lateral ±0.75 m,
     height +0.8–1.5 m above the root.
  2. After `throw_delay_range` of 0.5–1.5 s, launch it ballistically toward the catch zone:
     - aim point 0.4 m in front of the robot, lateral σ 0.05 m
     - height **+0.25–0.45 m above the root** (chest height, about 1.0–1.2 m)
     - flight time 0.3–0.6 s, velocity noise σ 0.3 m/s
  3. The ball counts as **caught** when, for 25 consecutive steps (0.5 s), all of the following hold
     (counter `secured_steps`):
     - distance to the closest hand < 0.2 m
     - speed relative to that hand < 0.5 m/s
     - ball z > 0.5 m
- Rewards:

  | Reward | Weight | Meaning |
  |---|---|---|
  | `hand_to_ball` | 2.0 | exp(−d²/0.3²) |
  | `catch_success` | 500 | one-time, effectively 10 |
  | `ball_secured_hold` | 5.0 | 0.1 per step |
  | `ball_height_penalty` | −5 | ball below 0.5 m |
  | `alive` | 0.15 | |
  | Posture crutches | | `base_height` −10 @0.76 m, `flat_orientation` −5, `joint_deviation_*`, `feet_slide`, `undesired_contacts` −1 (excluding ankles and wrists), further locomotion regularization |

- Terminations: timeout 20 s, `base_height` < 0.2, `bad_orientation` 0.8, **`ball_caught` (success)**,
  `ball_dropped` (z < 0.1), `ball_missed` (horizontal > 4 m).
- Curriculum:
  - Throw distribution from easy to final over steps 50k → 250k.
  - Posture tapers over steps 100k → 400k: `base_height` −10 → −2.5, `flat_orientation` −5 → −1,
    arm deviation −0.1 → −0.02.
  - The play cfg sets `curriculum=None` and therefore uses the final distribution.
- Runner: `BasePPORunnerCfg` (std 1.0, from scratch). Fine-tuning from Phase 1 is done on the command line
  (`--resume --resume_experiment … --finetune --finetune_std 0.2`, see Section 4): `init_noise_std` in a runner
  config has no effect after `--resume`, because the std is part of the loaded actor state.

### 7.3 Unified: `Unitree-G1-29dof-BallCatch-Unified` (recommended direction)

- DeepMimic style: Phase 1 tracking rewards **and** Phase 2 ball rewards at the same time. No dummy slots,
  **no** posture crutches and no robot reset events (the `MotionCommand` owns resets).
- `SyncedBallCommand`: launch at `launch_time = CATCH_MOTION_TIME ± 0.05 s − flight_time`, so that ball
  arrival coincides with the catch frame of the mocap.
  - If an episode starts after the launch window, no throw happens; the episode trains pure tracking.
  - When the clip wraps or is resampled, the ball is re-parked and re-armed.
  - **`motion` must be declared before `ball_throw` in `CommandsCfg`.**
- Terminations: Phase 1 tracking terminations (`ee_body_pos` loosened to 0.3) plus Phase 2 ball terminations.
  Drops count as failures in the adaptive sampler, which then samples the catch window more often.
- Curriculum:
  - Throw distribution 50k → 250k.
  - Body tracking tapers (pos/ori 1.0 → 0.4, vel 1.0 → 0.5) over 150k → 500k.
  - Anchor tracking keeps full weight.
- 6-DoF pushes every 2–5 s. Runner: `BasePPORunnerCfg` (std 1.0).
  Optional warm start: `--resume --resume_experiment unitree_g1_29dof_ballcatch_phase1 --load_run … --finetune --finetune_std 0.3`.
- **Before serious training:**
  - Set `CATCH_MOTION_TIME` (in `unified/ball_catch_env_cfg.py`) to the ball-contact moment of the current
    clip. The clip is 3.1 s long; 2.0 s is an unverified placeholder.
  - Check that the wrist height at the catch matches `catch_height_range`.

## 8. MDP Building Blocks (`BC/mdp/`)

| File | Contents |
|---|---|
| `commands/motion_command.py` | `MotionLoader`, `MotionCommand(Cfg)`: reference states, adaptive bin sampling, metrics `error_*`, `sampling_*` |
| `commands/ball_command.py` | `BallCommand(Cfg)`: parking, throw, `secured_steps`, metrics `ball_height`, `ball_secured_time`, `hand_ball_distance`. Helpers `_sample_spawn`, `_write_parked_state`, `_launch`, `_update_catch_state` |
| `commands/synced_ball_command.py` | `SyncedBallCommand(Cfg)`: launch coupled to the motion phase (Unified only) |
| `observations.py` | Motion obs (`motion_anchor_*`, `robot_body_*`), ball obs (`ball_pos_vel_b`, `ball_to_hands_b`, `ball_intercept_b`, `hand_pos_b`), `dummy_zeros` |
| `rewards.py` | Tracking exp rewards, `energy`, `hand_to_ball_distance_exp`, `ball_caught_bonus`, `ball_secured`, `ball_height_penalty` |
| `terminations.py` | Tracking terminations (`bad_anchor_*`, `bad_motion_body_pos*`), `ball_caught`, `ball_below_height`, `ball_far_from_robot` |
| `events.py` | `randomize_joint_default_pos`, `randomize_rigid_body_com` |
| `curriculums.py` | `modify_reward_weight_linear`, `ball_throw_curriculum` (linear interpolation over `common_step_counter`) |

`ball_intercept_b`: intersection of the (descending) ball parabola with the plane 0.35 m above the root.
Output `[x_b, y_b, t_go]`, with `t_go` clamped to [0, 3] s. If the plane is never reached, `t_go = 0` and the
current position is returned.

## 9. Invariants & Pitfalls

1. **Observation contract:** If you change a slot in one task (order, dim, scale, noise), mirror it in *all*
   tasks and *both* groups. Otherwise transfer and warm starts break and old checkpoints become incompatible.
2. **Action scale** is `UNITREE_G1_29DOF_MIMIC_ACTION_SCALE` in all tasks. A future deploy config must use the same.
3. Isaac Lab multiplies rewards by `weight × dt` (dt = 0.02). One-time bonuses are therefore weighted as `value / dt`.
4. Isaac Lab step order: terminations → rewards → reset → command update. Command thresholds and termination
   thresholds must not contradict each other (this bug was fixed in P0).
5. `undesired_contacts` must exclude the wrists (later: hand links), otherwise catching is penalized.
6. Checkpoints from before 2026-08-10 (pre-P0/P1: different obs scales, no `ball_intercept`) are **incompatible**.
7. For play and evaluation set `curriculum=None`, otherwise evaluation runs on the easy initial distribution.
8. Curriculum steps assume about 30k iterations (24 steps per iteration). Scale them for other run lengths.
9. GMR `.pkl` files interleave the hands with the arms (see Section 5). Never slice `dof_pos[:, :29]`; use
   `select_body_joints()` from the conversion scripts.
10. `--resume` restores actor (incl. noise std), critic, optimizer and iteration unless `--finetune` is given.
    `--experiment_name` decides where the new run is logged; `--resume_experiment` where the checkpoint is read from.

## 10. Training History & Artifacts

| Run | Task | Motion | Result |
|---|---|---|---|
| `ballcatch_phase1/2026-07-27_13-32-10` | Phase 1 | Gangnam placeholder | model_99 (smoke test, outdated, incompatible) |
| `ballcatch_phase1/2026-07-27_13-41-06` | Phase 1 | – | empty / aborted |
| `ballcatch_phase1/2026-08-31_14-45-09` | Phase 1 | 143_merged_filtered | model_1 (test run) |
| `ballcatch_phase1/2026-08-31_15-06-41` | Phase 1 | 143_merged_filtered | model_1500, 1657 iterations, tracking converged, but on the corrupt clip (see Section 2) → unusable |
| `ballcatch_phase2/` | Phase 2 | – | empty |
| Unified | – | – | never started |

## 11. Open Items & Next Steps (prioritized)

1. Visually check the regenerated clip with `replay_npz.py` (legs!), then **retrain Phase 1**. Convert further
   clips as needed (`context/guides/replay_guide.md`, step 5).
2. **Determine `CATCH_MOTION_TIME`** (via `replay_npz.py` on the regenerated clip) and check
   `catch_height_range` against the wrist height. Then **train Unified**, with Phase 2 in parallel as an
   A/B comparison (fine-tuned from the new Phase 1 run vs. from scratch).
3. Remaining review findings M1–M4 and L1–L5 in `context/REVIEW.md` (C1, H1, H2 are fixed).
4. Establish a catch-rate metric (`Episode_Termination/ball_caught` vs. `ball_dropped`/`ball_missed`) and decide
   which line (two phases or Unified) to pursue.
5. Check the ball size: the simulation uses ∅ 10 cm and 0.15 kg, while the mocap group now uses a larger ball or
   a football. Adapt the sim parameters to the real ball.
6. Model the static hand mount (per the proposal) as geometry and tune catch detection / `catch_radius` to it.
   Dex3 hands are **not** required per the proposal.
7. Sim2real hardening:
   - noise, 1–2 step delay and dropout on ball observations
   - actuator delay as DR
   - fit the throw distribution to real VICON throws
8. Sim2Sim in MuJoCo, ONNX export and a deploy config (`deploy/robots/g1_29dof/config/`). Build the live VICON
   pipeline:
   - robot base tracking
   - ballistic Kalman filter with latency forward prediction
   - **identical** yaw-frame math to `observations.py`
9. Select more / better clips (`pkl_isaac_lab_*`). `MotionCommand` currently loads only **one** file.

## 12. Context Documents

`context/` (versioned) contains:
- `unitree_rl_lab_spec.md`: this document
- `pipeline/PG-Antrag-H-ReACT.pdf`: official project proposal with the minimum goals
- `pipeline/H-React.drawio(.xml)`: architecture diagram of the overall pipeline
- `pipeline/LATENT_pipeline.md`: reference extracted from the LATENT tennis paper
- `protocols/`: weekly meeting minutes from 22 May to 10 Aug 2026 (German; team split, status of other subgroups)
- `planning/g1_catching_optimization_plan.md`: analysis, findings F-1…F-13, roadmap P0–P3 (sim2real, Dex3 concept)
- `planning/Phase2_BallCommand_plan.md`: early Phase 2 plan (partly outdated, e.g. multi-throw logic was removed)
- `changes/p0_changes.md`, `changes/p1_changes.md`, `changes/merge_changes.md`: changelogs of the 10 Aug implementation
- `changes/LATENT_mimic_adjustments.md`: LATENT ideas for the mimic stage (not all adopted yet)
- `guides/debug_instructions.md`: older debug guide (partly outdated, e.g. debug prints were removed)
- `guides/replay_guide.md`: step-by-step guide (German) to convert, check (`check_motion.py`) and replay a mocap clip
- `REVIEW.md`: code review of the ball-catching work (findings C1–L5; C1, H1, H2 fixed)

In case of conflicts: **code > this spec > other context documents.**

## 13. Glossary

- **Mimic / Phase 1:** Imitation of a reference motion via tracking rewards (BeyondMimic recipe).
- **Anchor:** Reference body (`torso_link`) relative to which body positions are compared.
- **Yaw frame:** Robot root frame with only the yaw rotation (no roll or pitch).
- **Secured:** Ball close to a hand, slow relative to it, above the ground. 25 consecutive steps mean "caught".
- **GMR:** General Motion Retargeting, human → G1.

---

## 14. Changelog

> Append new entries **at the bottom**. Format: `### YYYY-MM-DD: Short title` + bullet points
> (what, why, affected files / sections). Entries before 2026-09-28 were reconstructed from git history.

### 2026-06-19: Ball-catching task created
- Phase 1 task as a copy of the Gangnam mimic task (`ca57347`). Unitree model/ROS paths set, RSL-RL compatibility.

### 2026-06-25 to 2026-07-02: Phase 2 built
- Ball object, `BallCommand`, ball observations, catch rewards, terminations, ball DR (mass, material)
  (`8240bb8` … `2304c13`).

### 2026-07-17: Mocap conversion
- `scripts/pkl_to_csv_without_hands.py`: GMR .pkl → CSV, hand columns dropped (`93de3a4`).

### 2026-07-26/27: Transfer Phase 1 → Phase 2
- Observation padding with `dummy_zeros`, identical shapes. Fix in `cli_args.py`: `--experiment_name` is honored,
  so `--resume` works across tasks (`c5bd979`, `43a1391`).

### 2026-08-10: P0 fixes (`4cf10f5`)
- Ball and hand observations in the yaw frame; throws in the heading frame; aim height lowered to the chest.
- Same action scale in both phases; unified obs scales; `undesired_contacts` excludes the wrists.
- Single-throw episodes with a throw delay.
- New catch detection: `secured_steps`, one-time bonus, success termination.
- Hygiene fixes.
- Details: `context/changes/p0_changes.md`.

### 2026-08-10: P1 content & transfer integrity (`0ff1d94`)
- Auto-discovery of `phase1/motions/`, `ball_intercept_b`, mirrored DR, curriculum (throws + posture tapers),
  `Phase2PPORunnerCfg` (std 0.2). Details: `context/changes/p1_changes.md`.

### 2026-08-10: Unified task (`1e4515f`)
- `Unitree-G1-29dof-BallCatch-Unified` and `SyncedBallCommand`. Details: `context/changes/merge_changes.md`.

### 2026-08-31: Phase 1 trained on real mocap data
- Clip `143_merged_filtered.npz` in `phase1/motions/`, run `2026-08-31_15-06-41` (1657 iterations, converged).

### 2026-09-28: Direct .pkl → .npz conversion (`5362bca`)
- `scripts/mimic/pkl_to_npz.py`: GMR .pkl → .npz in one step, whole folder per Isaac session.
- `.gitignore`: `pkl_isaac_lab_*` folders.

### 2026-09-28: Spec document & CLAUDE.md introduced
- `unitree_rl_lab_spec.md` created (full state snapshot).
- `CLAUDE.md` created: substantial changes must first be reflected in the spec, then logged in the changelog.

### 2026-09-28: Context folder versioned and restructured; spec translated to English
- `context/` committed (`df84fd8`) with subfolders `pipeline/`, `protocols/`, `planning/`, `changes/`, `guides/`.
  Root-level plan / changelog documents moved there.
- Spec moved to `context/unitree_rl_lab_spec.md` and translated fully to English. Sections 3, 6.1, 12 and
  changelog paths updated to the new structure. `CLAUDE.md` now points to the new spec location.

### 2026-09-28: Review fixes C1, H1, H2 (see `context/REVIEW.md`)
- **C1** `scripts/mimic/pkl_to_npz.py`, `scripts/pkl_to_csv_without_hands.py`: the GMR `.pkl` columns 22-28 are
  left-hand finger joints, not the right arm. Body joints are now taken from columns 0-21 + 34-40
  (`select_body_joints()`), and any width other than 53/29 raises. Every `.npz` created before is corrupt
  (right arm frozen inside the torso); the Phase 1 run `2026-08-31_15-06-41` is marked unusable.
- New `scripts/mimic/check_motion.py`: numpy-only comparison of a `.npz` against its `.pkl` (per joint, Isaac ↔ SDK order).
  Verified: the old clip shows 7 mismatching right-arm joints; a clip regenerated with the fix matches exactly.
- **H1** `train.py`/`cli_args.py`: new `--finetune` (load actor only; fresh critic, optimizer, iteration) and
  `--finetune_std` (overwrite the loaded noise std). `Phase2PPORunnerCfg` removed (its `init_noise_std` had no
  effect after `--resume`); Phase 2 now uses `BasePPORunnerCfg`.
- **H2** `train.py`/`cli_args.py`: new `--resume_experiment` reads the checkpoint from another experiment folder,
  so Phase 2 fine-tuning no longer logs into the Phase 1 folder. Verified by a smoke run (logs in the given folder,
  std 0.20, iteration counter starts at 0).
- Spec sections 2, 3, 4, 5, 7.2, 7.3, 9, 10 and 11 updated.

### 2026-09-28: Training clip regenerated
- `phase1/motions/143_merged_filtered.npz` regenerated with the fixed `pkl_to_npz.py` from
  `pkl_isaac_lab_fixed_27_07_26/143_merged_filtered.pkl` (155 frames @ 50 fps). `check_motion.py`: all 29 joints match
  (maxdiff 0.000). Phase 1 probe: contact forces on `torso_link` and the right arm dropped from ~2000 N to 0.
- Spec sections 2, 5 and 11 updated.

### 2026-09-28: Corrupt clips removed, replay guide updated
- The five pre-fix `.npz` in `pkl_isaac_lab_fixed_27_07_26/npz/` were deleted (by the user); only the regenerated
  training clip in `phase1/motions/` exists.
- `context/guides/replay_guide.md` rewritten for the current paths (file overview, check/replay of the training clip,
  swapping in another clip). Spec sections 2, 3 and 11 updated.

### 2026-09-28: play.py fixed for Isaac Lab 2.3 / rsl-rl 5 (`d86ef57`)
- `scripts/rsl_rl/play.py` crashed on import (`isaaclab.utils.pretrained_checkpoint` moved to `isaaclab_rl.utils`).
  Also added `handle_deprecated_rsl_rl_cfg` (as in `train.py`) and JIT/ONNX export via
  `runner.export_policy_to_jit/onnx` for rsl-rl >= 4 (`runner.alg.policy` no longer exists).
- Verified by the user: playing the smoke checkpoint `/tmp/smoke_p1/…/model_9.pt` opens the sim and runs the policy.
- Spec section 3 updated.
