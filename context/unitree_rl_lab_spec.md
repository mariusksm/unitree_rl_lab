# unitree_rl_lab: Specification & Current State

> **Purpose of this document:** A fast, reliable entry point for humans and AI assistants.
> It describes *what* this repository does within the H-ReACT project group, *how* it is structured,
> and *where* it currently stands.
> **Maintenance rule:** For every substantial change, first update the affected sections, then append an
> entry to the [Changelog](#14-changelog) at the end (see `CLAUDE.md`).
>
> Last updated: **2026-10-16**. Paths are relative to the repo root. `BC/` is shorthand for
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

## 2. Status at a Glance (2026-10-27)

| Area | Status |
|---|---|
| Motion conversion (.pkl → .npz) | ✅ Fixed on 2026-09-28 (the right arm was filled with left-hand finger joints; all `.npz` created before that date were corrupt and have been deleted). Since 2026-10-27 the reference clips come prepared from the group's dataset (Section 5.2); `pkl_to_npz.py` remains for new takes. |
| Reference motion | ✅ 2026-10-27: the group's **underhand-catch dataset** (GitLab `imitation-data`: 108 clips at 50 Hz, 89 of them with a VICON recording of the take) replaces the single overhand clip. Phase 1 trains on all clips, Unified since 2026-10-29 only on the 84 with a usable recording (86 before 2026-10-28), throwing at each clip's recorded catch time. Section 5.2. |
| Phase 1 (mimic) on real mocap data | ✅ 2026-10-28: trained on the 108 underhand clips (workstation 2, 10k iterations, run `2026-10-28_14-28-00`, code before fixes 1–4). Every clip is tracked from start to end without a tracking termination, body error 5 cm, almost all of it from a height offset of the clips (Section 5.2). Since 2026-10-16 Phase 1 is a clip trackability check only (actor sees the reference), **not** a warm start for Phase 2. |
| VICON ball recordings | ✅ Found 2026-10-16: 21 takes of the 2026-07-06 session with ball CSV + retargeted `.pkl` (12 complete underhand throws), see Section 5.1. Used to calibrate throws, drag and the perception model. |
| Ball and throw model | ✅ 2026-10-16: mjlab ball (∅15 cm, 112 g; drag fits the recordings), underhand lob from 2–5 m released at 0.85–1.2 m with apex 1.75–2.1 m, quadratic air drag with drag-compensated aim, catch = ball held in a box in front of the torso while touching both arms for 1 s. Restitution is a placeholder until the real ball is measured. |
| Actor observations | ✅ 2026-10-16: deployable actor contract in Phase 2 and Unified (perceived ball + visibility flag + intercept + proprioception, no reference, no hand positions). Ball perceived through a VICON model (detection delay, latency, noise, dropout bursts, occlusion in the catch box, coasting). |
| Phase 2 (RL catching, stage B) | ⚠️ Baseline trained to 30k iterations (runs `2026-10-22_16-04-10` → `2026-10-22_21-14-31` → `2026-10-23_03-35-20`, final `model_29999.pt` on workstation 2): 99.6 % catches on the final throw distribution. The 2026-10-27 stress tests and video analysis (Section 10) show it is **not deployable as is**: joint-stop exploit at the ankle-roll joints (feet on their edges, torque saturated), a throw after 6–9 s of waiting gives 28 % catches and 28 % falls, latency and pushes cost 6 points each, the ball is squeezed to the left chest. Prioritized fixes in Section 11 and `context/analysis/2026-10-27-phase2-final.md`. |
| Unified task (mimic + catching simultaneously, stage A) | ⚠️ Two runs stopped (2026-10-28 at iteration 8000, 2026-10-29 at ≈ 6000 by a crash of workstation 2), catches up to ≈ 30 % of the episodes in training. Fixed since: throws aimed at the reference hands, `feet_flat` (feet now flat), every cycle starts before its throw and only the 84 clips with a usable catch are used (cycles without a ball caused most tracking terminations), a ball squeezed against the torso (> 50 N) does not count as held (both runs clamped it to the chest). Every clip runs as a wait-and-hold cycle (6 s lead-in, catch pose held 5 s), confirmed catches are held for 3 s. |
| Catch success in simulation | ⚠️ 99.6 % (Phase 2, deterministic policy, final throw distribution), but only within the trained episode structure (short wait, 1 s hold); see Section 10. Since 2026-10-28 Phase 2 and Unified train long waits and a 3 s hold phase (fix 1), no checkpoint yet. |
| Hands (Dex3 / mount) in the model | ❌ not present (intentional, see proposal) |
| Sim2Sim (MuJoCo), deployment, live VICON pipeline | ⚠️ Sim2Sim started 2026-10-29 (`sim2sim/`): locked Python environment and the G1 MJCF of `unitree_rl_mjlab` on both workstations, model comparison done (masses, inertia, kinematics, joint limits identical to Isaac; collision shapes differ, and the Isaac model has no collider on the forearms, Section 6). Standing works: the third Unified run's `model_3000.pt` stands stably in MuJoCo without a ball, with the posture it has in Isaac. The ball observations are ported to numpy and reproduce Isaac exactly. Throws work in MuJoCo (`model_4500.pt`: 23–30 % catches against 44 % in Isaac, but only with a damped ball contact; with a MuJoCo ball that bounces like Isaac's the wedged ball is shot out: 0 %). Deployment and live VICON pipeline not started (`deploy/` is unmodified Unitree upstream; the C++ side has none of the ball observation terms yet) |

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
  rsl_rl/eval_catch.py                 Evaluation suite: one checkpoint, one stress-test scenario (Phase 2, Unified),
                                       metrics as JSON (outcomes, feet, posture, hold geometry, joint limits, torques)
  rsl_rl/eval_catch_all.sh             All scenarios of a checkpoint + comparison table (eval_catch_report.py)
  list_envs.py
  pkl_to_csv_without_hands.py          GMR .pkl → CSV (29 body joints: cols 0-21 + 34-40, hands dropped)
  mimic/csv_to_npz.py                  CSV → .npz (Isaac replay, upstream)
  mimic/pkl_to_npz.py                  GMR .pkl → .npz directly, whole folder in one Isaac session
  mimic/check_motion.py                Numpy-only check: .npz joints vs .pkl columns, static-arm warning
  mimic/replay_npz.py                  Visually inspect a .npz
  mimic/catch_info.py                  Numpy-only: catch time / flight time / catch point per clip from the ball
                                       recordings, with the validity check of the Unified task (Section 5.2)
  mimic/check_dataset.py               Numpy-only: robot elbows/hands of each clip vs the human elbows/palms of its
                                       VICON recording (right arm motion, right side; no hand data in an arm)
  mimic/check_npz_fk.py                Isaac: clips posed on the training asset, forward kinematics vs the stored
                                       body positions (joint and body order), excess over hard and soft joint limits
source/unitree_rl_lab/unitree_rl_lab/
  assets/robots/unitree.py             UNITREE_G1_29DOF_MIMIC_CFG, UNITREE_G1_29DOF_MIMIC_ACTION_SCALE,
                                       local paths UNITREE_MODEL_DIR / UNITREE_ROS_DIR
  tasks/locomotion/, tasks/mimic/      Upstream tasks (Velocity, Gangnam, Dance)
  tasks/ball_catching/                 ← PROJECT TASK (= BC/)
    agents/rsl_rl_ppo_cfg.py           BasePPORunnerCfg (used by all three tasks)
    motion_data.py                     Clip discovery (search order in 5.2), LFS-pointer check, alignment of a
                                       take's VICON recording with its clip (numpy only)
    robot_assets.py                    ROBOT_CFG of the three ball tasks; optional forearm colliders (FOREARM_COLLIDERS)
    mdp/                               actions, commands/, observations, rewards, terminations, events, curriculums
    robots/g1_29dof/phase1/            Env cfg (owns the clip discovery) + motions/ (fallback folder, README)
    robots/g1_29dof/phase2/            Env cfg
    robots/g1_29dof/unified/           Env cfg
pkl_isaac_lab_fixed_27_07_26/          150 retargeted GMR .pkl (pkl_to_npz.py writes to npz/ inside), gitignored
pkl_isaac_lab_17_08_26/                7 newer .pkl (as of 17 Aug), gitignored
data/imitation/{g1,ball}/              Local copy of the group's clips and ball recordings (gitignored, README)
logs/rsl_rl/<experiment>/<timestamp>/  Checkpoints model_*.pt, params/, tfevents
deploy/                                C++ deployment (ONNX, unitree_sdk2), upstream, untouched for the ball task
sim2sim/                               MuJoCo sim2sim (README): setup.sh (locked venv ~/envs/sim2sim, G1 MJCF of
                                       unitree_rl_mjlab in ~/unitree_mjcf), model.py (MuJoCo G1 with the Isaac
                                       collision shapes), tools/ (environment check, Isaac model dump, model
                                       comparison, self-collision over the clips, ball-on-arm probe in Isaac)
context/                               Project background + this spec (see Section 12)
```

`.npz` files are excluded globally via `.gitignore`. The reference clips are therefore **not in git** and must not be (private group data, public repository); copy them to each machine as described in `data/imitation/README.md`.

## 4. Environment & Commands

- Python env (Marius' machine): `/home/marius/miniconda3/envs/env_isaaclab_downgrade/bin/python`
  (Isaac Lab 2.3.0, Isaac Sim 5.1.0). The system Python has no numpy.
- Group workstations (`ssh workstation` = uni.yakyaz.dev with an RTX 3090, `ssh workstation2` = uni2.yakyaz.dev with an RTX 4080; Cloudflare Access login, GPUs are reserved per person): shared conda env `/opt/miniconda3/envs/env_isaaclab` (Isaac Lab 2.3.2, Isaac Sim 5.0, rsl-rl 5.0.1) plus a per-user overlay `python -m venv --system-site-packages ~/envs/hreact` with this repo installed editable; its `activate` exports `UNITREE_MODEL_DIR=~/unitree_model`, `TMPDIR=~/tmp` and `OMNI_KIT_ACCEPT_EULA=YES`. Activate with `conda activate env_isaaclab && source ~/envs/hreact/bin/activate`, train with `python scripts/rsl_rl/train.py … --headless` (not `unitree_rl_lab.sh`, which uses the conda Python), long runs in tmux. The workstations check out branch `workstation` (= `main` + reading `UNITREE_MODEL_DIR` from the environment), rebased onto `main` and pulled with `git pull --rebase`. Camera rendering (videos) needs `LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6` and `--kit_args="--portable-root $HOME/tmp/kitroot"`, because the shared env is read-only.
- Registered tasks: `Unitree-G1-29dof-BallCatch-Phase1`, `-Phase2`, `-Unified`
  (plus upstream: `Unitree-G1-29dof-Velocity`, `-Mimic-Gangnanm-Style`, `-Mimic-Dance-102`, Go2/H1 Velocity).

```bash
./unitree_rl_lab.sh -l                                           # list tasks
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-BallCatch-Phase1  # train headless
./unitree_rl_lab.sh -p --task Unitree-G1-29dof-BallCatch-Phase1  # play (loads latest run)
# Phase 2 from scratch (std 1.0)
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-BallCatch-Phase2
# Continue an interrupted run (restores actor, critic, optimizer, iteration and curriculum progress).
# --max_iterations counts ADDITIONAL iterations: 30k in total from model_7000 -> 23000
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-BallCatch-Phase2 --resume --load_run <timestamp> \
  --checkpoint model_<N>.pt --max_iterations <30000 - N>
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-BallCatch-Unified # unified = stage A (from scratch, std 1.0)
# Stage B: Phase 2 fine-tuned from a Unified checkpoint: loads the actor only, sets the noise std,
# logs into the Phase 2 folder (NOT into the Unified folder). Phase 1 checkpoints do not fit (other actor input).
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-BallCatch-Phase2 --resume \
  --resume_experiment unitree_g1_29dof_ballcatch_unified --load_run <timestamp> --checkpoint model_<N>.pt \
  --finetune --finetune_std 0.2
# Smoke test: --num_envs 16 --max_iterations 2 --experiment_name /tmp/smoke  (absolute path keeps logs/ clean)
# Hydra overrides are passed without dashes, e.g. agent.policy.init_noise_std=1.0
tensorboard --logdir logs/rsl_rl/
# Evaluation suite (stress tests of a checkpoint; ~4 min per scenario, absolute checkpoint path):
scripts/rsl_rl/eval_catch_all.sh Unitree-G1-29dof-BallCatch-Phase2 logs/rsl_rl/<exp>/<run>/model_<N>.pt eval/<run>
python scripts/rsl_rl/eval_catch_report.py eval/<run_a> eval/<run_b>     # compare runs
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
data/imitation/g1/*.npz   (all clips are used; search order and the group dataset: Section 5.2)
```

- **Previous clip (until 2026-10-27):** `143_merged_filtered.npz`, 155 frames at 50 fps, i.e. **3.1 s**. Regenerated on 2026-09-28
  with the fixed `pkl_to_npz.py` from `pkl_isaac_lab_fixed_27_07_26/143_merged_filtered.pkl`; `check_motion.py`
  reports `RESULT: OK`, and a Phase 1 probe shows no more torso/arm self-contact (feet only).
  - Content (analysed 2026-10-16): overhand two-hand chest catch. Stand still until ≈1.3 s, both hands rise
    within 0.5 s (up to 2.2 m/s), hands stop at the chest at **≈1.96–2.0 s** (wrists ≈0.11–0.18 m in front of
    and ≈0.35 m above the pelvis, 0.21 m apart), hold until the end.
  - Leg artefacts: knees at ≈0 rad (below the 0.9 soft limit of 0.06 rad, so `joint_limit` penalizes tracking
    the whole clip) and pelvis pitched back by 13–20°.
  - The source `.pkl` has only 30 fps (mjlab gets 120 Hz retargeting output).
  - Replaced on 2026-10-27 by the group's underhand clips (Section 5.2).
- Known quirk: `pkl_to_npz.py` hangs while Isaac Sim shuts down after printing `[DONE]`. The files are
  complete at that point; stop it with Ctrl+C.
- The `.npz` column order is the Isaac articulation joint order (not SDK order);
  `check_motion.py` contains both orders.
- Without any clips (search order in Section 5.2), Phase 1 and Unified fall back to the Gangnam placeholder and print a loud warning.
- Quality check before training (see `phase1/motions/README.md`):
  - full reach–catch–retract cycle
  - wrist height at the catch matches `catch_height_range`
  - no foot sliding
- **Known data issues** (from the meeting minutes):
  - jitter and wrist flipping
  - self-collisions
  - ball position offset or delayed relative to the robot
  - the ball is only usable in few recordings

  The ball in training is synthetic; its distributions are calibrated on the VICON ball recordings (5.1).

### 5.1 VICON ball recordings

- The training `.pkl` files contain **no ball** (all 156 in `pkl_isaac_lab_*`: only `fps`, `root_pos`, `root_rot`,
  `dof_pos`). The ball is recorded separately by the VICON tracker script (report ch. 4.2.4) as `samples.csv` per take.
- The retargeting group's GMR fork (`../GMR_for_fbx_files`, branch `fbx`) links them: `third_party/extract_pkl_and_csv.py`
  keeps `tracked`/`acquired` rows, fills missing VICON frames by **linear interpolation (copying the previous
  timestamp, so interpolated rows show up as duplicate timestamps)**, drops takes with too large frame gaps or too
  short flight. Output for the 2026-07-06 session: `recordings/two_results_ver2.7z` with `results/` (unscaled, metres)
  and `results_scaled/` (positions ×0.9), each 21 × (`<n>_merged_filtered.csv` + `.pkl`). No 7z tool is installed
  on Marius' machine; the archive was read with a stdlib-only reader.
- CSV columns: `session_id, frame_index` (VICON frame, ≈120 Hz)`, timestamp_s` (workstation)`, elapsed_s, speed_m_s,
  event, x_m, y_m, z_m` (VICON world). Ball and retargeted pelvis share the world frame; the catcher's pelvis stands at
  ≈(0.9, −0.55) facing +x, the thrower at x ≈ 4.5 m. Time alignment between CSV and `.pkl` is only by take (the CSV
  starts when the tracker acquires the ball).
- Usable: takes 17, 18, 19, 21, 24, 25, 28, 29, 30, 33, 35, 40 (complete underhand throws). Takes 26, 31, 32, 36 lose
  the ball near the apex; takes 3–15 are a different setup without a complete throw (catcher at x ≈ −0.7…−1.1,
  ball mostly static).
- Measured on the 12 complete throws (unscaled `results/`):

  | Quantity | Value |
  |---|---|
  | Release distance from the catcher's pelvis | 3.1 m (2.8–3.3) |
  | Release height | 1.03 m (0.93–1.10) |
  | Apex height | 1.9 m (1.76–2.03) |
  | Flight time release → hands | 0.77 s (0.69–0.86) |
  | Release speed / arrival speed | 5.2 m/s / 4.4 m/s |
  | Arrival relative to the pelvis | 0.5–0.7 m in front, 0.3–0.8 m above (human catcher) |
  | Air drag k/m (quadratic) | 0.035 1/m (0.030–0.040); fit residual 1.2 → 0.65 cm with drag |
  | Position noise | ≈4–5 mm per axis |
  | Tracked before release (wind-up) | 0.7–1.7 s, ball moves 0.5–0.9 m |
  | Control steps (20 ms) in flight without a measurement | 6.8 %, in runs of 1–3 steps (11×1, 7×2, 2×3) |
  | After the catch | about a third lose the ball at once, the rest track it for 0.25–1.2 s |

  The k/m value matches a light ∅15 cm ball (112 g with c_d ≈ 0.37), not ∅10 cm / 150 g (0.015) or a football (0.025).

### 5.2 Group dataset: underhand-catch clips with ball recordings (current reference)

- Source: GitLab `pg-681-h-react/data/imitation-data`, folder `isaaclab/` (dataset commit `e1704e4`, as of 2026-10-27; the `.npz` files are Git LFS objects, so `git lfs pull` is needed after cloning). Maik's Isaac Lab repository (`pg-681-h-react/ml/isaac-lab`, task `g1mimic`) trains on the same files.
- `g1/<take>.npz`: 108 clips, 50 Hz, 1.9–4.6 s (median 3.1 s), BeyondMimic format in the Isaac joint and body order of our G1 asset (verified: 30 bodies, hands = bodies 28/29). The person stands still (pelvis travel ≤ 0.10 m per clip).
- `ball/<take>.npz`: 89 VICON recordings of the same takes, 120 Hz, 2.08 s each. Besides the ball (`ball_pos`, `ball_visible`, `ball_speed_m_s`) and the catch moment (`ball_catch_time_s`, `ball_catch_distance_m`, `palm_ball_distance_m`) they carry an own retarget of the take (`dof_pos` with `joint_names`, `root_pos`, `root_rot_xyzw`, finger joints, human skeleton, hand calibration) in the VICON world frame. Neither time window nor frame matches the clip.
- Alignment (`motion_data.compute_catch_info`): time offset by the best joint-angle match of the recording inside the clip (RMSE 0.18–0.30 rad, the two retargets differ; offset ≈ 0 for all but two takes), frame by the mean yaw rotation (≈ −93.5°, spread 0.3–3.6°) and translation (spread ≈ 5 cm) between the recording's root and the clip's pelvis.
- Result for the 89 recordings: launch 0.85 s, catch 1.73 s (0.84–2.16), flight 0.88 s (0.59–1.11) into the clip (medians); catch point 0.33 m in front of the pelvis, laterally centred, 1.04 m high (0.30 m above the pelvis), ball 0.21 m from the wrist midpoint and 0.13 m above the wrists (palms-up two-hand catch). Release 0.81 m high, apex 1.71 m (underhand lob). 84 recordings are usable; rejected: one catch outside its clip (52), two with the ball more than 0.3 m from the hands (45, 101) and, since 2026-10-28, two whose clip arms deviate more than 0.10 m from the human of the recording (51, 82; see the checks below).
- Search order of the clips (`motion_data.find_motion_data`): `$HREACT_MOTION_DIR` (+ `$HREACT_BALL_DIR`), `data/imitation/g1/` (local copy, ball recordings in `data/imitation/ball/`), `../imitation-data/isaaclab/g1/` (dataset repo next to this one), `phase1/motions/` (no recordings). LFS pointer files are detected and stop the import with a hint. On the workstations the dataset lies in `~/imitation-data/isaaclab/` (next to `~/unitree_rl_lab`).
- Checks: `python scripts/mimic/catch_info.py` (numpy only) prints the per-clip catch table with the validity decision of the Unified task. Joint assignment (the C1 failure mode, hand data in an arm) is checked in two steps, run on 2026-10-27:
  - `scripts/mimic/check_npz_fk.py --headless` (Isaac): all 108 clips posed on UNITREE_G1_29DOF_MIMIC_CFG reproduce their stored body positions to < 0.1 mm, so `joint_pos` is in the asset's joint order and `body_pos_w` in its body order (negative control: swapped elbow columns or SDK order give 0.27–0.39 m). The clips exceed the hard limits by at most 0.08 rad (right wrist roll).
  - `scripts/mimic/check_dataset.py` (numpy): the clips' elbows and hands follow the human elbows and palms of the recordings (scale 0.84) with 4–6 cm RMSE (worst point ≤ 0.07 m), on the same side, and the other side never fits better. Clips 51 and 52 (unusual time offsets of +0.76 s and +1.2 s) deviate by 0.12–0.22 m, and clip 82's right hand drifts 0.11 m off the human's palm although its offset is normal. No clip has a static arm. Joint angles are no test here: the recordings' own retarget distributes the arm motion differently (less elbow, more wrist roll).
  - Since 2026-10-28 this comparison (`motion_data.compare_with_human`) is part of the validity check of a recording: the worst of the four points must stay within `MAX_ARM_RMSE` = 0.10 m, otherwise the recorded catch point does not belong to the clip's hands.
  - The knees stay below the soft limit (0.06 rad) in 93–98 % of the frames (the person stands with straight knees), as with the old clip: the `joint_limit` penalty of Phase 1 and Unified acts during the whole clip. The Phase 1 run of 2026-10-28 bends its knees by 9.7° on average (reference 1.3°) at a penalty of −0.004 and tracks every clip (Section 10).
- **Self-collisions of the reference poses** (2026-10-29, `sim2sim/tools/check_self_collision.py`, Isaac collision shapes with PhysX's 64-vertex hulls, every second frame of all 108 clips): standing and default pose are free, but the clips put the right upper arm more than 2 mm into the torso in 20 % of the frames (left 2 %), up to 10 cm in clip 82; the hands overlap each other in 4 % (clip 70: 6.5 cm). With self-collisions on, Isaac pushes these apart, so the robot cannot follow those frames exactly. The forearm colliders add right forearm in torso in 8 % (left 0.3 %).
- **Height offset:** the clips sit about 4.5 cm too low. Their ankle links reach 1.2–1.8 cm below the ground plane, those of the standing robot are at about +3 cm, so the torso and feet of a robot standing on the ground are 4.4–4.6 cm above its reference in every clip (Phase 1 run of 2026-10-28). Consequences: a floor of about 5 cm in the body position error; every start from the reference pose (each reset and clip end) puts the robot 4.5 cm into the ground, and physics pushes it out by 4.2–4.8 cm within 0.2 s; in Unified the recorded catch point, which lives in the clip frame, is 4.5 cm too low relative to the robot.
  - Where it comes from (2026-10-28, numpy over the dataset): the lowest link origin of every clip (ankle roll link) lies at −1.2 cm (spread 0.3 cm), the clips' pelvis at 73.8 cm. The ball recordings carry their own retarget of the same takes; its pelvis is 3.8 cm higher (77.6 cm) at the same moments, its own metric `minimum_sole_height_m` puts the soles only 1.1 cm into the ground, and the human toe joints of the recordings stand at 0 cm. So the VICON floor is right and the offset arises in the retargeting or export that produced `isaaclab/g1`. The clips' lowest link *origin* (−1.2 cm) matching the recordings' *sole* height (−1.1 cm) suggests that the export grounded the link origin instead of the sole (about 3.3 cm below it).
  - **Temporary correction since 2026-10-28:** `motion_data.TEMP_CLIP_LIFT` = 0.045 m lifts the clips when they are loaded (`MotionCommandCfg.clip_lift` in Phase 1 and Unified) and the recorded catch points with them (`catch_data(lift=...)`, same frame). With it the Phase 1 checkpoint stands 0.1 cm above its reference instead of 4.6 cm, its body error drops from 5.0 to 1.6 cm (all 108 clips), and the ball still arrives 0.25 m from the reference wrists in Unified. It is only valid for the current dataset (`e1704e4`): with a re-exported dataset remove it (grep `TEMP_CLIP_LIFT`), otherwise every reset drops the robot from 4.5 cm.

## 6. Common Task Base (all three ball tasks)

| Parameter | Value |
|---|---|
| Robot | `robot_assets.ROBOT_CFG` = `UNITREE_G1_29DOF_MIMIC_CFG` (29 DoF, no hands), with `FOREARM_COLLIDERS` (off; pitfall 26) the same asset plus forearm colliders. Collision shapes of its USD (`unitree_model`, checked 2026-10-29 with `sim2sim/tools/dump_isaac_model.py`): convex hulls for pelvis, hip pitch and yaw, torso, head, upper arm (shoulder yaw), wrist roll piece and wrist yaw with the rubber hand; cylinders on knees and shoulder pitch; each foot 4 corner spheres (r 5 mm) plus a toe sphere and a box 1 cm higher. **No collider** on `elbow_link` and `wrist_pitch_link` (the forearm), hip roll, shoulder roll, ankle pitch and the waist links (pitfall 26) |
| Physics / control | `sim.dt = 0.005`, `decimation = 4`, i.e. **50 Hz policy** |
| Envs | 4096, spacing 2.5 m, flat plane |
| Actions | Term `JointPositionAction` = `ClampedJointPositionActionCfg` (since 2026-10-28), all joints, `scale = UNITREE_G1_29DOF_MIMIC_ACTION_SCALE` (per joint 0.25·effort/stiffness), `use_default_offset=True`. The joint targets are clamped to the asset's **hard** joint limits (not the soft ones, which would forbid straight knees) through the `clip` of Isaac Lab's `JointAction`; `export_deploy_cfg` writes the limits into `deploy.yaml` (`actions.JointPositionAction.clip`) and the C++ `JointAction` clamps the same way. Penalty `joint_target_excess` −1 in all tasks (sum over the joints of how far the unclamped target lies beyond the limit, rad). |
| Ball | `RigidObject` sphere **r = 0.075 m (∅15 cm), 0.112 kg** (= mjlab ball, interim report ch. 9.2), friction 0.5, restitution 0.4 (placeholder, real ball not measured). Startup DR: friction 0.3–0.8, restitution 0.2–0.6, mass ±0.015 kg. Quadratic air drag F = −½ρ c_d A \|v\| v as external force during the flight, c_d 0.3–0.45 per throw (k/m 0.028–0.043 1/m, fitted on the VICON throws: 0.030–0.040) (Phase 2 and Unified only) |
| "Hands" | `HAND_BODY_NAMES = ["left_wrist_yaw_link", "right_wrist_yaw_link"]` (the rubber hand collider is merged into these bodies). Midpoint = reach target of the ball rewards |
| Contact sensors | `contact_forces`: `Robot/.*` (whole body, required by `undesired_contacts`). `ball_contact` (Phase 2, Unified): sensor body = ball, filtered against the 8 arm links `{left,right}_{elbow,wrist_roll,wrist_pitch,wrist_yaw}_link`, history 4. Elbow and wrist pitch link have no collider and never report contact: the ball touches an arm only at the wrist roll piece or the hand |
| Observation history | 5 frames (policy and critic) |
| PPO | Actor/critic `[512,256,128]` ELU, lr 1e-3 adaptive, γ 0.99, λ 0.95, 24 steps/env/iteration, max 30k iterations, save every 500, `empirical_normalization=False` |

### 6.1 Observation Layout (contracts)

Since 2026-10-16 there are two contracts:
- **Actor contract (Phase 2 = Unified):** identical terms, no placeholders. `--finetune` moves an actor between
  them (Unified = stage A, Phase 2 = stage B). Only what the real robot has.
- **Critic layout (all three tasks):** same slots and dimensions, `dummy_zeros` where a task has no value. Not
  needed for `--finetune` (which loads the actor only), kept for uniformity.

Phase 1 is outside the actor contract: without a ball, only the reference tells its actor where in the clip it is.

**Actor, Phase 2 and Unified** (per frame, ×5 history → **103 per frame, 515 in total**):

| # | Slot | Dim | Source |
|---|---|---|---|
| 1 | `ball_state` = `ball_perceived_pos_vel_b` | 6 | perceived ball pos (relative to the root) + vel, yaw frame; zero while not visible |
| 2 | `ball_visible` | 1 | 1 while the robot has a ball estimate (measured or coasted) |
| 3 | `ball_intercept` = `ball_perceived_intercept_b` | 3 | ballistic crossing of the plane 0.15 m above the root, from the perceived state |
| 4 | `base_ang_vel` (×0.2, noise ±0.2) | 3 | |
| 5 | `projected_gravity` (noise ±0.05) | 3 | |
| 6 | `joint_pos_rel` (noise ±0.01) | 29 | |
| 7 | `joint_vel_rel` (×0.05, noise ±1.5) | 29 | |
| 8 | `last_action` | 29 | |

**Actor, Phase 1** (157 per frame, 785 in total): `motion_command` (58) · `motion_anchor_ori_b` (6, noise ±0.05) ·
the five proprioception terms above.

**Critic** (clean, no noise; 308 per frame, 1540 in total): `command`(58) · `ball_state`(6) · `ball_visible`(1) ·
`ball_relative`(3) · `hand_pos`(6) · `ball_intercept`(3) · `motion_anchor_pos_b`(3) · `motion_anchor_ori_b`(6) ·
`body_pos`(42) · `body_ori`(84) · `base_lin_vel`(3) · `base_ang_vel`(3) · `projected_gravity`(3) · `joint_pos`(29) ·
`joint_vel`(29) · `actions`(29). Phase 1 has zeros in the ball slots, Phase 2 in the motion slots, Unified none.
`ball_state` and `ball_intercept` are the true state here; `ball_visible` is what the actor sees.

**Conventions:**
- The ball perception (VICON model) lives in `BallCommand` (Section 7.2), so all actor ball terms see one consistent
  estimate. Give them **no** `ObsTerm` noise.
- `ball_intercept` uses the catch plane **0.15 m above the root** (belly height of the underhand catch; was 0.35 m
  until 2026-10-16). Change it in all ball tasks and both groups at once.
- All ball and hand observations are expressed in the **robot's yaw frame**, relative to the root.
- They contain no world coordinates; otherwise per-env origins would leak into the observations.
- This is the same frame that will be computed from VICON on the hardware.
- `BallCommand.command` (world coordinates) is **for logging only** and must not be used as an observation.

## 7. Task Specifications

### 7.1 Phase 1: `Unitree-G1-29dof-BallCatch-Phase1` (motion tracking)

- BeyondMimic style: `MotionCommand` with anchor `torso_link`, 14 tracked bodies, teleport reset at clip end, episode length 30 s. Since 2026-10-27 it loads all clips (Section 5.2, 108 underhand clips): every reset or clip end draws a new clip uniformly and a start bin from that clip's failure statistics (adaptive sampling per clip, one bin per second of the longest clip on each clip's normalized time).
- Rewards:
  - Tracking exp kernels: anchor pos/ori 0.5 each; body pos/ori/lin_vel/ang_vel 1.0 each.
  - Regularization: joint_acc, joint_torque, action_rate −0.1, joint_limit −10, joint_target_excess −1, undesired_contacts −0.1
    (excluding ankles and wrists).
- Terminations: `anchor_pos` (z > 0.25), `anchor_ori` (0.8), `ee_body_pos` (z > 0.25 at ankles and wrists), timeout.
- DR: material 0.3–1.6 / 0.3–1.2 / restitution 0–0.5, default joint offset ±0.01, torso CoM, torso mass −1…+3 kg,
  aggressive 6-DoF pushes.
- No ball in the scene. The actor sees the reference (`motion_command`, `motion_anchor_ori_b`) and proprioception,
  no ball slots; the critic keeps zeros in its ball slots. Runner: `BasePPORunnerCfg` (std 1.0).
- Role since 2026-10-16: **trackability check for new clips** (can the robot physically follow the reference?). It is
  no warm start: its actor input (785) differs from the Phase 2 / Unified actor (515), and `--finetune` fails loudly.

### 7.2 Phase 2: `Unitree-G1-29dof-BallCatch-Phase2` (RL catching)

- Standing and catching without a motion reference. Reset: root randomized ±0.5 m, **yaw ±π**, joints at the
  default pose (`reset_joints_by_scale` scales the zero default velocity, so joint velocities are not randomized).
- `BallCommand` (one throw per episode): **underhand lob into a two-arm catch in front of the belly**. Distance and
  direction follow mjlab (report ch. 9.4), the aim point LocoMuJoCo (ch. 7.6.2), release height and apex the VICON
  recordings (Section 5.1). The throw is sampled at reset in the robot's heading frame and **frozen** (walking toward
  the thrower gains nothing):
  1. **Park** the ball at the release point: horizontal distance **2–5 m** from the root, azimuth **±75°** around the
     heading, **0.85–1.2 m above the ground**.
  2. After a wait of **0.5–8 s** (`throw_delay_range`, log-uniform: median 2 s, a quarter of the throws after more than
     4 s; the curriculum widens it from 0.5–1.5 s; until 2026-10-27 always 0.5–1.5 s), launch it toward the aim point:
     - **0.25–0.45 m in front of the root, lateral ±0.3 m, 0.8–1.15 m above the ground** (belly to chest height; since 2026-10-29, before 0.3–0.4 m, ±0.1 m, 0.8–1.05 m). A real thrower misses: with the velocity noise 35 % of the throws now arrive more than 0.2 m and 12 % more than 0.3 m to the side (before 5 % and 0.4 %); 5 % land beyond the catch box's ±0.35 m and need a lean or a step. The dataset's human catches lie within 0.25 m of the body's centre line (90 % within −0.07…+0.12 m) and the catcher's pelvis moved at most 0.10 m before the catch, so the mocap throwers were accurate and never made the catcher step. Unified keeps aiming at the clip (Section 7.3): aimed elsewhere, the robot would have to leave the reference.
     - apex **1.75–2.1 m above the ground**; it fixes the flight time (0.71–1.02 s; recordings 0.69–0.86 s), so farther
       throws are faster. Velocity noise σ 0.1 m/s.
     - the launch velocity compensates the air drag (fixed-point iteration over the drag displacement, 16 trapezoid
       nodes). Without compensation the 15 cm ball falls ≈0.35 m short on average; with it, the PhysX flight passes
       the aim point within 2 cm on average, at most 4.5 cm (probe 2026-10-16, with the old c_d range).
  3. The ball counts as **caught** when, for `secure_steps` = **50 consecutive steps (1.0 s)**, all of the following
     hold (counter `secured_steps`):
     - ball inside the catch box x 0–0.6 m, y ±0.35 m, z −0.3…+0.5 m (root yaw frame)
     - speed relative to the root < 0.5 m/s, ball z > 0.5 m
     - ball touches **both arms** (`ball_contact` sensor, force > 0.1 N; a contact within the last 5 steps counts)
     - since 2026-10-29: ball **not squeezed against the torso** (force between ball and `torso_link` ≤ `max_torso_force` = 50 N; resting against the belly is fine). Without this the policies of Phase 2 and Unified clamped the ball to the chest with 196–236 N, the most reliable way to keep both arms in contact.

     0.5 s counted balls that rolled off afterwards; with 1.0 s every confirmed catch in a static-cradle probe was
     still held 1 s later. Reaching `secure_steps` **confirms** the catch (`confirmed`, `confirmed_now` on that step).
  3a. **Hold phase** (since 2026-10-28): the throw only succeeds once the confirmed catch has lasted
     `hold_after_catch_s` = **3 s** (`catch_complete`, then `ball_caught` ends the episode); the robot should keep the
     ball and stand still, as after a real catch (the baseline walked off with it). A drop during the hold phase is a
     failure (`ball_dropped`); the hold phase runs on if the secured streak breaks without a drop.
  4. **Perception (actor only)**, calibrated on the VICON recordings: every control step the tracker measures the ball
     relative to the root (yaw frame). A measurement is valid once the ball has flown for `detection_delay_range`
     (0.04–0.12 s; before that, e.g. during the wind-up, the ball is invisible), unless a **dropout burst** runs
     (starts with p 0.05 per step, lasts 1–3 steps) or the arms occlude it (`hide_in_catch_zone_prob` 0.5 per throw:
     invisible for its whole stay in the catch box). Measurements arrive `obs_delay_steps_range` (1–5, i.e. up to
     100 ms; 1–3 until 2026-10-27) steps late with noise (5 mm, 0.1 m/s) and a per-throw bias (±1 cm). The real latency
     is not measured yet; 80–120 ms cost the 2026-10-27 baseline (trained on 1–3) 6 points of catch rate. Gaps are bridged by ballistic prediction for 3 steps;
     after that the ball counts as not visible (perceived state 0, `ball_visible` 0). Probe (2026-10-16): in flight
     88.8 % fresh measurements, 8.6 % coasted, 2.6 % not visible; zero noise and fixed delay reproduce the true
     state of 2 steps earlier exactly.
- Rewards (mjlab ch. 9.4.1: small proximity < larger contact < largest retention reward; all ball terms are **zero
  before the launch**):

  | Reward | Weight | Meaning |
  |---|---|---|
  | `ball_reach` | 1.0 | exp(−d²/0.3²), d = ball to midpoint of both hands |
  | `ball_interception` | 2.0 | exp(−d²/0.2² − v²/1.0²), v = ball speed relative to the hands (eq. 9.1, soft catch) |
  | `ball_arm_contact` | 3.0 | fraction of arms touching the ball (eq. 9.2) |
  | `catch_success` | 500 | once per throw at the confirmation (`confirmed_now`), effectively 10 |
  | `ball_secured_hold` | 5.0 | 0.1 per step while secured; no incentive to delay the catch because `ball_caught` is a truncation |
  | `ball_height_penalty` | −5 | ball below 0.5 m |
  | `alive` | 1.0 | 0.02 per step; discounted at most ≈2, below the catch bonus |
  | `termination_penalty` | −200 | `is_terminated_term` for `base_height` / `bad_orientation` only: −4 per fall (catch and drop are not penalized) |
  | `stand_still_root` | 0 → −10 | ramped in over steps 50k → 250k (see Curriculum); before the launch and during the hold phase: \|v_xy\|² + 0.25·ω_z² of the root (baseline while waiting: 0.16 m/s, 15°/s) |
  | `stand_still_feet` | 0 → −5 | ramped like `stand_still_root`; same phases: summed horizontal speed of the feet in firm contact > 50 N (baseline ≈ 0.08 m/s per foot) |
  | `joint_target_excess` | −1.0 | joint targets beyond the clamp (Section 6): the baseline pressed the ankle-roll targets 2–4 rad past the stops |
  | `joint_torque` | −1e-5 | Σ torque², as in Phase 1 / Unified (`energy` ignores static torque; the baseline stood with saturated ankle torque) |
  | `feet_flat` | −2.0 | Σ over the feet in firm contact (> 50 N) of sin² of their tilt (projected gravity xy of `*_ankle_roll_link`); baseline 14–21° on the foot edges |
  | Posture crutches | | `base_height` −10 @0.76 m, `flat_orientation` −5, `joint_deviation_*`, `feet_slide`, `undesired_contacts` −1 (excluding ankles, elbow and wrist links, torso: the ball rests there), further locomotion regularization (`energy`, `action_rate`, `dof_pos_limits`, …) |

  `alive` and `termination_penalty` are required: Phase 2 has no other positive dense reward, so with `alive` 0.15 and
  no fall penalty the early per-step penalties (mostly `action_rate` from the exploration noise, ≈20× `alive`) make
  falling before the throw the best policy (pilot runs 2026-10-22, Section 10).

- Terminations: timeout 20 s (longest regular episode ≈ 8 s wait + 1 s flight + 1.5 s to the confirmation + 3 s hold), `base_height` < 0.2, `bad_orientation` 0.8, **`ball_caught` (success = confirmed catch held through the hold phase, a truncation with `time_out=True`)**, `ball_dropped` (z < 0.1), `ball_missed` (horizontal > 6 m, beyond the 5 m release distance). The longer episodes let `push_robot` (every 4–8 s) fire, which it practically never did with ≈ 4 s episodes.
  - `ball_caught` must stay a truncation: PPO then bootstraps the value of the catch state instead of 0. As a plain termination the catch ended the hold rewards (hold, contact, proximity and `alive`, ≈ 0.2 per step, ≈ 20 discounted at γ 0.99) for a bonus of 10 (≈ 14 in total), so the policy learned to hold the ball and break the secured streak just before `secure_steps` (run 2026-10-22, Section 10). The hold phase adds no new incentive to postpone: completion and timeout both bootstrap, and the bonus comes at the confirmation.
- Curriculum:
  - Throw distribution from easy to final over steps 50k → 250k. Easy: 2.0–2.5 m, ±15°, aim 0.3–0.4 m in front, lateral ±0.03 m, aim height
    0.85–0.95 m, apex 1.85–1.95 m, no velocity noise, `secure_steps` 25, and clean perception (delay 1 step, no
    dropouts, no occlusion), wait 0.5–1.5 s. Final = the values of `CommandsCfg.ball_throw` (`BallCommandCfg` defaults
    with the wider aim of 2026-10-29 and the 0.5–8 s wait), `secure_steps` 50.
  - Stand-still terms ramped in over steps 50k → 250k (with the waits): `stand_still_root` 0 → −10, `stand_still_feet`
    0 → −5. Active from the start, they made a 250-iteration pilot (2026-10-28) learn to fall within 7 steps (Section 10).
  - Posture tapers over steps 100k → 400k: `base_height` −10 → −2.5, `flat_orientation` −5 → −1,
    arm deviation −0.1 → −0.02.
  - The play cfg sets `curriculum=None` and therefore uses the final distribution.
- Runner: `BasePPORunnerCfg` (std 1.0, from scratch). As stage B it is fine-tuned from a **Unified** actor on the
  command line (`--resume --resume_experiment unitree_g1_29dof_ballcatch_unified … --finetune --finetune_std 0.2`,
  see Section 4; tested with a smoke checkpoint): `init_noise_std` in a runner config has no effect after `--resume`,
  because the std is part of the loaded actor state.

### 7.3 Unified: `Unitree-G1-29dof-BallCatch-Unified` (stage A, recommended direction)

- DeepMimic style: Phase 1 tracking rewards **and** Phase 2 ball rewards at the same time. **No** posture crutches
  and no robot reset events (the `MotionCommand` owns resets).
- The actor has the Phase 2 actor contract (perceived ball, no reference): the reference only enters the rewards,
  terminations and the critic, so the policy has to time the catch from the ball, as on the real robot. This makes it
  stage A of the mjlab recipe; stage B = fine-tuning its actor in Phase 2 (Section 4).
- Clips: since 2026-10-29 only the 84 clips with a usable catch (Section 5.2). In a clip without a throw the reference performs its catching motion into the void, which an actor that sees only the ball cannot follow.
- `SyncedBallCommand` (recorded mode): launch at `launch_time = catch_time[clip] ± 0.05 s − flight_time[clip]`, both from the take's recording, so the ball arrives when the human caught it and the human's reaction to the release stays consistent. The release point is sampled as in Phase 2 (distance, direction, height, drag); at the launch the target is the catch point re-anchored to the robot exactly like the tracked bodies (robot anchor xy and yaw, reference anchor height) plus ±0.05 m per axis.
  - Catch point since 2026-10-29: the clip's wrist midpoint at the catch moment raised by `CATCH_ABOVE_HANDS` = 0.08 m (`catch_data(aim_above_hands=...)`): 0.19 m in front of and 0.24 m above the pelvis (median), all inside the catch box. Before, it was the recorded ball position, which is in the human's scale while the clip is in the robot's: 0.21 m from the clip's wrists (0.13 m above them), 0.33 m in front of and 0.30 m above the pelvis, up to 0.57 m high (above the catch box). Aimed there, the robot had to leave the reference to catch and learned to clamp the ball against its upper chest and head (first run, Section 10). Kinematic replay (robot = reference, same conditions): the ball comes within 0.14 m of the reference wrist midpoint around the arrival (median; 0.23 m with the old target) and is 0.18 m from it at the arrival time (0.25 m); the rest is the intended jitter (±0.05 m, ±0.05 s), the moving hands and collisions with the reference arms.
  - Same ball, drag, perception, catch check, ball rewards, `undesired_contacts` exclusions and throw curriculum as Phase 2 (Section 7.2); in recorded mode the curriculum's target and apex ranges have no effect.
  - **Wait-and-hold cycle** (since 2026-10-28, `MotionCommandCfg` `start_hold_s`, `start_hold_prob`, `hold_from_s`, `end_hold_s`): every clip is preceded by its first frame held for 6 s (lead-in, zero velocities), cut 0.3 s after its recorded catch (in 33 of 86 clips the person already puts the ball down by the clip end) and that pose held for 5 s (3 s hold phase plus up to 2 s until the confirmation; if the clip ended first, the robot would be reset onto the next cycle before the catch completes). In fixed mode (no usable recordings) the clips keep their full length. Catch times are shifted by the lead-in.
  - 35 % of the cycles start uniformly inside the lead-in, the others by adaptive sampling over the whole extended cycle (nothing fails in the static lead-in, so the adaptive sampling alone would drop it). Since 2026-10-29 no cycle starts after its clip's launch (`MotionCommandCfg.latest_start_s` = launch − 0.1 s; later starts are drawn again, uniformly before it), so every cycle gets its ball. Until then 45 % of the cycles that reached their clip end got no ball (started after the launch or clip without a usable catch) while the reference performed its catch: 58 % of the tracking terminations of the second run happened in such cycles, and the adaptive sampler, which counts tracking failures where they happen, starts more cycles exactly there (likely part of why both runs fell from ≈ 25–30 % to ≈ 12 % completed catches after iteration ≈ 5000, while the throw curriculum got harder). Kinematic replay with the change (256 robots × 30 s, robot = reference): every cycle that reaches its clip end throws (55 % before); only cycles cut by the episode end before their launch stay without a ball, which is harmless because the reference has not started its catch yet; 0.11 throws per robot and second (0.09 before), wait from the cycle start to the launch median 3.2 s, up to 7.2 s; the ball comes within 0.16 m of the reference wrist midpoint around the arrival (median). Episodes last 20 s.
  - Catch data per clip is computed when the config is imported (`motion_data.catch_data`, a few seconds of numpy) and passed to both commands (`hold_from_s`, `SyncedBallCommandCfg.catch_data`), so `params/env.yaml` records it; `SyncedBallCommand` checks that the G1 constants it was computed with match the asset.
  - Hold phase as in Phase 2 (`hold_after_catch_s` 3 s). No stand-still terms: the static lead-in and the held catch pose train standing still through the tracking rewards.
  - A new cycle (reset or clip end) is detected through `MotionCommand.resample_count`, samples a fresh throw and re-parks and re-arms the ball.
  - Fixed mode (no usable recordings, e.g. the old single clip in `phase1/motions/`): one catch time for all clips (`FALLBACK_CATCH_MOTION_TIME = 2.0`, fits the old overhand clip), target and flight time sampled from the `BallCommandCfg` ranges.
  - **`motion` must be declared before `ball_throw` in `CommandsCfg`.**
- Terminations: Phase 1 tracking terminations (`ee_body_pos` loosened to 0.3) plus Phase 2 ball terminations (`ball_caught` = completed hold phase, a truncation, as in Phase 2). The adaptive sampler counts only real terminations as failures, a completed catch is none. Tracking failures are counted in the bin where they happen (BeyondMimic: start more often where the motion is hard). Since 2026-10-28 dropped and missed balls are counted in the bin where the failed cycle started (`start_bin_failure_terms`), so a clip where the ball falls gets more cycles from that start point; counted where the ball falls, they would raise bins after the launch, where no cycle starts since 2026-10-29. Each clip still gets 1/84 of the cycles, the sampler only moves the start time within a clip, and every cycle samples a new throw, so a hard throw is never repeated as such.
- Curriculum:
  - Throw distribution 50k → 250k.
  - Body tracking tapers (pos/ori 1.0 → 0.4, vel 1.0 → 0.5) over 150k → 500k.
  - Anchor tracking keeps full weight.
- 6-DoF pushes every 2–5 s. Runner: `BasePPORunnerCfg` (std 1.0), from scratch (Phase 1 checkpoints do not fit
  the actor).
- Regularization: Phase 1 set (`joint_torque`, `action_rate`, `joint_limit` −10, `joint_target_excess` −1, …) and, since 2026-10-29, `feet_flat` −2 as in Phase 2: tracking the feet's orientation alone left the left foot on its edge (17° in 84 % of the contact steps at iteration 7000 of the first run), because the orientation reward averages over 14 bodies with a wide kernel.
- Before serious training: run Phase 1 on the clips as a trackability check (legs).

## 8. MDP Building Blocks (`BC/mdp/`)

| File | Contents |
|---|---|
| `commands/motion_command.py` | `MotionLoader` (all clips as padded tensors (clip, frame, …), per-clip lengths, one frame rate; optional wait-and-hold cycle: lead-in `start_hold_frames`, cut at `hold_from_s`, end hold, held frames with zero velocities; `clip_lift` adds a height to all bodies), `MotionCommand(Cfg)`: reference states of each env's clip, per-clip adaptive bin sampling plus a fixed share of starts in the lead-in, no start after a clip's `latest_start_s`, failures of `start_bin_failure_terms` counted at the cycle start, `clip_ids`, `resample_count`, `cycle_start_steps`, metrics `error_*`, `sampling_*` |
| `commands/ball_command.py` | `flight_time_from_apex`. `BallCommand(Cfg)`: throw sampling (frozen at reset), parking, drag-compensated launch, air drag, catch check, perception model, `secured_steps`, `confirmed`, `confirmed_now`, `steps_since_confirm`, `catch_complete` (hold phase), log-uniform wait option, `arm_contact`, `torso_force` (ball against the torso; with `max_torso_force` a squeezed ball is not held), `launched`, `perceived_pos_b`, `perceived_vel_b`, `ball_visible`. Metrics `ball_height`, `ball_secured_time`, `hold_time_after_confirm`, `hand_ball_distance`, `ball_in_catch_zone`, `ball_arm_contact`, `ball_torso_force`. Helpers `_sample_throw`, `_write_parked_state`, `_launch`, `_solve_launch_velocity`, `_drag_displacement`, `_apply_drag`, `_update_arm_contact`, `_update_catch_state`, `_reset_perception`, `_update_perception` |
| `commands/synced_ball_command.py` | `SyncedBallCommand(Cfg)` (Unified only): launch coupled to the motion phase; recorded mode with per-clip catch time, flight time and re-anchored catch point from `catch_data` (computed in the env config from the take's VICON recording, shifted by the lead-in), fixed mode with one catch time; `_start_cycle` = new throw per motion cycle |
| `observations.py` | Motion obs (`motion_anchor_*`, `robot_body_*`), true ball obs for the critic (`ball_pos_vel_b`, `ball_to_hands_b`, `ball_intercept_b`, `hand_pos_b`), perceived ball obs for the actor (`ball_perceived_pos_vel_b`, `ball_visible`, `ball_perceived_intercept_b`), shared `_ballistic_intercept`, `dummy_zeros` |
| `actions.py` | `ClampedJointPositionAction(Cfg)`: `JointPositionAction` with the targets clamped to the hard joint limits (via `clip`, exported to `deploy.yaml`), `target_excess` per env |
| `rewards.py` | Tracking exp rewards, `energy`, `joint_target_excess`, `feet_flat`, `ball_interception_exp` (eq. 9.1, distance-only without `std_vel`), `ball_arm_contact`, `ball_caught_bonus` (once per throw), `ball_secured`, `stand_still_root`, `stand_still_feet` (waiting and hold phase), `ball_height_penalty` |
| `terminations.py` | Tracking terminations (`bad_anchor_*`, `bad_motion_body_pos*`), `ball_caught` (= `catch_complete`), `ball_below_height`, `ball_far_from_robot` |
| `events.py` | `randomize_joint_default_pos`, `randomize_rigid_body_com` |
| `../motion_data.py` | `find_motion_data` (search order), `is_lfs_pointer`, `ball_file_for`, `align_recording`, `compare_with_human` (clip arms vs the human of the recording), `compute_catch_info` / `catch_infos` (recording ↔ clip alignment, validity limits), `catch_data` (what the Unified config passes to its commands; optional `lift` of the catch points and `aim_above_hands`: aim at the clip's wrists instead of the recorded ball), G1 asset constants `G1_JOINT_NAMES`, `G1_HAND_BODY_INDEXES`, `G1_ARM_POINTS`, temporary `TEMP_CLIP_LIFT` (numpy only) |
| `curriculums.py` | `modify_reward_weight_linear`, `ball_throw_curriculum` (linear interpolation over `common_step_counter`; int fields such as `secure_steps` are rounded) |

`ball_intercept_b`: intersection of the (descending) ball parabola with the plane 0.15 m above the root.
Output `[x_b, y_b, t_go]`, with `t_go` clamped to [0, 3] s. If the plane is never reached, `t_go = 0` and the
current position is returned. It ignores air drag on purpose (same formula on hardware), so early in the flight it
predicts the 15 cm ball up to ~0.3 m too far; the error shrinks with `t_go`. The actor variant
`ball_perceived_intercept_b` applies the same function to the perceived state and is zero while the ball is not visible.

## 9. Invariants & Pitfalls

1. **Observation contracts (Section 6.1):** the actor terms of Phase 2 and Unified must stay identical (order, dim,
   scale, noise), otherwise `--finetune` from Unified into Phase 2 breaks. The critic slot layout is shared by all
   three tasks; mirror critic changes there too. Phase 1's actor is separate (reference instead of ball).
   Checkpoints from before 2026-10-16 (actor 875) do not fit any task anymore.
2. **Action scale** is `UNITREE_G1_29DOF_MIMIC_ACTION_SCALE` in all tasks. A future deploy config must use the same.
3. Isaac Lab multiplies rewards by `weight × dt` (dt = 0.02). One-time bonuses are therefore weighted as `value / dt`.
4. Isaac Lab step order: terminations → rewards → reset → command update. Command thresholds and termination
   thresholds must not contradict each other (this bug was fixed in P0).
5. `undesired_contacts` must exclude every body the ball may rest on (forearm/hand links, torso; later: hand
   links), otherwise catching is penalized.
6. Checkpoints from before 2026-08-10 (pre-P0/P1: different obs scales, no `ball_intercept`) are **incompatible**.
7. For play and evaluation set `curriculum=None`, otherwise evaluation runs on the easy initial distribution.
8. Curriculum steps assume about 30k iterations (24 steps per iteration). Scale them for other run lengths.
9. GMR `.pkl` files interleave the hands with the arms (see Section 5). Never slice `dof_pos[:, :29]`; use
   `select_body_joints()` from the conversion scripts.
10. `--resume` restores actor (incl. noise std), critic, optimizer and iteration unless `--finetune` is given, and then sets `common_step_counter` to iteration × `num_steps_per_env`, so the step-based curricula continue where the checkpoint left off (with `--finetune` they start over). `--max_iterations` counts additional iterations. `--experiment_name` decides where the new run is logged; `--resume_experiment` where the checkpoint is read from. The first logged iteration after a resume still shows the curriculum values of the initial reset.
11. Isaac Lab reports filtered contacts only **one-to-many**: the `ball_contact` sensor sits on the ball and filters
    against the arm links. `BallCommand` assigns the filter bodies to the arms by their `left_` / `right_` name prefix
    (column order verified with lateral throws on 2026-10-16).
12. The ball mass randomization must stay a **startup** event: `BallCommand` reads the per-env masses once (first
    launch) for the drag-compensated aim.
13. `ball_throw_curriculum` sets `BallCommandCfg` fields with `setattr`; a misspelled key is silently ignored.
14. `ball_missed` (`max_distance`) must exceed the largest release distance, otherwise it fires while the ball is
    still parked.
15. The perception ring buffer is sized from the default `obs_delay_steps_range` (the largest delay, 5 steps). The
    curriculum may only shrink the delay range; larger delays are clipped to the buffer. Scenarios with a longer delay
    (`eval_catch.py --scenario delay`) set the range before the env is built.
16. A success termination after which positive per-step rewards would keep flowing (hold, contact, `alive`, tracking) must be a truncation (`time_out=True`). As a plain termination it cuts those rewards, and the policy learns to postpone the success instead of completing it (Phase 2 run 2026-10-22).
17. Reference clips must run at the control rate (50 Hz, one frame per control step) and use the Isaac joint and body order of our G1 asset; `MotionCommand` refuses other frame rates. The dataset's `.npz` files are Git LFS objects: without `git lfs pull` only pointer files are checked out, which `motion_data.py` detects.
18. A VICON recording and its clip are two retargets of the same take with different time windows and frames; use the catch data only through `compute_catch_info` (alignment), never by frame index. Check new clips with `check_npz_fk.py` (joint and body order against the asset) and `check_dataset.py` (arm motion against the human).
19. Joint targets are clamped to the hard joint limits (`ClampedJointPositionAction`, Section 6). Without the clamp the policy can press a joint against its stop with a target far beyond it: the PD controller then delivers saturated torque at no visible cost (Phase 2 baseline, ankle roll). Keep the action term's `clip` set (it carries the clamp into `deploy.yaml`), and keep `joint_target_excess`, otherwise the raw actions behind the clamp can grow without effect.
20. The adaptive sampling of `MotionCommand` draws start times where episodes fail. A static part of a cycle (the lead-in) never fails, so without a fixed share of starts (`start_hold_prob`) it would vanish from training, and with it the long waits.
21. In Unified the held catch pose (`end_hold_s`) must outlast the confirmation plus the hold phase (`hold_after_catch_s`). If the clip ends first, the robot is reset onto the next cycle and the catch never completes.
22. Penalties that grow with motion (quadratic stand-still terms) are large for an untrained, flailing robot. Active from the start they make an early fall cheaper than flailing, and the policy learns to fall at once (Phase 2 pilot 2026-10-28, as with the missing fall penalty on 2026-10-22). Ramp such terms in once the robot can stand.
23. In Unified a failure must be counted where a restarted cycle can practise it again. The adaptive sampler restarts cycles in the bin of a failure, but cycles start only before the launch (pitfall 25): dropped and missed balls therefore belong to the bin where their cycle started (`MotionCommandCfg.start_bin_failure_terms`). New ball terminations must be added to that list.
24. The clips of the current dataset sit 4.5 cm too low and are lifted when loaded (`motion_data.TEMP_CLIP_LIFT`, Section 5.2). The lift belongs to this dataset only: remove it together with a re-exported dataset (grep `TEMP_CLIP_LIFT`), and lift anything else that lives in the clip frame (like the recorded catch points) by the same amount as long as it exists.
25. The Unified actor sees no reference, only the ball. Every reference motion it has to track must therefore be predictable from the ball: a cycle in which the reference catches without a ball (a start after the launch, a clip without a usable catch) cannot be tracked, ends by tracking terminations and, through the adaptive sampling, gets started more often.
26. The G1 USD of `unitree_model` has no collision shapes on `elbow_link` and `wrist_pitch_link`, so in Isaac the ball passes through the forearms (the static probe of 2026-10-16, in which a ∅10 cm ball fell through the forearm cradle, and 0 % elbow and wrist pitch contacts in every evaluation are this). The real robot and both MJCFs of `unitree_rl_mjlab` have solid forearms. Policies trained on this model have never felt a ball on the forearm; keep it in mind for Sim2Sim and the real robot, and re-check with `sim2sim/tools/dump_isaac_model.py` when the asset changes. It is the USD's own choice, not a conversion error: its base layer holds empty collider prototypes for these links and its physics layer leaves their `collisions` prims empty, while Unitree's URDF (`unitree_ros`, `unitree_rl_gym`) gives both links their mesh as collider; the USD also swaps the knee mesh for two cylinders, adds a box and a toe sphere to the feet and makes the rubber hand a collider. `tasks/ball_catching/robot_assets.py` (`FOREARM_COLLIDERS`) adds each link's visual mesh as convex-hull collider with the settings of the existing wrist roll collider, i.e. what the URDF has (checked 2026-10-29, changelog).
27. `params/deploy.yaml` (written by `utils/export_deploy_cfg.py` at the start of training) rounds every value to three significant digits, and until 2026-10-29 it exported env 0's randomized default pose (startup event `randomize_joint_default_pos`, ±0.01 rad) as `default_joint_pos` and as the action offset. The export now takes `default_joint_pos_nominal`; files of earlier runs (including the third Unified run) still hold the shifted pose, so take the default pose from the robot config for deployment. `sim2sim/run.py` takes default pose, gains, limits and action scale exactly from the Isaac dump and only compares them with `deploy.yaml`.

## 10. Training History & Artifacts

| Run | Task | Motion | Result |
|---|---|---|---|
| `ballcatch_phase1/2026-07-27_13-32-10` | Phase 1 | Gangnam placeholder | model_99 (smoke test, outdated, incompatible) |
| `ballcatch_phase1/2026-07-27_13-41-06` | Phase 1 | – | empty / aborted |
| `ballcatch_phase1/2026-08-31_14-45-09` | Phase 1 | 143_merged_filtered | model_1 (test run) |
| `ballcatch_phase1/2026-08-31_15-06-41` | Phase 1 | 143_merged_filtered | model_1500, 1657 iterations, tracking converged, but on the corrupt clip (see Section 2) → unusable |
| `ballcatch_phase2/2026-10-22_16-04-10` (workstation) | Phase 2 | – | 4096 envs, stopped after iteration 7000 (`model_7000.pt`): learned to hold the ball without completing the catch (below); continued with the truncation fix |
| `ballcatch_phase2/2026-10-22_21-14-31` (workstation 1) | Phase 2 | – | resumed from `model_7000.pt` with the truncation fix; slowed to 11 s per iteration by a second job on the GPU, ended at ≈ iteration 8935 when the tmux session was closed; last checkpoint `model_8500.pt` |
| `ballcatch_phase2/2026-10-23_03-35-20` (workstation 2, RTX 4080) | Phase 2 | – | resumed from `model_8500.pt`, ran to `model_29999.pt` (2.7 s per iteration); **current Phase 2 baseline**, evaluated on 2026-10-27 (below) |
| `ballcatch_phase1/2026-10-28_14-28-00` (workstation 2) | Phase 1 | 108 dataset clips | 10k iterations, 4096 envs, 4.9 h (1.5 s per iteration), code `71c1767` (before fixes 1–4). End of training: 99.4 % of the episodes run the full 30 s, body error 6.2 cm. Per-clip check (every clip once from its first frame, deterministic, no pushes or noise): no clip triggers a tracking termination, body error 5.0 cm (worst clip 5.7 cm), hands and feet at most 6.8 cm (worst 13 cm) off in height, feet in firm contact slide 1.2 cm/s, knees at 9.7° (reference 1.3°), torso and feet 4.5 cm above the reference (height offset of the clips, Section 5.2); deep squats (clips 28, 41, 15) are tracked less deep; ankle targets beyond the hard limits in 0.2 % of the steps (single clips up to ≈ 13 %). Side-by-side video, contact sheets, per-clip table and scripts: `analysis/2026-10-28_phase1/` on Marius' laptop |
| `ballcatch_unified/2026-10-28_21-24-50` (workstation 2) | Unified | 89 dataset clips (84 with throws) | First run with fixes 1–4, option B and the height correction; 2.2 s per iteration, stopped at iteration 8000 (`model_8000.pt`). Mean of iterations 4000–4500: 31 % of the episodes end with a completed catch, 62 % by `ee_body_pos`, body error 13 cm; after iteration 5500 the curves became unsteady while the throw curriculum got harder. Check of `model_7000.pt`: 92 % of the confirmed catches complete the 3 s hold phase, 1 % are cut by the clip end (`END_HOLD_S` suffices); in the video the robot clamps the ball between hands and head (ball 0.32 m above the pelvis, torso contact 196 N), staggers under the pushes and stands with its left foot on the edge (17°). Led to the changes of 2026-10-29 (Section 7.3); results and video in `analysis/2026-10-29_unified_7000/` on Marius' laptop |
| `ballcatch_unified/2026-10-29_03-31-07` (workstation 2) | Unified | 89 dataset clips (84 with throws) | Second run: throws aimed at the reference hands, `feet_flat`; 2.6 s per iteration, ended at iteration 6106 when workstation 2 crashed (last checkpoint `model_6000.pt`). Mean of iterations 4000–4500: 25 % of the episodes end with a completed catch, 64 % by `ee_body_pos`, body error 13 cm; after iteration 5000 the completed catches fell to 12 % and `ee_body_pos` rose to 81 %, as in the first run. Check of `model_6000.pt`: feet flat (left foot 4.7°, was 17°), 91 % of the confirmed catches complete the hold phase, 2 % are cut by the clip end; the ball is still pressed against the chest (0.35 m above the pelvis, torso force 227–233 N, the reference holds its wrists 0.15 m above the pelvis); 58 % of the tracking terminations happen in cycles without a ball, 36 % while waiting for it, 0 % while holding, mostly the left wrist below the reference; the pushes cause about half of the steps (0.71 lift-offs per second while waiting, 0.36 without pushes). Led to the changes of 2026-10-29 (every cycle with a ball, no chest clamp; Sections 7.2, 7.3); results and videos (with and without pushes) in `analysis/2026-10-29_unified2_6000/` on Marius' laptop |

As of 2026-10-16 both `logs/rsl_rl/unitree_g1_29dof_ballcatch_phase1/` and `…_phase2/` are empty: the runs above no
longer exist, and there is **no ball-catching checkpoint**. Smoke runs (64 envs, 5 iterations) of all three tasks
after the 2026-10-16 changes were logged outside the repo.

**Pilot runs 2026-10-22** (Marius' RTX 3080, logged outside the repo, no checkpoints kept):

| Run | Envs / iterations | Result |
|---|---|---|
| Phase 2 as of `24d448d` (`alive` 0.15, no fall penalty) | 4096 / 108 (stopped) | 100 % of the episodes end by `bad_orientation`; episode length 12.5 → ≈9 steps while the mean reward rises: the policy learns to fall before the throw, all ball rewards stay 0 |
| Variant B: + fall penalty −200 | 2048 / 250 | episode length 86 steps, 76 % falls, 25 % reach the throw (ball dropped) |
| Variant A: + fall penalty −200, `alive` 1.0 (**adopted**) | 2048 / 250 | episode length 104 steps, 15 % falls, 86 % reach the throw, hands–ball distance 2.1 → 0.7 m, no catches yet |
| Unified, unchanged config, overhand clip | 2048 / 250 | not affected: episode length 12.8 → 6.3 → 136 steps, mean reward −1.03 → +3.30 (dense tracking rewards); episodes still end by `ee_body_pos` / `anchor_pos` |

**Pilot runs 2026-10-28** (workstation 2, 2048 envs, 250 iterations from scratch, fixes 1–4 of branch `fixes-2026-10-28`, logged outside the repo; values are means over iterations 230–250):

| Run | Episode length (steps) | Falls | Episodes that reach the throw |
|---|---|---|---|
| Stand-still terms at full weight from the start | 14.6 | 100 % | 0 % (episode length 49 → 7 steps by iteration 50: the policy learns to fall at once) |
| Stand-still weights 0 (= the ramped curriculum before step 50k, **adopted**) | 95.6 | 32 % | 69 % |

Unified with the same setup (fixes 1–4, 2048 envs, 250 iterations) shows no such collapse: episode length 34 → 7 steps (iterations 40–110) → 57 steps at iterations 230–250, episodes end by the tracking terminations (`ee_body_pos`, `anchor_pos`). For comparison, the running Phase 1 run on the same clips (4096 envs, its actor sees the reference) was at 110 steps after 250 iterations.

Timing: 2.85 s per iteration at 4096 envs (≈24 h for 30k iterations on an RTX 3080), ≈6.5 GB GPU memory. Two Isaac
Sim trainings at once do not fit into 10 GB.

**Phase 2 run 2026-10-22** (workstation RTX 3090, 4096 envs, `26b64a3`; logs and checkpoints in `~/unitree_rl_lab/logs/rsl_rl/` on the workstation; 2.65 s per iteration, ≈ 7 GB GPU memory). Values are 50-iteration means:

| Iteration | `ball_caught` | `time_out` | Episode length (steps) | `Episode_Reward/ball_secured_hold` |
|---|---|---|---|---|
| 2000 | 73 % | 0 % | 192 | 0.34 |
| 2500 | 77 % | 1 % | 281 | 0.72 |
| 3000 | 63 % | 14 % | 439 | 1.35 |
| 4000 | 55 % | 22 % | 538 | 1.80 |
| 6000 | 56 % | 30 % | 578 | 2.06 |
| 6927 | 54 % | 32 % | 608 | 2.20 |

At iteration 6900 falls were ≈ 3 %, drops ≈ 11 %, misses 0, and the ball ended ≈ 88 % of the episodes in the catch box. A throw reaches the robot after ≈ 2.5 s, yet the episodes grew to ≈ 12 s and a third ran into the 20 s limit with the ball in the arms: on average the ball was secured ≈ 460 steps per episode, while a confirmed catch needed 39 in a row at that curriculum stage. Cause and fix: Section 7.2 (`ball_caught` as a truncation). The run was continued from `model_7000.pt` with the fix and the curriculum-aware resume.

**Phase 2 baseline, final evaluation 2026-10-27** (`model_29999.pt`, deterministic policy, play config; 512 robots × 60 s per scenario with the predecessor of `scripts/rsl_rl/eval_catch.py`). Training catch rate (stochastic) 92–95 % from iteration ≈ 15k on; `Policy/mean_std` rose from 0.69 to 0.93 during the run. Full findings, ranking of fixes and the method: `context/analysis/2026-10-27-phase2-final.md`; videos, raw JSON and the script are kept outside the repo in `analysis/2026-10-27_phase2_final/` on Marius' laptop.

| Scenario | Caught | Dropped | Fell |
|---|---|---|---|
| Base (play config) | 99.6 % | 0.4 % | 0.03 % |
| Throw after 6–9 s of waiting (training: 0.5–1.5 s) | 27.9 % | 43.9 % | 28.3 % |
| Pushes ±0.5 m/s every 1–2 s | 93.6 % | 6.3 % | 0.2 % |
| Perception latency 4–6 steps (training: 1–3) | 93.6 % | 6.3 % | 0.07 % |
| Throws aimed ±0.3 m sideways (training: ±0.1 m) | 99.5 % | 0.5 % | 0.06 % |
| Friction fixed high (1.0/0.8) or low (0.4/0.3) | 99.7–99.8 % | 0.2–0.3 % | ≤ 0.1 % |
| No catch termination, 12 s episodes | 99.4 % still hold the ball at the end | 0.1 % | 0.6 % |

- Joint-stop exploit: the ankle-roll targets lie 2–4 rad outside the joint range in 98–99 % of the steps; the joints sit on their stops with torque above 80 % of the limit during 100 % of the waiting time, and the feet stand tilted by 14–21° (on their edges). Real gliding is rare (< 1 % of the contact steps); the visible "sliding" is pivoting on the foot edges, tapping (1.9 lift-offs per second while waiting, 7.7 while catching) and drifting (≈ 15° rotation per second while waiting, 33 cm per catch). Cause: unclipped actions in all three tasks, and Phase 2 has no torque penalty (only `energy`, which ignores static torque).
- Hold: the ball touches only the hand links but is pressed against the left chest (0.20 m in front of, 0.14 m left of the pelvis) with a mean torso contact force of 236 N; the right arm crosses over, right shoulder yaw and wrist pitch are above 80 % of their torque limit for most of the hold. In 12 s episodes the robot keeps the ball but walks off by ≈ 4 m.
- Posture: waits in a deep knee bend (72° instead of the default 38°) with abducted arms; bends the pelvis forward by 34° while holding.
- `push_robot` (every 4–8 s) practically never fired: the interval timer restarts at every reset and episodes last ≈ 4 s since the truncation fix.

## 11. Open Items & Next Steps (prioritized)

1. **Fix the Phase 2 baseline, then retrain** (ranking and reasons in `context/analysis/2026-10-27-phase2-final.md`, implementation plan for fixes 1–4 in `context/planning/2026-10-27-fixes-unified-phase2-plan.md`; in `main` since 2026-10-28).
   - Implemented on 2026-10-28, tests pending: perception latency 1–5 steps (fix 3); recordings whose clip arms deviate from the human are no longer used (fix 4, verified: 84 of 108); joint targets clamped to the hard joint limits in all tasks with an excess penalty, Phase 2 additionally with torque and flat-feet terms (fix 2, smoke runs only); long waits and a 3 s hold phase after the confirmed catch in Phase 2 (log-uniform 0.5–8 s wait, stand-still terms, bonus once per throw) and Unified (wait-and-hold cycle), which also lets the pushes fire within episodes (fix 1).
   - Implemented on 2026-10-29 (Phase 2 and Unified): a ball pressed against the torso with more than 50 N is not held (Section 7.2), against the chest clamp of the baseline.
   - Implemented on 2026-10-29 (Phase 2): throws aimed up to ±0.3 m to the side, 0.25–0.45 m in front and up to 1.15 m high (Section 7.2), against throws that only arrive where the robot stands; the `lateral` scenario of the evaluation now aims ±0.45 m.
   - Still open: lower `entropy_coef`; a ready pose before the throw.
   - 15–20k iterations suffice. Evaluate with the stress-test scenarios, not the training catch rate.
2. **Measure the real ball:** diameter, mass, restitution (drop test, e = √(h_bounce / h_drop)), friction and how it behaves when squeezed slowly between two surfaces (Sim2Sim 2026-10-29: the policy's catches depend on the wedged ball staying put). Replace
   the placeholder restitution 0.4 / DR 0.2–0.6 and confirm the mjlab values (∅15 cm, 112 g; the drag of the
   recorded throws fits such a ball). Static probes on 2026-10-16 showed that restitution decides how many throws
   stay in the arms (static cradle: 8/45 at e = 0.6, 21/45 at e = 0.3, 32/45 at e = 0), and that a ∅10 cm ball
   falls through the forearm cradle.
3. **Train with the underhand clips** (integrated 2026-10-27, Section 5.2): Phase 1 trackability check done on 2026-10-28 (all 108 clips tracked, Section 10). Unified (stage A): two runs stopped (Section 10); third run with throws aimed at the reference hands, `feet_flat`, every cycle with a ball and no chest clamp. Then fine-tuning of its actor in Phase 2 (stage B), A/B against Phase 2 from scratch. Knees: keep `joint_limit` as is; the robot tracks with ~10° bent knees at a negligible penalty.
   - Height offset of the clips: corrected temporarily when loading (`motion_data.TEMP_CLIP_LIFT`, Section 5.2). The proper fix is a re-export of the dataset with the soles on the ground (to be raised with the data group, who also train on it in Maik's task); then remove `TEMP_CLIP_LIFT`.
4. Refine the throw and perception calibration with the new session's ball CSVs (Section 5.1 used 12 throws of
   2026-07-06 from one thrower at a fixed distance). Measure the latency of the live pipeline (VICON → MQTT →
   robot) to replace the assumed 1–5 steps.
5. Remaining review findings L1–L4 in `context/REVIEW.md` (C1, H1, H2, M1–M4 and L5 are resolved).
6. Evaluate every run with the suite (`scripts/rsl_rl/eval_catch_all.sh`, Section 4) instead of the training catch rate, and decide on that basis which line (Phase 2 from scratch or Unified → Phase 2) to pursue. For Unified, `wait` starts every cycle at the beginning of its lead-in.
7. End effector: the rubber hands (collider merged into `*_wrist_yaw_link`, 0.17 kg) stay for now. Decide what is
   mounted on the real robot (Inspire FTP hand ≈ 0.88 kg per hand per the `unitree_ros` URDF, or the 3D-printed
   mount from the proposal) and model its mass and collision geometry.
   - Forearm colliders: the USD has none on `elbow_link` and `wrist_pitch_link` (pitfall 26), the real forearms are solid. Implemented and checked as an option on 2026-10-29 (`FOREARM_COLLIDERS`, off): decide at iteration ≈ 6000 of the third Unified run whether to restart with it; at the latest the Phase 2 fine-tuning before the real robot needs it.
8. Sim2real hardening:
   - ball contacts: in the probe of 2026-10-29 (stiff arms, 200 Hz physics) the ball sinks 1–2 mm into the real surface at gentle contacts but 15–23 mm into the hand at 4.7–6.5 m/s impacts, the speed range of real throws (changelog). Check contact offsets, CCD for the ball or a shorter physics step if Sim2Sim shows catches that rely on it
   - actuator delay as DR
   - states the real robot sees: standing after a miss (long waits and holding after the catch are trained since 2026-10-28)
   - no-walking constraint (foot displacement) and a fall penalty
9. Sim2Sim in MuJoCo (`sim2sim/`, since 2026-10-29):
   - Done: environment on both workstations (identical simulation results), G1 MJCF, comparison with the Isaac model; MuJoCo model with exactly the Isaac collision shapes (`model.py`, also with solid forearms); `run.py` with PD loop and robot-side actor observations, ONNX export (`tools/export_onnx.py`). Standing test 2026-10-29: `model_3000.pt` of the third Unified run stands 20–30 s from the default pose and from the first frames of 8 clips (pelvis 0.78 m, torso tilt 7–8°, drift 2–6 cm, almost no steps; Isaac without pushes: 0.77 m, 9°, 0.6 lift-offs per second), results in `analysis/2026-10-29_sim2sim_stehen/` on Marius' laptop.
   - Done 2026-10-29: ball observations ported to numpy (`sim2sim/ball.py`: VICON model and intercept) and checked against Isaac (`tools/record_ball_obs.py`, `tools/check_ball_obs.py`): with a deterministic VICON model the port gives the same `ball_state`, `ball_visible` and `ball_intercept` as Isaac in every step (16 robots × 1000 steps, largest difference 3e-6 m; with a wide catch box also 170 coasted and 350 lost steps).
   - Done 2026-10-29: throws in MuJoCo (`run.py --throws`): Phase 2 lifecycle (throw sampled in the heading frame and launched with the drag-compensated velocity, drag as an external force, VICON model, catch check with both arms, catch box and torso force, hold phase, Phase 2 terminations), ports in `ball.py` (`Throw`, `CatchState`). Result with `model_4500.pt` on the Phase 2 throw distribution (Isaac `base` without pushes: 44 % caught, 38 % dropped, 18 % fallen): with a damped ball contact (damping ratio 0.7, drop bounce e = 0.14) 23 % caught, 68 % dropped, 10 % fallen; 1.0 (e = 0.08) 30 / 58 / 12 %; with the damping ratio that matches Isaac's free bounce (0.4, e = 0.36) 0 % and 51 % fallen. The robot wedges the ball low between forearms, hands and belly in both simulators; PhysX applies restitution only above its bounce threshold, so the slowly squeezed ball stays, while MuJoCo's contact springs at every speed and shoots it out (180–250 N within one physics step), and the bent-over robot falls; a ball that only exists for the perception never makes it fall. Confirmed catches are also lost in the hold phase more often in MuJoCo (23–31 % against 1.5 %). Results and videos in `analysis/2026-10-29_sim2sim_wuerfe/` on Marius' laptop.
   - Next: repeat with later checkpoints; the variant with solid forearms; measure how the real ball behaves when squeezed (Section 11, item 2), and consider training with restitution also for slow contacts (PhysX bounce threshold) if the real ball is springy.
   - Then: ONNX export, deploy config and C++ ball
   observation terms (`ball_state`, `ball_visible`, `ball_intercept`). The joint-target clamp is already covered: `deploy.yaml` carries the joint limits as `clip`, which the C++ `JointAction` applies (check it on the real robot). Build the live VICON pipeline so that it
   produces what the perception model assumes:
   - robot base tracking; ball relative to the root in the yaw frame (**identical** math to `observations.py`)
   - release detection (ballistic fit) before the ball becomes visible; nothing during the wind-up
   - ballistic filter with air drag, bridging gaps of up to 3 control steps, then "not visible"
   - latency forward prediction (then shrink `obs_delay_steps_range` to the residual latency)
10. Curate the clips: `MotionCommand` loads all of them with equal weight. Phase 1 tracks all 108 (2026-10-28), so none has to go for trackability; weighting stays an option.
11. Later, a Phase 3 (stage C) for throws beyond the arms' reach (decided 2026-10-29, not started): fine-tuning on top of the Phase 2 policy with throws 0.4–0.8 m off the centre, foot motion allowed during the flight and a stable stance required again for the hold phase, so that Phase 2 stays the safe policy for the real robot. Not part of Phase 2: lunges bring a fall risk and different rewards. First clarify:
   - whether real throws need it (test throws with VICON, meeting topics)
   - where the motion comes from: no take of the dataset contains a step; RL alone tends to find ugly steps, a few recorded catches with a lunge would give a reference

## 12. Context Documents

`context/` (versioned) contains:
- `unitree_rl_lab_spec.md`: this document
- `pipeline/PG-Antrag-H-ReACT.pdf`: official project proposal with the minimum goals
- `pipeline/H-React.drawio(.xml)`: architecture diagram of the overall pipeline
- `pipeline/LATENT_pipeline.md`: reference extracted from the LATENT tennis paper
- `protocols/`: weekly meeting minutes from 22 May to 10 Aug 2026 (German; team split, status of other subgroups)
- `planning/g1_catching_optimization_plan.md`: analysis, findings F-1…F-13, roadmap P0–P3 (sim2real, Dex3 concept)
- `planning/Phase2_BallCommand_plan.md`: early Phase 2 plan (partly outdated, e.g. multi-throw logic was removed)
- `planning/2026-10-27-fixes-unified-phase2-plan.md`: implementation plan (German) for fixes 1–4 of the Phase 2 analysis in Phase 2 and Unified (waiting and holding, joint-target clamp, latency, clip 51), with the test procedure and the baseline values
- `changes/p0_changes.md`, `changes/p1_changes.md`, `changes/merge_changes.md`: changelogs of the 10 Aug implementation
- `changes/2026-10-16-underhand-catch-p0.md`: P0 of the 2026-10-16 analysis (mjlab ball, underhand lob, air drag,
  arm-contact catch detection), with the report basis, probe results and open placeholders. New change documents
  are named `<date>-<title>-<part>.md`.
- `changes/2026-10-16-deployable-observations-p1.md`: P1 items 5 and 7 (deployable actor observations, VICON
  perception model) and the calibration on the VICON ball recordings (Section 5.1).
- `changes/LATENT_mimic_adjustments.md`: LATENT ideas for the mimic stage (not all adopted yet)
- `guides/debug_instructions.md`: older debug guide (partly outdated, e.g. debug prints were removed)
- `guides/replay_guide.md`: step-by-step guide (German) to convert, check (`check_motion.py`) and replay a mocap clip
- `analysis/2026-10-27-phase2-final.md`: evaluation of the Phase 2 baseline (stress tests, joint-stop exploit, hold and wait behavior) with a ranked list of fixes for Phase 2, Unified and the training setup
- `REVIEW.md`: code review of the ball-catching work (findings C1–L5; C1, H1, H2, M1–M4 and L5 resolved)

Outside `context/`: `data/imitation/README.md` (where the reference clips come from and how to install them on a machine).

In case of conflicts: **code > this spec > other context documents.**

## 13. Glossary

- **Mimic / Phase 1:** Imitation of a reference motion via tracking rewards (BeyondMimic recipe).
- **Anchor:** Reference body (`torso_link`) relative to which body positions are compared.
- **Yaw frame:** Robot root frame with only the yaw rotation (no roll or pitch).
- **Secured:** Ball in the catch box in front of the torso, touching both arms, slow relative to the root, above the
  ground. `secure_steps` consecutive steps (50 = 1 s; 25 early in the curriculum) mean "caught".
- **Underhand catch:** The ball, lobbed from below by the thrower, is caught with both forearms/hands in front of
  the belly (basket), as in the mjlab and LocoMuJoCo tasks of the interim report.
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

### 2026-10-16: Underhand lob, mjlab ball, air drag and arm-contact catch detection (P0)
- Why: the analysis of the interim report showed that ball, throw and catch definition matched neither the report
  nor the real setup (ball ∅10 cm vs. ∅15 cm in the mjlab chapter, throws from 1–2.5 m with 0.3–0.6 s flight vs.
  underhand lobs over ~4 m). Static Isaac probes: a ∅10 cm ball falls through a forearm cradle (0/45 held), a
  ∅15 cm ball stays (34/45). The 0.5 s "secured" criterion counted balls that rolled off afterwards. The reference
  motion will switch to underhand two-arm catches.
- Ball (Phase 2, Unified): ∅15 cm, 112 g (mjlab), restitution 0.4 (placeholder), DR restitution 0.2–0.6 and
  mass ±15 g. New `ball_contact` sensor (ball filtered against the 8 forearm/hand links).
- `BallCommand`: throw frozen at reset (release 2–5 m, ±75°, 1.1–1.7 m above ground; aim 0.3–0.4 m in front,
  ±0.1 m, 0.8–1.05 m above ground; flight 0.8–1.4 s; noise 0.1 m/s). Quadratic air drag applied as external
  force, launch velocity drag-compensated (PhysX flight passes the aim point within 2 cm on average vs. 35 cm
  without compensation). Catch = catch box + both arms touching + slow relative to the root + z > 0.5 m for
  `secure_steps` (50). New `launched` property and `arm_contact` state. `SyncedBallCommand` adapted
  (`_start_cycle`, same geometry).
- Rewards: `hand_to_ball` replaced by `ball_reach` (distance-only) and `ball_interception` (eq. 9.1) on the hands
  midpoint, plus `ball_arm_contact`; all ball shaping is zero before the launch (fixes review finding M1).
  `undesired_contacts` additionally excludes elbow/wrist links and the torso.
- Terminations: `ball_missed` at 6 m (was 4 m, would fire at a 5 m release point).
- Observation contract: `ball_intercept` catch plane 0.15 m above the root (was 0.35 m) in Phase 2 and Unified,
  policy and critic; dimensions unchanged (policy 875, critic 1535 in all three tasks).
- Curriculum: new throw fields (easy 2.0–2.5 m / ±15° / 1.0–1.2 s / `secure_steps` 25 → final defaults);
  `_lerp` rounds int fields.
- Verified: drag solver vs. RK4 (mean 0.3 cm error), PhysX throw accuracy, contact column mapping (lateral throws),
  catch detection in a static cradle (23/64 confirmed, all still held 1 s later), 5-iteration smoke runs of
  Phase 1, Phase 2 and Unified.
- Files: `BC/mdp/commands/ball_command.py`, `BC/mdp/commands/synced_ball_command.py`, `BC/mdp/rewards.py`,
  `BC/mdp/observations.py`, `BC/mdp/curriculums.py`, `BC/robots/g1_29dof/phase2/ball_catch_env_cfg.py`,
  `BC/robots/g1_29dof/unified/ball_catch_env_cfg.py`, `context/REVIEW.md` (M1 marked fixed).
- Spec sections 2, 5, 6, 6.1, 7.2, 7.3, 8, 9, 10, 11, 12 and 13 updated.

### 2026-10-16: Change document for the underhand-catch P0
- Added `context/changes/2026-10-16-underhand-catch-p0.md`: detailed description of the P0 changes above (report
  basis per parameter, problem/change per item, probe and validation results, placeholders). Naming scheme for new
  change documents: `<date>-<title>-<part>.md`.
- Spec section 12 updated.

### 2026-10-16: Deployable actor observations, VICON perception model, throws calibrated on recorded throws (P1)
- Why: the actor got the motion reference (Unified; a human thrower cannot sync to the clip clock), hand positions
  (forward kinematics in C++) and a perfect, always visible ball (a static ball before the throw, no latency, noise or
  dropouts). The interim report (ch. 10.3) already claimed noise on the policy inputs that the code did not have.
- VICON ball recordings found: the training `.pkl` have no ball, but `GMR_for_fbx_files/recordings/two_results_ver2.7z`
  holds 21 takes of 2026-07-06 with ball CSV + `.pkl` (12 complete underhand throws). Analysed: release 3.1 m away at
  1.03 m, apex 1.9 m, flight 0.77 s, drag k/m 0.035 1/m (fits a light ∅15 cm ball), noise 4–5 mm, 6.8 % of the control
  steps without a measurement in bursts of 1–3 steps, wind-up visible before the release (new spec section 5.1).
- Actor contract (Phase 2 = Unified, 103 per frame / 515): perceived ball pos+vel, `ball_visible`, intercept from the
  perceived state, proprioception. Removed from the actor: `motion_command`, `motion_anchor_ori_b`, `hand_pos`,
  `ball_relative` (still in the critic). Critic gets `ball_visible` (308 per frame / 1540; Phase 1 has a zero slot).
  Phase 1's actor keeps the reference and drops the ball slots (157 / 785): it is now a clip trackability check, not
  a warm start. Stage A → B = Unified → Phase 2 with `--finetune` (tested); Phase 1 → Phase 2 now fails on the shape.
- Perception model in `BallCommand` (`_reset_perception`, `_update_perception`): detection delay 0.04–0.12 s after the
  release, latency 1–3 steps, noise 5 mm / 0.1 m/s, per-throw bias ±1 cm, dropout bursts (p 0.05, 1–3 steps),
  occlusion in the catch box with p 0.5 per throw, ballistic coasting for 3 steps, then not visible. New observation
  functions `ball_perceived_pos_vel_b`, `ball_visible`, `ball_perceived_intercept_b`; `ball_intercept_b` refactored onto
  the shared `_ballistic_intercept` (same math for actor and critic).
- Throws re-calibrated on the recordings: release height 0.85–1.2 m (was 1.1–1.7), `flight_time_range` replaced by
  `apex_height_range` 1.75–2.1 m (flight time 0.71–1.02 s via `flight_time_from_apex`, farther throws faster), drag
  coefficient 0.3–0.45 (was 0.4–0.6). Curriculum: apex instead of flight time; easy stage with clean perception
  (delay 1, no dropouts, no occlusion).
- Verified: zero-noise probe reproduces the true state of 2 steps earlier exactly (pos, vel, intercept); default
  probe 88.8 % fresh / 8.6 % coasted / 2.6 % not visible in flight, never visible before the launch; sampled throws
  0.85–1.20 m release, 0.74–0.98 s flight, apex mean 1.90 m; smoke runs of all three tasks; Unified → Phase 2
  fine-tuning; exported `deploy.yaml` lists only robot-side terms.
- Files: `BC/mdp/commands/ball_command.py`, `BC/mdp/commands/synced_ball_command.py`, `BC/mdp/observations.py`,
  `BC/mdp/curriculums.py`, `BC/robots/g1_29dof/{phase1,phase2,unified}/ball_catch_env_cfg.py`, `CLAUDE.md` (key
  invariant), `context/REVIEW.md` (M4 resolved), `context/changes/2026-10-16-deployable-observations-p1.md`.
- Spec sections 2, 4, 5 (+ new 5.1), 6, 6.1, 7.1, 7.2, 7.3, 8, 9, 11 and 12 updated.

### 2026-10-22: Phase 2 fall penalty and larger alive bonus (pilot runs)
- Why: a 4096-env pilot of Phase 2 at `24d448d` converged to falling before the throw. 100 % of the episodes ended
  by `bad_orientation`, the episode length dropped to ≈9 steps while the mean reward rose, and all ball rewards stayed 0.
  Phase 2 has no positive dense reward apart from `alive`; early per-step penalties (mostly `action_rate` from the
  exploration noise) were ≈20× `alive` (0.15), and falling cost nothing.
- Change: `alive` 0.15 → 1.0 (discounted at most ≈2, below the catch bonus 10) and `termination_penalty`
  (`is_terminated_term` on `base_height` / `bad_orientation` only, weight −200 = −4 per fall).
- Evidence (2048 envs, 250 iterations): with both terms, episode length 104 steps, 15 % falls, 86 % of the episodes
  reach the throw; the fall penalty alone gives 86 steps and 76 % falls. A Unified pilot with unchanged config is not
  affected (episode length → 136 steps, reward → +3.3), so Unified and Phase 1 stay unchanged.
- Files: `BC/robots/g1_29dof/phase2/ball_catch_env_cfg.py`, `context/changes/2026-10-22-phase2-fall-penalty-p1.md`.
- Spec sections 2, 7.2 and 10 updated.

### 2026-10-22: Catch as a truncation (hold procrastination) and curriculum-aware resume
- Why: the first full Phase 2 run (workstation, 4096 envs, `26b64a3`) reached 77 % confirmed catches at iteration 2500 and then learned to hold the ball without completing the catch: by iteration 6900 time-outs rose 0 → 32 %, confirmed catches fell to 54 %, the episode length grew 192 → 608 steps and the ball was secured ≈ 460 steps per episode (a catch needs 39 in a row at that stage). Holding paid ≈ 0.2 per step (hold, contact, proximity, `alive`), ≈ 20 discounted at γ 0.99; the catch ended that stream for a bonus of 10 (≈ 14 in total), so breaking the secured streak just before `secure_steps` was optimal.
- Change: `ball_caught` is a truncation (`time_out=True`) in Phase 2 and Unified, so PPO bootstraps the value of the catch state and the bonus makes completing the catch strictly better. In Unified this also stops the adaptive sampler from counting confirmed catches as failures (it counts `terminated` only). Logging is unchanged.
- `train.py`: a full `--resume` sets `common_step_counter` to iteration × `num_steps_per_env`; before, a resumed run restarted the step-based curricula (throws, posture tapers) at the easy settings.
- Verified on the workstation (64 envs, resume from `model_7000.pt`): `ball_caught` is listed as a time-out term, the iterations continue at 7000, and from the third iteration on the curriculum is at 0.59 (throws) and −8.30 (`base_height`).
- Files: `BC/robots/g1_29dof/{phase2,unified}/ball_catch_env_cfg.py`, `scripts/rsl_rl/train.py`.
- Spec sections 2, 4, 7.2, 7.3, 9, 10 and 11 updated.

### 2026-10-27: Phase 2 baseline finished and evaluated
- The Phase 2 run with the truncation fix ran to 30k iterations (workstation 1 → workstation 2, final `model_29999.pt`): 99.6 % catches on the final throw distribution (deterministic).
- Stress tests (long waits, pushes, latency, wider aim, friction, holding without termination) and full-body videos show a joint-stop exploit at the ankle-roll joints, failures after long waits (28 % catches, 28 % falls), sensitivity to latency and pushes, and a squeeze-to-the-chest hold; pushes practically never fired in training.
- No code change. New context document `context/analysis/2026-10-27-phase2-final.md` (ranked fixes); workstation setup documented in Section 4; spec sections 2, 4, 10, 11 and 12 updated.

### 2026-10-27: Underhand-catch dataset as reference motion (all clips, recorded catch timing)
- Why: the single overhand chest clip did not match the underhand catch the ball tasks target, and the group now provides prepared underhand clips with VICON ball recordings (GitLab `imitation-data`, also used by Maik's `g1mimic` task).
- Data: 108 clips (50 Hz) and 89 recordings (120 Hz) of the same takes; the recordings are aligned with their clips by joint angles (time) and root pose (frame); 86 are usable (Section 5.2). Local copy in `data/imitation/` (gitignored, README committed; the repository is public, the data is private group data).
- `MotionLoader` / `MotionCommand`: all clips as padded tensors; each env tracks one clip, drawn uniformly at every reset or clip end, with adaptive start-bin sampling per clip; `resample_count` marks new cycles; the frame rate must equal the control rate.
- `SyncedBallCommand`: recorded mode with per-clip catch time, recorded flight time and the recorded catch point re-anchored to the robot at the launch (resolves review findings M2 and M3; `CATCH_MOTION_TIME` removed, fixed mode kept as a fallback); new cycles detected via `resample_count` instead of a backward jump of the motion time; `fps` is a float (L5).
- New `tasks/ball_catching/motion_data.py` (discovery, LFS check, alignment) and `scripts/mimic/catch_info.py`; Phase 1 uses all clips, Unified the clips with recordings.
- Verified on workstation 1: `catch_info.py` (86 of 108 usable), Phase 1 and Unified smoke runs (64 envs, 3 iterations), and a kinematic replay of the references in Unified (ball within 0.23 m of the reference wrists around the arrival; ≈ 29 % of the cycles throw).
- Files: `BC/motion_data.py`, `BC/mdp/commands/{motion_command,synced_ball_command}.py`, `BC/robots/g1_29dof/{phase1,unified}/ball_catch_env_cfg.py`, `BC/robots/g1_29dof/phase1/motions/README.md`, `scripts/mimic/{catch_info,pkl_to_npz}.py`, `data/imitation/README.md`, `context/REVIEW.md`.
- Spec sections 2, 3, 5 (+ new 5.2), 7.1, 7.3, 8, 9, 11 and 12 updated.

### 2026-10-27: Joint-assignment checks for the dataset clips
- Why: make sure Phase 1 / Unified get the right joints (no repeat of C1, hand data in an arm) for clips that have no GMR `.pkl` to compare with (`check_motion.py`).
- New `scripts/mimic/check_npz_fk.py` (Isaac: forward kinematics of the clips on the training asset vs their stored body positions) and `scripts/mimic/check_dataset.py` (numpy: robot elbows/hands vs the human points of the VICON recordings). `motion_data.align_recording` factored out of `compute_catch_info` (shared by both).
- Result: joint and body order match the asset (< 0.1 mm; negative control 0.27–0.39 m), elbows and hands follow the human on the right side in 87 of 89 clips (51 and 52 deviate through their time alignment), no static arms; knees below the soft limit in 93–98 % of the frames.
- Spec sections 3, 5.2, 9 and 11 updated.

### 2026-10-27: Evaluation suite in the repository
- Why: the training catch rate (≈ 95 %) hid the weaknesses that the stress tests of the Phase 2 baseline exposed (long waits, joint-stop exploit, latency, pushes); every run needs the same comparable measurement.
- New `scripts/rsl_rl/eval_catch.py` (one checkpoint and scenario: base, fric_high, fric_low, hold, wait, push, lateral, delay; deterministic policy in the play config; JSON with outcomes, foot behavior incl. gliding vs. pivoting and foot tilt, posture, hold geometry incl. torso contact, joints at their limits, torque saturation; optional full-body video), `scripts/rsl_rl/eval_catch_all.sh` (all scenarios of a task) and `scripts/rsl_rl/eval_catch_report.py` (comparison table). Works for Phase 2 and Unified; `wait` and `lateral` are refused for Unified.
- Spec sections 3, 4, 10 and 11 updated; `CLAUDE.md` lists the command.

### 2026-10-28: Perception latency up to 5 steps (fix 3; implemented, tests pending)
- Why: 4–6 steps of perception latency cost the 2026-10-27 baseline (trained on 1–3) 6 points of catch rate (93.6 % vs. 99.6 %), and the latency of the live chain VICON → MQTT → robot is not measured yet.
- `BallCommandCfg.obs_delay_steps_range` default (1, 3) → (1, 5) (it sizes the perception ring buffer), and the final curriculum stage of Phase 2 and Unified likewise; the easy stage keeps 1 step. The suite's `delay` scenario (4–6 steps) stays unchanged for comparison with the baseline.
- First of the fixes 1–4 on branch `fixes-2026-10-28` (plan: `context/planning/2026-10-27-fixes-unified-phase2-plan.md`, added to the repository).
- Files: `BC/mdp/commands/ball_command.py`, `BC/robots/g1_29dof/{phase2,unified}/ball_catch_env_cfg.py`, `scripts/rsl_rl/eval_catch.py` (docstring).
- Spec sections 7.2, 9, 11 and 12 updated.

### 2026-10-28: Ball recordings checked against the human of the take (fix 4)
- Why: the joint-angle alignment accepted clip 51 with an unusual time offset (+0.76 s) that puts the clip's hands 0.12 m away from the human's; its catch time and catch point were then used for throws.
- `motion_data.compute_catch_info` now also compares the clip's elbows and hands with the human elbows and palms of the recording (`compare_with_human`, moved from `check_dataset.py`) and rejects the recording if the worst point deviates more than `MAX_ARM_RMSE` = 0.10 m (new reason "arms … off the human of the recording"). The G1 asset constants (`G1_JOINT_NAMES`, `G1_HAND_BODY_INDEXES`, the four compared points) moved from `catch_info.py` / `check_dataset.py` into `motion_data.py`; `catch_info.py` prints the arm RMSE per clip.
- Verified on workstation 2 (numpy): 84 of 108 clips usable (86 before). Besides 51, the check rejects clip 82, whose time offset is normal (−0.08 s) but whose right hand drifts 0.11–0.15 m off the human's palm in the second half; the plan expected only 51 (85 clips) and assumed all other clips within 0.07 m. All accepted clips stay within 0.07 m. `check_dataset.py` uses the same limit and now reports 86 of 89 (51, 52, 82).
- Files: `BC/motion_data.py`, `BC/mdp/commands/synced_ball_command.py` (reason summary), `scripts/mimic/{catch_info,check_dataset}.py`.
- Spec sections 2, 5.2, 7.3, 8 and 11 updated.

### 2026-10-28: Joint targets clamped to the joint limits (fix 2; implemented, tests pending)
- Why: the actions were unbounded in all three tasks. The Phase 2 baseline drove the ankle-roll targets 2–4 rad past the joint range in 98–99 % of the steps, stood on the joint stops with saturated torque during the whole wait and with its feet tilted by 14–21°; Phase 2 had no torque penalty (`energy` ignores static torque).
- New `BC/mdp/actions.py`: `ClampedJointPositionAction(Cfg)`, a `JointPositionAction` whose targets are clamped to the asset's hard joint limits (the reachable range stays the same; the soft limits would have forbidden straight knees) through Isaac Lab's `clip`. `export_deploy_cfg` writes the limits into `deploy.yaml`, and the C++ `JointAction` already applies `clip`, so the robot clamps identically (smoke run: 29 ranges, e.g. ankle roll ±0.262 rad, knee −0.087…2.88 rad). `target_excess` (sum of |unclamped − clamped| per env) feeds the new reward `joint_target_excess`.
- All three tasks use the clamped term (term name `JointPositionAction` unchanged) with `joint_target_excess` −1. Phase 2 additionally gets `joint_torque` −1e-5 (as Phase 1 / Unified) and the new `feet_flat` −2 (sin² of the tilt of each foot in firm contact > 50 N).
- Old checkpoints still load (same network), but behave differently on the clamped targets; train new runs from scratch.
- Verified so far: smoke runs of all three tasks on workstation 2 (64 envs, 3 iterations), terms active, `deploy.yaml` with the limits; Unified reports recorded catches for 84 of 89 clips (fix 4).
- Files: `BC/mdp/{actions,rewards,__init__}.py`, `BC/robots/g1_29dof/{phase1,phase2,unified}/ball_catch_env_cfg.py`.
- Spec sections 3, 6, 7.1, 7.2, 7.3, 8, 9 (new pitfall 19) and 11 updated.

### 2026-10-28: Waiting and holding (fix 1; implemented, training pending)
- Why: the Phase 2 baseline trained on 0.5–1.5 s waits and a 1 s confirmation that ended the episode. After 6–9 s of waiting it caught 28 % and fell 28 %; in 12 s episodes it walked off with the ball by ≈ 4 m. In Unified only 29 % of the motion cycles threw, and 20 clips end less than 1 s after the catch (no time to confirm and hold).
- `BallCommand` (shared): the confirmation (`secured_steps` reaches `secure_steps`) happens once per throw (`confirmed`, `confirmed_now`), then a hold phase of `hold_after_catch_s` follows (`catch_complete`; `ball_caught` fires only then and stays a truncation); `ball_caught_bonus` pays once per throw at the confirmation; new metric `hold_time_after_confirm`; optional log-uniform wait (`throw_delay_log_uniform`).
- Phase 2: wait 0.5–8 s log-uniform (curriculum from 0.5–1.5 s), 3 s hold phase, `stand_still_root` (−10) and `stand_still_feet` (−5) before the launch and during the hold phase. The longer episodes let `push_robot` fire.
- Unified: wait-and-hold cycle in `MotionLoader` / `MotionCommand` (6 s lead-in with 35 % of the starts inside it; each clip cut 0.3 s after its recorded catch and that pose held for 5 s); catch data computed when the config is imported (`motion_data.catch_data`, replaces `SyncedBallCommandCfg.ball_data_dir` by `catch_data`) and shifted by the lead-in; check of the G1 asset constants; 3 s hold phase. Deviation from the plan: end hold 5 s instead of 3.5 s, otherwise most clips would end before the hold phase completes (confirmation up to ≈ 1.5 s after the arrival + 3 s).
- Suite: `wait` works for Unified (every cycle starts at the beginning of the lead-in); `hold` keeps the 0.5–1.5 s wait for Phase 2 (≈ 9 s of holding, as for the baseline); new outputs `confirmed_pct`, `after_confirmation_pct` (how confirmed catches ended, `ball_caught` = hold phase completed), `hold_after_confirm_s_median`, `pct_time_target_beyond_hard_limit` per joint and its maximum (`pct_time_target_outside_range` is measured against the soft limits, so a target clamped to the hard limit still counts there); report column `full`.
- Verified on workstation 2: smoke runs of all three tasks; Unified kinematic replay as in Section 7.3 (no cycle ended during a hold phase); old Phase 2 checkpoint on the new code: with the joint-target clamp of fix 2 it falls in 90 % of the episodes (it stood on the torque at the ankle-roll stops; a new training is required), without the clamp (diagnostic only) it catches 87 % with the 0.5–8 s waits (median 1.9 s), 99.9 % of its confirmed catches complete the hold phase (exactly 3.0 s), and in `hold` 98.5 % still hold the ball after 12 s (baseline 99.4 %). The new `pct_time_target_beyond_hard_limit` is 0.0 % for every joint with the clamp and 98.9 % (left ankle roll) without it.
- Watch in the first Phase 2 training: early on `stand_still_root` dominates the rewards (smoke run, iteration 2: −1.24 vs. `alive` +0.06), because flailing and falling robots move fast; check that standing is learned about as fast as in the baseline (episode length, `bad_orientation`).
- Files: `BC/mdp/commands/{ball_command,motion_command,synced_ball_command}.py`, `BC/mdp/{rewards,terminations}.py`, `BC/motion_data.py`, `BC/robots/g1_29dof/{phase2,unified}/ball_catch_env_cfg.py`, `scripts/rsl_rl/{eval_catch.py,eval_catch_all.sh,eval_catch_report.py}`, `context/planning/2026-10-27-fixes-unified-phase2-plan.md` (implementation status and deviations).
- Spec sections 2, 7.2, 7.3, 8, 9 (pitfalls 20, 21) and 11 updated.

### 2026-10-28: Stand-still terms ramped in (pilot run)
- Why: a 250-iteration Phase 2 pilot with fixes 1–4 (2048 envs, workstation 2) learned to fall within 7 steps: an untrained, flailing robot moves fast, so the quadratic `stand_still_root` cost about −17 per episode while flailing for 49 steps and a quick fall only about −6. The same pilot with both stand-still weights at 0 learned to stand like the 2026-10-22 pilot (episode length 95.6 steps, 32 % falls, 69 % of the episodes reach the throw at iterations 230–250), so the clamp, `feet_flat` and the torque penalty do not hinder the start.
- Change: `stand_still_root` (0 → −10) and `stand_still_feet` (0 → −5) are ramped in over steps 50k → 250k with `modify_reward_weight_linear`, together with the waits of the throw curriculum. In the first full run, watch that the episode length and `bad_orientation` do not collapse while the weights rise (iterations ≈ 2000–10000).
- Files: `BC/robots/g1_29dof/phase2/ball_catch_env_cfg.py`, `context/planning/2026-10-27-fixes-unified-phase2-plan.md`.
- Spec sections 7.2, 9 (pitfall 22) and 10 updated.

### 2026-10-28: Unified pilot with fixes 1–4
- 250 iterations, 2048 envs on workstation 2: no collapse as in the Phase 2 pilot with the stand-still terms; the episode length dips to 7 steps around iterations 40–110 and rises to 57 steps by iteration 250, episodes end by tracking terminations, the first balls arrive (0.7 % drops). Phase 1 on the same clips (old code, 4096 envs, actor with the reference) was at 110 steps after 250 iterations.
- Spec section 10 updated.

### 2026-10-28: Dropped balls counted where their cycle started (Unified, option B)
- Why: the adaptive sampler counted `ball_dropped` / `ball_missed` in the bin where the episode ended, after the throw. Cycles that start there get no ball (`SyncedBallCommand` throws only if the cycle starts before the launch), so every drop steered the starts away from the throws; simulated with the sampler's parameters, 57 % instead of 69 % of the cycles throw at a 50 % drop rate. The incentive to keep the ball up does not depend on the sampler (`ball_dropped` is a real termination, `ball_height_penalty`, lost rewards); the sampler decides how often the policy sees a throw.
- Change: new `MotionCommandCfg.start_bin_failure_terms`: failures of these terms are counted in the bin where the failed cycle started (`MotionCommand.cycle_start_steps`); disabled terms are skipped, unknown names raise. Unified sets `["ball_dropped", "ball_missed"]`; tracking failures are counted where they happen, as before. Considered and not chosen: counting tracking terminations only, backward smoothing (`adaptive_kernel_size`), more lead-in starts, no change. Option B gives clips where the ball falls more throws, and a hard throw cannot lock in: every cycle samples a new throw and every clip keeps 1/89 of the cycles.
- Verified on workstation 2: smoke runs of Unified and Phase 1; kinematic replay of Unified in which every throw is dropped (2048 envs, 2500 steps): with option B 98 % of the failure mass lies before the clips' launch bins and 75 % of the cycles throw (0.156 throws per robot and second), without it 100 % lies after them and 54 % throw (0.113).
- Files: `BC/mdp/commands/motion_command.py`, `BC/robots/g1_29dof/unified/ball_catch_env_cfg.py`, `context/planning/2026-10-27-fixes-unified-phase2-plan.md`.
- Spec sections 7.3, 8 and 9 (pitfall 23) updated.

### 2026-10-28: Phase 1 trained on the 108 dataset clips
- Run `unitree_g1_29dof_ballcatch_phase1/2026-10-28_14-28-00` on workstation 2: 10k iterations, 4096 envs, 4.9 h, code `71c1767` (before fixes 1–4). End of training: 99.4 % of the episodes without a tracking termination, body error 6.2 cm.
- Per-clip check of `model_9999.pt` (each clip once from its first frame, deterministic, no pushes or noise): all 108 clips are tracked without a tracking termination, body error 5.0 cm (worst 5.7 cm), no foot sliding, knees at 9.7° vs. 1.3° in the reference at a negligible `joint_limit` penalty; deep squats are tracked less deep.
- Found: the clips sit about 4.5 cm too low (reference ankles below the ground plane); the robot stands 4.5 cm higher than its reference, is pushed out of the ground after every start from the reference pose, and in Unified the recorded catch point is 4.5 cm too low relative to the robot. Not corrected yet.
- No code change. Side-by-side video (policy next to the reference), contact sheets, per-clip table and the scripts in `analysis/2026-10-28_phase1/` outside the repository.
- Spec sections 2, 5.2, 10 and 11 updated.

### 2026-10-28: Temporary height correction of the dataset clips
- Why: the Phase 1 evaluation showed that the clips sit about 4.5 cm too low (feet in the ground): the robot stands 4.5 cm above its reference, every start from the reference pose puts it into the ground, and Unified's catch points (same frame) are 4.5 cm too low. Better data and retargeting are expected, so the correction is a clearly marked temporary one.
- `motion_data.TEMP_CLIP_LIFT` = 0.045 m (block comment: temporary, current dataset `e1704e4` only, how to remove); new generic `MotionCommandCfg.clip_lift` / `MotionLoader(lift=...)` (added to all body heights) and `catch_data(lift=...)` (catch point height). Phase 1 and Unified set both from `TEMP_CLIP_LIFT`; `catch_info.py` notes that its heights are in the unlifted clip frame. Every use contains the name `TEMP_CLIP_LIFT`.
- Cause analysis (numpy over the dataset, Section 5.2): VICON floor and the recordings' own retarget are right (soles 1.1 cm in the ground); the exported clips' pelvis is 3.8 cm lower than that retarget of the same takes, and their lowest link origin sits where the recordings' soles are, so the export presumably grounded the link origin instead of the sole. To be raised with the data group.
- Verified on workstation 2: smoke runs of Phase 1 and Unified; Phase 1 checkpoint on all 108 clips with the lift: torso 0.1 cm (was 4.6 cm) above the reference, body error 1.6 cm (was 5.0 cm), no tracking termination; Unified kinematic replay: ball 0.25 m from the reference wrists at the arrival (0.26 m without the lift).
- Spec: the fix branch is in `main` (Section 11).
- Files: `BC/motion_data.py`, `BC/mdp/commands/motion_command.py`, `BC/robots/g1_29dof/{phase1,unified}/ball_catch_env_cfg.py`, `scripts/mimic/catch_info.py`.
- Spec sections 5.2, 8, 9 (pitfall 24) and 11 updated.

### 2026-10-29: Unified throws aimed at the reference hands, feet_flat in Unified
- Why: the first Unified run (`2026-10-28_21-24-50`, stopped at iteration 8000) learned to clamp the ball between its hands and head (ball 0.32 m above the pelvis, torso contact 196 N) and stood with its left foot on the edge (17° in 84 % of the contact steps at iteration 7000). The throws aimed at the recorded ball position, which is in the human's scale while the clip is in the robot's: at the catch the clip's wrists were 0.21 m from it (0.13 m below), the targets reached up to 0.57 m above the pelvis, above the catch box. Tracking the feet's orientation alone did not keep the feet flat.
- Change: `motion_data.compute_catch_info` also returns the clip's wrist midpoint at the catch (`hands_point`); `catch_data(aim_above_hands=...)` puts the catch point there, raised by that much. Unified aims 0.08 m above the reference wrists (`CATCH_ABOVE_HANDS`): targets 0.19 m in front of and 0.24 m above the pelvis (median), all inside the catch box. Catch time and flight time still come from the recording. Unified gets `feet_flat` −2 as Phase 2. The pushes stay as they are (they harden the tracking policy early; part of the staggering in the video is the recovery from them).
- Verified on workstation 2: Unified smoke run (`feet_flat` logged); kinematic replay under the same conditions: closest approach of the ball to the reference wrist midpoint 0.14 m (median; 0.23 m with the old target), 0.18 m at the arrival time (0.25 m).
- First run documented in Section 10 (results and video outside the repo in `analysis/2026-10-29_unified_7000/`); second run started from scratch.
- Files: `BC/motion_data.py`, `BC/robots/g1_29dof/unified/ball_catch_env_cfg.py`.
- Spec sections 2, 7.3, 8, 10 and 11 updated.

### 2026-10-29: Every Unified cycle gets its ball; a ball squeezed against the torso is not held
- Why (A): in the second Unified run (`2026-10-29_03-31-07`, ended at iteration 6106 by a crash of workstation 2) 58 % of the tracking terminations happened in cycles without a ball. 45 % of the cycles that reached their clip end had started after the launch or played a clip without a usable catch; the reference then performs its catching motion without a ball, which the actor (only the ball, no reference) cannot predict, and the adaptive sampler starts more cycles exactly where these failures happen. Both runs fell from ≈ 25–30 % to ≈ 12 % completed catches after iteration ≈ 5000.
- Change (A): new `MotionCommandCfg.latest_start_s` (per clip, seconds into the clip, `None` = anywhere): a cycle drawn to start later is drawn again, uniformly before that point. Unified sets it to the launch − 0.1 s (`LAUNCH_MARGIN_S`) and trains only on the 84 clips with a usable catch (all clips in fixed mode).
- Why (B): the Phase 2 baseline and both Unified runs press the ball against the chest (196–236 N) instead of holding it with the arms; the catch check only asked for contact with both arms, and the chest makes that contact the most reliable. The reference holds the ball in front of the belly (wrists 0.19 m in front of and 0.15 m above the pelvis).
- Change (B): `BallCommandCfg.torso_body_name` / `max_torso_force`: the `ball_contact` sensor also filters `torso_link` (appended last, the arm columns keep their order), and a ball pressed against it with more than 50 N is not secured, so it is neither confirmed nor rewarded by `ball_secured_hold` or the bonus; resting against the body is allowed. New metric `ball_torso_force`. Phase 2 and Unified (`TORSO_BODY_NAME`, `MAX_TORSO_FORCE`). A condition on the catch, no new reward term.
- `eval_catch.py`: `--no_push` switches the pushes off; with `--video` the command markers are hidden.
- Verified on workstation 2: smoke runs of Phase 2 and Unified (`ball_torso_force` logged); Unified kinematic replay (Section 7.3): every cycle that reaches its clip end throws (55 % before), 0.11 throws per robot and second (0.09); `model_6000.pt` of the second run with A and B (512 robots × 60 s): 15 % of the throws are confirmed (77 % before; most of its catches were chest clamps), the few confirmations come late (20 % of the hold phases cut by the clip end, 2 % before; a policy trained with B should hold the ball properly from the start), 4.6 instead of 8.3 tracking terminations per robot and minute, 3575 instead of 3444 throws.
- Second run documented in Section 10 (results and videos outside the repo in `analysis/2026-10-29_unified2_6000/`); third run from scratch.
- Files: `BC/mdp/commands/{motion_command,ball_command}.py`, `BC/robots/g1_29dof/{phase2,unified}/ball_catch_env_cfg.py`, `scripts/rsl_rl/eval_catch.py`.
- Spec sections 2, 7.2, 7.3, 8, 9 (pitfalls 23 and 25), 10 and 11 updated.

### 2026-10-29: Sim2Sim environment and model comparison
- Why: prepare Sim2Sim in MuJoCo while the third Unified run trains. The workstations already have MuJoCo (shared conda env `gmr`, the mjlab group's packages in the shared uv cache), so nothing large is installed again, and both workstations have to give the same results.
- `sim2sim/`: `requirements.in` and the locked `requirements.txt` (Python 3.10 = system Python; MuJoCo 3.10.0 as in `gmr`, onnxruntime 1.22.0 as in the C++ deploy, numpy, PyYAML, imageio-ffmpeg), `setup.sh` (venv `~/envs/sim2sim` from the shared uv cache, G1 MJCF and meshes of `unitree_rl_mjlab` at commit `1425b15` in `~/unitree_mjcf/g1`, environment check), `tools/check_env.py`, `tools/dump_isaac_model.py`, `tools/compare_models.py`, `tools/shapes.py`, README. `gmr` itself is not used: it differs between the workstations already and belongs to the retargeting work. `unitree_rl_mjlab` as a whole is not used either: it pins mjlab 1.2.0 (the cache has 1.3–1.6), would be a second training stack, and has nothing of the ball task.
- Set up on both workstations: the 1-s passive simulation gives the same state hash on W1 (Intel) and W2 (AMD); headless rendering (EGL) works on W2 (not tried on W1, whose GPU was in use by another user). On W2 uv copies the packages that other users put into the shared cache instead of hardlinking them (Linux forbids hardlinks to other users' files), ≈ 240 MB.
- Model comparison (Isaac `UNITREE_G1_29DOF_MIMIC_CFG` vs `g1.xml` and `scene_g1.xml` of `unitree_rl_mjlab`): masses (33.341 kg in total), centres of mass, inertia, forward kinematics at the zero and the default pose, joint ranges and effort limits identical. Armature only in Isaac (MJCFs: none or 0.01), passive damping 0.05 and friction loss 0.2 Nm only in `scene_g1.xml`. Collision shapes: `scene_g1.xml` uses the same meshes as the USD (identical bounding boxes, also the rubber hand mesh) but adds colliders where the USD has none and keeps the rubber hand visual only; `g1.xml` replaces everything with capsules. The USD has no collider on the forearm (`elbow_link`, `wrist_pitch_link`), hip roll, shoulder roll and ankle pitch (pitfall 26).
- Spec sections 2, 3, 6, 9 (pitfall 26) and 11 (items 7 and 9) updated.

### 2026-10-29: Optional forearm colliders; self-collision and ball-contact checks
- Why: the Isaac G1 has no forearm colliders (pitfall 26), the real robot does, so the policies have never felt a ball on the forearm. Whether the third Unified run restarts with them is decided at iteration ≈ 6000; until then the option is ready, tested and off.
- Change: `tasks/ball_catching/robot_assets.py` provides `ROBOT_CFG` for Phase 1, Phase 2 and Unified (their env cfgs import it instead of `UNITREE_G1_29DOF_MIMIC_CFG`). With `FOREARM_COLLIDERS = True` the spawn function `spawn_usd_with_forearm_colliders` references each forearm link's visual mesh (from the USD's base layer) below `<link>/collisions` and gives it the collision schemas and attributes of the existing wrist roll collider (convex hull), before the env prims are cloned.
- Checks on workstation 2 (test copy):
  - Isaac dump with the option: elbow and wrist pitch link get a convex hull with exactly the bounding box of Unitree's collider mesh (`scene_g1.xml`); masses and inertia unchanged. The Unified env has the colliders in every env (PhysX: 37 instead of 33 collision shapes per robot) and steps without errors.
  - Self-collision over the clips (MuJoCo, `sim2sim/model.py` and `tools/check_self_collision.py`): no overlap standing or in the default pose. In the clips the forearm colliders add the right forearm in the torso in 8 % of the frames (left 0.3 %), on top of the reference's right upper arm in the torso in 20 % (Section 5.2).
  - Ball on the arm (Isaac, `tools/probe_arm_contacts.py` and `tools/analyze_arm_contacts.py`; stiff arms, the ball of the tasks dropped onto forearm, wrist and hand at 0.6–6.5 m/s): without the option the forearm stops the ball only through its neighbours, 14–37 mm deep in the real forearm. With it the ball stays 1.4 mm in the real surface at 0.7 m/s and 3–4 mm at 2.5–6.5 m/s (one drop at 4.7 m/s: 20 mm), like at the hand (0.4–1.1 mm gentle, 15–23 mm at 4.7–6.5 m/s).
- Sim2Sim code: `sim2sim/model.py` (MuJoCo G1 with the Isaac collision shapes read from the dump, PhysX hull vertex limit) and the tools above. The dump records the shape poses, takes `--forearm_colliders` and exits without Kit's hanging shutdown.
- Spec sections 3, 5.2, 6, 9 (pitfall 26) and 11 (items 7 and 8) updated.

### 2026-10-29: Sim2Sim step 2, standing in MuJoCo; nominal default pose in deploy.yaml
- `sim2sim/run.py`: G1 from `model.py` with the Isaac collision shapes, armature, drive gains and effort limits of the Isaac asset (MuJoCo position actuators, implicitfast integrator, 5 ms step), policy every 20 ms on the actor observation of the run's `deploy.yaml` (term order, scales, 5-frame history filled with the first observation, ball terms "not visible"), targets `default + 0.25 * effort / stiffness * action` clamped to the hard limits; start in the default pose or in frame 0 of a clip (feet placed on the floor); summary every 5 s and video (EGL, camera of the Isaac videos). `sim2sim/tools/export_onnx.py`: actor of a checkpoint to ONNX (loaded with `weights_only=True`), with a check file that `run.py` compares with onnxruntime (max difference 4e-6).
- Result with `model_3000.pt` of the third Unified run: stands 30 s from the default pose and 20 s from the first frames of clips 1, 15, 28, 45, 63, 82, 97 and 107 (pelvis 0.777–0.781 m, torso tilt 6.6–8.1°, drift 2.4–6.3 cm, one step per foot after a clip start, no clamped targets, no saturated torques). Isaac, same checkpoint, `wait` without pushes: pelvis 0.773 m, torso tilt 8.9°, 0.62 lift-offs per second, drift median 13 cm per wait. Results and videos in `analysis/2026-10-29_sim2sim_stehen/` on Marius' laptop.
- `utils/export_deploy_cfg.py` exports the nominal default pose (pitfall 27).
- Spec sections 2, 9 (pitfall 27) and 11 (item 9) updated.

### 2026-10-29: Sim2Sim step 3, ball observations in numpy
- `sim2sim/ball.py`: port of the VICON model of `BallCommand._update_perception` (measurement in the root yaw frame with the world velocity rotated, detection delay, ring buffer with per-throw latency, per-throw bias, noise, dropout bursts, occlusion in the catch box, ballistic coasting for `max_coast_steps`) and of the actor terms `ball_state`, `ball_visible` and `ball_intercept`, for one robot; parameters from `commands.ball_throw` of the run's `params/env.yaml`.
- Check: `tools/record_ball_obs.py` records in Isaac (Phase 2 env, zero actions, final throw distribution without curriculum) the true states and the actor's ball terms with a deterministic VICON model (latency 3 steps, detection delay 70 ms, no noise, bias or dropouts, ball hidden in the catch box on every throw); `tools/check_ball_obs.py` replays the states through the port. 16 robots × 1000 steps: identical in every step (largest difference 3e-6 m, visibility and catch-box test exact); with a wide catch box (`--wide_zone`) also 172 coasted and 353 lost steps, likewise identical. A detection delay that is a multiple of the 20 ms step (60 ms) differed by one step: float32 in Isaac against float64 in numpy; irrelevant with the random delays of the real model.
- Spec sections 2 and 11 (item 9) updated.

### 2026-10-29: Phase 2 throws aimed wider
- Why: the final throw distribution aimed ±0.1 m to the side, 0.3–0.4 m in front and 0.8–1.05 m high; with the velocity noise only 0.4 % of the throws arrived more than 0.3 m off the centre, so the robot practically always caught where it stood. A real thrower misses; the robot should catch whatever its arms can reach. The dataset gives no wider spread: its human catches lie within 0.25 m of the centre line and the catcher never stepped.
- Change (Phase 2 only): `TARGET_FORWARD_RANGE` (0.25, 0.45), `TARGET_LATERAL_RANGE` (−0.3, 0.3), `TARGET_HEIGHT_RANGE` (0.8, 1.15) m in the command config and as the curriculum's final values (easy keeps 0.3–0.4 m, ±0.03 m, 0.85–0.95 m). With the velocity noise 35 % of the throws arrive more than 0.2 m and 12 % more than 0.3 m to the side, 5 % beyond the catch box (±0.35 m). The old baseline caught 99.5 % of throws aimed ±0.3 m without stepping. Unified unchanged: its throws follow the clip.
- `eval_catch.py`: the `lateral` scenario aims ±0.45 m (beyond the trained range; until now ±0.3 m against a trained ±0.1 m).
- Verified: Phase 2 smoke run (3 iterations).
- Files: `BC/robots/g1_29dof/phase2/ball_catch_env_cfg.py`, `scripts/rsl_rl/eval_catch.py`. Spec sections 7.2 and 11 (item 1) updated.

### 2026-10-29: Phase 3 for lunges planned
- Decided with Marius: catches that need a step (throws beyond the arms' reach) come later as a separate Phase 3 on top of Phase 2, not inside Phase 2, which stays the safe standing catcher for the real robot. Prerequisites: real throw accuracy and recorded catches with a lunge. Section 11, item 11.

### 2026-10-29: Sim2Sim step 4, throws in MuJoCo
- `sim2sim/run.py --throws N`: episodes with the Phase 2 lifecycle (start pose; throw sampled in the heading frame from `commands.ball_throw` of the run's `env.yaml`; ball parked until the wait is over, then launched with the drag-compensated velocity; drag as an external force every control step; VICON model; catch check; Phase 2 terminations); ball as a MuJoCo sphere with the radius, mass and friction of `env.yaml` and a contact damping ratio for the bounce (`--ball_dampratio`, default 0.7, `--drop_test`); `--easy` for the curriculum's easy stage, `--ghost_ball` and `--debug_ball` for debugging. `sim2sim/ball.py`: `Throw` (sampling, launch velocity solver with drag) and `CatchState` (secured steps, confirmation, hold phase), ports of `BallCommand`.
- Result with `model_4500.pt` of the third Unified run (Section 11, item 9): 23–30 % catches in MuJoCo against 44 % in Isaac with a damped ball contact, 0 % when the MuJoCo ball bounces like Isaac's (0.4: e = 0.36), because the wedged ball is shot out of the squeeze; MuJoCo's contacts spring at every speed, PhysX's restitution only acts above its bounce threshold. Real-ball measurement extended accordingly (Section 11, item 2).
- Spec sections 2 and 11 (items 2 and 9) updated.
