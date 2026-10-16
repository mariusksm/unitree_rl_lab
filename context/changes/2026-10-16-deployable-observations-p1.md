# P1 Changes (2026-10-16) — Deployable Actor Observations, VICON Perception Model, Calibration on Recorded Throws

Implements **P1 items 5 and 7** of the 2026-10-16 analysis:
- 5: the actor gets only what the real robot has.
- 7: the ball is perceived the way the VICON pipeline delivers it.

Before that, the motion data were checked for a tracked ball (Section 1). The training `.pkl` files have none, but
VICON ball recordings exist for 21 takes. They were used to calibrate the perception model and to correct the P0
throw parameters (Section 2).

Builds on `2026-10-16-underhand-catch-p0.md`. Hands remain unchanged (rubber hands).

Files touched (`BC/` = `source/unitree_rl_lab/unitree_rl_lab/tasks/ball_catching/`):

| File | Change |
|---|---|
| `BC/mdp/commands/ball_command.py` | Perception model (`_reset_perception`, `_update_perception`, `perceived_pos_b`, `perceived_vel_b`, `ball_visible`); `flight_time_from_apex`; `apex_height_range` replaces `flight_time_range`; recalibrated throw and drag defaults; new perception config fields |
| `BC/mdp/commands/synced_ball_command.py` | Perception reset/update per motion cycle; catch-time check from the apex range |
| `BC/mdp/observations.py` | New `ball_perceived_pos_vel_b`, `ball_visible`, `ball_perceived_intercept_b`; `ball_intercept_b` uses the shared `_ballistic_intercept` |
| `BC/mdp/curriculums.py` | Docstring only |
| `BC/robots/g1_29dof/phase2/ball_catch_env_cfg.py` | New actor terms, critic `ball_visible`, curriculum (apex, perception) |
| `BC/robots/g1_29dof/unified/ball_catch_env_cfg.py` | Same actor as Phase 2 (reference critic-only), critic `ball_visible`, curriculum |
| `BC/robots/g1_29dof/phase1/ball_catch_env_cfg.py` | Actor without ball slots (reference + proprioception), critic `ball_visible` zero slot |
| `CLAUDE.md` | Key invariant on the observation contracts |
| `context/unitree_rl_lab_spec.md` | Sections 2, 4, 5, new 5.1, 6, 6.1, 7.1–7.3, 8, 9, 11, 12, changelog |
| `context/REVIEW.md` | M4 resolved |

Validation done (Section 6): exact-delay check of the perception model, default perception statistics, throw
statistics against the recordings, smoke runs of all three tasks, Unified → Phase 2 fine-tuning, exported
`deploy.yaml`.

---

## 1. VICON ball recordings

**Question:** do the motion data contain a tracked ball?

**Answer:** not the training data. All 156 `.pkl` in `pkl_isaac_lab_fixed_27_07_26/` and `pkl_isaac_lab_17_08_26/`
contain only `fps`, `root_pos`, `root_rot`, `dof_pos` (plus the empty `local_body_pos`, `link_body_list`).
The ball is recorded separately by the VICON tracker script (report ch. 4.2.4) as `samples.csv` per take.

**Where the ball data are:** the retargeting group's GMR fork `../GMR_for_fbx_files` (branch `fbx`) links the
tracker CSVs to the retargeted `.pkl`.
- `third_party/extract_pkl_and_csv.py` processes the CSVs:
  - keeps `tracked` / `acquired` rows,
  - **fills missing VICON frames by linear interpolation**, copying the previous row's timestamp (so interpolated
    rows are recognisable as duplicate timestamps),
  - drops takes with too large frame gaps or too short flights.
- Output for the session of 2026-07-06: `recordings/two_results_ver2.7z`, with `results/` (metres) and
  `results_scaled/` (positions ×0.9), each 21 × (`<n>_merged_filtered.csv` + `.pkl`).
- There is no 7z tool on the machine. The archive was read with a small stdlib-only reader (LZMA via Python's
  `lzma`) into a scratch folder.
- No other ball data exist locally:
  - `recordings/two_results_ver.7z` (48 MB) was not opened.
  - The pipeline that writes the ball into `.npz` (report ch. 6.1.2, for LocoMuJoCo) is not in the local repos.

**Format:** `session_id, frame_index` (VICON frame, ≈120 Hz), `timestamp_s` (workstation), `elapsed_s,
speed_m_s, event, x_m, y_m, z_m` (VICON world, metres).
- Ball and retargeted pelvis share the world frame: the catcher stands at ≈(0.9, −0.55) facing +x, the thrower at
  x ≈ 4.5 m.
- CSV and `.pkl` are aligned per take only; the CSV starts when the tracker acquires the ball.

**Usable takes:**
- 17, 18, 19, 21, 24, 25, 28, 29, 30, 33, 35, 40 are complete underhand throws to the catcher.
- 26, 31, 32, 36 lose the ball near the apex.
- 3–15 are a different setup without a complete throw: pelvis at x ≈ −0.7…−1.1, ball starting ≈2.6 m away and
  mostly static.

**Method (12 complete throws, unscaled `results/`):**
- Ballistic segment: sliding windows of 13 real frames. A window counts when the quadratic fit in z gives
  g = −9.81 ± 2.5, the residuals stay below 1 cm and the horizontal speed exceeds 1.5 m/s.
- Drag: grid search of k/m with the initial state refined by least squares.
- Dropouts: real frames binned into 20 ms control steps.

| Quantity | Value |
|---|---|
| Release distance from the catcher's pelvis | 3.1 m (2.8–3.3) |
| Release height | 1.03 m (0.93–1.10) |
| Apex height | 1.9 m (1.76–2.03) |
| Flight time release → hands | 0.77 s (0.69–0.86) |
| Release speed / arrival speed | 5.2 m/s / 4.4 m/s, arriving 30–55° downward |
| Arrival relative to the pelvis | 0.5–0.7 m in front, 0.3–0.8 m above (human catcher) |
| Quadratic drag k/m | 0.035 1/m (0.030–0.040); fit residual 1.2 cm → 0.65 cm with drag |
| Position noise | ≈4–5 mm per axis |
| Tracked before release (wind-up) | 0.7–1.7 s, ball moves 0.5–0.9 m |
| Control steps in flight without a measurement | 6.8 %, runs of 1–3 steps (11×1, 7×2, 2×3), start p ≈ 0.047 |
| VICON frame gaps in flight | mean 2.2 frames, up to 8 (67 ms, the tracker's tolerance) |
| After the catch | 4/12 lose the ball within 0.1 s, the rest track it for another 0.25–1.2 s |

**What it means:**
- **Ball:** the drag k/m = 0.035 fits a light ∅15 cm ball (112 g with c_d ≈ 0.37). The old sim ball (∅10 cm / 150 g:
  0.015) and a football (0.025) do not fit. The mjlab ball of P0 is consistent with the recordings.
- **Throws:** P0 took release height (1.1–1.7 m) and flight time (0.8–1.4 s) from LocoMuJoCo. Real underhand throws
  release lower (≈1.0 m) and fly faster (≈0.8 s). Corrected in Section 2.
- **Perception:**
  - The wind-up is visible to VICON, so the deploy side needs release detection before the ball counts as visible.
  - Dropouts come in short bursts.
  - The arms occlude the ball at the catch in part of the throws.

**Caveats:**
- One session, one thrower, one distance.
- The CSVs are pre-filtered (takes with long gaps were dropped), so real dropouts can be worse.
- `results_scaled/` multiplies positions by 0.9 to match the scaled robot motion. For throw parameters the
  unscaled values are right, because the real robot faces a real-size human thrower.

## 2. Throws re-calibrated on the recordings

| Field | P0 | Now | Basis |
|---|---|---|---|
| `release_height_range` | 1.1–1.7 m | **0.85–1.2 m** | recorded 0.93–1.10 m |
| `flight_time_range` | 0.8–1.4 s | **removed** | – |
| `apex_height_range` | – | **1.75–2.1 m** | recorded 1.76–2.03 m |
| `drag_coefficient_range` | 0.4–0.6 | **0.3–0.45** | k/m 0.030–0.040 → c_d 0.32–0.42 |
| `throw_distance_range`, `throw_direction_range`, `target_*` | 2–5 m, ±75°, belly target | unchanged | mjlab / LocoMuJoCo |

The apex now fixes the flight time: T = √(2(apex − z_release)/g) + √(2(apex − z_target)/g) (`flight_time_from_apex`),
giving 0.71–1.02 s. A sampled flight time independent of the distance would give 5 m throws at 8 m/s or 2 m throws
far above the head. With a fixed apex, farther throws are simply faster, as with a human. The apex is clamped to at
least 5 cm above release and target. `SyncedBallCommand._validate_catch_time` now uses the shortest flight time
possible with the configured ranges.

Curriculum (Phase 2, Unified): `apex_height_range` easy 1.85–1.95 m → final 1.75–2.1 m.

## 3. Deployable actor observations (P1 item 5)

**Problem:**
- The Unified actor saw the reference (`motion_command`, `motion_anchor_ori_b`). Throws are always synchronized to
  the clip, so the policy can time the catch from the clip clock, and no human thrower follows that clock.
- Phase 2 and Unified gave the actor `hand_pos` / `ball_relative`, which the C++ deploy cannot compute without
  forward kinematics.
- Phase 2's actor carried 64 dims of placeholders (M4).

**Change: two contracts.**

- **Actor contract, Phase 2 = Unified** (103 per frame × 5 = **515**):

  | # | Term | Dim | Function |
  |---|---|---|---|
  | 1 | `ball_state` | 6 | `ball_perceived_pos_vel_b` |
  | 2 | `ball_visible` | 1 | `ball_visible` |
  | 3 | `ball_intercept` | 3 | `ball_perceived_intercept_b` (plane 0.15 m above the root) |
  | 4–8 | `base_ang_vel`, `projected_gravity`, `joint_pos_rel`, `joint_vel_rel`, `last_action` | 93 | unchanged scales and noise |

  The ball terms carry no `ObsTerm` noise; all corruption lives in the perception model, so the three terms are
  consistent. The exported `deploy.yaml` lists exactly these terms, with no motion and no hand terms.
- **Critic layout, all three tasks** (308 per frame × 5 = **1540**): unchanged slots plus `ball_visible` after
  `ball_state` (Phase 1: zero slot). The critic keeps the true ball state, hand positions and the reference.
- **Phase 1** leaves the actor contract. Without a ball, only the reference tells its actor where in the clip it is.
  - Actor: `motion_command`, `motion_anchor_ori_b` and proprioception (157 × 5 = **785**).
  - Role: a **clip trackability check** for new clips, not a warm start.
  - `--finetune` from Phase 1 into Phase 2 now fails with a size mismatch instead of transferring silently.
- **Stage A → B** (mjlab recipe): train Unified, then fine-tune its actor in Phase 2, e.g.
  `--resume --resume_experiment unitree_g1_29dof_ballcatch_unified --load_run … --finetune --finetune_std 0.2`.

Checkpoints from before 2026-10-16 (actor 875) fit no task anymore; none existed.

## 4. VICON perception model (P1 item 7)

`BallCommand._update_perception` runs every control step after the catch check; `_reset_perception` runs on every
new throw (also per motion cycle in `SyncedBallCommand`). It produces `perceived_pos_b`, `perceived_vel_b` (ball
relative to the root and its velocity, yaw frame, zero while not visible) and `ball_visible`. Rewards, catch check
and critic keep using the true state.

1. **Measurement:** each step the true relative state is written into a ring buffer together with a validity flag.
   - A measurement is valid once the ball has flown for `detection_delay_range` (0.04–0.12 s; models the
     release-detection window). Before that, including the whole wind-up and parking phase, nothing is visible.
   - It is invalid during a **dropout burst**: start probability `obs_dropout_prob` 0.05 per step, length uniform in
     1..`obs_dropout_max_steps` = 3 steps.
   - It is also invalid while the ball is in the catch box, if this throw draws `hide_in_catch_zone_prob` (0.5): the
     arms occlude it.
2. **Delivery:** the robot gets the buffer entry of `obs_delay_steps_range` (1–3) steps ago, sampled per throw. Noise
   is added (`obs_pos_noise_std` 5 mm, `obs_vel_noise_std` 0.1 m/s), plus a per-throw position bias (±1 cm). Validity
   is delayed together with the state, so the first visible sample is always from the flight, never the parked ball.
3. **Coasting:** without a valid measurement the estimate is extrapolated ballistically from the last one, for at
   most `max_coast_steps` (3, ≈ the tracker's 8-frame tolerance). After that the ball counts as not visible.

| Parameter | Value | Source |
|---|---|---|
| `detection_delay_range` | 0.04–0.12 s | ballistic-fit window (recordings: 13 frames ≈ 0.11 s suffice) |
| `obs_delay_steps_range` | 1–3 steps | assumption (latency not measurable from the CSVs) |
| `obs_pos_noise_std` | 5 mm | recordings: 4–5 mm |
| `obs_vel_noise_std` | 0.1 m/s | assumption (depends on the deploy filter) |
| `obs_pos_bias` | ±1 cm per throw | assumption (marker centroid, calibration) |
| `obs_dropout_prob` / `obs_dropout_max_steps` | 0.05 / 3 | recordings: start p 0.047, runs 1–3 |
| `hide_in_catch_zone_prob` | 0.5 | recordings: a third lost at the catch, the rest tracked 0.25–1.2 s longer (as mjlab, which always hides it) |
| `max_coast_steps` | 3 | tracker gives up after 8 frames ≈ 67 ms |

The ring buffer is sized from the default (largest) delay range; the curriculum may only shrink it. The curriculum's
easy stage has clean perception (delay 1 step, no dropouts, no occlusion); the final stage uses the defaults.

## 5. Observation functions (`BC/mdp/observations.py`)

- `_ballistic_intercept(pos_b, vel_b, catch_height, max_time)`: the drag-free crossing with the catch plane in the
  yaw frame. `ball_intercept_b` (true state, critic) is now a thin wrapper and mathematically unchanged.
- `ball_perceived_pos_vel_b`, `ball_visible`, `ball_perceived_intercept_b` read the command's perceived state; the
  intercept is zero while the ball is not visible.

## 6. Validation

| Check | Result |
|---|---|
| Perception without noise, bias, dropouts and occlusion, delay fixed at 2 | perceived = true state 2 steps earlier, error 0 in pos, vel and intercept; never visible before the launch; first visible 0.08–0.14 s after the launch |
| Perception with defaults (128 envs, fixed root) | in flight 88.8 % fresh, 8.6 % coasted, 2.6 % not visible (recordings: 6.8 % without a measurement, so slightly pessimistic); missing runs mostly 1–3 steps; occluded throws visible in the catch box only while coasting (256/578 steps), others 858/860 |
| Sampled throws | release 0.85–1.20 m, distance 2.0–5.0 m, flight 0.74–0.98 s (mean 0.87), apex observed 1.69–2.14 m (mean 1.90, recordings 1.9) |
| Smoke runs (64 envs, 5 iterations) | Phase 2 and Unified: actor 515, critic 1540; Phase 1: actor 785, critic 1540; all exit 0 |
| Stage A → B | Unified checkpoint fine-tuned in Phase 2: actor loaded, std 0.3, iterations from 0; Phase 1 checkpoint → Phase 2: size mismatch 785 vs. 515 |
| `deploy.yaml` | observations `ball_state`, `ball_visible`, `ball_intercept`, `base_ang_vel`, `projected_gravity`, `joint_pos_rel`, `joint_vel_rel`, `last_action` |

## 7. Consequences and open points

- **Deploy pipeline** (spec Section 11): the live VICON side has to produce what this model assumes:
  - ball relative to the root in the yaw frame,
  - release detection before the ball becomes visible,
  - a ballistic filter with drag that bridges up to 3 steps and then reports "not visible",
  - optionally latency prediction (then shrink `obs_delay_steps_range` to the residual).

  The C++ deploy still needs the three ball observation terms.
- Measure the real pipeline latency.
- Recalibrate with the ball CSVs of the new underhand session (ask the retargeting group for them).
- With ball CSVs for the new clips, Unified could reproduce the recorded throw and read `CATCH_MOTION_TIME` and the
  catch point from the data (M2, M3). Mind that CSV and `.pkl` are aligned per take only.
- Not part of this change: end effector, actuator delay, post-catch / post-miss states, no-walking constraint.
