# G1 Ball Catching — Analysis & Optimization Plan

**Scope:** `tasks/ball_catching` (Phase 1 = motion imitation / mimic, Phase 2 = RL catching), sim-to-real via VICON Shogun streaming, and the conceptual integration of the Unitree Dex3-1 dexterous hands.

**Status:** Analysis only — no code has been changed. All findings reference the current code with `file:line`.

---

## 1. Executive Summary

The two-phase architecture (mimic pre-training → RL fine-tuning with matched observation/network shapes via `dummy_zeros` padding) is a sound idea, and the Phase 1 tracking stack (BeyondMimic-style `MotionCommand` with adaptive sampling) is solid. However, **Phase 2 currently contains several defects that will prevent or severely degrade learning, and two of them silently break the Phase 1 → Phase 2 transfer entirely:**

1. **`ball_state` leaks absolute world coordinates including per-env origins** into the policy (values of ±80 m across the 4096-env grid, unnormalized). This alone can kill Phase 2 training.
2. **The action scaling differs between phases** (per-joint dict in Phase 1 vs. uniform `0.25` in Phase 2), so a transferred network commands different joint targets for the same output — the mimic prior is destroyed on load.
3. Ball-relative observations are expressed in **world axes while the robot's yaw is randomized over ±π** and yaw is not observable — the policy cannot consistently map "ball is at +x" to a body-frame motion.
4. The ball is always thrown from **world +x**, independent of the randomized robot heading.
5. The Phase 2 `undesired_contacts` penalty **includes the wrist links — the robot is punished for touching the ball with its hands.**
6. The catch reward (`speed < 0.5 && z > 0.5`) fires **without any proximity-to-hand condition** — a slow ball at its apex, or a ball resting on the head, yields the +10 catch bonus every step.
7. The ball terminations are **dead or stochastic** because the `BallCommand` re-throw thresholds are tighter than the termination thresholds and run in the same step loop.
8. The aim point of the throw (root + 1.0 m ≈ **1.76 m world height**) is **above the G1's head** (robot is ~1.32 m tall) — many throws are physically uncatchable.
9. Phase 1 still trains on the **Gangnam-style placeholder motion**, not a catching motion (known TODO, `phase1/ball_catch_env_cfg.py:89-90`).

Section 4 contains the full findings; Section 5 covers sim-to-real risks (VICON latency, velocity estimation, missing ball-observation noise); Section 6 is the dexterous-hand concept; Section 7 is the prioritized step-by-step roadmap.

**Strategic recommendation:** beyond the bug fixes, consider merging the two phases into a single DeepMimic-style task (motion-tracking rewards + ball rewards simultaneously, ball throw synchronized to the mocap catch instant). This removes the fragile "matched-shapes transfer" contract entirely and is the approach most likely to produce a natural, robust catching motion. Details in §4.4.

---

## 2. Current Architecture (as read)

| Component | Location | Summary |
|---|---|---|
| Phase 1 env | `tasks/ball_catching/robots/g1_29dof/phase1/ball_catch_env_cfg.py` | Motion tracking (BeyondMimic-style), 29-DOF joint-position actions, dummy ball obs (zeros), 4096 envs, 50 Hz control |
| Phase 2 env | `.../phase2/ball_catch_env_cfg.py` | Standing + ball catching, real ball obs, dummy motion obs (zeros), locomotion-style regularization rewards |
| MotionCommand | `mdp/commands/motion_command.py` | Loads npz (joint + body states), adaptive failure-bin sampling, teleport reset on motion end |
| BallCommand | `mdp/commands/ball_command.py` | Spawns ball 1.0–2.5 m in front (world +x), ballistic velocity toward a "catch zone", self-triggered re-throw on drop/miss/timeout |
| Rewards | `mdp/rewards.py` | Tracking exp-kernels (P1); `hand_to_ball_distance_exp`, `ball_caught`, `ball_height_penalty` (P2) |
| Obs helpers | `mdp/observations.py` | `ball_pos_relative`, `hand_body_pos` (world-axis relative), `dummy_zeros` padding |
| Mocap pipeline | `scripts/pkl_to_csv_without_hands.py`, `scripts/mimic/csv_to_npz.py` | GMR pkl (29 body + 24 hand dof) → drops hands → CSV → npz replay |
| PPO cfg | `agents/rsl_rl_ppo_cfg.py` | Shared runner for both phases, `[512,256,128]`, `empirical_normalization=False` |
| Robot asset | `assets/robots/unitree.py` (`UNITREE_G1_29DOF_MIMIC_CFG`) | 29-DOF G1, armature-derived PD gains — **no hands in the USD** |

Observation-shape contract between phases (verified consistent): `motion_command` 58 ⟷ `dummy_zeros(58)`, `ball_state` 6, `ball_relative` 3, `hand_pos` 6, `motion_anchor_ori_b` 6, critic `body_pos` 42 / `body_ori` 84. The *shapes* match; the *scales* do not (see F-2).

---

## 3. Phase 1 (Mimic) — Assessment

Overall this is the healthiest part of the codebase. It closely follows the proven BeyondMimic recipe: anchor-relative tracking rewards with exp-kernels, adaptive time-bin sampling of failure regions, pose/velocity/joint randomization on resample, early terminations on anchor/EE divergence. Observation noise and DR (friction, CoM, default-joint offset, pushes) are reasonable.

Findings (mostly minor):

- **P1-1 — Placeholder motion.** `phase1/ball_catch_env_cfg.py:90` still points at `G1_gangnam_style_V01.bvh_60hz.npz`. Everything downstream (catch-frame timing, arm workspace, hand aperture) depends on the real catching mocap. This is the single most important content item to deliver.
- **P1-2 — Mimic hand data is discarded.** `scripts/pkl_to_csv_without_hands.py` truncates `dof_pos[:, :29]`. When hands arrive (§6), this script and the npz schema are where hand joints re-enter the pipeline.
- **P1-3 — Catching is a *timed* skill, but adaptive sampling starts mid-motion.** For a catch motion, starting an episode in the middle of the catch (ball already "arrived" conceptually) is fine for pure imitation, but if you later synchronize a ball with the motion (§4.4), sampling must remain phase-aware (the `time_steps` already give you the motion phase — expose it as an observation, see F-9).
- **P1-4 — `undesired_contacts` regex is correct here** (excludes ankles *and* wrists, `phase1:286-293`) — note the asymmetry with Phase 2 (F-5).
- **P1-5 — Episode length 30 s vs. motion length.** With `resampling_time_range=(1e9, 1e9)` the motion loops via the internal teleport-reset (`motion_command.py:279-282`). Standard, but be aware every loop is a hard teleport; metrics spanning the boundary are polluted.

---

## 4. Phase 2 (RL Catching) — Findings

Ordered by severity. **F-1 … F-5 are training-breaking or transfer-breaking.**

### F-1 (critical) — `ball_state` observes absolute world coordinates including env origins
`phase2:213` uses `generated_commands("ball_throw")`, and `BallCommand.command` (`ball_command.py:51-56`) returns `root_pos_w` ‖ `root_lin_vel_w` raw. With 4096 envs at 2.5 m spacing, the grid spans roughly ±80 m — each env's policy sees a different, huge, constant position offset. With `empirical_normalization=False` (`rsl_rl_ppo_cfg.py:16`) these values enter the MLP unnormalized and will saturate the first layer. The critic gets the same values (`phase2:244`).

**Fix:** make the ball observation robot-centric: `pos_b = quat_apply_inverse(yaw_quat(robot.root_quat_w), ball.pos_w − robot.root_pos_w)` and the same rotation for velocity. This simultaneously fixes F-3 for this term and matches what you can actually compute on the real robot from VICON (§5).

### F-2 (critical) — Action scale differs between phases → transfer is broken
Phase 1: per-joint `UNITREE_G1_29DOF_MIMIC_ACTION_SCALE` (`phase1:127-129`, values ≈ 0.44–0.55 for most joints, computed as `0.25·effort/stiffness` in `unitree.py:706-717`). Phase 2: uniform `scale=0.25` (`phase2:196-198`). A network initialized from the Phase 1 checkpoint emits the same normalized actions, but they are mapped to roughly half-sized (and differently proportioned) joint offsets. The carefully pre-trained motor skills are scrambled at load time.

**Fix:** use `UNITREE_G1_29DOF_MIMIC_ACTION_SCALE` in Phase 2 as well. (Deployment configs must match too.)

### F-3 (critical) — Ball-relative observations in world axes + unobservable randomized yaw
`ball_pos_relative` (`observations.py:88-95`) and `hand_body_pos` (`observations.py:98-110`) subtract the root position but do **not** rotate into the base frame. `reset_base` randomizes yaw over (−π, π) (`phase2:151`) and the policy has no yaw observation (correctly — it isn't observable on-robot). The same physical situation ("ball 1 m in front of my chest") therefore produces arbitrarily rotated observation vectors. The task becomes partially unobservable; at best the policy learns a yaw-averaged mush.

**Fix:** rotate all relative vectors by `quat_apply_inverse(yaw_quat(root_quat_w), ·)`. This is also exactly the frame you will construct on the real robot.

### F-4 (critical) — Ball is thrown from world +x regardless of robot heading
`_resample_command` builds the spawn offset in world axes (`ball_command.py:100-109`), but the robot's yaw is randomized (F-3). A robot facing −x gets balls thrown at its back. Combined with F-1/F-3 it cannot even perceive where the ball is coming from.

**Fix:** rotate the spawn/catch-zone offsets by the robot's current yaw quaternion (spawn "in front of the robot"), or stop randomizing yaw. Keep yaw randomization — it is your defense against the world-frame/VICON-frame mismatch on hardware.

### F-5 (critical) — `undesired_contacts` punishes catching
Phase 2 regex `"(?!.*ankle.*).*"` (`phase2:361-368`) excludes only ankles. Ball-on-wrist contact — the desired terminal event — costs −1 per step. Phase 1 got this right (`phase1:286-293` also excludes `*_wrist_yaw_link`).

**Fix:** exclude wrist (later: hand/finger) links; ideally also filter to penalize only *robot-ground* and *self* collisions, not ball contacts, once hands are in.

### F-6 (high) — Catch detection has no notion of "in the hands"
`ball_caught` (`rewards.py:117-128`): `‖v_ball‖ < 0.5 ∧ z > 0.5`. Failure modes:
- Near-vertical throws have ‖v‖ ≈ horizontal speed ≈ small at apex → **+10/step for a ball in mid-air nowhere near the robot**.
- A ball resting on the head/shoulders/clavicle counts as caught.
- As a per-step reward (+10 × up to 250 steps until `max_flight_time` re-throws), it dwarfs everything else and invites trap-and-hold reward farming against the torso.

**Fix:** define caught as *ball within r of a hand/palm* ∧ *low ball velocity relative to that hand* (not world speed — you eventually want moving catches) ∧ sustained for N steps. Pay a **one-time** bonus (or better: terminate the episode on confirmed catch with a terminal bonus — clean credit assignment) plus a small bounded per-step "holding" reward.

### F-7 (high) — Dead / stochastic terminations vs. BallCommand re-throw
Step order in Isaac Lab (`manager_based_rl_env.py:204-232`): terminations → rewards → reset → **command update**. `BallCommand._update_command` re-throws at `z < 0.2` / `dist > 5.0` (`ball_command.py:167-175`), while the terminations trigger at `z < 0.1` / `dist > 6.0` (`phase2:379-386`).
- `ball_missed` can never fire (the ball would need to cross 5 m → 6 m in one 20 ms step).
- `ball_dropped` fires only when the ball falls > 0.1 m in one control step (needs > 5 m/s downward) — from typical throw heights the terminal speed is ~5–6 m/s, so this is a **coin flip per drop**. Episode semantics are effectively random.

**Fix — decide the episode contract explicitly.** Recommended: **single-throw episodes** first (BallCommand never self-resamples; drop/miss/catch all end the episode; the drop penalty becomes a terminal penalty). Multi-throw episodes are a later curriculum stage — then *remove* the ball terminations and let the command own the ball lifecycle, with a per-throw outcome metric logged in `BallCommand.metrics`.

### F-8 (high) — Throw target is above the robot's head
`catch_zone z = root_z + 1.0` (`ball_command.py:113-115`) ≈ 1.76 m world. The G1 is ~1.32 m tall; comfortable two-hand catch height is ~0.9–1.2 m world. Spawn heights (`root + 0.8…1.5` → 1.56–2.26 m) are fine as *origins*, but the aim point makes many throws uncatchable and teaches the robot that reaching is futile.

**Fix:** aim at `root_z + 0.25…0.45` (≈ 1.0–1.2 m world), and make lateral/height aim-scatter a curriculum variable. Verify against the wrist workspace of the actual mocap catch motion (P1-1).

### F-9 (medium) — Missing observations the task needs
- **No ball-flight phase/time information**: with history_length 5 (0.1 s) the policy sees a velocity estimate implicitly, but knowing *time-to-arrival* is the essence of catching. Add `time_since_throw` (already tracked, `ball_command.py:38`) or — better for sim2real — a **predicted interception point + time-to-intercept** computed from the ballistic model (same math you'll run on the VICON stream, §5).
- Policy `ball_state`/`ball_relative` have **no noise, no delay, no dropout** (`phase2:213-214`) while every proprioceptive term is corrupted. Real VICON data has occlusions, ~1–2 cm jitter after filtering, and 10–30 ms of pipeline latency.
- Critic (`phase2:236-266`) receives no privileged info beyond clean policy obs. Cheap wins: true ball mass/restitution, ball contact forces, true base lin vel is already there — add per-env ball material parameters.

### F-10 (medium) — Obs scaling/noise inconsistent between phases (transfer contract violated again)
`base_ang_vel`: P1 no scale (`phase1:151`), P2 `scale=0.2` (`phase2:223`). `joint_vel_rel`: P1 no scale, noise ±0.5 (`phase1:154`), P2 `scale=0.05`, noise ±1.5 (`phase2:226`). Same input feature, 5–20× different magnitudes across phases. Pick one set (P2's scaled versions are the better convention) and use it in both.

### F-11 (medium) — Posture rewards fight the catch
`base_height_l2` weight −10 pinned to 0.76 m (`phase2:343`) forbids crouching for low balls; `flat_orientation_l2` −5 (`phase2:342`) forbids leaning/reaching; `joint_deviation_arms` −0.1 (`phase2:311-324`) pulls arms to default against `hand_to_ball` +2. These are fine to bootstrap stable standing but must be tapered (curriculum weight decay) once catching starts working — or replaced by the mimic-style reward if you unify the phases (§4.4). Also note `alive` +0.15/step combined with the height/orientation pins yields a comfortable local optimum: *stand still, ignore ball* (the ball costs at most −5·(fraction of steps below 0.5 m) which the re-throw loop resets quickly).

### F-12 (low) — DR asymmetry between phases
Phase 2 lacks `randomize_joint_default_pos` and CoM randomization (Phase 1 has both, `phase1:208-225`); Phase 1 lacks the base-mass randomization Phase 2 has. Whichever phase produces the deployed policy needs the full sim2real DR set; keeping them identical also stabilizes transfer.

### F-13 (low) — Performance / hygiene
- `robot.find_bodies(...)` is called **every step** in `hand_to_ball_distance_exp` (`rewards.py:110`), `hand_body_pos` (`observations.py:108`), and the debug block (`ball_command.py:67`). Resolve indices once in `__init__`/via `SceneEntityCfg`.
- Debug print block in `_update_metrics` (`ball_command.py:61-90`) — remove for training runs.
- `mdp/__init__.py:6` imports the package **into itself**, and `.commands` is imported twice (`:8-9`). Harmless today, but the self-import is a latent circular-import trap.
- `tasks/ball_catching/__init__.py` is an empty file — works only because `import_packages` walks subpackages; fine, but document it.
- `ContactSensorCfg` on `Robot/.*` with `update_period = sim.dt` (`phase2:54-56, 410`) runs the sensor at 200 Hz over all ~30 bodies. Restrict the prim expression to the bodies actually used (ankles + wrists/hands) once catch detection is redesigned.
- Phase 2 `push_robot` interval is a degenerate range `(5.0, 5.0)` — synchronized pushes across all envs; widen to e.g. `(4.0, 8.0)`.

### 4.4 Strategic option — unify the phases (recommended direction)

The `dummy_zeros` shape-matching transfer is workable once F-2/F-10 are fixed, but it remains fragile (every obs edit must be mirrored twice) and Phase 2 can catastrophically forget the motion prior since nothing anchors it (PPO restarts with `init_noise_std=1.0`, which alone will wipe fine motor skills in the first iterations — if you keep two phases, lower it to ~0.2–0.3 for fine-tuning and/or freeze early layers initially).

The stronger design, used by DeepMimic and its descendants, is **one task**: keep the Phase 1 motion-tracking rewards active (possibly down-weighted, arms emphasized) *and* the ball rewards, with the **ball throw synchronized to the motion phase** — i.e., `BallCommand` launches the ball so that arrival time coincides with the catch frame of the mocap clip (the `MotionCommand.time_steps` gives you the phase; the ballistic flight time is chosen by you). The mocap provides the "how", RL learns the "when/where" corrections, and there is no transfer step at all. Curriculum: progressively widen throw distribution and decay tracking-reward weights so the policy can deviate from the clip for off-nominal balls.

---

## 5. Sim-to-Real Risk Register (VICON → robot)

| # | Risk | Where it bites | Mitigation |
|---|---|---|---|
| S-1 | **Frame convention**: policy trained on world-frame ball obs (F-1/F-3) cannot be fed from VICON | deployment blocked | Fix F-1/F-3: everything robot-yaw-frame. On HW: track robot base with VICON markers (or fuse robot odometry), compute `ball_in_base_yaw_frame` exactly as in sim |
| S-2 | **Ball velocity is not measured** — VICON gives positions; sim uses perfect `root_lin_vel_w` | obs distribution shift | Run a ballistic Kalman filter on the VICON stream; feed *filtered* pos/vel (or predicted intercept point, F-9). In sim, corrupt ball obs with matching noise + latency so train = deploy |
| S-3 | **Latency** (VICON pipeline + network + control loop, realistically 15–40 ms) with only 0.3–0.6 s flight time | timing skill fails | Model observation delay for ball terms in sim (1–2 step buffer, randomized); on HW forward-predict the ball by the measured latency using the ballistic model |
| S-4 | **Occlusion/dropout** of the ball marker near the hands (exactly at the catch) | catastrophic at the worst moment | Train with obs dropout on ball terms (hold-last-value); on HW the Kalman predictor coasts through gaps — the final 100 ms of a catch must work open-loop |
| S-5 | **Throw distribution mismatch** — sim's gravity-compensated synthetic throws vs. human throws | policy overfits to sampler | Record real throws with VICON now; fit the spawn/velocity distribution in `BallCommand` to the recorded data (spin/air drag are secondary at 0.15 kg / <8 m/s but measurable — bake into velocity noise) |
| S-6 | **No actuator latency/torque-delay modeling** anywhere | classic G1 sim2real gap | Add action delay randomization (0–2 sim steps) and the DR items from F-12 to the deployed phase |
| S-7 | **Catch verification on HW** differs from sim reward | none (reward is sim-only) — but define a success criterion for experiments (ball retained > 1 s within palm volume via VICON) |
| S-8 | **Safety**: 29-DOF whole-body policy reaching fast toward an incoming object near humans | commissioning | Joint-velocity/torque clamps in deploy config, dead-man switch, start with soft/light balls (current 0.15 kg is good), net testing area |

---

## 6. Dexterous Hand Integration Concept (Dex3-1)

**Hardware facts** (verified from `unitree_ros`): Dex3-1 = **7 joints per hand** (thumb 0/1/2, index 0/1, middle 0/1) → 14 additional DOF, 43 total. Ready-made combined model: `unitree_ros/robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf` (+ MJCF). No USD exists yet in `unitree_model` — asset conversion is Step H-1.

### 6.1 Asset & physics
- Convert `g1_29dof_with_hand_rev_1_0.urdf` via the existing `UnitreeUrdfFileCfg` path (or Isaac Lab's URDF converter) into a cached USD; create `UNITREE_G1_29DOF_DEX3_MIMIC_CFG` extending the current mimic config with a fifth actuator group `"hands"` (low stiffness ~0.5–1.5 N·m/rad, damping ~0.05–0.2, effort limit per Dex3-1 datasheet ≈ 2.45 N·m; verify against SDK limits).
- **Collision geometry is the make-or-break item for grasping**: replace/verify finger collision meshes with convex decompositions (or SDF colliders) — a 5 cm ball against coarse convex hulls will jitter or tunnel. Tune `contact_offset` (~5 mm) / `rest_offset` (~0 mm), raise solver iterations for contact-heavy scenes (position 8 → 16 to start), keep `sim.dt=0.005` initially but be prepared to go to 1/240 s if grasp contacts chatter. Watch `gpu_max_rigid_patch_count` (already raised) and `gpu_found_lost_pairs_capacity` with 4096 envs × finger contacts.
- Enable self-collision only between hand links and forearm, not the whole articulation, if solver cost explodes.

### 6.2 Action space — recommendation: **synergy (eigengrasp) actions, not raw joints**
Catching a single known ball does not need 14 independent finger DOFs, and raw finger actions cost exploration time and transfer poorly (finger PD behavior differs most between sim and real).

- **Recommended:** 1–2 synergy scalars per hand: action `g ∈ [0,1]` linearly interpolates each hand joint between a calibrated **open pre-shape** and a **closed sphere-grasp pose** (poses obtained by posing the hand around the 5 cm ball in sim once). Optional second synergy for thumb opposition. Action space grows 29 → 31 (or 33). The synergy mapping is trivially reproducible on the real Dex3 SDK (it is just a joint-position lookup), which keeps sim2real risk minimal.
- **Baseline/ablation (cheapest, do first):** hands not in the action space at all — a scripted reflex closes the hand when `‖ball − palm‖ < r_trigger` (event/action-term wrapper). This gives you a working full pipeline in days and quantifies how much the learned closure is actually worth.
- **Full 14-DOF joint actions:** only if synergy demonstrably limits performance (e.g., asymmetric one-hand catches). Defer.

### 6.3 Mimic phase with hands
VICON Shogun body capture almost certainly lacks reliable finger data (no gloves), and the GMR pipeline's 24 hand DOFs are currently discarded (`pkl_to_csv_without_hands.py`) and don't map 1:1 to Dex3's 7×2 anyway. Pragmatic approach:
- Keep hand joints **in the action space** during mimic (so network shapes and action semantics already include them) but **out of the tracking rewards**: no `body_names` entries for finger links, a weak `joint_deviation_l1` holding the open pre-shape instead.
- If the GMR hand channels turn out to be usable, retarget them to the two synergy values (aperture = mean finger flexion) rather than per-joint — that is robust to retargeting noise.
- Extend the npz schema (`csv_to_npz.py`) with optional `hand_joint_pos` now, so motion files don't need re-recording later.

### 6.4 Observations (added symmetrically to both phases / both groups)
- Hand joint positions (14) — real Dex3 provides joint feedback; add matching noise.
- Palm pose: replace `wrist_yaw_link`-based `hand_pos` with **palm-frame** positions (add palm sites/bodies), and add palm→ball vector in the palm frame for the pre-grasp alignment.
- Critic-only (privileged): binary finger-pad contact flags from a `ContactSensor` filtered on the ball prim, ball-in-palm-frame position.

### 6.5 Grasp-specific rewards (Phase 2 / unified task)
Staged shaping, gated by ball distance `d = ‖ball − palm‖`:
1. **Pre-grasp** (`d > 0.25 m`): palm normal alignment with ball approach direction (dot-product reward); aperture-open reward (penalize `g > 0.2` while ball far → prevents premature closing, the dominant failure mode).
2. **Contact** (`d < 0.25 m`): reward per finger-pad/palm contact with the *ball prim specifically* (ContactSensor `filter_prim_paths_expr` on the ball), bonus for ≥ 2 fingers + palm simultaneously ("caging" configuration).
3. **Secure**: ball velocity **relative to the palm** below threshold ∧ ball inside a palm-frame box, sustained N=25 steps (0.5 s) → terminal success bonus (this replaces `ball_caught`, fixing F-6 for good).
4. Keep all hand/finger links excluded from `undesired_contacts` (F-5), and exclude ball-robot pairs from that penalty entirely.
5. **Two-hand vs one-hand:** the current `min`-over-hands distance reward (`rewards.py:113`) actually encourages one-handed reaching. For a two-hand chest catch (most robust for G1), reward the *midpoint* of both palms to the ball plus an inter-palm-distance term near ball diameter — decide based on what your mocap catch actually does.

### 6.6 Curriculum for grasping
1. Scripted reflex close (6.2 baseline) → verifies physics & pipeline.
2. Learned synergy, ball thrown gently to a fixed chest-height target, tracking rewards strong.
3. Widen throw cone/speed, decay posture/tracking crutches, add obs latency/dropout.
4. (Stretch) one-hand catches, moving catches, multi-throw episodes.

### 6.7 Deployment notes for hands
- Dex3-1 is controlled via `unitree_sdk2` DDS topics at ≥ 100 Hz; the synergy→joint mapping runs in the deploy wrapper. Export the mapping into the deploy config (`deploy/robots/g1_29dof/config/policy/...`) alongside the existing policy config machinery (`utils/export_deploy_cfg.py`).
- The hand PD in sim must be re-identified against the real hand (log a chirp/step response once hardware is available) — finger dynamics are the least trustworthy part of the sim.

---

## 7. Prioritized Roadmap

Each step is small enough to validate independently. **Do not start hand integration (H-steps) before P0 is done and a no-hands baseline catches reliably** — otherwise you'll debug grasping on top of broken observations.

### P0 — Fix training-breaking defects (est. 1–2 days of edits, then retrain baseline)
1. **[F-1/F-3]** Robot-yaw-frame ball & hand observations: new obs functions `ball_pos_b`, `ball_vel_b`, `hand_pos_b` using `quat_apply_inverse(yaw_quat(...))`; replace `generated_commands("ball_throw")` in policy *and* critic. Delete env-origin leakage.
2. **[F-4]** Rotate `BallCommand` spawn + catch-zone offsets into the robot's yaw frame (`ball_command.py:100-115`).
3. **[F-8]** Lower catch-zone aim to `root_z + 0.25…0.45`; parametrize aim height/scatter in `BallCommandCfg` for later curriculum.
4. **[F-2]** Phase 2 action scale → `UNITREE_G1_29DOF_MIMIC_ACTION_SCALE`.
5. **[F-10]** Unify obs scales/noise across phases (adopt P2's scaled convention in P1).
6. **[F-5]** Fix Phase 2 `undesired_contacts` regex to exclude wrists.
7. **[F-7]** Single-throw episode contract: disable `BallCommand` self-resampling (or set its thresholds *looser* than terminations); terminations = dropped / caught-confirmed / robot-fell / timeout.
8. **[F-6]** Replace `ball_caught`: proximity-to-hand ∧ relative-velocity ∧ sustained → one-time/terminal bonus; keep `hand_to_ball` shaping.
9. **[F-13]** Hygiene: cache body indices, remove debug prints, fix `mdp/__init__.py`, narrow contact-sensor prims, widen push interval.

**Gate:** Phase 2 from scratch (no transfer) reaches a nonzero, climbing catch rate; log `catch_rate`, `time_to_first_contact`, `drop_rate` as `BallCommand` metrics.

### P1 — Content + transfer integrity (parallel to P0 where possible)
10. **[P1-1]** Record/retarget the real catching mocap (VICON → GMR → `pkl_to_csv_without_hands.py` → `csv_to_npz.py`); replace the Gangnam placeholder; visually verify with `replay_npz.py` (wrist workspace vs. catch-zone height!).
11. **Decide phase architecture**: (a) keep two-phase transfer → additionally lower `init_noise_std` to ~0.2 for Phase 2 fine-tuning, mirror DR sets (F-12); or (b) **unified DeepMimic-style task** (§4.4) with ball arrival synchronized to the motion catch frame — recommended. This decision determines where steps 12–13 land.
12. Add flight-phase information to obs: `time_since_throw` or predicted intercept point + time-to-go (F-9).
13. Curriculum: throw speed/cone/aim-scatter widening; taper `base_height`/`flat_orientation`/`joint_deviation_arms` weights on a schedule (F-11).

**Gate:** ≥ 80 % catch rate (wrist-trap catch, no hands) over the target throw distribution, with obs noise on ball terms enabled.

### P2 — Hands (§6)
14. **[H-1]** Build `g1_29dof_with_hand` USD + `UNITREE_G1_29DOF_DEX3_MIMIC_CFG` (actuator group "hands"); validate collision meshes with a static "ball dropped into posed hand" scene before any training.
15. **[H-2]** Scripted-reflex baseline (hands close on proximity) in the fixed Phase 2 env → measures physics readiness + baseline catch quality.
16. **[H-3]** Synergy action space (+2 DOF), open/closed calibration poses, mimic-phase handling per §6.3, obs additions per §6.4.
17. **[H-4]** Grasp rewards + ball-filtered contact sensing per §6.5, curriculum per §6.6.

**Gate:** secured-grasp success (ball held 0.5 s in palm frame) beats the scripted-reflex baseline.

### P3 — Sim-to-real hardening & deployment
18. Ball-obs corruption suite in sim: noise, 1–2 step randomized delay, dropout with hold-last (S-2/3/4); actuator delay DR (S-6); fit throw distribution to recorded human throws (S-5).
19. VICON deploy pipeline: robot-base tracking, ballistic Kalman filter + latency forward-prediction, base-yaw-frame transform **identical to the sim obs code** (share the math in one module), export via `export_deploy_cfg.py`.
20. Hardware bring-up ladder: replay policy on robot with *virtual* ball → real ball, gentle underhand throws, soft ball → target throws — with the safety measures from S-8.

---

## 8. Validation & Metrics to Add

- `BallCommand.metrics`: per-throw outcome (caught/dropped/missed/timeout), catch rate EMA, time-to-first-contact, intercept-point error.
- Phase 1: existing tracking metrics are good; add wrist-position error specifically during the catch window of the motion.
- Grasp phase: finger-contact count histogram, premature-close rate, ball-relative-velocity at first contact (proxy for "soft catching" — high values predict real-world drops).
- Always evaluate with the sim2real corruption suite ON — a policy that only catches with clean obs is a policy that only catches in RViz.
