# P0 Changes (2026-10-16) — Underhand Catch: mjlab Ball, Underhand Lob, Air Drag, Arm-Contact Catch

Implements the **P0 steps** of the 2026-10-16 analysis of the interim report (`zwischenbericht.pdf`) against the
repository: ball model, throw generation and catch definition. Target behavior is the **underhand two-arm catch**
(ball lobbed from below, caught with both forearms/hands in front of the belly) that the upcoming reference clips
will show. The hands are **intentionally unchanged**: the rubber hands (collider merged into `*_wrist_yaw_link`)
stay for now.

The cross-task observation contract keeps all dimensions (policy 875 = 175×5, critic 1535 = 307×5 in all three
tasks). Only the meaning of the `ball_intercept` slot changes (catch plane height, Section 8). Phase 1 is untouched
(its ball slots are zeros).

Files touched (`BC/` = `source/unitree_rl_lab/unitree_rl_lab/tasks/ball_catching/`):

| File | Change |
|---|---|
| `BC/mdp/commands/ball_command.py` | Rewritten: underhand lob sampled and frozen at reset, drag-compensated launch, air drag force, catch check (box + both arms + relative speed + height), `launched` property, `arm_contact` state, new metrics |
| `BC/mdp/commands/synced_ball_command.py` | Adapted to the new base class: `_start_cycle`, flight time from `_sample_throw`, `launched` override, drag |
| `BC/mdp/rewards.py` | `hand_to_ball_distance_exp` replaced by `ball_interception_exp` (eq. 9.1 of the report); new `ball_arm_contact` |
| `BC/mdp/observations.py` | `ball_intercept_b` default plane 0.35 → 0.15 m, docstring on drag |
| `BC/mdp/curriculums.py` | `_lerp` recurses into tuples and rounds int fields (`secure_steps`) |
| `BC/robots/g1_29dof/phase2/ball_catch_env_cfg.py` | Ball, `ball_contact` sensor, ball DR, rewards, `undesired_contacts`, `ball_missed`, intercept plane, curriculum |
| `BC/robots/g1_29dof/unified/ball_catch_env_cfg.py` | Same changes as Phase 2 |
| `context/unitree_rl_lab_spec.md` | Sections 2, 5, 6, 6.1, 7.2, 7.3, 8–13 and changelog |
| `context/REVIEW.md` | M1 marked fixed |

Validation done (details in Section 11): solver test against RK4, PhysX throw accuracy, contact column mapping,
catch detection in a static cradle, 5-iteration smoke runs of Phase 1 / Phase 2 / Unified, throughput at 4096 envs.

---

## 1. Basis: what the interim report specifies

The report does not say "underhand" in so many words, except for the draft note in ch. 4.2.3 ("gefangen wird von
unten, geworfen wird von unten?") and the meeting minutes of 2026-07-03 (bigger ball "um Ball besser von unten
fangen zu können"). The throw parameters of the other subgroups describe exactly that kind of throw, so they were
taken over:

| Parameter | Value used | Source |
|---|---|---|
| Ball diameter / mass | 15 cm / 112 g | ch. 9.2 (mjlab) |
| Throw distance | 2 m → 5 m over the curriculum | ch. 9.4 (mjlab) |
| Throw direction | ±15° → ±75° over the curriculum | ch. 9.4 (mjlab) |
| Release height | 1.1–1.7 m above the ground | ch. 7.6.2 (LocoMuJoCo) |
| Aim point | 0.35 m in front (here 0.3–0.4 m), 0.8–1.05 m above the ground | ch. 7.6.2 (LocoMuJoCo) |
| Flight time | 0.8–1.4 s | ch. 7.6.2 (LocoMuJoCo) |
| Required hold time increases | `secure_steps` 25 → 50 | ch. 9.4 (mjlab) |
| Catch region box in front of the torso | catch box | ch. 9.5 (mjlab) |
| Both palms in contact, support from below | both arms must touch the ball | ch. 9.4, 9.4.1 (mjlab) |
| Interception and contact rewards | eq. 9.1, eq. 9.2 | ch. 9.4.1 (mjlab) |
| Reward ordering: small proximity < larger contact < even larger retention | weights 1 / 2 / 3 / 5 + bonus | ch. 9.4.1 (mjlab) |

**Not in the report (own choices, see Section 12):** restitution, drag coefficient, catch box dimensions, contact
threshold and grace window, lateral aim scatter, velocity noise, reward weights.

What the report says about the old Isaac setup (ch. 10.4: 1.0–2.5 m, flight time 0.3–0.6 s, aim at the chest) did not
match the real setup (ch. 4: thrower about 4 m away).

## 2. Ball model

**Problem:** the sim ball was ∅10 cm / 150 g (DR ±50 g) with restitution 0.6 (DR 0.4–0.8). The report specifies
∅15 cm / 112 g. Static probes (Section 11.1) showed that the ∅10 cm ball falls through a forearm cradle (0/45
placements held) while the ∅15 cm ball stays (34/45). The size decides whether the underhand catch works at all.

**Change** (Phase 2 and Unified scene + events):
- `SphereCfg(radius=0.075, mass=0.112)`, restitution 0.4 (placeholder, see Section 12), friction 0.5 unchanged.
- Startup DR: restitution 0.2–0.6 (was 0.4–0.8), mass ±0.015 kg (was ±0.05 kg), friction 0.3–0.8 unchanged.
- The mass DR must stay a **startup** event: `BallCommand` reads the per-env masses once at the first launch for the
  drag-compensated aim (Section 4). The comment in both configs says so.

## 3. Throw generation: underhand lob, frozen at reset

**Problem:** throws came from 1.0–2.5 m, released 1.6–2.3 m high, with 0.3–0.6 s flight time toward the chest.
The aim point was computed **at launch** from the robot's current pose, and `hand_to_ball` paid for hands near the
parked ball (review finding M1: walking toward the thrower made the throw shorter and easier).

**Change** (`BallCommand._sample_throw`, called at reset): all of the following are sampled in the robot's heading
frame and **frozen** until the next reset.

- Release point: horizontal distance `throw_distance_range` (2–5 m) from the root at azimuth `throw_direction_range`
  (±75°, positive = robot's left), height `release_height_range` (1.1–1.7 m) **above the ground**.
- Aim point: `target_forward_range` (0.3–0.4 m) and `target_lateral_range` (±0.1 m) relative to the root, height
  `target_height_range` (0.8–1.05 m) **above the ground**.
- Flight time `flight_time_range` (0.8–1.4 s), drag coefficient `drag_coefficient_range` (0.4–0.6).

The ball is parked at the release point for `throw_delay_range` (0.5–1.5 s, unchanged), then launched with
`_launch`, plus Gaussian velocity noise `velocity_noise` (0.1 m/s per axis, was 0.3). Heights are now referenced to
the ground (env origin), not to the root, so a crouching robot does not change the throw.

Removed config fields: `throw_lateral_range`, `throw_height_range`, `catch_forward`, `catch_lateral_std`,
`catch_height_range`, `catch_radius`. Removed helper: `_sample_spawn` (replaced by `_sample_throw`).

## 4. Air drag and drag-compensated launch

**Problem:** the sim had no air drag. For the light ∅15 cm ball the drag deceleration is about 10 % of g at 5 m/s
(k/m ≈ 0.045 1/m). A 4 m throw with 0.8–1.2 s flight lands 0.35–0.43 m shorter than the drag-free parabola. A
policy trained without drag expects the real ball decimeters too far.

**Change:**
- `_apply_drag` (every control step, after launches and parking): F = −k·|v|·v with k = ½·ρ·c_d·π·r² per env,
  zero for parked balls. Applied through `ball.permanent_wrench_composer` in the world frame. The composer is reset
  first so that it reads the current ball orientation for the world-to-body conversion. The force is held constant
  over the 4 physics steps of a control step. (`set_external_force_and_torque` is deprecated in Isaac Lab 2.3 and
  logs a warning on every call, so it is not used.)
- `_solve_launch_velocity`: fixed-point iteration on p(T) = p0 + v0·T + ½·g·T² + D(v0), 3 iterations.
  `_drag_displacement` evaluates D = ∫0^T (T − s)·a_drag(s) ds with the trapezoid rule on 16 intervals. The velocity
  along the flight is refined by two Picard passes. Without drag (c_d = 0) this reduces to the plain ballistic
  solution.
- Result: launches pass the aim point within 2.1 cm on average in PhysX. Without compensation, the ball misses it by
  ≈35 cm on average (Section 11.2).
- `ball_intercept_b` deliberately stays drag-free (the same simple formula on hardware). Early in the flight it
  predicts the ball up to ~0.4 m too far; the error shrinks with the time-to-go.

## 5. Catch detection: box, both arms, 1 s

**Problem:** "secured" was distance to the closest `wrist_yaw_link` origin < 0.2 m, relative speed to that hand
< 0.5 m/s, ball z > 0.5 m, for 25 steps (0.5 s). In static probes, the 0.5 s criterion fired for balls that rolled
off the arms afterwards (e.g. 11–13/45 placements "secured", 0/45 still held after 2.5 s). The wrist origin is about
0.1 m behind the rubber hand's centre and says nothing about the forearms a basket catch rests on.

**Change** (`_update_catch_state`): the launched ball counts as held when **all** of these hold:
- inside the catch box `catch_zone_x/y/z` = x 0–0.6 m, y ±0.35 m, z −0.3…+0.5 m (root yaw frame), i.e. in front of
  the torso from below the belly up to the chest, not at the side or above the head (as in mjlab, ch. 9.5)
- speed relative to the **root** < `secure_rel_vel` (0.5 m/s), ball z > `secure_min_height` (0.5 m)
- touching **both arms** (`require_both_arms=True`): a ball–arm contact force > `contact_force_threshold` (0.1 N;
  the ball weighs ≈1.1 N) counts if it happened within the last `contact_grace_steps` (5) control steps, so a rolling
  or slightly bouncing ball does not reset the counter.

`secure_steps` = 50 (1.0 s) in the final distribution, 25 early in the curriculum.

Contact sensing (`ball_contact` in both scenes): Isaac Lab reports filtered contacts only **one-to-many**, i.e. one
sensor body per env against several filter bodies. The sensor therefore sits on the **ball** and filters against the
8 links `{left,right}_{elbow,wrist_roll,wrist_pitch,wrist_yaw}_link`. History length 4 = physics steps per control
step; `_update_arm_contact` takes the strongest contact per link over that history. `BallCommand` assigns the filter
columns to the arms by the `left_` / `right_` name prefix and raises if one side is missing. New state:
`arm_contact` (bool, num_envs × 2) for the rewards; new metrics `ball_in_catch_zone`, `ball_arm_contact`.
`contact_sensor_name=None` drops the contact condition.

`launched` (new property): `time_until_throw <= 0` in `BallCommand`, `_launched` in `SyncedBallCommand`. It gates
the catch check, the drag and the ball rewards.

## 6. Rewards

**Problem:** `hand_to_ball` paid for the closest wrist near the ball. This favored one-handed reaching, and the term
was also active while the ball was parked at the release point (M1).

**Change** (Phase 2 and Unified; all ball shaping is zero before the launch):

| Term | Weight | Function | Meaning |
|---|---|---|---|
| `ball_reach` | 1.0 | `ball_interception_exp(std_dist=0.3)` | exp(−d²/0.3²), d = ball to the **midpoint of both hands** |
| `ball_interception` | 2.0 | `ball_interception_exp(std_dist=0.2, std_vel=1.0)` | eq. 9.1: exp(−d²/0.2² − v²/1.0²), v = ball speed relative to the mean hand velocity (soft catch) |
| `ball_arm_contact` | 3.0 | `ball_arm_contact` | eq. 9.2: fraction of arms touching the ball |
| `catch_success` | 500 | unchanged | one-time bonus, effectively 10 |
| `ball_secured_hold` | 5.0 | unchanged | 0.1 per held step (now with the new definition) |
| `ball_height_penalty` | −5 | unchanged | ball below 0.5 m |

## 7. `undesired_contacts` and terminations

- `undesired_contacts` (Phase 2 −1, Unified −0.1) now also excludes the elbow and wrist links and `torso_link`:
  `^(?!.*_ankle_roll_link$)(?!.*_elbow_link$)(?!.*_wrist_(roll|pitch|yaw)_link$)(?!torso_link$).+$`. In the basket
  catch the ball rests on the forearms and against the belly; that contact is the goal, not a fault. Falls on the
  torso still end the episode through `base_height` / `bad_orientation` (Phase 2) or the tracking terminations
  (Unified).
- `ball_missed`: horizontal distance > **6 m** (was 4 m). With release points up to 5 m away, 4 m would have
  terminated episodes while the ball was still parked.

## 8. Observation contract: intercept plane

`ball_intercept_b(catch_height=…)` moves from 0.35 m to **0.15 m above the root** (belly height; aim points are
0.8–1.05 m above the ground, root ≈0.76–0.79 m). Changed in Phase 2 and Unified, policy and critic; the function
default is 0.15 as well. Dimensions are unchanged; Phase 1 keeps its zeros. The exported `deploy.yaml` carries the
new value (`ball_intercept: params: {catch_height: 0.15}`).

## 9. Curriculum

`ball_throw_curriculum` (Phase 2 and Unified, steps 50k → 250k):

| Field | Easy | Final (= `BallCommandCfg` defaults) |
|---|---|---|
| `throw_distance_range` | 2.0–2.5 m | 2.0–5.0 m |
| `throw_direction_range` | ±15° | ±75° |
| `target_lateral_range` | ±0.03 m | ±0.1 m |
| `target_height_range` | 0.85–0.95 m | 0.8–1.05 m |
| `flight_time_range` | 1.0–1.2 s | 0.8–1.4 s |
| `velocity_noise` | 0.0 m/s | 0.1 m/s |
| `secure_steps` | 25 | 50 |

`_lerp` now recurses into tuples and rounds when both ends are ints: `secure_steps` is compared with `==` / `>=`
against an integer step counter, so a float such as 37.5 would have disabled the one-time bonus. All keys were
checked against the `BallCommandCfg` fields, since `setattr` silently accepts misspelled names.

## 10. Unified task (`SyncedBallCommand`)

- Same ball, throw geometry, drag, catch check, rewards, `undesired_contacts`, `ball_missed` and curriculum as
  Phase 2.
- The flight time is now sampled in `_sample_throw`; `_arm` only derives `launch_time = CATCH_MOTION_TIME ± 0.05 −
  flight_time`. `_start_cycle` (sample, park, reset catch state, arm) is used on reset and on every motion wrap.
- With flight times of 0.8–1.4 s and `CATCH_MOTION_TIME = 2.0` the launch falls at 0.55–1.25 s, inside the standing
  phase of the current clip. For clip 143, 2.0 s matches the moment the hands stop at the chest (≈1.96–2.0 s).
- **Do not train Unified before the underhand clips are in.** The current overhand chest clip pulls the hands to the
  chest while the ball now arrives at the belly. The aim point is the configured belly target, not the reference
  catch pose (review finding M2 still open).

## 11. Validation

All probes ran headless in Isaac Sim with scratch scripts outside the repo.

### 11.1 Static probes (before the change, motivated it)

Robot root fixed, joints PD-held in a pose, 45 placements per case.
Underhand cradle pose: shoulder pitch −0.3, elbow −0.6, shoulder roll ±0.05, shoulder yaw ∓0.4 (left −0.4),
wrist roll ∓1.57 (left −1.57), rest default.

| Pose | Ball | Dropped from rest, held 2.5 s | Thrown from 3 m, 0.8 s flight, held |
|---|---|---|---|
| Overhand reference pose (clip 143, t = 2.0 s) | ∅10 cm | 0/45 | – |
| Overhand reference pose | ∅15 cm | 0/45 | – |
| Underhand forearm cradle | ∅10 cm | 0/45 | 0/45 (e = 0.3) |
| Underhand forearm cradle | ∅15 cm | 34/45 | 8/45 (e = 0.6), 21/45 (e = 0.3), 32/45 (e = 0.0) |

### 11.2 Throw accuracy

- Solver vs. RK4 reference (dt 0.5 ms), 4096 random throws from the final distribution: mean miss 0.32 cm, max
  0.71 cm (16 nodes, 3 iterations; 8 nodes gave 1.6 cm mean). The drag-free aim under drag misses by 39.6 cm on
  average, max 81.9 cm.
- PhysX, Phase 2 env with a fixed root, zero actions, final distribution, no velocity noise, 64 envs. Metric: closest
  approach to the aim point, interpolated between control steps.

  | Variant | Mean | Max |
  |---|---|---|
  | drag + compensation (new) | 2.1 cm | 4.5 cm |
  | no drag at all | 1.2 cm | 1.6 cm |
  | drag, no compensation | ≈35 cm | ≈79 cm |

  The 2.1 cm residual comes from the drag force being held constant per control step.

### 11.3 Catch detection

Phase 2 env, fixed root, arms held in the cradle pose by constant actions, frontal throws from 3–4 m (flight
0.9–1.1 s, aim 0.24 m in front, ±0.03 m, 0.95 m high), `ball_caught` termination disabled to watch what happens
afterwards, 64 envs:
- Filter order `[left_elbow, left_wrist_roll, left_wrist_pitch, left_wrist_yaw, right_…]`, columns L = 0–3, R = 4–7.
- Lateral aim +0.22 m: only left columns register; −0.22 m: only right columns.
- 23/64 throws confirmed as caught (≥50 steps) into the static cradle; **23/23 still held** at least 50 steps later;
  no ball still held without confirmation. Contact steps left/right symmetric (54.6 / 54.7 on average).

### 11.4 Smoke runs and throughput

- `train.py --num_envs 64 --max_iterations 5` for Phase 1, Phase 2 and Unified: no errors, obs 875 / 1535, new
  reward terms and metrics logged, `deploy.yaml` exported. Untrained robots fall before the throw
  (`bad_orientation` ≈1.0 in Phase 2, `ee_body_pos` in Unified), as before.
- Phase 2 at 4096 envs: collection time 2.81 s per iteration vs. 2.73 s before the change (+3 %).

## 12. Placeholders and open points

- **Restitution 0.4 / DR 0.2–0.6 is a guess.** Drop-test the real ball (e = √(h_bounce / h_drop)) and also measure
  diameter, mass and friction; the probes show how strongly restitution decides whether a throw stays in the arms.
- Drag coefficient 0.4–0.6 (smooth sphere ≈0.47; the taped ball may differ), ρ = 1.2 kg/m³.
- Catch box, contact threshold / grace window, lateral aim scatter, velocity noise and reward weights are own choices.
- ±75° follows mjlab and is harder than the real setup (fixed thrower in front); narrow it if needed.
- Fit release point, aim point and flight time to the real VICON throws (`samples.csv` / `parabolas.csv`).
- Unified: new underhand clips, `CATCH_MOTION_TIME`, aim at the reference catch point (M2).
- Not part of P0 (see spec Section 11): end effector of the real robot, deployable actor observations (no motion
  reference / hand positions in the actor, ball-visibility flag), noise / delay / dropout on ball observations,
  pre-throw waiting / post-catch / post-miss states, no-walking constraint, deploy pipeline.
