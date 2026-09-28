# Review: Marius H's changes since `cd953ed`

Scope: `git diff cd953ed..HEAD`, excluding `context/`. The reviewed code is:
- `tasks/ball_catching/**`
- `scripts/mimic/pkl_to_npz.py` and `scripts/pkl_to_csv_without_hands.py`
- the edits to `scripts/rsl_rl/cli_args.py`, `scripts/list_envs.py` and `assets/robots/unitree.py`

Target stack: Isaac Lab 0.54.3 (repo `IsaacLab`), Isaac Sim 5.1, rsl-rl-lib 5.0.1.
Date: 2026-09-28.

## 1. Summary

**What I ran:**
- Short smoke runs: Phase 2 (16 envs, 3 iterations) and Unified (16 envs, 6 iterations).
- Two read-only behaviour probes (zero actions, 8 envs).
- Phase 1 already has a real run (`2026-08-31_15-06-41`).

**Results of those runs:**
- **All three tasks build and step without crashing.**
- Observation sizes are policy 875 = 175×5 and critic 1535 = 307×5, identical in all three tasks and in the Phase 1 checkpoint. The cross-task shape contract therefore holds.
- Every reward, observation, termination and event function I checked exists in the installed Isaac Lab with the signature used.

**Main problem:** the data is not correct.
- The motion conversion scripts take the wrong 29 columns out of the GMR `.pkl`.
- In every generated motion, the **right arm is replaced by seven left-hand finger joints**.
- The reference therefore drives the right arm into the torso: a probe measured about 2000 N contact on `torso_link` and `right_shoulder_yaw_link`.
- The Phase 1 checkpoint was trained on this corrupted reference.

| Task | Likely to run | Likely to train properly |
|---|---|---|
| Phase 1 (`…-BallCatch-Phase1`) | Yes (already ran 1657 iterations) | **No.** It imitates a corrupted right arm (C1). The tracking metrics look fine because the policy tracks what it is given. |
| Phase 2 (`…-BallCatch-Phase2`) | Yes (smoke run OK) | Uncertain. From scratch, its default exploration is too timid (H1). With `--resume`, it inherits the corrupted Phase 1 prior (C1), keeps Phase 1's std and critic (H1), and logs into the Phase 1 folder (H2). There is also a plausible pre-throw exploit (M1). |
| Unified (`…-BallCatch-Unified`) | Yes (smoke run OK) | **No** until C1 is fixed and the catch frame is set (M3). Even then, the ball aim is not tied to the reference catch pose (M2). |

## 2. Findings

| ID | Severity | File | Title |
|---|---|---|---|
| C1 | Critical (✅ fixed 2026-09-28) | `scripts/mimic/pkl_to_npz.py:113`, `scripts/pkl_to_csv_without_hands.py:41` | `dof_pos[:, :29]` puts left-hand finger joints into the right arm |
| H1 | High (✅ fixed 2026-09-28) | `tasks/ball_catching/agents/rsl_rl_ppo_cfg.py:39-55` | `Phase2PPORunnerCfg.init_noise_std=0.2` has no effect on resume and is too low from scratch; resume also loads the Phase 1 critic, optimizer and iteration |
| H2 | High (✅ fixed 2026-09-28) | `scripts/rsl_rl/cli_args.py:88-89` | `--experiment_name` override sends Phase 2 fine-tuning runs into the Phase 1 log folder; `play` then silently loads the wrong policy |
| M1 | Medium | `mdp/commands/ball_command.py:114-130`, `phase2/ball_catch_env_cfg.py:308-316` | Pre-throw exploit: hands are rewarded near the parked ball, and the aim point is computed at launch from the robot's current pose |
| M2 | Medium | `mdp/commands/synced_ball_command.py:117-121`, `ball_command.py:125-130` | Unified: the ball is aimed at a fixed chest point, not at the reference's (left-hand) catch pose |
| M3 | Medium | `unified/ball_catch_env_cfg.py:42` | `CATCH_MOTION_TIME = 2.0` is unverified; the clip has no retract phase |
| M4 | Medium | `phase2/ball_catch_env_cfg.py:235,250,271,285-288` | Transfer design: 64 policy inputs that were always informative in Phase 1 become constant zeros in Phase 2 |
| L1 | Low | `agents/rsl_rl_ppo_cfg.py:47`, spec, `merge_changes.md` | Documented Hydra override syntax `--agent.policy.init_noise_std=…` is wrong |
| L2 | Low | `assets/robots/unitree.py:20-21` | Hard-coded absolute asset paths committed |
| L3 | Low | `phase1/ball_catch_env_cfg.py:53-65`, `.gitignore` | `*.npz` is gitignored, so a fresh clone silently falls back to the Gangnam dance (warning only) |
| L4 | Low | `tasks/ball_catching/mdp/*` | Mimic MDP code is copied verbatim (`MotionCommand`, rewards, terminations, events), so two copies can drift apart |
| L5 | Low | `mdp/commands/synced_ball_command.py:66,70` | `float(motion.motion.fps)` on a shape-(1,) ndarray (NumPy deprecation) |

## 3. Details

### C1: Right arm replaced by left-hand finger joints (Critical)

**Location:**
- `scripts/mimic/pkl_to_npz.py:113`: `dof_pos = …data["dof_pos"])[:, :NUM_BODY_JOINTS]`
- `scripts/pkl_to_csv_without_hands.py:41`: `dof_pos = data["dof_pos"][:, :29]`

**What is wrong:**
- Both scripts assume the 53 GMR columns are laid out as "29 body joints, then 24 hand joints".
- The actual layout is **legs (12) | waist (3) | left arm (7) | left hand (12) | right arm (7) | right hand (12)**.
- Columns 22–28 are therefore left-hand finger joints, and they are written into the robot's right arm (SDK indices 22–28).

**Evidence:**
1. Per-column statistics of `pkl_isaac_lab_fixed_27_07_26/143_merged_filtered.pkl`:
   - Cols 15–21 move like an arm: shoulder pitch std 0.36, elbow range −0.71…1.0.
   - Cols 22–33 look like finger joints: near-constant, 0.2–1.0, pairwise equal ranges (26/27, 28/29, …).
   - Cols 34–40 mirror 15–21 with mirrored roll sign (35: −0.48…0.02 vs 16: 0.27…0.40) and the same elbow range. These are the real right arm.
   - Cols 41–52 look like fingers again.
2. Matching the trained clip `phase1/motions/143_merged_filtered.npz` column by column against the `.pkl` (mean+std distance ≈ 0): the Isaac-order right-arm joints map to `.pkl` columns **22–28**, not 34–40.
3. Reference poses from a probe over the clip (world frame):
   - The right wrist is frozen at ≈ (0.10, 1.20, 0.83), next to the pelvis at (0.07, 1.26, 0.79), for the whole clip.
   - The left wrist rises to z ≈ 1.16 (the catch).
4. A Phase 1 probe right after a reference-state reset measured net contact forces of **2315 N on `torso_link`, 2007 N on `right_shoulder_yaw_link`, 1695 N on `right_wrist_roll_link`**. The reference pushes the right arm into the torso.

**Impact:**
- Every `.npz` produced by either script is wrong.
- Phase 1 `model_1500.pt` learned the corrupted arm.
- Unified tracks it too.
- The `undesired_contacts` penalty and the `ee_body_pos` termination are constantly polluted by the self-collision.

**Fix:** select the correct columns and fail loudly on any other layout. Ideally, derive the indices from the GMR robot's joint-name list instead of hard-coding them.

```python
# GMR "unitree_g1 with hands" layout (53 dof):
# 0-11 legs | 12-14 waist | 15-21 left arm | 22-33 left hand | 34-40 right arm | 41-52 right hand
BODY_COLS = list(range(0, 22)) + list(range(34, 41))   # -> 29 joints in SDK order

dof = np.asarray(data["dof_pos"])
if dof.shape[1] == 53:
    dof = dof[:, BODY_COLS]
elif dof.shape[1] != 29:
    raise ValueError(f"unexpected dof_pos width {dof.shape[1]} in {pkl_path}")
```

After the fix:
1. Regenerate all `.npz` files.
2. Check them with `replay_npz.py`: both arms visible, no arm inside the torso.
3. Retrain Phase 1. Treat `model_1500.pt` as unusable for transfer.

The legs are also worth a visual check: hip pitch ≈ +0.36 rad with knee ≈ 0 is unusual. I could not verify the legs statically.

**Confidence:** high.

### H1: Phase 2 fine-tuning config does not do what it claims (High)

**Location:** `tasks/ball_catching/agents/rsl_rl_ppo_cfg.py:39-55`, registered as Phase 2's default runner in `phase2/__init__.py:10`.

**What is wrong:**
1. **Resume case.** `init_noise_std` only initializes the parameter. `train.py` calls `runner.load(resume_path)`, and in rsl-rl 5.0.1 this loads the full `actor_state_dict`, including `distribution.std_param` (`rsl_rl/algorithms/ppo.py:444-466`). The noise std is therefore Phase 1's learned value, not 0.2. I measured it in the checkpoint: 0.23–0.48 per joint. The same `load()` also restores:
   - the **Phase 1 critic**: its value function was fitted to a different reward, so the advantages in early Phase 2 updates are garbage;
   - the Adam state and learning rate;
   - `iter = 1500`.
2. **From-scratch case.** Phase 2 is the default path whenever `--resume` is not passed. Here std 0.2 on a freshly initialized policy is timid exploration. The docstring acknowledges this, but the default is still the wrong way round for a from-scratch run.

The same applies to the documented Unified warm start with `init_noise_std=0.3`: it is overridden by the loaded std.

**Fix:** load the actor only and set the std explicitly after loading (`train.py`, right after `runner.load`):

```python
# train.py, after the checkpoint path is resolved
if agent_cfg.resume:
    runner.load(resume_path, load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False})
    if args_cli.finetune_std is not None:          # new CLI arg, e.g. --finetune_std 0.2
        runner.alg.actor.distribution.std_param.data.fill_(args_cli.finetune_std)
```

Then register Phase 2 with `BasePPORunnerCfg` (std 1.0). With this, from-scratch is the sane default, and fine-tuning is explicit.

**Confidence:** high for the mechanism (verified in rsl-rl source and in the checkpoint keys). Medium for the claim that loading the critic hurts.

### H2: `--experiment_name` override mixes Phase 1 and Phase 2 runs (High)

**Location:** `scripts/rsl_rl/cli_args.py:88-89`, used by `train.py:149-169` and `play.py:80-91`.

**What is wrong:**
- `train.py` uses `agent_cfg.experiment_name` both to find the checkpoint to resume from and to decide where the new run is logged.
- The documented Phase 2 resume command passes `--experiment_name unitree_g1_29dof_ballcatch_phase1`. The new Phase 2 run is therefore written into `logs/rsl_rl/unitree_g1_29dof_ballcatch_phase1/<new timestamp>/`.

**Consequences:**
- `play --task …-Phase2` looks in the `…_phase2` folder and finds nothing.
- `play --task …-Phase1` (default `load_run=".*"` means the latest run) loads the **Phase 2** checkpoint into the Phase 1 env. The observation and action shapes are identical by design, so this loads **without any error**.
- The same happens for any later Phase 1 resume.

**Fix:** keep the checkpoint source separate from the log destination.

```python
# cli_args.add_rsl_rl_args
arg_group.add_argument("--resume_experiment", type=str, default=None,
                       help="Experiment folder to load the resume checkpoint from (defaults to experiment_name).")
# (remove the experiment_name override added in update_rsl_rl_cfg, or keep it but don't use it for resume)

# train.py
resume_root = os.path.abspath(os.path.join("logs", "rsl_rl", args_cli.resume_experiment or agent_cfg.experiment_name))
resume_path = get_checkpoint_path(resume_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
```

**Confidence:** high.

### M1: Pre-throw exploit in Phase 2 (Medium)

**Location:**
- `mdp/commands/ball_command.py:114-130`: `_launch` computes the catch zone and the ballistic velocity from the robot pose **at launch**.
- `phase2/ball_catch_env_cfg.py:308-316`: `hand_to_ball` is active for the whole episode.

**What is wrong:**
- The ball is parked 1.0–2.5 m in front of the robot for 0.5–1.5 s. During the easy curriculum stage this is 1.2–1.8 m.
- During that time `hand_to_ball` pays up to 0.04 per step if a hand is near the parked ball.
- Because the aim point is recomputed at launch, a robot that walks up to the "thrower" gets a very short, slow throw that is trivial to "catch". This is 10 points plus the hold reward for a behaviour that does not transfer.
- The probe showed a zero-action robot falls within ~1.2 s, so the exploit needs locomotion skill. That is why this is only medium likelihood, but nothing prevents it.

**Fix:** gate reach shaping on the launch, and freeze the aim geometry at reset.

```python
# rewards.hand_to_ball_distance_exp(..., command_name="ball_throw")
cmd = env.command_manager.get_term(command_name)
launched = cmd.time_until_throw <= 0.0            # BallCommand; use cmd._launched for SyncedBallCommand
return torch.exp(-(min_dist**2) / std**2) * launched.float()
```

Also either compute `catch_zone` in `_resample_command` (at reset) and store it, or terminate when the root moves more than ~0.5 m from its reset position before launch.

**Confidence:** medium (plausible optimum, not observed).

### M2: Unified aims the ball at a fixed chest point, not at the reference catch (Medium)

**Location:** `synced_ball_command.py:117-121` calls `BallCommand._launch`, which targets `catch_forward=0.4`, lateral ~N(0, 0.05), height 0.25–0.45 above root.

**What is wrong:**
- The reference catch is **one-handed with the left hand**. At t = 2.0 s the left wrist is at z ≈ 1.16 m, about 0.37 m above the pelvis.
- The height is consistent with `catch_height_range`.
- The forward and lateral offsets relative to the pelvis heading are not taken from the reference. The ball is aimed at the body centreline.
- Tracking rewards pull the left wrist to the reference point, while the ball arrives elsewhere. The two reward groups therefore conflict by design, and the curriculum only loosens this later.

**Fix:** aim at the reference hand position at the catch frame, with scatter around it.

```python
# in SyncedBallCommand._launch override (per env_ids)
m = self.motion_command
k = int(self.cfg.catch_motion_time * float(m.motion.fps))
# MotionLoader.body_pos_w is already indexed in cfg.body_names order
ref = m.motion.body_pos_w[k, m.cfg.body_names.index("left_wrist_yaw_link")]
ref_anchor = m.motion.body_pos_w[k, m.motion_anchor_body_index]
# express ref relative to the reference anchor, rotate into the robot's current heading,
# add to robot anchor position, then add catch_lateral_std scatter
```

(Sketch only: attribute names follow `motion_command.py`. Verify the anchor and yaw handling against `MotionCommand.body_pos_relative_w`, which already does this transform.)

**Confidence:** medium (reference positions measured; conflict inferred).

### M3: `CATCH_MOTION_TIME = 2.0` placeholder; clip has no retract (Medium)

**Location:** `unified/ball_catch_env_cfg.py:42`.

**What is wrong:**
- The launch time is derived from this constant. If it is wrong, the ball arrives before or after the hand is there, and the whole synchronization mechanism trains against the wrong moment.
- The probe shows the left wrist rising between t = 1.5 s (z 0.77) and t = 2.0 s (z 1.16), then staying up until the clip ends at 3.1 s. There is no retract.
- The merge docs claim the retract phase gets trained by episodes starting after the catch, but that only covers ~1 s of "arm held up".
- `_validate_catch_time` only warns if the ball can never launch, not if the time is merely wrong.

**Fix:** after C1 is fixed, determine the contact frame from the (fixed) `.npz`, e.g. the frame where the left wrist's upward velocity changes sign near its peak, or visually with `replay_npz.py`. Set the constant, and consider storing it per clip, e.g. as an extra key `catch_time` in the `.npz`.

**Confidence:** high that the value is unverified; medium on the correct value (1.8–2.0 s from coarse sampling).

### M4: Phase 1 → Phase 2 transfer feeds constant zeros into previously informative inputs (Medium)

**Location:** `phase2/ball_catch_env_cfg.py:235,250` (policy) and `271,285-288` (critic).

**What is wrong:**
- In Phase 1 the actor always received the 58-dim `motion_command` and the 6-dim `motion_anchor_ori_b`. The learned first layer depends on them.
- In Phase 2 these are exactly zero (× 5 history), which is out of distribution: the transferred actor sees an input it never encountered.
- Conversely, the ball slots were zero throughout Phase 1, so their first-layer weights are untrained random values.
- The "prior" that survives is therefore unclear. This is a structural reason to prefer the Unified task, and to judge Phase 2 fine-tuning against a from-scratch Phase 2 baseline.

**Fix:** none in code. Run the A/B comparison (resumed vs from scratch) before relying on transfer.

**Confidence:** medium.

### L1: Wrong Hydra override syntax in docs (Low)

**Location:** `agents/rsl_rl_ppo_cfg.py:47`, spec §4/§7, `context/changes/*`.

**What is wrong:** they document `--agent.policy.init_noise_std=1.0`. `train.py:51-58` forwards the unknown args to Hydra, which expects `key=value` without dashes. The correct form is `agent.policy.init_noise_std=1.0`.

**Confidence:** medium (standard Isaac Lab/Hydra usage; not executed).

### L2: Absolute local paths committed (Low)

**Location:** `assets/robots/unitree.py:20-21`.

**What is wrong:** they point to `/home/marius/...`. On the workstation or a teammate's machine, asset loading fails.

**Fix:** use `os.environ.get("UNITREE_MODEL_DIR", "<default>")`, or keep the path local and out of git.

**Confidence:** high.

### L3: Silent fallback to the dance clip (Low)

**Location:** `phase1/ball_catch_env_cfg.py:53-65` together with `.gitignore` (`*.npz`).

**What is wrong:** a fresh clone has no motion file, so Phase 1 and Unified train on the Gangnam dance, with only a console print as warning.

**Fix:** raise unless an explicit opt-in is set, e.g. `BALLCATCH_ALLOW_PLACEHOLDER=1`, or version the one chosen `.npz` (small: 155 frames).

**Confidence:** high.

### L4: Duplicated mimic MDP code (Low)

**Location:** `tasks/ball_catching/mdp/`.

**What is wrong:** `mdp/commands/motion_command.py`, `rewards.py`, `terminations.py`, `observations.py` and `events.py` are copies of `tasks/mimic/mdp/*` with only comments and imports changed (verified with `diff`). Fixes to either copy will not propagate.

**Fix:** import the mimic versions and add only the ball-specific terms.

**Confidence:** high.

### L5: NumPy scalar conversion (Low)

**Location:** `synced_ball_command.py:66,70`.

**What is wrong:** `motion.motion.fps` is `np.load(...)["fps"]`, which has shape (1,). `float(...)` on it works but raises a NumPy DeprecationWarning.

**Fix:** `float(np.asarray(fps).item())`.

**Confidence:** high.

## 4. Not verifiable statically: check in a short run

**Quick commands** (logs to a scratch folder so the real `logs/` stays clean):

```bash
./unitree_rl_lab.sh -t --task <Task> --num_envs 16 --max_iterations 10 --experiment_name /tmp/smoke_<task>
```

**Checks:**
1. **After fixing C1:** replay the regenerated clip. Both arms move, the right arm is not inside the torso, and the feet do not slide or penetrate. Check that `Episode_Reward/undesired_contacts` in Phase 1 drops to near 0 at the start of training (right now self-contact is constant).
2. **Catch feasibility without hands:** can a ball be held within 0.2 m of a `wrist_yaw_link` origin with relative speed < 0.5 m/s for 0.5 s at all? Try a scripted pose in `play` with the arms forming a cradle. If not, `ball_caught` never fires and only `hand_to_ball` shapes behaviour. Modelling the static 3D-printed mount (per the proposal) would change this.
3. **Throw geometry:** log the ball position relative to the robot at `time_since_throw ≈ flight_time` for a standing robot. It should be at (0.4, ~0, 0.25–0.45). In the zero-action probe the robot was already falling, so I could not confirm this.
4. **Phase 2 episode statistics over ~200 iterations:** `ball_dropped` and `ball_missed` should dominate early. `time_out` should be ≈ 0 apart from the random initial episode length. If `bad_orientation` stays near 1.0, the policy never gets to see a throw (the zero-action robot falls after ~1.2 s, while the throw lands at ~1–2 s).
5. **Unified early training:** `ee_body_pos` terminated 100 % in the 6-iteration smoke run, which is normal for an untrained policy. Confirm it falls over the first few hundred iterations. If it stays high after C1 is fixed, the 0.3 m wrist threshold is too tight for the catch reach.
6. **Resume behaviour after the H1/H2 fixes:** confirm the new run's folder, and confirm the printed std after load equals the requested value.
7. **Curriculum logging:** `Curriculum/ball_throws` goes 0 → 1 between steps 50k and 250k (~2.1k–10.4k iterations). The `*_taper` values move linearly (Phase 2: 100k–400k; Unified: 150k–500k). Rescale if your runs are much shorter than 30k iterations.
