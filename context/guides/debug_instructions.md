# Debug Instructions

Quick-reference guide for verifying and debugging the ball catching pipeline at runtime.

---

## How to Run

### Phase 1 — Motion Imitation (mimic task)
```bash
cd /home/marius/Dokumente/Repositories/unitree_rl_lab
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-BallCatch-Phase1
```

### Phase 2 — Ball Catching (reward-based task)
```bash
cd /home/marius/Dokumente/Repositories/unitree_rl_lab
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-BallCatch-Phase2
```

### Phase 2 with viewer (non-headless)
```bash
cd /home/marius/Dokumente/Repositories/unitree_rl_lab
/home/marius/miniconda3/envs/env_isaaclab_downgrade/bin/python scripts/rsl_rl/train.py --task Unitree-G1-29dof-BallCatch-Phase2 --num_envs 32
```

### Phase 2 — transfer from Phase 1 checkpoint
```bash
# --resume is a boolean flag (not a path).
# --experiment_name:    the log subfolder name from Phase 1.
# --load_run:           the timestamp folder inside that experiment's logs.
# --checkpoint:         the .pt file name inside that run.

# Example with your actual data:
cd /home/marius/Dokumente/Repositories/unitree_rl_lab
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-BallCatch-Phase2 \
  --resume \
  --experiment_name unitree_g1_29dof_ballcatch_phase1 \
  --load_run 2026-07-27_11-15-04 \
  --checkpoint model_99.pt

# Smoke test:
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-BallCatch-Phase1 --max_iterations 100
# Check the generated checkpoint:
ls logs/rsl_rl/unitree_g1_29dof_ballcatch_phase1/
# Resume:
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-BallCatch-Phase2 \
  --resume \
  --experiment_name unitree_g1_29dof_ballcatch_phase1 \
  --load_run 2026-07-27_11-15-04 \
  --checkpoint model_99.pt \
  --max_iterations 10
```

### List all registered tasks
```bash
cd /home/marius/Dokumente/Repositories/unitree_rl_lab
/home/marius/miniconda3/envs/env_isaaclab_downgrade/bin/python scripts/list_envs.py
```

### Tensorboard
```bash
tensorboard --logdir logs/rsl_rl/
```

---

## 1. Ball Relative Position Debug

**What:** Verify that `ball_pos_relative` and `hand_body_pos` observation functions compute correct world-frame offsets from the robot root position.

**How:** Temporary debug logging is active in `BallCommand._update_metrics()`. It prints env 0's state every ~1000 steps (~20 seconds at 50 Hz).

**Output to look for:**
```
[DEBUG BallCommand] env=0 step=1000
  robot_root_pos    = [0.12, -0.03, 0.76]
  ball_world_pos    = [1.35, 0.18, 1.92]
  ball_relative     = [1.23, 0.21, 1.16]
  ball_height       = 1.920  (dropped=False)
  time_since_throw  = 0.12s
  left_wrist_yaw_link            = [-0.05, 0.22, 0.31]
  right_wrist_yaw_link           = [-0.05, -0.19, 0.30]
```

**Expected values:**

| Field | Expected | Check |
|---|---|---|
| `ball_relative` | `ball_world_pos - robot_root_pos` | Must be exact vector subtraction |
| `ball_relative.x` | ~0.5–2.5 | Ball should be in front of robot (positive x) |
| `ball_relative.y` | ~-0.75–0.75 | Random lateral offset |
| `ball_relative.z` | ~0.5–1.5 | Ball above robot base (chest height zone) |
| `dropped` | `ball_height < 0.2` | Signals re-throw |
| `time_since_throw` | ~0 after resample, increases between throws | Counts time in flight |
| `left_wrist.y` vs `right_wrist.y` | Opposite signs | Left wrist positive y, right wrist negative y |
| `wrist.x` | ~0 | Arms hang near body center when idle |

**Red flags:**
- `ball_relative == [0, 0, 0]` — ball not spawned or at robot origin
- `wrist.x > 1.0` — arms fully extended (unusual for idle pose)
- `time_since_throw` never resets — re-throw trigger broken
- `dropped` is `True` for many consecutive prints — ball spawns too low or falls immediately

**How to stop:** Remove the debug block from `ball_catching/mdp/commands/ball_command.py` lines 60–79, or set an environment variable guard.

---

## 2. Catching Rewards Verification (Step 6)

**What:** Verify the three new reward terms (`hand_to_ball`, `catch_success`, `ball_height_penalty`) appear correctly in the training log and behave sensibly.

**How:** Run training and check the episode reward breakdown in the iteration printout.

### Expected Terms in Log

You should see these in the `Episode_Reward/` section:
```
Episode_Reward/hand_to_ball: ____
Episode_Reward/catch_success: ____
Episode_Reward/ball_height_penalty: ____
```

### Expected Early Training Values (untrained policy)

| Reward | Expected Range | Why |
|---|---|---|
| `hand_to_ball` | 0.0–0.3 | Robot doesn't move hands toward ball yet — distance large → `exp(-d²/σ²)` small |
| `catch_success` | 0.0 | Robot doesn't catch — bonus never triggers |
| `ball_height_penalty` | -3 to -5 | Ball drops below 0.5m in most envs → penalty active |
| Total reward | -8 to -3 | Dominated by penalties + regularization |

### How to Read Progress

As the policy learns (after several hundred iterations):

| Signal | Meaning |
|---|---|
| `hand_to_ball` rising above 0.5 | Policy is moving hands toward the ball |
| `hand_to_ball` approaching 0.8–1.0 | Hands consistently near ball at some point in flight |
| `catch_success` showing occasional >0 values | Ball speed drops above ground — likely contact with robot |
| `ball_height_penalty` decreasing | Robot is reaching up to intercept the ball |
| `bad_orientation` no longer 1.0 | Robot stays balanced (not falling repeatedly) |

### Red Flags

| Observation | Problem |
|---|---|
| `hand_to_ball` stuck at 0.0 | Hands not moving — policy ignoring ball observation |
| `ball_height_penalty` always -5 | Ball drops instantly — spawn height too low |
| `catch_success` always ~1.0 | Threshold too loose — ball stops due to ground or other factor |
| No new reward terms visible in log | Import error — the 3 functions not found in `mdp` module |
| `hand_to_ball` values > 1.0 or NaN | Reward function bug — exp kernel should output [0, 1] |
| `bad_orientation` stays at 1.0 after sustained training | Robot never learns to balance — regularization too weak or catching reward too dominant |

### Tunable Parameters

| Param | Config Line | Purpose |
|---|---|---|
| `std: 0.15` | `hand_to_ball` → `params` | Catch tolerance radius. Increase to make the reward easier (wider catch zone). |
| `weight: 2.0` | `hand_to_ball` → `weight` | Main driving reward strength. Increase to prioritize reaching over posture. |
| `weight: 10.0` | `catch_success` → `weight` | One-time bonus magnitude. Keep larger than per-step rewards. |
| `vel_threshold: 2.0` | `catch_success` → `params` | Max ball speed (m/s) to count as caught. Decrease for stricter catch detection. |
| `min_height: 0.3` | `catch_success` → `params` | Min ball Z to count as caught. Must be above ground. |
| `min_height: 0.5` | `ball_height_penalty` → `params` | Ball Z below this triggers penalty. Increase to force earlier interception. |
| `weight: -5.0` | `ball_height_penalty` → `weight` | Penalty strength. More negative = robot prioritizes keeping ball up.

---

## 3. Ball Termination Verification (Step 7)

**What:** Verify `ball_dropped` and `ball_missed` terminations fire correctly.

**How:** The extended debug print in `BallCommand._update_metrics()` now includes termination flags for env 0. Also check the iteration printout.

### Debug Print Output

Every ~1000 steps you'll see:
```
[DEBUG BallCommand] env=0 step=2000
  ...
  time_since_throw  = 0.34s
  terminated        = False  (ball_dropped=0.0  ball_missed=0.0)
```

After the ball drops:
```
  terminated        = True  (ball_dropped=1.0  ball_missed=0.0)
```

### Iteration Printout

```
Episode_Termination/ball_dropped: 0.92
Episode_Termination/ball_missed: 0.08
```

### Expected Values

| Metric | Expected (untrained) | Why |
|---|---|---|
| `ball_dropped` in debug | 1.0 after ~0.5s | Ball hits ground → env terminates |
| `ball_missed` in debug | 0.0 | Ball drops before reaching 6m horizontal distance |
| `Episode_Termination/ball_dropped` | 0.8–1.0 | Drop is the dominant termination cause |
| `Episode_Termination/ball_missed` | 0.0–0.2 | Only fires if ball travels sideways fast enough |
| `Episode_Termination/bad_orientation` | may drop from 1.0 | Episodes end sooner from ball drops, robot falls less often |

### Red Flags

| Observation | Problem |
|---|---|
| Neither `ball_dropped` nor `ball_missed` seen in debug | Import error in `terminations.py` |
| `ball_missed` is 1.0 constantly | `max_distance` too low or throw velocity too high |
| `ball_dropped` stays 0.0 | Threshold `min_height: 0.1` too low (ball never reaches it) or ball not being thrown |
| Episodes last 20s (time_out=1.0) but ball dropped earlier | Termination not triggering — check function signature matches

---

## 4. Ball Event Randomization Verification (Step 8)

**What:** Verify that startup events randomize ball mass, friction, and restitution across environments.

**How:** The debug print now includes ball mass for env 0 and env 1. Mass is set at scene creation by `startup` events and does not change during training.

### Debug Print Output

```
[DEBUG BallCommand] env=0 step=1000
  ...
  ball_mass (env0)  = 0.1274 kg
  ball_mass (env1)  = 0.1832 kg
```

### Expected Values

| Observation | Expected | Why |
|---|---|---|
| `ball_mass (env0)` ≠ `ball_mass (env1)` | True | Startup event randomizes each env independently |
| Both masses near 0.15 | True | Range is ±0.05 (0.10–0.20 kg) |
| Masses never change between prints | True | Startup event runs once at scene load, not per episode |

### Red Flags

| Observation | Problem |
|---|---|
| Both masses exactly 0.1500 | `randomize_rigid_body_mass` not applied — check `asset_cfg: SceneEntityCfg("ball")` resolves correctly |
| Mass remains 0.0 | Ball not spawned or PhysX view not initialized for this env |
| Mass changes between debug prints | Event running in wrong mode ("reset" instead of "startup") |

### Note on Friction/Bounciness

Friction and restitution cannot be read back from PhysX via the public API. We rely on the fact that `randomize_rigid_body_material` is the same function used for the robot (line 92 in the env config), which already randomizes successfully. The ball passes the same `isinstance(asset, (RigidObject, Articulation))` check at `isaaclab/envs/mdp/events.py:199`.

---

## 5. Transfer Learning Checkpoint Resume

**What:** Verify that Phase 2 can load a Phase 1 checkpoint without dimension mismatch errors.

**Why:** Both phases now have identical observation shapes (policy=172, critic=304) via zero-padding. A dimension mismatch means the observation terms are not in the same order or have different counts.

### Smoke Test (Quick Verify)

```bash
cd /home/marius/Dokumente/Repositories/unitree_rl_lab

# Step 1 — Run minimal Phase 1 training to generate a checkpoint
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-BallCatch-Phase1 --max_iterations 100

# Step 2 — Find the generated checkpoint
ls logs/rsl_rl/unitree_g1_29dof_ballcatch_phase1/$(ls -t logs/rsl_rl/unitree_g1_29dof_ballcatch_phase1/ | head -1)/model_*.pt

# Step 3 — Resume Phase 2 from the Phase 1 checkpoint
# --resume (bool), --experiment_name (Phase 1 log folder), --load_run (timestamp), --checkpoint (.pt file)
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-BallCatch-Phase2 \
  --resume \
  --experiment_name unitree_g1_29dof_ballcatch_phase1 \
  --load_run 2026-07-27_11-15-04 \
  --checkpoint model_99.pt \
  --max_iterations 10
```

### Expected Output

When resume works correctly:
```
[INFO]: Loading model checkpoint from: logs/rsl_rl/.../model_100.pt
Actor Model: MLPModel(input_dim=172, hidden=[512, 256, 128], output=29)
Critic Model: MLPModel(input_dim=304, hidden=[512, 256, 128], output=1)
```
Then normal training output begins.

### Red Flags

| Observation | Problem |
|---|---|
| `size mismatch for ... weight: copying a param with shape` | Observation dimensions don't match between phases — check term order and count |
| `Missing key(s) in state_dict` | Phase 2 has observation terms Phase 1 doesn't — add dummy terms |
| `Unexpected key(s) in state_dict` | Phase 1 has terms Phase 2 doesn't — add dummy terms |
| `RuntimeError: Error(s) in loading state_dict` | Weight shapes incompatible — verify both configs have identical term order and dimensions |
| No `Loading model checkpoint` message | `--resume` flag not recognized or checkpoint path wrong |

### Production Resume Command

After full Phase 1 training completes:
```bash
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-BallCatch-Phase2 \
  --resume \
  --experiment_name unitree_g1_29dof_ballcatch_phase1 \
  --load_run 2026-07-27_11-15-04 \
  --checkpoint model_30000.pt
```

### Checkpoint Location

Checkpoints are saved automatically by RSL-RL at the interval set in the agent config (`save_interval = 500`):

```
logs/rsl_rl/<experiment_name>/<timestamp>/model_<iteration>.pt
```

For Phase 1 the experiment name is `unitree_g1_29dof_ballcatch_phase1`. Each run gets a timestamp directory.

