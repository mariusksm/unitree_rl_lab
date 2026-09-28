# LATENT Pipeline — Manager-Level Reference

> Extracted from [LATENT](https://github.com/GalaxyGeneralRobotics/LATENT) (commit: `main`, Jun 2025)
> Only **Phase 1 (motion tracking)** is released. Phases 2–3 (DAgger distillation, high-level tennis policy) are unreleased.

---

## 1. Overview

| Aspect | LATENT | Our Isaac Lab Setup |
|---|---|---|
| Simulator | MuJoCo (MJX) | Isaac Sim (PhysX) |
| RL Framework | Brax (JAX) | RSL-RL (PyTorch) |
| Robot Model | `g1_mjx_w_racket_wo_ball.xml` | G1 29DOF USD + `UNITREE_G1_29DOF_MIMIC_CFG` |
| Control Type | PD torque | Joint position (via `JointPositionAction`) |
| Phase | Motion tracking (mocap imitation) | Phase 1 = mimic (same), Phase 2 = ball catching |

---

## 2. Environment Configuration

```python
env_config = {
    ctrl_dt: 0.02,            # 50 Hz control frequency
    sim_dt: 0.002,            # 500 Hz physics (10 substeps per control step)
    action_repeat: 1,
    episode_length: 1000,     # 20 seconds
    action_scale: 1.0,
    soft_joint_pos_limit_factor: 0.95,
    num_envs: 32768,          # 1024 × 32 parallel environments
}
```

XML model: `scene_mjx_racket_wo_ball_flat_terrain.xml`
- Flat floor plane
- G1 robot with 29 actuators
- Tennis racket (mesh + collision cylinder) attached to `right_wrist_yaw_link`
- Racket collision geom: cylinder, size `(0.12, 0.005)`, offset `(0.1025, -0.004, 0.4)` from wrist
- Racket center site with velocity sensor (for future ball-hit detection)
- **No ball body, no ball physics** (filename: `wo_ball`)

---

## 3. Actions

### 3.1 Action Space

- **Dimension:** 26 (all 29 joints minus 3 right wrist joints)
- **Type:** Position deviation from reference trajectory
- **Computation:** `motor_targets = ref_qpos + action × action_scale`

```
excluded_joints: right_wrist_roll_joint, right_wrist_pitch_joint, right_wrist_yaw_joint
active_joints:   all other 26 joints
```

### 3.2 Excluded Joints (Right Wrist)

The right wrist (holding the racket) is NOT controlled by the policy. Instead:
- Random PD targets within joint limits
- Targets resampled every 0.5–2.0 seconds (uniform random)
- This prevents the policy from learning wrist-specific strategies (future phase)

### 3.3 PD Torque Control

For every control step (50 Hz), 10 physics substeps at 500 Hz:

```
target_q = ref_qpos + action × action_scale (or random target for excluded joints)
torque = Kp × (target_q - current_q) + Kd × (0 - current_dq)
torque = clip(torque, -torque_limit, torque_limit)
```

**PD Gains (Kp / Kd):**

| Joint Group | Joints | Kp | Kd |
|---|---|---|---|
| Left hip yaw | 1 | 100 | 2 |
| Left hip roll | 1 | 100 | 2 |
| Left hip pitch | 1 | 100 | 2 |
| Left knee | 1 | 200 | 4 |
| Left ankle pitch | 1 | 80 | 2 |
| Left ankle roll | 1 | 20 | 1 |
| Right hip yaw | 1 | 100 | 2 |
| Right hip roll | 1 | 100 | 2 |
| Right hip pitch | 1 | 100 | 2 |
| Right knee | 1 | 200 | 4 |
| Right ankle pitch | 1 | 80 | 2 |
| Right ankle roll | 1 | 20 | 1 |
| Waist yaw | 1 | 300 | 10 |
| Waist roll | 1 | 300 | 10 |
| Waist pitch | 1 | 300 | 10 |
| Left shoulder pitch | 1 | 90 | 2 |
| Left shoulder roll | 1 | 60 | 2 |
| Left shoulder yaw | 1 | 20 | 1 |
| Left elbow | 1 | 60 | 2 |
| Left wrist roll | 1 | 20 | 1 |
| Left wrist pitch | 1 | 20 | 1 |
| Left wrist yaw | 1 | 20 | 1 |
| Right shoulder pitch | 1 | 90 | 2 |
| Right shoulder roll | 1 | 60 | 2 |
| Right shoulder yaw | 1 | 20 | 1 |
| Right elbow | 1 | 60 | 2 |
| Right wrist roll | 1 | — (excluded) | — |
| Right wrist pitch | 1 | — (excluded) | — |
| Right wrist yaw | 1 | — (excluded) | — |

**Initial state noise at reset:**
- Root XY: N(0, 0.1)
- Root yaw: N(0, 0.27 rad ≈ 15°)

---

## 4. Observations

### 4.1 Policy Observation (`state`)

7 keys, total **~85 dims** (56 with default joint filtering):

| Key | Description | Dim | Noise (additive uniform) |
|---|---|---|---|
| `dif_joint_pos` | `ref_qpos - current_qpos` | 29 | 0.03 |
| `dif_joint_vel` | `ref_qvel - current_qvel` (×0.05 scale) | 29 | 1.5 |
| `gvec_pelvis` | Gravity vector in pelvis frame | 3 | 0.05 |
| `gyro_pelvis` | Angular velocity in pelvis frame (×0.2 scale) | 3 | 0.2 |
| `joint_pos` | Current joint positions (offset to default pos) | 26 obs joints | 0.03 |
| `joint_vel` | Current joint velocities (×0.05 scale) | 26 | 1.5 |
| `last_motor_targets` | Previous motor targets | 29 | — |

`obs_joint_names`: All 26 joints that are NOT `right_wrist_roll`, `right_wrist_pitch`, `right_wrist_yaw`.

Observation history (`history_len`): 0 by default (no recurrence).

### 4.2 Privileged Observation (`privileged_state`, for Critic)

19 keys, **~200+ dims** (no noise applied):

| Key | Description |
|---|---|
| All 7 policy keys | (same as above, no noise) |
| `linvel_pelvis` | Pelvis linear velocity (3) |
| `dif_torso_rp` | Torso roll-pitch difference from ref (2) |
| `feet_contact` | Binary left/right foot contact (2) |
| `dif_feet_height` | Foot site height difference from ref (2) |
| `dif_rigid_body_pos_local` | Body position errors in pelvis-local frame (N_bodies × 3) |
| `dif_rigid_body_rot_local` | Body rotation errors, quaternions in local frame (N_bodies × 4) |
| `dif_rigid_body_linvel_local` | Body linear velocity errors in local frame (N_bodies × 3) |
| `dif_rigid_body_angvel_local` | Body angular velocity errors in local frame (N_bodies × 3) |
| `dif_root_height` | Root height difference from ref (1) |
| `dif_root_linvel_local` | Root linear velocity error in local frame (3) |
| `dif_root_angvel_local` | Root angular velocity error in local frame (3) |

---

## 5. Rewards (16 Components)

All tracking rewards use **Gaussian functions** of the error: `exp(-error² / sigma²)`.

### 5.1 Tracking Rewards (Positive)

| Reward | Scale | Sigma | Error Type |
|---|---|---|---|
| `rigid_body_pos_tracking_upper` | 1.0 | 1.0 | Abs position difference (upper body) |
| `rigid_body_pos_tracking_lower` | 0.5 | 1.0 | Abs position difference (lower body) |
| `rigid_body_rot_tracking` | 0.5 | 1.0 | `arccos(|quat_dot|)` |
| `rigid_body_linvel_tracking` | 0.5 | 5.0 | MSE of linear velocity |
| `rigid_body_angvel_tracking` | 0.5 | 50.0 | MSE of angular velocity |
| `feet_pos_tracking` | 2.1 | 1.0 | Abs position difference (foot sites) |
| `feet_rot_tracking` | 1.0 | 1.0 | `arccos(|quat_dot|)` (foot sites) |
| `joint_pos_tracking` | 0.75 | 10.0 | Abs position difference (active joints) |
| `joint_vel_tracking` | 0.5 | 1.0 | Abs velocity difference (active joints) |
| `root_linvel_tracking` | 1.0 | 1.0 | Abs velocity difference (pelvis) |
| `root_angvel_tracking` | 1.0 | 10.0 | Abs angular velocity difference (pelvis) |
| `roll_pitch_tracking` | 1.0 | 0.2 | Abs roll-pitch difference (torso) |
| `root_height_tracking` | 1.0 | 0.1 | Abs height difference (pelvis) |
| `feet_height_tracking` | 1.0 | 0.1 | Abs height difference (foot sites) |

**Body grouping for upper/lower tracking:**
- Upper body: `torso_link`, left/right `shoulder_roll`, `elbow`, `wrist_yaw`
- Lower body: `pelvis`, left/right `hip_roll`, `knee`, `ankle_roll`

### 5.2 Penalty Rewards (Negative)

| Reward | Scale | Computation |
|---|---|---|
| `penalty_action_rate` | -0.5 | `sum((motor_targets - last_motor_targets)²)` |
| `penalty_torque` | -2e-5 | `sum(torque²)` |
| `smoothness_joint` | -1e-6 | `sum(0.02 × vel² + acc²)` |
| `dof_pos_limit` | -10 | `sum(max(q - upper, 0) + max(lower - q, 0))` |
| `dof_vel_limit` | -5 | `sum(max(|dq| - limit, 0))` |
| `collision` | -10 | Self-collision pairs (see below) |
| `termination` | -200 | Applied when episode terminates |

**Self-collision penalized pairs:**
- `left_hand` ↔ `left_thigh`
- `right_hand` ↔ `right_thigh`
- `left_hand` ↔ `right_hand`
- `left_hand` ↔ `right_wrist_pitch`
- `right_hand` ↔ `left_wrist_pitch`

### 5.3 Reward Computation

```python
total_reward = sum(scale_i × reward_i) × dt
total_reward = clip(total_reward, max=10000)
```

---

## 6. Terminations

| Condition | Threshold |
|---|---|
| `fall` | `|root_height - ref_root_height| > 0.3 m` |
| `body_deviation` | Any valid body local position norm > 0.5 m from reference |
| `NaN` | Any NaN in qpos or qvel |
| `truncation` | Step count ≥ 1000 (episode length) |
| `traj_change` | Reference clip changes to next in dataset |

Valid bodies for deviation check: all tracked bodies except `pelvis` (root).

---

## 7. Reference Trajectory System

### 7.1 Data Pipeline

```
Human Tennis Motion Data → Retarget to G1 Skeleton → .npz files → TrajectoryHandler
```

### 7.2 `.npz` File Format (`TrajectoryData`)

| Field | Shape | Description |
|---|---|---|
| `qpos` | (T, 36) | 7 root (xyz + quat) + 29 joint positions |
| `qvel` | (T, 35) | 6 root velocity + 29 joint velocities |
| `xpos` | (T, N_bodies, 3) | Body positions in world frame |
| `xquat` | (T, N_bodies, 4) | Body orientations in world frame |
| `cvel` | (T, N_bodies, 6) | Body linear + angular velocities |
| `subtree_com` | (T, N_bodies, 3) | Body subtree center-of-mass |
| `site_xpos` | (T, N_sites, 3) | Site positions in world frame |
| `site_xmat` | (T, N_sites, 3, 3) | Site rotation matrices |
| `fps` | scalar | Original motion capture framerate |

### 7.3 `TrajectoryHandler` Operations

1. **Filter & reorder**: Match trajectory joint/body/site ordering to the current MuJoCo model
2. **Add defaults**: Insert dummy values for model joints/bodies not in the trajectory
3. **Interpolate**: SLERP for quaternions, linear for positions → align to 50 Hz control rate
4. **Recalculate velocities**: Finite differences on interpolated data
   - Angular velocity: quaternion difference → axis-angle → velocity
   - Linear velocity: position difference × frequency
   - Joint velocity: joint position difference × frequency
5. **Random start**: At each reset, randomly sample a trajectory clip index and start frame within it
6. **Smooth transitions** (optional): IK-based transition from default standing pose to first motion frame

### 7.4 Smooth Start/End Transitions

Optional preprocessing (`--smooth_start_end True`):
- QP-based IK solver (OSQP) computes foot positioning
- Cubic polynomial swing foot trajectories
- Foot position/orientation + CoM balance constraints
- Auto-detects 1 or 2 foot steps based on foot placement gap
- Inserted before the main motion data

---

## 8. Domain Randomization

### 8.1 Push Events

- Random 2D force impulse applied to root velocity
- Interval: 5–10 seconds
- Magnitude: [0.1, 1.0]

### 8.2 Motor Control Randomization

- PD gain scaling: Kp × [0.75, 1.25], Kd × [0.75, 1.25]
- RFI noise: random force/torque × [0.5, 1.5] × 0.1 × torque_limit

### 8.3 Physics Randomization

| Parameter | Distribution |
|---|---|
| Floor friction | N(0.4, 1.5) |
| Joint frictionloss | ×[0.75, 1.25] |
| Armature | ×[1.0, 1.05] |
| Torso CoM | ±0.15 m |
| Link masses | ×[0.75, 1.25] |
| Torso added mass | ±1.0 kg |
| Default joint pos | ±0.05 rad |

---

## 9. PPO Training Configuration

### 9.1 Hyperparameters

```python
learning_rate: 3e-4
discounting (γ): 0.97
gae_lambda: 0.95
clipping_epsilon: 0.2
entropy_cost: 0.01
max_grad_norm: 1.0
normalize_advantage: True
normalize_observations: False   # raw observations
reward_scaling: 1.0

unroll_length: 20
batch_size: 1024
num_minibatches: 32
num_updates_per_batch: 4         # 4 PPO epochs per batch

num_timesteps: 3_000_000_000     # total training steps
num_envs: 32768
```

### 9.2 Network Architecture

| Component | Hidden Layers | Output |
|---|---|---|
| Policy (actor) | [512, 512, 256, 256, 128], Swish | `action_size × 2` (mean + log_std, Tanh distribution) |
| Value (critic) | [512, 512, 256, 256, 128], Swish | scalar |
| Policy input | `state` dict (~56 dim, noisy) | — |
| Value input | `privileged_state` dict (~200+ dim, clean) | — |

---

## 10. Relevance to Our Phase 2 (Ball Catching)

### 10.1 What Ports Directly

| LATENT Concept | Isaac Lab Equivalent |
|---|---|
| Gaussian tracking rewards (`exp(-error²/σ²)`) | Already used in our mimic task rewards |
| Joint/body tracking error observation | `joint_pos_rel`, `joint_vel_rel`, mimic body observations |
| PD gain table | Relevant if switching from joint position to torque control |
| Action rate penalty | `action_rate_l2` reward term |
| Joint limit penalty | `joint_pos_limits` reward term |
| Collision penalty | `undesired_contacts` reward term |
| Fall / body deviation termination | `base_height`, `bad_orientation` terminations |
| Domain randomization (masses, friction, pushes) | Already in our mimic events |
| Smooth start transitions | IK-based default pose → motion (useful for mocap setup) |

### 10.2 What We Must Design (Not in LATENT)

| Needed for Phase 2 | How |
|---|---|
| Ball rigid body in scene | `RigidObjectCfg` + `SphereCfg` in `RobotSceneCfg` |
| Ball throw mechanism | `BallCommand` (extends `CommandTerm`): random spawn + initial velocity |
| Ball observations | `ball_pos`, `ball_vel`, `ball_pos_relative`, `hand_pos` |
| Ball-to-hand distance reward | `exp(-||hand - ball||²)` |
| Catch success reward | Contact sensor on wrist/hand + ball velocity drop |
| Ball dropped / missed termination | Ball z < 0.2, ball xy distance > 3m |
| Ball reset event | Reset ball position/velocity on episode reset |
| Transfer from Phase 1 | `--resume` loading Phase 1 checkpoint, same observation dimension |

### 10.3 Key Difference: Control Mode

| | LATENT | Our Phase 2 |
|---|---|---|
| Control | PD **torque** (26-dim) | Joint **position** (29-dim via `JointPositionAction`) |
| Output | Deviation from ref + PD | Absolute target angle |
| Substeps | 10 per control step (sim_dt=0.002) | 4 per control step (sim_dt=0.005, decimation=4) |
| Excluded joints | 3 right wrist joints | None (all 29 controlled) |

This means we control all joints including wrists — important for hand positioning to catch.
