# LATENT Mimic Adjustments

> Improvements from LATENT's motion tracking pipeline to integrate into
> `unitree_rl_lab/tasks/ball_catching/mdp/`

---

## 1. Action Smoothness Reward

**Status:** ❌ Not in unitree_rl_lab  
**Map to:** New reward function in `mdp/rewards.py`

LATENT combines velocity and acceleration penalties into a single term:

```
smoothness_joint = 0.02 × vel² + acc²   (weight: -1e-6)
```

Current unitree_rl_lab has them separately:
- `joint_vel_l2` (weight: -0.001)
- `joint_acc_l2` (weight: -2.5e-7)

**To add:** Implement `smoothness_joint()` in `ball_catching/mdp/rewards.py`.
Optionally remove the separate vel/acc terms or reduce their weights.

```python
def smoothness_joint(env, asset_cfg):
    asset = env.scene[asset_cfg.name]
    vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    acc = asset.data.joint_acc[:, asset_cfg.joint_ids]
    return torch.sum(0.02 * vel**2 + acc**2, dim=1)
```

---

## 2. Foot Tracking Rewards

**Status:** ❌ Not in unitree_rl_lab  
**Map to:** New reward function in `mdp/rewards.py`

LAENT tracks foot **site** positions and orientations — not just body link positions.

| Reward | Weight | Sigma | Purpose |
|---|---|---|---|
| `feet_pos_tracking` | 2.1 | 1.0 | Foot site position vs reference |
| `feet_rot_tracking` | 1.0 | 1.0 | Foot site orientation vs reference |

This is more precise than the generic `motion_body_pos` reward because it targets the
exact foot contact points, preventing sliding and ensuring proper foot placement.

**To add:** In `MotionCommand`, expose foot site data from the `.npz` (requires
`site_xpos` in the NPZ or computing from body transforms). Then add reward functions
in `rewards.py` that compare `command.foot_site_pos` / `command.foot_site_quat` with
actual robot foot state.

---

## 3. Root Velocity Tracking Rewards

**Status:** ❌ Not in unitree_rl_lab  
**Map to:** New reward function in `mdp/rewards.py`

| Reward | Weight | Sigma | Purpose |
|---|---|---|---|
| `root_linvel_tracking` | 1.0 | 1.0 | Pelvis linear velocity |
| `root_angvel_tracking` | 1.0 | 10.0 | Pelvis angular velocity |

Current mimic only tracks body positions and joint states — root velocity tracking
ensures smooth body motion, not just correct position per frame.

**To add:** Functions in `rewards.py` comparing `command.anchor_lin_vel_w` /
`command.anchor_ang_vel_w` with `command.robot_anchor_lin_vel_w` /
`command.robot_anchor_ang_vel_w`. The `MotionCommand` already exposes these properties —
just need the reward wrappers.

```python
def motion_root_linvel_tracking(env, command_name, std):
    command = env.command_manager.get_term(command_name)
    error = torch.sum((command.anchor_lin_vel_w - command.robot_anchor_lin_vel_w)**2, dim=-1)
    return torch.exp(-error / std**2)

def motion_root_angvel_tracking(env, command_name, std):
    command = env.command_manager.get_term(command_name)
    error = torch.sum((command.anchor_ang_vel_w - command.robot_anchor_ang_vel_w)**2, dim=-1)
    return torch.exp(-error / std**2)
```

---

## 4. Roll-Pitch Tracking Reward

**Status:** ❌ Not in unitree_rl_lab  
**Map to:** New reward function in `mdp/rewards.py`

| Reward | Weight | Sigma | Purpose |
|---|---|---|---|
| `roll_pitch_tracking` | 1.0 | 0.2 | Torso roll + pitch vs reference |

Keeps the upper body upright during motion. Critical for ball catching where torso
stability directly affects arm accuracy.

**To add:** Extract roll/pitch from `command.anchor_quat_w` and
`command.robot_anchor_quat_w`, compare, use Gaussian reward.

```python
def motion_torso_roll_pitch_tracking(env, command_name, std):
    command = env.command_manager.get_term(command_name)
    # Extract roll/pitch from quaternion -> euler or gravity projection
    ref_grav = quat_apply_inverse(command.anchor_quat_w, GRAVITY_VEC)
    robot_grav = quat_apply_inverse(command.robot_anchor_quat_w, GRAVITY_VEC)
    error = torch.sum((ref_grav[:, :2] - robot_grav[:, :2])**2, dim=-1)
    return torch.exp(-error / std**2)
```

---

## 5. Root Height + Foot Height Tracking

**Status:** ❌ Not in unitree_rl_lab  
**Map to:** New reward functions in `mdp/rewards.py`

| Reward | Weight | Sigma | Purpose |
|---|---|---|---|
| `root_height_tracking` | 1.0 | 0.1 | Pelvis Z position |
| `feet_height_tracking` | 1.0 | 0.1 | Foot site Z positions |

Current mimic checks `bad_anchor_pos_z_only` as a **termination** (threshold 0.25m)
but doesn't reward staying close. These rewards continuously encourage correct height.

**To add:** Compare Z component of `command.anchor_pos_w` with
`command.robot_anchor_pos_w` for root, and foot site Z for feet.

```python
def motion_root_height_tracking(env, command_name, std):
    command = env.command_manager.get_term(command_name)
    error = (command.anchor_pos_w[:, 2] - command.robot_anchor_pos_w[:, 2])**2
    return torch.exp(-error / std**2)
```

---

## 6. Velocity Limit Penalty

**Status:** ❌ Not in unitree_rl_lab  
**Map to:** New reward function in `mdp/rewards.py`

```
dof_vel_limit = sum(max(|dq| - vel_limit, 0))   (weight: -5)
```

Current mimic has `joint_pos_limits` (position limits, weight -10) but no velocity limit.
Catching motions involve fast arm swings — without this penalty, joints can overspeed.

**To add:** Compare `asset.data.joint_vel` with `asset.data.soft_joint_vel_limits`.

```python
def joint_vel_limits(env, asset_cfg):
    asset = env.scene[asset_cfg.name]
    vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    limits = asset.data.soft_joint_vel_limits[:, asset_cfg.joint_ids]
    return torch.sum(
        torch.clamp(vel - limits[:, :, 1], min=0) +
        torch.clamp(limits[:, :, 0] - vel, min=0),
        dim=1
    )
```

---

## 7. Self-Collision Pairs (Explicit)

**Status:** ❌ Not in unitree_rl_lab  
**Map to:** Replace `undesired_contacts` regex with explicit pairs in env config

Current mimic uses a regex filter on bodies:
```python
body_names=[
    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)..."
]
```

LATENT uses **explicit collision pairs** and penalizes force on them:

| Pair | Why |
|---|---|
| `left_hand ↔ left_thigh` | Arm hitting leg |
| `right_hand ↔ right_thigh` | Arm hitting leg |
| `left_hand ↔ right_hand` | Hands clapping |
| `left_hand ↔ right_wrist_pitch` | Cross-body contact |
| `right_hand ↔ left_wrist_pitch` | Cross-body contact |

**To add:** Either switch to explicit body name pairs in `undesired_contacts` or create
a new `self_collision_penalty` function that checks contact forces on specific body pairs.

Due to Isaac Lab's `ContactSensor` API, the simplest approach is to add a separate
reward term with explicit body name pairs:

```python
self_collision = RewTerm(
    func=mdp.self_collision_pairs,
    weight=-10,
    params={
        "sensor_cfg": SceneEntityCfg("contact_forces"),
        "body_pairs": [
            ("left_wrist_yaw_link", "left_hip_roll_link"),
            ("right_wrist_yaw_link", "right_hip_roll_link"),
            ("left_wrist_yaw_link", "right_wrist_yaw_link"),
            ("left_wrist_yaw_link", "right_wrist_pitch_link"),
            ("right_wrist_yaw_link", "left_wrist_pitch_link"),
        ],
        "threshold": 1.0,
    },
)
```

---

## 8. Termination Penalty

**Status:** ❌ Not in unitree_rl_lab  
**Map to:** New reward function in `mdp/rewards.py`

```
termination_penalty = -200  (applied on the step where episode terminates)
```

Current mimic applies **no penalty** when the episode ends — the agent only learns
from the stopped reward stream. The termination penalty gives the critic a strong
signal that the pre-termination state was undesirable, improving credit assignment.

**To add:** A reward term that checks `env.termination_manager.terminated` and returns
`-200` for terminated environments. This needs to be applied **after** terminations
are computed in the step loop.

⚠️ In Isaac Lab's `ManagerBasedRLEnv`, rewards are computed **before** terminations
in the default `step()` order. You may need to either:
- Use the **previous step's** termination as a flag, or
- Override `step()` to compute termination penalty after `self.termination_manager.compute()`

```python
def termination_penalty(env):
    return env.termination_manager.terminated.float() * -200.0
```

---

## 9. Root Noise at Reset

**Status:** ❌ Not in unitree_rl_lab (mimic)  
**Map to:** Add event in env config

LATENT perturbs root XY (±0.1m) and yaw (±15° ≈ 0.27 rad) at each reset:

```python
root_noise = EventTerm(
    func=mdp.reset_root_state_uniform,
    mode="reset",
    params={
        "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.27, 0.27)},
        "velocity_range": {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        },
    },
)
```

This builds robustness to imperfect starting positions — critical for Phase 2
where the robot might start from anywhere relative to the ball.

---

## 10. Richer Privileged Observations

**Status:** ❌ Not in unitree_rl_lab (mimic)  
**Map to:** New observation terms in env config

LATENT adds for the critic (privileged):

| Observation | Purpose |
|---|---|
| `feet_contact` (binary) | Left/right foot contact flag (2 dims) |
| `dif_body_pos_local` | Per-body position errors in pelvis frame (N_bodies × 3) |
| `dif_body_rot_local` | Per-body orientation errors as quaternions (N_bodies × 4) |
| `dif_root_linvel_local` | Root linear velocity error in local frame (3) |
| `dif_root_angvel_local` | Root angular velocity error in local frame (3) |

All in pelvis-local frame — invariant to world rotation.

**To add:** Observation functions in `mdp/observations.py`:

```python
def feet_contact(env, sensor_cfg, threshold=1.0):
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    force = contact_sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids]
    contact = (torch.norm(force, dim=-1) > threshold).float()

def body_pos_error_local(env, command_name):
    command = env.command_manager.get_term(command_name)
    # Convert body positions to pelvis-local frame, subtract
    pos_err = subtract_frame_transforms(...)
    return pos_err
```

---

## 11. Reward Clipping

**Status:** ❌ Not in unitree_rl_lab  
**Map to:** Configuration change (reward manager or post-processing)

```
total_reward = clip(sum(weights × values) × dt, max=10000)
```

Prevents rare reward spikes from destabilizing training. The Isaac Lab
`RewardManager` does not clip by default.

**To add:** Could be done in a wrapper or custom reward aggregation. Most
pragmatic approach: add `reward_clip` to `ManagerBasedRLEnvCfg` and apply after
`self.reward_manager.compute(dt)` in `step()`. Requires modifying the env base class
or overriding `step()`.

---

## Summary: Implementation Priority

| # | Feature | Phase | Effort | Impact |
|---|---|---|---|---|
| 1 | Root noise at reset | Both | Trivial (1 config line) | High |
| 2 | Termination penalty | Both | Medium (needs step order change) | High |
| 3 | Velocity limit penalty | Both | Easy (1 reward function) | Medium |
| 4 | Smoothness joint | Both | Easy (1 reward function) | Medium |
| 5 | Root height + foot height tracking | Phase 1 | Easy (2 reward functions) | Medium |
| 6 | Root velocity tracking | Phase 1 | Easy (2 reward functions) | Medium |
| 7 | Roll-pitch tracking | Both | Easy (1 reward function) | Medium |
| 8 | Foot tracking rewards | Phase 1 | Medium (needs NPZ site data) | Medium |
| 9 | Self-collision pairs | Phase 2 | Easy (config change) | Medium |
| 10 | Richer privileged obs | Phase 1 | Medium (observation functions) | Low |
| 11 | Reward clipping | Both | Medium (env base class change) | Low |
