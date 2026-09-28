# Phase 2 — BallCommand Implementation Plan

> Goal: Ball spawns in front of the robot and gets thrown with physics.  
> Robot learns to intercept it via reward shaping. Transfer from Phase 1 checkpoint.

---

## Overview: What Gets Changed

| # | File | Change | Why |
|---|---|---|---|
| 1 | `mdp/commands/ball_command.py` | **NEW** — `BallCommand` + `BallCommandCfg` | Generates throws, writes ball state to sim, exposes ball data |
| 2 | `mdp/commands/__init__.py` | **NEW** — re-exports `BallCommandCfg` | Makes `mdp.BallCommandCfg` available in env config |
| 3 | `mdp/__init__.py` | Add import for `ball_command` | Wires new command into MDP namespace |
| 4 | `phase2/ball_catch_env_cfg.py` | Multiple sections (see below) | Scene, commands, observations, rewards, terminations |

---

## 1. Ball Object in `RobotSceneCfg`

Add a programmatic sphere as a `RigidObject` — no USD file needed.

```python
from isaaclab.assets import RigidObjectCfg
import isaaclab.sim as sim_utils

class RobotSceneCfg(InteractiveSceneCfg):
    # ... existing: terrain, robot, lights, contact_forces ...

    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.05,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5,
                dynamic_friction=0.5,
                restitution=0.6,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.3, 0.0),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 5.0),      # out of the way at startup
        ),
    )
```

**Why each parameter:**
- `radius=0.05` — tennis ball size (~67mm diameter) mapped to generic ball
- `mass=0.15` — realistic for a small ball (adjustable)
- `collision_enabled=True` — ball physics needs collision to bounce
- `static_friction=0.5, dynamic_friction=0.5` — moderate grip with robot/hands
- `restitution=0.6` — some bounce but not too elastic (catches feel realistic)
- `init_state` — placed far away at startup; `BallCommand` repositions it on first resample

**Collision groups:** Default group 0 for both robot and ball means they DO collide.
The `collision_group=-1` on `terrain` puts it in a separate group, so ball-terrain
collision is handled via the global collision filter — ball falls through ground unless
terrain also collides. This is fine since the ball should hit the ground.

---

## 2. Contact Sensor (Widen Scope)

Current Phase 2 sensor only watches the robot:
```python
contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", ...)
```

Widen to cover **everything** in the environment:
```python
contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/.*", ...)
```

**Why:** We need to detect:
- Ball contact with robot hands/wrists → catch success (reward +85088+cvbn)
- Ball contact with ground → ball dropped (termination)
- Ball contact with robot body → ball bounced off (reward penalty)

The `SceneEntityCfg` filter in individual reward/termination terms selects which
bodies to check — the sensor just needs to cover all of them.

---

## 3. `BallCommand` Implementation (`mdp/commands/ball_command.py`)

### 3.1 Class Interface

```python
class BallCommand(CommandTerm):
    """Generates ball throws at random intervals. PhysX handles the physics."""

    cfg: BallCommandCfg

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.ball: RigidObject = env.scene[cfg.asset_name]      # "ball"
        self.robot: Articulation = env.scene["robot"]

        # Per-environment state buffers
        num_envs = env.num_envs
        self.time_since_throw = torch.zeros(num_envs, device=env.device)

    def __del__(self):
        super().__del__()

    @property
    def command(self) -> torch.Tensor:
        """Returns (num_envs, 6): ball world position + ball world velocity."""
        return torch.cat([self.ball.data.root_pos_w, self.ball.data.root_lin_vel_w], dim=1)

    def _update_metrics(self):
        """Track ball state for logging."""
        self.metrics["ball_height"] = self.ball.data.root_pos_w[:, 2].clone()
        self.metrics["ball_speed"] = torch.norm(self.ball.data.root_lin_vel_w, dim=1).clone()
        self.metrics["time_since_throw"] = self.time_since_throw.clone()

    def _resample_command(self, env_ids):
        """Throw the ball toward the robot from a random position."""
        if len(env_ids) == 0:
            return

        # Get robot base positions for the env_ids being resampled
        robot_pos = self.robot.data.root_pos_w[env_ids]

        # Spawn position: in front of robot, above, with noise
        spawn_pos = robot_pos.clone()
        spawn_pos[:, 0] += 1.0 + torch.rand(len(env_ids), device=self.device) * 1.0     # 1–2m ahead
        spawn_pos[:, 1] += (torch.rand(len(env_ids), device=self.device) - 0.5) * 1.5    # ±0.75m lateral
        spawn_pos[:, 2] += 1.2 + torch.rand(len(env_ids), device=self.device) * 0.8      # 1.2–2.0m height

        # Throw velocity: toward the robot's catch zone, with noise
        # Catch zone: robot base + (0.5, 0, 1.0) → chest height, slightly in front
        catch_zone = robot_pos.clone()
        catch_zone[:, 0] += 0.5
        catch_zone[:, 2] += 1.0 + torch.randn(len(env_ids), device=self.device) * 0.1

        direction = catch_zone - spawn_pos
        flight_time = 0.3 + torch.rand(len(env_ids), device=self.device) * 0.3   # 0.3–0.6s flight

        # Account for gravity: v0 = (Δp - 0.5*g*t²) / t
        g = torch.tensor([0, 0, -9.81], device=self.device)
        throw_vel = (direction - 0.5 * g.unsqueeze(0) * flight_time.unsqueeze(1)**2) / flight_time.unsqueeze(1)
        # Add velocity noise
        throw_vel += torch.randn(len(env_ids), 3, device=self.device) * 0.3

        # Build root state: [pos(3), quat(4), lin_vel(3), ang_vel(3)]
        default_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)  # wxyz
        zero_ang_vel = torch.zeros(len(env_ids), 3, device=self.device)
        root_state = torch.cat(
            [spawn_pos, default_quat.unsqueeze(0).expand(len(env_ids), -1), throw_vel, zero_ang_vel],
            dim=1,
        )
        self.ball.write_root_state_to_sim(root_state, env_ids=env_ids)

        self.time_since_throw[env_ids] = 0.0

    def _update_command(self):
        """Advance time, detect ball dropped/out-of-bounds, trigger resample."""
        self.time_since_throw += self._env.step_dt

        # Ball dropped below catch zone → resample
        ball_z = self.ball.data.root_pos_w[:, 2]
        dropped = ball_z < 0.2

        # Ball too far from robot → resample
        ball_xy = self.ball.data.root_pos_w[:, :2]
        robot_xy = self.robot.data.root_pos_w[:, :2]
        too_far = torch.norm(ball_xy - robot_xy, dim=1) > 5.0

        # Ball timed out → resample
        timed_out = self.time_since_throw > 5.0

        resample_ids = (dropped | too_far | timed_out).nonzero(as_tuple=False).flatten()
        if len(resample_ids) > 0:
            self._resample(resample_ids)

    def _set_debug_vis_impl(self, debug_vis):
        # TODO: add sphere marker to visualize throw trajectory
        pass  # skip for initial implementation
```

### 3.2 Config Class

```python
@configclass
class BallCommandCfg(CommandTermCfg):
    """Configuration for the ball command."""

    class_type: type = BallCommand

    asset_name: str = MISSING       # name of the ball RigidObject in the scene ("ball")
```

### 3.3 Key Design Decisions

| Decision | Rationale |
|---|---|
| **Resample on drop/too_far/timeout** instead of fixed interval | Ball naturally resets after being caught or missed — no need for a timer that interrupts mid-flight |
| **Physics-based throw** not scripted path | PhysX handles gravity, collision, bouncing — more realistic |
| **Throw velocity computed with gravity compensation** | Ball arrives near the catch zone regardless of distance |
| **Randomized spawn distance (1–2m), height (1.2–2.0m), lateral (±0.75m)** | Diverse trajectories prevent overfitting to one throw pattern |
| **Flight time 0.3–0.6s** | Fast enough to challenge the robot, slow enough to be catchable |

---

## 4. CommandsCfg Replacement (Phase 2 env config)

Replace the velocity command placeholder with the ball command:

```python
@configclass
class CommandsCfg:
    ball_throw = mdp.BallCommandCfg(
        asset_name="ball",
        resampling_time_range=(1e9, 1e9),  # effectively never auto-resample; manual via conditions
        debug_vis=False,
    )
```

**Why `resampling_time_range=(1e9, 1e9)`:** We don't want time-based auto-resampling.
The ball is thrown once, and only re-thrown when it's caught, dropped, or times out.
The `_resample()` call inside `_update_command()` handles all resampling triggers.

---

## 5. ObservationsCfg — Ball Observations (Phase 2 env config)

Add ball-specific observation terms to the policy and critic groups.

**Policy observations:**
```python
ball_pos = ObsTerm(func=mdp.generated_commands, params={"command_name": "ball_throw"})
```
This returns `[ball_pos_w(3), ball_vel_w(3)]` from the `BallCommand.command` property.

Also add hand/wrist body positions for catching accuracy:
```python
hand_pos = ObsTerm(
    func=mdp.robot_body_pos_b,
    params={"command_name": "ball_throw"},
)
```

Full policy observation pipeline (replace existing):
```python
@configclass
class PolicyCfg(ObsGroup):
    ball_cmd = ObsTerm(func=mdp.generated_commands, params={"command_name": "ball_throw"})
    hand_pos = ObsTerm(...)                # wrist positions relative to base
    joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, ...)
    joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, ...)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, ...)
    projected_gravity = ObsTerm(func=mdp.projected_gravity, ...)
    last_action = ObsTerm(func=mdp.last_action)
```

**Why these observations:** The policy needs to know where the ball is (ball_cmd),
where its hands are (hand_pos), and its own state (joints, base, gravity) to
coordinate a catching motion.

---

## 6. RewardsCfg — Catching Rewards (Phase 2 env config)

**Note:** This is where we add LATENT-inspired improvements (from `LATENT_mimic_adjustments.md`).

Keep (*) or replace (~) existing locomotion rewards:
- ~ `track_lin_vel_xy` → remove (no velocity commands in Phase 2)
- ~ `track_ang_vel_z` → remove
- ~ `feet_gait` → remove (static stance for catching)
- * All regularization rewards stay (`joint_vel_l2`, `joint_acc_l2`, `action_rate_l2`,
  `dof_pos_limits`, `joint_deviation_*`, `flat_orientation_l2`, `base_height_l2`,
  `feet_slide`, `feet_clearance`, `undesired_contacts`, `alive`)

Add catching-specific rewards:
```python
hand_to_ball = RewTerm(
    func=mdp.hand_to_ball_distance_exp,
    weight=2.0,
    params={"ball_asset": "ball", "hand_body_names": ["left_wrist_yaw_link", "right_wrist_yaw_link"], "std": 0.15},
)
catch_success = RewTerm(
    func=mdp.ball_caught,
    weight=10.0,
    params={"ball_asset": "ball", "contact_sensor": "contact_forces",
            "hand_body_names": [".*wrist.*", ".*hand.*"], "vel_threshold": 2.0},
)
ball_height = RewTerm(
    func=mdp.ball_height_above_threshold,
    weight=-5.0,
    params={"ball_asset": "ball", "min_height": 0.5},
)
```

**Why each reward:**
- `hand_to_ball_distance_exp` — main driving reward: hands should be near the ball.
  Gaussian so it saturates when hands are close (no over-optimization).
- `catch_success` — Large episodic bonus when ball contact detected on hands/wrists
  AND ball speed drops below threshold (ball velocity absorbed by hands).
- `ball_height` — Penalize ball falling below catch height. Ensures robot doesn't
  learn to wait for the ball to drop — it must reach up to catch.

---

## 7. TerminationsCfg — Ball Terminations (Phase 2 env config)

Add to existing terminations:
```python
ball_dropped = DoneTerm(
    func=mdp.ball_below_height,
    params={"ball_asset": "ball", "min_height": 0.1},
)
ball_missed = DoneTerm(
    func=mdp.ball_far_from_robot,
    params={"ball_asset": "ball", "max_distance": 6.0},
)
```

**Why:** Reset the environment when the ball is clearly un-catchable, so the agent
doesn't waste time (and compute) on lost balls. This also gives the termination
penalty reward a clean signal about what states to avoid.

---

## 8. EventsCfg — Ball Randomization (Phase 2 env config, optional)

Optional domain randomization for robustness:
```python
ball_physics = EventTerm(
    func=mdp.randomize_rigid_body_material,
    mode="startup",
    params={
        "asset_cfg": SceneEntityCfg("ball"),
        "static_friction_range": (0.3, 0.8),
        "dynamic_friction_range": (0.3, 0.8),
        "restitution_range": (0.4, 0.8),
        "num_buckets": 16,
    },
)
ball_mass = EventTerm(
    func=mdp.randomize_rigid_body_mass,
    mode="startup",
    params={
        "asset_cfg": SceneEntityCfg("ball"),
        "mass_distribution_params": (-0.05, 0.05),
        "operation": "add",
    },
)
```

**Why:** Ball properties (friction, mass) vary in the real world. Randomizing during
training prevents the policy from memorizing a specific ball behavior.

---

## 9. Wiring: `mdp/__init__.py`

```python
from .commands.ball_command import *      # exports: BallCommandCfg
```

Create empty `mdp/commands/__init__.py` to mark as package.

The env config already imports `unitree_rl_lab.tasks.ball_catching.mdp as mdp`,
so `mdp.BallCommandCfg` is directly accessible.

---

## 10. Additional MDP Functions Needed

These reward/termination/observation functions need to be added to the MDP module
(files exist — just add functions):

| Function | File | Purpose |
|---|---|---|
| `hand_to_ball_distance_exp` | `mdp/rewards.py` | Gaussian reward for hand-ball proximity |
| `ball_caught` | `mdp/rewards.py` | Binary bonus for catch detection via contact + velocity |
| `ball_height_above_threshold` | `mdp/rewards.py` | Penalty for ball dropping below catch zone |
| `ball_below_height` | `mdp/terminations.py` | Termination when ball hits ground |
| `ball_far_from_robot` | `mdp/terminations.py` | Termination when ball is too far |
| `hand_body_pos` | `mdp/observations.py` | Hand/wrist positions relative to robot base |

---

## 11. Implementation Order (Recommended)

| Step | Description | Effort |
|---|---|---|
| 1 | Add ball to `RobotSceneCfg` + widen contact sensor | 5 lines |
| 2 | Create `ball_command.py` with `BallCommand` + `BallCommandCfg` | ~80 lines |
| 3 | Wire MDP imports (`__init__.py` files) | 3 lines |
| 4 | Replace `CommandsCfg` with `BallCommandCfg` | 3 lines |
| 5 | Test: ball spawns and falls (physics verified before adding RL) | — |
| 6 | Add `hand_to_ball_distance_exp` reward | ~15 lines |
| 7 | Add ball terminations | ~20 lines |
| 8 | Add ball observations | ~10 lines in config |
| 9 | Full training test with Phase 1 checkpoint resume | — |

---

## 12. What This Does NOT Cover (Future)

- Left-out from this sprint: LATENT-inspired rewards (smoothness, foot tracking, etc.)
  — those are separate, documented in `LATENT_mimic_adjustments.md`
- Ball trajectory prediction (where will the ball be at time t+Δt?)
- Multi-ball scenarios
- Adversarial throws (intentionally hard trajectories)
- Catching with position commands for base (robust footwork)
