import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import unitree_rl_lab.tasks.ball_catching.mdp as mdp
from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_MIMIC_ACTION_SCALE
from unitree_rl_lab.tasks.ball_catching.robot_assets import ROBOT_CFG  # FOREARM_COLLIDERS, CONCAVE_PALMS

HAND_BODY_NAMES = ["left_wrist_yaw_link", "right_wrist_yaw_link"]
FEET_BODY_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]
# after the confirmed catch the robot keeps the ball this long (s) before the throw counts as completed
HOLD_AFTER_CATCH_S = 3.0
# forearm + hand links the ball may rest on in the two-arm underhand catch (left first, then right)
ARM_CONTACT_BODY_NAMES = [
    f"{side}_{link}"
    for side in ("left", "right")
    for link in ("elbow_link", "wrist_roll_link", "wrist_pitch_link", "wrist_yaw_link")
]
# a ball squeezed against the torso harder than this (N) does not count as held (catch with the arms, not a chest clamp)
TORSO_BODY_NAME = "torso_link"
MAX_TORSO_FORCE = 50.0
# held only in both palms from below (BallCommandCfg.palm_hold, since 2026-10-29): forearm, wrist or a squeeze between
# them do not count (the third Unified run wedged the ball low between forearms, hands and belly)
PALM_HOLD = True
# aim point of the final throw distribution (root heading frame; height above the ground). A real thrower misses, so
# since 2026-10-29 throws are aimed up to 0.3 m to the side (was 0.1 m; with the launch velocity noise 12 % then
# arrive more than 0.3 m off the centre), a bit shorter or longer and up to chest height: what the arms can reach
# without a step (the baseline caught 99.5 % of throws aimed 0.3 m to the side). Unified aims at the clip instead.
TARGET_FORWARD_RANGE = (0.25, 0.45)
TARGET_LATERAL_RANGE = (-0.3, 0.3)
TARGET_HEIGHT_RANGE = (0.8, 1.15)


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the ball catching scene."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )
    # robot
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    # sensors
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )
    # ball-arm contact for catch detection: the ball is the single sensor body, filtered against the arm
    # links (Isaac Lab reports filtered contacts only one-to-many); history = physics steps per control step
    ball_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        filter_prim_paths_expr=[
            f"{{ENV_REGEX_NS}}/Robot/{name}" for name in ARM_CONTACT_BODY_NAMES + [TORSO_BODY_NAME]
        ],
        history_length=4,
    )
    # ball — physics-simulated rigid object thrown toward the robot (same ball as the mjlab task,
    # interim report ch. 9.2: ∅15 cm, 112 g)
    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.075,                 # [BALL SIZE] radius in meters (∅15cm)
            mass_props=sim_utils.MassPropertiesCfg(
                mass=0.112,               # [BALL MASS] kg
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,    # ball flies under gravity
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,   # ball collides with robot and ground
            ),
            activate_contact_sensors=True,  # needed by the ball_contact sensor
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5,      # [BALL FRICTION] with hands/ground
                dynamic_friction=0.5,
                restitution=0.4,          # [BALL BOUNCINESS] 0=clay, 1=superball; not measured yet
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.3, 0.0),   # [BALL COLOR] orange-red
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 5.0),         # [BALL START POS] far away; BallCommand repositions
        ),
    )


@configclass
class EventCfg:
    """Configuration for events."""

    # startup (robot DR mirrored with Phase 1 / Unified)
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.01, 0.01),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    # ball randomization
    ball_physics = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "static_friction_range": (0.3, 0.8),     # [BALL FRIC RANGE] randomized surface grip
            "dynamic_friction_range": (0.3, 0.8),
            "restitution_range": (0.2, 0.6),          # [BALL BOUNCE RANGE] until a drop test of the real ball
            "num_buckets": 16,
        },
    )
    ball_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",  # must stay startup: BallCommand reads the masses once for the drag-compensated aim
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "mass_distribution_params": (-0.015, 0.015),  # [BALL MASS RANGE] ±15g around 112g default
            "operation": "add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(4.0, 8.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    ball_throw = mdp.BallCommandCfg(
        asset_name="ball",
        resampling_time_range=(1.0e9, 1.0e9),  # single throw per episode; new throw only on env reset
        hand_body_names=HAND_BODY_NAMES,
        contact_sensor_name="ball_contact",
        torso_body_name=TORSO_BODY_NAME,
        max_torso_force=MAX_TORSO_FORCE,
        palm_hold=PALM_HOLD,
        # wait as in front of a real thrower (final range, the curriculum starts at 0.5-1.5 s; the baseline trained on
        # 0.5-1.5 s caught 28 % after 6-9 s), then keep the ball (it walked off with it)
        throw_delay_range=(0.5, 8.0),
        throw_delay_log_uniform=True,
        target_forward_range=TARGET_FORWARD_RANGE,
        target_lateral_range=TARGET_LATERAL_RANGE,
        target_height_range=TARGET_HEIGHT_RANGE,
        hold_after_catch_s=HOLD_AFTER_CATCH_S,
        debug_vis=False,
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # NOTE: must match Phase 1 exactly, otherwise transferred policies command
    # different joint targets for the same network output.
    JointPositionAction = mdp.ClampedJointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=UNITREE_G1_29DOF_MIMIC_ACTION_SCALE, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group: only what the real robot has (actor contract, identical in the
        Unified task so that ``--finetune`` can move the actor between them)."""

        # -- ball as the robot perceives it via VICON (yaw frame; noise, latency and dropouts come from the
        #    ball command, so no ObsTerm noise here). Zeros while not visible. --
        ball_state = ObsTerm(func=mdp.ball_perceived_pos_vel_b, params={"command_name": "ball_throw"})
        ball_visible = ObsTerm(func=mdp.ball_visible, params={"command_name": "ball_throw"})
        # -- ballistic interception prediction at belly height, from the perceived state --
        ball_intercept = ObsTerm(
            func=mdp.ball_perceived_intercept_b, params={"command_name": "ball_throw", "catch_height": 0.15}
        )
        # -- proprioception --
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group (privileged; same slot layout in all three tasks)."""

        # -- motion placeholders (zero in Phase 2, real in Phase 1 / Unified) --
        command = ObsTerm(func=mdp.dummy_zeros, params={"dim": 58})
        # -- ball state (robot-yaw frame, clean, no noise) --
        ball_state = ObsTerm(func=mdp.ball_pos_vel_b)
        # -- whether the actor currently sees the ball --
        ball_visible = ObsTerm(func=mdp.ball_visible, params={"command_name": "ball_throw"})
        ball_relative = ObsTerm(
            func=mdp.ball_to_hands_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=HAND_BODY_NAMES)},
        )
        hand_pos = ObsTerm(
            func=mdp.hand_pos_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=HAND_BODY_NAMES)},
        )
        # -- ballistic interception prediction at belly height, clean (zeros in Phase 1) --
        ball_intercept = ObsTerm(func=mdp.ball_intercept_b, params={"catch_height": 0.15})
        # -- motion placeholders --
        motion_anchor_pos_b = ObsTerm(func=mdp.dummy_zeros, params={"dim": 3})
        motion_anchor_ori_b = ObsTerm(func=mdp.dummy_zeros, params={"dim": 6})
        body_pos = ObsTerm(func=mdp.dummy_zeros, params={"dim": 42})
        body_ori = ObsTerm(func=mdp.dummy_zeros, params={"dim": 84})
        # -- proprioception (clean) --
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 5

    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- ball catching (mjlab ch. 9.4.1: small proximity < larger contact < largest retention reward;
    #    all zero before the launch)
    ball_reach = RewTerm(
        func=mdp.ball_interception_exp,
        weight=1.0,
        params={
            "command_name": "ball_throw",
            "asset_cfg": SceneEntityCfg("robot", body_names=HAND_BODY_NAMES),
            "std_dist": 0.3,        # [REACH SHAPING] hands midpoint to ball; gradient reaches ~0.8m
        },
    )
    ball_interception = RewTerm(
        func=mdp.ball_interception_exp,
        weight=2.0,
        params={
            "command_name": "ball_throw",
            "asset_cfg": SceneEntityCfg("robot", body_names=HAND_BODY_NAMES),
            "std_dist": 0.2,
            "std_vel": 1.0,         # [SOFT CATCH] eq. 9.1: hands move with the ball instead of blocking it
        },
    )
    ball_arm_contact = RewTerm(
        func=mdp.ball_arm_contact,
        weight=3.0,                 # 0.06/step with both arms in contact
        params={"command_name": "ball_throw"},
    )
    catch_success = RewTerm(
        func=mdp.ball_caught_bonus,
        weight=500.0,               # once per throw at the confirmation; effective value = weight * dt = 10
        params={"command_name": "ball_throw"},
    )
    ball_secured_hold = RewTerm(
        func=mdp.ball_secured,
        weight=5.0,                 # 0.1/step while held; ball_caught is a truncation (see TerminationsCfg)
        params={"command_name": "ball_throw"},
    )
    ball_height_penalty = RewTerm(
        func=mdp.ball_height_penalty,
        weight=-5.0,
        params={
            "ball_name": "ball",
            "min_height": 0.5,       # [CATCH ZONE] penalize ball dropping below 0.5m
        },
    )
    # -- staying up: without these the per-step penalties (mostly action_rate from the exploration noise) outweigh
    #    alive, falling costs nothing and the policy learns to fall before the throw (pilot run 2026-10-22)
    alive = RewTerm(func=mdp.is_alive, weight=1.0)  # 0.02/step; discounted at most ~2 < catch bonus 10
    termination_penalty = RewTerm(
        func=mdp.is_terminated_term,
        weight=-200.0,              # -4 per fall
        params={"term_keys": ["base_height", "bad_orientation"]},  # falls only: catch and drop are not penalized
    )

    # -- standing still while waiting for the throw and while holding the confirmed catch (the baseline drifted
    #    0.16 m/s and turned 15°/s while waiting, slid its feet and walked off with the ball). Final weights; the
    #    curriculum ramps them in from 0 (see CurriculumCfg.stand_still_*_ramp)
    stand_still_root = RewTerm(func=mdp.stand_still_root, weight=-10.0, params={"command_name": "ball_throw"})
    stand_still_feet = RewTerm(
        func=mdp.stand_still_feet,
        weight=-5.0,
        params={
            "command_name": "ball_throw",
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_NAMES, preserve_order=True),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_NAMES, preserve_order=True),
        },
    )

    # -- regularization
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    # the 2026-10-27 baseline pressed the ankle-roll targets 2-4 rad past the joint stops (feet on their edges,
    # torque saturated while waiting); the clamp removes the lever, these two terms the incentive
    # targets beyond the joint limits change nothing since the clamp (ClampedJointPositionAction): keep them inside
    joint_target_excess = RewTerm(func=mdp.joint_target_excess, weight=-1.0)
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1e-5)  # as Phase 1 / Unified; energy ignores static torque
    energy = RewTerm(func=mdp.energy, weight=-2e-5)

    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*",
                ],
            )
        },
    )
    joint_deviation_waists = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["waist.*"],
            )
        },
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )

    # -- posture
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    base_height = RewTerm(func=mdp.base_height_l2, weight=-10, params={"target_height": 0.76})

    # -- feet
    # TEMP: removed gait reward — depends on base_velocity command.
    # Re-add or adapt for catching footwork.
    # gait = RewTerm(...)
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    feet_flat = RewTerm(
        func=mdp.feet_flat,
        weight=-2.0,  # sin² of the tilt per foot in firm contact: 15° -> 0.067; baseline 14-21°, target < 5°
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_NAMES, preserve_order=True),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_NAMES, preserve_order=True),
        },
    )
    # TEMP: removed foot_clearance_reward — only in locomotion MDP, not ball_catching.
    # Add back or replace with catching-specific foot reward.
    # feet_clearance = RewTerm(func=mdp.foot_clearance_reward, ...)

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1,
            # exclude feet, forearms/hands and torso — in the two-arm underhand catch the ball rests on the
            # forearms and against the belly; that contact is the goal, not a fault
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    r"^(?!.*_ankle_roll_link$)(?!.*_elbow_link$)(?!.*_wrist_(roll|pitch|yaw)_link$)(?!torso_link$).+$"
                ],
            ),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # Single-throw episode contract: every throw ends in exactly one of
    # caught / dropped / missed / robot fell / time out.
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})
    # The completed catch (confirmed, then held for HOLD_AFTER_CATCH_S) is a truncation (time_out=True): PPO
    # bootstraps the value of that state instead of cutting it to 0. As a plain termination the catch forfeited the
    # hold rewards that follow, so holding the ball while breaking the secured streak just before secure_steps paid
    # more than the bonus (run 2026-10-22). A drop during the hold phase is a failure (ball_dropped).
    ball_caught = DoneTerm(
        func=mdp.ball_caught,
        time_out=True,
        params={"command_name": "ball_throw"},  # [TERM SUCCESS] confirmed catch, held for the hold phase
    )
    ball_dropped = DoneTerm(
        func=mdp.ball_below_height,
        params={"ball_name": "ball", "min_height": 0.1},  # [TERM DROP] ball z below 10cm = dead
    )
    ball_missed = DoneTerm(
        func=mdp.ball_far_from_robot,
        params={"ball_name": "ball", "max_distance": 6.0},  # [TERM MISS] beyond max release distance (5m) + margin
    )


@configclass
class CurriculumCfg:
    """Curriculum: start with easy, slow throws and strong posture crutches, then
    widen the throw distribution and taper the crutches so the policy becomes free
    to crouch/lean/reach for the ball.

    Steps are ``common_step_counter`` values (one per env-step; at 24 steps/iteration
    a full 30k-iteration run is 720k steps). Tune alongside run length.
    """

    ball_throws = CurrTerm(
        func=mdp.ball_throw_curriculum,
        params={
            "command_name": "ball_throw",
            "start_step": 50_000,
            "end_step": 250_000,
            "easy": {  # short lobs from the front straight into the arms, clean perception; mjlab: 2 m / ±15° first
                "throw_distance_range": (2.0, 2.5),
                "throw_direction_range": (-math.radians(15.0), math.radians(15.0)),
                "target_forward_range": (0.3, 0.4),
                "target_lateral_range": (-0.03, 0.03),
                "target_height_range": (0.85, 0.95),
                "apex_height_range": (1.85, 1.95),
                "velocity_noise": 0.0,
                "secure_steps": 25,                # mjlab raises the required hold time over training
                "obs_delay_steps_range": (1, 1),
                "obs_dropout_prob": 0.0,
                "hide_in_catch_zone_prob": 0.0,
                "throw_delay_range": (0.5, 1.5),   # short waits first
            },
            "final": {  # = CommandsCfg.ball_throw (BallCommandCfg defaults, wider aim, long waits)
                "throw_distance_range": (2.0, 5.0),
                "throw_direction_range": (-math.radians(75.0), math.radians(75.0)),
                "target_forward_range": TARGET_FORWARD_RANGE,
                "target_lateral_range": TARGET_LATERAL_RANGE,
                "target_height_range": TARGET_HEIGHT_RANGE,
                "apex_height_range": (1.75, 2.1),
                "velocity_noise": 0.1,
                "secure_steps": 50,
                "obs_delay_steps_range": (1, 5),
                "obs_dropout_prob": 0.05,
                "hide_in_catch_zone_prob": 0.5,
                "throw_delay_range": (0.5, 8.0),
            },
        },
    )

    # Standing still is only asked for once the robot can stand: a flailing robot moves fast, the quadratic penalties
    # then cost more than a fall, and the policy learned to fall within 7 steps (pilot 2026-10-28, 250 iterations).
    # Ramped together with the waits of ball_throws.
    stand_still_root_ramp = CurrTerm(
        func=mdp.modify_reward_weight_linear,
        params={
            "term_name": "stand_still_root",
            "start_weight": 0.0,
            "end_weight": -10.0,
            "start_step": 50_000,
            "end_step": 250_000,
        },
    )
    stand_still_feet_ramp = CurrTerm(
        func=mdp.modify_reward_weight_linear,
        params={
            "term_name": "stand_still_feet",
            "start_weight": 0.0,
            "end_weight": -5.0,
            "start_step": 50_000,
            "end_step": 250_000,
        },
    )

    base_height_taper = CurrTerm(
        func=mdp.modify_reward_weight_linear,
        params={
            "term_name": "base_height",
            "start_weight": -10.0,
            "end_weight": -2.5,
            "start_step": 100_000,
            "end_step": 400_000,
        },
    )
    flat_orientation_taper = CurrTerm(
        func=mdp.modify_reward_weight_linear,
        params={
            "term_name": "flat_orientation_l2",
            "start_weight": -5.0,
            "end_weight": -1.0,
            "start_step": 100_000,
            "end_step": 400_000,
        },
    )
    arm_deviation_taper = CurrTerm(
        func=mdp.modify_reward_weight_linear,
        params={
            "term_name": "joint_deviation_arms",
            "start_weight": -0.1,
            "end_weight": -0.02,
            "start_step": 100_000,
            "end_step": 400_000,
        },
    )


@configclass
class RobotEnvCfgPhase2(ManagerBasedRLEnvCfg):
    """Configuration for the ball catching Phase 2 (autonomous catching) environment."""

    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        self.scene.contact_forces.update_period = self.sim.dt


class RobotPlayEnvCfgPhase2(RobotEnvCfgPhase2):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        # evaluate on the final throw distribution and reward weights, not the
        # step-0 curriculum stage
        self.curriculum = None
