"""Unified ball-catching task: motion imitation + catching in ONE environment.

DeepMimic-style combination of the former Phase 1 (mimic) and Phase 2 (RL):
the policy tracks the reference catching motion *while* a physics ball is thrown
so that it arrives exactly when the mocap reaches its catch frame
(:class:`SyncedBallCommand`). The mocap supplies the "how" (whole-body catching
form), RL learns the "when/where" corrections for off-nominal balls — no
checkpoint transfer, no observation placeholders, no forgetting.
"""

from __future__ import annotations

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
from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_MIMIC_CFG as ROBOT_CFG

# single source of truth for the catching motion (auto-discovered drop-in directory,
# see phase1/motions/README.md)
from ..phase1.ball_catch_env_cfg import MOTION_FILE

HAND_BODY_NAMES = ["left_wrist_yaw_link", "right_wrist_yaw_link"]

# [CATCH FRAME] Time (s) into the motion clip at which the mocap catch (ball-hand
# contact) happens. READ THIS OFF YOUR RETARGETED CLIP (scripts/mimic/replay_npz.py)
# and update it when the real catching mocap lands in phase1/motions/.
CATCH_MOTION_TIME = 2.0

VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Scene: robot + physics ball."""

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
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )
    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.05,                  # [BALL SIZE] radius in meters (∅10cm)
            mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            activate_contact_sensors=True,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5,
                dynamic_friction=0.5,
                restitution=0.6,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.3, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 5.0),         # parked/thrown by SyncedBallCommand
        ),
    )


@configclass
class CommandsCfg:
    """Command terms. ORDER MATTERS: the motion command must come first so that on
    env reset the ball command reads the freshly resampled motion time."""

    motion = mdp.MotionCommandCfg(
        asset_name="robot",
        motion_file=MOTION_FILE,
        anchor_body_name="torso_link",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        velocity_range=VELOCITY_RANGE,
        joint_position_range=(-0.1, 0.1),
        body_names=[
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ],
    )

    ball_throw = mdp.SyncedBallCommandCfg(
        asset_name="ball",
        resampling_time_range=(1.0e9, 1.0e9),  # lifecycle driven by the motion phase
        hand_body_names=HAND_BODY_NAMES,
        motion_command_name="motion",
        catch_motion_time=CATCH_MOTION_TIME,
        arrival_jitter=0.05,
        debug_vis=False,
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=UNITREE_G1_29DOF_MIMIC_ACTION_SCALE, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """One observation space — no placeholders. The layout intentionally matches the
    Phase 1/Phase 2 contract (each side's dummy_zeros slots filled with real values),
    so phase checkpoints remain shape-compatible for warm-starting."""

    @configclass
    class PolicyCfg(ObsGroup):
        # -- motion tracking target --
        motion_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        # -- ball state (robot-yaw frame) --
        ball_state = ObsTerm(func=mdp.ball_pos_vel_b)
        ball_relative = ObsTerm(
            func=mdp.ball_to_hands_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=HAND_BODY_NAMES)},
        )
        hand_pos = ObsTerm(
            func=mdp.hand_pos_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=HAND_BODY_NAMES)},
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )
        ball_intercept = ObsTerm(func=mdp.ball_intercept_b, params={"catch_height": 0.35})
        # -- motion tracking --
        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05)
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

    @configclass
    class CriticCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        # -- ball state (clean) --
        ball_state = ObsTerm(func=mdp.ball_pos_vel_b)
        ball_relative = ObsTerm(
            func=mdp.ball_to_hands_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=HAND_BODY_NAMES)},
        )
        hand_pos = ObsTerm(
            func=mdp.hand_pos_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=HAND_BODY_NAMES)},
        )
        ball_intercept = ObsTerm(func=mdp.ball_intercept_b, params={"catch_height": 0.35})
        # -- motion tracking (privileged) --
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        # -- proprioception (clean) --
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 5

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Full DR set (union of the former phases). No reset events for the robot —
    the MotionCommand owns robot resets (reference-state initialization)."""

    # startup
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

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    # ball randomization
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

    # interval — gentler than the pure-mimic task: a push during ball flight makes
    # the catch itself noisy training signal, but robustness still needs pushes
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(2.0, 5.0),
        params={"velocity_range": VELOCITY_RANGE},
    )


@configclass
class RewardsCfg:
    """Tracking rewards (the 'how') + catch rewards (the 'when/where') + regularization.

    No standing-posture crutches (base height / flat orientation / joint deviation):
    the reference motion already defines posture, and crutches would fight the
    crouch/lean/reach the catch needs.
    """

    # -- motion tracking
    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 1.0},
    )
    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 3.14},
    )

    # -- ball catching
    hand_to_ball = RewTerm(
        func=mdp.hand_to_ball_distance_exp,
        weight=2.0,
        params={
            "ball_name": "ball",
            "asset_cfg": SceneEntityCfg("robot", body_names=HAND_BODY_NAMES),
            "std": 0.3,
        },
    )
    catch_success = RewTerm(
        func=mdp.ball_caught_bonus,
        weight=500.0,               # one-time bonus; effective value = weight * dt = 10
        params={"command_name": "ball_throw"},
    )
    ball_secured_hold = RewTerm(
        func=mdp.ball_secured,
        weight=5.0,
        params={"command_name": "ball_throw"},
    )
    ball_height_penalty = RewTerm(
        func=mdp.ball_height_penalty,
        weight=-5.0,
        params={"ball_name": "ball", "min_height": 0.5},
    )

    # -- regularization (mimic-task set)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1e-5)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-1)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
                ],
            ),
            "threshold": 1.0,
        },
    )


@configclass
class TerminationsCfg:
    """Tracking terminations keep the robot honest to the motion; ball terminations
    end the current catch attempt. Ball drops register as failures in the
    MotionCommand's adaptive sampling, which then oversamples the catch window."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # -- motion tracking
    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.25},
    )
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 0.8},
    )
    ee_body_pos = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            # slightly looser than the pure-mimic task (0.25): catching an
            # off-nominal ball legitimately pulls the wrists off the reference
            "threshold": 0.3,
            "body_names": [
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ],
        },
    )
    # -- ball
    ball_caught = DoneTerm(
        func=mdp.ball_caught,
        params={"command_name": "ball_throw"},  # [TERM SUCCESS] confirmed catch
    )
    ball_dropped = DoneTerm(
        func=mdp.ball_below_height,
        params={"ball_name": "ball", "min_height": 0.1},
    )
    ball_missed = DoneTerm(
        func=mdp.ball_far_from_robot,
        params={"ball_name": "ball", "max_distance": 4.0},
    )


@configclass
class CurriculumCfg:
    """Widen the throw distribution, then loosen the tracking leash so the policy
    may deviate from the reference for off-nominal balls. Anchor tracking stays at
    full weight (keeps the robot in place); only relative body tracking is tapered.
    Steps are common_step_counter values (24 per iteration)."""

    ball_throws = CurrTerm(
        func=mdp.ball_throw_curriculum,
        params={
            "command_name": "ball_throw",
            "start_step": 50_000,
            "end_step": 250_000,
            "easy": {
                "throw_distance_range": (1.2, 1.8),
                "throw_lateral_range": (-0.3, 0.3),
                "flight_time_range": (0.5, 0.7),
                "velocity_noise": 0.1,
                "catch_lateral_std": 0.02,
            },
            "final": {
                "throw_distance_range": (1.0, 2.5),
                "throw_lateral_range": (-0.75, 0.75),
                "flight_time_range": (0.3, 0.6),
                "velocity_noise": 0.3,
                "catch_lateral_std": 0.05,
            },
        },
    )

    body_pos_taper = CurrTerm(
        func=mdp.modify_reward_weight_linear,
        params={
            "term_name": "motion_body_pos",
            "start_weight": 1.0,
            "end_weight": 0.4,
            "start_step": 150_000,
            "end_step": 500_000,
        },
    )
    body_ori_taper = CurrTerm(
        func=mdp.modify_reward_weight_linear,
        params={
            "term_name": "motion_body_ori",
            "start_weight": 1.0,
            "end_weight": 0.4,
            "start_step": 150_000,
            "end_step": 500_000,
        },
    )
    body_lin_vel_taper = CurrTerm(
        func=mdp.modify_reward_weight_linear,
        params={
            "term_name": "motion_body_lin_vel",
            "start_weight": 1.0,
            "end_weight": 0.5,
            "start_step": 150_000,
            "end_step": 500_000,
        },
    )
    body_ang_vel_taper = CurrTerm(
        func=mdp.modify_reward_weight_linear,
        params={
            "term_name": "motion_body_ang_vel",
            "start_weight": 1.0,
            "end_weight": 0.5,
            "start_step": 150_000,
            "end_step": 500_000,
        },
    )


@configclass
class RobotEnvCfgUnified(ManagerBasedRLEnvCfg):
    """Unified motion-imitation + ball-catching environment."""

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


class RobotPlayEnvCfgUnified(RobotEnvCfgUnified):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        # evaluate on the final throw distribution and reward weights
        self.curriculum = None
