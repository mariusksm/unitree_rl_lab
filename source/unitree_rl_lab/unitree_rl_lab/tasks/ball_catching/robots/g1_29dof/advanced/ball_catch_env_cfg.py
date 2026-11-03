"""Advanced tasks (stage B): catching without a motion reference, fine-tuned from the Unified actor (stage A).

Since 2026-11-03 they replace Phase 2 (trained from scratch with additive rewards, it found loophole after loophole:
ankle stops, foot edges, ball clamped to the chest with 236 N, deep squat). The chain Advanced-A -> B -> C -> D -> E
adds one change per stage, each stage a subclass of the previous one and trained from the previous stage's checkpoint;
every stage is measured (evaluation suite, Sim2Sim) against the stage before and against the unchanged Unified actor.

- **A:** the Unified definition of holding and posture without the reference, under Unified-like conditions (throws
  aimed where the clips catch, the same pushes, randomization and perception), fine-tuned with the safeguards of
  ``AnchoredPPO`` (critic warm-up, small actor rate, KL anchor to the Unified actor). The stage checks that the
  fine-tuning machinery and the reference-free reward keep what Unified learned.
- **B:** ball rebound at low speeds (PhysX bounces only above its threshold of 0.5 m/s, B lowers it to 0.1 m/s; in
  MuJoCo a springy ball halves the catches of the sixth Unified run).
- **C:** the real lifecycle: start in the FixStand pose of the deploy FSM, several throws per episode, the ball taken
  out of the hands after the hold.
- **D:** pushes in the ball phase ramped from 20 % up to full strength by a success-gated curriculum.
- Planned: E, wider, realistic throws. F (sketch): lunges for throws beyond the arms' reach.

Analysis and sources: H-ReACT/analysis/2026-11-03_phase2_analyse/ (outside the repository).

The actor observations, the action term and the critic slot layout must stay identical to Unified (``--finetune``
moves the actor between them).
"""

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
from unitree_rl_lab.tasks.ball_catching.motion_data import G1_HOLD_POINT
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
TORSO_BODY_NAME = "torso_link"

# ── what "held" means, as in Unified (since 2026-11-01): both palms carry the ball, at the hold point, and nothing else
#    carries it. Without a reference the hold point is the median of the clips' hold points (motion_data.G1_HOLD_POINT,
#    0.26 m in front of and 0.235 m above the pelvis) with a larger radius than Unified's per-clip 0.15 m: the clips'
#    own hold points spread by 0.09 m around it (median; 90 % within 0.19 m). ──
PALM_HOLD = True
HOLD_RADIUS = 0.20
FORBIDDEN_BODY_NAMES = [
    f"{side}_{link}" for side in ("left", "right") for link in ("shoulder_pitch_link", "shoulder_yaw_link")
]
MAX_TORSO_FORCE_CATCHING = 5.0
MAX_TORSO_FORCE_HOLDING = 10.0
# arm links whose pressure on the torso is punished (reward self_contact, sensor arm_torso_contact), as in Unified
ARM_SELF_CONTACT_BODY_NAMES = [
    f"{side}_{link}"
    for side in ("left", "right")
    for link in ("shoulder_yaw_link", "elbow_link", "wrist_roll_link", "wrist_pitch_link", "wrist_yaw_link")
]

# ── posture factor P (mdp.posture_quality, reward `stand`, factor of the hold reward and the catch bonus) as in
#    Unified, against a fixed standing height instead of the reference pelvis: the clips stand at 0.783 m, the robot
#    at 0.78 m; 3 cm lower are free, deeper falls off with 5 cm ──
POSTURE = {"tilt_std": 0.35, "height_tolerance": 0.03, "height_std": 0.05, "ref_height": 0.78}

# ── Advanced-A throws: where Unified's throws arrive. Unified aims at the clips' palm midpoint + 0.05 m at the recorded
#    catch, ±0.05 m per axis (catch_point_jitter); relative to the pelvis that is 0.17-0.36 m in front, -0.05..+0.12 m
#    to the side and 0.90-1.23 m above the ground (p5-p95 of the 84 clips with a usable catch; script aim_points.py in
#    the analysis folder). Distances, directions, apex, noise and perception are the final ones of both tasks
#    (BallCommandCfg defaults). Wider throws (Phase 2 aimed up to ±0.3 m to the side) come in stage E. ──
TARGET_FORWARD_RANGE = (0.15, 0.40)
TARGET_LATERAL_RANGE = (-0.10, 0.17)
TARGET_HEIGHT_RANGE = (0.85, 1.28)
# wait before the throw: log-uniform 0.5-8 s (median 2 s), as Phase 2; Unified waits up to its 6 s lead-in plus the
# time to the clip's launch
THROW_DELAY_RANGE = (0.5, 8.0)

# ── pushes as in Unified: full strength while waiting, this share of it from the launch to the end of the hold (a push
#    moves the robot, not the free ball; full pushes in the ball phase rewarded clamping the ball to the chest). Stage
#    D ramps the share up. ──
PUSH_BALL_PHASE_SCALE = 0.2
PUSH_VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Ground, robot (ball-task asset), ball and the contact sensors of the hold definition."""

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
    # ball contact for the hold definition: the ball is the single sensor body, filtered against arm links, torso and
    # the forbidden upper arms (Isaac Lab reports filtered contacts only one-to-many); history = physics steps per step
    ball_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        filter_prim_paths_expr=[
            f"{{ENV_REGEX_NS}}/Robot/{name}"
            for name in ARM_CONTACT_BODY_NAMES + [TORSO_BODY_NAME] + FORBIDDEN_BODY_NAMES
        ],
        history_length=4,
    )
    # arms pressing on the torso (reward self_contact)
    arm_torso_contact = ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{TORSO_BODY_NAME}",
        filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/Robot/{name}" for name in ARM_SELF_CONTACT_BODY_NAMES],
        history_length=4,
    )
    # ball: ∅15 cm, 112 g (as the mjlab task, interim report ch. 9.2)
    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.075,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.112),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            activate_contact_sensors=True,  # needed by the ball_contact sensor
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5,
                dynamic_friction=0.5,
                restitution=0.4,  # not measured yet (spec Section 11, item 2)
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.3, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 5.0)),  # BallCommand repositions it
    )


@configclass
class EventCfg:
    """Randomization as in Unified; the robot starts in the default pose (Phase 2 reset)."""

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
    ball_physics = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "static_friction_range": (0.3, 0.8),
            "dynamic_friction_range": (0.3, 0.8),
            "restitution_range": (0.2, 0.6),  # until a drop test of the real ball
            "num_buckets": 16,
        },
    )
    ball_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",  # must stay startup: BallCommand reads the masses once for the drag-compensated aim
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "mass_distribution_params": (-0.015, 0.015),
            "operation": "add",
        },
    )

    # reset: standing in the default pose somewhere around the origin, facing anywhere (the throw is sampled in the
    # robot's heading frame)
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
        params={"position_range": (1.0, 1.0), "velocity_range": (-1.0, 1.0)},
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity_phased,
        mode="interval",
        interval_range_s=(2.0, 5.0),
        params={
            "velocity_range": PUSH_VELOCITY_RANGE,
            "command_name": "ball_throw",
            "ball_phase_scale": PUSH_BALL_PHASE_SCALE,
        },
    )


@configclass
class CommandsCfg:
    """One throw per episode (wait, flight, catch, 3 s hold), the hold definition of Unified."""

    ball_throw = mdp.BallCommandCfg(
        asset_name="ball",
        resampling_time_range=(1.0e9, 1.0e9),  # single throw per episode; new throw only on env reset
        hand_body_names=HAND_BODY_NAMES,
        contact_sensor_name="ball_contact",
        torso_body_name=TORSO_BODY_NAME,
        palm_hold=PALM_HOLD,
        hold_point=G1_HOLD_POINT,
        hold_radius=HOLD_RADIUS,
        hold_sigma=0.10,
        contact_map=True,
        forbidden_body_names=FORBIDDEN_BODY_NAMES,
        max_torso_force_catching=MAX_TORSO_FORCE_CATCHING,
        max_torso_force_holding=MAX_TORSO_FORCE_HOLDING,
        throw_delay_range=THROW_DELAY_RANGE,
        throw_delay_log_uniform=True,
        target_forward_range=TARGET_FORWARD_RANGE,
        target_lateral_range=TARGET_LATERAL_RANGE,
        target_height_range=TARGET_HEIGHT_RANGE,
        hold_after_catch_s=HOLD_AFTER_CATCH_S,
        debug_vis=False,
    )


@configclass
class ActionsCfg:
    """Must match Unified exactly, otherwise a transferred actor commands other joint targets."""

    JointPositionAction = mdp.ClampedJointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=UNITREE_G1_29DOF_MIMIC_ACTION_SCALE, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Actor: only what the real robot has (contract shared with Unified). Critic: the shared slot layout."""

    @configclass
    class PolicyCfg(ObsGroup):
        # ball as the robot perceives it via VICON (yaw frame; noise, latency and dropouts come from the ball command)
        ball_state = ObsTerm(func=mdp.ball_perceived_pos_vel_b, params={"command_name": "ball_throw"})
        ball_visible = ObsTerm(func=mdp.ball_visible, params={"command_name": "ball_throw"})
        # ballistic interception prediction at belly height, from the perceived state
        ball_intercept = ObsTerm(
            func=mdp.ball_perceived_intercept_b, params={"command_name": "ball_throw", "catch_height": 0.15}
        )
        # proprioception
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
        # motion placeholders (zero here, real in Phase 1 / Unified)
        command = ObsTerm(func=mdp.dummy_zeros, params={"dim": 58})
        # ball state (robot yaw frame, clean)
        ball_state = ObsTerm(func=mdp.ball_pos_vel_b)
        ball_visible = ObsTerm(func=mdp.ball_visible, params={"command_name": "ball_throw"})
        ball_relative = ObsTerm(
            func=mdp.ball_to_hands_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=HAND_BODY_NAMES)},
        )
        hand_pos = ObsTerm(
            func=mdp.hand_pos_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=HAND_BODY_NAMES)},
        )
        ball_intercept = ObsTerm(func=mdp.ball_intercept_b, params={"catch_height": 0.15})
        # motion placeholders
        motion_anchor_pos_b = ObsTerm(func=mdp.dummy_zeros, params={"dim": 3})
        motion_anchor_ori_b = ObsTerm(func=mdp.dummy_zeros, params={"dim": 6})
        body_pos = ObsTerm(func=mdp.dummy_zeros, params={"dim": 42})
        body_ori = ObsTerm(func=mdp.dummy_zeros, params={"dim": 84})
        # proprioception (clean)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)
        # phase of the throw (since 2026-11-01)
        catch_phase = ObsTerm(func=mdp.catch_phase, params={"command_name": "ball_throw"})

        def __post_init__(self):
            self.history_length = 5

    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Unified's reward without the tracking terms (the reference is gone), plus standing still while waiting and
    holding (without the reference nothing else keeps the robot from drifting or walking off with the ball).

    No posture crutches of the old Phase 2 (`base_height`, `flat_orientation`, joint deviation toward the default
    pose): P covers torso and pelvis, and the arm pose of the Unified actor is kept by the KL anchor of
    ``AnchoredPPO``, not by a fixed pose that would compete with it. If the arms drift while waiting once the anchor
    weight has fallen, a ready pose from the clips is the next step (spec Section 11, item 12).
    """

    # -- posture factor P (also a factor of the hold reward and the catch bonus)
    stand = RewTerm(func=mdp.posture_quality, weight=5.0, params=POSTURE)

    # -- ball catching (all zero before the launch); approach until the confirmed catch, then the hold reward alone
    ball_reach = RewTerm(
        func=mdp.ball_interception_exp,
        weight=1.0,
        params={
            "command_name": "ball_throw",
            "asset_cfg": SceneEntityCfg("robot", body_names=HAND_BODY_NAMES),
            "std_dist": 0.3,
            "until_confirmed": True,
        },
    )
    ball_interception = RewTerm(
        func=mdp.ball_interception_exp,
        weight=2.0,
        params={
            "command_name": "ball_throw",
            "asset_cfg": SceneEntityCfg("robot", body_names=HAND_BODY_NAMES),
            "std_dist": 0.2,
            "std_vel": 1.0,
            "until_confirmed": True,
        },
    )
    hold_quality = RewTerm(func=mdp.hold_quality, weight=8.0, params={"command_name": "ball_throw", "posture": POSTURE})
    catch_success = RewTerm(
        func=mdp.ball_caught_bonus,
        weight=500.0,  # once per throw at the confirmation, times P; effective value = weight * dt = 10
        params={"command_name": "ball_throw", "posture": POSTURE},
    )
    ball_body_contact = RewTerm(func=mdp.ball_body_contact, weight=-1.0, params={"command_name": "ball_throw"})
    ball_height_penalty = RewTerm(
        func=mdp.ball_height_penalty, weight=-5.0, params={"ball_name": "ball", "min_height": 0.5}
    )
    self_contact = RewTerm(
        func=mdp.self_contact,
        weight=-5.0,
        params={"sensor_cfg": SceneEntityCfg("arm_torso_contact"), "threshold": 10.0},
    )

    # -- standing still while waiting for the throw and while holding the confirmed catch (Phase 2 weights, without its
    #    ramp: the loaded actor already stands)
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

    # -- regularization (Unified's set)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1e-5)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-1)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits, weight=-10.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])}
    )
    joint_target_excess = RewTerm(func=mdp.joint_target_excess, weight=-1.0)
    feet_flat = RewTerm(
        func=mdp.feet_flat,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_NAMES, preserve_order=True),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_NAMES, preserve_order=True),
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            # exclude feet, forearms/hands and torso (ball contact of the two-arm underhand catch)
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    r"^(?!.*_ankle_roll_link$)(?!.*_elbow_link$)(?!.*_wrist_(roll|pitch|yaw)_link$)(?!torso_link$).+$"
                ],
            ),
            "threshold": 1.0,
        },
    )


@configclass
class TerminationsCfg:
    """Every throw ends in exactly one of: caught (after the hold phase), dropped, missed, robot fell, time out."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})
    # the completed catch is a truncation (time_out=True): PPO bootstraps its value; as a plain termination the policy
    # learned to postpone the confirmation (spec pitfall 16)
    ball_caught = DoneTerm(func=mdp.ball_caught, time_out=True, params={"command_name": "ball_throw"})
    ball_dropped = DoneTerm(func=mdp.ball_below_height, params={"ball_name": "ball", "min_height": 0.1})
    ball_missed = DoneTerm(func=mdp.ball_far_from_robot, params={"ball_name": "ball", "max_distance": 6.0})


@configclass
class AdvancedAEnvCfg(ManagerBasedRLEnvCfg):
    """Advanced-A: the Unified definition of holding and posture without the reference, Unified-like conditions."""

    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum = None  # the loaded actor already catches these throws

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0  # longest regular episode ≈ 8 s wait + 1 s flight + 1.5 s confirmation + 3 s
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.contact_forces.update_period = self.sim.dt


class AdvancedAPlayEnvCfg(AdvancedAEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32


# ── Advanced-B: ball rebound at low speeds. PhysX applies restitution only to contacts whose normal relative speed
#    exceeds the scene's bounce threshold (Isaac Lab default 0.5 m/s), so a ball that lands softly in the palms or is
#    jostled while held never bounces; MuJoCo's contacts spring at every speed (Sim2Sim of the sixth Unified run: 44 %
#    caught with a springy ball, e = 0.36, against 98 % with a damped one, e = 0.14). B lowers the threshold to
#    0.1 m/s. The ball-robot restitution stays the average of the randomized ball (0.2-0.6) and arm link (0-0.5)
#    values, 0.1-0.55: the startup randomization gives every shape its own material with combine mode average
#    (measured 2026-11-03 with balls dropped onto a fixed robot; a combine mode authored on the ball is overwritten).
#    Ground contacts keep restitution 0 (terrain material, multiply). Replace the ranges once the real ball is measured
#    (spec Section 11, item 2).
#    Measured before training B: the sixth Unified run's actor catches 81 % in B against 80 % in A (base scenario,
#    ~4000 episodes each), so PhysX's low-speed restitution does not reproduce the MuJoCo failure with the springy ball:
#    MuJoCo's soft contact stores energy while the ball is squeezed and releases it, a rigid contact with restitution
#    does not. If the real ball turns out springy, compliant contacts for the ball
#    (RigidBodyMaterialCfg.compliant_contact_stiffness / _damping) are the next candidate. ──
BOUNCE_THRESHOLD_VELOCITY = 0.1


@configclass
class AdvancedBEnvCfg(AdvancedAEnvCfg):
    """Advanced-B: Advanced-A plus ball rebound at low speeds (bounce threshold 0.1 m/s instead of 0.5 m/s).

    Fine-tuned from Advanced-A's final checkpoint (``--finetune``: the A actor is the new anchor)."""

    def __post_init__(self):
        super().__post_init__()
        self.sim.physx.bounce_threshold_velocity = BOUNCE_THRESHOLD_VELOCITY


class AdvancedBPlayEnvCfg(AdvancedBEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32


# ── Advanced-C: the real lifecycle. On the robot the deploy FSM hands over to the policy from its FixStand state, and
#    the policy then runs throw after throw: wait, catch, hold until a person takes the ball out of the hands, wait
#    while the thrower walks back, the next throw. A and B start every episode in the default pose and end it after
#    one throw, so the policy never learned to let go and get ready again. C starts in the FixStand pose of the deploy
#    config plus noise and re-throws after every completed catch instead of ending the episode (the ball is moved
#    out of the hands to the next release point); a drop, a miss or a fall still ends it. ──
# FixStand pose of deploy/robots/g1_29dof/config/config.yaml (qs), in which the policy takes over: straighter legs than
# the default pose (knees 0.3 instead of 0.669 rad), upper arms down, elbows bent 0.97 rad; joints not listed are 0.
# Change it together with the deploy config. The default pose stays the offset of actions and observations.
DEPLOY_STAND_POSE = {
    ".*_hip_pitch_joint": -0.1,
    ".*_knee_joint": 0.3,
    ".*_ankle_pitch_joint": -0.2,
    "left_shoulder_roll_joint": 0.25,
    "right_shoulder_roll_joint": -0.25,
    ".*_elbow_joint": 0.97,
    "left_wrist_roll_joint": 0.15,
    "right_wrist_roll_joint": -0.15,
}
# pelvis height of that pose with the soles on the ground: 0.784 m (forward kinematics of the G1 MJCF, which gives
# 0.757 m for the default pose, spawned at 0.76 m), plus the same 3 mm
DEPLOY_STAND_PELVIS_HEIGHT = 0.787
# the hand-over is not exact: FixStand holds the pose with lower PD gains than the policy's, so gravity sags the joints
START_POSE_NOISE = 0.05
# wait before a throw that follows a completed catch: the person takes the ball, walks back 2-5 m and throws again
# (log-uniform like the first wait, median 4 s)
RETHROW_DELAY_RANGE = (2.0, 8.0)
# several throws per episode: a throw takes about 2-4 s of waiting + 1 s flight + 1 s to the confirmation + 3 s hold
EPISODE_LENGTH_C_S = 30.0


@configclass
class AdvancedCEnvCfg(AdvancedBEnvCfg):
    """Advanced-C: Advanced-B plus the real lifecycle: start in the deploy FixStand pose, several throws per episode,
    the ball taken out of the hands after the hold.

    Fine-tuned from Advanced-B's final checkpoint (or A's, if B is not trained)."""

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = EPISODE_LENGTH_C_S
        self.commands.ball_throw.rethrow_after_catch = True
        self.commands.ball_throw.rethrow_delay_range = RETHROW_DELAY_RANGE
        # the completed catch arms the next throw; time-outs (truncations) and failures end the episode
        self.terminations.ball_caught = None
        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_to_pose,
            mode="reset",
            params={"joint_pos": DEPLOY_STAND_POSE, "position_noise": START_POSE_NOISE},
        )
        lift = DEPLOY_STAND_PELVIS_HEIGHT - ROBOT_CFG.init_state.pos[2]
        self.events.reset_base.params["pose_range"]["z"] = (lift, lift)


class AdvancedCPlayEnvCfg(AdvancedCEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32


# ── Advanced-D: pushes in the ball phase ramped up. A to C push at 20 % of the waiting strength from the launch to the
#    end of the hold (under the old hold definition full pushes in the ball phase rewarded clamping the ball to the
#    chest). With the clean grip defined, D raises the share toward full strength by a success-gated curriculum: each
#    push in the ball phase draws its share between 20 % and 20 % + level · 80 %, the level uniform in [0, difficulty],
#    and the difficulty rises while at least 70 % of the finished throws succeed and falls below 50 %. Phillip's
#    g1-catch raises at 80 %; the stochastic training catch rate of Advanced-A sat near 77 %, so 80 % would stall the
#    curriculum. ≈ 24 calls per iteration at 1e-4 each: the full ramp takes at least ≈ 420 iterations. Measured with the
#    suite's `push` scenario (pushes every 1-2 s; the sixth Unified run caught 65 % of them in A). ──
PUSH_BALL_PHASE_SCALE_FINAL = 1.0
SUCCESS_GATE = {"up": 0.7, "down": 0.5, "step": 1e-4, "smoothing": 0.01}


@configclass
class AdvancedDCurriculumCfg:
    push_difficulty = CurrTerm(func=mdp.success_gated_difficulty, params={"target": "pushes", **SUCCESS_GATE})


@configclass
class AdvancedDEnvCfg(AdvancedCEnvCfg):
    """Advanced-D: Advanced-C plus pushes in the ball phase ramped from 20 % up to full strength (success-gated).

    Fine-tuned from Advanced-C's final checkpoint."""

    def __post_init__(self):
        super().__post_init__()
        self.curriculum = AdvancedDCurriculumCfg()
        self.events.push_robot.params["ball_phase_scale_final"] = PUSH_BALL_PHASE_SCALE_FINAL
        self.events.push_robot.params["difficulty"] = 0.0  # set by the curriculum


class AdvancedDPlayEnvCfg(AdvancedDEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.curriculum = None
        self.events.push_robot.params["difficulty"] = 1.0  # final level: shares of 20-100 % in the ball phase
