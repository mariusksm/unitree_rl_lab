"""Unified ball-catching task: motion imitation + catching in ONE environment.

DeepMimic-style combination of the former Phase 1 (mimic) and Phase 2 (RL):
the policy tracks a reference catching clip *while* a physics ball is thrown so that
it arrives where and when the human caught the ball in that take (catch time and
catch point from the take's VICON recording, :class:`SyncedBallCommand`). The mocap
supplies the "how" (whole-body catching form), RL learns the "when/where" corrections
for off-nominal balls. Each clip runs as a wait-and-hold cycle: standing in its first
pose before the throw (lead-in), the catch, then the catch pose held while the robot
keeps the ball (hold phase).

The actor sees the ball and proprioception only (the reference is critic-only), so it
must react to the ball as on the real robot. Its observations equal those of the Advanced
tasks, which makes this task stage A: its actor is fine-tuned there (stage B, Advanced-A..E).
"""

from __future__ import annotations

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
from unitree_rl_lab.tasks.ball_catching.robot_assets import (  # FOREARM_COLLIDERS, CONCAVE_PALMS
    ARM_JOINT_NAMES,
    ARM_TORQUE_THRESHOLD,
    ARM_TORQUE_WEIGHT,
    HAND_CONTACT_THRESHOLD,
    HAND_CONTACT_WEIGHT,
    OTHER_ARM_LINKS,
    ROBOT_CFG,
    hand_contact_sensor,
)

from unitree_rl_lab.tasks.ball_catching.motion_data import (
    G1_HOLD_POINT,
    HOLD_ABOVE_PALMS,
    TEMP_CLIP_LIFT,
    ball_file_for,
    catch_data,
)

# single source of truth for the reference clips (auto-discovered, see motion_data.py)
from ..phase1.ball_catch_env_cfg import BALL_DATA_DIR, MOTION_FILES

# the clip is cut this long after the recorded catch and that pose held: in 33 of 86 clips the person already puts the
# ball down by the clip end, a reference that lets go of the ball does not go with the hold phase
CATCH_SETTLE_S = 0.3

# Catch time, flight time, catch point and hold point per clip from its recording (numpy, a few seconds at import); in
# the config, so that the motion command can cut the clips after the catch and params/env.yaml records what was used.
# Since 2026-11-01 the ball is aimed at the hold point of the reference: its palm midpoint at the catch moment, raised
# by HOLD_ABOVE_PALMS (0.05 m), where it is also held (hold point, relative to the pelvis, CATCH_SETTLE_S after the
# catch). Until then it was aimed at the wrist midpoint + 0.08 m, 0.08 m behind the palms at the heel of the hands
# (median); before 2026-10-29 at the recorded ball position, which is in the human's scale while the clip is in the
# robot's (the robot then clamped the ball against its upper chest and head, run of 2026-10-28).
_RECORDED = [f for f in MOTION_FILES if ball_file_for(f, BALL_DATA_DIR)]
_DATA = catch_data(  # TEMP_CLIP_LIFT: temporary height fix, as for the clips
    _RECORDED, BALL_DATA_DIR, lift=TEMP_CLIP_LIFT, aim_at_palms=HOLD_ABOVE_PALMS, hold_settle_s=CATCH_SETTLE_S
)
# Unified trains only on clips with a usable catch: without a ball the reference performs its catching motion into the
# void, which an actor that sees only the ball cannot follow (58 % of the tracking terminations of the second run).
# Without any usable recording (fixed mode, e.g. the old overhand clip in phase1/motions/) all clips are used.
if any(d["valid"] for d in _DATA):
    CATCH_MOTION_FILES = [f for f, d in zip(_RECORDED, _DATA) if d["valid"]]
    CATCH_DATA = [d for d in _DATA if d["valid"]]
else:
    CATCH_MOTION_FILES = MOTION_FILES
    CATCH_DATA = catch_data(MOTION_FILES, None)
# For the same reason no cycle starts after its launch: latest start shortly before it (margin > arrival jitter 0.05 s)
LAUNCH_MARGIN_S = 0.1
LATEST_START_S = [max(0.0, d["catch_time"] - d["flight_time"] - LAUNCH_MARGIN_S) if d["valid"] else None
                  for d in CATCH_DATA]

# Fixed mode only (no usable recordings, e.g. the old single overhand clip in phase1/motions/): catch time (s)
# into every clip. With recordings, every clip brings its own catch time and catch point.
FALLBACK_CATCH_MOTION_TIME = 2.0

# ── wait-and-hold cycle ──
# lead-in: the first frame is held this long before the clip, a share of the cycles starts inside it (waiting for the
# throw as in front of a real thrower; the 2026-10-27 Phase 2 baseline caught 28 % after 6-9 s of waiting)
START_HOLD_S = 6.0
START_HOLD_PROB = 0.35
HOLD_FROM_S = [d["catch_time"] + CATCH_SETTLE_S if d["valid"] else None for d in CATCH_DATA]
# after the confirmed catch the robot keeps the ball this long before the throw counts as completed
HOLD_AFTER_CATCH_S = 3.0
# the held pose must outlast the confirmation (secure_steps, up to ~1.5 s after the arrival) plus the hold phase,
# otherwise the clip ends first: the robot is reset onto the next cycle and the catch never completes
END_HOLD_S = HOLD_AFTER_CATCH_S + 2.0

HAND_BODY_NAMES = ["left_wrist_yaw_link", "right_wrist_yaw_link"]
FEET_BODY_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]
# forearm + hand links the ball may rest on in the two-arm underhand catch (left first, then right)
ARM_CONTACT_BODY_NAMES = [
    f"{side}_{link}"
    for side in ("left", "right")
    for link in ("elbow_link", "wrist_roll_link", "wrist_pitch_link", "wrist_yaw_link")
]
TORSO_BODY_NAME = "torso_link"
# held only in both palms from below (BallCommandCfg.palm_hold, since 2026-10-29): forearm, wrist or a squeeze between
# them do not count (the third Unified run wedged the ball low between forearms, hands and belly)
PALM_HOLD = True

# ── what "held" means (since 2026-11-01; concept in H-ReACT/analysis/2026-11-01_festhalten_haltung/konzept.md, outside
#    the repo): both palms carry the ball, where the reference holds it, and nothing else carries it. The fifth run met
#    the old definition (catch box up to 0.5 m above the pelvis, torso force up to 50 N) by pressing the ball against
#    chest and shoulder with 20 N, 0.06 m in front of and 0.38 m above the pelvis; the reference holds it 0.26 m in
#    front of and 0.24 m above the pelvis in its palms. ──
# the ball counts as held only within this distance (m) of the clip's hold point (curriculum: 0.25 m early on)
HOLD_RADIUS = 0.15
# contact map (BallCommandCfg.contact_map): while catching only the palms count and the torso may be touched with up to
# 5 N; after the confirmation forearm, wrist and belly (up to 10 N) may support the ball as long as both palms carry it;
# the upper arms, shoulders and a torso pressed with more than 10 N are forbidden
FORBIDDEN_BODY_NAMES = [
    f"{side}_{link}" for side in ("left", "right") for link in ("shoulder_pitch_link", "shoulder_yaw_link")
]
MAX_TORSO_FORCE_CATCHING = 5.0
MAX_TORSO_FORCE_HOLDING = 10.0
# arm links whose pressure on the torso is punished (reward self_contact, sensor arm_torso_contact): the fifth run
# pressed its arms against the torso with more than 140 N to stiffen its grip
ARM_SELF_CONTACT_BODY_NAMES = [
    f"{side}_{link}"
    for side in ("left", "right")
    for link in ("shoulder_yaw_link", "elbow_link", "wrist_roll_link", "wrist_pitch_link", "wrist_yaw_link")
]

# ── correct posture: upright and at standing height (posture factor P, mdp.posture_quality). The height part is
#    one-sided against the reference pelvis of the current frame: 3 cm lower are free (knees up to ~35-40°), deeper
#    falls off with 5 cm (fifth run: 0.71 m while holding -> 0.43, 0.67 m while waiting -> 0.07). P is a reward of
#    its own (`stand`) and multiplies the hold reward and the catch bonus, so a hold in a bad posture is worth little
#    instead of costing a little: five runs traded posture for a more robust grip whenever posture was only added. ──
POSTURE = {"tilt_std": 0.35, "height_tolerance": 0.03, "height_std": 0.05, "ref_command_name": "motion"}

# ── pushes: full strength while waiting for the throw, this share of it from the launch to the end of the hold. A
#    push changes the velocity of the whole robot, the free ball does not follow it, so full pushes rewarded clamping
#    the ball against the chest. Unified learns the clean form; stage B (Advanced-D) ramps them up again. ──
PUSH_BALL_PHASE_SCALE = 0.2

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
    # ball contact for the catch check (ball = single sensor body, filtered against the arm links, the torso and the
    # forbidden upper arms; the torso stays column 8, the evaluation reads it there)
    ball_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        filter_prim_paths_expr=[
            f"{{ENV_REGEX_NS}}/Robot/{name}"
            for name in ARM_CONTACT_BODY_NAMES + [TORSO_BODY_NAME] + FORBIDDEN_BODY_NAMES
        ],
        history_length=4,
    )
    # arms pressing on the torso (reward self_contact): torso = sensor body, filtered against the arm links; history =
    # the physics steps of one control step
    arm_torso_contact = ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{TORSO_BODY_NAME}",
        filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/Robot/{name}" for name in ARM_SELF_CONTACT_BODY_NAMES],
        history_length=4,
    )
    # a hand pressing on the other hand or forearm (rewards hand_contact_*, since 2026-11-03; the hand-hand contact is
    # counted once)
    hand_contact_left = hand_contact_sensor("left", "right", OTHER_ARM_LINKS)
    hand_contact_right = hand_contact_sensor("right", "left", OTHER_ARM_LINKS[1:])
    # same ball as the Advanced tasks / the mjlab task (interim report ch. 9.2: ∅15 cm, 112 g)
    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.075,                 # [BALL SIZE] radius in meters (∅15cm)
            mass_props=sim_utils.MassPropertiesCfg(mass=0.112),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            activate_contact_sensors=True,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5,
                dynamic_friction=0.5,
                restitution=0.4,          # not measured yet
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
        motion_files=CATCH_MOTION_FILES,
        clip_lift=TEMP_CLIP_LIFT,  # temporary height fix of the current dataset (motion_data.py)
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
        start_hold_s=START_HOLD_S,
        start_hold_prob=START_HOLD_PROB,
        hold_from_s=HOLD_FROM_S,
        end_hold_s=END_HOLD_S,
        latest_start_s=LATEST_START_S,
        # a dropped or missed ball counts where its cycle started (before the throw), not where the ball falls (after
        # the launch, where no cycle may start): clips where the ball falls get more cycles from that start point
        start_bin_failure_terms=["ball_dropped", "ball_missed"],
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
        contact_sensor_name="ball_contact",
        torso_body_name=TORSO_BODY_NAME,
        palm_hold=PALM_HOLD,
        # what "held" means (see above): hold point per clip from CATCH_DATA (the fixed one only in fixed mode)
        hold_point=G1_HOLD_POINT,
        hold_radius=HOLD_RADIUS,
        hold_sigma=0.10,
        contact_map=True,
        forbidden_body_names=FORBIDDEN_BODY_NAMES,
        max_torso_force_catching=MAX_TORSO_FORCE_CATCHING,
        max_torso_force_holding=MAX_TORSO_FORCE_HOLDING,
        motion_command_name="motion",
        catch_data=CATCH_DATA,
        catch_motion_time=FALLBACK_CATCH_MOTION_TIME,
        arrival_jitter=0.05,
        catch_point_jitter=(0.05, 0.05, 0.05),
        hold_after_catch_s=HOLD_AFTER_CATCH_S,
        debug_vis=False,
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = mdp.ClampedJointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=UNITREE_G1_29DOF_MIMIC_ACTION_SCALE, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """The actor gets only what the real robot has: the same terms as the Advanced tasks (actor contract), so
    this task's actor can be fine-tuned there (stage A -> stage B). The reference motion goes to the critic
    only; otherwise the policy could take the catch timing from the clip clock, which no human thrower
    follows. The critic keeps the slot layout shared by all three tasks."""

    @configclass
    class PolicyCfg(ObsGroup):
        # -- ball as the robot perceives it via VICON (BallCommand) --
        ball_state = ObsTerm(func=mdp.ball_perceived_pos_vel_b, params={"command_name": "ball_throw"})
        ball_visible = ObsTerm(func=mdp.ball_visible, params={"command_name": "ball_throw"})
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

    @configclass
    class CriticCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        # -- ball state (clean) --
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
        ball_intercept = ObsTerm(func=mdp.ball_intercept_b, params={"catch_height": 0.15})
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
        # -- phase of the throw (since 2026-11-01): the reward depends on it, the history cannot show it
        catch_phase = ObsTerm(func=mdp.catch_phase, params={"command_name": "ball_throw"})

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
            "restitution_range": (0.2, 0.6),
            "num_buckets": 16,
        },
    )
    ball_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",  # must stay startup: the ball command reads the masses once for the drag-compensated aim
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "mass_distribution_params": (-0.015, 0.015),
            "operation": "add",
        },
    )

    # interval — gentler than the pure-mimic task (every 2-5 s), and from the launch to the end of the hold only at
    # PUSH_BALL_PHASE_SCALE of the strength (see there)
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity_phased,
        mode="interval",
        interval_range_s=(2.0, 5.0),
        params={
            "velocity_range": VELOCITY_RANGE,
            "command_name": "ball_throw",
            "ball_phase_scale": PUSH_BALL_PHASE_SCALE,
        },
    )


@configclass
class RewardsCfg:
    """Tracking rewards (the 'how') + catch rewards (the 'when/where') + posture + regularization.

    Since 2026-11-01 (concept: H-ReACT/analysis/2026-11-01_festhalten_haltung/konzept.md, outside the repo):
    - tracking keeps its full weight (no taper: the throws aim at the reference hands, nothing calls for leaving it);
    - approach terms (reach, interception) only until the catch is confirmed;
    - one graded hold reward instead of the flat contact and secured terms, times the posture factor P;
    - P (`stand`: torso upright, pelvis at standing height) instead of `torso_upright`;
    - small penalties for forbidden ball contacts and for arms pressing on the torso.
    No other standing-posture crutches (joint deviation etc.): the reference defines the posture.
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
    # the hands on their own (since 2026-11-01): in the mean over 14 bodies they barely count (the fifth run waited with
    # its hands at its face for about 0.15 per step); the reference hands start moving only 0.34 s after the launch
    # (median), when the robot sees the ball already, so they can be followed while waiting
    motion_hand_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.10, "body_names": HAND_BODY_NAMES},
    )

    # -- correct posture: torso upright, pelvis at standing height (P, see POSTURE). Replaces torso_upright (tilt only,
    # since 2026-10-30), under which the fifth run kept its torso upright but sank into a squat (knees 69°)
    stand = RewTerm(func=mdp.posture_quality, weight=5.0, params=POSTURE)

    # -- ball catching (all zero before the launch). Approach until the confirmed catch, then the hold reward alone
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
    # how well the ball is held (both palms, at the clip's hold point, contact map; BallCommand.hold_quality) times P.
    # Replaces ball_arm_contact (3) and ball_secured_hold (5), which paid the same for any hold inside loose limits
    hold_quality = RewTerm(
        func=mdp.hold_quality,
        weight=8.0,
        params={"command_name": "ball_throw", "posture": POSTURE},
    )
    catch_success = RewTerm(
        func=mdp.ball_caught_bonus,
        weight=500.0,               # once per throw at the confirmation, times P; effective value = weight * dt = 10
        params={"command_name": "ball_throw", "posture": POSTURE},
    )
    # forbidden ball contacts (upper arm, shoulder, torso above 10 N): small on purpose, the lost hold reward is the
    # main cost, so keeping the ball always pays more than dropping it (harsh penalties teach avoiding the ball)
    ball_body_contact = RewTerm(func=mdp.ball_body_contact, weight=-1.0, params={"command_name": "ball_throw"})
    ball_height_penalty = RewTerm(
        func=mdp.ball_height_penalty,
        weight=-5.0,
        params={"ball_name": "ball", "min_height": 0.5},
    )

    # -- arms free of the body: share of the physics steps with an arm pressing on the torso above 10 N (light touches
    # of the reference, whose right upper arm overlaps the torso in 20 % of the frames, stay below)
    self_contact = RewTerm(
        func=mdp.self_contact,
        weight=-5.0,
        params={"sensor_cfg": SceneEntityCfg("arm_torso_contact"), "threshold": 10.0},
    )
    # -- joint load (since 2026-11-03, for the next run): Advanced-A, fine-tuned from the sixth run, pressed its hands
    # against each other with 20-30 N and drove the left wrist yaw (5 Nm) to its torque limit; the sixth run did the
    # same to a lesser degree. A hand pressing on the other arm above 5 N and arm torques above 60 % of the limit cost.
    hand_contact_left = RewTerm(
        func=mdp.self_contact,
        weight=HAND_CONTACT_WEIGHT,
        params={"sensor_cfg": SceneEntityCfg("hand_contact_left"), "threshold": HAND_CONTACT_THRESHOLD},
    )
    hand_contact_right = RewTerm(
        func=mdp.self_contact,
        weight=HAND_CONTACT_WEIGHT,
        params={"sensor_cfg": SceneEntityCfg("hand_contact_right"), "threshold": HAND_CONTACT_THRESHOLD},
    )
    arm_torque_limit = RewTerm(
        func=mdp.torque_limit_excess,
        weight=ARM_TORQUE_WEIGHT,
        params={"threshold": ARM_TORQUE_THRESHOLD, "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)},
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
    # targets beyond the joint limits change nothing since the clamp (ClampedJointPositionAction): keep them inside
    joint_target_excess = RewTerm(func=mdp.joint_target_excess, weight=-1.0)
    # tracking the feet's orientation alone let the left foot stand on its edge (17° in 84 % of the contact steps at
    # iteration 7000 of the run of 2026-10-28); same term and weight as the Advanced tasks
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
    # The completed catch (confirmed, then held for HOLD_AFTER_CATCH_S) is a truncation (time_out=True), as in
    # Phase 2: it must not forfeit the tracking and hold rewards that follow, and the adaptive sampler counts only
    # real terminations as failures.
    ball_caught = DoneTerm(
        func=mdp.ball_caught,
        time_out=True,
        params={"command_name": "ball_throw"},  # [TERM SUCCESS] confirmed catch, held for the hold phase
    )
    ball_dropped = DoneTerm(
        func=mdp.ball_below_height,
        params={"ball_name": "ball", "min_height": 0.1},
    )
    ball_missed = DoneTerm(
        func=mdp.ball_far_from_robot,
        params={"ball_name": "ball", "max_distance": 6.0},  # beyond max release distance (5m) + margin
    )


@configclass
class CurriculumCfg:
    """Widen the throw distribution and narrow the hold radius. Steps are common_step_counter values (24 per
    iteration).

    No tracking taper since 2026-11-01: it loosened the body tracking to 40-50 % so that the policy could leave the
    reference for off-nominal balls, but the throws aim at the reference hands since 2026-10-29, and the fifth run sank
    into a squat once the body tracking weighed less than about 0.75 (iteration 11500 -> end)."""

    ball_throws = CurrTerm(
        func=mdp.ball_throw_curriculum,
        params={
            "command_name": "ball_throw",
            "start_step": 50_000,
            "end_step": 250_000,
            "easy": {
                "throw_distance_range": (2.0, 2.5),
                "throw_direction_range": (-math.radians(15.0), math.radians(15.0)),
                "target_lateral_range": (-0.03, 0.03),
                "target_height_range": (0.85, 0.95),
                "apex_height_range": (1.85, 1.95),
                "velocity_noise": 0.0,
                "secure_steps": 25,
                "hold_radius": 0.25,               # the stricter hold definition starts generous
                "obs_delay_steps_range": (1, 1),
                "obs_dropout_prob": 0.0,
                "hide_in_catch_zone_prob": 0.0,
            },
            "final": {  # = BallCommandCfg defaults and CommandsCfg.ball_throw
                "throw_distance_range": (2.0, 5.0),
                "throw_direction_range": (-math.radians(75.0), math.radians(75.0)),
                "target_lateral_range": (-0.1, 0.1),
                "target_height_range": (0.8, 1.05),
                "apex_height_range": (1.75, 2.1),
                "velocity_noise": 0.1,
                "secure_steps": 50,
                "hold_radius": HOLD_RADIUS,
                "obs_delay_steps_range": (1, 5),
                "obs_dropout_prob": 0.05,
                "hide_in_catch_zone_prob": 0.5,
            },
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
