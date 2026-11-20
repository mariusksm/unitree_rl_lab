"""Chain v2 (branch ``chain-v2``, built 2026-11-20): the Advanced stages reordered on a seventh Unified run.

The first chain (Advanced-A -> A2 -> B -> C -> D -> E -> S, ``ball_catch_env_cfg.py``) fine-tuned the sixth Unified
run and added each lesson as one more stage. Chain v2 starts from a Unified run that already has the fixes that belong
to it (A2's penalties against pressed hands and arm torques near the limit, the ball reported before the throw in half
of the cycles, a lost ball that ends the episode only 3 s later and stays visible on the floor in half of the cycles),
and puts every later lesson into the stage where it belongs:

- **V2-A:** the reference dropped, holding and posture as defined in A, plus A2's penalties, the approach rewards
  times the posture factor (S), every throw aimed at the robot at the launch (S), the ball before the throw reported
  in half of the throws (constant: the seventh Unified run learned it) and the arms resting in the pose of the human
  recordings while waiting (new, ``mdp.arm_rest_pose_exp``).
- **V2-C:** the real lifecycle (C) together with what follows a lost ball (S): a loss ends only its throw, the lost ball
  stays visible in half of the throws (as the seventh Unified run learned it), a fall costs -200, the capture point over
  the feet pays while standing.
- **V2-D:** pushes in the ball phase ramped up (D), plus pushes timed into the flight for 30 % of the throws (D2,
  ``mdp.push_during_flight``).
- **V2-E:** wider throws and frontal throws (E), plus 15 % of the throws aimed 0.45-1.0 m to the side (S).

Every stage is fine-tuned from the previous one's final checkpoint (V2-A from the seventh Unified run) with
``V2PPORunnerCfg`` (anchor 1.0 -> 0.1 within 500 iterations, masked anchor as Advanced-S: the observation group
``anchor_mask`` of ``SObservationsCfg``). B (ball rebound) stays a placeholder until the real ball
is measured; its bounce threshold is inherited as in C to S. F (lunges) comes later, if real throws need it.
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import unitree_rl_lab.tasks.ball_catching.mdp as mdp

from .ball_catch_env_cfg import (
    CAPTURE_POINT_STD,
    CAPTURE_POINT_WEIGHT,
    FALL_PENALTY_WEIGHT,
    FEET_BODY_NAMES,
    HAND_BODY_NAMES,
    HIDE_LOST_BALL_PROB,
    POSTURE,
    PUSH_VELOCITY_RANGE,
    SHOW_PARKED_BALL_PROB,
    SUCCESS_GATE,
    A2RewardsCfg,
    AdvancedBEnvCfg,
    AdvancedECurriculumCfg,
    EventCfg,
    SObservationsCfg,
    apply_ball_phase_push_ramp,
    apply_loss_evaluation,
    apply_loss_lifecycle,
    apply_real_lifecycle,
    apply_unreachable_throws,
    apply_wide_throws,
)

# ── V2-A: arm pose of the human catchers at the start of the 108 underhand clips (first frame, mean over the clips,
#    mirrored to be symmetric; std over the clips 0.03-0.08 rad per joint): upper arms slightly out and not raised,
#    elbows bent about 62°. The Advanced stages wait with the forearms raised in front of the chest instead. ──
ARM_REST_POSE = {
    ".*_shoulder_pitch_joint": 0.072,
    "left_shoulder_roll_joint": 0.408,
    "right_shoulder_roll_joint": -0.408,
    "left_shoulder_yaw_joint": -0.376,
    "right_shoulder_yaw_joint": 0.376,
    ".*_elbow_joint": 1.089,
    "left_wrist_roll_joint": -0.078,
    "right_wrist_roll_joint": 0.078,
    ".*_wrist_pitch_joint": 0.071,
    "left_wrist_yaw_joint": -0.413,
    "right_wrist_yaw_joint": 0.413,
}
ARM_REST_WEIGHT = 1.0  # per second, against 5 for `stand`: a nudge, low priority (Marius, 2026-11-20)
ARM_REST_STD = 0.5  # rad, root mean square over the 14 arm joints: 0.5 rad off pays exp(-1) = 0.37; wide enough to
#   pull from E-like raised forearms (with E's actor in V2-A the term paid almost nothing at 0.3)

# ── V2-D: pushes timed into the flight (D2) ──
FLIGHT_PUSH_SHARE = 0.3
FLIGHT_PUSH_WINDOW = (0.2, 0.9)  # fractions of the planned flight time after the launch


@configclass
class V2ARewardsCfg(A2RewardsCfg):
    """A2's rewards with the approach terms times the posture factor, the ball-height penalty only while the ball is in
    play (both from S) and the arms resting in the recorded pose while waiting."""

    ball_reach = RewTerm(
        func=mdp.ball_interception_exp,
        weight=1.0,
        params={
            "command_name": "ball_throw",
            "asset_cfg": SceneEntityCfg("robot", body_names=HAND_BODY_NAMES),
            "std_dist": 0.3,
            "until_confirmed": True,
            "posture": POSTURE,
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
            "posture": POSTURE,
        },
    )
    ball_height_penalty = RewTerm(
        func=mdp.ball_height_penalty,
        weight=-5.0,
        params={"ball_name": "ball", "min_height": 0.5, "command_name": "ball_throw"},
    )
    arm_rest = RewTerm(
        func=mdp.arm_rest_pose_exp,
        weight=ARM_REST_WEIGHT,
        params={"target": ARM_REST_POSE, "std": ARM_REST_STD, "command_name": "ball_throw"},
    )


@configclass
class V2AEnvCfg(AdvancedBEnvCfg):
    """V2-A: the reference dropped (as Advanced-A, with A2's penalties and B's bounce threshold), approach rewards times
    P, throws aimed at the robot at the launch, the ball before the throw reported in half of the throws, the arms
    resting while waiting. Fine-tuned from the seventh Unified run."""

    rewards: V2ARewardsCfg = V2ARewardsCfg()
    observations: SObservationsCfg = SObservationsCfg()  # with the group anchor_mask of the masked anchor

    def __post_init__(self):
        super().__post_init__()
        cmd = self.commands.ball_throw
        cmd.aim_at_launch = True
        cmd.show_parked_ball_prob = SHOW_PARKED_BALL_PROB


class V2APlayEnvCfg(V2AEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.ball_throw.show_parked_ball_prob = 0.0  # --show_ball_before_throw reports it


@configclass
class V2CRewardsCfg(V2ARewardsCfg):
    """V2-A's rewards plus S's fall penalty and capture point."""

    fall = RewTerm(
        func=mdp.is_terminated_term,
        weight=FALL_PENALTY_WEIGHT,
        params={"term_keys": ["bad_orientation", "base_height"]},
    )
    capture_point = RewTerm(
        func=mdp.capture_point_exp,
        weight=CAPTURE_POINT_WEIGHT,
        params={
            "std": CAPTURE_POINT_STD,
            "command_name": "ball_throw",
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_NAMES, preserve_order=True),
        },
    )


@configclass
class V2CEnvCfg(V2AEnvCfg):
    """V2-C: V2-A plus the real lifecycle (C) and what follows a lost ball (S): a loss ends only its throw, the lost
    ball visible in half of the throws (as in the seventh Unified run), a fall costs -200, the capture point pays.
    Fine-tuned from V2-A's final checkpoint."""

    rewards: V2CRewardsCfg = V2CRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        apply_real_lifecycle(self)
        apply_loss_lifecycle(self)
        self.commands.ball_throw.hide_lost_ball_prob = HIDE_LOST_BALL_PROB


class V2CPlayEnvCfg(V2CEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        apply_loss_evaluation(self)


@configclass
class V2DEventCfg(EventCfg):
    """Events with the pushes timed into the flight (D2)."""

    push_in_flight = EventTerm(
        func=mdp.push_during_flight,
        mode="interval",
        interval_range_s=(0.02, 0.02),  # every control step; the event keeps its own plan per throw
        params={"velocity_range": PUSH_VELOCITY_RANGE, "share": FLIGHT_PUSH_SHARE, "window": FLIGHT_PUSH_WINDOW},
    )


@configclass
class V2DCurriculumCfg:
    push_difficulty = CurrTerm(func=mdp.success_gated_difficulty, params={"target": "pushes", **SUCCESS_GATE})


@configclass
class V2DEnvCfg(V2CEnvCfg):
    """V2-D: V2-C plus pushes in the ball phase ramped up (D) and pushes timed into the flight of 30 % of the throws
    (D2), both at the level of D's success-gated curriculum. Fine-tuned from V2-C's final checkpoint."""

    events: V2DEventCfg = V2DEventCfg()

    def __post_init__(self):
        super().__post_init__()
        apply_ball_phase_push_ramp(self)
        self.curriculum = V2DCurriculumCfg()


class V2DPlayEnvCfg(V2DEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        apply_loss_evaluation(self)
        self.events.push_robot.params["difficulty"] = 1.0  # final level
        self.events.push_in_flight = None  # evaluated like the earlier stages (pushes every 2-5 s only)


@configclass
class V2EEnvCfg(V2DEnvCfg):
    """V2-E: V2-D plus wider throws and frontal throws (E, success-gated) and 15 % of the throws aimed 0.45-1.0 m to the
    side (S). Fine-tuned from V2-D's final checkpoint."""

    def __post_init__(self):
        super().__post_init__()
        self.curriculum = AdvancedECurriculumCfg()
        self.events.push_robot.params["difficulty"] = 1.0  # V2-D's final level, fixed
        apply_wide_throws(self)
        apply_unreachable_throws(self)


class V2EPlayEnvCfg(V2EEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        apply_loss_evaluation(self)
        self.commands.ball_throw.difficulty = 1.0  # final level: throws from the easy to the hard ranges
        self.events.push_in_flight = None  # evaluated like the earlier stages (pushes every 2-5 s only)
