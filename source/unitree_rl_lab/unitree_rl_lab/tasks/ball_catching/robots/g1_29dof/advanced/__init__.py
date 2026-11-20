import gymnasium as gym

AGENTS = "unitree_rl_lab.tasks.ball_catching.agents.rsl_rl_ppo_cfg"

# Stage B: the chain Advanced-A -> ... -> E fine-tunes the Unified actor without a reference, one change per stage,
# each stage starting from the previous stage's checkpoint (replaces Phase 2 since 2026-11-03)
gym.register(
    id="Unitree-G1-29dof-BallCatch-Advanced-A",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ball_catch_env_cfg:AdvancedAEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.ball_catch_env_cfg:AdvancedAPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{AGENTS}:AdvancedPPORunnerCfg",
    },
)

# A2: A plus penalties against pressing the hands together and arm torques near the limit, from A's checkpoint
gym.register(
    id="Unitree-G1-29dof-BallCatch-Advanced-A2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ball_catch_env_cfg:AdvancedA2EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.ball_catch_env_cfg:AdvancedA2PlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{AGENTS}:AdvancedA2PPORunnerCfg",
    },
)

# B: A2 plus ball rebound at low speeds (PhysX bounce threshold 0.1 m/s), from A2's checkpoint
gym.register(
    id="Unitree-G1-29dof-BallCatch-Advanced-B",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ball_catch_env_cfg:AdvancedBEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.ball_catch_env_cfg:AdvancedBPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{AGENTS}:AdvancedArmAnchorPPORunnerCfg",
    },
)

# C: B plus the real lifecycle (deploy FixStand start pose, several throws per episode), from B's (or A2's) checkpoint
gym.register(
    id="Unitree-G1-29dof-BallCatch-Advanced-C",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ball_catch_env_cfg:AdvancedCEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.ball_catch_env_cfg:AdvancedCPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{AGENTS}:AdvancedArmAnchorPPORunnerCfg",
    },
)

# D: C plus pushes in the ball phase ramped up to full strength (success-gated), from C's checkpoint
gym.register(
    id="Unitree-G1-29dof-BallCatch-Advanced-D",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ball_catch_env_cfg:AdvancedDEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.ball_catch_env_cfg:AdvancedDPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{AGENTS}:AdvancedArmAnchorPPORunnerCfg",
    },
)

# E: D plus wider, realistic throws (aim and arc widened, success-gated; half from the front), from D's checkpoint
gym.register(
    id="Unitree-G1-29dof-BallCatch-Advanced-E",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ball_catch_env_cfg:AdvancedEEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.ball_catch_env_cfg:AdvancedEPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{AGENTS}:AdvancedArmAnchorPPORunnerCfg",
    },
)

# S: E plus losing a ball safely (no episode end at a lost ball, fall penalty, unreachable throws), from E's checkpoint
gym.register(
    id="Unitree-G1-29dof-BallCatch-Advanced-S",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ball_catch_env_cfg:AdvancedSEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.ball_catch_env_cfg:AdvancedSPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{AGENTS}:AdvancedSPPORunnerCfg",
    },
)

# ── chain v2 (v2_env_cfg.py): the stages reordered on the seventh Unified run, each from the previous one's checkpoint
_V2_STAGES = {
    "A": "the reference dropped, A2's penalties, approach x P, throws aimed at the launch, arms resting while waiting",
    "C": "real lifecycle plus what follows a lost ball (no episode end, fall penalty, capture point)",
    "D": "pushes in the ball phase ramped up, plus pushes timed into the flight (D2)",
    "E": "wider and frontal throws, plus 15 % of the throws 0.45-1.0 m to the side",
}
for _stage in _V2_STAGES:
    gym.register(
        id=f"Unitree-G1-29dof-BallCatch-V2-{_stage}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.v2_env_cfg:V2{_stage}EnvCfg",
            "play_env_cfg_entry_point": f"{__name__}.v2_env_cfg:V2{_stage}PlayEnvCfg",
            "rsl_rl_cfg_entry_point": f"{AGENTS}:V2PPORunnerCfg",
        },
    )
