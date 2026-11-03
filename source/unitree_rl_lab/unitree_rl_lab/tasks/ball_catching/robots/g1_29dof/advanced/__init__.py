import gymnasium as gym

# Stage B: the chain Advanced-A -> ... -> E fine-tunes the Unified actor without a reference, one change per stage,
# each stage starting from the previous stage's checkpoint (replaces Phase 2 since 2026-11-03)
gym.register(
    id="Unitree-G1-29dof-BallCatch-Advanced-A",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ball_catch_env_cfg:AdvancedAEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.ball_catch_env_cfg:AdvancedAPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.ball_catching.agents.rsl_rl_ppo_cfg:AdvancedPPORunnerCfg",
    },
)
