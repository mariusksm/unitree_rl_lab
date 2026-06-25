import gymnasium as gym

gym.register(
    id="Unitree-G1-29dof-BallCatch-Phase2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ball_catch_env_cfg:RobotEnvCfgPhase2",
        "play_env_cfg_entry_point": f"{__name__}.ball_catch_env_cfg:RobotPlayEnvCfgPhase2",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.ball_catching.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
