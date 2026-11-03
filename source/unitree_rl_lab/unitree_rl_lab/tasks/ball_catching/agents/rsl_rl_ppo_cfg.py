# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 500
    experiment_name = ""
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # 0.005 until 2026-10-29: it drove the action std to 0.8, where the third Unified run collapsed twice
        # (catches 56 % -> 0 % around iterations 8000 and 9000) and recovered at std 0.64 after each collapse
        entropy_coef=0.002,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


# Fine-tuning from another task's checkpoint is configured on the command line, not here:
# init_noise_std only initializes the std parameter, which --resume overwrites with the
# checkpoint's value. Use: --resume --resume_experiment <exp> --load_run <run> --finetune [--finetune_std 0.2]


@configclass
class AnchoredPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO with the fine-tuning safeguards of ``anchored_ppo.AnchoredPPO`` (all off by default)."""

    class_name: str = "unitree_rl_lab.tasks.ball_catching.agents.anchored_ppo:AnchoredPPO"
    actor_learning_rate: float | None = None
    """Actor learning rate; ``learning_rate`` is then the critic's. None: one rate for both."""
    actor_lr_warmup_iterations: int = 0
    """Iterations after the critic warm-up over which the actor rate ramps up linearly from 0."""
    critic_warmup_iterations: int = 0
    """Iterations at the start in which only the critic learns (the actor keeps the loaded weights)."""
    anchor_kl_coef: float = 0.0
    """Weight of KL(anchor || actor), the anchor being the actor at the first update (the loaded one)."""
    anchor_kl_final_coef: float = 0.0
    anchor_kl_decay_iterations: int = 0
    """Iterations after the critic warm-up over which the anchor weight falls linearly to its final value."""
    max_std: float | None = None
    """Upper bound of the action std, applied after every update."""


@configclass
class AdvancedPPORunnerCfg(BasePPORunnerCfg):
    """Stage B (Advanced tasks): fine-tunes a stage-A actor, started with
    ``--resume --resume_experiment <exp> --load_run <run> --checkpoint model_<N>.pt --finetune``. The std stays the
    loaded one (Unified, sixth run: 0.25-0.39, the wrist pitch and yaw joints 0.57-0.60). Analysis and sources:
    H-ReACT/analysis/2026-11-03_phase2_analyse/ (outside the repository).

    - The critic learns alone for 100 iterations, then the actor rate ramps up to 1e-4 over 100 iterations (10x below
      training from scratch; fixed, with the KL guard of ``AnchoredPPO`` only lowering it).
    - KL to the stage-A actor with weight 1.0, falling to 0.1 over 3000 iterations; it replaces the entropy bonus
      (as in VPT): stays close to stage A where the task does not ask for a change.
    - The std is capped at 0.6 (Andy's cap), just above the largest one of the sixth Unified run, so none is cut at
      the start.
    """

    max_iterations = 4000
    algorithm = AnchoredPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,  # critic
        schedule="adaptive",  # actor: KL guard only, never above actor_learning_rate
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        actor_learning_rate=1.0e-4,
        actor_lr_warmup_iterations=100,
        critic_warmup_iterations=100,
        anchor_kl_coef=1.0,
        anchor_kl_final_coef=0.1,
        anchor_kl_decay_iterations=3000,
        max_std=0.6,
    )
