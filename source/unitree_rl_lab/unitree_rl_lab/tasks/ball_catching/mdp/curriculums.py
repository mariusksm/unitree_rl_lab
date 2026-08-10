from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _progress(env: ManagerBasedRLEnv, start_step: int, end_step: int) -> float:
    """Linear training progress in [0, 1] based on the common step counter."""
    if end_step <= start_step:
        return 1.0
    return min(max((env.common_step_counter - start_step) / (end_step - start_step), 0.0), 1.0)


def _lerp(a, b, p: float):
    """Linear interpolation for floats and (nested) tuples/lists of floats."""
    if isinstance(a, (tuple, list)):
        return type(a)(ai + (bi - ai) * p for ai, bi in zip(a, b))
    return a + (b - a) * p


def modify_reward_weight_linear(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    start_weight: float,
    end_weight: float,
    start_step: int,
    end_step: int,
) -> float:
    """Linearly anneal a reward term's weight from ``start_weight`` to ``end_weight``.

    Used to taper posture crutches (base height, flat orientation, arm deviation)
    once the policy has learned to stand, so it becomes free to crouch/lean/reach
    for the ball. Returns the current weight for logging.
    """
    p = _progress(env, start_step, end_step)
    weight = start_weight + (end_weight - start_weight) * p
    env.reward_manager.get_term_cfg(term_name).weight = weight
    return weight


def ball_throw_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    easy: dict,
    final: dict,
    start_step: int,
    end_step: int,
    command_name: str = "ball_throw",
) -> float:
    """Widen the BallCommand throw distribution from ``easy`` to ``final`` values.

    ``easy``/``final`` map BallCommandCfg field names (e.g. ``flight_time_range``,
    ``throw_lateral_range``, ``velocity_noise``) to values; each field is linearly
    interpolated over training progress. New throws sample from the updated cfg
    immediately. Returns the progress in [0, 1] for logging.
    """
    p = _progress(env, start_step, end_step)
    cfg = env.command_manager.get_term(command_name).cfg
    for key, easy_val in easy.items():
        setattr(cfg, key, _lerp(easy_val, final[key], p))
    return p
