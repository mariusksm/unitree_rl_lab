from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import CurriculumTermCfg


def _progress(env: ManagerBasedRLEnv, start_step: int, end_step: int) -> float:
    """Linear training progress in [0, 1] based on the common step counter."""
    if end_step <= start_step:
        return 1.0
    return min(max((env.common_step_counter - start_step) / (end_step - start_step), 0.0), 1.0)


def _lerp(a, b, p: float):
    """Linear interpolation for floats, ints (rounded) and (nested) tuples/lists of them."""
    if isinstance(a, (tuple, list)):
        return type(a)(_lerp(ai, bi, p) for ai, bi in zip(a, b))
    if isinstance(a, int) and isinstance(b, int):
        # int fields (e.g. secure_steps) are compared with ==/>= against step counters
        return int(round(a + (b - a) * p))
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

    ``easy``/``final`` map BallCommandCfg field names (e.g. ``apex_height_range``,
    ``throw_direction_range``, ``secure_steps``) to values; each field is linearly
    interpolated over training progress. New throws sample from the updated cfg
    immediately. Returns the progress in [0, 1] for logging.
    """
    p = _progress(env, start_step, end_step)
    cfg = env.command_manager.get_term(command_name).cfg
    for key, easy_val in easy.items():
        setattr(cfg, key, _lerp(easy_val, final[key], p))
    return p


class success_gated_difficulty(ManagerTermBase):
    """Success-gated curriculum (as in Phillip's g1-catch; Advanced-D and -E): a difficulty in [0, 1] that rises by
    ``step`` per call while the success rate of the finished throws is at least ``up`` and falls by ``step`` while it is
    below ``down``. The success rate is an exponential moving average over the calls (``smoothing`` per call). The
    consumer draws a level uniform in [0, difficulty] per sample, so easier samples stay in the mix (against
    forgetting).

    Called at every reset, before the terminations and the command are reset: a completed catch counts as a success, a
    dropped or missed ball or a fall as a failure, a throw cut off by the time-out not at all. With one throw per
    episode the ``ball_caught`` termination marks the success; with several (``throws_completed`` of the command) each
    completed throw counts at the next call after it happened, not only when its episode ends: episodes end early by a
    failure but late after successes, so counting at the episode end would see the failures first.

    ``target``: ``"pushes"`` sets ``difficulty`` of the event term ``push_robot`` (pushes in the ball phase, D),
    ``"throws"`` the ball command's ``difficulty`` (throw distribution, E). ``start`` is the difficulty at the start;
    a resumed training starts there again and climbs back while the policy succeeds. Logged as
    ``Curriculum/<term>/difficulty`` and ``/success_rate``.
    """

    FAILURES = ("ball_dropped", "ball_missed", "base_height", "bad_orientation")

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.difficulty = float(cfg.params.get("start", 0.0))
        self.success_rate = 0.0
        self._counted = torch.zeros(env.num_envs, device=env.device)  # completed throws already counted, per env
        self._apply(env, cfg.params["target"], cfg.params.get("command_name", "ball_throw"))

    def _apply(self, env: ManagerBasedRLEnv, target: str, command_name: str):
        if target == "pushes":
            env.event_manager.get_term_cfg("push_robot").params["difficulty"] = self.difficulty
        elif target == "throws":
            env.command_manager.get_term(command_name).cfg.difficulty = self.difficulty
        else:
            raise ValueError(f"success_gated_difficulty: unknown target '{target}' (pushes, throws)")

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        target: str,
        up: float = 0.7,
        down: float = 0.5,
        step: float = 1e-4,
        smoothing: float = 0.01,
        start: float = 0.0,
        command_name: str = "ball_throw",
    ) -> dict[str, float]:
        tm = env.termination_manager
        failed = None
        for name in self.FAILURES:
            if name in tm.active_terms:
                hit = tm.get_term(name)[env_ids]
                failed = hit if failed is None else failed | hit
        failures = int(failed.sum()) if failed is not None else 0
        successes = int(tm.get_term("ball_caught")[env_ids].sum()) if "ball_caught" in tm.active_terms else 0
        command = env.command_manager.get_term(command_name)
        if "throws_completed" in command.metrics:
            completed = command.metrics["throws_completed"]
            successes += int((completed - self._counted).sum())
            self._counted = completed.clone()
            self._counted[env_ids] = 0.0  # the command zeroes the counts of the resetting envs right after this call
        if successes + failures > 0:
            self.success_rate += smoothing * (successes / (successes + failures) - self.success_rate)
            if self.success_rate >= up:
                self.difficulty = min(1.0, self.difficulty + step)
            elif self.success_rate < down:
                self.difficulty = max(0.0, self.difficulty - step)
            self._apply(env, target, command_name)
        return {"difficulty": self.difficulty, "success_rate": self.success_rate}
