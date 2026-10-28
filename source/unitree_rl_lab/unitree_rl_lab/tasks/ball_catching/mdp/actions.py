from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class ClampedJointPositionAction(JointPositionAction):
    """Joint position targets clamped to the joint limits of the asset.

    With the BeyondMimic action scale the targets are otherwise unbounded: the Phase 2 baseline (2026-10-27) drove the
    ankle-roll targets 2-4 rad past the joint range and stood on the hard stops with saturated torque. Clamping the
    targets to the hard limits removes that lever without shrinking the reachable range (the soft limits would forbid
    straight knees): with the joint and its target on the stop, the PD error and thus the torque is zero.

    The clamp is the ``clip`` of :class:`JointAction` (applied to the processed targets), intersected with the asset's
    hard joint limits at startup. ``export_deploy_cfg`` therefore writes the limits into ``deploy.yaml`` and the C++
    deployment (``JointAction::process_actions``) clamps the same way. ``target_excess`` feeds a penalty: beyond the
    clamp a larger raw action changes nothing, so without it the raw actions (``last_action`` observation) can run off.
    """

    cfg: ClampedJointPositionActionCfg

    def __init__(self, cfg: ClampedJointPositionActionCfg, env: ManagerBasedEnv):
        if cfg.clip is None:
            raise ValueError("ClampedJointPositionActionCfg.clip must not be None: the joint limits are applied through it")
        super().__init__(cfg, env)
        limits = self._asset.data.joint_pos_limits[:, self._joint_ids]
        self._clip = torch.stack(
            [torch.maximum(self._clip[..., 0], limits[..., 0]), torch.minimum(self._clip[..., 1], limits[..., 1])], dim=-1
        )
        self.target_excess = torch.zeros(self.num_envs, device=self.device)
        """Per env: sum over the joints of |unclamped - clamped| target (rad) of the last processed action."""

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        targets = self._raw_actions * self._scale + self._offset
        self._processed_actions = torch.clamp(targets, min=self._clip[..., 0], max=self._clip[..., 1])
        self.target_excess = (targets - self._processed_actions).abs().sum(dim=-1)


@configclass
class ClampedJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for :class:`ClampedJointPositionAction`."""

    class_type: type = ClampedJointPositionAction

    clip: dict[str, tuple] | None = {".*": (-math.inf, math.inf)}
    """Clip range of the targets (dict of regex expressions), intersected with the asset's hard joint limits; the
    default leaves just the joint limits. Must not be ``None`` (``export_deploy_cfg`` exports the clamp only if set)."""
