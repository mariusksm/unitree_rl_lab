"""PPO for stage B: fine-tuning a pretrained actor on a new task (the Advanced tasks, since 2026-11-03).

Analysis and sources: H-ReACT/analysis/2026-11-03_phase2_analyse/ (outside the repository).
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO


class AnchoredPPO(PPO):
    """PPO with the safeguards for fine-tuning a stage-A actor (``--finetune`` loads only the actor, the critic starts
    untrained). In the project group fine-tuning went wrong three times: the adaptive learning rate fell to its floor
    in the first iteration, the value loss rose to tens of millions, and a policy forgot how to catch. All additions are
    off with the defaults, which gives plain PPO:

    - **Critic warm-up:** for the first ``critic_warmup_iterations`` updates only the value loss is optimized. The actor
      keeps its weights, and the learning rate is not adapted: a frozen actor has KL 0, which would drive the adaptive
      rate to its maximum. PIRLNav (2023): +3 points from the warm-up, +8 together with learning-rate schedules.
    - **Actor learning rate:** ``actor_learning_rate``, ramped up linearly over ``actor_lr_warmup_iterations`` after
      the critic warm-up; the critic keeps ``learning_rate``. With ``schedule="adaptive"`` the actor rate is only
      lowered when the KL exceeds twice ``desired_kl`` and recovers up to ``actor_learning_rate``, never above. The
      plain adaptive schedule multiplies the rate by 1.5 per mini-batch while the KL is small and would undo a small
      fine-tuning rate within one iteration.
    - **Anchor:** ``anchor_kl_coef`` · KL(anchor ‖ actor) on the batch observations. The anchor is a frozen copy of the
      actor at the first update, i.e. the loaded stage-A actor. Its weight falls linearly to ``anchor_kl_final_coef``
      over ``anchor_kl_decay_iterations`` updates after the critic warm-up. KL to the pretrained policy (kickstarting)
      is the best-supported measure against forgetting (Wołczyk et al. 2024, VPT, TWIST, OP3 soccer). The anchor is
      saved with every checkpoint and restored on a full resume; ``--finetune`` (actor only) takes a new anchor.
    - **Std cap:** ``max_std`` clamps the action std after every update (with unbounded std Andy's runs diverged).

    Supports the non-recurrent MLP actor and critic used here; RND, symmetry and multi-GPU training are not supported.
    """

    def __init__(
        self,
        *args,
        actor_learning_rate: float | None = None,
        actor_lr_warmup_iterations: int = 0,
        critic_warmup_iterations: int = 0,
        anchor_kl_coef: float = 0.0,
        anchor_kl_final_coef: float = 0.0,
        anchor_kl_decay_iterations: int = 0,
        max_std: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if self.rnd or self.symmetry or self.is_multi_gpu:
            raise ValueError("AnchoredPPO supports neither RND, symmetry nor multi-GPU training")
        if self.actor.is_recurrent or self.critic.is_recurrent:
            raise ValueError("AnchoredPPO supports only non-recurrent models")
        self.critic_learning_rate = self.learning_rate
        self.actor_base_learning_rate = self.learning_rate if actor_learning_rate is None else actor_learning_rate
        self.actor_lr_warmup_iterations = actor_lr_warmup_iterations
        self.critic_warmup_iterations = critic_warmup_iterations
        self.anchor_kl_coef = anchor_kl_coef
        self.anchor_kl_final_coef = anchor_kl_final_coef
        self.anchor_kl_decay_iterations = anchor_kl_decay_iterations
        self.max_std = max_std
        self.anchor = None
        self.updates_done = 0  # PPO updates (= iterations) of this algorithm, restored on a full resume
        self.kl_guard = 1.0  # factor of the adaptive schedule on the actor rate, at most 1
        # one optimizer with two parameter groups: the actor rate is set at every update, the critic rate stays
        self.optimizer = type(self.optimizer)(
            [
                {"params": list(self.actor.parameters()), "lr": self.actor_base_learning_rate},
                {"params": list(self.critic.parameters()), "lr": self.critic_learning_rate},
            ],
            lr=self.critic_learning_rate,
        )
        self.learning_rate = self.actor_base_learning_rate  # the runner logs this as the learning rate

    def set_anchor(self) -> None:
        """Freeze a copy of the current actor as the anchor of the KL term."""
        self.anchor = copy.deepcopy(self.actor)
        self.anchor.eval()
        for param in self.anchor.parameters():
            param.requires_grad_(False)

    def _schedule(self) -> tuple[bool, float, float]:
        """(critic warm-up running, actor learning rate, anchor weight) for the current update."""
        if self.updates_done < self.critic_warmup_iterations:
            return True, 0.0, 0.0
        k = self.updates_done - self.critic_warmup_iterations
        lr = self.actor_base_learning_rate * self.kl_guard
        if self.actor_lr_warmup_iterations > 0:
            lr *= min(1.0, (k + 1) / self.actor_lr_warmup_iterations)
        coef = self.anchor_kl_coef
        if self.anchor_kl_decay_iterations > 0:
            coef += (self.anchor_kl_final_coef - self.anchor_kl_coef) * min(1.0, k / self.anchor_kl_decay_iterations)
        return False, lr, coef

    def update(self) -> dict[str, float]:
        """PPO update with critic warm-up, separate actor rate and anchor (see the class docstring)."""
        if self.anchor is None and self.anchor_kl_coef > 0.0:
            self.set_anchor()  # no update has happened yet: this is the loaded stage-A actor
        warmup, actor_lr, anchor_coef = self._schedule()
        self.optimizer.param_groups[1]["lr"] = self.critic_learning_rate
        sums = dict.fromkeys(("value", "surrogate", "entropy", "anchor_kl", "kl"), 0.0)
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for batch in generator:
            self.optimizer.param_groups[0]["lr"] = actor_lr
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)
            self.actor(
                batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[0], stochastic_output=True
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            distribution_params = self.actor.output_distribution_params
            entropy = self.actor.output_entropy

            with torch.inference_mode():
                kl_mean = torch.mean(self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params))
            # KL guard on the actor rate only, and not while the actor is frozen
            if not warmup and self.schedule == "adaptive" and self.desired_kl is not None:
                if kl_mean > self.desired_kl * 2.0:
                    self.kl_guard = max(1e-5 / self.actor_base_learning_rate, self.kl_guard / 1.5)
                elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                    self.kl_guard = min(1.0, self.kl_guard * 1.5)
                actor_lr = self._schedule()[1]

            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
            advantages = torch.squeeze(batch.advantages)
            surrogate = -advantages * ratio
            surrogate_clipped = -advantages * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_loss = torch.max((values - batch.returns).pow(2), (value_clipped - batch.returns).pow(2)).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            anchor_kl = torch.zeros((), device=self.device)
            if self.anchor is not None:
                with torch.no_grad():
                    self.anchor(batch.observations, stochastic_output=True)
                    anchor_params = tuple(p.detach() for p in self.anchor.output_distribution_params)
                anchor_kl = torch.mean(self.actor.get_kl_divergence(anchor_params, distribution_params))

            loss = self.value_loss_coef * value_loss
            if not warmup:  # during the warm-up the actor is not part of the loss and keeps its weights
                loss = loss + surrogate_loss - self.entropy_coef * entropy.mean() + anchor_coef * anchor_kl

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            sums["value"] += value_loss.item()
            sums["surrogate"] += surrogate_loss.item()
            sums["entropy"] += entropy.mean().item()
            sums["anchor_kl"] += anchor_kl.item()
            sums["kl"] += kl_mean.item()

        self._cap_std()
        self.updates_done += 1
        self.learning_rate = actor_lr
        self.storage.clear()
        num_updates = self.num_learning_epochs * self.num_mini_batches
        loss_dict = {key: value / num_updates for key, value in sums.items()}
        loss_dict["anchor_coef"] = anchor_coef
        loss_dict["critic_warmup"] = float(warmup)
        return loss_dict

    def _cap_std(self) -> None:
        """Clamp the action std of the actor to ``max_std`` (if set)."""
        if self.max_std is None:
            return
        distribution = self.actor.distribution
        with torch.no_grad():
            if hasattr(distribution, "std_param"):
                distribution.std_param.clamp_(max=self.max_std)
            elif hasattr(distribution, "log_std_param"):
                distribution.log_std_param.clamp_(max=float(torch.log(torch.tensor(self.max_std))))

    def save(self) -> dict:
        """Models and optimizer as in PPO, plus the anchor and the update counter."""
        saved = super().save()
        if self.anchor is not None:
            saved["anchor_state_dict"] = self.anchor.state_dict()
        saved["anchored_ppo_updates_done"] = self.updates_done
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """As PPO; a full resume (``load_cfg`` None) also restores the anchor and where the schedules stand."""
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        # cap the loaded std now, before the first rollout: the anchor, copied at the first update, then has the capped
        # std too, so the cap alone never shows up as a distance to it (capping in place after a rollout would also
        # break that copy, the rollout leaves inference-mode views of the std parameter)
        self._cap_std()
        if load_cfg is None:
            if "anchor_state_dict" in loaded_dict:
                self.set_anchor()
                self.anchor.load_state_dict(loaded_dict["anchor_state_dict"], strict=strict)
            self.updates_done = loaded_dict.get("anchored_ppo_updates_done", 0)
        return load_iteration
