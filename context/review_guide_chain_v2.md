# Review guide: robot safety and chain v2 (2026-11-20)

This guide is the entry point for reviewing two pieces of work:
- **On `main` (2026-11-19/20, commits `3df7c99` to `a7365ea`):** the robot-safety work. Measuring what the robot does after a lost ball, the stage Advanced-S (third run with a masked anchor in training), the catchability gate, and the ball visibility options.
- **On branch `chain-v2` (2026-11-20):** a seventh Unified run that also learns to leave a visible ball alone, plus the Advanced stages reordered on it (V2-A, V2-C, V2-D, V2-E). Built and smoke-tested, not trained.

It lists what changed and why, where it is in the code, which measurements stand behind it, and which decisions are open for discussion. The specification (`context/unitree_rl_lab_spec.md`) stays the reference for details; section numbers below point into it.

## 1. Background

- **Pipeline.**
  - Stage A, **Unified** (`Unitree-G1-29dof-BallCatch-Unified`): learns catching from scratch while tracking human underhand-catch clips. The actor sees only what the real robot has.
  - Stage B, the **Advanced chain**: fine-tunes Unified's actor without the reference. Each stage starts from the previous checkpoint and adds one change. The order is A (holding and posture definition), A2 (no pressed hands, arm torque limit), B (ball rebound, placeholder), C (real lifecycle), D (pushes in the ball phase) and E (wider throws).
  - Fine-tuning uses `AnchoredPPO`: critic warm-up, low actor learning rate, KL anchor to the previous actor (Section 7.2).
- **E is the best stage.** It catches 88.8 % per throw in its own conditions and 96 % in MuJoCo (Section 10).
- **New requirement.** In November 2026 the project group required that the robot must not hurt itself. Above all, no fall when it cannot catch a ball, and nothing bad after it drops one (Section 11, item 17).
- **What the measurements showed:**
  - Every stage ended the episode at a lost ball, so the time after a loss was never trained or measured, and a fall cost the policy no more than the lost ball.
  - With the episode kept running, E fell after 96 % of its lost balls: it goes after the ball on the floor (pitfall 39).
  - When the ball is reported before the throw, still in the thrower's hand, E chases it and falls in 99.9 % of the episodes (pitfall 40).
- **Responses:**
  - stage S on `main`: second run trained and evaluated, third run (masked anchor) in training;
  - the catchability gate, a rule for the live pipeline that is evaluated in simulation;
  - ball visibility randomization, because VICON hides nothing by itself;
  - chain v2 on the branch, which puts all lessons into a clean order.

## 2. What changed, by topic

### 2.1 Evaluation (main)
- `scripts/rsl_rl/eval_catch.py`:
  - `--after_loss_s T` keeps the episode running T s after a loss and records falls, pelvis, hands, tilt, lift-offs, displacement, ball-to-foot distance and P in new per-throw columns (`lost_as`, `post_loss_*`).
  - Scenario `unreachable`: aims up to ±1.0 m sideways.
  - `--hide_lost_ball` hides the lost ball.
  - `--gate_deviation D` adds the gate (column `gated`).
  - `--show_ball_before_throw` reports the ball in the thrower's hand.
- `sim2sim/run.py` and `sim2sim/ball.py` have the same options in MuJoCo.
- Analysis scripts outside the repository: `analysis/2026-11-19_ballverlust/skripte/` (`nach_verlust.py` for the post-loss tables, `gate_offline.py` for the gate estimate from the recorded throws).

### 2.2 Perception model (`BC/mdp/commands/ball_command.py`, main; Section 7.2)
- **Lost ball:** `hide_lost_ball_below` / `_beyond` stop reporting a lost ball (below 0.4 m, or more than 3 m away after the planned arrival). `hide_lost_ball_prob` applies this in a share of the throws only.
- **Ball before the throw:** `show_parked_ball_prob` reports the parked ball before the launch in a share of the throws, swaying as if held in a hand (`parked_ball_sway`).
- **Catchability gate:** `gate_max_deviation`, `gate_center`, `gate_height_range`, `gate_drag_coefficient`, using `closest_catchable_approach`.
  - Decided once per throw, on the first measurement of the flying ball.
  - The path is predicted with gravity and nominal drag.
  - The throw is never reported if it comes no closer than the threshold to the catch point at a catchable height.
- **Random draws:** shares of 0 or 1 draw no random numbers, so tasks without these options keep their random stream.

### 2.3 Throws (main)
- **`aim_at_launch`:** the aim point follows the robot until the launch. Without it, pushes and drift during a wait of up to 8 s moved throws out of the trained ranges (pitfall 38).
- **`unreachable_share`, `unreachable_lateral_range`:** a share of the throws aimed to either side.

### 2.4 Losing a ball safely: Advanced-S (main; Section 7.2)
- `rethrow_after_loss`: a loss ends only its throw. The ball lies there 1–5 s, then the next throw is armed. Only a fall or the time-out ends the episode.
- Rewards:
  - approach terms times the posture factor P;
  - ball-height penalty only while the ball is in play;
  - one-time fall penalty of −200 (`is_terminated_term`, weight −10000);
  - capture point over the feet while the robot should stand still (`capture_point_exp`).
- Throws: 15 % aimed 0.45–1.0 m to the side, all throws aimed at the launch.
- Visibility: in half of the throws the ball is reported before the throw, and in half the lost ball is hidden. Both shares are ramped in between iterations 300 and 1500 (`mdp.ramp_ball_visibility`).
- Training: `AdvancedSPPORunnerCfg`, with the anchor down to 0.1 within 500 iterations after the warm-up, from E's final checkpoint.
  - The first run (02:41) did not improve in 600 iterations and was stopped.
  - The second run (03:13) has the ramp and the faster anchor.
  - S is trained without the gate, so that the policy is safe on its own.
- **Third run: masked anchor** (`BC/agents/anchored_ppo.py`, since 2026-11-20 14:20). The second run did not unlearn E's reflex to a visible ball that is not in play (pitfall 41).
  - The observation group `anchor_mask` (`SObservationsCfg`, term `mdp.ball_shown_not_in_play`: a ball in view before the launch or after the loss) flags samples; it is no input of actor or critic.
  - In flagged samples the anchor term is KL(anchor(observation with the 50 ball entries of `policy` zeroed) ‖ actor): it points at what E does when it sees no ball. Over all joints, weight 1.0, not decayed.
  - The plain anchor and the KL guard use only the other samples. `train.py` fills in the ball indices (`masked_anchor_indices`, terms `ball_*`) from the observation manager.

### 2.5 Chain v2 (branch `chain-v2`; Section 7.2 "Chain v2", 7.3 "Seventh run", Section 11 item 18)
- **Unified, seventh run:** A2's penalties (in the config since 2026-11-03), the ball before the throw in half of the cycles, and a lost ball that no longer ends the episode at once:
  - `BallCommandCfg.track_loss` marks the loss (below 0.1 m or 6 m away, as the old terminations), and `ball_dropped` = `mdp.ball_lost_since` ends the episode 3 s later, or on the clip's last step if that comes first (the next cycle would clear the loss). `ball_missed` is part of it.
  - The reference holds its catch pose meanwhile, so the robot learns from scratch to stand next to the lost ball. It stays visible on the floor in half of the cycles and is hidden in the other half.
  - The approach rewards are zero after the loss, `ball_height_penalty` counts only while the ball is in play; the play config evaluates as before.
  - The critic sees the loss and the time since it: `catch_phase` has two more entries in all three tasks (1565 instead of 1555), as it already had the held share for the hold phase.
- **V2-A** (from Unified 7): A's definition with A2's penalties and B's bounce threshold, plus:
  - approach rewards times P;
  - `aim_at_launch`;
  - ball before the throw in half of the throws, constant;
  - `arm_rest_pose_exp`: arm joints toward the human catchers' pose at the start of the 108 clips while the robot waits.
- **V2-C:** C's real lifecycle plus S's loss lifecycle: fall −200, capture point, and the lost ball visible in half of the throws from the start (no ramp: Unified 7 learned it).
- **V2-D:** D's pushes in the ball phase plus `push_during_flight` (D2): one push in the flight of 30 % of the throws, with strength from D's curriculum.
- **V2-E:** E's wider and frontal throws plus S's 15 % throws at 0.45–1.0 m.
- **All V2 stages:** S's runner (`V2PPORunnerCfg(AdvancedSPPORunnerCfg)`: anchor down to 0.1 within 500 iterations, the arm joints at 10 %, masked anchor); V2-A on carry S's group `anchor_mask`.
- **Refactor:** the stage features of the first chain are now shared functions (`apply_real_lifecycle`, `apply_ball_phase_push_ramp`, `apply_wide_throws`, `apply_loss_lifecycle`, `apply_unreachable_throws`, `apply_loss_evaluation` in `ball_catch_env_cfg.py`). A dump of all task configs before and after shows the first chain unchanged.

## 3. Measurements behind the decisions

All in Isaac with 512 robots × 180 s unless noted. Sources are under `analysis/` (outside the repository) and in Sections 9 and 10.

| Question | Result | Source |
|---|---|---|
| What does E do after a lost ball (its own throws)? | falls after 96 % of the losses; 10 % of all throws end with a fall with pushes, 5.7 % without | pitfall 39, `2026-11-19_ballverlust/` |
| Does hiding the lost ball help? | falls after a loss 96 → 12 %, the robot stands still (0.1 m, P 0.99) | same |
| Throws up to 1 m sideways | 43 % of throws end with a fall (32 % with the ball hidden); 92 % beyond 0.7 m | same |
| Gate on the line of flight only | lets through oblique throws that cross at head height (E falls after 44 % of them) | same, `gate_offline.py` |
| Gate prediction error (state 0.08 s after release) | 2 cm with nominal drag, 24 cm without drag, 9 cm with 0.1 m/s velocity noise | Section 7.2 |
| Gate at 0.6 m (noisy, Isaac) | 34 % of throws gated, none followed by a fall; falls 31 → 6 %, catches 55 → 53 % | pitfall 39 |
| Gate at 0.6 m (MuJoCo) | falls 27 → 7.5 % at the same catch rate | same |
| Gate cost on E's own throws (estimate) | 0.3–0.9 % gated at 0.45–0.5 m, at most 0.6 points of catches | same |
| Ball reported in the thrower's hand | E falls in 99.9 % of the episodes, 0.6 s after the start (without: 92 % caught, 0.7 % falls) | pitfall 40 |
| S's first run (full visibility shares, slow anchor) | 93 % of episodes ended with a fall, flat for 500 iterations after the warm-up | Section 7.2 |
| S's second run (ramp, fast anchor), evaluated like E | catches as E (88.6 / 94.3 % with / without pushes); falls before a loss on wide throws 8.5 → 4.3 %; with the gate and the lost ball hidden 6.1 → 3.5 % falls; **with a visible lost ball or a ball before the throw as many falls as E** (96 %, 99.9 %); training falls rose with the visibility to 92 % and stayed there for 2500 iterations | Section 10, pitfall 41, `analysis/2026-11-20_advanced_s_final/` |
| Throws aimed at the launch | E in S's play config 92.2 % against 89.5 % in its own: about 3 points | Section 10 |
| Unified 6's actor in Unified 7's training config (256 robots × 60 s; the post-loss window) | two runs: 54 and 58 losses; about half of the windows ended by `ball_dropped` after 3 s (one at the clip end), the other half by a tracking termination; none cleared by a new cycle; ball reported in 29 and 36 % of the post-loss steps; critic loss entries as intended; split by visibility (512 robots, 104 windows): with the ball in view 90 % of the windows end with a tracking termination after 1.3 s on average, with it hidden 96 % reach the end after 3 s. Unified 6 already has E's reflex | Section 7.3 |
| Arm pose while waiting | human clips: upper arms hanging slightly out, elbows ≈ 62°; the Advanced stages since A raise the forearms in front of the chest | Section 7.2 "Chain v2" |

## 4. Decisions open for discussion

Each item gives our choice, the alternatives, and what would change it.

1. **Fall price −200 (weight −10000).** Chosen so that the policy accepts at most one fall per about ten extra catches (research of 2026-11-19, `analysis/2026-11-19_stufe_f/`). Too high a price makes a robot timid in the literature, too low makes falls free. Alternative: a Lagrangian on the fall rate. Check: S's catch rate on E's throws should stay within a point of E.
2. **Visibility randomization of 0.5 each** (ball before the throw, lost ball hidden). VICON tracks the ball everywhere, and the bridge's filters may miss cases, for example a bounce that looks ballistic. Alternative: rely on the bridge only. That is fragile, given E's 99.9 % falls with a parked ball.
3. **Where the robustness to a visible ball is learned.** S's second run answered this: anchored fine-tuning did not unlearn E's reflex (pitfall 41). **Decided by Marius on 2026-11-20:**
   - Unified 7 learns both visible balls from scratch: the ball before the throw and the lost ball on the floor for 3 s.
   - V2-C keeps the lost ball visible in half of the throws, without a ramp.
   - All V2 stages use S's masked anchor, so the fine-tuning keeps what Unified 7 learned.
   - S's third run tests the masked anchor alone, by fine-tuning E. Nothing of chain v2 is trained before it is evaluated.
4. **Fast anchor decay (500 instead of 3000 iterations).** The earlier stages gained mostly in their last 1000 iterations. A faster decay risks forgetting catching skills. Check: the catch rate in S's suite and S evaluated in E's environment.
5. **Gate design.**
   - Decided once, on the first measurement, rather than re-deciding (table tennis: decide once).
   - Closest approach at a catchable height, rather than the line of flight or the arrival point.
   - Nominal drag.
   - Threshold 0.45–0.5 m for E; a new threshold after S.
   - Where it lives: the ball bridge of the live pipeline, not the policy.
   - Open: the real bridge fits over several frames and should decide more sharply than our single noisy measurement. Spin of the real ball is unmeasured.
6. **Train without the gate.** The policy must be safe on its own; the gate is a second layer. Alternative: train with the gate, which makes the policy simpler but depends on the bridge.
7. **Let-go throws: 15 % at 0.45–1.0 m.** This covers the band where E still catches half but falls after a fifth to a third, up to clearly unreachable throws. Alternatives: a privileged "let go" flag that turns the reach shaping off, or a smaller band.
8. **Capture-point reward.** The pelvis stands in for the centre of mass, with weight 2 and std 0.12 m, active in all stand-still phases (waiting, holding, after a loss). Alternative: only after a loss.
9. **`aim_at_launch` changes the throw distribution relative to E.** Comparisons therefore evaluate S also in E's play configuration.
10. **Chain v2 order.**
    - Safety goes into C, the lifecycle stage, so that D and E keep it.
    - D2 at a 30 % share, with the success gate on all throws rather than only the pushed ones.
    - The arm rest term in A, at weight 1 and std 0.5 rad; low priority for Marius.
    - Alternatives: keep S as the last stage, or one combined stage.
11. **When to start Unified 7.** Now, or after the real ball and the latency are measured. With those measurements no stage B would be needed later.
12. **Post-loss window of 3 s in Unified 7, cut at the clip end.** E's reflex shows within a second (with the ball before the throw it fell 0.6 s after the start), so 3 s cover the reflex and some standing. The reference holds the catch pose for 5 s after the catch, so the window always fits after a miss, but a ball dropped late in the hold phase reaches the clip end first; there the loss now ends the episode, otherwise the next cycle would clear it. Alternative: an end hold longer by 3 s, which changes the start sampling over the extended cycle.
13. **The loss phase in the critic of all tasks.** Without it the critic cannot predict the end of the window 3 s after the loss (5 frames of history show no time). The entries go into all three tasks because the critic layout is shared (pitfall 31); checkpoints from before cannot continue with a plain `--resume`, but `--finetune`, `eval_catch.py` and `play.py` load only the actor. Alternative: a window without a fixed end, which leaves the drop without a cost.
14. **Masked anchor: weight 1.0, constant, all joints.** The reflex is in the legs, and the flagged states are where E falls. Alternatives: decay it like the plain anchor, or apply it only to the legs.
15. **Deployment layers** (Section 11, item 17):
    - catch policy only around a throw;
    - tilt check to Passive, which is not protective;
    - gantry for the first tests;
    - the bridge rules: no ball before the throw, the lost ball hidden, the gate.

## 5. Risks and known limitations

- **S** is the fallback for the first robot tests only if the live pipeline hides the ball before the throw and the lost ball (pitfall 41).
- **Chain v2 is untrained.** The arm rest weight, the D2 share, the window of 3 s and the masked anchor's weight are first guesses.
- **Unified 7 learns to stand next to a lost ball while it tracks a static reference.** The V2 stages have no reference; the masked anchor and V2-C's rewards have to keep that behaviour without it.
- **The masked anchor also moves the actor elsewhere.** In S's third run the plain anchor KL stayed around 1 in the first 300 iterations, before the visibility ramp started (the second run ended at 0.07). Its catch rate is the check.
- **Simulation details:**
  - The parked ball sways sinusoidally; a real thrower's wind-up (fast and non-ballistic) is not modelled.
  - In the simulation the lost ball lies still or rolls. In reality a person picks it up while VICON reports it.
- **MuJoCo throws** are simpler than Isaac's (no hard ranges, no speed cap, no unreachable share). The MuJoCo numbers use the earlier Phase 2 throw distribution.
- **The gate has not been tested on real throws.** Spin and wind would bend the path.
- **Evaluation of V2-D and V2-E** runs without flight pushes, for comparability. The eval's push columns record only `push_robot`.
- **`push_during_flight`** runs as an interval event every control step and keeps its plan on the env (`env._flight_push`). It does nothing while `push_robot` is off.

## 6. Where to look

| File | What |
|---|---|
| `BC/mdp/commands/ball_command.py` | perception options, gate (`closest_catchable_approach`), `aim_at_launch`, loss lifecycle, `track_loss`, parked ball |
| `BC/mdp/commands/synced_ball_command.py` | Unified's command: `track_loss` after the perception update; a new cycle clears a loss |
| `BC/mdp/terminations.py` | `ball_lost_since` (Unified 7's post-loss window) |
| `BC/mdp/observations.py` | `ball_shown_not_in_play` (flag of the masked anchor), `catch_phase` with the loss phase |
| `BC/agents/anchored_ppo.py`, `scripts/rsl_rl/train.py` | masked anchor; train.py fills in the ball indices |
| `BC/mdp/rewards.py` | `ball_interception_exp(posture=...)`, `ball_height_penalty(command_name=...)`, `capture_point_exp`, `arm_rest_pose_exp` |
| `BC/mdp/events.py` | `push_during_flight` (D2) |
| `BC/mdp/curriculums.py` | `ramp_ball_visibility` (S only) |
| `BC/agents/rsl_rl_ppo_cfg.py` | `AdvancedSPPORunnerCfg`, `V2PPORunnerCfg` |
| `BC/robots/g1_29dof/advanced/ball_catch_env_cfg.py` | first chain A … S, stage features as `apply_*` functions |
| `BC/robots/g1_29dof/advanced/v2_env_cfg.py` | chain v2 (V2-A, -C, -D, -E) |
| `BC/robots/g1_29dof/unified/ball_catch_env_cfg.py` | Unified (seventh run: `SHOW_PARKED_BALL_PROB`, `POST_LOSS_S`, `HIDE_LOST_BALL_*`) |
| `scripts/rsl_rl/eval_catch.py`, `sim2sim/{run,ball}.py` | evaluation options |
| `scripts/rsl_rl/dump_task_cfgs.py` | dump and compare the configs of all ball tasks (refactor check) |
| `context/unitree_rl_lab_spec.md` | Sections 2, 6.1, 7.2, 7.3, 8, 9 (pitfalls 31, 38–41), 11 (items 16–18), 14 (changelog) |

## 7. How to verify

- **Configs of all tasks:** `python scripts/rsl_rl/dump_task_cfgs.py <out.json>` once with `main` and once with the branch on the `PYTHONPATH`, then `python scripts/rsl_rl/dump_task_cfgs.py --compare main.json branch.json`. On 2026-11-20 (afternoon) the first chain's configs were identical apart from the new field `track_loss` (off), and Unified differed only in the intended fields (spec Section 11, item 18). The dump shows configs, not code: the critic's `catch_phase` returns 5 entries in every task.
- **Smoke test of a stage:**

  ```
  python scripts/rsl_rl/train.py --task Unitree-G1-29dof-BallCatch-V2-C --headless --num_envs 64 --max_iterations 3 --resume --resume_experiment unitree_g1_29dof_ballcatch_advanced_e --load_run 2026-11-04_04-03-34 --checkpoint model_3999.pt --finetune
  ```

  E's checkpoint has the same actor contract, so it can stand in for a V2 checkpoint.
- **Unified 7's post-loss window:** step Unified's training config with a Unified 6 actor and count the losses, the `ball_dropped` ends (3 s after the loss or at the clip end), other terminations inside the window, losses cleared by a new cycle without an episode end (must be 0) and the share of post-loss steps with the ball visible.
- **Safety evaluation of a checkpoint:**

  ```
  eval_catch.py --scenario base|unreachable --after_loss_s 5 [--no_push] [--hide_lost_ball] [--gate_deviation 0.6] [--show_ball_before_throw]
  ```

  Then use `analysis/2026-11-19_ballverlust/skripte/nach_verlust.py` and `gate_offline.py` on the per-throw CSVs.
- **Acceptance** (Section 11, item 17):
  - no fall after about 3000 lost balls (zero in n bounds the rate at 3/n);
  - no more falls before a loss than E;
  - catch rate within a point of E;
  - MuJoCo agrees.

## 8. Status and next steps (2026-11-20, 15:00)

- S's second run is trained and evaluated (`analysis/2026-11-20_advanced_s_final/`): same catches as E, half the dives on wide throws, no robustness to a visible ball.
- S's third run (masked anchor) is in training until about 17:15 and is then evaluated like the second.
- Chain v2 has decision 3 built in. It waits for this review, S's third run and the decision on the ball and latency measurements (decision 11) before Unified 7 starts.
