# Review guide: robot safety and chain v2 (2026-11-20)

This guide is the entry point for reviewing two pieces of work:
- **On `main` (2026-11-19/20, commits `3df7c99` to `b2498d6`):** the robot-safety work. Measuring what the robot does after a lost ball, the stage Advanced-S, the catchability gate, and the ball visibility options.
- **On branch `chain-v2` (2026-11-20):** a seventh Unified run plus the Advanced stages reordered on it (V2-A, V2-C, V2-D, V2-E). Built and smoke-tested, not trained.

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
  - stage S on `main`, now in training;
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

### 2.5 Chain v2 (branch `chain-v2`; Section 7.2 "Chain v2", 7.3 "Seventh run", Section 11 item 18)
- **Unified, seventh run:** A2's penalties (in the config since 2026-11-03) plus the ball before the throw in half of the cycles.
- **V2-A** (from Unified 7): A's definition with A2's penalties and B's bounce threshold, plus:
  - approach rewards times P;
  - `aim_at_launch`;
  - ball before the throw in half of the throws, constant;
  - `arm_rest_pose_exp`: arm joints toward the human catchers' pose at the start of the 108 clips while the robot waits.
- **V2-C:** C's real lifecycle plus S's loss lifecycle: fall −200, capture point, and the lost ball hidden in a share that falls from 1 to 0.5.
- **V2-D:** D's pushes in the ball phase plus `push_during_flight` (D2): one push in the flight of 30 % of the throws, with strength from D's curriculum.
- **V2-E:** E's wider and frontal throws plus S's 15 % throws at 0.45–1.0 m.
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
| Arm pose while waiting | human clips: upper arms hanging slightly out, elbows ≈ 62°; the Advanced stages since A raise the forearms in front of the chest | Section 7.2 "Chain v2" |

## 4. Decisions open for discussion

Each item gives our choice, the alternatives, and what would change it.

1. **Fall price −200 (weight −10000).** Chosen so that the policy accepts at most one fall per about ten extra catches (research of 2026-11-19, `analysis/2026-11-19_stufe_f/`). Too high a price makes a robot timid in the literature, too low makes falls free. Alternative: a Lagrangian on the fall rate. Check: S's catch rate on E's throws should stay within a point of E.
2. **Visibility randomization of 0.5 each** (ball before the throw, lost ball hidden). VICON tracks the ball everywhere, and the bridge's filters may miss cases, for example a bounce that looks ballistic. Alternative: rely on the bridge only. That is fragile, given E's 99.9 % falls with a parked ball.
3. **Where the robustness to a visible ball is learned.** S's second run answered this: anchored fine-tuning did not unlearn E's reflex (pitfall 41). Chain v2 already moves the ball before the throw into Unified 7. Open: V2-C still ramps in the visible lost ball by fine-tuning and would most likely fail the same way. Proposal: Unified 7 also trains losses that do not end the cycle, with the lost ball visible in half of them, so the policy learns from scratch while the reference keeps it upright. Alternative: an anchor that, in such states, points at the anchor actor's output with the ball masked.
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
12. **Deployment layers** (Section 11, item 17):
    - catch policy only around a throw;
    - tilt check to Passive, which is not protective;
    - gantry for the first tests;
    - the bridge rules: no ball before the throw, the lost ball hidden, the gate.

## 5. Risks and known limitations

- **S** is the fallback for the first robot tests only if the live pipeline hides the ball before the throw and the lost ball (pitfall 41).
- **Chain v2 is untrained.** The arm rest weight, the D2 share and the visibility ramps are first guesses.
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
| `BC/mdp/commands/ball_command.py` | perception options, gate (`closest_catchable_approach`), `aim_at_launch`, loss lifecycle, parked ball |
| `BC/mdp/rewards.py` | `ball_interception_exp(posture=...)`, `ball_height_penalty(command_name=...)`, `capture_point_exp`, `arm_rest_pose_exp` |
| `BC/mdp/events.py` | `push_during_flight` (D2) |
| `BC/mdp/curriculums.py` | `ramp_ball_visibility` |
| `BC/agents/rsl_rl_ppo_cfg.py` | `AdvancedSPPORunnerCfg`, `V2PPORunnerCfg` |
| `BC/robots/g1_29dof/advanced/ball_catch_env_cfg.py` | first chain A … S, stage features as `apply_*` functions |
| `BC/robots/g1_29dof/advanced/v2_env_cfg.py` | chain v2 (V2-A, -C, -D, -E) |
| `BC/robots/g1_29dof/unified/ball_catch_env_cfg.py` | Unified (seventh run: `SHOW_PARKED_BALL_PROB`) |
| `scripts/rsl_rl/eval_catch.py`, `sim2sim/{run,ball}.py` | evaluation options |
| `scripts/rsl_rl/dump_task_cfgs.py` | dump and compare the configs of all ball tasks (refactor check) |
| `context/unitree_rl_lab_spec.md` | Sections 2, 7.2, 7.3, 9 (pitfalls 38–40), 11 (items 16–18), 14 (changelog) |

## 7. How to verify

- **Configs of all tasks:** `python scripts/rsl_rl/dump_task_cfgs.py <out.json>` once with `main` and once with the branch on the `PYTHONPATH`, then `python scripts/rsl_rl/dump_task_cfgs.py --compare main.json branch.json`. On 2026-11-20 the first chain was identical, and Unified differed only in `show_parked_ball_prob`.
- **Smoke test of a stage:**

  ```
  python scripts/rsl_rl/train.py --task Unitree-G1-29dof-BallCatch-V2-C --headless --num_envs 64 --max_iterations 3 --resume --resume_experiment unitree_g1_29dof_ballcatch_advanced_e --load_run 2026-11-04_04-03-34 --checkpoint model_3999.pt --finetune
  ```

  E's checkpoint has the same actor contract, so it can stand in for a V2 checkpoint.
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

## 8. Status and next steps (2026-11-20, 04:00)

- S is trained and evaluated (2026-11-20, `analysis/2026-11-20_advanced_s_final/`): same catches as E, half the dives on wide throws, no robustness to a visible ball.
- Chain v2 waits for this review, the decision on V2-C's visible lost ball (decision 3) and on the ball and latency measurements before Unified 7 starts.
