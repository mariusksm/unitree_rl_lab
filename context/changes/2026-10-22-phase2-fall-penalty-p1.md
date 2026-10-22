# P1 Changes (2026-10-22) — Phase 2: Fall Penalty and Larger Alive Bonus

A short pilot run before the first full training on the workstation showed that Phase 2, as of commit `24d448d`,
converges to falling before the throw. This change adds a penalty for falls and raises the `alive` bonus. Only
`BC/robots/g1_29dof/phase2/ball_catch_env_cfg.py` changes. The observation contract, the ball and the throw logic
stay as they are.

Files touched (`BC/` = `source/unitree_rl_lab/unitree_rl_lab/tasks/ball_catching/`):

| File | Change |
|---|---|
| `BC/robots/g1_29dof/phase2/ball_catch_env_cfg.py` | `alive` 0.15 → 1.0; new `termination_penalty` (−200, falls only) |
| `context/unitree_rl_lab_spec.md` | Sections 2, 7.2, 10, changelog |

---

## 1. Problem: falling is the best policy

Phase 2 has no positive dense reward apart from `alive` (Unified and Phase 1 have the tracking rewards). At the
start of training the per-step penalties are far larger than `alive`:
- `action_rate` alone costs ≈ −0.056 per step, about 20× `alive` (+0.003 per step). The exploration noise (std 1.0)
  produces it unavoidably.
- Falling cost nothing extra.

So the shortest episode had the highest return.

Pilot, Phase 2 at `24d448d`, 4096 envs, stopped after 108 iterations:
- 100 % of the episodes end by `bad_orientation`.
- Mean episode length 12.5 → 6.6 → ≈9 steps (0.13–0.25 s), while the mean reward rises from −1.37 to −0.62.
- All ball rewards are 0, because the episodes end before the throw (throw delay 0.5–1.5 s).

A 30k-iteration run would most likely have stayed in this optimum.

## 2. Change

```python
alive = RewTerm(func=mdp.is_alive, weight=1.0)  # 0.02/step; discounted at most ~2 < catch bonus 10
termination_penalty = RewTerm(
    func=mdp.is_terminated_term,
    weight=-200.0,              # -4 per fall
    params={"term_keys": ["base_height", "bad_orientation"]},  # falls only: catch and drop are not penalized
)
```

- `is_terminated_term` restricted to the fall terminations. The plain `is_terminated` would also hit the success
  termination `ball_caught` and the ball terminations.
- `alive` 1.0: with γ = 0.99 an endless stream of 0.02 per step is worth at most ≈2, less than the catch bonus (10).
  Delaying the catch to keep collecting `alive` does not pay.
- The weight −200 is the usual Isaac Lab locomotion value.

## 3. Evidence (2048 envs, 250 iterations each, run as scratch variants of the Phase 2 config)

| Variant | Episode length | Ends by fall | Ends after the throw (ball dropped) | Hands–ball distance |
|---|---|---|---|---|
| As of `24d448d` (after 108 iterations, 4096 envs) | ≈9 steps | 100 % | 0 % | 2.2 m |
| B: + fall penalty −200 | 86 steps | 76 % | 25 % | 1.3 m |
| **A: + fall penalty −200, `alive` 1.0 (adopted)** | **104 steps** | **15 %** | **86 %** | **0.7 m** |

Variant A over training:

| Iteration | Episode length | Falls | Ball dropped |
|---|---|---|---|
| 0 | 12.8 | 0 % | 0 % |
| 100 | 39.7 | 99.9 % | 0.05 % |
| 150 | 76.0 | 91.7 % | 8.8 % |
| 200 | 91.1 | 64.5 % | 36.7 % |
| 249 | 104.3 | 14.7 % | 86.0 % |

No catches after 250 iterations, as expected this early. Both variants fix the optimum; A learns to stand about
twice as fast. A 3-iteration smoke run of the real Phase 2 config confirms both terms are active.

Timing on the RTX 3080: 2.85 s per iteration at 4096 envs (≈24 h for the planned 30k iterations), ≈6.5 GB GPU memory.
Two Isaac Sim trainings at once do not fit into 10 GB.

## 4. Unified and Phase 1 are not affected

Both get dense, positive tracking rewards (Unified: weights summing to 5.0), which make staying alive worthwhile
despite the stronger `action_rate` (−0.1). Phase 1 is the BeyondMimic recipe; its earlier run converged.

Unified pilot as of this change (2048 envs, 250 iterations, current overhand clip), unchanged config:

| Iteration | Episode length | Mean reward | `ee_body_pos` | `anchor_pos` |
|---|---|---|---|---|
| 0 | 12.8 | −1.03 | 100 % | 0 % |
| 50 | 6.3 | −0.06 | 100 % | 0 % |
| 100 | 9.7 | 0.11 | 100 % | 0.1 % |
| 150 | 61.0 | 0.66 | 85 % | 21 % |
| 200 | 107.7 | 2.06 | 66 % | 44 % |
| 249 | 135.9 | 3.30 | 68 % | 42 % |

After a short initial dip the episodes grow and the reward turns positive. Episodes still end by tracking
terminations; no ball termination occurred yet. No change was made to Unified or Phase 1.
