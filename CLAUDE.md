# CLAUDE.md

Fork of `unitree_rl_lab` (Isaac Lab 2.3.0 / Isaac Sim 5.1.0) for the **H-ReACT** project group:
a Unitree G1 (29 DoF) is trained to catch a thrown ball. The project work lives in
`source/unitree_rl_lab/unitree_rl_lab/tasks/ball_catching/` and `scripts/mimic/`.

## Read first

**`context/unitree_rl_lab_spec.md`** is the central specification and state snapshot. It covers the
architecture, tasks, observation contract, data pipeline, training status, open items and changelog.
Read it before doing any work. Background material (proposal, meeting minutes, plans) lives in `context/`.

## Mandatory: maintain the spec document

Every **substantial change** must be recorded in `context/unitree_rl_lab_spec.md`, in this order:

1. **Update the specification first.** Revise the affected sections (tasks, observation layout, rewards,
   pipeline, status, open items, …) so they describe the new current state. The spec always describes the
   *current* state, not history.
2. **Then append a log entry.** Add a new entry **at the bottom** of the "Changelog" section at the end of
   the document: `### YYYY-MM-DD: Short title` plus bullet points (what, why, affected files / sections).
   Never modify or delete existing log entries, so the history is preserved.

Substantial changes include, among others:
- new or changed tasks, env configs, rewards, observations, terminations, curricula
- PPO hyperparameters
- changes to data formats or pipeline scripts
- meaningful new training results or checkpoints
- architecture decisions
- deployment and sim2real work

Not substantial: pure formatting, typos, comments.

## Key invariants (details in the spec)

- The observation slot contract spans Phase 1 / Phase 2 / Unified. Mirror every change in all tasks and in
  both groups (policy/critic).
- Ball and hand observations are in the robot's yaw frame, never in world coordinates.
- The action scale is `UNITREE_G1_29DOF_MIMIC_ACTION_SCALE` everywhere.

## Environment

- Python: `/home/marius/miniconda3/envs/env_isaaclab_downgrade/bin/python`
- Train: `./unitree_rl_lab.sh -t --task <Task>` · Play: `-p` · List: `-l`
