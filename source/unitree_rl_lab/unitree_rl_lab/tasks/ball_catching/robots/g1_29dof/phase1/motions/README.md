# Catching Motion Files

Place the retargeted **catching** motion here as a single `.npz` file — Phase 1
(`ball_catch_env_cfg.py`) picks up the first `*.npz` in this directory automatically.
Until one exists, training falls back to the Gangnam-style placeholder and prints a
warning (the policy then imitates dancing, not catching).

## Pipeline (VICON Shogun → npz)

1. Record the catching motion with VICON Shogun and retarget it to the G1 with GMR
   (produces a `.pkl` with 29 body + 24 hand joint columns).
2. Extract the 29 body joints and convert to the Phase-1 CSV layout
   (`base_xyz, quat_xyzw, joint_0..28`):

   ```bash
   python scripts/pkl_to_csv_without_hands.py -i pkl_result/ -o <csv_out>/
   ```

3. Replay through Isaac Sim to compute body kinematics and write the `.npz`
   (requires the Isaac Lab python environment):

   ```bash
   python scripts/mimic/csv_to_npz.py -f <csv_out>/<file>.csv --input_fps 30 --output_fps 50
   ```

4. Copy the resulting `.npz` into this directory and verify it visually:

   ```bash
   python scripts/mimic/replay_npz.py --motion_file <this_dir>/<file>.npz
   ```

## Quality checklist before training on it

- The clip should contain the full reach–catch–retract cycle, ideally with a short
  lead-in/lead-out of quiet standing so episode resets don't start mid-reach.
- During the catch window, check the **wrist heights** in the replay: they must be
  compatible with the Phase-2 catch zone (aim point ≈ 1.0–1.2 m world height,
  `catch_height_range = (0.25, 0.45)` above the root in `BallCommandCfg`). If the
  mocap catches at a different height, adjust `catch_height_range` to match the
  motion rather than the other way around.
- Feet should not slide in the replay; if they do, re-check the GMR retarget before
  burning GPU-hours on Phase 1.
