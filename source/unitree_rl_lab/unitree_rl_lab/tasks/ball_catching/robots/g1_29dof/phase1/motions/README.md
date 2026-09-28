# Catching Motion Files

Place the retargeted **catching** motion here as a single `.npz` file — Phase 1
(`ball_catch_env_cfg.py`) picks up the first `*.npz` in this directory automatically.
Until one exists, training falls back to the Gangnam-style placeholder and prints a
warning (the policy then imitates dancing, not catching).

## Pipeline (VICON Shogun → npz)

1. Record the catching motion with VICON Shogun and retarget it to the G1 with GMR
   (produces a `.pkl` with 53 joint columns; the hands sit between the arms:
   `0-11 legs | 12-14 waist | 15-21 left arm | 22-33 left hand | 34-40 right arm | 41-52 right hand`).
2. Convert to `.npz` (picks the 29 body joints, replays them through Isaac Sim; requires the
   Isaac Lab python environment). It may hang on shutdown after `[DONE]` — Ctrl+C is safe then:

   ```bash
   python scripts/mimic/pkl_to_npz.py -i <file_or_folder>.pkl --headless -o <npz_out>/
   ```

3. Check that the `.npz` contains the right joints (numpy only, prints `RESULT: OK`):

   ```bash
   python scripts/mimic/check_motion.py --pkl <file>.pkl --npz <npz_out>/<file>.npz
   ```

4. Verify it visually, then copy it into this directory:

   ```bash
   python scripts/mimic/replay_npz.py -f <npz_out>/<file>.npz
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
