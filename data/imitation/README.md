# Reference motions (local copy, not in git)

Phase 1 and Unified look for their reference clips here first (`g1/*.npz`, ball recordings in `ball/*.npz`). The files are not committed: `*.npz` is gitignored, and the mocap data belongs to the project group's private dataset, while this repository is public.

Source: GitLab `pg-681-h-react/data/imitation-data` (folder `isaaclab/`). The `.npz` files there are stored with Git LFS, so after cloning run `git lfs pull`, otherwise the folders only contain small pointer files (the tasks detect this and stop with a hint).

```bash
git clone git@gitlab.fachschaften.org:pg-681-h-react/data/imitation-data.git
cd imitation-data && git lfs pull
cp -r isaaclab/g1 isaaclab/ball <this repo>/data/imitation/
```

Instead of copying, the dataset repository can also sit next to this one (`../imitation-data/isaaclab/`), or the folders can be set with `HREACT_MOTION_DIR` / `HREACT_BALL_DIR` (see `source/unitree_rl_lab/unitree_rl_lab/tasks/ball_catching/motion_data.py`).

Contents as of 2026-10-27 (dataset commit `e1704e4`): 108 clips of underhand catches at 50 Hz (1.9–4.6 s, BeyondMimic format, Isaac joint and body order of the G1 29-DoF asset) and 89 VICON recordings of the same takes at 120 Hz (ball trajectory, catch moment, an own retarget in the VICON frame). Check the data and the derived catch timing with `python scripts/mimic/catch_info.py`.
