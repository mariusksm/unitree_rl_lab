# Sim2Sim (MuJoCo)

Runs a policy trained in Isaac Lab unchanged in MuJoCo: the exported ONNX policy, the same robot, the same control loop (policy at 50 Hz, PD control on 5 ms physics steps) and the same actor observations, computed from the run's `params/deploy.yaml` and `params/env.yaml`. Only the physics engine changes, so a policy that leans on PhysX quirks shows up here before the real robot. Status and plan: spec Section 11 (item 9).

## Setup (group workstations)

```bash
bash sim2sim/setup.sh
```

- **Python environment** `~/envs/sim2sim`: system Python 3.10 with the locked packages of `requirements.txt` (MuJoCo 3.10.0 as in the shared conda env `gmr`, onnxruntime 1.22.0 as in the C++ deploy). uv takes them from the shared cache `/opt/uv-cache`. The shared environments `env_isaaclab` and `gmr` stay untouched.
- **Robot model** `~/unitree_mjcf/g1`: the G1 (29 DoF) MJCF and meshes of [unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab) at a pinned commit (Apache-2.0, `LICENCE` and `SOURCE` are copied along). `g1.xml` is the model mjlab trains with (collision capsules), `scene_g1.xml` Unitree's model for their C++ simulator (collision meshes, torque motors). Override the location with `UNITREE_MJCF_DIR`.
- **Check** `tools/check_env.py`: versions, both models load, a 1-s passive simulation. Its state hash is the same on both workstations (checked 2026-10-29 on W1 and W2), so results are comparable between them. `--render frame.png` tests headless rendering (EGL).

Change a package: edit `requirements.in`, lock again with the command at its top, run `setup.sh` on both workstations.

## Model comparison

`tools/dump_isaac_model.py` (Isaac Lab environment) writes the G1 as PhysX simulates it to JSON, `tools/compare_models.py` (sim2sim environment) compares it with MJCF files: masses, centres of mass, inertia, forward kinematics, collision shapes, joint ranges, armature and effort limits.

```bash
# Isaac Lab environment
python sim2sim/tools/dump_isaac_model.py --headless --device cpu --out isaac_g1.json
# sim2sim environment
~/envs/sim2sim/bin/python sim2sim/tools/compare_models.py isaac_g1.json ~/unitree_mjcf/g1/g1.xml ~/unitree_mjcf/g1/scene_g1.xml
```
