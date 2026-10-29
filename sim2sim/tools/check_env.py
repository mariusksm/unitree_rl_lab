"""Check the sim2sim environment: package versions, the G1 MJCFs load, a short passive simulation, optionally a frame.

The hash of the simulated state matches between two machines only if both run the same MuJoCo build on the same
model with the same floating-point behaviour; compare it across the workstations to confirm identical conditions.

    python sim2sim/tools/check_env.py --mjcf_dir ~/unitree_mjcf/g1 [--render frame.png]
"""

import argparse
import hashlib
import os
import platform
import sys

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--mjcf_dir", type=str, required=True)
parser.add_argument("--render", type=str, default=None, help="Write one offscreen frame (EGL) of g1.xml to this PNG.")
args = parser.parse_args()
if args.render:
    os.environ.setdefault("MUJOCO_GL", "egl")  # headless rendering on the workstations

import imageio  # noqa: E402
import imageio_ffmpeg  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime  # noqa: E402
import yaml  # noqa: E402


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            return next(line.split(":", 1)[1].strip() for line in f if line.startswith("model name"))
    except (OSError, StopIteration):
        return platform.processor()


def with_floor(xml: str) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(xml)
    if not any(g.type == mujoco.mjtGeom.mjGEOM_PLANE for g in spec.worldbody.geoms):
        spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05])
    return spec.compile()


print(f"host {platform.node()}, {cpu_name()}, Python {platform.python_version()}")
print(
    f"mujoco {mujoco.__version__}, numpy {np.__version__}, onnxruntime {onnxruntime.__version__}, "
    f"pyyaml {yaml.__version__}, imageio {imageio.__version__}, ffmpeg {imageio_ffmpeg.get_ffmpeg_version()}"
)
source = os.path.join(args.mjcf_dir, "SOURCE")
if os.path.exists(source):
    print("model:", open(source).read().strip())

ok = True
for name in ("g1.xml", "scene_g1.xml"):
    try:
        m = with_floor(os.path.join(args.mjcf_dir, name))
    except Exception as e:  # report and go on with the other model
        print(f"{name}: FAILED to load: {e}")
        ok = False
        continue
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    for _ in range(round(1.0 / m.opt.timestep)):  # 1 s without actuation: the robot collapses (deterministic)
        mujoco.mj_step(m, d)
    digest = hashlib.sha256(d.qpos.tobytes()).hexdigest()[:16]
    mass = sum(m.body_mass[1:])
    print(
        f"{name}: {m.nbody - 1} bodies ({mass:.3f} kg), {m.njnt} joints, {m.nu} actuators, {m.ngeom} geoms, "
        f"step {m.opt.timestep * 1000:g} ms; "
        f"after 1 s passive: pelvis z {d.qpos[2]:.6f} m, state hash {digest}"
    )
    if args.render and name == "g1.xml":
        renderer = mujoco.Renderer(m, height=480, width=640)
        renderer.update_scene(d)
        imageio.imwrite(args.render, renderer.render())
        print(f"frame -> {args.render}")
sys.exit(0 if ok else 1)
