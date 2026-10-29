"""Does the numpy perception (``sim2sim/ball.py``) give the same ball terms as Isaac?

Replays the true states recorded by ``record_ball_obs.py`` through ``ball.Perception`` (one per env, reset at every
new throw, as the Isaac command) and compares ``ball_state``, ``ball_visible`` and ``ball_intercept`` with the
observations Isaac computed, plus the catch-box test. With a deterministic recording they must agree to float
precision; the coverage line shows how many steps were measured, coasted and lost.

    python sim2sim/tools/check_ball_obs.py ball_obs.npz
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ball import Perception  # noqa: E402  # isort: skip

TERMS = ("ball_state", "ball_visible", "ball_intercept")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recording")
    args = parser.parse_args()
    d = np.load(args.recording)
    cfg = json.loads(str(d["perception"]))
    steps, envs = d["reset"].shape
    print(f"{steps} steps x {envs} envs, deterministic: {bool(d['deterministic'])}, perception {cfg}")

    worst = {name: 0.0 for name in TERMS}
    zone_mismatch, compared = 0, 0
    measured = coasted = lost = 0
    for e in range(envs):
        perception = Perception(cfg, float(d["step_dt"]), np.random.default_rng(e))
        started = False
        for t in range(steps):
            if d["reset"][t, e]:
                perception.reset()
                started = True
            if not started:
                continue  # the perception state before the first recorded reset is unknown
            ball_pos, ball_vel = d["ball_pos"][t, e], d["ball_vel"][t, e]
            root_pos, root_quat = d["root_pos"][t, e], d["root_quat"][t, e]
            zone_mismatch += perception.in_zone(ball_pos, root_pos, root_quat) != bool(d["in_zone"][t, e])
            perception.update(
                bool(d["launched"][t, e]), float(d["time_since_throw"][t, e]), ball_pos, ball_vel, root_pos, root_quat
            )
            terms = perception.actor_terms()
            for name in TERMS:
                worst[name] = max(worst[name], float(np.abs(terms[name] - d[name][t, e]).max()))
            compared += 1
            if perception.visible:
                measured += perception.steps_since_measurement == 0
                coasted += perception.steps_since_measurement > 0
            else:
                lost += perception.steps_since_measurement < 1_000_000

    print(f"compared {compared} steps: measured {measured}, coasted {coasted}, lost after a measurement {lost}")
    print(f"catch box test differs in {zone_mismatch} steps")
    for name, err in worst.items():
        print(f"{name:15s} max |numpy - Isaac| = {err:.2e}")
    ok = all(err < 1e-3 for err in worst.values()) and zone_mismatch == 0
    print("OK: the port reproduces Isaac" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
