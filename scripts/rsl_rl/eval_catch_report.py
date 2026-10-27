"""Comparison table of the eval_catch.py results in a directory (plain Python, no Isaac Sim).

    python scripts/rsl_rl/eval_catch_report.py <dir with <scenario>.json>
    python scripts/rsl_rl/eval_catch_report.py run_a/ run_b/      # several runs one after the other
"""

import glob
import json
import os
import sys

ORDER = ("base", "fric_high", "fric_low", "hold", "wait", "push", "lateral", "delay")


def row(name: str, r: dict) -> str:
    o = r["outcomes_pct"]
    fell = o.get("bad_orientation", 0) + o.get("base_height", 0)
    tracking = sum(v for k, v in o.items() if k in ("anchor_pos", "anchor_ori", "ee_body_pos"))
    caught = r.get("after_confirmed_catch_pct", {}).get("time_out", None) if name == "hold" else o.get("ball_caught", 0)
    caught_txt = f"{caught:6.1f}" if caught is not None else "     -"
    tilt = max(r["foot_tilt_L"]["mean_deg"], r["foot_tilt_R"]["mean_deg"])
    sat = r["torque_saturation"][0]
    drift = r.get("wait_drift_cm", {}).get("median", float("nan"))
    return (f"{name:10s} {r['episodes']:6d} {caught_txt} {o.get('ball_dropped', 0):6.1f} {fell:6.1f} {tracking:6.1f} "
            f"{tilt:6.1f} {r['feet_wait']['liftoffs_per_s']:6.2f} {drift:7.1f} "
            f"{r['held']['pct_torso_contact']:6.0f}  {sat['joint']} {sat['pct_steps_over_80pct']:.0f} %")


def main():
    dirs = sys.argv[1:] or ["."]
    for d in dirs:
        results = {}
        for path in glob.glob(os.path.join(d, "*.json")):
            with open(path) as f:
                r = json.load(f)
            if "scenario" in r and "outcomes_pct" in r:
                results[r["scenario"]] = r
        if not results:
            print(f"{d}: no eval_catch results")
            continue
        any_r = next(iter(results.values()))
        print(f"\n{d}  ({any_r.get('task')}, {os.path.basename(str(any_r.get('checkpoint')))})")
        print("scenario   episod caught  dropp   fell  track  tilt°  lift/s  driftcm torso%  most saturated joint (steps > 80 %)")
        for name in sorted(results, key=lambda n: ORDER.index(n) if n in ORDER else len(ORDER)):
            print(row(name, results[name]))
        print("caught: % of the episodes (hold: % of the confirmed catches still held at the end); fell: bad orientation "
              "or base height; track: tracking terminations (Unified); tilt: mean foot tilt in firm contact; lift/s: "
              "lift-offs per second while waiting; driftcm: median drift until the launch; torso%: share of the hold "
              "with ball-torso contact")


if __name__ == "__main__":
    main()
