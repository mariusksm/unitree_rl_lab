"""Catch rate per throw type from the per-throw records of ``eval_catch.py`` (``<out>_throws.csv``), as Markdown tables.

    python scripts/rsl_rl/eval_catch_throws.py E=eval/e_in_e/base_throws.csv D=eval/d_in_e/base_throws.csv
    python scripts/rsl_rl/eval_catch_throws.py --detail E=eval/e_in_e/base_throws.csv   # how the misses ended

Every throw is described relative to the robot at its launch: lateral offset of the aim point (+ = left), its height
above the ground and its distance in front of the pelvis, apex height, flight time, release direction off the heading
and release distance, horizontal ball speed, perception delay, whether the tracker hides the ball near the body, pushes
during the flight (launch to the planned arrival: the same window for caught and lost balls; pushes after the arrival
are not split, since a lost ball ends its throw early and so gets fewer of them) and the largest of them, pushes while
holding (confirmed catches only: did the hold survive them), the wait before the throw, how far the robot turned while
waiting, and whether the throw lies within the ranges of Advanced-A to -D (aim 0.15-0.40 m in front, -0.10..+0.17 m to
the side, 0.85-1.28 m high, apex 1.75-2.1 m; the aim gets 3 cm for the robot's drift while it waits).

Each cell of the main tables: throws in the bin and the share caught, one column per record file. With ``--detail`` a
table per file says how the throws of each bin ended: caught, ball lost without touching arms or torso (the filter
bodies of the ball's contact sensor), after touching them, after lying on a palm, after being held (before or after the
confirmation), fall. Throws without a launch (a fall while waiting) are in no bin. With about 3000 throws per file a bin
of 300 throws has a standard error of about two points on its catch rate.
"""

import csv
import math
import sys

INF = math.inf
AD_RANGES = {"aim_fwd_m": (0.15, 0.40), "aim_lat_m": (-0.10, 0.17), "aim_h_m": (0.85, 1.28), "apex_h_m": (1.75, 2.10)}
AD_TOL = {"aim_fwd_m": 0.03, "aim_lat_m": 0.03, "aim_h_m": 0.01, "apex_h_m": 0.01}
LOSSES = ("ball_dropped", "ball_missed")
FALLS = ("bad_orientation", "base_height")


def load(path: str) -> list[dict]:
    with open(path) as f:
        return [{k: (v if k == "outcome" else (float(v) if v != "" else None)) for k, v in row.items()}
                for row in csv.DictReader(f)]


def end(row: dict) -> str:
    """caught, fell, or how far a lost ball got (never touched, touched, palm, held)."""
    if row["outcome"] == "ball_caught":
        return "caught"
    if row["outcome"] in FALLS:
        return "fell"
    if row["outcome"] not in LOSSES:
        return "other"
    if row["confirmed"] or row["max_secured_s"] > 0:
        return "held"
    if row["palm"]:
        return "palm"
    return "touched" if row["touched"] else "never_touched"


def ranges(key: str, edges: list[float], unit: str = "", absolute: bool = False):
    """Bins [lo, hi) of ``key`` (of its absolute value with ``absolute``; say so in the split's title)."""
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo == -INF:
            label = f"< {hi:g}{unit}"
        elif hi == INF:
            label = f"≥ {lo:g}{unit}"
        else:
            label = f"{lo:g} to {hi:g}{unit}"

        def pred(r, lo=lo, hi=hi):
            v = r[key]
            if v is None:
                return False
            v = abs(v) if absolute else v
            return lo <= v < hi

        bins.append((label, pred))
    return bins


def counts(key: str, top: int):
    """Bins 0, 1, ..., top - 1 and ≥ top of an integer count."""
    bins = [(str(k), lambda r, k=k: r[key] is not None and round(r[key]) == k) for k in range(top)]
    bins.append((f"≥ {top}", lambda r: r[key] is not None and r[key] >= top))
    return bins


def values(key: str, labels: dict):
    return [(label, lambda r, v=v: r[key] is not None and round(r[key]) == v) for v, label in labels.items()]


def inside_ad(r: dict) -> bool:
    return all(r[k] is not None and lo - AD_TOL[k] <= r[k] <= hi + AD_TOL[k] for k, (lo, hi) in AD_RANGES.items())


SPLITS = [
    ("aim point to the side (m, + left)", ranges("aim_lat_m", [-INF, -0.30, -0.10, 0.17, 0.30, INF])),
    ("aim point height above the ground (m)", ranges("aim_h_m", [-INF, 0.85, 1.00, 1.15, 1.28, INF])),
    ("aim point in front of the pelvis (m)", ranges("aim_fwd_m", [-INF, 0.15, 0.25, 0.40, INF])),
    ("apex height (m)", ranges("apex_h_m", [-INF, 1.75, 2.10, 2.35, INF])),
    ("flight time (s)", ranges("flight_s", [0, 0.7, 0.8, 0.9, 1.0, INF])),
    ("release direction off the heading (deg, either side)",
     ranges("release_az_deg", [0, 15, 30, 50, 75, INF], "", True)),
    ("release distance (m)", ranges("release_dist_m", [0, 3, 4, INF])),
    ("horizontal ball speed (m/s)", ranges("speed_h_mps", [0, 2.5, 3.5, 4.5, INF])),
    ("perception delay (control steps)", values("obs_delay_steps", {k: str(k) for k in range(0, 8)})),
    ("ball hidden near the body by the tracker", values("hidden_in_zone", {0: "no", 1: "yes"})),
    ("pushes during the flight (launch to the planned arrival)", counts("pushes_flight", 2)),
    ("largest push during the flight (m/s)", ranges("push_max_flight_mps", [0, 0.01, 0.2, 0.4, INF])),
    ("pushes while holding (confirmed catches only)",
     [(name, lambda r, p=p: r["confirmed"] and p(r)) for name, p in counts("pushes_hold", 1)]),
    ("wait before the throw (s)", ranges("wait_s", [0, 2, 4, 6, INF])),
    ("robot turned while waiting (deg, either way)", ranges("turn_wait_deg", [0, 10, 30, INF], "", True)),
    ("within the throw ranges of Advanced-A to -D",
     [("yes", lambda r: r["launched"] and inside_ad(r)), ("no", lambda r: r["launched"] and not inside_ad(r))]),
]
ENDS = ("caught", "never_touched", "touched", "palm", "held", "fell")


def pct(k: int, n: int) -> str:
    return f"{100 * k / n:.1f} %" if n else "–"


def main():
    args = sys.argv[1:]
    detail = "--detail" in args
    files = [a.split("=", 1) for a in args if a != "--detail"]
    if not files:
        sys.exit(__doc__)
    data = {label: load(path) for label, path in files}
    for label, rows in data.items():
        launched = [r for r in rows if r["launched"]]
        caught = sum(end(r) == "caught" for r in rows)
        print(f"{label}: {len(rows)} throws with an outcome, {len(launched)} launched, caught {pct(caught, len(rows))}")
    for title, bins in SPLITS:
        print(f"\n### {title}\n")
        print("| bin | " + " | ".join(f"{label}: throws, caught" for label in data) + " |")
        print("|---|" + "---|" * len(data))
        for name, pred in bins:
            cells = []
            for rows in data.values():
                sel = [r for r in rows if r["launched"] and pred(r)]
                cells.append(f"{len(sel)}, {pct(sum(end(r) == 'caught' for r in sel), len(sel))}" if sel else "–")
            if any(c != "–" for c in cells):
                print(f"| {name} | " + " | ".join(cells) + " |")
        if detail:
            for label, rows in data.items():
                print(f"\n{label}: how the throws ended\n")
                print("| bin | throws | " + " | ".join(ENDS) + " |")
                print("|---|---|" + "---|" * len(ENDS))
                for name, pred in bins:
                    sel = [r for r in rows if r["launched"] and pred(r)]
                    if sel:
                        print(f"| {name} | {len(sel)} | "
                              + " | ".join(pct(sum(end(r) == e for r in sel), len(sel)) for e in ENDS) + " |")


if __name__ == "__main__":
    main()
