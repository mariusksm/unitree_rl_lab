"""Paired comparison of two evaluations on the same fixed throw list (``eval_catch.py --fixed_throws SEED`` with the same
task, scenario, options and ``--num_envs``): the per-throw records are matched by robot and throw index (columns ``env``
and ``throw_index``), so both checkpoints are compared on the same throws. The standard error of a difference then
comes only from the throws on which the two disagree, which is much smaller than for two independent runs.

    python scripts/rsl_rl/eval_catch_paired.py A=eval/a/base_throws.csv B=eval/b/base_throws.csv

Every robot works through its list at its own pace (a dropped ball ends an episode earlier than a caught one), so only
throws that both runs finished are compared; the last throws of a robot may be missing in one of them. As a check that
the lists match, the parameters a throw draws from the list (perception delay, ball hidden near the body, ball reported
before the throw, lost ball hidden) must agree on the matched throws; the script reports the share that does. Pushes,
tracker dropouts and measurement noise are not part of the list (``--no_push`` gives the tightest pairing).

Rows: caught; fell (before or after a loss); left the reference (Unified: a tracking termination, also in the window
after a loss); lost. For each: the rate in A and B on the matched throws, the difference B - A with its paired
standard error, the standard error two independent runs of that size would have, and how many throws only A or only B
counted (e.g. for "caught": caught by A but not by B, and the reverse).
"""

import csv
import math
import sys

FALLS = ("bad_orientation", "base_height")
TRACKING = ("anchor_pos", "anchor_ori", "ee_body_pos")
LOSSES = ("ball_dropped", "ball_missed")
CHECKED = ("obs_delay_steps", "hidden_in_zone", "shown_before", "lost_hidden")
METRICS = {
    "caught": lambda r: r["outcome"] == "ball_caught",
    "fell": lambda r: r["outcome"] in FALLS or r.get("post_loss_end") == "fell",
    "left the reference": lambda r: r["outcome"] in TRACKING or r.get("post_loss_end") == "left_reference",
    "lost": lambda r: r["outcome"] in LOSSES,
}


def load(path: str) -> dict:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if rows and ("env" not in rows[0] or "throw_index" not in rows[0]):
        sys.exit(f"{path}: no columns env and throw_index (per-throw records of eval_catch.py since 2026-11-24 needed)")
    return {(int(r["env"]), int(r["throw_index"])): r for r in rows}


def main():
    args = [a.split("=", 1) for a in sys.argv[1:]]
    if len(args) != 2 or any(len(a) != 2 for a in args):
        sys.exit(__doc__)
    (la, pa), (lb, pb) = args
    a, b = load(pa), load(pb)
    keys = sorted(set(a) & set(b))
    n = len(keys)
    print(f"{la}: {len(a)} throws, {lb}: {len(b)} throws, matched by robot and throw index: {n}")
    if n == 0:
        return
    agree = {c: [a[k][c] == b[k][c] for k in keys if a[k].get(c, "") != "" and b[k].get(c, "") != ""] for c in CHECKED}
    print("same throw parameters on the matched throws: "
          + ", ".join(f"{c} {100 * sum(v) / len(v):.1f} % of {len(v)}" for c, v in agree.items() if v))
    print()
    print(f"| | {la} | {lb} | {lb} - {la} (points) | paired SE | SE of independent runs | only {la} | only {lb} |")
    print("|---|---|---|---|---|---|---|---|")
    for name, f in METRICS.items():
        xa = [1.0 if f(a[k]) else 0.0 for k in keys]
        xb = [1.0 if f(b[k]) else 0.0 for k in keys]
        rate_a, rate_b = sum(xa) / n, sum(xb) / n
        d = [y - x for x, y in zip(xa, xb)]
        mean = sum(d) / n
        var = sum((v - mean) ** 2 for v in d) / max(1, n - 1)
        se_paired = math.sqrt(var / n)
        se_indep = math.sqrt((rate_a * (1 - rate_a) + rate_b * (1 - rate_b)) / n)
        only_a = sum(1 for x, y in zip(xa, xb) if x > y)
        only_b = sum(1 for x, y in zip(xa, xb) if y > x)
        print(f"| {name} | {100 * rate_a:.2f} % | {100 * rate_b:.2f} % | {100 * mean:+.2f} | {100 * se_paired:.2f} | "
              f"{100 * se_indep:.2f} | {only_a} | {only_b} |")


if __name__ == "__main__":
    main()
