"""Dump the env configs (training and play) of all ball-catching tasks to one JSON file, to compare two code versions,
for example before and after a refactor of the stage configs.

    python scripts/rsl_rl/dump_task_cfgs.py <out.json>        # with the code under test importable (PYTHONPATH)
    python scripts/rsl_rl/dump_task_cfgs.py --compare a.json b.json

Functions and classes are written as their dotted names, other objects as their repr.
"""

import json
import sys


def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if callable(x):
        return getattr(x, "__module__", "") + "." + getattr(x, "__name__", type(x).__name__)
    if isinstance(x, (int, float, str, bool)) or x is None:
        return x
    return repr(x)


def _diff(x, y, path=""):
    if isinstance(x, dict) and isinstance(y, dict):
        out = []
        for k in sorted(set(x) | set(y)):
            if k not in x:
                out.append(f"+ {path}/{k}")
            elif k not in y:
                out.append(f"- {path}/{k}")
            else:
                out += _diff(x[k], y[k], f"{path}/{k}")
        return out
    return [] if x == y else [f"~ {path}: {str(x)[:80]} -> {str(y)[:80]}"]


def compare(path_a: str, path_b: str):
    a, b = json.load(open(path_a)), json.load(open(path_b))
    for key in sorted(set(a) | set(b)):
        if key not in a or key not in b:
            print(f"{'only in ' + (path_b if key not in a else path_a):40s} {key}")
            continue
        lines = _diff(a[key], b[key])
        print(f"{'same' if not lines else f'{len(lines)} differences':40s} {key}")
        for line in lines[:20]:
            print("    ", line)


def dump(out_path: str):
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    import gymnasium as gym

    import unitree_rl_lab.tasks  # noqa: F401
    from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

    out = {}
    for task in sorted(t for t in gym.registry if "BallCatch" in t):
        for key in ("env_cfg_entry_point", "play_env_cfg_entry_point"):
            try:
                cfg = parse_env_cfg(task, device="cuda:0", num_envs=64, entry_point_key=key)
                out[f"{task}|{key}"] = _clean(cfg.to_dict())
            except Exception as e:  # noqa: BLE001
                out[f"{task}|{key}"] = f"ERROR {type(e).__name__}: {e}"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print(f"[dump] {len(out)} configs written to {out_path}", flush=True)
    app.close()


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--compare":
        compare(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 2:
        dump(sys.argv[1])
    else:
        sys.exit(__doc__)
