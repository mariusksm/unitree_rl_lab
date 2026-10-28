#!/usr/bin/env bash
# Runs every stress-test scenario of eval_catch.py for one checkpoint (one Isaac process each), then prints the
# comparison table. Run it from the repository root in the Isaac Lab environment.
#
#   scripts/rsl_rl/eval_catch_all.sh <task> <checkpoint.pt> <out_dir> [more eval_catch.py arguments]
#   scripts/rsl_rl/eval_catch_all.sh Unitree-G1-29dof-BallCatch-Phase2 logs/rsl_rl/<exp>/<run>/model_29999.pt eval/run1
#
# Takes about 4 minutes per scenario with the defaults (512 robots, 60 s).
set -u
if [ $# -lt 3 ]; then
    sed -n '2,8p' "$0"
    exit 1
fi
TASK=$1
CKPT=$(realpath "$2")
OUT=$3
shift 3
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case "$TASK" in
    *Unified*) SCENARIOS="base fric_high fric_low hold wait push delay" ;;  # lateral: the clips set the aim
    *) SCENARIOS="base fric_high fric_low hold wait push lateral delay" ;;
esac
mkdir -p "$OUT"
for s in $SCENARIOS; do
    echo "[eval] $(date +%H:%M:%S) $s"
    rm -f "$OUT/$s.json"
    python "$HERE/eval_catch.py" --headless --task "$TASK" --checkpoint "$CKPT" --scenario "$s" \
        --out "$OUT/$s" "$@" > "$OUT/$s.log" 2>&1
    # Isaac Sim exits with 0 even after an error, so success means: the result file exists
    if [ -f "$OUT/$s.json" ]; then
        grep "^\[eval\]" "$OUT/$s.log" | head -1
    else
        echo "[eval] $s failed, see $OUT/$s.log"
    fi
done
python "$HERE/eval_catch_report.py" "$OUT"
