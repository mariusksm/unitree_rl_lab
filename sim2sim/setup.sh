#!/usr/bin/env bash
# Set up the sim2sim environment on a group workstation (or any Linux machine with uv, git and Python 3.10):
#   1. venv $SIM2SIM_VENV (default ~/envs/sim2sim) with the locked packages of sim2sim/requirements.txt. On the
#      workstations uv takes them from the shared cache /opt/uv-cache and hardlinks them, so nothing is stored twice.
#   2. the G1 MJCF and meshes of unitree_rl_mjlab at a pinned commit in $UNITREE_MJCF_DIR/g1 (default ~/unitree_mjcf),
#      next to ~/unitree_model (the USD of Isaac Lab). Only fetched if missing (sparse checkout, ~35 MB).
#   3. tools/check_env.py: versions, both MJCFs load, a short passive simulation (its hash shows identical conditions
#      on different machines).
# Re-running is safe: the venv is synced to the lock file again.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${SIM2SIM_VENV:-$HOME/envs/sim2sim}"
PYTHON="${SIM2SIM_PYTHON:-/usr/bin/python3.10}"
MJCF_DIR="${UNITREE_MJCF_DIR:-$HOME/unitree_mjcf}"
MJLAB_REPO=https://github.com/unitreerobotics/unitree_rl_mjlab.git
MJLAB_COMMIT=1425b15f73bd4095f0df53709d7c389c3eb9e790  # 2026-04-13
G1_PATH=src/assets/robots/unitree_g1/xmls

# workstations: shared uv cache, group-writable files
if [ -f /etc/profile.d/uv_setup.sh ]; then
    source /etc/profile.d/uv_setup.sh
fi

if [ ! -x "$VENV/bin/python" ]; then
    uv venv --python "$PYTHON" "$VENV"
fi
uv pip sync --python "$VENV/bin/python" "$HERE/requirements.txt"

if [ ! -f "$MJCF_DIR/g1/g1.xml" ]; then
    tmp="$(mktemp -d)"
    git -C "$tmp" init -q
    git -C "$tmp" remote add origin "$MJLAB_REPO"
    git -C "$tmp" config core.sparseCheckout true  # works with the git 2.34 of Ubuntu 22.04
    printf '/%s/\n/LICENCE\n' "$G1_PATH" > "$tmp/.git/info/sparse-checkout"
    git -C "$tmp" fetch -q --depth 1 --filter=blob:none origin "$MJLAB_COMMIT"
    git -C "$tmp" checkout -q FETCH_HEAD
    mkdir -p "$MJCF_DIR/g1"
    cp -r "$tmp/$G1_PATH/." "$MJCF_DIR/g1/"
    cp "$tmp/LICENCE" "$MJCF_DIR/g1/LICENCE"
    echo "$MJLAB_REPO $MJLAB_COMMIT $G1_PATH" > "$MJCF_DIR/g1/SOURCE"
    rm -rf "$tmp"
fi

"$VENV/bin/python" "$HERE/tools/check_env.py" --mjcf_dir "$MJCF_DIR/g1"
