#!/usr/bin/env bash
# Continue an RSL-RL training after it stopped unintentionally: after the workstation came back up (crash, power loss,
# reboot), after the training process alone crashed (2026-11-02: SIGBUS from a hardware memory error on workstation 2),
# or when it hangs.
#
# Two cron jobs:
# - `autoresume.sh boot` after every boot: if a training is armed, it waits for the GPU and continues the training.
# - `autoresume.sh watch` every 10 min: if the armed training is unfinished and its process is gone (at two checks in a
#   row, so a restart in progress is not started twice), or it runs but has written nothing to its log folder for
#   HANG_MIN minutes (then it is stopped first), it continues the training. Nothing happens during the first 15 min
#   after a boot, while `boot` is busy, or while someone else's job uses the GPU (it retries at the next check).
# Continuing means: the newest checkpoint of the training with --resume in a tmux session, a new log folder with the
# suffix `_autoresume`, checkpoints keep counting the iterations, the step-based curricula continue at the checkpoint
# (train.py sets common_step_counter). At most `save_interval` (500) iterations are lost. A crash while a checkpoint is
# written leaves a truncated file: only checkpoints of full size that load are used, else the next older one.
#
#   autoresume.sh enable <task> <experiment> <run folder> [total iterations, 30000] [tmux session, unified]
#   autoresume.sh disable     # BEFORE stopping a training on purpose, otherwise `watch` starts it again within 20 min
#   autoresume.sh status
#   autoresume.sh dry-run     # what `boot` would start now, without starting it
#   autoresume.sh boot        # what cron calls after a boot
#   autoresume.sh watch       # what cron calls every 10 min
#
# Install once per workstation (user crontab, no root; GPUs are reserved per person, so neither job starts next to
# someone else's GPU job):
#   mkdir -p ~/bin && cp scripts/rsl_rl/autoresume.sh ~/bin/hreact-autoresume && chmod +x ~/bin/hreact-autoresume
#   (crontab -l 2>/dev/null | grep -v hreact-autoresume; echo "@reboot $HOME/bin/hreact-autoresume boot";
#    echo "*/10 * * * * $HOME/bin/hreact-autoresume watch") | crontab -
# Log: ~/tmp/autoresume.log; training output: ~/tmp/<session>_autoresume_train.log
set -uo pipefail

CONF="${HREACT_AUTORESUME_CONF:-$HOME/.config/hreact/autoresume.conf}"
LOG="$HOME/tmp/autoresume.log"
LOCK="$HOME/tmp/.autoresume.lock"     # held by `boot` and `watch` while they check and resume
MISS="$HOME/tmp/.autoresume.missing"  # `watch` found no training process at its last check
LAST="$HOME/tmp/.autoresume.last"     # last `watch` message, so a waiting state is logged once, not every 10 min
HANG_MIN="${HREACT_AUTORESUME_HANG_MIN:-30}"
REPO="${HREACT_REPO:-$HOME/unitree_rl_lab}"
ENV_SETUP="source /opt/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab"
ENV_SETUP="$ENV_SETUP && source $HOME/envs/hreact/bin/activate"
export PATH="$PATH:/usr/local/bin:/usr/bin:/bin"  # cron starts with a minimal PATH

log() {
    mkdir -p "$(dirname "$LOG")"
    echo "$(date '+%F %T') $*" | tee -a "$LOG"
}

log_once() {  # log a `watch` message only if it differs from the last one
    [ "$(cat "$LAST" 2> /dev/null)" = "$*" ] && return 0
    echo "$*" > "$LAST"
    log "$*"
}

candidates() {  # "<iteration> <run folder>" of the full-size checkpoints of the armed chain, newest first
    local run f size max=0
    for run in $CHAIN; do
        for f in "$REPO/logs/rsl_rl/$EXPERIMENT/$run"/model_*.pt; do
            [ -e "$f" ] || continue
            size=$(stat -c %s "$f")
            [ "$size" -gt "$max" ] && max=$size
        done
    done
    for run in $CHAIN; do  # all checkpoints of a training have the same size; a smaller one is truncated
        for f in "$REPO/logs/rsl_rl/$EXPERIMENT/$run"/model_*.pt; do
            [ -e "$f" ] && [ "$(stat -c %s "$f")" -eq "$max" ] || continue
            f=$(basename "$f" .pt)
            echo "${f#model_} $run"
        done
    done | sort -rn
}

LOAD_CHECK='import sys, torch; torch.load(sys.argv[1], map_location="cpu", weights_only=True)'

loads() {  # the checkpoint file can be read by torch
    bash -c "$ENV_SETUP && python -c '$LOAD_CHECK' \"\$0\"" "$1" < /dev/null > /dev/null 2>&1
}

pick_checkpoint() {  # sets IT and RUN to the newest loadable full-size checkpoint; fails if there is none
    RUN=""
    while read -r IT RUN; do
        loads "$REPO/logs/rsl_rl/$EXPERIMENT/$RUN/model_$IT.pt" && return 0
        log "$RUN/model_$IT.pt does not load, trying the next older one"
        RUN=""
    done < <(candidates)
    return 1
}

training_pids() {  # the python process of the armed training; not the tmux server or the bash wrapper, whose command
    # lines contain the train.py command too (2026-11-02: `pgrep -f train.py` found the tmux server of a crashed run)
    pgrep -f "^[^ ]*python[0-9.]* [^ ]*train\.py --task $TASK( |\$)"
}

gpu_free() {
    [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2> /dev/null)" ]
}

writes_to_chain() {  # process $1 has a file of the newest chain folder open (its TensorBoard event file): only such a
    # process is stopped when it hangs, never a pilot or test of the same task that logs elsewhere
    ls -l "/proc/$1/fd" 2> /dev/null | grep -qF "$REPO/logs/rsl_rl/$EXPERIMENT/${CHAIN##* }/"
}

output_age() {  # seconds since the newest file in the log folder of the running training (last folder of the chain)
    local newest
    newest=$(find "$REPO/logs/rsl_rl/$EXPERIMENT/${CHAIN##* }" -type f -printf "%T@\n" 2> /dev/null | sort -n | tail -n 1)
    [ -n "$newest" ] || { echo 999999; return; }
    echo $(($(date +%s) - ${newest%.*}))
}

train_command() {  # $1 run folder, $2 iteration
    echo "$ENV_SETUP && cd $REPO && python scripts/rsl_rl/train.py --task $TASK --headless --resume --load_run $1" \
        "--checkpoint model_$2.pt --max_iterations $((TOTAL - $2)) --run_name autoresume" \
        "2>&1 | tee -a $HOME/tmp/${SESSION}_autoresume_train.log; exec bash"
}

start_resume() {  # $1 run folder, $2 iteration: start the training in tmux and append its new log folder to the chain
    local session="$SESSION" marker new
    # the session of the stopped training usually still exists (its shell stays open)
    tmux has-session -t "$session" 2> /dev/null && session="${SESSION}_resume_$(date +%m%d_%H%M)"
    marker="$HOME/tmp/.autoresume_started"
    touch "$marker"
    tmux new-session -d -s "$session" "$(train_command "$1" "$2")"
    log "resumed $1/model_$2.pt in tmux $session, $((TOTAL - $2)) iterations to go"
    for _ in $(seq 60); do  # remember the new log folder, so that the next resume starts from there
        new=$(find "$REPO/logs/rsl_rl/$EXPERIMENT" -mindepth 1 -maxdepth 1 -type d -name "*_autoresume" \
            -newer "$marker" -printf "%f\n" | sort | tail -n 1)
        if [ -n "$new" ]; then
            sed -i "s/^CHAIN=\"\(.*\)\"$/CHAIN=\"\1 $new\"/" "$CONF"
            log "new log folder $new"
            return 0
        fi
        sleep 10
    done
    log "no new log folder after 10 min, check tmux $session"
}

case "${1:-}" in
    enable)
        [ $# -ge 4 ] || { echo "usage: $0 enable <task> <experiment> <run folder> [total] [session]"; exit 1; }
        [ -d "$REPO/logs/rsl_rl/$3/$4" ] || { echo "no run folder $REPO/logs/rsl_rl/$3/$4"; exit 1; }
        mkdir -p "$(dirname "$CONF")"
        printf 'TASK=%s\nEXPERIMENT=%s\nCHAIN="%s"\nTOTAL=%s\nSESSION=%s\n' \
            "$2" "$3" "$4" "${5:-30000}" "${6:-unified}" > "$CONF"
        rm -f "$MISS" "$LAST"
        log "armed: $2, $3/$4, $((${5:-30000})) iterations in total, tmux ${6:-unified}"
        ;;
    disable)
        rm -f "$CONF" "$MISS" "$LAST"
        log "disarmed"
        ;;
    status)
        if [ -f "$CONF" ]; then
            cat "$CONF"
            source "$CONF"
            echo "newest full-size checkpoint: $(candidates | head -n 1)"
            echo "training process: $(training_pids | tr '\n' ' ')"
            echo "log folder last written $(($(output_age) / 60)) min ago"
        else
            echo "not armed"
        fi
        echo "cron: $(crontab -l 2>/dev/null | grep autoresume | tr '\n' ';' || echo 'no entries')"
        tail -n 5 "$LOG" 2>/dev/null
        ;;
    dry-run | boot)
        [ -f "$CONF" ] || { [ "$1" = boot ] || echo "not armed"; exit 0; }
        source "$CONF"
        if [ "$1" = boot ]; then
            mkdir -p "$(dirname "$LOCK")"
            exec 9> "$LOCK"
            flock 9  # a `watch` that is resuming right now finishes first
            log "boot: armed for $TASK ($EXPERIMENT: $CHAIN)"
            for _ in $(seq 60); do nvidia-smi > /dev/null 2>&1 && break; sleep 10; done  # driver up, at most 10 min
            nvidia-smi > /dev/null 2>&1 || { log "no GPU after 10 min, not resuming"; exit 1; }
            sleep 60  # let the system settle
            if [ -n "$(training_pids)" ]; then log "already running, nothing to do"; exit 0; fi
            for _ in $(seq 30); do
                gpu_free && break
                sleep 60
            done
            if ! gpu_free; then
                log "GPU busy after 30 min, not resuming (watch retries every 10 min)"
                exit 1
            fi
        fi
        pick_checkpoint || { log "no loadable checkpoint in $EXPERIMENT: $CHAIN, not resuming"; exit 1; }
        if [ "$IT" -ge $((TOTAL - 1)) ]; then
            [ "$1" = boot ] && rm -f "$CONF"
            log "training finished ($RUN/model_$IT.pt), disarmed"
            exit 0
        fi
        if [ "$1" = dry-run ]; then
            echo "would resume $RUN/model_$IT.pt, $((TOTAL - IT)) iterations to go, in tmux $SESSION:"
            train_command "$RUN" "$IT"
            exit 0
        fi
        start_resume "$RUN" "$IT"
        rm -f "$MISS" "$LAST"
        ;;
    watch)
        [ -f "$CONF" ] || exit 0
        source "$CONF"
        [ "$(cut -d. -f1 /proc/uptime)" -ge 900 ] || exit 0  # the first 15 min after a boot belong to `boot`
        mkdir -p "$(dirname "$LOCK")"
        exec 9> "$LOCK"
        flock -n 9 || exit 0  # `boot` or another `watch` is busy
        pids=$(training_pids)
        if [ -n "$pids" ]; then
            pid=$(echo "$pids" | head -n 1)
            running=$(ps -o etimes= -p "$pid" | tr -d ' ')
            age=$(output_age)
            if [ "${running:-0}" -lt $((HANG_MIN * 60)) ] || [ "$age" -lt $((HANG_MIN * 60)) ]; then
                rm -f "$MISS" "$LAST"
                exit 0  # running and writing (or started less than HANG_MIN ago)
            fi
            if ! writes_to_chain "$pid"; then
                log_once "watch: pid $pid ($TASK) runs without writing to ${CHAIN##* } for $((age / 60)) min; not the" \
                    "armed training or hung before logging, left alone"
                exit 0
            fi
            log "watch: training (pid $pid) has written nothing for $((age / 60)) min, stopping it"
            kill "$pid" 2> /dev/null
            sleep 30
            kill -9 "$pid" 2> /dev/null
            sleep 10
            touch "$MISS"  # if the GPU is not free yet, resume at the next check
        elif [ ! -f "$MISS" ]; then
            touch "$MISS"  # first check without a process: a restart may be in progress, decide at the next check
            exit 0
        fi
        pick_checkpoint || { log_once "watch: no loadable checkpoint in $EXPERIMENT: $CHAIN"; exit 1; }
        if [ "$IT" -ge $((TOTAL - 1)) ]; then
            rm -f "$CONF" "$MISS" "$LAST"
            log "watch: training finished ($RUN/model_$IT.pt), disarmed"
            exit 0
        fi
        if ! gpu_free; then
            log_once "watch: training not running, GPU busy with another job, retrying every 10 min"
            exit 0
        fi
        log "watch: training not running (crashed?), resuming"
        start_resume "$RUN" "$IT"
        rm -f "$MISS" "$LAST"
        ;;
    *)
        sed -n '2,30p' "$0"
        exit 1
        ;;
esac
