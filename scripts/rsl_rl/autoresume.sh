#!/usr/bin/env bash
# Continue an RSL-RL training after the workstation came back up, whether it crashed, lost power or was rebooted.
#
# A cron job runs `autoresume.sh boot` after every boot. If a training is armed, it waits for the GPU, takes the newest
# checkpoint of that training and continues it with --resume in a tmux session: a new log folder with the suffix
# `_autoresume`, checkpoints keep counting the iterations, the step-based curricula continue at the checkpoint
# (train.py sets common_step_counter). At most `save_interval` (500) iterations are lost. A crash while a checkpoint is
# written leaves a truncated file: only checkpoints of full size that load are used, else the next older one.
#
#   autoresume.sh enable <task> <experiment> <run folder> [total iterations, 30000] [tmux session, unified]
#   autoresume.sh disable     # BEFORE stopping a training on purpose, otherwise the next boot starts it again
#   autoresume.sh status
#   autoresume.sh dry-run     # what `boot` would start now, without starting it
#   autoresume.sh boot        # what cron calls
#
# Install once per workstation (cron's @reboot needs no root; GPUs are reserved per person, so `boot` never starts next
# to someone else's GPU job and gives up after 30 min):
#   mkdir -p ~/bin && cp scripts/rsl_rl/autoresume.sh ~/bin/hreact-autoresume && chmod +x ~/bin/hreact-autoresume
#   (crontab -l 2>/dev/null; echo "@reboot $HOME/bin/hreact-autoresume boot") | crontab -
# Log: ~/tmp/autoresume.log; training output: ~/tmp/<session>_autoresume_train.log
set -uo pipefail

CONF="${HREACT_AUTORESUME_CONF:-$HOME/.config/hreact/autoresume.conf}"
LOG="$HOME/tmp/autoresume.log"
REPO="${HREACT_REPO:-$HOME/unitree_rl_lab}"
ENV_SETUP="source /opt/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab"
ENV_SETUP="$ENV_SETUP && source $HOME/envs/hreact/bin/activate"
export PATH="$PATH:/usr/local/bin:/usr/bin:/bin"  # cron starts with a minimal PATH

log() {
    mkdir -p "$(dirname "$LOG")"
    echo "$(date '+%F %T') $*" | tee -a "$LOG"
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

train_command() {  # $1 run folder, $2 iteration
    echo "$ENV_SETUP && cd $REPO && python scripts/rsl_rl/train.py --task $TASK --headless --resume --load_run $1" \
        "--checkpoint model_$2.pt --max_iterations $((TOTAL - $2)) --run_name autoresume" \
        "2>&1 | tee -a $HOME/tmp/${SESSION}_autoresume_train.log; exec bash"
}

case "${1:-}" in
    enable)
        [ $# -ge 4 ] || { echo "usage: $0 enable <task> <experiment> <run folder> [total] [session]"; exit 1; }
        [ -d "$REPO/logs/rsl_rl/$3/$4" ] || { echo "no run folder $REPO/logs/rsl_rl/$3/$4"; exit 1; }
        mkdir -p "$(dirname "$CONF")"
        printf 'TASK=%s\nEXPERIMENT=%s\nCHAIN="%s"\nTOTAL=%s\nSESSION=%s\n' \
            "$2" "$3" "$4" "${5:-30000}" "${6:-unified}" > "$CONF"
        log "armed: $2, $3/$4, $((${5:-30000})) iterations in total, tmux ${6:-unified}"
        ;;
    disable)
        rm -f "$CONF"
        log "disarmed"
        ;;
    status)
        if [ -f "$CONF" ]; then
            cat "$CONF"
            source "$CONF"
            echo "newest full-size checkpoint: $(candidates | head -n 1)"
        else
            echo "not armed"
        fi
        echo "cron: $(crontab -l 2>/dev/null | grep autoresume || echo 'no @reboot entry')"
        tail -n 5 "$LOG" 2>/dev/null
        ;;
    dry-run | boot)
        [ -f "$CONF" ] || { [ "$1" = boot ] || echo "not armed"; exit 0; }
        source "$CONF"
        if [ "$1" = boot ]; then
            log "boot: armed for $TASK ($EXPERIMENT: $CHAIN)"
            for _ in $(seq 60); do nvidia-smi > /dev/null 2>&1 && break; sleep 10; done  # driver up, at most 10 min
            nvidia-smi > /dev/null 2>&1 || { log "no GPU after 10 min, not resuming"; exit 1; }
            sleep 60  # let the system settle
            if pgrep -f "[t]rain.py --task $TASK" > /dev/null; then log "already running, nothing to do"; exit 0; fi
            for _ in $(seq 30); do
                [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break
                sleep 60
            done
            if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
                log "GPU busy after 30 min, not resuming"
                exit 1
            fi
        fi
        RUN=""
        while read -r IT RUN; do
            loads "$REPO/logs/rsl_rl/$EXPERIMENT/$RUN/model_$IT.pt" && break
            log "$RUN/model_$IT.pt does not load, trying the next older one"
            RUN=""
        done < <(candidates)
        [ -n "$RUN" ] || { log "no loadable checkpoint in $EXPERIMENT: $CHAIN, not resuming"; exit 1; }
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
        tmux has-session -t "$SESSION" 2> /dev/null && SESSION="${SESSION}_resume"
        marker="$HOME/tmp/.autoresume_started"
        touch "$marker"
        tmux new-session -d -s "$SESSION" "$(train_command "$RUN" "$IT")"
        log "resumed $RUN/model_$IT.pt in tmux $SESSION, $((TOTAL - IT)) iterations to go"
        for _ in $(seq 60); do  # remember the new log folder, so that the next boot resumes from there
            new=$(find "$REPO/logs/rsl_rl/$EXPERIMENT" -mindepth 1 -maxdepth 1 -type d -name "*_autoresume" \
                -newer "$marker" -printf "%f\n" | sort | tail -n 1)
            if [ -n "$new" ]; then
                sed -i "s/^CHAIN=\"\(.*\)\"$/CHAIN=\"\1 $new\"/" "$CONF"
                log "new log folder $new"
                exit 0
            fi
            sleep 10
        done
        log "no new log folder after 10 min, check tmux $SESSION"
        ;;
    *)
        sed -n '2,19p' "$0"
        exit 1
        ;;
esac
