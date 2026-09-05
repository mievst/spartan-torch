#!/usr/bin/env bash
# TinyLlama stage chain driver: pretrain -> continue_pretrain -> cooldown.
#
# For each stage: if its DONE marker exists -> skip (already complete), else
# run train_stage.py with --resume-from auto (picks up the last checkpoint-XXX
# if one exists, so an interrupted run continues instead of restarting).
#
# Exit behaviour:
#   * stage returns 0 -> write DONE, move to next.
#   * stage crashes (nonzero/signal) -> exit nonzero, keep the last
#     checkpoint-XXX. Re-running the same command resumes from there.
#
# All output is tee'd to runs/<stage>.log so progress is visible from the host
# even in the background, and the stage's eval/checkpoint dots land in the
# checkpoint state as usual.

set -euo pipefail

STAGES=("pretrain" "continue_pretrain" "cooldown")

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS="$DIR/runs"
mkdir -p "$RUNS"

# Python to use. Defaults to the project venv (posix layout in the
# devcontainer); honour PYTHON if the caller sets one. Also allow a bare
# `python` fallback for a plain activated environment.
PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
    if [[ -x "$DIR/../../../.venv/bin/python" ]]; then
        PYTHON="$DIR/../../../.venv/bin/python"
    elif command -v python >/dev/null 2>&1; then
        PYTHON="python"
    else
        echo "[chain] ERROR: no python found. Set PYTHON=... explicitly." >&2
        exit 1
    fi
fi

# per-stage dirs mirror train_stage.py
declare -A STAGE_DIR=(
    [pretrain]="1.pretrain"
    [continue_pretrain]="2.continue_pretrain"
    [cooldown]="3.cooldown"
)

for stage in "${STAGES[@]}"; do
    dir="${STAGE_DIR[$stage]}"
    marker="$DIR/$dir/DONE"
    log="$RUNS/$stage.log"

    if [[ -f "$marker" ]]; then
        echo "[chain] $stage: DONE present, skipping."
        continue
    fi

    echo "[chain] ===> $stage (log: $log)"
    if "$PYTHON" "$DIR/train_stage.py" --stage "$stage" --resume-from auto 2>&1 | tee "$log"; then
        echo "[chain] $stage finished OK, writing DONE."
        touch "$marker"
    else
        echo "[chain] ERROR: $stage failed (see $log)." >&2
        echo "[chain] Re-run this script to resume from the last checkpoint." >&2
        exit 1
    fi
done

echo "[chain] all stages complete."
