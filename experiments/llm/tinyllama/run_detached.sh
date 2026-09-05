#!/usr/bin/env bash
# Launch the full tinyllama chain fully detached, surviving SSH/terminal drops.
#
# Detaches the process from the shell session (setsid + nohup) so you can
# close the SSH connection and walk away from the machine; the chain keeps
# running and resumes on crash. Logs go to runs/chain.log.
#
# Usage:
#   ./run_detached.sh              # background the whole chain, print PID + log
#   tail -f runs/chain.log         # follow progress from any session
#   kill $(cat runs/chain.pid)     # stop the chain (SIGTERM; HF saves on handler)
#
# NOTE: for the A100 run, make sure this runs INSIDE the devcontainer (with
# GPU + workspace mounted) and that the directory is a real disk (bind mount),
# not a container-local fs — checkpoints/DONE must survive the host reboot.

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$DIR/runs"
PIDFILE="$DIR/runs/chain.pid"
LOGFILE="$DIR/runs/chain.log"

# Already running?
if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "chain already running (pid $(cat "$PIDFILE"), log: $LOGFILE)"
    exit 0
fi

# setsid detaches from the controlling terminal; nohup ignores SIGHUP so the
# process is not killed when the SSH session closes. All output -> chain.log.
setsid nohup "$DIR/run_chain.sh" >"$LOGFILE" 2>&1 &
pid=$!
echo "$pid" >"$PIDFILE"
echo "chain started in background."
echo "  pid : $pid"
echo "  log : $LOGFILE"
echo "detach freely; reattach with: tail -f $LOGFILE"
