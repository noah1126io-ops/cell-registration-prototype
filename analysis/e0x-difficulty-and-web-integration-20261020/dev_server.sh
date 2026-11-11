#!/bin/bash
# Registration Explorer development server (localhost only). Usage: dev_server.sh start|stop|status [port]
# Stops only the process recorded in the PID file, after checking that it is this app's Streamlit on this port.
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PORT="${2:-8511}"
PY="${CELLREG_PYTHON:-/mnt/mib-dbia08_1/users/noa.ganeko/miniconda3/envs/cell-rg/bin/python}"
DIR="$REPO/runs/registration-explorer-dev"
PIDF="$DIR/streamlit-$PORT.pid"
LOG="$DIR/streamlit-$PORT.log"
mkdir -p "$DIR"

is_ours() {  # pid -> 0 if it is our streamlit on PORT
  [ -n "${1:-}" ] && ps -o cmd= -p "$1" 2>/dev/null | grep -q "streamlit run app.py --server.port $PORT"
}

case "${1:-status}" in
  start)
    if [ -f "$PIDF" ] && is_ours "$(cat "$PIDF")"; then echo "already running (pid $(cat "$PIDF"))"; exit 0; fi
    if ss -ltn | awk '{print $4}' | grep -qE "[:.]$PORT\$"; then echo "port $PORT is in use by another process; not starting"; exit 1; fi
    cd "$REPO"
    CELLREG_RESEARCH_MODE=1 STREAMLIT_BROWSER_GATHER_USAGE_STATS=false setsid "$PY" -m streamlit run app.py --server.port "$PORT" \
      --server.address 127.0.0.1 --server.headless true --server.fileWatcherType none --browser.gatherUsageStats false \
      >> "$LOG" 2>&1 < /dev/null &
    echo $! > "$PIDF"
    for _ in $(seq 1 30); do
      sleep 1
      if curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/_stcore/health" | grep -q 200; then
        echo "started pid $(cat "$PIDF") on http://127.0.0.1:$PORT"; exit 0
      fi
    done
    echo "did not become healthy; see $LOG"; exit 1 ;;
  stop)
    P="$(cat "$PIDF" 2>/dev/null || true)"
    if is_ours "$P"; then kill "$P"; sleep 2; echo "stopped pid $P"; rm -f "$PIDF"; else echo "no recorded server for port $PORT (nothing stopped)"; fi ;;
  status)
    P="$(cat "$PIDF" 2>/dev/null || true)"
    if is_ours "$P"; then echo "running pid $P; health $(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/_stcore/health")"; else echo "not running"; fi ;;
esac
