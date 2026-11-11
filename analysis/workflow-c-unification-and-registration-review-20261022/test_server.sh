#!/bin/bash
# Isolated Streamlit server for browser E2E (port 8512): own research store and manual-JSON dir, so tests never write
# into the user's research record.  Usage: test_server.sh start|stop
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PORT=8512
PY=/mnt/mib-dbia08_1/users/noa.ganeko/miniconda3/envs/cell-rg/bin/python
DIR="$(cd "$(dirname "$0")" && pwd)/e2e_state"
PIDF="$DIR/server.pid"
mkdir -p "$DIR"
case "${1:-}" in
  start)
    cd "$REPO"
    CELLREG_RESEARCH_MODE=1 CELLREG_RESEARCH_STORE="$DIR/research_e2e.sqlite" CELLREG_EXPLORER_MANUAL_DIR="$DIR/manual" \
      setsid "$PY" -m streamlit run app.py --server.port $PORT --server.address 127.0.0.1 --server.headless true \
      --server.fileWatcherType none --browser.gatherUsageStats false >> "$DIR/server.log" 2>&1 < /dev/null &
    echo $! > "$PIDF"
    for _ in $(seq 1 40); do sleep 1; curl -s "http://127.0.0.1:$PORT/_stcore/health" | grep -q ok && { echo "test server pid $(cat $PIDF)"; exit 0; }; done
    echo "not healthy"; exit 1 ;;
  stop)
    P="$(cat "$PIDF" 2>/dev/null || true)"
    if [ -n "$P" ] && ps -o cmd= -p "$P" | grep -q "server.port $PORT"; then kill "$P"; echo "stopped $P"; fi ;;
esac
