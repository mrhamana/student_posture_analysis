#!/usr/bin/env bash
# ─── Posture Analysis – Local Runner ───────────────────────────────────────────
# Starts all three services (model_server, backend, frontend) locally.
# Press Ctrl+C to stop everything.

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
PIDS=()

cleanup() {
    echo ""
    echo "Stopping all services..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null
    echo "All services stopped."
}
trap cleanup EXIT INT TERM

# ─── 1. Model Server (port 8001) ──────────────────────────────────────────────
echo "==> Starting Model Server on http://localhost:8001 ..."
cd "$ROOT/model_server"
python server.py &
PIDS+=($!)

# ─── 2. Backend API (port 8000) ────────────────────────────────────────────────
echo "==> Starting Backend API on http://localhost:8000 ..."
cd "$ROOT/backend"
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &
PIDS+=($!)

# ─── 3. Frontend Dev Server (port 5173) ───────────────────────────────────────
echo "==> Starting Frontend on http://localhost:5173 ..."
cd "$ROOT/frontend"
npm run dev &
PIDS+=($!)

echo ""
echo "All services started:"
echo "  Frontend  → http://localhost:5173"
echo "  Backend   → http://localhost:8000"
echo "  Model API → http://localhost:8001"
echo ""
echo "Press Ctrl+C to stop all services."

wait
