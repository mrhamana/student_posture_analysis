#!/usr/bin/env bash
# ─── One-time setup: install all dependencies ─────────────────────────────────
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "==> Installing backend dependencies..."
cd "$ROOT/backend"
pip install -r requirements.txt

echo ""
echo "==> Installing model server dependencies..."
cd "$ROOT/model_server"
pip install -r requirements.txt

echo ""
echo "==> Installing frontend dependencies..."
cd "$ROOT/frontend"
npm install

echo ""
echo "Setup complete! Run ./run.sh to start all services."
