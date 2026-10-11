#!/usr/bin/env bash
# sync_to_jetson.sh - Lean incremental sync to Jetson Orin Nano
# Usage:
#   ./sync_to_jetson.sh              # Syncs code + production model weights (excludes all media/datasets)
#   ./sync_to_jetson.sh --code-only  # Ultra-fast: syncs python code & configs only (skips all .pt and .zip weights)

set -e

JETSON_IP="192.168.55.1"
JETSON_USER="philip"
REMOTE_DIR="/home/philip/Development/autonomous-drone-tracking"

EXTRA_EXCLUDES=()
if [ "$1" == "--code-only" ] || [ "$1" == "-c" ]; then
  echo "⚡ Running CODE-ONLY sync (skipping model weights)..."
  EXTRA_EXCLUDES+=(--exclude '*.pt' --exclude '*.zip' --exclude '*.bin' --exclude 'control/weights/' --exclude 'perception/weights/')
else
  echo "🚀 Syncing code & active models to Jetson (${JETSON_USER}@${JETSON_IP})..."
fi

rsync -avz --progress \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude 'venv/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.DS_Store' \
  --exclude 'runs/' \
  --exclude 'outputs/' \
  --exclude 'output/' \
  --exclude 'logs/' \
  --exclude 'docs/' \
  --exclude 'datasets/' \
  --exclude 'data/' \
  --exclude 'perception/raw_data/' \
  --exclude 'control/rendering/' \
  --exclude 'control/weights/*/checkpoints/' \
  --exclude 'control/weights/*/*/checkpoints/' \
  --exclude 'control/weights/*/eval_results/' \
  --exclude 'control/weights/recurrent_ppo/archived_*' \
  --exclude '*.mp4' \
  --exclude '*.avi' \
  --exclude '*.mov' \
  --exclude '*.mkv' \
  --exclude '*.jpg' \
  --exclude '*.jpeg' \
  --exclude '*.png' \
  --exclude '*.gif' \
  --exclude '*.BIN' \
  --exclude '*.ulg' \
  --exclude '*.tlog' \
  --exclude '*.engine' \
  "${EXTRA_EXCLUDES[@]}" \
  ./ ${JETSON_USER}@${JETSON_IP}:${REMOTE_DIR}/

echo "✅ Sync complete! Directory structure preserved."
