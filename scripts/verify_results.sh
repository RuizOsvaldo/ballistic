#!/bin/bash
# Ballistic nightly results verification
# Runs at midnight PST (08:00 UTC) to reconcile yesterday's predictions

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$PROJECT_DIR/data/logs"
LOG_FILE="$LOG_DIR/verify_$(date +%Y-%m-%d).log"

mkdir -p "$LOG_DIR"

echo "=== Ballistic Results Verification $(date) ===" >> "$LOG_FILE"

"$PROJECT_DIR/venv/bin/python3" -m src.jobs.verify_results >> "$LOG_FILE" 2>&1

if [ $? -eq 0 ]; then
    echo "=== Done $(date) ===" >> "$LOG_FILE"
else
    echo "=== FAILED $(date) ===" >> "$LOG_FILE"
fi
