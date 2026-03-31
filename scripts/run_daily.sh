#!/bin/bash
# Ballistic daily predictions runner
# Called by cron at 8:00am every day

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$PROJECT_DIR/data/logs"
LOG_FILE="$LOG_DIR/daily_$(date +%Y-%m-%d).log"

mkdir -p "$LOG_DIR"

cd "$PROJECT_DIR"

echo "=== Ballistic Daily Run $(date) ===" >> "$LOG_FILE"

"$PROJECT_DIR/venv/bin/python3" -m src.jobs.daily_predictions >> "$LOG_FILE" 2>&1

echo "=== Done $(date) ===" >> "$LOG_FILE"
