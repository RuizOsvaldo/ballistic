#!/bin/bash
# Ballistic daily predictions runner
# Called by cron at 8:00am every day

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$PROJECT_DIR/data/logs"
LOG_FILE="$LOG_DIR/daily_$(date +%Y-%m-%d).log"
SENT_FLAG="$LOG_DIR/sent_$(date +%Y-%m-%d).flag"

mkdir -p "$LOG_DIR"

# Prevent duplicate emails — exit if already sent today
if [ -f "$SENT_FLAG" ]; then
    echo "=== Skipping: email already sent today ($(date)) ===" >> "$LOG_FILE"
    exit 0
fi

cd "$PROJECT_DIR"

echo "=== Ballistic Daily Run $(date) ===" >> "$LOG_FILE"

"$PROJECT_DIR/venv/bin/python3" -m src.jobs.daily_predictions >> "$LOG_FILE" 2>&1

if [ $? -eq 0 ]; then
    touch "$SENT_FLAG"
    echo "=== Done $(date) ===" >> "$LOG_FILE"
else
    echo "=== FAILED $(date) — flag not set, will retry if run again ===" >> "$LOG_FILE"
fi
