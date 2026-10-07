#!/bin/bash
# run_fetch.sh — download + box overnight, logged, survives logout.
# Writes ONLY to dataset_new/ (the old dataset/ is left untouched; combine later).
# Runs the LOOP: download -> box -> recheck good counts -> top up species still
# under target, repeating until target is met or sources run dry.
# Usage:  ./run_fetch.sh        (foreground, see live)
#         nohup ./run_fetch.sh > /dev/null 2>&1 &   (detached, check the log)

cd "$(dirname "$0")"          # run from this script's folder
LOG="fetch_$(date +%Y%m%d_%H%M).log"

echo "=== FETCH START $(date) ===" | tee -a "$LOG"
python3 fetch_v2.py 2>&1 | tee -a "$LOG"
echo "=== FETCH DONE $(date) ===" | tee -a "$LOG"
