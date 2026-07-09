#!/bin/bash
# batch_with_claude.sh — batch_entry.sh + headless Claude analysis afterwards.
# Submit this via the Slurm web GUI when you want the batch job to not only run
# the experiment queue but also have Claude summarize results and update
# RESULTS_DOSSIER.md unattended.
#
# Requirements baked in: portable claude binary + auth live on lustre at
# claude_portable/ (node-local ~/.claude does not exist on batch nodes).
# If the batch node has no outbound network, the claude step fails gracefully
# and the experiment results are still on lustre for later analysis.
set -u
cd "$(dirname "$0")"
PORTABLE=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/claude_portable

# 1. run the experiment queue (idempotent; skips completed runs)
bash batch_entry.sh "${1:-BATCH_QUEUE.txt}"

# 2. headless Claude: read fresh eval.json files, append a dated results table
#    to RESULTS_DOSSIER.md. Keep the prompt narrow — unattended run.
CLAUDE_CONFIG_DIR="$PORTABLE/config" "$PORTABLE/bin/claude" -p \
  "You are resuming the TTT camera-embedding research (read CLAUDE.md and RESULTS_DOSSIER.md first).
   New eval results just finished in lact_nvs/outputs/*/eval.json. For every eval.json newer than
   the last dossier entry: compute per-variant mean +- std across seeds, then append a dated
   section to RESULTS_DOSSIER.md in the established table style with a 2-3 line reading of what
   the numbers mean against the established findings (F-series). Also update EXPERIMENT_QUEUE.md
   statuses. Commit both files to git with a concise message. Do not launch any training." \
  --dangerously-skip-permissions \
  --add-dir /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding \
  2>&1 | tail -20
echo "[batch_with_claude] done $(date)"
