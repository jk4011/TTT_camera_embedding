#!/bin/bash
# batch_remote.sh — interactive-Claude batch entry point. Submit via Slurm web GUI:
#
#   bash /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_nvs/batch_remote.sh
#
# Starts, on the batch node:
#   1. RE10K bootstrap (reshard into node-local /tmp if missing)
#   2. queue_daemon.sh  — keeps GPUs busy from BATCH_QUEUE.txt, re-read every
#      60 s, so appended experiments launch automatically
#   3. a Remote-Control Claude Code session inside tmux — chat with it from
#      https://claude.ai/code (session named "ttt-batch"); it runs ON this node
#      with full GPU/repo access, can add queue lines, launch/analyze runs.
#      The pairing URL is also written to outputs/REMOTE_SESSION_URL.txt.
# If the Claude session exits it is restarted (new URL appended to the file).
# Everything durable lives on lustre; killing/resubmitting this job is safe.
set -u
cd "$(dirname "$0")"
PY=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/python
PORTABLE=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/claude_portable
RE10K_SRC=/NHNHOME/WORKSPACE/26msit001_A/V-LAB/Datasets/re10k
WORKSPACE=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok

# Permission mode for the remote Claude session. The remote session is a FULL
# Claude Code instance (any command: downloads, clones, edits, new tasks).
#   default            — Claude asks before tools run; you approve from the
#                        claude.ai web/phone UI (safe, small friction)
#   acceptEdits        — file edits auto-approved, Bash still asks
#   bypassPermissions  — fully autonomous, no prompts (research-box mode)
PERMISSION_MODE=${PERMISSION_MODE:-acceptEdits}
mkdir -p outputs
exec > >(tee -a "outputs/batch_remote_$(date '+%F_%H%M%S').log") 2>&1
echo "[remote] $(date) node=$(hostname)"

# ---- 1. data bootstrap ----
if [ ! -f /tmp/re10k/train_index.json ]; then
  echo "[remote] resharding RE10K -> /tmp/re10k"
  mkdir -p /tmp/re10k
  $PY data_preprocess/reshard_re10k.py --src "$RE10K_SRC/test"  --odir /tmp/re10k/test  --index /tmp/re10k/test_index.json  --workers 32
  $PY data_preprocess/reshard_re10k.py --src "$RE10K_SRC/train" --odir /tmp/re10k/train --index /tmp/re10k/train_index.json --workers 32
fi

# ---- 2. queue daemon ----
nohup bash queue_daemon.sh > outputs/daemon.log 2>&1 &
DAEMON_PID=$!
echo "[remote] queue daemon pid $DAEMON_PID"

# ---- 3. remote-control Claude in tmux, restarted if it exits ----
# Transcripts live in $PORTABLE/config (lustre), so they survive job death.
# Which conversation to resume, in priority order:
#   1. $PORTABLE/RESUME_SESSION holding a session id -> `--resume <id>`
#      (deterministic; pin the exact conversation to carry across jobs)
#   2. any transcript for this repo -> `--continue` (most recent conversation)
#   3. none -> fresh session
# Resuming keeps the same claude.ai session URL (verified), so the web thread
# continues seamlessly across batch jobs.
# NOTE: Claude Code's project slug replaces EVERY non-alphanumeric char with
# '-' (underscores too) — deriving it with only / replaced silently misses the
# transcripts and starts a fresh session (bug found 2026-07-09).
REPO_ROOT="$(cd .. && pwd)"
SLUG=$(echo "$REPO_ROOT" | sed 's/[^A-Za-z0-9]/-/g')
PIN="$PORTABLE/RESUME_SESSION"
URLFILE=outputs/REMOTE_SESSION_URL.txt
while true; do
  RESUME=""
  if [ -s "$PIN" ]; then
    RESUME="--resume $(cat "$PIN")"
  elif ls "$PORTABLE/config/projects/${SLUG}"/*.jsonl >/dev/null 2>&1; then
    RESUME="--continue"
  fi
  tmux kill-session -t rc 2>/dev/null
  tmux new-session -d -s rc -x 220 -y 50 -c "$REPO_ROOT" \
    "CLAUDE_CONFIG_DIR=$PORTABLE/config $PORTABLE/bin/claude $RESUME --remote-control ttt-batch \
     --permission-mode $PERMISSION_MODE --add-dir $WORKSPACE"
  URL=""
  for _ in $(seq 1 24); do
    sleep 5
    URL=$(tmux capture-pane -t rc -p 2>/dev/null | grep -oE "https://claude.ai/code/session_[A-Za-z0-9]+" | head -1)
    [ -n "$URL" ] && break
  done
  echo "$(date '+%F %T')  ${URL:-<no url captured — check tmux session 'rc'>}" >> "$URLFILE"
  echo "[remote] claude session up: ${URL:-unknown} (also listed as 'ttt-batch' on claude.ai/code)"
  while tmux has-session -t rc 2>/dev/null; do sleep 60; done
  echo "[remote] $(date '+%F %T') claude session ended — restarting"
done
