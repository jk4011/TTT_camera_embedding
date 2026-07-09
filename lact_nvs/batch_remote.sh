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
URLFILE=outputs/REMOTE_SESSION_URL.txt
while true; do
  tmux kill-session -t rc 2>/dev/null
  tmux new-session -d -s rc -x 220 -y 50 \
    "CLAUDE_CONFIG_DIR=$PORTABLE/config $PORTABLE/bin/claude --remote-control ttt-batch"
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
