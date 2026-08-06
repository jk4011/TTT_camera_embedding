#!/bin/bash
# As each currently-running cell frees its GPU, that GPU becomes a band-sweep worker.
#   gpu4 <- Q39 gobj_gentle done      gpu6 <- Q39 gentle (re10k) done
#   gpu7 <- decomposition prope_raw done
# (gpu5 is taken by the camimg chain until ~06:30; it is NOT enrolled here.)
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
( until [ -f outputs/gobj_gentle_s95/eval.json ]; do sleep 240; done
  bash run_bandsweep.sh 4 ) &
( until [ -f outputs/gentle_s95/eval.json ]; do sleep 240; done
  bash run_bandsweep.sh 6 ) &
( until [ -f outputs/gobj_prope_raw_s95/eval.json ]; do sleep 240; done
  bash run_bandsweep.sh 7 ) &
wait
echo "[bandsweep chain] all workers done"
