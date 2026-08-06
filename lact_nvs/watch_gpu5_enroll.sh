#!/bin/bash
# When the camimg chain releases gpu5 (~06:30), enroll it as a fourth sweep worker.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
until [ -f outputs/gobj_camimg_s95/eval.json ]; do sleep 300; done
bash run_bandsweep.sh 5
