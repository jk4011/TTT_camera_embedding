#!/usr/bin/env bash
# Append one NODE2_RESULTS.md line per finished Table4-lin cell, exactly once, then exit when all nine are in.
# Idempotent and durable: markers in outputs/.t4lin_reported/ mean a re-run after a node reset re-reports nothing.
#   Usage: nohup setsid bash report_table4_abs.sh >/dev/null 2>&1 &
main() {
  cd /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope/lact_nvs
  local PY=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/python
  local RES=../NODE2_RESULTS.md MARK=outputs/.t4lin_reported
  local CELLS="" d v
  for d in re10k gobj dl3dvu; do for v in both in h; do CELLS="$CELLS ${d}_dplin_pdir_${v}_s137"; done; done
  mkdir -p "$MARK"
  while :; do
    local left=0 c
    for c in $CELLS; do
      if [ -f "outputs/$c/eval.json" ]; then
        [ -f "$MARK/$c" ] && continue
        local line
        line=$($PY - "$c" <<'PYEOF'
import json, sys
c = sys.argv[1]
d = json.load(open(f"outputs/{c}/eval.json"))
print(f"- `{c}`  PSNR {d['psnr']:.3f}  SSIM {d['ssim']:.4f}  LPIPS {d['lpips']:.4f}  "
      f"(n={d['num_scenes']}, se {d['psnr_std_err']:.3f}, eval.json)")
PYEOF
) || { sleep 300; continue; }
        echo "$(date '+%F %H:%M') $line" >> "$RES"
        touch "$MARK/$c"
      else
        left=$((left+1))
      fi
    done
    [ "$left" = 0 ] && break
    sleep 300
  done
  echo "$(date '+%F %H:%M') → T4-lin 9셀 전부 완료." >> "$RES"
  echo "$(date '+%F %T') [table4] reporter: all nine reported" >> outputs/queue_table4_lin.log
}
main "$@"
