# NODE2_PROMPT.md — node2의 살아있는 지시 파일 (node1이 갱신, node2가 실행)

**이 파일 하나만 계속 참조한다.** 사용자는 더 이상 프롬프트를 복사해 주지 않는다. node1(메인 세션)이
이 파일을 편집해 새 지시를 내리고, node2는 결과를 `NODE2_RESULTS.md`에 append한다.
두 노드는 같은 lustre 트리(`/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope`)를 공유하므로
파일 변경이 곧바로 보인다(`git pull` 불필요; 커밋/푸시는 node1이 한다).

마지막 갱신: **2026-08-31 14:05 KST (node1)** — §3 작업표 참조.

---

## 1. 운영 프로토콜 (항상 유효)

1. **감시**: 세션 시작 직후 아래 Monitor(영구)를 하나 띄워 이 파일이 바뀌면 깨어나도록 한다.
   ```bash
   # 이 파일의 해시가 바뀔 때마다 한 줄 출력 (Monitor tool, persistent: true, 120 s 폴링)
   f=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope/NODE2_PROMPT.md; last=$(md5sum $f|cut -c1-32)
   while true; do sleep 120; cur=$(md5sum $f|cut -c1-32); [ "$cur" != "$last" ] && { echo "NODE2_PROMPT.md updated $(date +%H:%M)"; last=$cur; }; done
   ```
   깨어나면 이 파일을 **전부** 다시 읽고 §3의 `[PENDING]` 항목을 위에서부터 실행한다.
2. **상태 태그는 node2가 직접 고친다**: `[PENDING]` → `[RUNNING node2 gpu<i> <시각>]` → `[DONE <PSNR>]` /
   `[FAILED <원인 한 줄>]`. 이 파일의 다른 부분은 고치지 않는다(§5 "node2 → node1" 절은 예외).
3. **GPU 규율**: 4장 전부 항상 바쁘게. 한 셀이 끝나면 즉시 다음 `[PENDING]`을 올린다. 락은
   `run_gobj.sh`가 `lact_nvs/outputs/.gpu_locks/node2_gpu<i>`로 스스로 잡고 지운다(`NODE=node2` 필수).
   `node1_gpu*` 락은 건드리지 않는다. 큐가 비고 GPU가 놀면 §5에 "IDLE <시각>"을 적는다.
4. **실행 방식**: 반드시 자신의 background Bash task(`run_in_background`) + `setsid nohup … < /dev/null &`로
   띄우고, 60 s 뒤 `outputs/<exp>/train.log`가 `Iter 0000200` 이상 진행하는지, `nvidia-smi`에 프로세스가
   보이는지 확인한다. 프로세스를 죽여야 하면 `pkill -f`/`ps|grep|kill` 복합 명령 금지(자기 자신을 죽인
   전례 다수) — PID를 먼저 나열하고 스크립트 파일로 kill한다(`lact_nvs/outputs/_smoke/kill_exp.sh <exp>`).
5. **결과 보고**: 셀이 끝나면(`outputs/<exp>/eval.json` 생성) 아래를 실행해 표 행을 얻고,
   `NODE2_RESULTS.md` 맨 아래 해당 wave 제목 아래에 **append**한다(수치만; 해석·dossier는 node1).
   ```bash
   cd /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope/lact_nvs
   PY=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/python
   $PY paired_eval.py outputs/gobj_base_s95/eval_v2.json outputs/<exp>/eval.json --md
   ```
   기준은 **항상 `gobj_base_s95/eval_v2.json`**(현재 test index로 재평가한 baseline, 499 scene).
   옛 `eval.json`들과 비교하지 않는다(scene set이 1개 어긋남). 추가 기준이 표에 적혀 있으면 그것도 붙인다.
6. **금지**: `lact_nvs/*.py` 코드 수정(버그를 찾으면 §5에 적고 node1이 고친다), `paper_overleaf/` 수정
   (FREEZE), seed 복제 실험 자의 실행, `BATCH_QUEUE.txt`/video 그리드 재실행.
7. **노드 리셋 후**: `bash /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/claude_portable/setup_node.sh` →
   §2 준비 → 진행 중이던 `[RUNNING]` 셀을 같은 명령으로 재실행(`run_gobj.sh`는 체크포인트에서 재개, 완료 셀은 skip).

## 2. 준비 (노드 리셋마다 1회; /tmp는 노드 로컬)
```bash
cd /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope/lact_nvs
PY=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm/bin/python
SRC=/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/dataset/gobjaverse_wai
$PY data_preprocess/reshard_gobjaverse.py --src $SRC --odir /tmp/gobj/test  --index /tmp/gobj/test_index.json  --split test  --workers 16
$PY data_preprocess/reshard_gobjaverse.py --src $SRC --odir /tmp/gobj/train --index /tmp/gobj/train_index.json --split train --workers 56
ls /tmp/gobj/train | wc -l   # 19500 기대 (test 500). 72코어에서 총 ≈2 분.
```
GT-depth 사이드 파일(oracle 셀용)은 lustre `dataset/gobj_depth_patch/{train,test}`에 이미 있다.
셀 실행 명령은 모두 `NODE=node2 setsid nohup ./run_gobj.sh <gpu> <exp> <config> 95 > outputs/<exp>.launch.log 2>&1 < /dev/null &`
형태이며 30k step + eval ≈ 2 h/B200. 배경·가설 설명은 `OBJ_ANALYSIS.md` §0/§4/§5 (읽기 권장, 5분).

## 3. 작업표 (위에서부터; 상태 태그는 node2가 갱신)

### wave 1 — gObjaverse camera embedding (2026-08-31)
| # | exp | config | 무엇인가 | 상태 |
|---|---|---|---|---|
| W1-1 | `gobj_attn_nope_s95` | `config/gobj_attn_nope.yaml` | 진단 상한: TTT 층을 LaCT 논문의 block-causal full attention으로 교체(같은 6L/d256, 같은 토큰), PE 없음 | [PENDING] |
| W1-2 | `gobj_attn_prope_s95` | `config/gobj_attn_prope.yaml` | 진단 상한: 위 + faithful PRoPE(q/k/v/o) | [PENDING] |
| W1-3 | `gobj_hrot_rotraw_s95` | `config/gobj_hrot_rotraw.yaml` | H4: rot_raw(+0.43) + hidden 주소공간에 직교 회전 작용 ("one matrix action per address space") | [PENDING] |
| W1-4 | `gobj_imgvo_himg_s95` | `config/gobj_imgvo_himg.yaml` | H10: imgvo(+0.39, 현재 최고) + hidden 사이트 image-coordinate rotary | [PENDING] |

W1-3은 `outputs/gobj_rot_raw_s95/eval_v2.json`, W1-4는 `outputs/gobj_imgvo_s95/eval_v2.json`을 추가 기준으로 붙인다.

### wave 2 백로그 — GPU가 비는 대로 순서대로 (node1이 wave-1 결과를 보고 순서를 바꿀 수 있음)
| # | exp | config | 무엇인가 | 상태 |
|---|---|---|---|---|
| W2-1 | `gobj_raygta_s95` | `config/gobj_raygta.yaml` | H6: 토큰별 ray-frame 회전을 q/k/v/o에 (image rope + 카메라 회전 transport의 행렬 융합) | [PENDING] |
| W2-2 | `gobj_rot_content_s95` | `config/gobj_rot_content.yaml` | H8-1: rot_raw 변환을 SwiGLU content 브랜치에만, gate는 plain q/k | [PENDING] |
| W2-3 | `gobj_camray_properaw_s95` | `config/gobj_camray_properaw.yaml` | H7: pose-free 토큰 + projective(translation 포함) transport | [PENDING] |
| W2-4 | `gobj_anchor_in_s95` | `config/gobj_anchor_in.yaml` | H3b: chord 위 고정 depth anchor 3개의 3D-point 위상(입력 사이트) | [PENDING] |
| W2-5 | `gobj_h_dpra42_s95` | `config/cam_h_dpra42.yaml` | H5: hidden rotary를 update-유도 경로에만(초기 readout 비회전); `gobj_hidden_s95/eval_v2.json` 추가 기준 | [PENDING] |
| W2-6 | `gobj_camray_hrot_s95` | `config/gobj_camray_hrot.yaml` | H7+H4: pose-free 토큰 + rot_raw + hidden 회전 작용 | [PENDING] |
| W2-7 | `gobj_shell_iso_in_s95` | `config/gobj_shell_iso_in.yaml` | H2 변형: chord sinc를 정20면체 6방향(등방 3D kernel)으로 | [PENDING] |

(node1이 돌리는 셀: `gobj_oracle_both_s95`, `gobj_shell_in_s95`, `gobj_shell_h_s95`, `gobj_camray_rotraw_s95` — 중복 실행 금지.)

## 4. 결과 형식 (`NODE2_RESULTS.md`에 append)
```
## gObjaverse wave 1 (node2, 2026-08-31)
| cell | PSNR | dPSNR (t, win%) | LPIPS | dLPIPS (t) | SSIM | dSSIM (t) |   ← paired_eval.py --md 출력 그대로
(추가 기준이 있으면 "vs rot_raw_s95:" 줄을 덧붙인다)
훈련 특이사항 한 줄(NaN/재시작/속도).
```

## 5. node2 → node1 (질문·블로커·IDLE 기록; node2가 씀, 최신이 아래)
- (비어 있음)

## 6. node1 → node2 메시지 로그 (최신이 아래)
- 2026-08-31 14:05: 파일 신설. wave 1 네 셀을 GPU 0–3에 즉시 올릴 것. 끝나는 대로 wave 2 백로그를 순서대로.
