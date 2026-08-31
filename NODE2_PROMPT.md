# NODE2_PROMPT.md — node2의 살아있는 지시 파일 (node1이 갱신, node2가 실행)

**이 파일 하나만 계속 참조한다.** 사용자는 더 이상 프롬프트를 복사해 주지 않는다. node1(메인 세션)이
이 파일을 편집해 새 지시를 내리고, node2는 결과를 `NODE2_RESULTS.md`에 append한다.
두 노드는 같은 lustre 트리(`/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope`)를 공유하므로
파일 변경이 곧바로 보인다(`git pull` 불필요; 커밋/푸시는 node1이 한다).

마지막 갱신: **2026-08-31 17:55 KST (node1)** — V2-0c rot_shell 추가(V2-1보다 먼저). — V2-0a foot_in, V2-0b shell_all 추가(V2-0보다 먼저). — V2-0 `gobjvi_shell_h_vo` 추가(맨 앞). — 사용자 결정: **vi가 주축**. wave 1-vi 다음은 wave 2-vi(V2-1…5); orbit 백로그는 [HOLD]. — §2 vi 데이터 추가, §3에 wave 1-vi 블록(wave 1 다음, 기존 백로그보다 먼저).

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
**두 번째 데이터 (2026-08-31 15:20 추가)** — RayRoPE의 vary-intrinsics 스크립트로 재렌더한 objaverse(`gobj_vi`,
24 views/object, 뷰별 FOV·거리 무작위; F69의 데이터). 사용자 요청으로 이 데이터가 논문용 주축이 된다.
```bash
$PY data_preprocess/reshard_rayrope_renders.py      # 인자 없음, /tmp/gobj_vi/{train,test} + index, ≈3 분
ls /tmp/gobj_vi/train | wc -l                        # 20000 기대 (test 500)
```
vi 셀은 `DATA=gobj_vi NODE=node2 setsid nohup ./run_gobj.sh <gpu> gobjvi_<name>_s95 <config> 95 …` 로 띄운다
(`DATA=gobj_vi`가 데이터 경로·min_frames 24·eval을 모두 바꾼다; exp 이름은 `gobjvi_` 접두). 기준은
`outputs/gobjvi_base_s95/eval_v2.json`(node1이 15:30경 생성; F69 base 21.98, 500 scene).
셀 실행 명령은 모두 `NODE=node2 setsid nohup ./run_gobj.sh <gpu> <exp> <config> 95 > outputs/<exp>.launch.log 2>&1 < /dev/null &`
형태이며 30k step + eval ≈ 2 h/B200. 배경·가설 설명은 `OBJ_ANALYSIS.md` §0/§4/§5 (읽기 권장, 5분).

## 3. 작업표 (위에서부터; 상태 태그는 node2가 갱신)

### wave 1 — gObjaverse camera embedding (2026-08-31)
| # | exp | config | 무엇인가 | 상태 |
|---|---|---|---|---|
| W1-1 | `gobj_attn_nope_s95` | `config/gobj_attn_nope.yaml` | 진단 상한: TTT 층을 LaCT 논문의 block-causal full attention으로 교체(같은 6L/d256, 같은 토큰), PE 없음 | [DONE 22.898 (+0.705)] |
| W1-2 | `gobj_attn_prope_s95` | `config/gobj_attn_prope.yaml` | 진단 상한: 위 + faithful PRoPE(q/k/v/o) | [DONE 23.630 (+1.437)] |
| W1-3 | `gobj_hrot_rotraw_s95` | `config/gobj_hrot_rotraw.yaml` | H4: rot_raw(+0.43) + hidden 주소공간에 직교 회전 작용 ("one matrix action per address space") | [DONE 22.603 (+0.410 vs base, -0.010 vs rot_raw)] |
| W1-4 | `gobj_imgvo_himg_s95` | `config/gobj_imgvo_himg.yaml` | H10: imgvo(+0.39, 현재 최고) + hidden 사이트 image-coordinate rotary | [DONE 22.529 (+0.336 vs base, -0.059 vs imgvo)] |

W1-3은 `outputs/gobj_rot_raw_s95/eval_v2.json`, W1-4는 `outputs/gobj_imgvo_s95/eval_v2.json`을 추가 기준으로 붙인다.

### wave 1-vi — 같은 네 가지를 RayRoPE 재렌더 데이터에서 (wave 1이 끝나는 GPU부터 **이것을 먼저**; 2026-08-31 15:20)
node1 wave-1 판정: chord-3D-point rotary(`shell_*`)가 orbit 데이터에서 입력 +0.38 / hidden +0.32로 Plücker
ladder(−0.41/−0.57)를 뒤집었다. vi 데이터에서도 같은지가 논문의 핵심 표가 된다.
| # | exp | config | 무엇인가 | 상태 |
|---|---|---|---|---|
| V1-1 | `gobjvi_shell_in_s95` | `config/gobj_shell_in.yaml` | H2 입력 사이트 chord rotary (`DATA=gobj_vi`) | [RUNNING node1 gpu0 15:40] — node1이 가져감, node2는 V1-2부터 |
| V1-2 | `gobjvi_shell_h_s95` | `config/gobj_shell_h.yaml` | H2 hidden 사이트 | [DONE 22.043 (+0.062 vs base, +0.162 vs hidden)] |
| V1-3 | `gobjvi_shell_both_s95` | `config/gobj_shell_both.yaml` | 입력+hidden chord | [DONE 22.356 (+0.375 vs base)] |
| V1-4 | `gobjvi_shell_vo_s95` | `config/gobj_shell_vo.yaml` | 입력 chord + 회전 v/o transport | [DONE 22.490 (+0.509 vs base)] |
| V1-5 | `gobjvi_rot_raw_s95` | `config/cam_rot_raw.yaml` | 대조: orbit 최고 행렬 셀을 vi에서 | [DONE 22.514 (+0.533 vs base)] |
| V1-6 | `gobjvi_imgvo_s95` | `config/cam_imgvo.yaml` | 대조: orbit 현 최고 imgvo를 vi에서 | [RUNNING node2 gpu0 17:26] |
기준: `gobjvi_base_s95/eval_v2.json` + (V1-1/2/3은) `gobjvi_input_s95/eval_v2.json`, `gobjvi_hidden_s95/eval_v2.json`.

### wave 2-vi — 가설 셀을 vi 데이터로 (wave 1-vi 다음; 2026-08-31 16:25, 사용자 결정: vi가 주축)
node1이 vi에서 `gobjvi_shell_in`, `gobjvi_raygta`, `gobjvi_anchor_in`, `gobjvi_prope_raw`를 맡는다(중복 금지). node2는 아래 순서.
| # | exp | config | 무엇인가 | 상태 |
|---|---|---|---|---|
| V2-0a | `gobjvi_foot_in_s95` | `config/gobj_foot_in.yaml` | 사용자 질문(17:15): 가장 단순한 3D 점 = ray의 focus point 최근접점 `x_c = o + t_c d` (적분 없음, 파라미터 0) — sinc 적분이 필요한지 확인 | [RUNNING node2 gpu1 17:42] |
| V2-0b | `gobjvi_shell_all_s95` | `config/gobj_shell_all.yaml` | 사용자 제안: chord 입력 + chord hidden + 회전 v/o 모두 (한 레시피 후보) | [RUNNING node2 gpu2 17:47] |
| V2-0 | `gobjvi_shell_h_vo_s95` | `config/gobj_shell_h_vo.yaml` | F74 후속: hidden chord + 회전 v/o transport (orbit에서 shell_in+vo가 +0.53으로 최고 → hidden 쪽 합성 확인) | [RUNNING node2 gpu3 17:41] |
| V2-0c | `gobjvi_rot_shell_s95` | `config/gobj_rot_shell.yaml` | F75 후속: rot_raw(행렬 주소 + carrier) **위에** chord 위상까지 (두 주소 변환 중첩; vi에서 chord가 rot_raw에 추가 이득을 주는지) | [PENDING] |
| V2-1 | `gobjvi_anchor_h_s95` | `config/gobj_anchor_h.yaml` | H3b: chord 위 고정 depth anchor 3개의 3D-point 위상, hidden 사이트 | [PENDING] |
| V2-2 | `gobjvi_shell_iso_in_s95` | `config/gobj_shell_iso_in.yaml` | H2 변형: chord sinc를 정20면체 6방향(등방 3D kernel)으로 | [PENDING] |
| V2-3 | `gobjvi_rot_content_s95` | `config/gobj_rot_content.yaml` | H8-1: rot_raw 변환을 SwiGLU content 브랜치에만 | [PENDING] |
| V2-4 | `gobjvi_h_dpra_s95` | `config/gobj_h_dpra.yaml` | H5: hidden Plücker 위상을 update-유도 경로에만; 기준 `gobjvi_hidden_s95/eval_v2.json` | [PENDING] |
| V2-5 | `gobjvi_camray_hrot_s95` | `config/gobj_camray_hrot.yaml` | H7+H4 (vi에서의 재확인용, 후순위) | [PENDING] |
모두 `DATA=gobj_vi`, 기준 `gobjvi_base_s95/eval_v2.json`(21.981). orbit 백로그(아래)는 vi 큐가 빈 뒤에만.

### (보류) orbit 백로그 — vi 큐가 완전히 빈 뒤에만, node1이 별도 지시할 때
| # | exp | config | 무엇인가 | 상태 |
|---|---|---|---|---|
| W2-0a | `gobj_anchor_in_s95` | `config/gobj_anchor_in.yaml` | H3b 입력 (orbit) | [HOLD] |
| W2-0b | `gobj_anchor_h_s95` | `config/gobj_anchor_h.yaml` | H3b hidden (orbit) | [HOLD] |
| W2-1 | `gobj_raygta_s95` | `config/gobj_raygta.yaml` | H6 (orbit) | [HOLD] |
| W2-2 | `gobj_rot_content_s95` | `config/gobj_rot_content.yaml` | H8-1 (orbit) | [HOLD] |

(node1이 돌리는 셀: `gobj_oracle_both_s95`, `gobj_shell_in_s95`, `gobj_shell_h_s95`, `gobj_camray_rotraw_s95` — 중복 실행 금지.)

## 4. 결과 형식 (`NODE2_RESULTS.md`에 append)
```
## gObjaverse wave 1 (node2, 2026-08-31)
| cell | PSNR | dPSNR (t, win%) | LPIPS | dLPIPS (t) | SSIM | dSSIM (t) |   ← paired_eval.py --md 출력 그대로
(추가 기준이 있으면 "vs rot_raw_s95:" 줄을 덧붙인다)
훈련 특이사항 한 줄(NaN/재시작/속도).
```

## 5. node2 → node1 (질문·블로커·IDLE 기록; node2가 씀, 최신이 아래)
- 2026-08-31 14:14 (node2): 세션 시작. /tmp/gobj 리샤딩 완료(train 19500 / test 500, 7.1 G).
  wave-1 4셀 GPU 0–3 정상 기동(Iter 200+ 확인, it/s: attn_nope 11.8 / attn_prope 9.6 / hrot_rotraw 16.2 / imgvo_himg 15.7).
- 2026-08-31 14:14 (node2, 질문/W2-5): `config/cam_h_dpra42.yaml`의 cam_mode는 `qk_rope_cam+h_dpra`로
  **input+hidden 양 사이트**다. 표에 적힌 추가 기준 `gobj_hidden_s95`는 hidden-only(h_pra)로 보이는데,
  사이트가 어긋난 비교다. 양 사이트 대조군(TTT-RoPE both, −0.89 셀)의 eval_v2.json 경로를 알려주면
  그것도 함께 붙이겠다. 답이 없으면 표에 적힌 대로 base + gobj_hidden_s95 두 기준만 쓴다.
- 2026-08-31 15:14 (node2): 15:20 지시 확인. `/tmp/gobj_vi` 리샤딩 완료(train 20000 / test 500, +3.6 G).
  V1-1…V1-6 여섯 config 모두 존재·프로토콜 일치(L6/d256/p16) 확인. wave-1 GPU가 비는 대로 V1부터 올린다.
  wave-1 현황 15:14: 16.0k–19.0k/30k, LPIPS 구간 진입 후 ~4.2 it/s, NaN 0 — 완료 예상 16:00–16:15 KST.
  (`gobjvi_{base,input,hidden}_s95/eval_v2.json`은 아직 없음 — node1 15:30 예정대로면 V1 종료 시점엔 문제 없다.)
- 2026-08-31 15:16 (node2): 15:42 경고 확인. wave-1 4셀은 아직 학습 중(17.4k–20.8k)이라 아직 죽지 않았다.
  GPU별 체인 `outputs/_node2/chain.sh`를 4개 띄워 자동 처리하도록 했다: (1) `model_0030000.pth` 생성 +
  train 프로세스 종료를 감지 → (2) 같은 run_gobj.sh를 eval-only로 재실행 → (3) 같은 GPU에 다음 셀 투입.
  락 파일이 아니라 체크포인트/프로세스로 판정하므로 죽은 wrapper가 락을 남겨도 안전하다.
  V1-1은 node1 것으로 두고 node2 배정: gpu0→V1-2 shell_h, gpu1→V1-3 shell_both, gpu2→V1-4 shell_vo,
  gpu3→V1-5 rot_raw. V1-6 imgvo는 다음으로 비는 GPU에 올린다.
- 2026-08-31 16:06 (node2): **wave 1 완료**, 4행 모두 `NODE2_RESULTS.md`에 append.
  attn_nope 22.898 (+0.705) / attn_prope 23.630 (+1.437) / hrot_rotraw 22.603 (+0.410 base, **−0.010 vs rot_raw**) /
  imgvo_himg 22.529 (+0.336 base, **−0.059 vs imgvo**). 체인 복구 정상 작동(eval-only 재실행 4/4 성공, GPU 공백 1–3분).
  현재 GPU 0–3 = V1-2 shell_h(15:47) / V1-3 shell_both(15:55) / V1-4 shell_vo(16:01) / V1-5 rot_raw(16:05), 모두 ~17 it/s로 기동 확인.
  다음 대기열은 V1-6 → W2-0a → W2-0b …로 잡고 있다. vi 결과를 보고 순서를 바꾸려면 18:00 KST 전에 §3을 갱신해 두면 반영된다.
- 2026-08-31 16:32 (node2): 16:25 지시(vi 주축, orbit [HOLD]) 반영. V2 config 5개 모두 존재·프로토콜 일치 확인,
  `gobj_camray_hrot.yaml`은 `gobj_hrot_rotraw.yaml`과 `input_raymap: camray` 한 줄만 다른 것도 확인(H7 부분 정상).
  vi 기준 3개 모두 생성됨: base 21.981 / input 22.187 / hidden 21.881.
  **선체인 완료** — GPU가 비는 즉시 다음이 자동으로 뜬다(`outputs/_node2/chain2.sh`):
  gpu0→V1-6 imgvo, gpu1→V2-1 anchor_h, gpu2→V2-2 shell_iso_in, gpu3→V2-3 rot_content.
  남은 V2-4 h_dpra, V2-5 camray_hrot는 그다음 비는 GPU 2장에 올린다(≈20:00).
  ⚠ 순서를 바꾸려면 **18:00 KST 전에** §3을 갱신해라 — 그 이후엔 이미 뜬 셀을 죽여야 반영된다(대기 중 체인은 죽여도 무해).
- 2026-08-31 16:53 (node2): 17:05 지시(V2-0 맨 앞) 반영 완료. 대기 중이던 gpu1–3 체인을 죽이고(학습 프로세스는 무영향,
  4셀 모두 계속 진행 중 확인) 새 순서로 다시 걸었다: gpu0→V1-6 imgvo, gpu1→**V2-0 shell_h_vo**, gpu2→V2-1 anchor_h,
  gpu3→V2-2 shell_iso_in. 밀려난 V2-3 rot_content는 다시 [PENDING](V2-4 h_dpra, V2-5 camray_hrot와 함께 ≈20:00 배정).
  `config/gobj_shell_h_vo.yaml` 확인: `h_shell+vo_rel`, num_freqs_hseg 84, L6/d256/p16 정상.
- 2026-08-31 17:22 (node2): 17:20 지시(V2-0a/0b 우선) 반영. 대기 체인 gpu1–3만 교체(학습 3셀 무영향).
  새 배정: gpu0→V1-6 imgvo, gpu1→**V2-0a foot_in**, gpu2→**V2-0b shell_all**, gpu3→V2-0 shell_h_vo.
  V2-1 anchor_h·V2-2 shell_iso_in은 다시 [PENDING](V2-3/4/5와 함께 다음 회차).
  config 확인: foot_in(`foot_in`, F_seg 42) / shell_all(`shell_sinc+h_shell+vo_rel`, F 42/84) 모두 L6/d256/p16,
  `foot_in`은 `lact_ttt_cam.py`의 known·seg_in_modes에 등록되어 있고 1304행에서 sinc 적분 경로를 명시적으로 제외한다(적분 없음 = 의도대로).
  V1-2 shell_h는 17:20경 30k 학습 완료, 현재 eval 중.

## 6. node1 → node2 메시지 로그 (최신이 아래)
- 2026-08-31 14:05: 파일 신설. wave 1 네 셀을 GPU 0–3에 즉시 올릴 것. 끝나는 대로 wave 2 백로그를 순서대로.
- 2026-08-31 15:20: wave-1 결과 요약(orbit): shell_in +0.377 (t=+21), shell_h +0.324 (t=+19), camray_rotraw +0.343
  (rot_raw 대비 −0.08 → pose-free 토큰 기각). 사용자 요청으로 RayRoPE 재렌더(vi)를 주축으로 추가: §2에 리샤드,
  §3에 wave 1-vi 블록. **wave 1이 끝나는 GPU부터 V1-1…V1-6을 먼저** 올리고, 그다음 orbit 백로그(W2-*).
- 2026-08-31 17:55: F75 — vi에서 rot_raw +0.533 ≈ shell_vo +0.509(동률), shell_both +0.375(vi에서는 사이트 합성), shell_in +0.247. V2-0c `gobjvi_rot_shell` 추가.
- 2026-08-31 17:20: V2-0a `gobjvi_foot_in`, V2-0b `gobjvi_shell_all` 추가 — V2-0 shell_h_vo보다 먼저 올릴 것 (둘 다 스모크 통과).
- 2026-08-31 17:05: F74 — orbit에서 shell_vo(chord 입력 + v/o transport) **22.725 = +0.53**, PE-only 최고; shell_both는 비합성(−0.08 vs shell_in);
  oracle_in은 shell_in +0.05뿐(oracle 이득은 타깃 depth). V2-0 `gobjvi_shell_h_vo` 추가.
- 2026-08-31 16:25: 사용자 결정으로 **vi 데이터가 주축**. V1 다음에 V2-1…V2-5(vi)를 올리고, orbit 백로그(W2-*)는 [HOLD] — node1이 따로 풀기 전에는 돌리지 말 것.
- 2026-08-31 15:55: F73 — oracle_both(GT depth 3D-point rotary) **24.274 = +2.08 dB (t=+49.8, 499/499 scene)**;
  shell_in +0.377 / shell_h +0.324. 위상 계열의 상한이 매우 높으므로 orbit 백로그 맨 앞에 anchor 셀을 넣었다(W2-0a/b).
- 2026-08-31 15:42 ⚠ **중요**: node1이 15:15에 `run_gobj.sh`를 제자리 편집하는 바람에, 그 전에 시작된 wrapper
  (너의 wave-1 네 개 포함)가 학습이 끝나는 순간 `unexpected EOF while looking for matching '"'` 로 죽고 **eval을
  건너뛴다** (체크포인트 `model_0030000.pth`는 정상 저장됨). 대처: 학습이 끝나면 **같은 run_gobj.sh 명령을 한 번 더
  실행**하라 — 학습은 skip하고 eval만 돈다(파일은 이제 `main()` 구조라 재발 안 함). `outputs/<exp>/eval.json`이
  생긴 것을 확인한 뒤에만 결과를 보고할 것. 또한 V1-1은 node1이 가져갔으니 node2는 V1-2부터 올려라.
- 2026-08-31 14:20 (답변/W2-5): 지적이 맞다. `cam_h_dpra42.yaml`은 양 사이트(F16 입력 + delta-hidden 42)라 비교가
  어긋난다. W2-5를 **hidden-only** 셀 `gobj_h_dpra_s95` / `config/gobj_h_dpra.yaml`(cam_mode `h_dpra`, F_h 42;
  스모크 통과)로 교체했다 — 기준은 base + `gobj_hidden_s95/eval_v2.json`. 양 사이트 대조군은 `outputs/gobj_both_s95/eval_v2.json`
  (21.305)에 있으니 필요하면 붙여도 된다. 기동 보고 고맙다; it/s 수치까지 좋다.
