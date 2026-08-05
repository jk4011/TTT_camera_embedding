# NODE2_RESULTS.md — 노드2 실행 결과 (append-only)

node2가 완료한 태스크의 수치만 기록한다. 해석·다음 실험 설계는 node1이 RESULTS_DOSSIER.md /
lact_llm/ga_honly/LEDGER.md에 반영한다.

형식:
| 날짜 | 태스크 | 설정 | 핵심 수치 | 비고 |
|---|---|---|---|---|

## P0. DNA 데이터 경로 구현 + sanity (2026-08-03)
- 구현 파일: `lact_llm/dna_data.py` (hg38 char-LM 데이터 경로: DnaCharTokenizer vocab8,
  DnaBlockStream = data_seed-셔플 연속 4096블록 + state()/restore() 정확재개,
  get_or_build_dna_val_set = chr20 첫 488블록), `lact_llm/sanity_dna.py` (스트림 sanity),
  `lact_llm/train_small.py` (`--data {fineweb,dna}` 추가, fineweb 경로 불변; `data`를
  resume-critical arg에 추가), `lact_llm/self_heal.sh` (8회/60초 재시도 래퍼).
- 전처리 산출물(비커밋): `datasets/dna/hg38_train_blocks_4096.npy` (695,152블록×4096, 2.85GB,
  chr1-22+chrX−chr20, N>50% 블록 제거), `lact_llm/val_cache_hg38_4096.pt` (488×4096, chr20).
- **Sanity 3종 전부 PASS**:
  - (a) fineweb 무회귀: step1-100 loss 9.99→7.20, 누적평균 ≈8.44 (기존 참조 step100≈8.4 일치), VAL ppl 1205.5.
  - (b) dna: loss ln(8)=2.08 근방 시작 → step40 1.34, NaN 없음, VAL ppl 3.81(step30).
  - (c) resume 정확: 스트림 해시 일치 + epoch-wrap 해시 일치 + 트레이너 crash-resume
    (`resumed from ckpt_step20 step=20`, tokens_seen 끊김없이 이어짐).

## P0-W2. seq_len 파라미터화 + 32k 실현가능성 실측 (2026-08-03)

**구현**: `dna_data.py` seq_len 파라미터화 (`train_blocks_path(S)`/`val_cache_path(S)`,
`ensure_train_blocks(S)`, `get_or_build_dna_val_set(n, path, S)`; 4096 별칭 유지),
`train_small.py` dna 분기가 `args.seq_len`으로 블록/val 캐시 선택, `sanity_dna.py` seq_len 인자화,
`bench_seqlen.sh` 신규(실측 하네스). **WAVE 1 4096 산출물 그대로 보존·재사용**, 4096 경로 회귀 없음
(sanity_dna.py 4096 ALL PASS).
**32k 산출물**: train 86,911블록 × 32768 (2.8GB), val **61 × 32768 = 1,998,848 토큰** (WAVE 1과 동일).
64k(21,763×2)·128k 블록도 실측용으로 생성.

**실현가능성 실측** (`--data dna --bs 1 --window_size 128`, val은 기본 `--val_bs 8`, B200 1장):

| seq_len | tok/s (정상상태) | peak GPU MiB | OOM | val 1회 소요 |
|---|---|---|---|---|
| 4,096 (대조) | 30,384 | 8,738 | no | 26.0 s |
| **32,768** | **29,275–40,000** | **51,604** | no | 139.1 s |
| 65,536 | 30,092 | 100,222 | no | 253.1 s |
| 131,072 | — (12분간 1스텝도 미완료) | 157,466 | 클린 OOM 아님, **사실상 정지**(util 0%) | 도달 못함 |

**핵심 실측 결론 (수치만, 판단은 node1)**:
1. **tok/s는 seq_len과 무관하게 ~30k로 일정**. WAVE 1(seq 4096 **bs 8**)의 ~195,000 tok/s 대비
   6.5배 저하는 **전적으로 `bs 1` 효과**이고 시퀀스 길이 탓이 아니다 — seq 4096을 bs 1로 돌린
   대조군이 30,384 tok/s로 32k와 같기 때문. (어텐션은 flash_attn 2.8.3 sliding window라 O(s·w).)
2. 따라서 **WAVE 2 1런 예상 ≈ 27.8h** (3B ÷ 30k tok/s), + val 91회 × 139s ≈ 3.5h → **총 ≈ 31h**.
   WAVE 1은 4.5h였다. 4런 동시라 벽시계도 ≈31h.
3. 메모리는 seq_len에 선형(8.7/51.6/100.2 GB). 128k는 157 GB에서 진행 불가 → bs 1로도 상한.
4. `--actckpt` 류 CLI 옵션은 `train_small.py`에 **없음**. 모델은 `supports_gradient_checkpointing
   = True`(modeling_lact.py:163)라 배선하면 쓸 수 있으나, 트레이너 변경이라 하지 않았다.
5. **`max_position_embeddings` 변경 불필요**(큐가 확인 요청한 항목): 코드가
   `max_seqlen = max(q_len, mpe)`로 하한만 잡아 32768 > 4096이라 무영향이고, hidden rope는
   `h_inv_freq × pos`를 forward마다 즉석 계산(사전 테이블 없음). 기본값 4096 그대로 32k 정상 동작
   확인 → **T5-T8은 큐 커맨드 원문 그대로 실행, 아무것도 바꾸지 않음**.
6. 측정 편차 기록: 128k만 5스텝으로 계획(스텝당 시간 과다 예상) — 실제로는 1스텝도 완료 못함.

**Sanity (통과)**: 32k 20스텝 loss 1.7048 → 1.3565 (ln(8)=2.08 근방 시작, 감소, NaN 없음).
재개 정확성 — 스트림 해시 일치 + epoch-wrap 일치(4096·32768 양쪽 ALL PASS) + 트레이너
crash-resume: ckpt_step308에서 재개 후 step 310/312 loss가 중단 전과 **완전 동일**(1.3031/1.3113).


## 논문 표 채우기 (NODE2_PROMPT.md, 2026-08-05) — P1/P3/P5

### P1. NVS NoPE baseline `base_s95`  [DONE]
스톡 `lact_l6_d256_p16` (cam.mode 없음 = `lact_ttt.FastWeightGluMLPMultihead`), seed 95,
30k iters, bs16, lr 1e-4, LPIPS 5k부터, 8 input + 8 target, RE10K 256x256. 프로토콜 무편차.
훈련 EXIT 0, `model_0030000.pth` 생성. Eval: 256 held-out scenes, 8 uniform in / 4 midpoint target.

**base_s95: PSNR 21.8252 (SE 0.1423) / LPIPS 0.2874** (n=256)

seed 95로 전 arm이 일치하므로 **per-scene paired** (arm − base):

| arm | PSNR | ΔPSNR | t | win% | LPIPS | ΔLPIPS | t | win% |
|---|---|---|---|---|---|---|---|---|
| pra_hi_s95 | 22.3332 | +0.5080 | +20.65 | 89.8 | 0.2751 | −0.0123 | −20.92 | 93.4 |
| h_pra_hi_s95 | 22.7239 | +0.8987 | +41.14 | 99.6 | 0.2661 | −0.0213 | −41.53 | 99.6 |
| pra_h_hi_s95 | 22.7966 | +0.9715 | +31.70 | 98.0 | 0.2685 | −0.0189 | −27.09 | 97.3 |

(LPIPS는 낮을수록 좋음: win%는 base 대비 낮아진 scene 비율.) 단일 시드 결과이므로
F18 기준(~0.1-0.3 dB는 init 노이즈)에 비추어 arm 간 서열이 아니라 base 대비 크기만 읽을 것.

**Figure 1 view sweep: 네 arm 6/6 완성.** `ARM_RUN`에 `base`(`base_s95`) 추가.
base 곡선: v2 18.0641 / v4 20.6474 / v8 21.8252 / v16 22.0309 / v24 21.9368 / v32 21.9314.

**스크립트 버그 수정(node1 Figure 1을 막고 있던 원인)**: `run_fig1_viewsweep.sh`가
`launch_exp.sh`와 달리 triton/inductor 캐시 env를 export하지 않아, 기본 `/tmp/torchinductor_*`
(noexec tmpfs)에서 `ImportError: ... failed to map segment from shared object`로 죽었다.
이 때문에 `pra_hi` v=2/v=4가 실패해 있었다(114분 방치). 캐시 env를 추가하고 두 점을 재실행해
채웠다: pra_hi v2 17.8663 / v4 20.6609. 이제 네 arm 모두 6/6.

### P3. Video input-only + Both  [BLOCKED — node1 판단 필요]
두 가지 선행 문제를 확인했다(둘 다 실행 전 단계, 아직 아무것도 돌리지 않음).
1. **`abl_video_*.yaml` 경로가 리셋 후 미마이그레이션**: `output_path`/`data_root`가 죽은
   `26msit001_T_B/POSTECH-CGLAB/...`을, 체크포인트가 노드-로컬 `/tmp/wan_ckpt`(리셋으로 소실)를
   가리킨다. 세 경로 모두 부재 확인. `abl_ccv_*.yaml`은 `26msit001_A/jinhyeok/datasets/...`로
   갱신돼 있어, video config만 빠진 것으로 보인다.
2. **F22 대조군의 per-step 로그가 소실**: F21/F22는 데이터 순서+deterministic noise를 공유하는
   *paired per-step* 비교인데, 평문 video 런 로그·체크포인트가 리셋에서 전부 사라졌다
   (`lact_ar_video/outputs/`에 `ccv_*`만 생존). 게다가 데이터가 리셋 후 재구축돼(새 clip index)
   설령 로그가 남았어도 스텝 단위 페어링이 성립하지 않는다.
   → 지시대로 2런(input-only, Both)만 돌리면 **기존 base/hidden 셀과 짝지을 수 없다**.
   비교 가능한 Table 6을 만들려면 base(및 h_pra)까지 같은 재구축 데이터로 재실행해야 하므로
   2런이 아니라 3-4런이 된다. 프로토콜 변경이라 node2가 임의 결정하지 않는다.

플래그 자체는 확인됨: `ARFastWeightSwiGLU`에 `ttt_input_rope`(입력 사이트)와
`ttt_hidden_rope`(hidden 사이트)가 있어 input-only = `ttt_input_rope: true`,
Both = 둘 다 true. `abl_video_full`은 `ttt_hidden_rope+ttt_learnable_freqs`라 Both가 아님(명명 함정 확인).

### P5. CCV site ablation (3셀)  [RESTARTED 2026-08-05 at ttt_hrope_frac=1.0]
**재시작 사유(NODE2_PROMPT_RESTART_CCV.md)**: 최초 투입분은 `ttt_hrope_frac`을 기본값 0.5로
두어 hidden이 50%만 회전했다(입력은 98.4%). NVS는 98.4/98.4, 3D 재구성은 100/100으로 재실행
중이므로, CCV만 half-width 사다리면 input-vs-hidden 차이가 **사이트 때문인지 사다리 폭 때문인지
구분되지 않는다**. `ttt_hrope_frac: 1.0` 한 줄만 바꿔 재투입(다른 파라미터 불변).

폐기분은 삭제하지 않고 `outputs/ccv_site_{in,h,both}_frac05_ABANDONED`로 **이름만 비켜 뒀다**.
이유: 같은 exp_name으로 재실행하면 `find_latest_checkpoint`가 frac0.5 체크포인트를 물어
오염된 상태로 auto-resume 된다. 재시작 후 세 셀 모두 **step 1부터** 시작함을 로그로 확인.

런타임 config 덤프 기준 positive 확인(3셀 공통: use_cam_encoder=True, ttt_hrope_frac=1.0,
ttt_learnable_freqs=False, cam_phase_mode=plucker, max_fwdbwd_passes=20000, save_every=250,
keep_last_iter=1000, seed=1, deterministic_noise=True):
`ccv_site_in` input=True/hidden=False · `ccv_site_h` input=False/hidden=True ·
`ccv_site_both` input=True/hidden=True.

node1이 NODE2_PROMPT.md를 갱신해 arm 정의를 확정했다(F30 라벨이 틀렸었고, 저장된 config 기준으로
정정됨). 사용자 결정: **FIXED ladder**. 세 셀 모두 cam_encoder ON, `ttt_learnable_freqs: false`,
`cam_phase_mode: plucker`, seed 1, deterministic_noise, index_seed 42, 20,000 steps,
save_every 250 / keep_last_iter 1000.

| 셀 | run | ttt_input_rope | ttt_hidden_rope | GPU |
|---|---|---|---|---|
| input | `ccv_site_in` | true | false | 0 |
| hidden | `ccv_site_h` | false | true | 1 |
| both | `ccv_site_both` | true | true | 2 |

신규 파일: `minVid/configs/ar/abl_ccv_site_{in,h,both}.yaml`, `run_ccv_site.sh`(셀당 1 GPU,
**master_port를 GPU id에서 유도** — 루프 인덱스 유도 시 전 셀이 같은 포트를 받아 EADDRINUSE),
`watch_ccv_site.sh`(첫 체크포인트 생성 확인 + step 13999 별도 보관).

**착수 중 고친 stale 경로 2건** (기록 목적): 저장소가 `TTT_camera_embedding` → `TTT_rope`로
바뀐 뒤 ccv config의 절대경로가 갱신되지 않았다. `output_path`는 존재하지 않는 구 경로를
가리켰고(실제 산출물은 TTT_rope 아래), `api_key_path`는 wandb가 disabled인데도 코드가 파일
존재를 단언해 3셀 모두 즉시 죽었다(`AssertionError: API key file does not exist`).
둘 다 현행 경로로 수정. 프로토콜 파라미터는 일절 건드리지 않았다.

기동 검증: 덤프된 config에서 3셀의 `ttt_input_rope`/`ttt_hidden_rope`/`ttt_learnable_freqs=false`/
`use_cam_encoder=true`/`max_fwdbwd_passes=20000` 모두 의도대로 확인. 실측 ~11 s/step → **~61h 예상**.

### (구) P5 질문 — 해소됨
데이터·체크포인트·config 경로는 모두 정상이라 실행 자체는 가능하다. 다만 **arm 정의가
문서와 config에서 어긋난다**: RESULTS_DOSSIER F30 표는 `ccv_pra`를 "(input, learnable)"로
적었는데, `abl_ccv_pra.yaml`은 `ttt_input_rope: true`와 `ttt_hidden_rope: true`를 **둘 다**
켜고 `use_cam_encoder: false`만 다르다(즉 "input 전용"이 아니라 "cam_encoder 없는 PRA 양쪽").
따라서 "hidden-only"가 (a) `ttt_hidden_rope`만 켜고 cam_encoder 없음인지, (b) cam_encoder 있고
hidden만인지가 갈린다. ~46h 런이라 추측으로 시작하지 않고 node1 확인을 기다린다.


## Multi-chunk NVS 그리드 (NODE2_PROMPT_MULTICHUNK / _MC_RESTART, 2026-08-06)  [DONE]

**프로토콜**: 4 arm × `ttt_num_chunks: [1,2,4,8]`(forward마다 n 1개 draw, arm당 모델 1개),
32 input views 고정(청크 8192/4096/2048/1024, 전부 Muon 상각점 427 초과), 그 외 표준
(RE10K 256x256, 30k iters, bs16, lr 1e-4, LPIPS 5k부터, seed 95). gpu3 순차.
평가: 256 held-out scenes, 32 input / 4 target, n=1,2,4,8.
**4런 전부 동일 커밋(16db3e2)에서 재실행** — rotary fusion(6d2f388)이 bf16 반올림을 ~1 ULP
바꿔 이전/이후 런이 비교 불가라 최초 mc_base/mc_in을 폐기하고 처음부터 다시 돌렸다.

### Delta(rotary − NoPE), per-scene paired, n=256 scenes

**PSNR** (Δ / t / win%)

| arm | n=1 | n=2 | n=4 | n=8 |
|---|---|---|---|---|
| input (`mc_in`) | +0.690 / +40.8 / 99.6% | +0.771 / +42.7 / 99.6% | +0.845 / +36.7 / 99.6% | +0.686 / +25.9 / 97.3% |
| hidden (`mc_h`) | +1.276 / +59.8 / 100% | +1.439 / +64.3 / 100% | +1.555 / +61.7 / 100% | +1.361 / +53.5 / 99.6% |
| Both (`mc_both`) | +1.333 / +57.7 / 100% | +1.471 / +60.6 / 100% | +1.607 / +55.6 / 100% | +1.498 / +50.5 / 99.6% |

**LPIPS** (Δ / t / win%; 낮을수록 좋음)

| arm | n=1 | n=2 | n=4 | n=8 |
|---|---|---|---|---|
| input | −0.012 / −26.7 / 97.3% | −0.016 / −33.6 / 98.8% | −0.020 / −29.7 / 98.8% | −0.019 / −23.9 / 97.7% |
| hidden | −0.032 / −62.4 / 99.6% | −0.038 / −70.6 / 100% | −0.043 / −62.9 / 99.6% | −0.038 / −53.2 / 99.6% |
| Both | −0.024 / −42.9 / 99.6% | −0.029 / −50.6 / 100% | −0.034 / −45.6 / 100% | −0.032 / −35.9 / 99.2% |

절대 PSNR(참고용, **n 사이 비교 금지**): base 22.047/21.841/21.410/21.128,
in 22.737/22.612/22.255/21.814, h 23.323/23.280/22.965/22.489, both 23.381/23.312/23.018/22.626.

**실행 메모**: (1) `3e07ad9`의 리스트 config가 OmegaConf 경유로는 크래시했다 — YAML 리스트가
`ListConfig`라 `isinstance(x,(list,tuple))`이 False가 되어 `ttt_num_chunks > 1`로 떨어짐.
`model.py.__init__`에서 int가 아니면 `list[int]`로 정규화하도록 수정(int 경로 불변).
(2) 32뷰는 `min_frames = num_views*3` 때문에 train split이 59,411 → 32,361 씬(49%)으로 줄지만
4 arm에 동일 적용이라 paired delta에는 영향 없음. (3) fusion 이후 10.2 it/s로 런당 ~50분.

## WAVE 2 — DNA 장거리 (seq 32768, bs 1, w128, 3B 토큰, s42)
**WAVE 2는 WAVE 2 4셀끼리만 비교** (seq가 달라 ppl 절대값이 WAVE 1과 다름).

| 날짜 | 태스크 | 설정 | val ppl | 비고 |
|---|---|---|---|---|
| 2026-08-04 | T5 dna32k_nope | `ttt_nope=true` | **3.1680** | step 91552, 2,999,975,936 tok, 무중단·재시도0 |
| 2026-08-04 | T6 dna32k_rope | (input q/k rope) | **3.1612** | step 91552, 2,999,975,936 tok, 무중단·재시도0 |
| 2026-08-05 | T7 dna32k_honly_g1 | `ttt_nope=true, ttt_hidden_rope=true, gain 1.0` | **3.1678** | step 91552, 2,999,975,936 tok, 무중단·재시도0 |
| 2026-08-05 | T8 dna32k_hpra_g1 | `ttt_hidden_rope=true, gain 1.0` | **3.1795** | step 91552, 2,999,975,936 tok, 무중단·재시도0 |

**WAVE 2 최종 4셀 (전원 step 91552 / 2,999,975,936 tok, 무중단·재시도0·NaN없음, 1시드 s42)**:
rope 3.1612 / honly 3.1678 / nope 3.1680 / hpra 3.1795. 4셀 범위 **0.0183**.

**분해능 실측 (node1 참고용, 판정 아님)**: 종반 step>=85000(LR≈0, 모델 거의 정지) 구간의
eval 간 val ppl 변동 — WAVE 2 표준편차 0.0115-0.0221 / 범위 0.0345-0.0722,
WAVE 1 표준편차 0.0063-0.0230 / 범위 0.0183-0.0732. 즉 **셀 간 격차(WAVE 2 0.0183,
WAVE 1 0.0490)가 같은 런 내부의 eval 변동폭과 같은 자릿수**다. DNA는 셀당 1시드뿐이라
시드 노이즈 추정치는 없다(자연어 실험은 3시드였다).

**val 노이즈 관측 (node1 참고용, 판단 아님)**: 큐 지시대로 val 토큰 수를 WAVE 1과 맞추면
(≈2M) 블록 길이가 8배가 된 만큼 **독립 시퀀스가 488개 → 61개로 8배 감소**한다. 그 결과
val ppl 추정이 눈에 띄게 노이지해졌다 — step 30k-57k 구간 val ppl 표준편차가
WAVE 1 0.061-0.084 (488블록) 대 WAVE 2 0.101-0.148 (61블록)으로 약 1.7배.
또한 val 곡선 모양(최소점 후 중반 상승 → 종반 회복)은 **양쪽 공통**이다:
WAVE 1 최소 2.99@19k → 40-60k에서 3.15-3.25 → 최종 3.08-3.13 회복.
WAVE 2 최소 3.04@18k → 중반 3.16-3.47 → T5 최종 3.168. 훈련도 bs 1이라 스텝당
유전체 한 구간만 보므로 train loss 편차도 WAVE 1보다 크다.

## DNA 교차-태스크 검증 (프로토콜: 200M LaCT, seq 4096, w128, 3B 토큰, s42)
| 날짜 | 태스크 | 설정 | val ppl | 비고 |
|---|---|---|---|---|
| 2026-08-03 | T1 dna_nope_w128 | `ttt_nope=true` | **3.0951** | step 91552, 2,999,975,936 tok, val_loss 1.129817 |
| 2026-08-03 | T2 dna_rope_w128 | (input q/k rope, 기본) | **3.0988** | step 91552, 2,999,975,936 tok, val_loss 1.131008 |
| 2026-08-03 | T3 dna_honly_g1_w128 | `ttt_nope=true, ttt_hidden_rope=true, gain 1.0` | **3.0845** | step 91552, 2,999,975,936 tok, val_loss 1.126399 |
| 2026-08-03 | T4 dna_hpra_g1_w128 | `ttt_hidden_rope=true, gain 1.0` | **3.1335** | step 91552, 2,999,975,936 tok, val_loss 1.142158 |

실행 무결성: 4런 모두 단일 시드(s42) 1회, 무중단(self-heal 재시도 0회, `train.log.crash*` 없음),
non-finite loss 없음, 동일 토큰수(2,999,975,936)에서 종료, final.pt 저장 완료.
GPU 0-3 동시 실행, 총 소요 ≈ 4.5h.

부수 관측(실행 성능, 결과 해석 아님): 위치 인코딩 사이트가 처리량에 유의미한 영향.
로그 전체 중앙값 tok/s — nope 204,157 / honly 199,936 (−2.1%) / rope 193,665 (−5.1%) /
hpra 188,331 (−7.8%). 두 사이트 비용이 거의 가산적. FLOPs가 아니라 비융합(PyTorch) TTT
경로의 메모리 이동·커널 런치 오버헤드로 보임(input rope는 rotary 적용 위해 시퀀스 전체
fast_q/k를 einops로 레이아웃 왕복 복사; hidden rotary는 청크 텐서에 제자리 적용이라 더 쌈).
토큰예산 고정 프로토콜이라 비교 타당성에는 영향 없음(완료 시각만 어긋남).
