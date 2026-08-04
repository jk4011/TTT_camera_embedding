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
