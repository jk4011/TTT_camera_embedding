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
