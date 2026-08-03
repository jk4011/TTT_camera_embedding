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
