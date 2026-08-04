# NODE_QUEUE.md — node2 실행 큐 (TTT-RoPE 프로젝트)

node1(메인 세션)이 설계·판정·큐잉을 담당하고, node2는 이 파일의 태스크를 위에서부터 실행한다.
상태 태그는 node2가 직접 수정한다: `[PENDING]` → `[RUNNING node2 gpu<i> <시각>]` →
`[DONE <핵심수치>]` / `[FAILED <원인 한 줄>]`.

## 현재 노드2 배정 과제: DNA 언어모델 (교차-태스크 검증 #2)

**왜 하는가 (배경, 실행에는 불필요하지만 맥락)**: 우리 TTT-RoPE는 fast-weight 메모리의
*주소 공간*에 작용한다. 자연어에서는 장거리 회수가 내용-주소라 hidden 사이트가 패리티에 그쳤고
(3-시드 ~18.58 구분 불가), NVS/CCV처럼 회수가 좌표-주소인 과제에서만 이득이 났다. 유전체는
알파벳이 4글자뿐이라 같은 모티프가 수없이 반복 → **내용-주소가 원리적으로 모호하고 위치가
회수의 주소로 기능**한다. 자연어와 *동일한 아키텍처·동일한 예산*에서 서열이 뒤집히면 F35
(copy task) 해리의 자연-데이터 확장이 된다.

**프로토콜은 LLM 실험과 동일하게 고정한다** (비교 가능성이 실험의 전부):
200M LaCT / 12층 / hidden 768 / fw-head 4 / chunk 1024 / **window_size 128** / seq 4096 /
bs 8 / 3B 토큰 / data_seed 42 / model seed 42. 1런 ≈ 5.5h (B200 1장).

---

## 준비 절 (최초 1회 + 노드 리셋 후마다)

```bash
bash /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/claude_portable/setup_node.sh
cd /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope
ls .venv_llm/bin/python && .venv_llm/bin/python -c "import fla, torch; print(torch.__version__, torch.cuda.device_count())"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader   # GPU 4개 기대
```
GPU 점유는 `lact_nvs/outputs/.gpu_locks/node2_gpu<i>`에 용도를 적어 표시하고
(`echo <exp> > ...`), 끝나면 지운다. **락 디렉터리는 lustre라 두 노드가 공유하므로
반드시 `node2_` 접두사를 쓴다** (node1은 `node1_gpu<i>`를 쓴다). 접두사 없는 예전
`gpu<i>` 파일은 상대 노드 것일 수 있으니 지우지 마라.
훈련은 반드시 `lact_llm/run_llm.sh`를 통해 실행한다(triton/inductor 캐시 env가 그 안에 있음;
`/tmp`는 noexec라 직접 실행하면 중간에 죽는다).

---

## P0. DNA 데이터 경로 구현 + sanity  [DONE sanity3 PASS: fineweb≈8.44 / dna ln8→1.34 / resume 해시일치+crash-resume OK]

`lact_llm`의 기존 fineweb 스트림과 **같은 인터페이스**로 hg38을 공급하는 경로를 추가한다.
설계는 아래에 고정되어 있으니 그대로 구현만 하면 된다(설계 변경 금지, 막히면 FAILED로 보고).

1. **다운로드** (무인증 직접 HTTP, ~1GB gz → 3.1GB):
   ```bash
   mkdir -p /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/datasets/dna && cd $_
   wget -c https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
   ```
2. **토크나이저**: 문자 단위. vocab = `{PAD:0, BOS:1, EOS:2, A:3, C:4, G:5, T:6, N:7}`
   (소문자 soft-mask는 대문자로 변환, ACGT 외 문자는 전부 N). vocab_size=8.
3. **스플릿**: `chr1..chr22, chrX`만 사용(스캐폴드/`chrM`/`chrY` 제외).
   **val = chr20 전체**, train = 나머지. 텔로미어/센트로미어의 연속 N 구간은
   "N 비율 > 50%인 블록은 버림" 규칙으로 제외.
4. **블록화**: seq_len 4096 연속 블록(문서 경계 개념 없음 → EOS 삽입 없이 연속 스트림),
   train은 염색체를 순회하며 data_seed로 셔플된 순서로 읽고, 기존
   `PackedBlockStream`과 동일하게 `state()`/`restore()`로 **정확 재개**를 지원해야 한다
   (체크포인트 재개가 이 프로젝트의 절대 규칙).
5. **val 캐시**: 488블록(≈2M 토큰)을 뽑아
   `lact_llm/val_cache_hg38_4096.pt`로 저장(기존 `get_or_build_val_set`과 같은 형식).
6. **train_small.py 연결**: `--data {fineweb,dna}` 플래그 추가(기본 fineweb, 기존 동작
   100% 불변). dna일 때 vocab_size=8, tokenizer 없이 위 경로 사용.

**Sanity (통과해야 T1 진행)**:
- (a) 기존 fineweb 경로가 바뀌지 않았음: `--data fineweb`로 20스텝 돌려 loss가 기존 로그
  초반값(step100 loss ≈ 8.4)과 같은 범위인지 확인.
- (b) dna 20스텝: loss가 ln(8)=2.08 근방에서 시작해 감소, NaN 없음.
- (c) resume 정확성: dna로 60스텝 돌려 ckpt 저장 → 재개 후 다음 배치가 중단 없이 이어지는지
  (기존 gold-test 방식대로 배치 텐서 해시 비교).
결과 요약(구현 파일, sanity 수치 3개)을 NODE2_RESULTS.md에 append.

## T1. dna_nope_w128  [DONE ppl 3.0951]
```bash
cd /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope/lact_llm
./run_llm.sh 0 dna_nope_w128 --data dna --window_size 128 --bs 8 --token_budget 3000000000 \
  --extra_json '{"ttt_nope": true}'
```

## T2. dna_rope_w128  [DONE ppl 3.0988]
```bash
./run_llm.sh 1 dna_rope_w128 --data dna --window_size 128 --bs 8 --token_budget 3000000000
```

## T3. dna_honly_g1_w128  [DONE ppl 3.0845]
```bash
./run_llm.sh 2 dna_honly_g1_w128 --data dna --window_size 128 --bs 8 --token_budget 3000000000 \
  --extra_json '{"ttt_nope": true, "ttt_hidden_rope": true, "ttt_hrope_gain": 1.0}'
```

## T4. dna_hpra_g1_w128  [DONE ppl 3.1335]
```bash
./run_llm.sh 3 dna_hpra_g1_w128 --data dna --window_size 128 --bs 8 --token_budget 3000000000 \
  --extra_json '{"ttt_hidden_rope": true, "ttt_hrope_gain": 1.0}'
```

T1–T4는 GPU 0–3에 **동시** 투입한다(GPU당 1개). 각 런은 self-heal 래퍼로 감싸라:
실패 시 같은 커맨드 재실행 → `--auto_resume` 기본값으로 체크포인트에서 이어진다.

## 판정 방법 (node2가 수치만 뽑아 기록, 해석은 node1)
```bash
for r in dna_nope_w128 dna_rope_w128 dna_honly_g1_w128 dna_hpra_g1_w128; do
  python3 -c "import json;print('$r', round(json.loads(open('outputs/$r/val_log.jsonl').readlines()[-1])['ppl'],4))"
done
```
LLM 대조군(자연어, 동일 프로토콜 3-시드 평균): nope 18.685 / rope 18.582 / honly 18.593 /
hpra 18.578. DNA는 vocab이 8이라 ppl 절대값이 완전히 다르다(참고: 무작위 = 8.0,
문헌상 좋은 모델 ≈ 3.2–3.4). **비교는 오직 DNA 4셀 사이에서만 한다.**

---

---

# === WAVE 2 (2026-08-03, node1 지시): 장거리 DNA 재설계 ===

**WAVE 1 결과와 왜 재설계하는가**: 4셀 결과 nope 3.0951 / rope 3.0988 / honly 3.0845 /
hpra 3.1335. 서열은 가설 방향(honly 최선, input rope는 NoPE보다도 나쁨)이지만 효과가 0.3%로
너무 작다. 원인은 설정에 있다: seq 4096 bp는 사람 유전자 하나(중앙값 24 kb)도 못 담고,
attention 윈도우 128을 빼면 메모리 전담 구간이 128~4,096 bp뿐인데 이 대역은 국소 모티프
통계로 대부분 설명된다. 검증하려던 장거리 위치-주소 회수(인핸서-프로모터 10 kb~1 Mb,
뉴클레오솜 주기)를 사실상 훈련시키지 못했다. → **시퀀스를 32,768 bp로 늘려 메모리 전담
구간을 8배로 키운다.**

**새 프로토콜 (WAVE 1과 토큰/스텝·예산·스텝수가 정확히 동일 → 직접 비교 가능)**
seq_len **32,768** / **bs 1** (토큰/스텝 32,768 = 4096x8과 동일) / window_size 128 /
lact_chunk_size 1024 (시퀀스당 32청크) / 3B 토큰 = 91,552 스텝 / data_seed 42 / model seed 42.
나머지(200M LaCT, 12층, hidden 768, fw-head 4)는 그대로.

## P0-W2. seq_len 파라미터화 + 실현가능성 실측  [DONE sanity PASS; 32k 30k tok/s·51.6GB, 64k 100GB, 128k 진행불가; 병목은 seq가 아니라 bs=1 → 1런 ≈31h]
1. `dna_data.py`의 `SEQ_LEN = 4096` 하드코딩과 파일명(`hg38_train_blocks_4096.npy`,
   `val_cache_hg38_4096.pt`)을 seq_len 인자로 파라미터화해라. 기존 4096 산출물은 그대로
   두고(재사용), 32768용 블록/val 캐시를 새로 만든다. **val 토큰 수는 WAVE 1과 맞춘다:
   32k 블록 61개(= 1,998,848 토큰)**. train 블록은 같은 규칙(N 비율 50% 초과 블록 폐기,
   chr20 = val, chr1-22+X만).
2. `train_small.py`가 `--seq_len 32768 --data dna`로 동작하는지 확인(플래그가 이미 있으면 그대로).
3. **실측 보고**(각각 20스텝만 돌려 측정, 학습은 하지 마라): seq_len ∈ {32768, 65536, 131072}
   × bs 1에서 (a) 피크 GPU 메모리, (b) tok/s, (c) OOM 여부. `--actckpt` 류 옵션이 있으면
   그것도 함께. 이 수치로 node1이 후속 확장을 결정한다.
4. sanity: 32k에서 20스텝 loss가 ln(8)=2.08 근방에서 시작해 감소, NaN 없음 + 체크포인트
   재개 정확성(기존 방식대로 배치 해시 비교).
결과를 NODE2_RESULTS.md에 append.

## T5. dna32k_nope  [DONE ppl 3.1680]
```bash
cd /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope/lact_llm
./run_llm.sh 0 dna32k_nope --data dna --seq_len 32768 --bs 1 --window_size 128 \
  --token_budget 3000000000 --extra_json '{"ttt_nope": true}'
```

## T6. dna32k_rope  [DONE ppl 3.1612]
```bash
./run_llm.sh 1 dna32k_rope --data dna --seq_len 32768 --bs 1 --window_size 128 \
  --token_budget 3000000000
```

## T7. dna32k_honly_g1  [DONE ppl 3.1678]
```bash
./run_llm.sh 2 dna32k_honly_g1 --data dna --seq_len 32768 --bs 1 --window_size 128 \
  --token_budget 3000000000 \
  --extra_json '{"ttt_nope": true, "ttt_hidden_rope": true, "ttt_hrope_gain": 1.0}'
```

## T8. dna32k_hpra_g1  [DONE ppl 3.1795]
```bash
./run_llm.sh 3 dna32k_hpra_g1 --data dna --seq_len 32768 --bs 1 --window_size 128 \
  --token_budget 3000000000 --extra_json '{"ttt_hidden_rope": true, "ttt_hrope_gain": 1.0}'
```

T5-T8은 GPU 0-3 동시 투입, self-heal 래퍼 필수. P0-W2 sanity 통과 전에는 시작하지 마라.
**중요**: `max_position_embeddings`가 32768보다 작으면 config에서 올려야 한다(확인해서
필요하면 extra_json에 추가하고 무엇을 바꿨는지 기록).

## 판정 (수치만 기록, 해석은 node1)
WAVE 1 참조값(seq 4096): nope 3.0951 / rope 3.0988 / honly 3.0845 / hpra 3.1335.
**seq가 다르면 ppl 절대값이 달라지므로 WAVE 2는 WAVE 2 4셀끼리만 비교한다.**

---

---

# === WAVE 3 (2026-08-03, node1 지시): 심볼릭 음악 — 교차-태스크 검증 #3 ===
# 상태: 데이터 준비는 node1이 선행(GPU 불필요). 학습은 어느 노드든 GPU 4장이 비면 투입.

**가설**: 음악의 핵심 연산이 위치-주소 회수다 — "8마디 전 주제를 조옮김해 반복"은 내용이
아니라 정확한 음악적 거리로 과거를 지목한다(마디 4박, 프레이즈 4/8마디의 강한 주기 구조).
자연어(내용-주소, 패리티)와 DNA(약한 신호) 사이에서, 음악은 좌표-주소 쪽에 가장 가까운
1D 자연 데이터다. 여기서 hidden 사이트가 이기면 F35 copy-task 해리의 자연-데이터 확장이 된다.

**프로토콜은 자연어/DNA와 동일하게 고정** (비교 가능성):
200M LaCT / 12층 / hidden 768 / fw-head 4 / chunk 1024 / **window_size 128** / seq 4096 /
bs 8 / 3B 토큰 / data_seed 42 / model seed 42.
**좌표는 평범한 토큰 인덱스를 쓴다** — 마디/박자 좌표는 쓰지 않는다(사용자 정책: 표준 세팅
이어야 하며 방법론에 맞춰 세팅을 쥐어짜지 않는다). 음악의 주기성은 토큰 거리에 이미
반영되므로 별도 좌표가 필요 없다는 것이 이 실험의 논지다.

## P0-W3. 데이터/토크나이저 준비  [node1 선행 중]
Lakh MIDI(LMD-full, 176k 파일, 무인증) → miditok REMI 토크나이즈 → uint16 packed 블록 +
val 캐시(자연어/DNA와 동일한 ≈2M 토큰). 산출물은 datasets/music/ 및
lact_llm/val_cache_music_4096.pt, 데이터 경로는 `--data music`.

## T9.  music_nope       [RUNNING node2 gpu0 2026-08-05 02:35] ./run_llm.sh <g> music_nope --data music --window_size 128 --bs 8 --token_budget 3000000000 --extra_json '{"ttt_nope": true}'
## T10. music_rope       [RUNNING node2 gpu1 2026-08-05 02:35] ./run_llm.sh <g> music_rope --data music --window_size 128 --bs 8 --token_budget 3000000000
## T11. music_honly_g1   [RUNNING node2 gpu2 2026-08-05 02:35] ./run_llm.sh <g> music_honly_g1 --data music --window_size 128 --bs 8 --token_budget 3000000000 --extra_json '{"ttt_nope": true, "ttt_hidden_rope": true, "ttt_hrope_gain": 1.0}'
## T12. music_hpra_g1    [RUNNING node2 gpu3 2026-08-05 02:35] ./run_llm.sh <g> music_hpra_g1 --data music --window_size 128 --bs 8 --token_budget 3000000000 --extra_json '{"ttt_hidden_rope": true, "ttt_hrope_gain": 1.0}'

판정: 4셀끼리만 비교(vocab이 달라 자연어/DNA와 ppl 절대값 비교 불가).
참조 서열 — 자연어(3-시드): nope 18.685 / rope 18.582 / honly 18.593 / hpra 18.578.
DNA 4k(1-시드): nope 3.0951 / rope 3.0988 / honly 3.0845 / hpra 3.1335.

---

## 완료 로그 (node2가 append)
<!-- 형식: <시각> <노드/GPU> <태스크> <상태> <핵심수치> -->
- 2026-08-03 16:45 node2 시작: setup_node OK, torch 2.9.1+cu130, 4 GPU 유휴, fla import OK, hg38 다운로드(983MB) 완료.
- 2026-08-03 16:50 node2 P0 DONE: dna_data.py 구현 + sanity 3종 PASS(위 상태 참조). 전처리 695,152 train블록.
- 2026-08-03 16:56 node2 T1-T4 RUNNING: dna_{nope,rope,honly_g1,hpra_g1}_w128 → gpu0-3, self_heal 래퍼, ~4.3h 예상(91,553 step, ~195k tok/s).
- 2026-08-03 21:27 node2 T1-T4 DONE (4.5h, 무중단·재시도0·NaN없음, 전원 step 91552 / 2,999,975,936 tok):
  nope 3.0951 / rope 3.0988 / honly 3.0845 / hpra 3.1335. 수치는 NODE2_RESULTS.md 표에 기록.
  → **큐 전부 소진. node2 GPU 0-3 유휴, 락 해제. 다음 태스크 대기 중** (10분 간격 큐 감시 루프 가동).
- 2026-08-03 21:50 node2 P0-W2 DONE: seq_len 파라미터화 + 32k 산출물(train 86,911블록, val 61x32768
  =1,998,848 tok) + sanity PASS. 실측: tok/s가 seq와 무관하게 ~30k → **병목은 bs=1**(seq 4096 bs1도
  30,384). 32k peak 51.6GB / 64k 100GB / 128k는 157GB에서 진행 불가. max_position_embeddings는
  변경 불필요(하한 역할, 무영향) → T5-T8 큐 원문 그대로 실행.
- 2026-08-03 22:25 node2 T5-T8 RUNNING: dna32k_{nope,rope,honly_g1,hpra_g1} → gpu0-3, self_heal,
  91,552 step / 2,999,975,936 tok(WAVE 1과 동일). **예상 ≈31h/런**(WAVE 1의 4.5h 대비 ~7배).
- 2026-08-04 16:10 node2 T5 DONE: dna32k_nope ppl 3.1680 (step 91552 / 2,999,975,936 tok, 무중단).
  gpu0 유휴·락 해제. T6/T7/T8 진행중(각 86.2k/77.8k/80.3k step). WAVE 3(T9-T12)은 음악 데이터가
  준비 완료(datasets/music/, val_cache_music_4096.pt)지만 큐 조건이 "GPU 4장이 비면 투입"이라 대기.
  T6-T8 완료 예상 각 ~1.6h / ~4.1h / ~3.4h → 4장 확보는 약 4시간 후.
- 2026-08-04 22:40 node2 T6 DONE: dna32k_rope ppl 3.1612 (step 91552 / 2,999,975,936 tok, 무중단).
  gpu1 락 해제. 남은 T7 honly(82.9k)·T8 hpra(85.4k) 진행중, 각 잔여 ~2.4h/~1.5h.
- 2026-08-05 01:05 node2 T8 DONE: dna32k_hpra_g1 ppl 3.1795 (step 91552 / 2,999,975,936 tok, 무중단).
  gpu3 락 해제. 마지막 T7 honly 88.6k step, 잔여 ~1.1h. 완료 시 GPU 4장 확보 → T9-T12(음악) 착수.
- 2026-08-05 02:10 node2 T7 DONE → **WAVE 2 완료** (4셀 전원 step 91552 / 2,999,975,936 tok, 무중단):
  rope 3.1612 / honly 3.1678 / nope 3.1680 / hpra 3.1795 (범위 0.0183).
  참고 분해능: 종반 eval 변동 표준편차 0.0115-0.0221 → 셀 간 격차와 동일 자릿수, 1시드뿐.
  GPU 4장 확보 → T9-T12(음악) 착수 준비(sanity 먼저).
- 2026-08-05 02:35 node2 T9-T12 RUNNING: music_{nope,rope,honly_g1,hpra_g1} → gpu0-3, self_heal.
  선행 sanity_music.py ALL PASS (val 488x4096 vocab451, shuffle/resume/epoch-wrap 해시일치,
  20스텝 loss ln(451)=6.111→5.207 감소, ckpt 왕복 배치해시·가중치 일치, 코퍼스 2.92B tok=1.03에폭).
  91,552 step / 2,999,975,936 tok, 실측 ~155k tok/s → 약 5.4h 예상.

