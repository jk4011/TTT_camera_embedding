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

## T1. dna_nope_w128  [RUNNING node2 gpu0 2026-08-03 16:56]
```bash
cd /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope/lact_llm
./run_llm.sh 0 dna_nope_w128 --data dna --window_size 128 --bs 8 --token_budget 3000000000 \
  --extra_json '{"ttt_nope": true}'
```

## T2. dna_rope_w128  [RUNNING node2 gpu1 2026-08-03 16:56]
```bash
./run_llm.sh 1 dna_rope_w128 --data dna --window_size 128 --bs 8 --token_budget 3000000000
```

## T3. dna_honly_g1_w128  [RUNNING node2 gpu2 2026-08-03 16:56]
```bash
./run_llm.sh 2 dna_honly_g1_w128 --data dna --window_size 128 --bs 8 --token_budget 3000000000 \
  --extra_json '{"ttt_nope": true, "ttt_hidden_rope": true, "ttt_hrope_gain": 1.0}'
```

## T4. dna_hpra_g1_w128  [RUNNING node2 gpu3 2026-08-03 16:56]
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

## 완료 로그 (node2가 append)
<!-- 형식: <시각> <노드/GPU> <태스크> <상태> <핵심수치> -->
- 2026-08-03 16:45 node2 시작: setup_node OK, torch 2.9.1+cu130, 4 GPU 유휴, fla import OK, hg38 다운로드(983MB) 완료.
- 2026-08-03 16:50 node2 P0 DONE: dna_data.py 구현 + sanity 3종 PASS(위 상태 참조). 전처리 695,152 train블록.
- 2026-08-03 16:56 node2 T1-T4 RUNNING: dna_{nope,rope,honly_g1,hpra_g1}_w128 → gpu0-3, self_heal 래퍼, ~4.3h 예상(91,553 step, ~195k tok/s).
