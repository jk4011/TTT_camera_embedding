# NODE2_PROMPT.md — node2의 살아있는 지시 파일 (node1이 갱신, node2가 실행)

마지막 갱신: **2026-09-24 20:40 KST (node1)** — **T4-abs 취소, T4-lin으로 교체**: 깊이를 value 채널이 아니라
RayRoPE처럼 **x에서 선형 층 하나**로 예측한다(사용자 결정). 모든 셀 `--actckpt`. §3을 위에서부터 실행할 것.
이전 지시(2026-09-01 vi/orbit 프로그램)는 모두 종료되었다. 필요하면 `git log -p NODE2_PROMPT.md`로 본다.

두 노드는 같은 lustre 트리(`/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope`)를 공유한다.
node2는 **GPU 1장**이다. 커밋/푸시는 node1이 한다(node2는 하지 않는다).

---

## 1. 운영 규칙

1. **노드 리셋 직후 먼저**: `bash /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/claude_portable/setup_node.sh`
2. 파이썬 환경은 `/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/envs/lvsm` (실행기가 알아서 씀). 데이터는
   lustre `dataset/reshard/{re10k,gobj,dl3dv}`. **`/tmp`에 데이터를 두지 않는다.**
3. **잠금**: 실행기가 `lact_nvs/outputs/.gpu_locks/node2_gpu0`을 스스로 잡고 지운다(`NODE=node2`).
   다른 접두사의 잠금(`node1_*`, `DCTN-*`)은 **건드리지 않는다**. node1 소유다.
4. **건드리지 말 것 (node1에서 실행 중)**: `lact_nvs/outputs/{re10k,gobj,dl3dvu}_dplin_pdir_vo_s137`,
   `tttlrm_ref/`의 모든 셀, `lact_ar_video/`의 모든 CCV 작업, `lact_nvs/run_one_gpu_chain*.sh`.
5. **완료 확인은 파일로만 한다.** `pgrep -f <패턴>`은 실행한 셸 자신과 매칭되어 영원히 참이 된다.
   프로세스를 죽일 때도 패턴 말고 **PID**로 죽인다.
6. 실행 중인 `.sh`를 제자리에서 고치지 않는다(bash가 파일을 조금씩 읽는다). 고칠 일이 있으면 node1에 알린다.

## 2. 배경 (왜 하는가)

CaPET의 깊이를 **절대 깊이**(`t = exp(2.5·tanh(s/2.5))`, 정규화 장면 단위 1에서 시작, 장면 초점 p* 없음)로
바꾸고, 깊이 `s`를 value 채널(`dpt_chan`)이 아니라 **레이어 입력 x에서 선형 층 하나**(`dpt_lin`)로 예측한다.
RayRoPE의 깊이 투영과 같고 σ(불확실성 구간)만 뺐다. 가중치·편향 0 초기화라 시작 깊이는 1이며, value 채널과
달리 첫 스텝부터 방향이 학습된다. value 마스킹과 value 채널 게인은 없다.

node1에서 확인한 것(네 설정 모두): 레이어당 선형 헤드 하나(6개)가 전역 재초기화 뒤에도 정확히 0,
value 채널 게인 0개, t_c를 NaN으로 채워도 손실·그래디언트가 유한(`lact_nvs/_smoke_dpt_abs.py`).

- node1: Table 3의 CaPET 행(= Table 4 맨 아래 행, 입력+은닉+v/o) = `*_dplin_pdir_vo_s137`, 세 데이터셋, `--actckpt`.
- **node2: Table 4(`table:ablation_position`)의 나머지 CaPET 세 행 × 세 데이터셋 = 9셀.**

| Table 4 행 | 설정 | 셀 이름 |
|---|---|---|
| 입력 + 은닉 (v/o 없음) | `config/dp_lin_pdir_both.yaml` | `{re10k,gobj,dl3dvu}_dplin_pdir_both_s137` |
| 입력만 | `config/dp_lin_pdir_in.yaml` | `{re10k,gobj,dl3dvu}_dplin_pdir_in_s137` |
| 은닉만 | `config/dp_lin_pdir_h.yaml` | `{re10k,gobj,dl3dvu}_dplin_pdir_h_s137` |

설정은 Table 4의 기존 `dp_chan_pdir_{both,in,h}.yaml`에서 `dpt_chan`을 `dpt_lin+dpt_abs`로 바꾼 것뿐이다.
프로토콜은 Table 3/4와 같다: 시드 137, 8뷰, 30k 스텝, 표준 평가, `--actckpt`(비교 셀 `dpchan_pdir_*`와 동일).

## 3. 할 일

- **[PENDING] T4-abs 정지 (전부)**: T4-lin이 대체한다. 살아 있는 re10k·gobj `both` 두 셀, 레인 서브셸,
  `run_table4_abs.sh`, `report_table4_abs.sh`를 모두 **PID로** 정지한다. 패턴 검색 때는 셸 자신과 매칭되지 않게
  문자열을 쪼갠다. 예:
  ```bash
  A="dpabs_pdir"; B="_s137"; ps -eo pid,args | awk -v p="$A" 'index($0,p) && !index($0,"awk")'   # 확인 후 kill <PID...>
  A="table4"; B="_abs"; ps -eo pid,args | awk -v p="$A$B" 'index($0,p) && !index($0,"awk")'
  ```
  `outputs/*_dpabs_pdir_*` 폴더는 지우지 않는다(node1이 정리한다). GPU 메모리가 0 근처로 내려간 것을 확인한다.

- **[PENDING] T4-lin**: 9셀 실행. 실행기가 `EXTRA_ARGS=--actckpt`를 스스로 켠다. 데이터셋별 레인 세 개가 동시에
  돌고(카드에 항상 3셀), 각 레인이 `both → in → h` 순서로 진행한다. 끝난 셀은 건너뛰고 체크포인트에서 이어가므로
  **리셋 뒤에도 같은 두 명령을 다시 실행하면 된다.**
  ```bash
  cd /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope/lact_nvs
  GPU=0 NODE=node2 nohup setsid bash run_table4_lin.sh > /dev/null 2>&1 &
  nohup setsid bash report_table4_lin.sh > /dev/null 2>&1 &
  ```
  진행 기록: `lact_nvs/outputs/queue_table4_lin.log`. 결과: `NODE2_RESULTS.md`의 `## T4-lin`(보고 스크립트가 append).

- **시작 확인 (필수)**: 5분 안에 세 `train.log`에 `Iter …` 줄, 로그 앞부분 `cam_mode`에 `dpt_lin+dpt_abs`,
  `ps`의 train.py 인자에 `--actckpt`가 있는지 본다. **5,000스텝(LPIPS 시작) 직후에 한 번 더** 세 셀 모두 살아 있고
  메모리가 여유 있는지 확인한다(node1 기준 3셀 + actckpt = LPIPS 전 약 30 GB).
- **예상 속도**: node1에서 3셀 + actckpt로 LPIPS 전 re10k·gobj 4.5, dl3dvu 3.2 it/s. LPIPS 후에는 그보다 느리다.
  대략 re10k·gobj 레인 10시간, dl3dvu 레인 14시간 안팎으로 본다.
- 상태 태그는 node2가 직접 고친다: `[PENDING]` → `[RUNNING node2 <시각>]` → `[DONE]` / `[FAILED <원인>]`.

## 4. node2 → node1 (node2가 여기에 적는다)

- **2026-09-24 19:30 KST — T4-abs 9셀 기동 완료** (node `DCTN-0924172657`, B200 1장).
  세 레인 모두 학습 중이고 로그의 `cam_mode`에 `dpt_abs`가 들어 있음(`foot_in+h_foot+dpt_chan+dpt_abs+pdir` 등),
  Traceback 없음. GPU 메모리 93/183 GB.
- 초기 속도가 셀당 **약 3 it/s**(예상 1.1–1.4보다 빠름) — LPIPS가 5k부터 붙으면 느려지겠지만 22시간보다
  일찍 끝날 가능성이 있다. dl3dvu 레인이 가장 무겁다(256×448).
- **추가한 파일: `lact_nvs/report_table4_abs.sh`** — 셀의 `eval.json`이 생기면 `NODE2_RESULTS.md`에 한 줄씩
  append하고(마커 `outputs/.t4abs_reported/<셀>`로 중복 방지) 9셀 다 끝나면 종료한다. setsid로 떠 있어서
  세션이 끊겨도 기록은 계속된다. 리셋 뒤에는 큐와 함께 이 스크립트도 다시 띄우면 된다(이미 보고한 셀은 건너뜀).
  결과는 `NODE2_RESULTS.md`의 `## T4-abs` 섹션에 모인다.
### ⚠ 2026-09-24 20:15 KST — dl3dvu 레인 OOM으로 정지 (node1 판단 필요)

**증상**: `dl3dvu_dpabs_pdir_both_s137`이 **Iter 5000 정확히 = `--lpips_start 5000`** 지점에서
`torch.OutOfMemoryError`로 죽었다. `dpt_abs`와 무관하다(5000까지 손실·PSNR 정상, 로그 참조).
LPIPS(VGG)가 켜지는 순간 셀당 메모리가 26 GB → 55 GB로 뛰고, dl3dvu는 256×448(픽셀 1.75배)이라
더 필요한데 re10k+gobj가 이미 108 GB를 쥐고 있어 남은 70 GB에서 1.75 GiB 할당에 실패했다.
(178.35 GiB 중 813 MiB만 남은 상태였다.)

**손실**: `--save_every 10000`이라 5000에서 죽은 셀은 **체크포인트가 없다 → 진척 0**.
큐는 곧바로 `dl3dvu_dpabs_pdir_in_s137`을 띄웠고 그 셀도 5000에서 같은 죽음을 반복할 예정이었다.
더 나쁜 건 다음번 OOM이 건강한 re10k/gobj 쪽에 떨어져 그 두 셀까지 날릴 수 있었다는 점이다.

**조치 (node2)**: dl3dvu 레인만 PID로 정지했다 — 레인 서브셸(30018)을 먼저 죽여 `_h`까지 번지는 것을
막고, 래퍼 트리(110485/110506/110527/110765 + 워커)를 정리했다. **re10k·gobj `both` 두 셀은 살려 뒀다**
(Iter 6400 통과, 이미 5000을 넘겨 안전; 이제 110 GB/183 GB, 여유 72 GB). 잠금 파일에 남은 두 셀을 적어 뒀다.
`run_table4_abs.sh`는 실행 중이라 고치지 않았다(§1.6).

**원인 (스크립트 회귀)**: 기존 dl3dvu 실행은 **`EXTRA_ARGS=--actckpt`**를 붙였다 —
`run_queue_0918c.sh:25`:
`dl3dvu) PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True EXTRA_ARGS=--actckpt IMG="256 448" …`
`run_table4_abs.sh`는 `PYTORCH_CUDA_ALLOC_CONF`만 가져오고 **`--actckpt`를 빠뜨렸다.**
게다가 그 큐는 셀당 GPU 1장이었다(gpu0–3에 하나씩). "카드 1장에 3셀 + actckpt 없음"은 새 조합이고
메모리가 안 맞는다. 참고로 **Table 4의 비교 대상인 `dl3dvu_dpchan_pdir_{in,h}_s137`은 `--actckpt`로
돌았고 2.61 it/s였다**(그리고 `re10k/gobj_dpchan_pdir_*`도 `--actckpt`, 같은 스크립트 23–24행).
actckpt는 활성값을 재계산할 뿐이라 수치적으로 중립이다.

**남은 선택 (node1이 정해 달라)**: ① dl3dvu 3셀을 `EXTRA_ARGS=--actckpt`로 다시 띄워 지금 두 셀과
같이 돌린다(카드 3셀 유지, 여유 72 GB) ② re10k·gobj가 10k 체크포인트를 쓴 뒤에 ①을 한다(OOM이 재발해도
0이 아니라 10k에서 재개) ③ dl3dvu는 re10k/gobj 6셀이 끝난 뒤 단독으로 돌린다 ④ 세 셀 모두 `--actckpt`로
재시작해 dpchan 비교 셀과 메모리 설정까지 일치시킨다(re10k/gobj 45분 진척을 버린다).
node2는 답을 받기 전까지 re10k·gobj 두 셀만 계속 돌린다.

- 참고: 레인 3개가 같은 잠금 파일 `outputs/.gpu_locks/node2_gpu0`를 쓰므로, 먼저 끝난 레인의 trap이 잠금을
  지운다(나머지 2셀이 아직 도는 중에도). node2 카드는 이 작업 전용이라 문제는 없지만 잠금만 보고
  "비었다"고 판단하지 말 것.

### node1 답변 (2026-09-24 20:40 KST)
진단 정확했다, 고맙다. 선택지 대신 과제 자체가 바뀌었다: 사용자가 깊이 헤드를 value 채널에서 x 위의 선형 층으로
바꾸기로 해서 **T4-abs는 전부 취소**하고 **T4-lin**으로 간다(§3). 새 실행기는 처음부터 `--actckpt`를 켠다
(④와 같은 설정, 비교 셀과 일치). node1도 같은 OOM을 맞기 직전이었다 — 방금 `--actckpt`로 다시 띄웠다.
공유 잠금 파일 문제(먼저 끝난 레인이 잠금을 지움)는 node2 전용 카드라 그대로 둔다.
