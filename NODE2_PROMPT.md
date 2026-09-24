# NODE2_PROMPT.md — node2의 살아있는 지시 파일 (node1이 갱신, node2가 실행)

마지막 갱신: **2026-09-24 20:10 KST (node1)** — 새 과제: **Table 4를 절대 깊이(`dpt_abs`)로 다시 측정**.
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
4. **건드리지 말 것 (node1에서 실행 중)**: `lact_nvs/outputs/{re10k,gobj,dl3dvu}_dpabs_pdir_vo_s137`,
   `tttlrm_ref/`의 모든 셀, `lact_ar_video/`의 모든 CCV 작업, `lact_nvs/run_one_gpu_chain.sh`.
5. **완료 확인은 파일로만 한다.** `pgrep -f <패턴>`은 실행한 셸 자신과 매칭되어 영원히 참이 된다.
   프로세스를 죽일 때도 패턴 말고 **PID**로 죽인다.
6. 실행 중인 `.sh`를 제자리에서 고치지 않는다(bash가 파일을 조금씩 읽는다). 고칠 일이 있으면 node1에 알린다.

## 2. 배경 (왜 하는가)

CaPET의 깊이를 **장면 초점 p* 기준 배율**(`t = t_c · exp(…)`)에서 **절대 깊이**(`t = exp(2.5·tanh(s/2.5))`,
정규화 장면 단위 1에서 시작)로 바꾸는 안을 검토 중이다. p*는 입력 뷰 구성에 따라 바뀌고, 제자리에서 도는
카메라에서는 점이 카메라 원점에 뭉개진다(RESULTS_DOSSIER F98). 절대 깊이는 p*를 전혀 쓰지 않는다
(t_c를 NaN으로 채워도 손실·그래디언트가 유한하고 손실이 같음을 확인, `lact_nvs/_smoke_dpt_abs.py`).

- node1: Table 3의 CaPET 행(= Table 4 맨 아래 행, 입력+은닉+v/o)을 세 데이터셋에서 학습 중.
- **node2: Table 4(`table:ablation_position`)의 나머지 CaPET 세 행을 세 데이터셋에서.** 총 9셀.

| Table 4 행 | 설정 | 셀 이름 |
|---|---|---|
| 입력 + 은닉 (v/o 없음) | `config/dp_abs_pdir_both.yaml` | `{re10k,gobj,dl3dvu}_dpabs_pdir_both_s137` |
| 입력만 | `config/dp_abs_pdir_in.yaml` | `{re10k,gobj,dl3dvu}_dpabs_pdir_in_s137` |
| 은닉만 | `config/dp_abs_pdir_h.yaml` | `{re10k,gobj,dl3dvu}_dpabs_pdir_h_s137` |

세 설정은 기존 `dp_chan_pdir_{both,in,h}.yaml`에서 `cam_mode`에 `+dpt_abs`만 더한 것이다(다른 차이 없음).
세 개 모두 node1에서 스모크 테스트를 통과했다. 프로토콜은 Table 3/4와 같다: 시드 137, 8뷰, 30k 스텝, 표준 평가.

## 3. 할 일

- **[PENDING] T4-abs**: 9셀 실행. 아래 한 줄이면 된다. 데이터셋별 레인 세 개가 동시에 돌고(카드에 항상 3셀),
  각 레인이 `both → in → h` 순서로 진행한다. 끝난 셀은 건너뛰고 체크포인트에서 이어가므로,
  **리셋 뒤에도 같은 명령을 다시 실행하면 된다.**
  ```bash
  cd /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope/lact_nvs
  GPU=0 NODE=node2 nohup setsid bash run_table4_abs.sh > /dev/null 2>&1 &
  ```
  진행 기록: `lact_nvs/outputs/queue_table4_abs.log`. 셀별 학습 로그: `lact_nvs/outputs/<셀>/train.log`.

- **시작 확인 (필수)**: 띄운 뒤 5분 안에 세 `train.log` 모두에 `Iter 00000xx, PSNR: …` 줄이 찍히는지,
  로그 앞부분의 `cam_mode`에 `dpt_abs`가 들어 있는지 확인한다. 처음 한 번은 LPIPS용 VGG 가중치를 받느라
  1–2분 걸린다. 셋 중 하나라도 `Traceback`이면 멈추고 §4에 적는다.
- **예상 속도**: 한 카드에 3셀이면 셀당 약 1.1–1.4 it/s. 한 행(3셀)에 약 7시간, **9셀 전체 약 22시간**.
  행 순서대로 `both`(~7 h) → `in`(~14 h) → `h`(~22 h)에 완성된다.
- **보고**: 셀이 하나 끝날 때마다 `NODE2_RESULTS.md` 끝에 한 줄을 append한다.
  형식: `2026-09-25 03:10 re10k_dpabs_pdir_both_s137  PSNR 22.xx  SSIM 0.xxx  LPIPS 0.xxx  (eval.json)`.
  값은 그 셀의 `eval.json`에서 읽는다. 쌍체 통계와 표 작성은 node1이 한다.
- 상태 태그는 node2가 직접 고친다: `[PENDING]` → `[RUNNING node2 <시각>]` → `[DONE]` / `[FAILED <원인>]`.

## 4. node2 → node1 (node2가 여기에 적는다)

(비어 있음)
