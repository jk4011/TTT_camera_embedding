# node4 작업 지시 (node1이 작성, 2026-09-24 21:50 KST)

node4는 GPU 1장짜리 노드다. 이 노드의 일은 **tttLRM tab:recon의 CaPET 행을 최종 레시피로 다시 학습하고 평가하는 것 하나**다.
이 파일이 node4의 우편함이다. 지시가 바뀌면 node1이 이 파일을 고친다.

## 배경
- 최종 레시피는 `foot_in+h_foot+dpt_lin+dpt_abs+pdir+vo_rope`다(사용자 결정, 2026-09-24).
  - 깊이는 value 채널이 아니라 레이어 입력 x 위의 선형 층 하나(`capet_dlin`, 0 초기화)로 예측한다.
  - 절대 깊이 t = exp(2.5·tanh(s/2.5))라서 장면 초점 p*를 쓰지 않는다.
- 코드 이식(`tttlrm_ref/model/lact_ttt_cam.py`)과 12스텝 스모크 테스트는 node1에서 끝냈다. 24개 층 모두 깊이 헤드에 기울기가 흐른다.
- 설정은 `tttlrm_ref/configs/scratch_capet_lin_1gpu.yaml`이다(1 GPU × grad_accum 2 × bs 4 = 유효 배치 8, 다른 tab:recon 셀과 같다).
  2 GPU 노드용 `scratch_capet_lin.yaml`도 같은 체크포인트 폴더를 쓴다.
- 비교 대상인 No Encoding / PRoPE / RayRoPE 셀은 이미 끝났다. 이 셀만 남았다.

## 할 일
1. 노드 준비: `bash /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/claude_portable/setup_node.sh`
2. `nvidia-smi`로 GPU가 비어 있는지 본다. 다른 프로젝트가 수십 GB를 쓰고 있으면 시작하지 말고 사용자에게 알린다.
3. 실행한다. 학습에서 평가까지 자동으로 가고, 재실행하면 `latest.pt`에서 이어진다.
   ```bash
   NODE=node4 nohup setsid bash /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope/lact_nvs/run_tttlrm_capet_lin.sh >/dev/null 2>&1 &
   ```
4. 5분 뒤 확인한다.
   - `grep BATCH-COMPOSITION tttlrm_ref/outputs/scratch_capet_lin.log`에 `world_size=1 x grad_accum=2 x bs_per_gpu=4`가 찍혀야 한다.
   - `tail -1 tttlrm_ref/outputs/scratch_capet_lin/train_log.jsonl`의 `s_per_step`이 약 5.8이어야 한다.
   - 둘 다 맞으면 정상이다. 오류가 나면 로그 끝부분을 아래 "상태"에 적고 사용자에게 알린다.
5. 노드가 리셋되면 1~3을 다시 하면 된다. 체크포인트는 250스텝(약 24분)마다 lustre에 저장된다.

## 예상
- 학습 15,000스텝 × 5.8초 ≈ 24시간이다. 21:50에 시작하면 25일 22:00쯤 끝난다. 평가(DL3DV 140장면)는 약 0.5시간이다.
- 진행 로그는 `lact_nvs/outputs/queue_tttlrm.log`다.
- 결과는 `tttlrm_ref/evaluation/scratch_capet_lin_step15000/`에 쌓이고, 140개가 차면 끝이다.
- 이 노드에서 다른 일은 돌리지 않는다. 쌍체 통계와 tab:recon 작성은 node1이 한다.
- 2 GPU 노드가 나중에 할당되어 옮기게 되면, **반드시 node4 학습을 먼저 멈춘 뒤** 그쪽에서 같은 명령을 실행한다.
  두 학습이 같은 체크포인트 폴더에 쓰기 때문이다.
- 다른 노드의 작업은 건드리지 않는다. node1은 NVS와 CCV 평가, node2는 Table 4, 셋째 노드(fork 세션)는 Table 5를 맡고 있다.

## 상태 (node4가 갱신)
- (시작 시각, 5분 확인 결과, 종료 시각을 여기에 적는다)
