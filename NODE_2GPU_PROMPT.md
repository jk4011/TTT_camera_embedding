# 2-GPU 노드 작업 지시 (node1이 작성, 2026-09-24 21:20 KST)

이 노드의 일은 **tttLRM tab:recon의 CaPET 행을 최종 레시피로 다시 학습하고 평가하는 것 하나**다.
최종 레시피 = `foot_in+h_foot+dpt_lin+dpt_abs+pdir+vo_rope`. 깊이를 value 채널이 아니라 레이어 입력 x 위의
선형 층 하나(0 초기화)로 예측하고, 절대 깊이 t = exp(2.5·tanh(s/2.5))라서 장면 초점 p*를 쓰지 않는다.
코드 이식과 12스텝 스모크 테스트는 node1에서 끝냈다. 24개 층 모두 깊이 헤드에 기울기가 흐른다.

## 할 일
1. 노드 준비: `bash /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/claude_portable/setup_node.sh`
2. `nvidia-smi -L`로 GPU가 2장인지 확인한다. 1장이면 스크립트가 알아서 1-GPU 설정(grad_accum 2, 약 24시간)으로 돈다.
3. 실행 (학습 → 평가까지 자동, 재실행하면 `latest.pt`에서 이어서 한다):
   ```bash
   NODE=node2g nohup setsid bash /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope/lact_nvs/run_tttlrm_capet_lin.sh >/dev/null 2>&1 &
   ```
4. 5분 뒤 확인: `grep BATCH-COMPOSITION tttlrm_ref/outputs/scratch_capet_lin.log`에 `world_size=2 x grad_accum=1 x bs_per_gpu=4`,
   `tail -1 tttlrm_ref/outputs/scratch_capet_lin/train_log.jsonl`의 `s_per_step`이 약 2.7이면 정상이다.

## 예상
- 학습 15,000스텝 × 2.7초 ≈ 11.3시간, 평가(DL3DV 140장면) 약 0.5시간.
- 진행 로그: `lact_nvs/outputs/queue_tttlrm.log`, 결과: `tttlrm_ref/evaluation/scratch_capet_lin_step15000/` (140개가 차면 끝).
- 이 노드에서 다른 일은 돌리지 않는다. 한 셀이 두 GPU를 다 쓴다. 쌍체 통계와 표 작성은 node1이 한다.
