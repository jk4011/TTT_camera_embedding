# 야간 결과 요약 (2026-09-01 02:45 → 10:00, 29셀 착지; 세부: RESULTS_DOSSIER.md F81, OBJ_ANALYSIS.md §7/§7.1)

## 1. 헤드라인 — vi에서 PE-only 첫 +1 dB
- **`foot_all_iso_h2x` 22.997 = +1.016 dB** (t=50.4, 승률 98.8%, seed 95, n=500; base 21.981)
  = foot 점 좌표를 **6방향 축(iso)** 으로 양 사이트(입력 q/k + hidden h)에 위상 코드 + 회전 carrier(v/o)
  + **hidden 주파수 사다리 ×2**. 어제 밤 최고(rot_hshell/foot_all +0.716) 대비 **+0.30**.
- 구성 요소는 독립적이고 초가산: iso 6방향 +0.134(t=13), hidden ×2 +0.073(t=7.2) → 합쳐 +0.299.
- 시드 재현: `foot_all_iso`(iso만) s137 = **+0.890** vs base_s137(s95 +0.851). +1.016 셀의 s137/s211,
  orbit 쌍둥이는 진행 중(≈11:10–11:30 착지) — 착지 시 별도 보고.
- 추가 파라미터 0(고정 사다리 + 기존 학습 게인), 추가 FLOPs ≈0(per-token 위상 곱), 구조 변경 없음.

## 2. 교차 데이터셋 (F80 finalist를 재조정 없이 그대로)
| 데이터 (n) | rot_hshell | foot_all | 읽기 |
|---|---|---|---|
| vi 500 | +0.716 | +0.717 (iso +0.851 / iso×h2 **+1.016**) | wide-baseline 본령 |
| orbit 91° 499 | +0.642 | +0.564 | 유지 (hidden은 chord 우위) |
| RE10K narrow 256 | **+0.266** (t=12) | **+0.481** (t=15) | 좁은 베이스라인도 이득, foot_all 우위 |
| DL3DV walking 140 | −0.001 | −0.015 | **정확히 중립** — 전진 보행에선 p*가 비정칙 → 코드 무음화, 무해 |

## 3. 야간 신규 방법 판정 (5개 subagent 브레인스토밍 → 필터 → 셀)
| 방법 (출처) | 결과 | 판정 |
|---|---|---|
| **SPEC-2x** hidden ladder ×2 (TTT 대수 유도 P4) | foot_all +0.073, iso 위 +0.165 | ✅ **예측 적중** — 헤드라인 재료 |
| **iso 6방향 foot 좌표** | foot_all +0.134 | ✅ 단, foot 양 사이트에서만(chord·입력 단독에선 −) |
| foot-좌표 **위상 carrier** (사용자 질문 "Vo도 foot?") | 행렬 carrier +0.041 | ✅ 위상 carrier 실패는 ray 좌표 탓이었음; iso 위에선 포화(+0.007) |
| hidden anchors (rot_hanchor) | vi +0.049 / orbit −0.066 vs chord | ➖ 데이터 의존 손잡이 |
| QH 쿼터니언 반각 hidden(비음 kernel, 대수 P1) | rot_raw −0.252 | ❌ **반증** — hidden의 부호 민감성은 기능 |
| CFR 각도정합 per-token 회전 | rot_hshell −0.188 / rot_raw −0.067 | ❌ per-view 전역 회전이 관건 |
| Householder 반사 코드 | base −0.129 | ❌ |
| near-shell 앞 교차점(불투명 prior) | foot_all −0.176 | ❌ 최근접점이 옳다 |
| per-token foot-frame carrier | foot_all −0.279 | ❌ carrier는 per-view 직교 |
| 층상 메모리(head=깊이층, 4-head) | 자기 대조군 +0.750, foot_all −0.131 | ❌ head 분할 비용(−0.163) 못 메움 |
| 층별 좌표 변주 / anchor 혼합 / 비대칭 store-read / camray | 0 / −0.042 / −0.119 / −1.90 | ➖❌❌❌ |
| ENV² 깊은 null 봉투 (대수 P3) | rot_hshell −0.048 | ❌ 반증(약) — 대수 스코어: P4 적중, P1·P3 반증 |
| ω-split(사다리 ×0.5) / store-only carrier(o 역사상 없음) | foot_all −0.063 / −0.150 | ❌❌ — 사다리는 높여야, carrier는 양방향이어야 |

## 4. 깊이 추정 경로의 정량 종결
oracle σ-곡선(orbit): σ=0 +2.08 → 0.04 +0.98 → 0.07 +0.77 → **0.12 +0.08**. 추정기는 장면 스케일 4% 안에
들어야 유효한데 메모리 삼각측량 probe는 ~0.07(최상층만) → "TTT 성질로 공짜 깊이"는 정밀도 문턱 미달.
좌표(foot/iso-foot)로 승부한 것이 옳았고, 남은 oracle 격차(~1.0 dB)는 query의 미지 깊이 그 자체.

## 5. 왜 그런가 — input/hidden rope의 특성과 맞는 embedding (OBJ_ANALYSIS §7, 대수 유도 + 실험 정합)
weight-norm이 고정한 스케일에서 SwiGLU hidden 유닛은 입력의 거의 순수한 2차 특징이라,
- **입력 사이트(q/k)**: 코드가 회수 계수에 **제곱(짝수 kernel)** 으로 들어감 → wrap·부호에 둔감 →
  **정확한 per-view 회전 행렬/날카로운 점 코드**가 맞다(softmax 없는 대비 증강). 그래서 유도 스펙트럼이 2ω →
  hidden 사다리를 ×2로 맞추면 이득(SPEC-2x 적중).
- **hidden 사이트(h)**: 계수에 **선형(부호 민감)** 으로 곱해짐 → Δθ>π/2면 옳은 value를 빼 버림 → 대응점에서
  차이 0이 보장되는 **ray 위 3D 점 좌표(foot)** 만 허용, 불확실성 봉투(chord)는 데이터에 맞추는 손잡이.
  비음 kernel로 "안전하게" 만들면 오히려 손해(QH) — 빼기는 선택 기능.
- **carrier(v/o)**: W1에 선형 저장되어 임의 가역 사상에 대수적으로 정확하지만, Muon·열별 weight-norm·
  RMSNorm의 스펙트럼 수술과 **교환하는 직교 사상**만 살아남음 → per-view 회전 행렬(또는 foot 좌표 위상).
  per-token 프레임·projective·크기 코드는 전부 죽음.
- **합성 법칙**: 세 슬롯은 하나의 회수 항의 서로 다른 인자라 가산; 두 사이트 코드가 같은 쌍을 승인할 때만
  날카로워지므로 **일관성 > 혼합**(iso가 foot 양 사이트에서만 이기는 이유).

## 6. 진행 중 / 다음
- 착지 예정: env2(node2), orbit foot_all_iso(~10:10), base_s211(~10:50), iso_h2x s137(~11:30), iso s211(~11:10);
  큐: orbit iso_h2x(W5-23), **×4 반증 셀**(W5-24; 유도는 "×2가 최적, ×4는 개선 없음" 예측), node2 시드 트리오.
- 판단 요청: (a) 헤드라인을 `foot_all_iso_h2x`로 확정할지(시드/orbit 결과 후), (b) s211 3-seed 표 범위,
  (c) 논문 프레이밍 — "TTT 슬롯별 kernel 대수 → 사이트별 코드 처방" 을 중심으로 갈지.
