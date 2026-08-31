# TTT layer에 camera pose를 어떻게 넣어야 wide-baseline(Objaverse)에서도 효과적인가

작성 2026-08-31. 사용자 요청: "camera embedding task에 집중, Objaverse/DL3DV 성능을 올릴 method 개발.
실험 전에 지금까지의 camera-embedding 결과를 전부 분석하고 그에 맞는 가설을 세울 것."
근거 자료: `RESULTS_DOSSIER.md` (F1–F70, 특히 gObjaverse F51/F55–F63/F65–F70), `lact_ttt_cam.py`,
4개 관점의 ideation 서브에이전트(3D geometry / equivariance / fast-weight memory / PE literature)와
문헌조사 서브에이전트 보고서(CaPE, GTA, PRoPE, RayRoPE, DPPE, URoPE, LVT, ViewRoPE, SPAD 등).

---

## 0. 한 문단 요약

TTT의 readout은 `o_j ≈ h(q_j)W1⁰ + Σ_i lr_i ⟨h(q_j), h(k_i)⟩ v_i + …` 이고, 카메라 정보가 들어갈 자리는
세 곳뿐이다: **주소 계수 ⟨·,·⟩ (input/hidden 두 사이트)**, **value carrier v_i**, **초기 가중치 항**.
RE10K(뷰 간 ~4°)에서는 "같은 ray 근처"가 곧 correspondence라서 Plücker 위상 코드가 주소 계수를
정확히 relative하게 만들어 +1.2~+1.7 dB를 냈다. gObjaverse(~90°)에서는 correspondence가 **ray 좌표의
작은 차이가 아니라 3D 교차 관계**(Plücker reciprocal product = 0, bilinear)이므로 ray 좌표 위상 코드는
어떤 주파수 대역에서도 정보가 없고(F57 단조 하강), 오히려 content 유사도에 cos(Δθ) 세금만 물린다.
반면 carrier를 회전으로 정렬하는 v/o transport(rot_raw +0.43)는 "주파수를 고를 필요가 없는" 정확한
군 작용이라 살아남는다. 따라서 wide baseline용 설계 원칙은 **(a) 주소 좌표를 ray가 아니라 3D 점(깊이
prior/oracle)으로 바꾸거나, (b) 위상이 아니라 행렬 군 작용을 두 주소 공간 모두에 쓰거나, (c) 토큰에서
절대 pose를 빼고 모든 pose를 relative transform으로만 넣는(GTA/PRoPE 체제)** 세 갈래이며, 아래 12개
가설이 이 세 갈래를 채운다. 1차 실험(8 GPU)에서 가장 정보량이 큰 것은 **GT-depth oracle 3D-point
rotary**(위상 코드가 "올바른 좌표"에서는 90°에서도 사는지 판정)와 **pose-free 토큰 + rot_raw**이다.

---

## 1. 수식: 카메라가 들어갈 수 있는 자리

LaCT fast weight (head_dim d=256, hidden d_h=512): `f_W(x) = (silu(xW0) ⊙ (xW2)) W1`, `h(x) := silu(xW0)⊙(xW2)`.
update(입력 2048 토큰, 한 번): `ΔW1 = Muon(Σ_i lr_i h(k_i)ᵀ v_i)`, `ΔW0 = Muon(Σ_i lr_i k_iᵀ g_i)`,
`ΔW2 = Muon(Σ_i lr_i k_iᵀ c_i)` (g, c는 SwiGLU backward의 per-token 벡터), 이후 열 단위 weight-norm.
apply: `o_j = f_{W+ΔW}(q_j)`. 1차 전개(내적 주소화 보조정리, WRITING_BRIEF):

```
o_j ≈ h(q_j) W1⁰                              … (A) 초기 readout: 절대 항
    + Σ_i lr_i ⟨h(q_j), h(k_i)⟩ · v_i          … (B) hidden 주소 채널 × carrier  (지배적)
    + Σ_i lr_i ⟨q_j, k_i⟩ · c_ij               … (C) input 주소 채널 (gate 보정)
```

카메라 정보가 개입할 수 있는 슬롯은 정확히 네 개다.
1. **input 주소** q,k (L2-norm 후): 직교 변환 T_j, T_i → ⟨T_j q, T_i k⟩. 위상(rotary) 또는 행렬(gta/ogta/rot_raw).
2. **hidden 주소** h: ⟨R_j h(q_j), R_i h(k_i)⟩ (h-PRA; h_ga/h_rot는 행렬).
3. **carrier** v_i → P_i⁻¹ v_i, o_j → P_j o_j: W1 채널에서 **정확히** `P_j P_i⁻¹ v_i` (이론노트 2026-08-07,
   임의의 가역 P; 직교 P만 저장 norm 보존).
4. **초기 가중치 항 (A)**: 변환된 q가 W0⁰/W1⁰을 지나면 `P_j h(P_jᵀq_j) W1⁰` 같은 **절대 pose 함수**가 남는다.
   RE10K에서는 유용한 prior(F12: 우회 시 −0.59), 90°에서는 미지수.

여기에 슬롯 밖의 한 가지: **토큰 자체**가 world-frame Plücker raymap을 갖고 있어 NoPE도 절대 pose를 안다.
PRoPE/RayRoPE/GTA의 최선 설정은 토큰이 pose-free(intrinsics만)이고 pose가 relative 변환으로만 들어간다.

---

## 2. 지금까지의 결과가 말하는 것

### 2.1 두 데이터 체제의 대비 (같은 백본, 같은 프로토콜, seed 95, paired)

| 메커니즘 | RE10K (3.85°) | gObjaverse (91°) |
|---|---|---|
| input Plücker rotary (F21) | +0.51 | **−0.41** |
| hidden Plücker rotary (F_h42) | +0.90 | **−0.57** |
| both (TTT-RoPE) | +0.97 (3-seed +1.08~+1.23) | **−0.89** |
| 완만한 ladder (wrap 불가 대역) | +0.04 (고주파가 이득) | +0.04 (어느 대역도 0 근방) |
| projective q/k만 (prope_in) | −0.18 | −0.11 |
| rigid/orthogonal q/k만 (gta_in / ogta) | −0.14 / +0.08 | −0.15 / −0.09 |
| v/o transport만 (vo_only) | — | −0.11 |
| **R on q/k + v/o rotation transport (rot_raw)** | +0.07 (중립) | **+0.43** |
| projective q/k/v/o (prope_raw) | −0.29 | +0.48 (3-seed +0.34) |
| image-coordinate ropes (prope_imgrope) | +0.38 | +0.34 |
| **imgrope + rotation transport (imgvo)** | +0.58 | **+0.39 (3-seed 22.58, LPIPS 최선)** |
| Plücker ladder + transport (pra_vo) | ≈ pra_hi | −0.18 (ladder 손해가 transport 이득을 잠식) |
| hidden projective 행렬 (h_ga) | — | −0.18 |
| 학습 depth 3D-point (point_rope) / sinc segment | +0.10 / +0.30 | **미실험** |
| 3-layer fast weight + 3 rotary (fw3l_rot3) | +1.69 (기록) | 미실험 |

attention 사이트(prope 코드베이스, 2-view, 우리 orbit 렌더): NoPE 12.4 / GTA 20.7 / PRoPE 20.8 /
모든 위상 코드(RayRoPE 자체 릴리즈 포함) 12~16 (F65–F70). 단, 그 NoPE는 **pose-free 토큰**이라
"pose 정보 자체가 없음"에 해당한다. LaCT-LVSM NoPE(22.19)는 world-frame raymap 토큰을 가지므로 공정한
attention 잣대는 "absolute-Plücker attention vs GTA"(문헌상 +1~2 dB)이지 +8 dB가 아니다.

### 2.2 확립된 교훈 (가설 설계의 제약 조건)
- **E1 (F1/F3)** 내적 주소화를 거치는 **직교** 인코딩만 이득. norm 왜곡(projective q/k), feature 주입
  (FiLM, q-bias, register, hyper-init, camera-lr)은 전부 손해 → 새 가설은 직교 변환·carrier·주소 좌표만 건드린다.
- **E2 (F56/F57/F65)** wide baseline에서 작동 축은 **위상 vs 행렬**이다: ray 좌표 위상은 대역과 무관하게 0~손해,
  행렬 군 작용은 이득. 이는 ray 좌표에서의 "작은 차이" 관계가 correspondence가 아니기 때문(§3).
- **E3 (F59/F61)** transport는 **carrier 슬롯**의 이득이고, 짝이 되는 q/k 변환 없이는 지불되지 않는다
  (vo_only −0.11 vs rot_raw +0.43). imgvo는 "세금 없는 주소(patch 위상) + carrier 정렬"의 조합.
- **E4 (F60)** 손해 보는 주소 코드 위에 transport를 얹으면 손해가 이긴다(가산성). → 주소 코드가 단독으로
  ≥0일 때만 transport와 조합할 가치가 있다.
- **E5 (F12/F23)** 절대 성분은 RE10K에서 무해하거나 유익한 prior였다. 90°에서는 8개 뷰가 서로 다른 회전으로
  같은 prior를 돌리므로 부호가 바뀔 수 있다 — **미검증** (h_dpra, delta-path transport를 objaverse에서 안 돌렸다).
- **E6 (F69)** 같은 객체를 RayRoPE의 vary-intrinsics 스크립트로 재렌더(윈도우 안에 nested near-duplicate 쌍이
  생김)하면 input ladder가 −0.41 → **+0.21**로 뒤집힌다. 위상 코드의 필요조건은 "윈도우 안의 near-duplicate 쌍".
- **E7 (F40/F41/F24)** rotary의 가치는 입력 뷰 수, 순차 update 단계 수, fast-weight 깊이에 따라 **커진다**
  (주소 공간을 주는 효과). 90°에서도 "주소 공간이 필요하다"는 사실은 같다 — 다만 올바른 좌표여야 한다.
- **E8 (F9/F25)** 좁은 baseline에서는 hidden 채널이 이득의 대부분(+0.96/1.08)을 나른다; 90°에서는 input
  사이트의 행렬 작용만 살아남았다(F58) → 사이트별 처방이 기하에 따라 뒤집힌다.

---

## 3. 왜 90°에서 위상 코드가 죽는가 (수학적 직관)

**3.1 correspondence는 ray 좌표의 "거리"가 아니다.** 뷰 i, j가 같은 표면점 X를 본다면 두 ray
`(o_i, d_i)`, `(o_j, d_j)`는 X에서 교차한다. Plücker 좌표 π = (d, m = o×d)로 쓰면 교차 조건은 reciprocal
product `B(π_i, π_j) = d_i·m_j + d_j·m_i = 0` — **bilinear** 관계이지 `π_i ≈ π_j`가 아니다. 90° 떨어진
대응 ray는 `|Δd| = O(1)`, `|Δm| = O(ρ)`로 무작위 쌍과 같은 크기다. rotary의 kernel `Σ_ω cos(ω·Δπ)`는
Δπ ≈ 0 근방에서만 최대이므로 (i) 고주파는 대응 쌍을 오히려 decorrelate하고, (ii) 저주파는 대응/비대응을
구별 못 한다 → F57의 "어느 대역도 +0.04 이하". RE10K에서는 대응 쌍이 near-duplicate라 Δπ ≈ 0이므로
같은 코드가 정확히 옳았다. 이것이 E2·E6의 근본 원인이다.
(참고: `B(π_i,π_j) = −½ (π_i−π_j)ᵀ J (π_i−π_j)`, J=[[0,I],[I,0]], 즉 교차 관계는 **부정부호(indefinite)
계량의 정상 kernel**이라 양의 가중 cos 합(Bochner)으로는 표현되지 않는다. 다만 `B²`은 Plücker의 2차
특징 `vec(ππᵀ)`의 내적으로 **정확히** 쓸 수 있다 → 가설 H7.)

**3.2 content 세금과 선형 readout.** rotary는 pair마다 `|q_p||k_p| cos(Δθ_p + φ_content)`를 곱한다.
softmax attention은 head/layer/차원 분업으로 세금을 피할 여지가 있지만(RNoPE, MLA decoupling), fast
weight의 sum-of-outer-products readout은 회수 채널이 하나뿐이라 세금이 그대로 떨어진다
(RELATED_WORK_SURVEY content-tax 절). 90°에서는 "옳은 pair가 Δθ 큰 pair"이므로 세금이 최대다.

**3.3 왜 행렬 작용은 사는가.** 회전 transport `R_j R_iᵀ`는 180°까지 단사이고 고를 주파수가 없다.
carrier를 한 프레임으로 정렬해 주면 fast weight가 서로 다른 뷰의 note를 **합산**하더라도 기하적으로 호환
가능한 양을 섞게 된다(분석노트 2026-08-07). GTA Table 5(value transform 제거 시 −2.45 dB CLEVR-TR),
RayRoPE ablation(v/o 제거 −0.4~−0.5), DPPE(object-centric에서는 R·t 결합 4×4 carrier가 해롭고 **R만**이
최선)와 정합적이다.

**3.4 왜 TTT의 transport 이득(+0.43)이 attention(+8)보다 작은가.**
(a) 우리 토큰은 이미 world-frame 절대 pose를 갖고 있어 NoPE가 강하고(22.19), transport가 중복된
    채널이 된다 — 공정 잣대는 +1~2 dB.
(b) hidden 맵 h(·)는 표현(intertwiner)이 아니다: `h(Rx) ≠ ρ(R)h(x)`. 지배 채널 (B)의 계수
    `⟨h(P_jᵀq_j), h(P_i⁻¹k_i)⟩`는 relative도 절대도 아닌 "뷰별로 뒤섞인" 값이 되고, slow weight가 이를
    견디도록 학습해야 한다. transport가 정확한 것은 W1 채널뿐이다.
(c) (A)항 `P_j h(P_jᵀ q_j) W1⁰`은 순수 절대 pose 함수: 90°에서는 8개 프레임으로 돌아간 prior.
(d) Muon은 직교 등변이지만 **W1의 열별 weight-norm**은 3-블록으로 회전되는 출력 차원을 독립적으로
    재스케일하므로 transport와 정확히 교환하지 않는다(블록별 norm이 등변 버전).
(e) 선형 kernel 회귀(one-step)는 softmax의 날카로운 pair 선택을 못 한다 — 90°에서는 crosstalk
    자체가 병이다(memory 관점 에이전트).

---

## 4. 가설 (12개) — 메커니즘 · 수식 · 예측 · 반증 조건

표기: Δ = gObjaverse seed-95 paired PSNR 변화(base 22.193). 구현 상태: ✅ 구현됨(이번), ◐ 기존 코드, ☐ 미구현.

### 갈래 (a): 주소 좌표를 3D 점으로

**H1. Oracle 3D-point rotary (`pt_gt`, `h_pt_gt`) — 진단, 최우선.** ✅
GT depth(EXR, z-depth 검증: 재투영 오차 0.0008)로 패치 중심 ray의 표면점 `x = o + t_gt d`를 위상 좌표로 쓴다
(input·hidden 두 사이트, 3축 × F). 대응 쌍은 `Δx = 0`이므로 세금이 정확히 0. 90°에서 "위상 코드가 올바른
좌표에서는 사는가"를 결정한다. RayRoPE는 attention에서 known depth로 Objaverse +2.8 dB를 보고.
예측 ≥ +0.5. **반증**: ≤ +0.1이면 좌표와 무관하게 위상 계열은 이 사이트에서 죽은 것 → 행렬/carrier
계열로만 간다. 변형 `pt_gt_in`(입력 토큰만 GT, 타깃은 shell chord): 타깃 depth가 병목인지 분리.

**H2. Object-shell chord sinc rotary (`shell_sinc` 입력 / `h_shell` hidden).** ✅
학습 depth 없이, ray가 focus point p*(입력 뷰 optical axis들의 최소제곱 교점 = look-at 렌더에서 정확히
물체 중심) 주위 반지름 r(학습)의 구를 지나는 **chord** `[t_c − h, t_c + h]` 위로 3D-point 위상을 적분한다:
`∫cos(ω u·x(t))dt = sinc(ω u·(h d))·cos(ω u·x_mid)`. 두 chord가 교차하면 kernel 최대. 실루엣 ray는 짧은
chord → 날카로운 주소, 중심 ray는 지름 → 저주파만 생존. plucker_sinc(RE10K +0.30, best win% 94)의
object-centric 버전이며 objaverse에서는 미실험. 예측 +0.1~+0.4. 반증: ≤ 0. 변형 `shell_iso`: 3축 대신
정20면체 6방향(등방 3D kernel).

**H3. (제외) 학습 depth head.** 사용자 지침(2026-08-31): "depth를 예측하는 layer 추가는 우리 방향이
아니다 — 모델 구조 변경 없이 positional embedding만 개발." 따라서 depth head 계열(point_rope 재설계)은
계획에서 뺀다. 이 갈래에서 허용되는 것은 구조 변경이 없는 H2(chord sinc)와 그 변형 H3b(✅ `anchor_in` /
`h_anchor`): chord 위 K=3 **고정** anchor 점(chord 분율 0.25/0.5/0.75) 각각에 주파수 블록(plane-sweep 위상;
URoPE는 고정 anchor가 학습 depth보다 낫다고 보고). H1(oracle)은 이 갈래의 상한을 재는 진단일 뿐 방법이 아니다.

### 갈래 (b): 위상 대신 행렬 군 작용, 모든 주소 공간에

**H4. Hidden-site 회전 작용 + transport (`rot_raw+h_rot`).** ✅
h_ga(−0.18)는 norm을 왜곡하는 projective P였다. 4-블록마다 `blockdiag(R_c2w, 1)`(직교)을 update·apply
양쪽에 적용하면 지배 채널이 `h_jᵀ R_jᵀ R_i h_i`로 정확히 relative가 되고 Muon/weight-norm은 그대로다.
"one matrix action per address space" = fw3l_rot3(F24)의 90° 버전. 예측 rot_raw 대비 +0.1~+0.3.
반증: ≤ rot_raw → hidden 사이트는 90°에서 위상뿐 아니라 군 작용도 죽음(memory 에이전트의 예측).

**H5. Delta-path transport / h_dpra at 90° (절대 (A)항 검증).** ◐ h_dpra 존재, delta-transport ☐
`o_j = f_{W⁰}(q_j) + P_j[f_{W'}(P_jᵀq_j) − f_{W⁰}(P_jᵀq_j)]`: 초기 readout은 pose-free로 두고 update-유도
항만 transport. RE10K에서는 절대 prior 제거가 −0.59였지만(F12) 90°에서는 부호가 뒤집힐 것으로 예측
(정준 프레임 = 장면마다 무작위 방위각 → nuisance). 먼저 기존 `h_dpra42`를 objaverse에 돌려 h_pra(−0.57)와
비교(예측: 덜 나쁘거나 +). 반증: h_dpra ≤ h_pra.

**H6. Ray-frame per-token transport (RayGTA, `raygta`).** ✅
뷰별 R_cam 대신 `R_tok = R_cam · R_pix(u,v)`(광축을 픽셀 ray로 돌리는 회전)을 q/k/v/o에 적용:
image rope(+0.34)와 rot transport(+0.43)를 하나의 SO(3) 작용으로 융합(둘의 위상 조합 imgvo는 +0.39로
가산되지 않았다). 예측 +0.5~+0.8. 반증: ≤ rot_raw.

### 갈래 (c): 토큰에서 절대 pose 제거 — 전부 relative

**H7. Pose-free 토큰(camray) + matrix transport (`input_raymap: camray` + rot_raw / prope_raw).** ✅
raymap을 identity extrinsics로 계산(o=0, d=K⁻¹[u,v,1], m=0) → 토큰은 intrinsics·픽셀 위치만 갖고, pose는
TTT 사이트의 `R_j R_iᵀ`(주소)·`R_j R_iᵀ v`(carrier)로만 들어간다 = GTA/PRoPE/RayRoPE의 실제 체제(F65에서
행렬 작용이 20.7을 낸 조건). 문헌 LVT: intrinsics-only 토큰 + relative pose가 world-Plücker 토큰보다
+1.4 dB. 위험: cross-view 정보 통로가 TTT 층뿐이라 pose 병목; 고정 반경 orbit에서는 회전이 위치를
결정하므로 rot_raw로 충분, 일반 데이터용은 prope_raw(translation 포함). 예측 +0.5~+1.0. 반증: < rot_raw.

**H8. 등변 SwiGLU (gate는 불변량만, content 경로는 등변).** ◐ 1단계 `rot_content` 구현됨(gate 브랜치 plain q/k, content 브랜치 R-변환 + v/o transport); 완전 등변 커널은 H7 양성 시
gate `silu(xW0)`가 l=0 불변량(3-블록 norm, 스칼라 슬롯)만 읽고 W2·W1이 블록-스칼라(`A⊗I₃ ⊕ B`)이면
`h(Rx) = ρ(R)h(x)`가 되어 (B)채널이 정확히 GTA 구조가 되고 (A)항이 pose-free가 된다(3.4(b)(c) 해결).
1단계(코드 재사용): `fast_weight_swish_glu_branch_input_rotary_apply`로 gate 브랜치에 plain q/k, content
브랜치에 R-변환 q/k(+ v/o transport) = "gate 불변, content relative". 예측 H7 위에 +0.3~+0.8.

### 갈래 (d): 세금 없는 부가 채널 / TTT 고유 메커니즘

**H9. Epipolar 2차 특징 bias (`epi_quad`).** ☐ (후순위 — fast weight에 기하 특징 행을 추가하므로 "PE만"
기준에서 구조 변경에 가깝다; 사용자 지침 반영)
hidden 주소에 37차원 블록을 덧붙여 `⟨φ(π_j), ψ(π_i)⟩ = c − β²B(π_i,π_j)²`(정확한 "ray 교차" kernel,
SE(3) 불변, wrap 없음)을 (B)채널에 **가산**한다: W1에 37행 추가(zero-init), update는 기하 basis로
value를 splat, apply는 자기 ray의 epipolar 이웃 value의 가중합을 읽는다 = fast-weight 판 epipolar
attention bias(SPAD: Objaverse +1.3 dB). attention에서는 O(N²)인 bias가 TTT에서는 공짜. 한계: 곱이 아닌
합이라 AND 선택이 아니다. 예측 +0.1~+0.3(imgvo와 적층 가능). 반증: β → 0.

**H10. Hidden-site image rope + transport (`prope_imgrope+vo_rel+h_img`).** ✅
양 체제에서 유일하게 양수인 위상 코드(patch 좌표, 뷰 간 세금 0)를 지배 채널(hidden)에도 둔다. 좁은
baseline에서 hidden 사이트가 이득의 대부분을 날랐으므로(E8) 세금 없는 좌표라면 90°에서도 가산될 수 있다.
예측 imgvo 대비 +0.1~+0.2. 반증: ≤ imgvo.

**H11. View-direction soft partition of hidden units (`hgate`).** ☐ (후순위 — hidden 유닛 gating은 PE보다
구조 변경에 가깝다; 사용자 지침 반영)
뷰 방향 f_v를 G개 anchor에 softmax로 배정하고 hidden 유닛에 `1+β·(a−1/G)`의 **비음·봉우리형** 대각
gating(update·apply 동일). cos kernel의 음의 로브 없이 "이웃 뷰에서 더 많이 읽기"를 구현, β zero-init으로
baseline과 정확히 일치. 예측 +0.1~+0.4. 반증: β → 0.

**H12. Delta-rule(잔차) update + transport (`dres2`).** ☐ (embedding 밖, 후순위)
Hebbian 목표 `⟨f(k),v⟩` 대신 `v − α·f_{W_now}(k)` 잔차(α zero-init)로 2단계 update: 뷰 간 crosstalk을
update가 스스로 상쇄. 90° 특유의 간섭 문제에 직접 대응하지만 embedding이 아니라 dynamics 변경이므로
1·2차 결과 후 결정.

**진단 대조군 (가설 아님)**: `attn_nope` / `attn_prope` — TTT 층을 LaCT 논문의 block-causal full attention으로
바꾼 같은 크기(6L/d256) 모델, PRoPE 유/무. 같은 토큰 입력에서 attention+relative encoding이 얼마나 높이
가는지 = 이 스케일에서 camera embedding의 headroom. 또 `gobj_base_s95`·`imgvo_s95`·`rot_raw_s95`·
`prope_raw_s95` 체크포인트를 새 test index로 재평가해 모든 paired 비교의 scene set을 일치시킨다.

---

## 4.5 wave-1 결과 (2026-08-31, node1; dossier F73)
| 셀 | Δ vs base 22.193 | 판정 |
|---|---|---|
| H1 oracle_both (GT 3D point, in+hidden) | **+2.081 (t=+49.8, 100%)** | 갈래 (a) 확정: 좌표가 문제였다. 위상 계열의 상한 = +2.1 dB |
| H2 shell_in (chord, 입력) | **+0.377 (t=+21)**, Plücker 입력 대비 +0.79 | 채택; imgvo와 PSNR 동률 |
| H2 shell_h (chord, hidden) | **+0.324 (t=+19)**, Plücker hidden 대비 +0.89 | 채택; hidden 사이트도 양수 → 합성 후보 |
| H7 camray + rot_raw | +0.343, rot_raw 대비 −0.08 (t=−5) | 기각 |
| 대조 attn_nope / attn_prope (node2) | +0.705 / **+1.437** | attention 상한; TTT+oracle(+2.08)이 attn+PRoPE를 +0.65 앞섬 |
| H4 hrot_rotraw (node2) | +0.410, rot_raw 대비 −0.01 (t=−0.7) | 기각 (hidden 행렬 작용 무효) |
| H10 imgvo_himg (node2) | +0.336, imgvo 대비 −0.06 (t=−3.4) | 기각 |
| shell_vo (chord 입력 + 회전 v/o transport) | **+0.532 (t=+26)**, shell_in 대비 +0.155, imgvo 대비 +0.137 | **orbit PE-only 최고**; 주소·carrier 슬롯 합성 성립 |
| shell_both (chord 입력+hidden) | +0.299, shell_in 대비 −0.078 (t=−6.1) | 두 사이트는 90°에서 비합성 (F52와 동일) |
| oracle_in (입력 토큰만 GT) | +0.426, shell_in 대비 +0.05; oracle_both 대비 −1.66 | oracle 이득은 **타깃 토큰의 depth** 효과 |
**vi(RayRoPE 재렌더) 결과 (F75)**: rot_raw **+0.533**, shell_vo +0.509(rot_raw와 동률), shell_both +0.375(shell_in 대비 +0.129 → vi에서는 두 사이트 합성), shell_in +0.247(Plücker +0.206 대비 +0.04), shell_h +0.062. vi에서는 carrier가 지배적이고 chord 좌표의 몫은 작다(기하가 좁을수록 좌표 효과 감소).
**vi wave 2 (F76)**: foot_in(최근접점 1개, 파라미터 0) **+0.472** = 주소 단독 최고(anchor +0.07, chord +0.23 상회); shell_all(입력 chord + hidden chord + carrier) **+0.630** = vi 최고 — 3슬롯 모두 합성. shell_h_vo +0.33(carrier의 짝은 입력 주소), imgvo +0.26(vi에서는 약함).
**wave 3 전반 (F77)**: rot_hshell(입력 회전 행렬 + hidden chord + carrier) **+0.716** = vi 최고 — 사이트 역할 배정 성립(rot_raw +0.18, shell_all +0.09, rot_shell +0.15). foot_vo +0.595(rot_raw +0.06). 비대칭 코드(asym_ck_qa) +0.228 = 대칭 chord와 동률 → key=chord 형태는 기각; anchor+carrier 비합성(+0.03).
**F78**: foot_all(foot 입력+hidden + carrier) **+0.717** = rot_hshell과 동률, 가장 단순한 레시피. od_in((o,d) 6D) −0.02 = Plücker보다 −0.23 → ray 좌표는 어떤 형태든 실패, 3D 점만 작동.
판정 트리 결과: H1 ≫ 0 → 갈래 (a)에 집중. 다음: shell_both, shell_vo(진행 중), anchor_*, oracle_in(타깃 depth가 병목인지), vi 데이터 재현.

## 4.6 wave 3 — TTT 고유 PE (사용자 요청, 5개 에이전트 ~57개 아이디어 → 7계열, 2026-08-31 저녁)
제약: PE만(q/k/v/o/h에 곱하는 카메라 기하의 함수), 추가 연산 ≈0(apply K배 기각), 새 layer·depth head 없음.
1. **비대칭 store/read 코드** (`asym_in`, 5/5 수렴): key=chord/foot, query=K anchor 블록 → query 깊이 가설의 OR을 apply 1회로; Hebbian readout(정규화 없음)에서만 의미. V3-1~4.
2. **비음 hidden kernel**: Fejér 조화 ladder(`fejer_h`), 뷰방향 bump 진폭 코드(`h_bump`). 선형 readout은 kernel 부호를 그대로 쓰므로 hidden 고유.
3. **사이트 역할 배정**: input=회전 행렬(제곱되어 들어감), hidden=3D-point kernel(선형), carrier=transport (`rot_raw+h_shell/fejer`).
4. **SwiGLU 브랜치 곱 kernel** (`gate_shell_rot`): gate=chord, content=회전 → AND.
5. **Vernier 2-사이트 ladder** (`omega_scale`≠`omega_scale_h`): 저주파 input이 고주파 hidden의 alias를 거부.
6. 층별 깊이 sweep / coarse→fine (미구현), 7. 좌표 정제(앞쪽 anchor, shell 교차점, 극좌표) (미구현).
기각: 뷰별 메모리 혼합, sweep 읽기(K배), 특징 주입, l≥2, 학습 query 코드. 구현됐지만 미실행: `sweep_in`(비용), `head_anchor`(층상 메모리; 백로그).

## 5. 실험 계획

프로토콜: F51과 동일(LaCT-LVSM L6/d256/p16, gObjaverse 256², 8+8 뷰, 30k, bs16, lr1e-4, LPIPS 5k~, seed 95,
`--min_frames 40`; eval 500 scene(499 유효), 8 uniform 입력 / 4 midpoint 타깃, per-scene paired t).
한 셀 ≈ 2 h/B200. 비교 기준은 항상 LaCT-LVSM NoPE baseline(같은 seed, paired). 승자만 s137/s211 확장.

| wave | node1 (GPU 0–3) | node2 (GPU 0–3) |
|---|---|---|
| 1 | H1 `oracle_both` · H2 `shell_in` · H2 `shell_h` · H7 `camray_rotraw` | 대조군 `attn_nope` · `attn_prope` · H4 `hrot_rotraw` · H10 `imgvo_himg` |
| 2 | H1 `oracle_in` · H2 `shell_iso_in` · H7 `camray_properaw` · H5 `h_dpra42` | H6 RayGTA · H8 rot_content · H3b depth-anchor · camray+h_rot (구현 후) |
| 3 | wave 1–2 승자 조합(예: shell + rot transport, camray + h_rot), RE10K 역검증(one-recipe 조건), 3-seed | (후순위) H9 epi_quad · H11 hgate · H12 dres2 |

**범위 규칙 (사용자, 2026-08-31)**: 모델 구조 변경 없이 positional embedding만 개발한다. 허용: q/k/v/o·hidden에
곱하는 직교 변환(위상·회전 행렬), 위상 좌표의 선택(ray → 3D 점/chord/anchor), 입력 raymap의 프레임(camray).
비허용: depth 예측 layer, 새 sub-layer. 진단(oracle, attention 대조군)은 방법이 아니며 논문의 방법 표에 들어가지 않는다.

판정 트리: H1 ≤ +0.1 → 갈래 (a) 폐기, (b)(c) 집중. H1 ≫ 0 & `oracle_in` ≈ 0 → 타깃 depth가 병목 →
H3(학습 depth/anchor)로. H7 > rot_raw → pose-free 체제 채택 후 H8·H4 적층. H4 ≤ rot_raw → hidden 사이트는
90°에서 포기하고 input+carrier 전용 처방.

---

## 6. 문헌 근거 요약 (서브에이전트 조사, 2026-08-31)
- **GTA** (2310.10375) Table 5: value/output 변환 제거 시 CLEVR-TR −2.45 dB, MSN-Hard −0.81; SO(3) irrep은
  object-rich 데이터에서만 추가 이득(translation 불변 carrier).
- **PRoPE** (2507.10496) Objaverse: Plücker 21.44 / CaPE 19.68 / GTA=PRoPE 23.70(공유 intrinsics면 동일);
  CamRay(pose-free 토큰) + GTA가 varying intrinsics 붕괴를 구제. Objaverse 카메라 분포는 미공개(issue #12).
- **RayRoPE** (2601.15275) v/o 제거 −0.40(CO3D)/−0.51(RE10K); **known depth: Objaverse 22.42 → 25.19**;
  ray-only(p_∞) −1.34; 불확실성(sinc) −1.12 — "3D 점 + 불확실성"이 wide baseline 이득의 원천.
- **DPPE** (2606.31585) object-centric MVImgNet2에서 projective/SE(3) carrier가 후반 학습을 망치고 **R만** 최선
  (R·t 결합의 per-token 비식별성) → rot_raw가 옳은 carrier.
- **URoPE** (2604.18747) 고정 depth anchor가 학습 depth보다 우수(26.01 vs 25.57), Objaverse에서 world-ray
  RoPE가 가장 약함. **LVT** (2509.25001) intrinsics-only 토큰 + relative pose 20.59 → 22.02.
- **ViewRoPE** (2602.07854) l=1 방향 rotary가 90–180°에서 열세. **SPAD** epipolar mask Objaverse +1.3 dB,
  단 180° 전후 모호성.
