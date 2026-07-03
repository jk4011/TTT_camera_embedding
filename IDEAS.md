# Camera Embedding for LaCT/TTT — Synthesized Ideas

4개 ideation 서브에이전트(A: equivariance 이론, B: optimization dynamics, C: token-level 주입, D: 3D geometry)의 24개 아이디어를 사용자 기준으로 종합.

**사용자 우선순위 기준**: ① TTT/SwiGLU 구조 활용 ② 수학적 깊이 ③ relative 표현 ④ SE(3) invariance ⑤ 최소 아키텍처 변경

## 핵심 수학적 관찰 (Agent A의 Lemma)

LaCT fast weight `f_W(x) = (silu(x·w0) ⊙ (x·w2))·w1`, one-chunk update 후 query 출력:

```
o_j = h¹(q_j)·w1⁰  +  Σ_i lr1_i · ⟨h¹(q_j), h⁰(k_i)⟩ · v_i          (†)
```

1. **value 경로는 선형** — `v`는 (†)의 선형 채널로만 출력에 도달 (누출: `dhidden = v·w1ᵀ`가 w0/w2 grad에 미치는 영향뿐). 따라서 **per-view 변환 `O_i`를 v에(쓰기), `O_j⁻¹`를 출력에(읽기) 적용하면 payload가 정확히 relative** (`O_j⁻¹O_i`)로 전달됨 — SwiGLU 비선형성에도 불구하고 **exact**. PRoPE의 value-pathway를 TTT에 이식하는 이론적 근거.
2. **q–k 경로는 "초기 가중치까지만" relative** — update-유도 상호작용은 전부 `⟨q_j, k_i⟩` 내적 → per-view orthogonal `G_i` 적용 시 `⟨G_j q_j, G_i k_i⟩ = q_jᵀG_jᵀG_i k_i` (relative). 누출은 `q̃·w0⁰` 같은 초기-가중치 항뿐 (Gaussian init은 회전 불변이라 분포적 무해).
3. **Per-scene state = 공짜 gauge fixing** — fast weight는 장면당 하나이므로 canonical scene frame의 절대 인코딩은 이미 scene-relative. (데이터 전처리의 `normalize_with_mean_pose`가 canonical frame 제공.)
4. **Muon+weight-norm에서 살아남는 것** — global scale은 소멸, gradient 방향과 **토큰 간 상대적 lr 가중치**는 생존. per-token lr은 step size가 아니라 **token-selection 메커니즘**.

## 실험 변형 목록 (우선순위순)

| # | slug | 메커니즘 | ①TTT ②수학 ③rel ④SE3 ⑤최소변경 |
|---|------|---------|-----|
| 1 | `vo_rel` | 쓰기 시 `v ← rep(R_i)v`, 읽기 시 `o ← rep(R_jᵀ)o` (c2w 회전 85×3×3 블록). Lemma 1에 의해 Δw1 경로에서 **exact relative** `R_jᵀR_i` value transport | ★★★★★ |
| 2 | `qk_rope_cam` | L2-norm 후 q/k에 per-token orthogonal 변환: ray-direction rotary + translation phase (CaPE/GTA류). update-유도 커널이 정확히 relative | ★★★★★ |
| 3 | `prope_ttt` | 완전한 PRoPE 이식: `q←rep(P_jᵀ)q, k,v←rep(P_i⁻¹)k,v, o←rep(P_j)o` + re-L2-norm (P = lift(K̂)·w2c). 비직교 스케일은 lr/temperature 변조로 흡수 | ★★★★☆ |
| 4 | `plucker_sinc` | ray-segment 적분 rotary: 위상이 t에 선형이므로 `E[cos φ] = sinc(ω·d·Δt/2)·cos(φ_mid)` closed-form. fast weight가 3D 공간의 field가 됨. 파라미터 0개 | ★★★☆★ |
| 5 | `cam_lr` | camera-aware per-token 쓰기 lr (`lr_fc` 입력 확장, zero-init). **attention에 대응물이 없는 TTT-native 채널** (Lemma 4) | ★★☆–★ |
| 6 | `adaln_cam` | TTT layer마다 pose FiLM (zero-init) — "depth에서의 pose 접근"이 병목인지 확인하는 과학적 대조군 | ★☆☆–★ |
| 7 | `q_reinject` | query에만 zero-init pose 추가 (쓰기 경로에 증명 가능하게 무해) — read-path 대조군 | ★☆☆–★ |
| 8 | `cam_registers` | per-view 카메라 register KV쌍을 update set에만 추가 (v zero-init → 초기 baseline과 동일). update-set augmentation은 TTT 고유 | ★★☆–☆ |
| 9 | `hyper_init` | 카메라 집합 → DeepSets → fast weight 초기값 low-rank delta. inner optimization의 init을 조건화 (TTT 고유) | ★★☆–☆ |
| 10 | `point_rope` | depth head + 3D point rotary + 불확실성 수축 (RayRoPE 이식). 수학은 깊지만 arch 변경 필요 → 후순위 | ☆★★☆☆ |

**공통 설계**: 모든 rotary/변환은 q/k L2-norm과 직교성 호환; zero-init 게이트 변형은 초기에 정확히 baseline과 일치. Scene canonicalization은 데이터 레벨 `normalize_with_mean_pose`가 담당.

## 실험 라운드 (4× B200, 30k iters, L6/d256/p16)

- **R1**: baseline · vo_rel · qk_rope_cam · cam_lr
- **R2**: prope_ttt · plucker_sinc · adaln_cam · q_reinject
- **R3**: cam_registers · hyper_init · point_rope · (R1/R2 승자 조합, 예: vo_rel+qk_rope = full GTA-TTT)
- **R4**: 승자 조합/ablation

후보군에서 제외(추후 고려): steer-glu (수학적으로 가장 깊으나 arch 대변경), cam-tour-chunking (병렬성 훼손), two-pass-depth (복잡도), coverage-lr (cam_lr에 특징으로 흡수).

## 추가 Idea Direction (2026-07-03, Round 3 이후)

사용자 추가 기준: ⑥ 다른 방법론(PRoPE/GTA/RoPE)과 차별화 ⑦ 다른 task(LLM, video)로 일반화.

### 결과 기반 교훈 (R1–R2)
- 성공: qk_rope_cam +0.41, plucker_sinc +0.30 (기하 rotary addressing 계열만 성공)
- 실패: prope_ttt(projective 이식) −0.12, vo_rel(value transport) −0.12, 주입/변조 계열 −0.1~−0.6
- 결론: attention 레시피 직이식은 실패. Lemma 1 조건(직교성·post-norm·내적 경유)을 지킨 설계만 성공 → 이것 자체가 차별화 논거.

### Round 4 후보 (기준 ⑥⑦ 반영)
1. **pra_sinc**: 회전 예산 분할(line 96쌍 + segment 63쌍). 정확 좌표+구간 좌표 혼합 주소 공간
   = "uncertainty-aware rotary addressing" 프레임워크 (LLM span/video 시간 불확실성으로 일반화). R1·R2 1·2위의 상보 조합.
2. **steer_glu**: 위상-등변 SwiGLU (commutant 제약 초기 가중치 + modulus gate). 초기-가중치 누출을
   구조적으로 제거 → 비선형 fast weight의 exact equivariance. TTT 고유 이론, 좌표-불문 일반화. 구현 리스크 高.
3. point_rope (잔여, 후순위), R3 승자 조합.
