# Related-work survey — positional encoding in fast-weight / linear-attention / SSM states
(2026-08-03, agent survey; citation counts = Semantic Scholar unless noted. NOT a paper file —
this is a working record, the paper freeze does not apply here.)

## Verdict on our core site
**Partially occupied.** Rotating the **q/k of a LINEAR state** is crowded and well published.
Rotating the **hidden activation of a NONLINEAR (SwiGLU/MLP) fast weight** — our h-PRA site —
is occupied by **nobody**, under any phrasing searched.

## Two findings that constrain the claim (verified in code/text, not from search snippets)
1. **The upstream LaCT and TTT-LM code ALREADY applies RoPE to the fast-weight q/k.**
   `lact_llm/lact_model/layer_lact_swiglu.py` (first-import commit 77c9ff0) and
   `ttt-lm-pytorch/ttt.py` (`apply_rotary_pos_emb(XQ, XK, ...)`, TTT-Linear AND TTT-MLP).
   **Neither paper documents it.** => our input-rotary result is a REPRODUCTION, not a novelty.
   Contrast: `lact_nvs/lact_ttt.py` has zero rotary references (the NVS input site IS ours).
2. **LaCT §7 names our problem as open**: "our SwiGLU and Linear Fast Weight components do not
   exhibit [rotation invariance] ... the practical implications of this absence remain
   underexplored." => pitch the lemma as resolving a limitation the LaCT authors stated.

## Competitor table (most dangerous first)
| Work | Affiliation | Venue/year | Cites | Class | What it occupies |
|---|---|---|---|---|---|
| **SANA-WM** (2605.15178) | **NVIDIA** | preprint 2026-05-14 | 11 | **closest collision** | Camera 4x4 transform composed block-diagonally with RoPE on the Q/K of a **Gated DeltaNet**. Camera rotation inside a LINEAR fast-weight state. |
| Selective RoPE (2511.17388) | ELLIS/MPI-IS + EPFL | ICLR 2026 | 7 | adjacent | Input-dependent learnable rotation for gated linear transformers/SSMs; 1-D text only. |
| Mamba-3 (2603.15569) | CMU/Princeton | ICLR 2026 | 67 | adjacent | Complex SSM == real SSM with data-dependent RoPE on B,C. "Rotation IS the recurrence's position mechanism." |
| PaTH Attention (2505.16381) | MIT+IBM (Songlin Yang, a LaCT co-author) | NeurIPS 2025 | 28 | adjacent | Householder-product PE; delta-rule-derived but implemented in attention. |
| LRPE (2307.09270) | Shanghai AI Lab | TMLR 2023 | 18 | adjacent | Theory of which RPEs survive the linear-attention kernel decomposition. |
| **Decay design space** (2509.05282) | Qin et al. | COLM 2025 | 2 | **the negative result** | "RoPE ... typically fails to provide tangible benefits to the majority of linear attention mechanisms." Our LLM parity is consistent with this; must address head-on. |
| FLT (2302.01925) | Google DeepMind | AISTATS 2024 | 13 | closest IDEA | Learned RPE for linear attention over multi-dimensional GEOMETRIC coordinates (3D molecules). |
| RetNet (2307.08621) | MSRA | arXiv 2023 | 731 | adjacent | Complex decay = xPos-style relative position baked into the recurrent state. |
| cosFormer (2202.08791) | — | ICLR 2022 | 334 | adjacent | Cosine re-weighting as decomposable relative position in linear attention. |
| RWKV-7 (2503.14456) | RWKV/EleutherAI | preprint 2025 | 131 | superficial | Generalized delta rule (fast-weight family) but NO explicit PE; moved away from rotary. |
| **Titans** (2501.00663) | Google Research | NeurIPS 2025 | 291 | **key negative evidence** | The other major NONLINEAR MLP fast weight: verified no rotary / no PE on its addressing. Also already has a DNA eval table => never claim "first TTT for DNA". |
| DeltaProduct (2502.10297) | Freiburg | NeurIPS 2025 | 55 | superficial | Householder products as state transitions, for expressivity not position. |
| LieRE (2406.10322) | Stanford | ICML 2025 | 13 | superficial | Lie-algebra generators generalizing RoPE; attention only. (Ancestor of our ttt_liere.) |
| TTT-KV-Binding (2602.21204) | NVIDIA/Berkeley (incl. a PRoPE author) | ICML 2026 | 6 | supporting + scoop risk | TTT == learned linear attention. No pose/RoPE, but this group is one step away. |

Camera-pose RoPE **in attention** (the lineage we port): PRoPE (NeurIPS 2025, 82), ViewRope
(2602.07854, 7), UCPE (CVPR 2026, 26), RayRoPE, DPPE, GTA (ICLR 2024).

## Domain verdicts
| Domain | TTT/LaCT-layer paper? | Closest adjacent | Why not our claim |
|---|---|---|---|
| Ego-pose world models | **NO for TTT**; YES for a fast-weight (GDN) with camera-transformed Q/K | SANA-WM | Linear state (rotation invariance is textbook there); camera transform sits in a zero-init SIDE BRANCH, not the main state's addressing; encoding transplanted from attention (UCPE). |
| DNA / genomics | **NO** | Titans has a DNA eval table; HAD (2505.20836) = bidirectional Gated DeltaNet distilled from NT-v2 | Downstream eval / distillation claims, no PE in the memory. GENEB benchmarks 40 genomic models: zero TTT/DeltaNet/Titans. |
| Symbolic music / audio | **NO** (12 query formulations, ISMIR/Interspeech/ICASSP sweeps; LaCT never mentions audio) | MIDI-RWKV (SFU, 2506.13001) | RWKV-7 is fast-weight family but positionally agnostic; its "test-time" part is OFFLINE state tuning for style. Music Transformer's relative position is a term in the QK^T logit matrix made tractable by "skewing" — presupposes an LxL matrix a fast weight never forms. |

## Where we are genuinely unclaimed
- **h-PRA**: rotating the SwiGLU hidden between h and W1 on update AND apply with the inverse
  rotation in the manual backward. No analogue exists in attention (there is no hidden layer
  between logits and values), and no work rotates the hidden activation of any nonlinear fast weight.
- The **readout-decomposition lemma** (two independently rotatable channels in a gradient-updated
  nonlinear fast weight).
- **6-DoF camera/Plucker phases for a memory that is provably not rotation-invariant** (LaCT §7's gap).
- TTT layers in **DNA** and **symbolic music** at all.

## What we must concede
- Input-side "RoPE on fast-weight q/k" is NOT novel for 1-D language (upstream code).
- "Camera-derived rotation composed with RoPE on a fast-weight state" is occupied by SANA-WM
  (2026-05-14). Lead with **nonlinear state + hidden channel + derived-not-transplanted**.
- Correct cross-task phrasing: "first to make the fast-weight ADDRESSING itself positional, with
  DNA and music as cross-task evidence" — not "first TTT for DNA", not "first fast-weight music model".

## Two specific answers
- **CVPR 2025 TTT video paper** (2504.05298, 109 cites): text-storyboard -> one-minute cartoons.
  ZERO mentions of camera / ego-motion / viewpoint / 3D. Does not contest anything of ours.
- **LaCT** itself = "Test-Time Training Done Right", **ICLR 2026**, MIT+Adobe, 105 cites.
  Follow-ups are ALL 2026 and ALL vision, none doing pose-in-fast-weights: tttLRM (CVPR 2026,
  9-D ray maps CONCATENATED to input tokens, no rotary), FSM (2604.07350, Plucker maps
  concatenated; RoPE only as a timestamp ablation), TTT-KV-Binding (ICML 2026), ZipMap, TTT3R.
  => input-token conditioning is still the field default, which is exactly what we displace.

---

# ADDENDUM (2026-08-04): the streaming-3D LaCT family — SAME-YEAR COMPETITORS
The table above was materially incomplete. Four+ 2026 papers already use LaCT SwiGLU fast weights
for streaming 3D/4D reconstruction. **Every one conditions camera by concatenating Plucker/ray maps
to INPUT TOKENS** — i.e. they all use our baseline, the thing our NVS ablation beats by +1.7 dB.

| Work | arXiv | Memory | Camera conditioning | Threat |
|---|---|---|---|---|
| tttLRM | 2602.20160 (CVPR 2026) | LaCT fast weight | 9-ch ray embedding concatenated "as the positional embedding" | baseline, not competitor |
| FSM (Fast Spatial Memory) | 2604.07350 | LaCT SwiGLU FW + elastic (EWC) prior | Plucker maps concatenated; RoPE only as a *timestamp* ablation | **best adapter host** (code + HF ckpts, RE10K/DL3DV, 256x256 = our setting) |
| Mem3R | 2604.07279 | multihead SwiGLU FW (W0,W1,W2 in R^64x64 x12) = our layer | input-token conditioning | best host for the HIDDEN site |
| ZipMap | 2603.04385 | TTT state | input-token conditioning | — |
| TTT3R | 2509.26645 (ICLR 2026) | recasts CUT3R state as a fast weight | **learning-RATE conditioning** (beta_t = sigmoid(sum Q_S K_X^T)); "query/key projections and attention computation remain frozen" | closest philosophy competitor — and our F6 says lr-conditioning HURTS |
| Point3R | 2507.02863 | growing pointer bank + cross-attn | **"we change the 1D token index n in RoPE to a 3D token position"** | validates 3D-relative addressing, but pays quadratic memory |
| CUT3R | 2501.12387 | cross-attn over 768 state tokens | raymap-only query ("state is not updated here") | **no hidden site exists there** — do not target |
| HorizonStream | — | hybrid | "Spatiotemporal RoPE" on its LOCAL QUADRATIC branch only | — |

**Reading**: the field converged on our exact layer and spent every intervention on the LEARNING RATE,
never on the addressing. arXiv sweeps for `"recurrent state" AND "camera pose" AND "rotary"`,
`"spatial memory" AND "relative pose"`, `"Plucker" AND "linear attention"` return ZERO.
Licences here are mostly CC-BY-NC-SA (MapAnything's apache variant is the exception).

## Corrections to earlier claims in this file / my analysis
1. **IPA is NOT the same trick as PRoPE.** Verified in OpenFold structure_module.py: IPA logits are
   q.k + linear_b(z) + an ADDITIVE term -0.5*sum||T_i o q_vec - T_j o k_vec||^2, with points pushed to
   the global frame and outputs pulled back. Expanding the square does give a q^T(R_i^T R_j)k bilinear
   term (so it IS relative-SE(3)), but it is parameterized as an additive pairwise bias in an LxL logit
   matrix plus value transport — precisely the form that CANNOT port to a fast weight. (Our F4 already
   found value transport neutral-to-harmful.) The multiplicative rotary form is the one that transfers.
   NOTE: this expansion is our derivation from verified code, not a published claim.
2. **AlphaFold's sequence PE is 1-D**: AF2 relpos = one-hot bucketed residue-index difference (r_max 32)
   into the PAIR rep. AF3/Boltz/Chai/Protenix kept a 1-D token index + chain/entity one-hot and
   **dropped frames and IPA from the trunk entirely**. ESM-2's RoPE is on the 1-D residue index.
   => protein LANGUAGE models sit on our NULL side; only the IPA/frame family is on the NVS side.
3. **The protein bottleneck is the LxL PAIR tensor + triangle ops** (triangle attn 8L^3 C_z), NOT the
   sequence-axis attention. "TTT makes AlphaFold long-context" is FALSE and a reviewer kills it in one
   line. The surviving pitch: IPA is a relative-SE(3) inner product trapped inside a softmax; FlashIPA
   (2505.11580, MIT) already proved it factorizes per-token (rank-r pair rep, "avoiding materializing
   any quadratic object") and then KEPT the softmax; we supply the linear-time fast-weight memory that
   factorization was missing — in the generative FRAME family (FrameDiff/FrameFlow/FoldFlow), which has
   already shed the pair tensor. UNRESOLVED: FlashIPA's abstract claims linear wall-clock, body
   reportedly says compute is still O(L^2) due to softmax — must crack the PDF before claiming.
4. **Robot manipulation**: 3D Diffuser Actor / Act3D use RotaryPositionEncoding3D which is
   **axis-wise sin/cos over x,y,z only — translation-only, abelian; gripper ORIENTATION never enters
   the phase**. A concrete code-verified gap our SE(3) formulation fills. But standard protocols use
   n_obs_steps=2 (Diffusion Policy) / current observation only (3D Diffuser Actor) / single frame
   (OpenVLA) => criterion (c) FAILS; linear-time memory buys nothing unless we build the long-context
   setting ourselves. RoboTTT (2607.15275, NVIDIA) does scale TTT to 8K timesteps with no PE on the
   fast-weight addressing — but is real-robot-only, no code/checkpoints.

## Domains that FAIL our criteria (do not pursue)
cryo-EM (particles encoded independently -> no update->apply inner product to relativize; no pretrained
backbone); event cameras (state addressed by ARRAY INDEX not an inner product; (x,y,t) lattice is
abelian/separable = exactly the published linear-attention RoPE null); MLIPs (already O(N.k); long-range
already solved classically by Ewald message passing exp(ik.r) — severe novelty hazard); static point
clouds (PTv3 REMOVED relative position bias in favour of xCPE; already O(N)); gridded weather (no
retrieval — dense field transform; Aurora max_history_size=2); geospatial RS (pre-registered grids
collapse geography to token index); motion FORECASTING (QCNet radius graphs make attention already
sparse O(E)); medical 3D/4D (sliding windows keep token counts small); crystals (MP-20 is <=20 atoms).
Bonus theory-fit but no baseline: radio interferometry (van Cittert-Zernike kernel exp(-2pi i(ul+vm+w(n-1)))
IS a rotary phase in (u,v,w); 10^8-10^10 visibilities; but no adaptable model exists).
