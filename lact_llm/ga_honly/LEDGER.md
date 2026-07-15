# Q9: honly variant evolutionary search — ledger

Goal (user, 2026-07-10): make hidden-only positional encoding WIN in 1D LLM
("position 정보가 있는 게 없는 것보다 낫다"). Evolve variants: evaluate a small
population at proxy budget, select the best, mutate, repeat.

## Protocol (fitness)
- 200M LaCT (768/12L), fineweb-edu streaming, data_seed 42, bs 8x4096.
- Proxy budget: token_budget 655,360,000 = exactly 20,000 steps (~1.5h/run on one B200).
  Cosine schedule over the 20k steps (self-consistent proxy protocol; NOT comparable to
  mid-training values of the 91.5k runs — controls are rerun at this budget).
- Fitness = final val ppl on the ds42 val cache (lower is better).
- All variants: ttt_nope=true (hidden-only regime) + ttt_hidden_rope=true + genes.
- F27 context: at the FULL budget, honly (18.85) lost to nope (18.62) by +0.23 ppl —
  a flat absolute-phase tax on the initial readout with ~zero relative gain in 1D (F27b/c).

## Genome
| gene | meaning | F27 value |
|---|---|---|
| ttt_hrope_frac | fraction of hidden dims rotated | 0.5 |
| ttt_hrope_gain | global ladder multiplier | 1.0 |
| ttt_hrope_theta | ladder base (None -> rope_theta=1e6) | 1e6 |
| ttt_hrope_delta_only | rotate only the fast-weight delta path on apply (kills the F27b tax) | false |

## Generation 0 (2026-07-10, theory-guided seeds)
| run | genes | hypothesis | fitness (val ppl @20k) | vs nope |
|---|---|---|---|---|
| ga_honly_gain01 | gain=0.1 | gentler ladder: less scrambling, keeps coarse position | **25.68** | **−0.34 WIN** |
| ga_honly_delta | delta_only=true | tax removed, relative recall kept | 25.80 | −0.22 WIN |
| ga_honly_g01delta | gain 0.1 + delta_only (crossover) | genes compose | 25.85 | −0.17 (worse than gain01 alone: NOT additive — gentle ladder already shrinks the tax delta removes, and delta discards the cheap absolute prior gain01 keeps) |
| ga_nope_ctrl | (no hidden rope) | floor reference | 26.02 | 0 |
| ga_honly_ctrl | F27 defaults | tax reproduces at proxy budget | 26.10 | +0.08 |
| ga_honly_frac25 | frac=0.25 | fewer rotated dims | 26.11 | +0.09 (culled) |

GEN-0 VERDICT (2026-07-10): the user's hypothesis holds at proxy budget — hidden-only
position encoding CAN beat no-position (3 variants do). Best gene: LOW LADDER GAIN.
Note honly_ctrl is only +0.08 behind nope at 20k (vs +0.23 at 3B, F27) — consistent
with early-help/late-erosion (F27d). Full-budget risk therefore remains.

Context (F27d, same day): at 0.5B budget on ds43, plain hpra BEAT rope by −0.26 ppl —
the hidden rotary already wins early; the enemy is late-training erosion. The GA is
therefore optimizing "keep the early benefit, remove the late tax".

## Confirmation + gen 0.5 (running)
- ga_honly_gain01_full: WINNER at 3B tokens, full protocol (gpu7, ~6h) — vs F27 3B
  anchors nope 18.62 / honly_ctrl 18.85. THE decisive test of late-budget erosion.
- ga_honly_gain003: gain=0.03 (mutation, gain-direction resolution) — gpu0.

## Gen-1 (gain-direction line search, ds42 proxy)
| gain | fitness | note |
|---|---|---|
| 1.0 (ctrl) | 26.10 | |
| 0.1 | 25.68 | |
| 0.03 | **25.50** | leader 2026-07-10; top ladder freq period ~200 tok — no fast scrambling, recency-scale modulation only |
| 0.01 | (running, gpu0) | bracketing: gain->0 limit is nope (26.02), so an interior optimum exists in (0, 0.1) |
The win over BOTH nope (26.02) and gain->0 proves a genuine benefit of a slow, coarse
hidden position signal in 1D — not merely "less rotation is less bad".

## Draw-robustness (ds43, proxy budget) — SURPRISE REVERSAL
| run | ds42 | ds43 | 2-draw mean |
|---|---|---|---|
| nope_ctrl | 26.02 | 25.31 | 25.67 |
| honly_ctrl (plain) | 26.10 (+0.08) | **24.83 (−0.48!)** | 25.47 |
| gain01 | 25.68 (−0.34) | 25.15 (−0.16) — wins BOTH draws | 25.42 |
| gain003 | 25.50 (−0.52) | 25.26 (−0.05) | **25.38** |
Findings (2026-07-10):
1. ALL hidden variants beat nope on the 2-draw mean — the user's core hypothesis holds
   on average, not just per-draw.
2. But the optimal GAIN reverses across draws (ds42: 0.03 best, plain worst; ds43:
   plain best by a wide margin). The effect scale we are optimizing (±0.3–0.5 ppl) is
   the same scale as draw-to-draw swings.
3. PROTOCOL CHANGE: fitness = mean over ds42+ds43 from here on; and before further
   generations, MEASURE the init-seed noise of a proxy gap: gain01-vs-nope pair rerun
   at seed 137 (running, gpu1/6). If the gap noise is ~0.2+, single-run proxy deltas
   below that are not decisions.
gain01 remains the only variant that beat nope on both draws individually.

## Init-seed noise measurement (2026-07-10) — PROXY GAPS ±0.3 ARE NOISE
| pair (ds42, 20k) | seed 42 | seed 137 |
|---|---|---|
| nope | 26.02 | 25.72 |
| gain01 | 25.68 | 25.78 |
| gap (gain01−nope) | −0.34 | **+0.06 (sign flip)** |
Single-run proxy gaps of ±0.3 ppl flip with the init seed. PROTOCOL v3: fitness =
mean over (2 draws x 2 init seeds) = 4 runs per genome (~1.5h on 4 GPUs); only
differences > ~2x the empirical SE count as decisions. Single-run "leaders"
(gain003 25.50 etc.) are hereby demoted to candidates pending 4-cell evaluation.
Being filled: ds43 x s137 cells (nope, gain01), gain003 ds42-s137 + ds43-s137.
The 3B run remains the decisive test: full-budget gaps (~0.2, e.g. nope->rope −0.22)
have reproduced across environments, unlike 20k proxy gaps.
gain01_full trajectory vs same-step ds42 anchors: 53k: 21.11 vs nope 21.21 (−0.10);
65k: 19.78 vs nope 19.88 (−0.10); plain honly at 65k was 20.14 (+0.26). Holding.

## Gen-2
| run | genes | fitness | verdict |
|---|---|---|---|
| ga_honly_g003_frac100 | gain .03, frac 1.0 | 26.10 (ds42 s42) | REJECTED: rotating ALL hidden dims kills the gain even at slow frequencies — the position-free half of the hidden space (pure content pathway) is load-bearing. Theory-relevant negative. |

## PROXY PHASE CLOSED (2026-07-10): full 4-cell (2 draws x 2 seeds) results
| variant | gaps vs nope per cell | mean +- sd |
|---|---|---|
| gain01 | −0.34 / +0.06 / −0.16 / +0.21 | −0.06 +- 0.24 |
| gain003 | −0.52 / +0.13 / −0.05 / +0.23 | −0.05 +- 0.33 |
The 20k proxy cannot resolve these variants (effects ~0.05-0.1 vs noise sd ~0.25-0.33).
DECISION BUDGET = 3B. There, gain01 honly 18.53 beats nope 18.62 (F28), trajectory-stable.
Overnight 3B batch (launched): gain003_full (is 0.03 better at decision budget?),
gain01_full_s137 + nope_full_s137 (seed replicate of the F28 headline pair),
ga_hpra_gain01_full (does the low-gain hidden stack on input rope? vs rope 18.40).
  -> RESULT (recorded late, 2026-07-13): 18.415 vs rope 18.405 — stacking tax gone (plain hpra 18.639), increment nil (+0.01). See F28 addendum.

## Gen-3.5: hnorm variants (user idea 2026-07-11, implemented + verified)
Idea: the input site is rotation-friendly because q/k are L2-normalized (F27c geometry
hypothesis) — so RMS-normalize the hidden BEFORE the rotation and see if the hidden
site's tax drops. Gene ttt_hrope_hnorm: "rms" (whole hidden) / "rms_rot" (rotated dims
only; keeps the content half's magnitudes — frac100 lesson). Exact RMSNorm Jacobian in
the update backward (autograd-verified 3e-8); default "none" bit-identical.
Queue (45k matched-gap protocol, launch as gen-3 slots free):
- g3_hnorm_rms_45k: hnorm=rms, gain 1.0 — purest test: does normalization alone fix the
  PLAIN ladder (which lost by +0.23 at 3B)? If the geometry hypothesis is right, this
  should remove most of the tax without slowing the ladder.
- g3_hnorm_rmsrot_45k: hnorm=rms_rot, gain 1.0.
- (then) best hnorm x gain 0.1 — compose with the current champion.

## Gen-3 results (45k matched-gap protocol; fitness = last-10k mean gap vs g3_nope_45k 21.03)
| run | genes | gap | verdict |
|---|---|---|---|
| g3_theta100_g01_45k | theta 100 x gain 0.1 (all 48 pairs active) | **+0.17** | REJECTED — activating all pairs fails like frac100. UNIFIED PICTURE: at theta 1e6 the "frozen" pairs act as extra untagged content dims; what wins is FEW active pairs (~15 at gain 0.1: periods 63..4096 tok), all slow, none fast. The knob is "number of active slow pairs", not ladder coverage. |
| g3_gain02_45k | gain 0.2 | (running gpu0) | |
| g3_frac75_g01_45k | frac 0.75 x gain 0.1 | (running gpu7) | |
| g3_hnorm_rms_45k | hnorm=rms, gain 1.0 | (running gpu1) | |
| g3_hnorm_rmsrot_45k | hnorm=rms_rot, gain 1.0 | (running gpu6) | |
Gen-4 sketch: frac 0.25 x gain 0.1 (~8 active pairs) to probe the active-pair optimum
from below; hnorm x gain 0.1 composition if hnorm shows tax reduction.

## CALIBRATION SHOCK (2026-07-11): the 45k screen is ANTI-predictive
g3_gain01_45k (champion's own 45k anchor): +0.194 vs nope — the variant that WINS at 3B
(−0.09) LOSES at 45k-full-cosine. The gentle-hidden benefit only materializes late in
the schedule (LR-phase dependence, cf. F27d budget sensitivity). Corrected gen-3 reading
(relative to the +0.19 anchor):
- frac75 x g01: +0.02 -> −0.17 BETTER than champion at 45k. Projected 3B (if the
  45k->91.5k shift −0.28 is variant-additive): ~−0.26 — would BEAT ROPE. Promoted to 3B
  (g4_frac75_g01_full, gpu0).
- hnorm rms_rot (at plain gain 1.0): +0.12 -> −0.07 better than champion — the user's
  normalization idea WORKS on the tax axis (vs plain unnormalized at gain 1.0 which was
  ~+0.4-0.5 at this scale). Full-vector rms (+0.51) rejected: value magnitudes matter.
- gain02 (+0.25), theta100 (+0.17): at-or-below anchor, culled.
Methodology: abandon 45k as an absolute screen; use it only variant-vs-variant at the
same budget, and confirm anything promising at 3B.

## Gen-4 3B results (2026-07-11)
- g4_frac75_g01_full: **18.65** (vs nope +0.03, vs rope +0.24) — FAILED. The 45k winner
  (−0.17 better than champion there) is +0.12 WORSE than the champion at 3B. The
  variant-additive-shift hypothesis is DEAD: the 45k screen is uninformative for 3B
  outcomes even relatively. ALL ranking decisions in this family now require full 3B
  runs (6h each). In-flight 45k screens (rmsrot x g01, f75 x rmsrot x g01) demoted to
  curiosity data.
- Champion unchanged: gain 0.1 / frac 0.5 / no hnorm = 18.53 (−0.09 vs nope).
- Still live at 3B: g4_hnorm_rmsrot_full (gpu1). Untried-at-3B axes if the user wants
  to continue: rms_rot x g01, frac 0.25 x g01, per-layer gains (needs code).
- Honest odds assessment: every structured deviation from the champion has failed at 3B
  (0.03, 0.2, theta100, frac75, frac100); the remaining gap to rope (−0.13) exceeds the
  champion's entire measured gain (−0.07..−0.09 over two seeds). Beating rope hidden-only
  looks unlikely with ladder-shape genes alone.

## Gen-5: NEW CHAMPION — hnorm rms_rot at PLAIN gain (2026-07-11)
- g4_hnorm_rmsrot_full: **18.51** (vs nope −0.11, trajectory-stable) — beats gain01
  (18.53). The user's normalization idea wins WITHOUT slowing the ladder: RMS-normalizing
  just the rotated hidden dims removes the tax at full frequency coverage. Direct 3B
  validation of the F27c geometry hypothesis. Gap to rope: +0.11.
- (45k screen of rmsrot x g01: 21.80 — but 45k screens are proven uninformative.)
- Gen-5 launched (all 3B): rmsrot x gain0.1 composition (gpu0), rmsrot seed-137
  replicate (gpu1), delta_only at plain gain (gpu6 — the other tax-removal mechanism,
  never run at 3B).

## Gen-5 verdict (2026-07-11): the rms_rot championship DOES NOT REPLICATE
| run | 3B ppl | gap vs seed-matched nope |
|---|---|---|
| g5_rmsrot_full_s137 | 18.95 | **+0.27 — FAILS on seed 137** (s42 was −0.11) |
| g5_rmsrot_g01_full | 18.86 | composition worse than both parents (every composition in this program has failed) |
| g5_delta_full | 18.72 | +0.10, fails at 3B |
| g5_rope_rmsrot_full | 19.00 FINAL | stacking v2 fails (+0.60 vs rope — worse than plain stacking +0.24) |
CORRECTION: rms_rot is seed-FRAGILE (−0.11 / +0.27), not a champion. The only variant
with 2-seed 3B consistency remains gain 0.1 (−0.09 / −0.04, mean −0.07).
PROGRAM CEILING ESTIMATE: after ~15 3B runs + ~20 proxy runs, the hidden-only channel's
robust 1D gain is ~−0.07 ppl — about a third of rope's −0.22 — and every attempt to push
past it (slower/faster ladders, fractions, normalization, delta path, compositions,
stacking) has failed at 3B or proven seed-fragile. RECOMMENDATION to the user: close
Q9-EXT with F28 (honest small-but-consistent + the anatomy of what wins), redirect to
Q10 video where the coordinate is multi-dimensional and the theory says the hidden
address space has real relative structure to exploit.

## Gen-1 design notes (pending gain003)
- Map the gain curve: 1.0 (26.10) >> 0.1 (25.68); 0.03 running; if 0.03 < 0.1 ppl-wise,
  try 0.01; if worse, try 0.3 to bracket the optimum.
- Other mutation axes for gen-1: theta (1e4 — compresses the ladder range, different
  shape than a global gain); frac 0.75 x gain 0.1 (more dims, gentle); delta_only x
  gain 0.3 (tax removal where the tax is still sizable).

Selection rule: keep top-2 by fitness, mutate around them (one gene per child ±,
plus one crossover), population 3-4 per generation. Winners at proxy budget get a
full 3B confirmation run before any paper claim.


## Q12: STACKING program (user goal 2026-07-13/14: make hidden ADD on top of input rope; /goal = hidden increment in all 3 tasks)
Target: rope+hidden < rope 18.405 (3B ds42). Baseline stacked: hpra-g0.1 = 18.415 (neutral).
Wave 1 (3B finals, 2026-07-14): ALL FAIL —
| variant | ppl | vs rope |
| stack gain 0.2 | 18.46 | +0.05 |
| stack gain 0.05 | 18.65 | +0.24 |
| stack frac 0.25 x gain 0.2 | 18.70 | +0.29 |
Reading: the stacked optimum sits at ~gain 0.1 and is neutral; simple gain/frac axes do
not stack. Wave 2 (in flight): (a) q12w2_stack_hnrot_g01 = rope + hnorm rms_rot x gain
0.1 (F27c geometry hypothesis applied to stacking; hidden-only it was seed-fragile),
(b) q13L pair = learnable ladder init at the champion spectrum, per-layer vs SHARED
across layers (Q13 idea; shared = cross-layer regularization). Note: q13L_shared runs
at ~70k tok/s vs 141k (perlayer) — shared Parameter appears to halve throughput
(compile/fusion effect); science unaffected, wall doubles.
Q13-NVS side result (seed 95): shared learnable 22.338 vs per-layer 22.420 — sharing
does NOT help the NVS input-site gains (single seed, within noise of fixed 22.389).


## Q12 wave 3+4: BREAKTHROUGH — shared learnable hidden ladder stacks (2026-07-14)
Wave 2 results (all fail): stack hnorm-rms_rot 19.11 / learnable per-layer 18.63 /
delta-only 18.62. 2-seed check of the fixed gentle ladder: s137 rope 18.19 vs
hpra-g0.1 18.38 (+0.19) — the s42 "neutral" (+0.01) was the favorable draw; the FIXED
gentle ladder does not stack across seeds.
Wave 3 attribution triangle (s42): shared learnable {hidden+input} 18.371 (-0.034,
gap stable -0.030..-0.034 over the last 6.5k steps) -> decomposed:
  sharedH (hidden ladder only) **18.204 (-0.200 vs rope)** | sharedI (input deltas
  only) 18.73 (+0.33, harmful) | combined s137 18.44 (+0.25, input deltas dominate).
THE ACTIVE INGREDIENT: ONE learnable hidden ladder SHARED by all 12 layers, init at
the gentle g0.1 spectrum, input rope untouched. Gap trajectory -0.20..-0.28 from step
7k to the end (not an LR-tail artifact). Same recipe per-layer degenerates (18.63) —
cross-layer sharing is what rescues the learnable ladder (Q13 idea; opposite of NVS
where sharing was neutral-negative).
Wave 4 (in flight): sharedH s137 + s211 (3-seed), honly+sharedH s42 (new 2x2 cell).
If replicated: LLM increment = -0.20, comparable to the input rotary's own -0.22,
and the /goal (hidden adds in all three tasks) is complete.


## Q12 wave 4 verdict: sharedH is SEED-FRAGILE (2026-07-15)
3-seed replication FAILED: s42 18.204 (-0.200) / s137 18.75 (+0.56 vs rope 18.19) /
s211 18.92 (worse than nope; rope_s211 reference queued). honly+sharedH s42 = 18.36
(beats nope by -0.26 and fixed honly-g0.1 18.53 — but s42-only, same suspicion).
Diagnosis: both s42 and s137 gap trajectories are SMOOTH and sign-stable from step
~4k — the outcome is set by the initial condition (the +-10% random tilt draw x model
init), not by late training. Same failure family as rms_rot (F29).
Wave 6 (in flight): tilt=0 deterministic init at the g0.1 spectrum, s42+s137 — if the
fragility is the tilt lottery, this rescues the recipe; if it is the learnable-ladder
training dynamics itself, the learnable line dies for good. Also queued: rope s211
reference. Side wave-5 runs (sharedHI0, initg1, both s42-only) will be read in this
light.


## Q14 wave 1: dense U-basis fails via NORM DISTORTION; rotation earns vs its own control (2026-07-15)
s42, refs rope 18.405 / hpra-g0.1 18.415 / honly-g0.1 18.53 / nope 18.62:
| ropeUctl (U only, zero phases) | 18.68 | the dense in-path matrix ALONE costs +0.28 |
| hpraU (rope + hidden g0.1 + U) | 18.55 | loses to rope, but beats its control by -0.13 |
| honlyU | 18.74 | |
Reading: a dense learnable matrix in the hidden ADDRESS path distorts address norms
-> collides with the weight-norm regime — the same mechanism as F3's projective
failure, now reproduced in 1D. The organize-a-subspace hypothesis is not refuted
(rotation earned -0.13 on top of the U burden); the implementation was norm-unsafe.
Wave 2: ORTHOGONAL U (h_basis = matrix_exp(A - A^T), A zero-init: exact reduction at
init, norm-preserving always, wd pulls toward I) — queued after the Q15 NVS runs.


## Q14 wave 2 verdict: orthogonal U also fails — learned-basis line CLOSED (2026-07-16)
s42: ropeUorthctl (expm(A-A^T), zero phases) 18.55 (+0.15 tax; orthogonality halves the
dense +0.28 but does not eliminate it) | hpraUorth 18.56 (increment over its own control
= ZERO; dense showed -0.13). Residual tax = feature-axis identity mixing + optimization
interference, not norm distortion. Conclusion: no learned-basis parameterization (dense,
orthogonal) unlocks a 1D hidden increment. LLM attack moves entirely to the structural
levers: Q16 (exact-offset copy, precision+load maxed) and Q17-A (window 128, load-bearing
memory in natural language).
