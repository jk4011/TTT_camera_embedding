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

## Gen-1 design notes (pending gain003)
- Map the gain curve: 1.0 (26.10) >> 0.1 (25.68); 0.03 running; if 0.03 < 0.1 ppl-wise,
  try 0.01; if worse, try 0.3 to bracket the optimum.
- Other mutation axes for gen-1: theta (1e4 — compresses the ladder range, different
  shape than a global gain); frac 0.75 x gain 0.1 (more dims, gentle); delta_only x
  gain 0.3 (tax removal where the tax is still sizable).

Selection rule: keep top-2 by fitness, mutate around them (one gene per child ±,
plus one crossover), population 3-4 per generation. Winners at proxy budget get a
full 3B confirmation run before any paper claim.
