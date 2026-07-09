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
| run | genes | hypothesis | GPU | fitness (val ppl @20k) |
|---|---|---|---|---|
| ga_nope_ctrl | (no hidden rope) | floor reference | gpu7 (after mlp2) | |
| ga_honly_ctrl | F27 defaults | tax reproduces at proxy budget | gpu0 #2 | |
| ga_honly_delta | delta_only=true | tax removed, relative recall kept -> should beat ctrl, target: beat nope | gpu0 #1 | |
| ga_honly_gain01 | gain=0.1 | gentler ladder: less scrambling, keeps coarse position | gpu6 (after mlp2) | |
| ga_honly_frac25 | frac=0.25 | fewer rotated dims: tax scales down, dictionary keeps more freedom | gpu7 #2 | |

Selection rule: keep top-2 by fitness, mutate around them (one gene per child ±,
plus one crossover), population 3-4 per generation. Winners at proxy budget get a
full 3B confirmation run before any paper claim.
