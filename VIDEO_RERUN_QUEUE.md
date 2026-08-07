# Queued: video re-run matched to the CURRENT TTT-RoPE recipe (user, 2026-08-07)

F21/F22's "video null" tested a setting THREE steps removed from today's recipe:
hidden site only (input rope existed only in ccv configs), theta=10000 inverse-power
ladder on the (t,y,x) grid carrier, hrope_frac 0.5. None of {input site, pi-logspace
band, full coverage} was ever tried on plain video. Under the retrieval-disambiguation
hypothesis the memory's cargo (frames >= 2 SWA windows back, same scene) is
near-duplicate content, so the CURRENT recipe might well earn there -- the null may be
a setting artifact, not a task property.

## Cells (v20k protocol = F22: minVid ablation_small deriv., 20k steps, deterministic
## noise, paired per-step loss, 1 GPU each)

| cell | setting |
|---|---|
| video2_base | F22 base rerun under current code (paired reference) |
| video2_ttt  | input+hidden sites on the (t,y,x) grid, pi-logspace band scaled to the grid ranges, coverage ~98%, frozen freqs |

## Implementation prerequisite (NOT yet written)

`ttt_input_rope` currently asserts `cam_phase_mode == "plucker"`; plain video needs
the input site fed by GRID phases. Extend ar_lact_swa_repeat.py:
  - allow ttt_input_rope with cam_phase_mode "none" -> reuse the h-site grid carrier
    phases for fast q/k (t,y,x three-way split, same band scaling)
  - band: pi*logspace scaled so the top rung stays wrap-free over the grid ranges
    (t<=21, y/x<=latent hw) -- compute from the config, don't hardcode
  - verify: coords live (output moves when t/y/x permuted), stock parity at gain 0

## Where to run

1 GPU per cell, ~1+ day each (F22 scale). Slot AFTER the vo program drains on this
node, or node3 if it frees first. Do not co-schedule with Q35 pv (gpus 0-3).
