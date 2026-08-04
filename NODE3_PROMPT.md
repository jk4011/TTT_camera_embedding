# NODE3 prompt — paste this whole file as the first message

You are the node3 Claude session for the TTT-RoPE project. Repo:
`/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope` (on lustre, shared with node1).
Read `CLAUDE.md` first. node1 is running the tttLRM 3D-reconstruction grid; you
own the language-model diagnostics on node3's 4 GPUs.

**Before anything else: `git pull`.** node1 just fixed a bug that silently
invalidated half of the previous run; without the pull you would reproduce it.

---

## Background: what went wrong last time, so you can recognise it

The first Q29 grid completed all 7 cells and returned **bit-identical numbers for
every 2d/1d pair**. Cause: `train_small.py` populated `ttt_layers` only inside
the Q24-intervention branch, so `set_ext_coords()` looped over an empty list.
Nothing raised — every arm just ran the stock rotary, which is indistinguishable
from a genuine null result.

Two things now exist because of it, and you must not disable either:

1. `ttt_layers` is populated unconditionally.
2. `verify_clrs_coords_active()` runs at startup for `--data clrs` and
   `--data grid`: one forward per coordinate mode on the same batch, and with a
   rotary enabled the two **must** differ. Each cell's log must contain
   `[clrs] COORD VERIFIED ACTIVE ... |d|=...` (or, for the `base`/nope cell,
   `address correctly inert`). **If that line is missing, the cell is invalid —
   stop and report rather than letting it run.**

The invalid runs are quarantined as `outputs/q29_*_INVALID_stockrotary`. Do not
delete them and do not reuse them.

---

## JOB 1 — finish Q29 (CLRS-Text), 3 cells

```bash
cd /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope && git pull
cd lact_llm
./run_clrs_grid.sh 0,1,2,3
```

The script skips cells that already have `final.pt`, so only `q29_in_2d`,
`q29_h_2d`, `q29_both_2d` will run. `q29_base` and the three `*_1d` cells are
already valid and complete — `coord_mode=1d` feeds `(t, t)`, which recombines
the split ladder into `inv_freq * t`, i.e. bit-identically the stock rotary
(verified `0.000e+00` by `sanity_clrs_coords.py` mode a). That is exactly what
those runs executed, so they ARE the 1-D arm.

Already in hand from the valid cells (`answer_acc`, held-out CLRS test seeds):

| arm (all 1-D address) | answer_acc |
|---|---|
| base (NoPE) | 0.8559 |
| h (honly) | 0.8240 |
| both | 0.8182 |
| in (rope) | 0.8164 |

So with a 1-D address the rotary *hurts* on CLRS, and the hidden site hurts
least. Job 1 fills in the 2-D arm.

**Pre-registered prediction:** the h-over-in increment grows in the 2d arms and
does not in the 1d arms. If 2d looks just like 1d, the dimensionality hypothesis
(F20) is refuted for this task and we report that.

## JOB 2 — Q30, N-dimensional tensor recall (new)

```bash
cd /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope/lact_llm
python synthetic_grid.py            # self-test, expect ALL PASS
python synthetic_grid.py strides    # prints the quantity the sweep is about
DIMS="2 4 6" ./run_grid_diag.sh 0,1,2,3
```

**Why this task exists.** F35 (`RESULTS_DOSSIER.md:469`) was our exact-offset
copy diagnostic and it saturated: NoPE 0.2%, but rope / honly / hpra **all** hit
100%, so it separated the arms only by convergence speed and could not support
any claim that the hidden site beats the input rotary. The cause is structural —
the copy offset is one hardcoded constant (2560), so the model must represent
exactly ONE relative distance, and any positional code can do that.

**The data is genuinely d-dimensional.** A tensor of shape `SHAPES[d]` holding
1024 random tokens is serialised row-major into the sequence. The element count
is 1024 at every d, so memory load is constant and only the lattice changes. The
query names an AXIS, fixes the other d-1 indices, and the answer is that FIBER.
Enough fibers are asked per sequence to keep supervision at exactly 32 tokens for
every d, so answer length is never a confound. All tokens come from one
distribution — nothing is findable by content, only by address — and the tensor
sits >128 tokens before every answer, so the fast-weight path must carry it.

**What the sweep measures.** In the flat serialisation a fiber along axis a sits
at stride `prod(shape[a+1:])`, so a FLAT address has to resolve one stride per
axis:

| d | shape | strides a flat address must resolve |
|---|---|---|
| 2 | (32,32) | 32, 1 |
| 3 | (16,8,8) | 64, 8, 1 |
| 4 | (8,8,4,4) | 128, 16, 4, 1 |
| 5 | (4,4,4,4,4) | 256, 64, 16, 4, 1 |
| 6 | (4,4,4,4,2,2) | 256, 64, 16, 4, 2, 1 |

A d-D address gets all of them for free: the d-1 fixed axes contribute phase
difference **zero** between query and source, and only the free axis carries the
offset.

**Cells.** Per d: `base` (no rotary — the floor, i.e. how much harder the task
itself gets as the lattice deepens) plus `{in, h}` x `{nd, flat}`, where `nd`
supplies the true d-D index and `flat` supplies the token position, which
recombines the ladder bands into `inv_freq*t` and is therefore **bit-identically
the stock rotary**. So `nd` vs `flat` isolates address dimensionality and nothing
else. 5 cells per d, 800M tokens each. Start with `DIMS="2 4 6"` (15 cells,
~9 h on 4 GPUs); fill in d=3,5 afterwards if the trend is worth refining.

**Pre-registered prediction — two forces, so expect a peak not a ramp.** Rising d
gives the flat address more stride scales to resolve, so the `nd - flat` gap
should GROW. But `_ext_angles` partitions the ladder into d contiguous bands, so
each axis keeps only ~P/d frequencies and eventually becomes too coarse for its
own (already short) side. The load-bearing question is whether the **hidden**
site's `nd - flat` gap grows faster, or peaks later, than the **input** site's —
that is exactly the claim that the hidden site is the multi-dimensional-address
mechanism. Where the peak sits also speaks directly to NVS, which splits the
ladder **6 ways** for Plücker coordinates and has never been checked against a
dimension sweep.

## JOB 3 — Q31, remove the ATTENTION rotary (new, 4 cells) — **run this BEFORE Job 2**

```bash
cd /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope/lact_llm
./run_attnnope_grid.sh 0,1,2,3
```

**The hypothesis, and why it outranks the dimension sweep.** Our 1-D language null
has always been explained as "language is content-addressed". There is a simpler
candidate nobody tested: LaCT's LM keeps a rotary on its **sliding-window
attention**, so every local position already carries an explicit relative code and
the fast-weight rotary has little left to contribute. A reviewer will ask this about
Table 5, so we should have the answer.

It is a cleaner lever than F36's window shrink:

| lever | what it changes | side effect |
|---|---|---|
| window 1024 → 128 | attention **capacity** | absolute ppl degrades (18.40 → 18.61) |
| `attn_nope` | the positional **channel** only | capacity untouched |

With `attn_nope` the TTT rotary is the model's only explicit positional code.
**Prediction:** the four arms — which sit at 18.58 ± 0.06 and are mutually
indistinguishable at w128 across 3 seeds — should SEPARATE. If they do not,
"attention already supplies position" is refuted and the content-addressing
explanation stands on its own. Either outcome is publishable.

Only the attn-rope-OFF column runs: the ON column is F27 at the identical protocol
(200M LaCT, 3B tokens, ds42, bs 8×4096, window 1024), so do not re-run it.

Wiring is already verified on node1: `attn_nope` reaches every layer and changes the
forward (|Δloss| = 2.7e-04 on a matched state_dict). It is a plain config flag —
`getattr(config, 'attn_nope', False)` — so old checkpoints and configs are unaffected.

**Honest scope for the write-up:** causal masking still leaks position, so this is
"no explicit code", not "position-free" — that leak is exactly why NoPE transformers
work at all (Kazemnejad et al.). And absolute ppl will probably get *worse*, because
we removed a useful code. Report it as a channel-value decomposition, never as a
SOTA claim.

---

## Order

1. **Job 1** — finish Q29 (3 cells). Half-done already, cheapest to close.
2. **Job 3** — Q31 attn_nope (4 cells). Directly answers a question the LLM table
   invites; small.
3. **Job 2** — Q30 dimension sweep (15 cells). Largest and most exploratory; run last.

---

## Rules that apply to both jobs

- **GPU locks** are on lustre and therefore SHARED BETWEEN NODES. Both launchers
  already scope them by hostname (`<host>_gpu<i>`). Never drop the prefix.
- **Launch as your own background Bash task** (`run_in_background: true`) so the
  harness notifies you. Externally-started nohup daemons finish silently.
- **Report paired**, never unpaired: per-item paired deltas with a t-statistic,
  seed-matched, win-rate alongside the mean.
- **PAPER FREEZE.** Record results in `RESULTS_DOSSIER.md` /
  `lact_llm/ga_honly/LEDGER.md` only. Do not touch paper claims.
- If a run finishes and the queue is empty, say so explicitly rather than
  leaving GPUs idle.
- Both of these are **diagnostics on synthetic or procedurally-generated data**.
  A win here is not a natural-language result and not a real-data result; our
  3-seed LM null stands. Say so in any writeup.

## Honest statement of where the project stands

Outside camera tasks we have **no** win over the input rotary yet. F35 was a
convergence-speed result, not a capability gap. These two jobs are the attempt
to find one, and both are designed so that a null is reportable rather than
ambiguous.
