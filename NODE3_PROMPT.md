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

## JOB 2 — Q30, the grid-recall diagnostic (new), 7 cells

```bash
cd /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope/lact_llm
python synthetic_grid.py            # self-test, expect 12/12 PASS
./run_grid_diag.sh 0,1,2,3
```

**Why this task exists.** F35 (`RESULTS_DOSSIER.md:469`) was our exact-offset
copy diagnostic. It saturated: NoPE 0.2%, but rope / honly / hpra **all** hit
100%, so it separated the arms only by convergence speed and could not support
any claim that the hidden site beats the input rotary. The cause is structural —
the copy offset is one hardcoded constant (2560), so the model must represent
exactly ONE relative distance, and any positional code can do that.

`synthetic_grid.py` stores an R x C grid of random tokens and asks for one ROW
or one COLUMN:

* a **row** query is C contiguous tokens — the easy control, roughly what the
  copy task already measured;
* a **column** query is R tokens at **stride C**. Under a 1-D address that is R
  distinct offsets held at once; under a `(row, col)` address it is one constant
  axis. This is the sharpest 1-D-vs-2-D discriminator we can build, and it is
  the synthetic counterpart of the CLRS matrix.

Every cell is drawn from one distribution, so no cell is findable by content —
only by address. The grid sits ~3,500 tokens before the answer, far outside the
128 window, so the fast-weight update->apply path must carry it.

Defaults: 32x32, `QUERY=col`, 800M tokens/cell (matching F35's budget).
Knobs if you need a difficulty ladder: `ROWS=`, `COLS=`, `QUERY=row|mix`.

**Run `col` first.** Only if `col` separates the arms is the `row` control worth
GPU time — it is the "should show nothing" arm.

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
