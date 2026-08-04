# NODE3 prompt — Q29: CLRS-Text address-dimensionality grid

Paste this whole file as the first message to the Claude session on node3.

---

## What you are running

Q29 tests the one thing our paper claims but has never measured inside a single
task: **does the hidden rotary site need a multi-dimensional address?**

Our ledger says gains track two axes, and each axis has exactly one clean case:

| case | coordinate | memory load-bearing? | hidden site |
|---|---|---|---|
| natural language / hg38 DNA / REMI music | 1-D | yes | null |
| F35 copy diagnostic | 1-D | yes | works, but **ties** input rope (both 100%) |
| plain video (F21/F22) | 3-D (t,y,x) | **no** (idle memory) | null |
| ccv (F30) / NVS | 6-D Plucker | yes | **earns** (+3.9~4.6% over input) |

So the standing hypothesis is: the hidden site earns iff the retrieval is forced
through the fast weights AND the address is multi-dimensional (F20, F33, F27c).
Every existing datum for that is a comparison ACROSS unlike tasks. Q29 makes it
a controlled measurement inside one data distribution.

## Why CLRS-Text

CLRS-Text (DeepMind, `smcleish/CLRS-Text-*`, Apache-2.0) serializes an n x n
adjacency matrix with **no row/column indices printed** and every entry the
character `0` or `1`:

```
bfs:
s: 7, A: [[0 0 0 0 1 ...], [0 1 0 0 0 ...], ...], initial_trace: [0 1 2 ...]
trace | pi:
```

Thousands of identical tokens, so content matching cannot resolve WHICH zero.
The only usable address is (row, col). At n=41 the question is ~3,644 chars
against a 128-token attention window, so the matrix is memory-exclusive.

## The design (read this before touching anything)

The ablation is **on the coordinate, not on the data**. Same problems, same
tokens, same length, same memory load in every arm; only the address fed to
the rotary changes:

- `--clrs_coord_mode 2d` -> the parsed `(outer, inner)` = (row, col)
- `--clrs_coord_mode 1d` -> `(t, t)`, which recombines the split frequency
  ladder into `inv_freq * t`, i.e. **exactly the stock rotary**

That equivalence is verified bit-exact, not argued: `sanity_clrs_coords.py`
mode a reports `0.000e+00` against the stock `fast_rotary` path, against the
manual chunkq C=1 path, and against the plain hidden ladder. So `2d` vs `1d`
isolates address dimensionality and nothing else.

**Pre-registered prediction — write it down before reading any number:**
the h-over-in increment appears in the `2d` arms and vanishes in the `1d` arms.
If `h` beats `in` equally in both, the dimensionality hypothesis is refuted and
we report that.

## Grid: 7 cells

`base` has no rotary, so its coordinate is provably inert (sanity mode b:
`0.000e+00`) — it is ONE run, not two.

| cell | arm | coord | flags |
|---|---|---|---|
| q29_base | nope | (inert) | `ttt_nope=true` |
| q29_in_2d / q29_in_1d | rope | 2d / 1d | `ttt_nope=false` |
| q29_h_2d / q29_h_1d | honly | 2d / 1d | `ttt_nope=true, ttt_hidden_rope=true, ttt_hrope_gain=1.0` |
| q29_both_2d / q29_both_1d | hpra | 2d / 1d | `ttt_nope=false, ttt_hidden_rope=true, ttt_hrope_gain=1.0` |

`ttt_hrope_gain=1.0` (standard ladder, not the gentle 0.1) because F35 showed
the standard ladder wins on precision tasks; the gentle-ladder rule was a
property of the w1024 natural-language band.

## How to run

```bash
cd /NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_rope/lact_llm
./run_clrs_grid.sh 0,1,2,3            # 7 cells striped over 4 GPUs, 2 waves
```

The script stripes cells over the GPU list, skips cells that already have
`final.pt` + a non-empty `val_log.jsonl` (resubmission = resume), and takes
hostname-scoped GPU locks in `lact_nvs/outputs/.gpu_locks/<host>_gpu<i>`
(the lock dir is on lustre and therefore SHARED between nodes — do not drop
the hostname prefix).

Launch it as your own background Bash task (`run_in_background: true`) so the
harness notifies you on completion; nohup'd daemons finish silently.

## Metrics

`evaluate_clrs` logs three numbers per validation to `outputs/<cell>/val_log.jsonl`:

- `answer_acc` — teacher-forced argmax accuracy over the answer region. **This
  is the primary metric.** A from-scratch 200M char model may sit at 0% exact
  match, which would make an exact-match-only readout indistinguishable from
  broken wiring.
- `exact_match` — whole answer correct. Secondary.
- `val_loss` / `ppl` — answer-region only (question is context, labels -100).

Val comes from the held-out test repo (seeds test_1..test_5), disjoint from
train, so it is clean regardless of how many epochs the train corpus takes.

## Report as paired, never unpaired

House standard: per-item paired deltas with a t-statistic, seed-matched. Report
win-rate alongside the mean. Do NOT compare unpaired means. Record results in
`RESULTS_DOSSIER.md` / `lact_llm/ga_honly/LEDGER.md` only — the paper is under
FREEZE, so do not touch paper claims.

## Verification that already passed on node1 (do not redo, but do not assume
## it covers your node either — rerun the sanity if anything looks off)

- `clrs_data.py selftest` — 13/13, including on real CLRS samples: bfs n=41
  sweeps the outer axis to 40, insertion_sort keeps it at 1.
- `sanity_clrs_coords.py` — 11/11: 1d == stock rotary to 0.000e+00 at both
  sites; 2d changes the loss; rotary-off makes the address inert to
  0.000e+00; per-sequence addressing; finite backward.
- `infer_n()` (train repo has no `length` column) cross-checked against the
  test repo's `length`: 37,144/40,000 overall, and **100% on all five
  algorithms we use**. The two mismatching algorithms (segments_intersect,
  optimal_bst) are not in any cell.

## Known limits — state these in any writeup

- A win here does NOT say anything about natural language; the 3-seed LM null
  stands. CLRS-Text is procedurally generated algorithmic text.
- It is not SOTA on CLRS-Text: that benchmark is built for fine-tuning
  pretrained LLMs, and a from-scratch 200M char model is nowhere near
  Gemma 2B / Gemini 1.5 Flash / TransNAR. The claim is strictly internal,
  seed-matched, paired.
- A row/column grid is separable and abelian: this tests DIMENSIONALITY, not
  the non-commutative SE(3) structure of the NVS story.
- Novelty hazard to disclose: **Selective RoPE (arXiv 2511.17388, ICLR 2026)**
  reportedly applies input-dependent rotations to q/k of gated linear
  attention / DeltaNet — our INPUT site — with gains on MQAR/MAD/copy. Our
  contribution is the SITE. (This citation came from a survey agent and has
  not yet been verified first-hand; verify before it goes in related work.)
