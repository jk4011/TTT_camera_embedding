# -*- coding: utf-8 -*-
"""Grid-recall diagnostic: the harder successor to synthetic_copy (Q30).

WHY a new task. F35's exact-offset copy saturated: nope 0.2%, but rope / honly /
hpra ALL reached 100%, so it separated the arms only by convergence speed and
could not support any claim that the hidden site BEATS the input rotary. The
cause is structural -- the copy offset is a single hardcoded constant (2560), so
the model has to represent exactly ONE relative distance. Any positional code
that can express one distance suffices.

WHAT MAKES THIS HARDER, and specifically hard in the direction our thesis
predicts. The model stores an R x C grid of random tokens and is then asked for
one ROW or one COLUMN of it:

  * a ROW query is C consecutive tokens -- one contiguous run, the easy case,
    and roughly what the copy task already measured.
  * a COLUMN query is R tokens at STRIDE C. Under a 1-D address that is R
    distinct offsets the code must hold simultaneously; under a (row, col)
    address it is "col fixed, row sweeps" -- constant in one axis. This is the
    cleanest discriminator we can build between a 1-D and a 2-D address, and it
    is the synthetic counterpart of the CLRS matrix (Q29).

Every grid cell is drawn from the same distribution, so no cell can be located
by content -- only by address. The grid sits far enough back that the attention
window cannot reach it and the fast-weight update->apply path must carry it.

Blocks are emitted in the SAME 4-channel format as clrs_data
(token, c_outer, c_inner, loss_mask), so they reuse the trainer's existing
clrs_split / evaluate_clrs / _ext_coords plumbing unchanged, including the
2d-vs-1d coordinate ablation and the COORD VERIFIED ACTIVE startup guard.

Determinism matches synthetic_copy: every sequence is a pure function of
(data_seed, sample_index), so all arms see an identical stream and resume is
exact. Val indices start at VAL_INDEX_BASE, disjoint from training.
"""

import numpy as np
import torch

SEQ_LEN = 4096

# vocab: 0..9 reserved, 10..1009 content. Same shape as synthetic_copy so the
# two diagnostics stay comparable.
PAD, BOS, EOS = 0, 1, 2
GRID_MARKER_ID = 4
QUERY_MARKER_ID = 5
ROW_MODE_ID = 6
COL_MODE_ID = 7
NOISE_LO, NOISE_HI = 10, 1010
VOCAB_SIZE = 1024
IGNORE_INDEX = -100

VAL_INDEX_BASE = 10 ** 9

# Default difficulty. 32x32 = 1024 grid tokens; a column query is 32 tokens at
# stride 32. GRID_START is chosen so the whole grid sits >= 2 chunk boundaries
# (chunk 1024) and many windows (128) before the answer region.
DEF_ROWS, DEF_COLS = 32, 32
GRID_START = 512


class GridCharTokenizer:
    """Minimal stub with the attributes train_small.build_config reads."""
    bos_token_id = BOS
    eos_token_id = EOS
    vocab_size = VOCAB_SIZE


def _layout(rows, cols):
    """Fixed positions for one difficulty setting."""
    grid_n = rows * cols
    grid_end = GRID_START + grid_n
    # query block sits well after the grid; the answer region is the last thing
    # in the sequence so every answer token is far from its source.
    q_marker = SEQ_LEN - (max(rows, cols) + 8)
    ans_start = q_marker + 3          # marker, mode, index, then the answer
    return grid_end, q_marker, ans_start


def make_block(data_seed, index, rows=DEF_ROWS, cols=DEF_COLS,
               query_mode="col", seq_len=SEQ_LEN):
    """Deterministic block for (data_seed, index) -> uint8-free int64 [seq_len, 4].

    query_mode: 'col' (stride-C gather, the hard case), 'row' (contiguous, the
    easy control), or 'mix' (per-sample coin flip, so one model must do both).
    """
    assert seq_len == SEQ_LEN, f"grid layout is defined for seq_len={SEQ_LEN}"
    grid_n = rows * cols
    grid_end, q_marker, ans_start = _layout(rows, cols)
    assert ans_start + max(rows, cols) <= seq_len, "grid too large for seq_len"

    g = np.random.RandomState((int(data_seed) * 1_000_003 + int(index)) & 0x7FFFFFFF)
    tok = g.randint(NOISE_LO, NOISE_HI, size=seq_len).astype(np.int64)

    grid = g.randint(NOISE_LO, NOISE_HI, size=grid_n).astype(np.int64)
    tok[GRID_START:grid_end] = grid
    tok[GRID_START - 1] = GRID_MARKER_ID

    mode = query_mode
    if mode == "mix":
        mode = "col" if g.randint(2) == 0 else "row"
    # index of the requested row / column, encoded as a content token so the
    # QUERY is content and only the RETRIEVAL is positional
    limit = cols if mode == "col" else rows
    q_idx = int(g.randint(limit))
    tok[q_marker] = QUERY_MARKER_ID
    tok[q_marker + 1] = COL_MODE_ID if mode == "col" else ROW_MODE_ID
    tok[q_marker + 2] = NOISE_LO + q_idx

    if mode == "col":
        answer = grid[q_idx::cols]                      # R tokens, stride C
    else:
        answer = grid[q_idx * cols:(q_idx + 1) * cols]  # C contiguous tokens
    a_len = len(answer)
    tok[ans_start:ans_start + a_len] = answer

    # --- coordinates -------------------------------------------------------
    # Grid tokens carry their true (row, col). Everything else carries
    # (0, running position) so non-grid text still has a usable 1-D address.
    c_out = np.zeros(seq_len, dtype=np.int64)
    c_in = np.arange(seq_len, dtype=np.int64) % 256
    rr, cc = np.divmod(np.arange(grid_n), cols)
    c_out[GRID_START:grid_end] = rr
    c_in[GRID_START:grid_end] = cc
    # The answer region is addressed in the SAME space as its source, so the
    # phases can cancel: answer element k of column j is grid cell (k, j).
    if mode == "col":
        c_out[ans_start:ans_start + a_len] = np.arange(a_len)
        c_in[ans_start:ans_start + a_len] = q_idx
    else:
        c_out[ans_start:ans_start + a_len] = q_idx
        c_in[ans_start:ans_start + a_len] = np.arange(a_len)

    mask = np.zeros(seq_len, dtype=np.int64)
    mask[ans_start:ans_start + a_len] = 1

    return np.stack([tok, c_out, c_in, mask], axis=1)


class GridStream:
    """Resumable deterministic stream, same contract as ClrsBlockStream."""

    def __init__(self, data_seed, seq_len=SEQ_LEN, rows=DEF_ROWS, cols=DEF_COLS,
                 query_mode="col"):
        self.data_seed = int(data_seed)
        self.seq_len = seq_len
        self.rows, self.cols, self.query_mode = rows, cols, query_mode
        self.n_emitted = 0
        self.N = 10 ** 9  # effectively unbounded; kept for log parity

    def __iter__(self):
        return self

    def __next__(self):
        b = make_block(self.data_seed, self.n_emitted, self.rows, self.cols,
                       self.query_mode, self.seq_len)
        self.n_emitted += 1
        return b.tolist()

    def state(self):
        return {"n_raw_consumed": self.n_emitted,
                "buf": torch.empty(0, dtype=torch.int64)}

    def restore(self, state):
        self.n_emitted = int(state["n_raw_consumed"])


def build_val_set(data_seed, n_seqs=64, seq_len=SEQ_LEN, rows=DEF_ROWS,
                  cols=DEF_COLS, query_mode="col"):
    """Fixed val set from indices >= VAL_INDEX_BASE (disjoint from training)."""
    out = [make_block(data_seed, VAL_INDEX_BASE + j, rows, cols, query_mode,
                      seq_len) for j in range(n_seqs)]
    return torch.from_numpy(np.stack(out))


def selftest():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{'  ' + detail if detail else ''}")

    print("[grid] self-test")
    for mode in ("col", "row"):
        b = make_block(42, 0, 8, 8, mode, SEQ_LEN)
        tok, co, ci, mk = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        grid_end, q_marker, ans_start = _layout(8, 8)
        sup = np.flatnonzero(mk)
        check(f"{mode}: supervised span is contiguous and 8 long",
              len(sup) == 8 and sup[-1] - sup[0] == 7, f"{len(sup)} tokens")
        # the answer must actually equal the requested slice of the grid
        grid = tok[GRID_START:grid_end]
        q_idx = int(tok[q_marker + 2] - NOISE_LO)
        want = grid[q_idx::8] if mode == "col" else grid[q_idx * 8:(q_idx + 1) * 8]
        check(f"{mode}: answer equals the requested {mode}",
              np.array_equal(tok[sup], want))
        # every answer token must share its source cell's coordinate
        src = (np.arange(8) * 8 + q_idx) if mode == "col" \
            else (q_idx * 8 + np.arange(8))
        src_pos = GRID_START + src
        check(f"{mode}: answer coords match their source cell coords",
              np.array_equal(co[sup], co[src_pos]) and
              np.array_equal(ci[sup], ci[src_pos]))
        # the retrieval must be out of window reach
        check(f"{mode}: source is far outside the 128 window",
              int(sup[0]) - int(src_pos.max()) > 128,
              f"gap {int(sup[0]) - int(src_pos.max())}")
    # content cannot identify a cell: all cells share one distribution
    b = make_block(42, 1, 32, 32, "col", SEQ_LEN)
    grid_end, _, _ = _layout(32, 32)
    grid = b[GRID_START:grid_end, 0]
    check("grid cells are not content-separable (many duplicates)",
          len(np.unique(grid)) < len(grid), f"{len(np.unique(grid))} unique of {len(grid)}")
    # determinism
    check("deterministic in (seed, index)",
          np.array_equal(make_block(42, 7), make_block(42, 7)))
    check("different index -> different block",
          not np.array_equal(make_block(42, 7), make_block(42, 8)))
    check("val indices disjoint from train",
          not np.array_equal(make_block(42, 0), make_block(42, VAL_INDEX_BASE)))
    print(f"[grid] self-test {'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(selftest())
