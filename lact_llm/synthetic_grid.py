# -*- coding: utf-8 -*-
"""N-dimensional tensor recall: does a d-D address beat a flat one, and how does
that change as d grows? (Q30)

WHY a new task. F35's exact-offset copy saturated -- nope 0.2%, but rope / honly
/ hpra ALL reached 100% -- so it separated the arms only by convergence speed and
could not support any claim that the hidden site BEATS the input rotary. The
cause is structural: the copy offset is one hardcoded constant (2560), so the
model must represent exactly ONE relative distance, and any positional code can.

THE DATA IS GENUINELY d-DIMENSIONAL. A tensor of shape SHAPES[d] holding 1024
random tokens is serialised row-major into the sequence. Total element count is
1024 at EVERY d, so the memory load is constant and only the lattice changes:

    d=2  [32, 32]          d=5  [4, 4, 4, 4, 4]
    d=3  [16, 8, 8]        d=6  [4, 4, 4, 4, 2, 2]
    d=4  [8, 8, 4, 4]

THE QUERY needs the lattice. It names an AXIS a and fixes the other d-1 indices,
and the answer is the resulting FIBER: the tensor entries that vary only along
axis a. In the flat serialisation those entries sit at stride
prod(shape[a+1:]), so:

  * a d-D address makes this trivial -- the d-1 fixed axes contribute phase
    difference ZERO between query and source, and the free axis carries the
    offset. Nothing to resolve.
  * a flat address must resolve ONE stride per axis, and the strides span
    1 .. 1024/shape[0]. That is what grows with d.

Enough fibers are queried per sequence to keep the supervised token count at 32
for every d (K = 32 / shape[a] of them), so answer length is never a confound.

WHY THE CURVE IS NOT OBVIOUS. Two forces oppose each other. Rising d gives the
flat address more distinct stride scales to resolve, so the d-D advantage should
GROW. But `_ext_angles` partitions the ladder into d contiguous bands, so each
axis keeps only ~P/d frequencies -- past some point every axis is too coarse to
resolve its own (already short) side. The prediction is a peak, and finding
where it sits speaks directly to NVS, which splits the ladder 6 ways for Plucker
coordinates.

Blocks are (token, c_0 .. c_{d-1}, loss_mask) -- token first, mask last -- the
same convention clrs_data uses at d=2, so the trainer's clrs_split /
evaluate_clrs / _ext_coords plumbing and the COORD VERIFIED ACTIVE startup guard
apply unchanged. Feeding --clrs_coord_mode 1d drives every axis with the token
index, which recombines the bands into inv_freq*t, i.e. bit-identically the
stock rotary: that is the flat-address control arm.

Determinism matches synthetic_copy: every sequence is a pure function of
(data_seed, sample_index), so all arms see an identical stream and resume is
exact. Val indices start at VAL_INDEX_BASE, disjoint from training.
"""

import os

import numpy as np
import torch

SEQ_LEN = 4096

PAD, BOS, EOS = 0, 1, 2
GRID_MARKER_ID = 4
QUERY_MARKER_ID = 5
AXIS_ID_BASE = 6                  # axis a is token AXIS_ID_BASE + a  (6..11)
# Content vocab, 1000 ids by design. GRID_CONTENT_VOCAB is a LEARNABILITY-PROBE
# knob only: at the designed size every arm -- including the d-D address arm the
# design calls trivial -- sat exactly at chance (loss 6.909 == ln(1000)), so the
# sweep had no dynamic range. Shrinking the vocab lowers the bits per cell while
# leaving the lattice, the addresses and the retrieval pattern untouched, which
# separates "cannot address" from "cannot restore content this precisely".
# Keep >= 32: query tokens encode a fixed axis index as NOISE_LO + idx, and the
# largest index in SHAPES is 31.
_CONTENT_VOCAB = int(os.environ.get("GRID_CONTENT_VOCAB", "1000"))
assert _CONTENT_VOCAB >= 32, "content vocab must cover the fixed-index encoding"
NOISE_LO = 16
NOISE_HI = NOISE_LO + _CONTENT_VOCAB
if _CONTENT_VOCAB != 1000:
    # provenance: a probe run must be identifiable from its log alone
    print(f"[grid] PROBE: content vocab overridden to {_CONTENT_VOCAB} "
          f"(chance = {1.0 / _CONTENT_VOCAB:.5f}); NOT the designed task",
          flush=True)
VOCAB_SIZE = 1024
IGNORE_INDEX = -100

VAL_INDEX_BASE = 10 ** 9

# Genuine d-dimensional lattices, all holding exactly 1024 elements so the
# memory load is identical across the sweep.
SHAPES = {
    2: (32, 32),
    3: (16, 8, 8),
    4: (8, 8, 4, 4),
    5: (4, 4, 4, 4, 4),
    6: (4, 4, 4, 4, 2, 2),
}
GRID_N = 1024
SUPERVISED = 32                   # supervised tokens per sequence, every d
GRID_START = 512


def _strides(shape):
    """Row-major strides of `shape` in the flat serialisation."""
    st, acc = [0] * len(shape), 1
    for a in range(len(shape) - 1, -1, -1):
        st[a] = acc
        acc *= shape[a]
    return st


def _unravel(p, shape):
    st = _strides(shape)
    return [(int(p) // st[a]) % shape[a] for a in range(len(shape))]


def _layout(d, axis, shape):
    """Tail layout: K query blocks, each followed by its answer fiber."""
    fiber = shape[axis]
    K = SUPERVISED // fiber
    q_len = 2 + (d - 1)                 # marker, axis id, d-1 fixed indices
    tail = K * (q_len + fiber)
    return fiber, K, q_len, SEQ_LEN - tail


def make_block(data_seed, index, coord_dims=2, seq_len=SEQ_LEN):
    """Deterministic block for (data_seed, index) -> int64 [seq_len, d+2]."""
    assert seq_len == SEQ_LEN, f"layout is defined for seq_len={SEQ_LEN}"
    assert coord_dims in SHAPES, f"coord_dims must be one of {sorted(SHAPES)}"
    d = coord_dims
    shape = SHAPES[d]
    st = _strides(shape)

    g = np.random.RandomState((int(data_seed) * 1_000_003 + int(index)) & 0x7FFFFFFF)
    tok = g.randint(NOISE_LO, NOISE_HI, size=seq_len).astype(np.int64)
    grid = g.randint(NOISE_LO, NOISE_HI, size=GRID_N).astype(np.int64)
    grid_end = GRID_START + GRID_N
    tok[GRID_START:grid_end] = grid
    tok[GRID_START - 1] = GRID_MARKER_ID

    axis = int(g.randint(d))
    fiber, K, q_len, tail_start = _layout(d, axis, shape)
    assert grid_end + 128 < tail_start, "tail would sit inside the attention window"

    C = np.zeros((seq_len, d), dtype=np.int64)
    # non-grid tokens get a plain running position on axis 0 so ordinary
    # positions still have a (degenerate) address
    C[:, 0] = np.arange(seq_len) % 256
    for p in range(GRID_N):
        C[GRID_START + p] = _unravel(p, shape)

    mask = np.zeros(seq_len, dtype=np.int64)

    # K distinct index-tuples for the fixed axes -> K distinct fibers
    other = [a for a in range(d) if a != axis]
    n_other = GRID_N // fiber
    picks = g.choice(n_other, size=K, replace=False)
    cur = tail_start
    for pick in picks:
        # decode `pick` into indices of the non-free axes
        fixed, rem = {}, int(pick)
        for a in reversed(other):
            fixed[a] = rem % shape[a]
            rem //= shape[a]
        tok[cur] = QUERY_MARKER_ID
        tok[cur + 1] = AXIS_ID_BASE + axis
        for n, a in enumerate(other):
            tok[cur + 2 + n] = NOISE_LO + fixed[a]
        cur += q_len
        # the fiber itself
        base = sum(fixed[a] * st[a] for a in other)
        for f in range(fiber):
            p = base + f * st[axis]
            tok[cur + f] = grid[p]
            # the answer carries its SOURCE cell's coordinate, so the update and
            # apply phases can cancel -- without this the rotary has nothing to
            # align on
            C[cur + f] = _unravel(p, shape)
            mask[cur + f] = 1
        cur += fiber
    assert cur == seq_len, f"tail ended at {cur}, expected {seq_len}"
    assert int(mask.sum()) == SUPERVISED, f"{int(mask.sum())} supervised, want {SUPERVISED}"

    return np.concatenate([tok[:, None], C, mask[:, None]], axis=1)


class GridStream:
    """Resumable deterministic stream, same contract as ClrsBlockStream."""

    def __init__(self, data_seed, seq_len=SEQ_LEN, coord_dims=2):
        self.data_seed = int(data_seed)
        self.seq_len = seq_len
        self.coord_dims = coord_dims
        self.n_emitted = 0
        self.N = 10 ** 9  # effectively unbounded; kept for log parity

    def __iter__(self):
        return self

    def __next__(self):
        b = make_block(self.data_seed, self.n_emitted, self.coord_dims, self.seq_len)
        self.n_emitted += 1
        return b.tolist()

    def state(self):
        return {"n_raw_consumed": self.n_emitted,
                "buf": torch.empty(0, dtype=torch.int64)}

    def restore(self, state):
        self.n_emitted = int(state["n_raw_consumed"])


def build_val_set(data_seed, n_seqs=256, seq_len=SEQ_LEN, coord_dims=2):
    """Fixed val set from indices >= VAL_INDEX_BASE (disjoint from training)."""
    out = [make_block(data_seed, VAL_INDEX_BASE + j, coord_dims, seq_len)
           for j in range(n_seqs)]
    return torch.from_numpy(np.stack(out))


class GridCharTokenizer:
    """Minimal stub with the attributes train_small.build_config reads."""
    bos_token_id = BOS
    eos_token_id = EOS
    vocab_size = VOCAB_SIZE


def stride_report():
    """The strides a FLAT address would have to resolve, per d. This is the
    quantity the sweep is about."""
    print(f"{'d':>2s} {'shape':<22s} {'strides':<26s} fiber lengths")
    for d, sh in SHAPES.items():
        st = _strides(sh)
        fl = [f"{sh[a]}@{st[a]}" for a in range(d)]
        print(f"{d:2d} {str(sh):<22s} {str(st):<26s} {' '.join(fl)}")


def selftest():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{'  ' + detail if detail else ''}")

    print("[grid] self-test")
    for d, sh in SHAPES.items():
        prod = 1
        for x in sh:
            prod *= x
        check(f"d={d}: shape {sh} holds exactly {GRID_N}", prod == GRID_N, f"{prod}")

    for d in SHAPES:
        shape = SHAPES[d]
        st = _strides(shape)
        # unravel is a bijection
        seen = {tuple(_unravel(p, shape)) for p in range(GRID_N)}
        check(f"d={d}: unravel is injective over all {GRID_N} cells",
              len(seen) == GRID_N, f"{len(seen)} distinct")

        for idx in (0, 5, 17):
            b = make_block(42, idx, d)
            tok, C, mk = b[:, 0], b[:, 1:-1], b[:, -1]
            check(f"d={d} idx={idx}: {SUPERVISED} supervised tokens",
                  int(mk.sum()) == SUPERVISED, str(int(mk.sum())))
            check(f"d={d} idx={idx}: block has {d}+2 channels",
                  b.shape[1] == d + 2, str(b.shape))
            grid = tok[GRID_START:GRID_START + GRID_N]
            sup = np.flatnonzero(mk)
            # every answer token must equal the grid cell its coordinate names,
            # and that cell must be far outside the attention window
            good_val, good_far, good_free = True, True, True
            axis = int(tok[np.flatnonzero(tok == QUERY_MARKER_ID)[0] + 1] - AXIS_ID_BASE)
            for s in sup:
                coord = C[s]
                p = sum(int(coord[a]) * st[a] for a in range(d))
                good_val &= (tok[s] == grid[p])
                good_far &= (s - (GRID_START + p)) > 128
            check(f"d={d} idx={idx}: answers equal the cells their coords name",
                  good_val)
            check(f"d={d} idx={idx}: every source is >128 before its answer",
                  good_far)
            # within one fiber only the free axis may vary
            fiber = shape[axis]
            for start in range(0, SUPERVISED, fiber):
                blk = C[sup[start:start + fiber]]
                for a in range(d):
                    if a == axis:
                        good_free &= len(set(blk[:, a].tolist())) == fiber
                    else:
                        good_free &= len(set(blk[:, a].tolist())) == 1
            check(f"d={d} idx={idx}: within a fiber only axis {axis} varies",
                  good_free)

    # content cannot identify a cell
    grid = make_block(42, 1, 3)[GRID_START:GRID_START + GRID_N, 0]
    check("stored tokens are not content-separable",
          len(np.unique(grid)) < len(grid),
          f"{len(np.unique(grid))} unique of {len(grid)}")

    check("deterministic in (seed, index)",
          np.array_equal(make_block(42, 7, 3), make_block(42, 7, 3)))
    check("different index -> different block",
          not np.array_equal(make_block(42, 7, 3), make_block(42, 8, 3)))
    check("val indices disjoint from train",
          not np.array_equal(make_block(42, 0, 3),
                             make_block(42, VAL_INDEX_BASE, 3)))

    print(f"[grid] self-test {'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "strides":
        stride_report()
        raise SystemExit(0)
    raise SystemExit(selftest())
