# -*- coding: utf-8 -*-
"""Grid-recall diagnostic with an ADDRESS-DIMENSION sweep (Q30).

WHY a new task. F35's exact-offset copy saturated: nope 0.2%, but rope / honly /
hpra ALL reached 100%, so it separated the arms only by convergence speed and
could not support any claim that the hidden site BEATS the input rotary. The
cause is structural -- the copy offset is a single hardcoded constant (2560), so
the model has to represent exactly ONE relative distance, and any positional
code can do that.

THE TASK. A flat block of 1024 random tokens is stored early in the sequence.
The query asks for 32 of them, either

  * `stride` : every 32nd token starting at offset j  (the hard case), or
  * `contig` : 32 consecutive tokens starting at 32*i (the easy control).

Every token is drawn from one distribution, so no element is findable by
content -- only by address. The block sits ~3,500 tokens before the answer, far
outside the 128 window, so the fast-weight update->apply path must carry it.

THE SWEEP, which is the point of this module. The stored tokens are IDENTICAL
for every setting -- byte for byte. What varies is only how the flat index is
FACTORISED into coordinate axes handed to the rotary:

    flat index p = i * 32 + j,  i in [0,32)  (position within the fiber)
                                j in [0,32)  (which fiber)

    coord_dims k     axes
    -------------    ---------------------------------
    1                [p]                    <- the stock rotary
    2                [i, j]
    3                [i, j/4, j%4]
    4                [i, j/8, (j/2)%4, j%2]
    5                [i, ...4 binary-ish digits of j...]
    6                [i, ...5 binary digits of j...]

So this is a pure address-REPRESENTATION experiment: same data, same retrieval
pattern, same memory load, same answer length (32 tokens) at every k. Only the
coordinate changes.

WHAT THE SWEEP IS EXPECTED TO SHOW, and why it is not a monotone story.
`_ext_angles` partitions the frequency ladder into k contiguous bands, one per
axis, so each axis carries only ~P/k frequencies. Adding axes buys structure but
SPENDS resolution. The prediction is therefore an optimum, not a ramp: k=1
cannot express the stride at all, k=2 matches the data's true structure, and
large k should decay as each axis becomes too coarse. Where that optimum sits
matters directly for the paper, because NVS splits the ladder 6 ways for Plucker
coordinates.

Blocks are emitted as (token, c_0 .. c_{k-1}, loss_mask) -- token first, mask
last -- which is the same convention clrs_data uses at k=2, so the trainer's
clrs_split / evaluate_clrs / _ext_coords plumbing and the COORD VERIFIED ACTIVE
startup guard all apply unchanged.

Determinism matches synthetic_copy: every sequence is a pure function of
(data_seed, sample_index), so all arms see an identical stream and resume is
exact. Val indices start at VAL_INDEX_BASE, disjoint from training.
"""

import numpy as np
import torch

SEQ_LEN = 4096

PAD, BOS, EOS = 0, 1, 2
GRID_MARKER_ID = 4
QUERY_MARKER_ID = 5
STRIDE_MODE_ID = 6
CONTIG_MODE_ID = 7
NOISE_LO, NOISE_HI = 10, 1010
VOCAB_SIZE = 1024
IGNORE_INDEX = -100

VAL_INDEX_BASE = 10 ** 9

FIBER = 32          # answer length, constant at every coord_dims
NFIBER = 32         # number of fibers; total stored tokens = FIBER * NFIBER
GRID_N = FIBER * NFIBER
GRID_START = 512

# How the fiber index j in [0, 32) is factorised for each coord_dims k.
# Axis 0 is always i (position within the fiber); these are the REMAINING axes,
# most-significant first, and their product is always NFIBER = 32.
_FACTORS = {
    1: None,            # special-cased: a single flat axis p
    2: (32,),
    3: (8, 4),
    4: (4, 4, 2),
    5: (4, 2, 2, 2),
    6: (2, 2, 2, 2, 2),
}
MAX_COORD_DIMS = max(_FACTORS)


class GridCharTokenizer:
    """Minimal stub with the attributes train_small.build_config reads.

    Blocks are emitted as integer ids already (make_block), so there is no text
    to encode; encode/decode exist only for parity with ClrsCharTokenizer.
    """
    bos_token_id = BOS
    eos_token_id = EOS
    vocab_size = VOCAB_SIZE

    @staticmethod
    def encode(ids):
        return list(ids)

    @staticmethod
    def decode(ids):
        return " ".join(str(int(i)) for i in ids)


def _digits(j, factors):
    """Mixed-radix digits of j, most-significant first."""
    # Index by POSITION, not by value: factors repeat (4,2,2,2), and
    # factors.index(f) would resolve every 2 to the first one, collapsing the
    # digits so distinct j map to the same coordinate. The self-test's
    # injectivity check exists to catch exactly that.
    out, rem = [], int(j)
    for idx, _ in enumerate(factors):
        step = 1
        for g in factors[idx + 1:]:
            step *= g
        out.append(rem // step)
        rem = rem % step
    return out


def _coords_for(i, j, k):
    """Coordinate vector of the element at fiber position i, fiber index j."""
    if k == 1:
        return [i * NFIBER + j]
    return [i] + _digits(j, _FACTORS[k])


def _layout():
    grid_end = GRID_START + GRID_N
    q_marker = SEQ_LEN - (FIBER + 8)
    ans_start = q_marker + 3
    return grid_end, q_marker, ans_start


def make_block(data_seed, index, coord_dims=2, query_mode="stride",
               seq_len=SEQ_LEN):
    """Deterministic block for (data_seed, index) -> int64 [seq_len, coord_dims+2].

    The TOKENS do not depend on coord_dims -- only the coordinate channels do.
    """
    assert seq_len == SEQ_LEN, f"grid layout is defined for seq_len={SEQ_LEN}"
    assert coord_dims in _FACTORS, f"coord_dims must be one of {sorted(_FACTORS)}"
    grid_end, q_marker, ans_start = _layout()

    g = np.random.RandomState((int(data_seed) * 1_000_003 + int(index)) & 0x7FFFFFFF)
    tok = g.randint(NOISE_LO, NOISE_HI, size=seq_len).astype(np.int64)
    grid = g.randint(NOISE_LO, NOISE_HI, size=GRID_N).astype(np.int64)
    tok[GRID_START:grid_end] = grid
    tok[GRID_START - 1] = GRID_MARKER_ID

    mode = query_mode
    if mode == "mix":
        mode = "stride" if g.randint(2) == 0 else "contig"
    q_idx = int(g.randint(NFIBER))
    tok[q_marker] = QUERY_MARKER_ID
    tok[q_marker + 1] = STRIDE_MODE_ID if mode == "stride" else CONTIG_MODE_ID
    tok[q_marker + 2] = NOISE_LO + q_idx

    if mode == "stride":
        # every NFIBER-th element: flat p = i*NFIBER + q_idx, i = 0..FIBER-1
        src_flat = np.arange(FIBER) * NFIBER + q_idx
        src_ij = [(i, q_idx) for i in range(FIBER)]
    else:
        # FIBER consecutive elements starting at q_idx*FIBER
        src_flat = q_idx * FIBER + np.arange(FIBER)
        src_ij = [(int(p // NFIBER), int(p % NFIBER)) for p in src_flat]
    tok[ans_start:ans_start + FIBER] = grid[src_flat]

    # --- coordinates -------------------------------------------------------
    k = coord_dims
    C = np.zeros((seq_len, k), dtype=np.int64)
    # non-grid tokens: a plain running position on axis 0, zeros elsewhere, so
    # ordinary text still has a usable (if degenerate) address
    C[:, 0] = np.arange(seq_len) % 256
    for p in range(GRID_N):
        C[GRID_START + p] = _coords_for(p // NFIBER, p % NFIBER, k)
    # the answer carries its SOURCE cell's coordinate, so the update and apply
    # phases can cancel -- without this the rotary has nothing to align
    for a, (i, j) in enumerate(src_ij):
        C[ans_start + a] = _coords_for(i, j, k)

    mask = np.zeros(seq_len, dtype=np.int64)
    mask[ans_start:ans_start + FIBER] = 1

    return np.concatenate([tok[:, None], C, mask[:, None]], axis=1)


class GridStream:
    """Resumable deterministic stream, same contract as ClrsBlockStream."""

    def __init__(self, data_seed, seq_len=SEQ_LEN, coord_dims=2,
                 query_mode="stride"):
        self.data_seed = int(data_seed)
        self.seq_len = seq_len
        self.coord_dims = coord_dims
        self.query_mode = query_mode
        self.n_emitted = 0
        self.N = 10 ** 9  # effectively unbounded; kept for log parity

    def __iter__(self):
        return self

    def __next__(self):
        b = make_block(self.data_seed, self.n_emitted, self.coord_dims,
                       self.query_mode, self.seq_len)
        self.n_emitted += 1
        return b.tolist()

    def state(self):
        return {"n_raw_consumed": self.n_emitted,
                "buf": torch.empty(0, dtype=torch.int64)}

    def restore(self, state):
        self.n_emitted = int(state["n_raw_consumed"])


def build_val_set(data_seed, n_seqs=64, seq_len=SEQ_LEN, coord_dims=2,
                  query_mode="stride"):
    """Fixed val set from indices >= VAL_INDEX_BASE (disjoint from training)."""
    out = [make_block(data_seed, VAL_INDEX_BASE + j, coord_dims, query_mode,
                      seq_len) for j in range(n_seqs)]
    return torch.from_numpy(np.stack(out))


def selftest():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{'  ' + detail if detail else ''}")

    print("[grid] self-test")
    grid_end, q_marker, ans_start = _layout()

    # 1. the mixed-radix factorisations are exact and injective
    for k, f in _FACTORS.items():
        if f is None:
            continue
        prod = 1
        for x in f:
            prod *= x
        seen = {tuple(_digits(j, f)) for j in range(NFIBER)}
        check(f"k={k}: factors {f} multiply to {NFIBER} and are injective",
              prod == NFIBER and len(seen) == NFIBER, f"prod={prod} unique={len(seen)}")

    # 2. tokens are IDENTICAL across coord_dims -- the sweep changes only address
    ref = make_block(42, 3, 2, "stride")[:, 0]
    for k in _FACTORS:
        b = make_block(42, 3, k, "stride")
        check(f"k={k}: tokens byte-identical to k=2", np.array_equal(b[:, 0], ref))
        check(f"k={k}: block has {k}+2 channels", b.shape[1] == k + 2,
              str(b.shape))

    # 3. the answer really is the requested fiber, and its coords match the source
    for mode in ("stride", "contig"):
        for k in (2, 5):
            b = make_block(42, 11, k, mode)
            tok, C, mk = b[:, 0], b[:, 1:-1], b[:, -1]
            grid = tok[GRID_START:grid_end]
            q_idx = int(tok[q_marker + 2] - NOISE_LO)
            src = (np.arange(FIBER) * NFIBER + q_idx) if mode == "stride" \
                else (q_idx * FIBER + np.arange(FIBER))
            sup = np.flatnonzero(mk)
            check(f"{mode} k={k}: answer equals the requested fiber",
                  np.array_equal(tok[sup], grid[src]))
            check(f"{mode} k={k}: answer coords == source coords",
                  np.array_equal(C[sup], C[GRID_START + src]))
            check(f"{mode} k={k}: source is far outside the 128 window",
                  int(sup[0]) - int((GRID_START + src).max()) > 128)

    # 4. a stride query really is stride-NFIBER in the flat layout
    b = make_block(42, 5, 2, "stride")
    q_idx = int(b[q_marker + 2, 0] - NOISE_LO)
    src = np.arange(FIBER) * NFIBER + q_idx
    check("stride query gathers at stride 32", np.all(np.diff(src) == NFIBER))

    # 5. content cannot identify an element
    grid = make_block(42, 1, 2, "stride")[GRID_START:grid_end, 0]
    check("stored tokens are not content-separable",
          len(np.unique(grid)) < len(grid),
          f"{len(np.unique(grid))} unique of {len(grid)}")

    # 6. determinism / val disjointness
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
