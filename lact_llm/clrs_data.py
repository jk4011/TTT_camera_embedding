# -*- coding: utf-8 -*-
"""CLRS-Text char-level data path for train_small.py (cross-task validation #3).

Same interface as dna_data / music_data so the trainer's checkpoint/resume
machinery (state()/restore(), batch_generator with_state) works unchanged --
with ONE addition: every block carries a per-token 2-D coordinate, because this
task exists to separate the two axes the ledger says gate the hidden site
(F20 dimensionality, F30 memory-exclusive workload).

Why this task (see EXPERIMENT_QUEUE.md Q29):
  * CLRS-Text serializes an n x n adjacency matrix as
      'bfs:\\ns: 7, A: [[0 0 0 ...], [0 0 0 ...], ...], initial_trace: [...]'
    with NO row/column indices printed and every entry the character '0' or '1'.
    Thousands of identical tokens -> content matching cannot resolve WHICH zero;
    the only usable address is (row, col). That is the 2-D cell.
  * The array-input algorithms (insertion_sort, task_scheduling, ...) have the
    identical serialization grammar with a flat list -> outer axis is constant 0.
    That is the 1-D control, in the SAME corpus under the SAME builder.

Coordinate builder (the load-bearing piece):
  Coordinates are recovered by parsing the serialized string ONLY -- never from
  dataset metadata -- so a reviewer cannot call this structure injection. A
  bracket-counter stack yields, for every character:
      c_outer = element counter one level above the deepest open level
      c_inner = element counter at the deepest open level
  For a nested matrix  [[..], [..]]  this is (row, col).
  For a flat list      [..]          this is (0, index).
  For the answer's comma-separated trace arrays it is (trace_step, index).
  So the 1-D vs 2-D contrast falls out of the DATA, with one builder and one
  rotary -- exactly the site-ablation discipline the grid needs.

Layout: uint8 [n_blocks, seq_len, 4] = (token, c_outer, c_inner, loss_mask).
Loss is scored on the answer region only (loss_mask=1), mirroring
train_small.evaluate_copy's supervised-region accuracy.
"""

import os
import json
import time

import numpy as np
import torch

LACT_LLM_DIR = os.path.dirname(os.path.abspath(__file__))
CLRS_DIR = "/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/datasets/clrs"

SEQ_LEN = 4096

HF_TRAIN = "smcleish/CLRS-Text-train"
HF_TEST = "smcleish/CLRS-Text-test"
TEST_SPLITS = ["test_1", "test_2", "test_3", "test_4", "test_5"]

# ---------------------------------------------------------------------------
# Cells. Chosen so that TOTAL (question + answer) length is matched across the
# 1-D and 2-D subsets -- measured on test_1, all within 1.3% of ~3950 chars.
# Matching TOTAL length (not question length) is what makes the contrast clean:
# for the array algorithms the trace in the ANSWER is what fills the sequence.
# ---------------------------------------------------------------------------
# Values are INCLUSIVE n bands, not single sizes: CLRS-Text spreads 2.15M train
# rows over ~30 algorithms x n=4..64, so any single (algo, n) cell holds only
# ~1k examples -- far too few for a multi-epoch budget. A band keeps every
# example inside the target length window while multiplying the corpus, and it
# matches how the benchmark itself trains (a range of n, extrapolate beyond).
#
# 2-D input (n x n matrix), memory load-bearing (>> window 128).
# bfs n=41 -> 3950 chars, n=30 -> 2552; bellman_ford n=33 -> 3937, n=24 -> 1918.
# Band floor is set by the WINDOW, not by seq_len: memory-exclusive only needs
# the matrix to exceed window 128, and bfs n=16 is already 643 chars (5x). The
# wide band triples the corpus (30k -> ~90k blocks) and varies the address
# range, which the 2-D arm should exploit and the 1-D arm should not.
CELL_2D_LONG = {"bfs": (16, 41), "bellman_ford": (14, 33)}
# 1-D input (flat array), memory load-bearing.
CELL_1D_LONG = {"insertion_sort": (18, 25), "task_scheduling": (30, 41),
                "activity_selector": (30, 40)}
# 2-D input but the whole problem fits INSIDE the attention window -> the
# F21/F22 idle-memory regime reproduced inside this benchmark (predict: null)
CELL_2D_SHORT = {"bfs": (4, 8), "bellman_ford": (4, 6)}
# 1-D input, in-window (the fourth quadrant of the 2x2)
CELL_1D_SHORT = {"insertion_sort": (4, 6), "task_scheduling": (4, 8),
                 "activity_selector": (4, 8)}

QUADRANTS = {
    "2d_long": CELL_2D_LONG,
    "1d_long": CELL_1D_LONG,
    "2d_short": CELL_2D_SHORT,
    "1d_short": CELL_1D_SHORT,
}

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
PAD, BOS, EOS = 0, 1, 2
_SPECIAL = 3

# Fixed, explicit character inventory: CLRS-Text is machine-serialized, so the
# alphabet is closed. Anything outside it is a hard error (never silently
# mapped), so a corpus change cannot quietly corrupt the address space.
CHARS = (
    "0123456789.-"          # numbers (floats are '0.315', negatives '-0.93')
    " ,:|()[]\n"            # structure
    "_"                     # algo names use snake_case
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"   # field names: 'A:' (adjacency matrix)
)
CHAR2ID = {c: i + _SPECIAL for i, c in enumerate(CHARS)}
ID2CHAR = {v: k for k, v in CHAR2ID.items()}
VOCAB_SIZE = _SPECIAL + len(CHARS)

MAX_COORD = 255  # uint8 storage; asserted at build time


class ClrsCharTokenizer:
    """Minimal stub with the attributes train_small.build_config reads."""
    bos_token_id = BOS
    eos_token_id = EOS
    vocab_size = VOCAB_SIZE

    @staticmethod
    def encode(s):
        return [CHAR2ID[c] for c in s]

    @staticmethod
    def decode(ids):
        return "".join(ID2CHAR.get(int(i), "") for i in ids)


def encode_text(s):
    try:
        return np.array([CHAR2ID[c] for c in s], dtype=np.uint8)
    except KeyError as e:
        raise ValueError(f"character outside the CLRS alphabet: {e!r}") from None


# ---------------------------------------------------------------------------
# Coordinate builder
# ---------------------------------------------------------------------------
_OPEN = {"[": "]", "(": ")"}
_CLOSE = {"]", ")"}


def build_coords(s):
    """Per-character (c_outer, c_inner) recovered from the serialization alone.

    A counter stack tracks how many elements have been *completed* at each
    nesting level. Level 0 is the top level (counts comma-separated trace
    arrays); level 1 is inside the outermost bracket; level 2 inside a nested
    bracket. For any character we report the counters of the two deepest open
    levels, so:
        [[a b] [c d]]  -> the 'c' gets (row=1, col=0)
        [a b c]        -> the 'b' gets (0, 1)
    Separators (space / comma) are attributed to the element they terminate,
    which keeps the coordinate piecewise-constant over each element's characters.

    Returns uint8 arrays (c_outer, c_inner), both length len(s).
    """
    n = len(s)
    c_out = np.zeros(n, dtype=np.uint8)
    c_in = np.zeros(n, dtype=np.uint8)
    counters = [0]           # counters[d] = elements completed at depth d
    # 'pending' marks that at least one non-separator char was seen at this
    # depth since the last separator, so repeated separators don't over-count.
    pending = [False]
    for i, ch in enumerate(s):
        d = len(counters) - 1
        c_in[i] = min(counters[d], MAX_COORD)
        c_out[i] = min(counters[d - 1], MAX_COORD) if d >= 1 else 0
        if ch in _OPEN:
            counters.append(0)
            pending.append(False)
        elif ch in _CLOSE:
            # A closed bracket IS one completed element of its parent level --
            # unconditionally, since the bracket's own contents set `pending`
            # at the child level, never at the parent's.
            if len(counters) > 1:
                counters.pop()
                pending.pop()
                d = len(counters) - 1
                counters[d] += 1
                pending[d] = False
        elif ch in " ,":
            # Separators only delimit elements INSIDE a bracket. At top level
            # they merely separate prose ('bfs:', 's: 7, A:'), which must not
            # advance the address -- otherwise the outer coordinate would count
            # words rather than structure.
            if d >= 1 and pending[d]:
                counters[d] += 1
                pending[d] = False
        elif ch == "\n":
            # a newline ends a field; do not let counters leak across lines
            pending[d] = False
        else:
            pending[d] = True
    return c_out, c_in


def infer_n(question):
    """Problem size n recovered from the serialization.

    The TRAIN repo ships only (question, answer, algo_name) -- no `length`
    column -- so n must be parsed. n is the element count of the FIRST
    bracketed list: the adjacency matrix's row count for graph problems, the
    key array's length for array problems. Cross-checked against the test
    repo's `length` column by selftest() (which is the whole reason to trust
    it on the train split).
    """
    depth = 0
    count = 0
    pending = False
    for ch in question:
        if ch in _OPEN:
            depth += 1
            if depth > 1:          # nested element (a matrix row); ignore inside
                continue
            count, pending = 0, False
        elif ch in _CLOSE:
            if depth == 1:
                return count + (1 if pending else 0)
            depth -= 1
            if depth == 1:
                count += 1
                pending = False
        elif depth == 1 and ch in " ,":
            if pending:
                count += 1
                pending = False
        elif depth >= 1:
            if depth == 1:
                pending = True
    return 0


def coord_report(question, answer):
    """Diagnostic: max coordinate reached on each axis (used by the self-test
    to prove the 2-D cells really vary the outer axis and the 1-D cells do
    not)."""
    co, ci = build_coords(question + answer)
    qn = len(question)
    return {
        "q_outer_max": int(co[:qn].max()), "q_inner_max": int(ci[:qn].max()),
        "a_outer_max": int(co[qn:].max()) if answer else 0,
        "a_inner_max": int(ci[qn:].max()) if answer else 0,
    }


# ---------------------------------------------------------------------------
# Block construction
# ---------------------------------------------------------------------------
def make_block(question, answer, seq_len):
    """One problem -> one block. Returns uint8 [seq_len, 4] or None if it does
    not fit. Channels: (token, c_outer, c_inner, loss_mask).

    Layout: BOS + question + answer + EOS, PAD to seq_len.
    loss_mask=1 on the answer characters and the EOS only -- the question is
    context, exactly like evaluate_copy scores only the copy region.
    """
    qt, at = encode_text(question), encode_text(answer)
    total = 1 + len(qt) + len(at) + 1
    if total > seq_len:
        return None
    co, ci = build_coords(question + answer)
    blk = np.zeros((seq_len, 4), dtype=np.uint8)
    blk[0, 0] = BOS
    p = 1
    blk[p:p + len(qt), 0] = qt
    blk[p:p + len(qt), 1] = co[:len(qt)]
    blk[p:p + len(qt), 2] = ci[:len(qt)]
    p += len(qt)
    blk[p:p + len(at), 0] = at
    blk[p:p + len(at), 1] = co[len(qt):]
    blk[p:p + len(at), 2] = ci[len(qt):]
    blk[p:p + len(at), 3] = 1
    p += len(at)
    blk[p, 0] = EOS
    blk[p, 3] = 1
    return blk


TRAIN_RAW_DIR = os.path.join(CLRS_DIR, "train_raw")


def _iter_rows(repo, splits):
    """Yield dict rows. Prefers locally downloaded parquet (the train repo is
    531 MB and HF streaming times out on this cluster); falls back to
    streaming."""
    local = os.path.join(TRAIN_RAW_DIR, "data")
    if repo == HF_TRAIN and os.path.isdir(local):
        import glob
        import pyarrow.parquet as pq
        for f in sorted(glob.glob(os.path.join(local, "*.parquet"))):
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(batch_size=4096):
                for r in batch.to_pylist():
                    yield r
        return
    from datasets import load_dataset
    for split in splits:
        for r in load_dataset(repo, split=split, streaming=True):
            yield r


def _iter_hf(repo, splits, quadrant, max_per_cell):
    """Keep only (algo, n) pairs inside the requested quadrant's bands.
    Yields (algo, n, question, answer).

    n comes from the `length` column when present (test repo) and from
    infer_n() otherwise (train repo ships no length column). selftest()
    cross-checks the two on the test split."""
    want = QUADRANTS[quadrant]
    kept = {a: 0 for a in want}
    for r in _iter_rows(repo, splits):
        a = r["algo_name"]
        if a not in want or kept[a] >= max_per_cell:
            continue
        lo, hi = want[a]
        n = int(r["length"]) if "length" in r else infer_n(r["question"])
        if lo <= n <= hi:
            kept[a] += 1
            yield a, n, r["question"], r["answer"]
        if all(kept[x] >= max_per_cell for x in want):
            return


def blocks_path(quadrant, seq_len=SEQ_LEN, split="train"):
    return os.path.join(CLRS_DIR, f"clrs_{split}_{quadrant}_{seq_len}.npy")


def val_cache_path(quadrant, seq_len=SEQ_LEN):
    return os.path.join(LACT_LLM_DIR, f"val_cache_clrs_{quadrant}_{seq_len}.pt")


def ensure_train_blocks(quadrant="2d_long", seq_len=SEQ_LEN, max_per_cell=200_000):
    """Build (once, cached) the train block array for one quadrant."""
    path = blocks_path(quadrant, seq_len, "train")
    if os.path.exists(path):
        return path
    os.makedirs(CLRS_DIR, exist_ok=True)
    t0 = time.time()
    out, dropped = [], 0
    for a, n, q, ans in _iter_hf(HF_TRAIN, ["train"], quadrant, max_per_cell):
        blk = make_block(q, ans, seq_len)
        if blk is None:
            dropped += 1
            continue
        out.append(blk)
        if len(out) % 20000 == 0:
            print(f"[clrs] {quadrant}: {len(out):,} blocks "
                  f"({time.time() - t0:.0f}s)", flush=True)
    assert out, f"no blocks built for quadrant {quadrant}"
    arr = np.stack(out)
    tmp = path + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)
    print(f"[clrs] {quadrant}: {arr.shape[0]:,} blocks x {seq_len} "
          f"({arr.nbytes / 1e9:.2f} GB, {dropped} oversize dropped) -> {path} "
          f"in {time.time() - t0:.0f}s", flush=True)
    return path


def get_or_build_clrs_val_set(n_blocks, cache_path, quadrant="2d_long",
                              seq_len=SEQ_LEN):
    """Fixed val set from the HELD-OUT test repo (seeds test_1..test_5 are
    disjoint from train). int64 [n_blocks, seq_len, 4]."""
    if os.path.exists(cache_path):
        v = torch.load(cache_path, map_location="cpu")
        if v.dim() == 3 and v.shape[0] >= n_blocks and v.shape[1] == seq_len:
            print(f"[clrs] reusing cached val set {cache_path} "
                  f"({v.shape[0]} blocks)", flush=True)
            return v[:n_blocks]
        print(f"[clrs] WARNING: cached val {cache_path} shape {tuple(v.shape)} "
              f"insufficient; rebuilding.", flush=True)
    out = []
    for a, n, q, ans in _iter_hf(HF_TEST, TEST_SPLITS, quadrant, n_blocks):
        blk = make_block(q, ans, seq_len)
        if blk is not None:
            out.append(blk)
        if len(out) >= n_blocks:
            break
    assert len(out) >= n_blocks, (
        f"val set for {quadrant}: only {len(out)} blocks (< {n_blocks})")
    val = torch.from_numpy(np.stack(out[:n_blocks]).astype(np.int64))
    tmp = cache_path + ".tmp"
    torch.save(val, tmp)
    os.replace(tmp, cache_path)
    print(f"[clrs] saved val set to {cache_path} {tuple(val.shape)}", flush=True)
    return val


def _epoch_seed(data_seed, epoch):
    """Deterministic per-epoch shuffle seed (stable across processes)."""
    return (int(data_seed) * 1_000_003 + int(epoch) * 2_654_435_761) & 0x7FFFFFFF


class ClrsBlockStream:
    """Resumable shuffled-block stream. Emits uint8 [seq_len, 4] blocks as
    python lists-of-lists, matching DnaBlockStream's contract (exact O(1)
    resume via n_emitted)."""

    def __init__(self, blocks_path, data_seed, seq_len):
        self.blocks = np.load(blocks_path, mmap_mode="r")
        assert self.blocks.ndim == 3 and self.blocks.shape[2] == 4, (
            f"expected [n, seq_len, 4], got {self.blocks.shape}")
        assert self.blocks.shape[1] == seq_len, (
            f"train blocks seq_len {self.blocks.shape[1]} != {seq_len}")
        self.N = int(self.blocks.shape[0])
        assert self.N > 0, "no train blocks"
        self.seq_len = seq_len
        self.data_seed = int(data_seed)
        self.n_emitted = 0
        self._epoch = -1
        self._perm = None
        self._ensure_epoch(0)

    def _ensure_epoch(self, epoch):
        if epoch != self._epoch:
            g = np.random.RandomState(_epoch_seed(self.data_seed, epoch))
            self._perm = g.permutation(self.N)
            self._epoch = epoch

    def __iter__(self):
        return self

    def __next__(self):
        epoch, idx = divmod(self.n_emitted, self.N)
        self._ensure_epoch(epoch)
        block = self.blocks[self._perm[idx]].astype(np.int64)
        self.n_emitted += 1
        return block.tolist()

    def state(self):
        return {"n_raw_consumed": self.n_emitted,
                "buf": torch.empty(0, dtype=torch.int64)}

    def restore(self, state):
        self.n_emitted = int(state["n_raw_consumed"])
        self._ensure_epoch(self.n_emitted // self.N)


# ---------------------------------------------------------------------------
# Self-test: proves the coordinate builder does what the design claims.
# ---------------------------------------------------------------------------
def selftest():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{'  ' + detail if detail else ''}")

    print("[clrs] coordinate builder self-test")

    # 1. nested matrix -> (row, col)
    s = "A: [[9 8], [7 6]]"
    co, ci = build_coords(s)
    got = {s[i]: (int(co[i]), int(ci[i])) for i in range(len(s)) if s[i] in "9876"}
    check("nested matrix gives (row, col)",
          got == {"9": (0, 0), "8": (0, 1), "7": (1, 0), "6": (1, 1)}, str(got))

    # 2. flat list -> outer axis constant 0, inner axis = index
    s = "key: [5 4 3]"
    co, ci = build_coords(s)
    got = {s[i]: (int(co[i]), int(ci[i])) for i in range(len(s)) if s[i] in "543"}
    check("flat list gives (0, index)",
          got == {"5": (0, 0), "4": (0, 1), "3": (0, 2)}, str(got))

    # 3. multi-char elements keep one constant coordinate
    s = "[0.315 0.014]"
    co, ci = build_coords(s)
    check("float element has a single constant coord",
          len(set(zip(co[1:6].tolist(), ci[1:6].tolist()))) == 1
          and int(ci[7]) == 1,
          f"first={set(zip(co[1:6].tolist(), ci[1:6].tolist()))} second_idx={int(ci[7])}")

    # 4. comma-separated trace arrays -> (step, index)
    s = "[1 2], [3 4], [5 6]"
    co, ci = build_coords(s)
    got = {s[i]: (int(co[i]), int(ci[i])) for i in range(len(s)) if s[i] in "123456"}
    check("trace arrays give (step, index)",
          got == {"1": (0, 0), "2": (0, 1), "3": (1, 0),
                  "4": (1, 1), "5": (2, 0), "6": (2, 1)}, str(got))

    # 5. block round-trip
    q, a = "bfs:\ns: 1, A: [[0 1], [1 0]]\ntrace | pi:\n", "[0 1] | [0 0]\n\n"
    blk = make_block(q, a, 128)
    check("block builds", blk is not None)
    if blk is not None:
        toks = blk[:, 0]
        text = ClrsCharTokenizer.decode(toks[1:1 + len(q) + len(a)])
        check("token round-trip is exact", text == q + a)
        nmask = int(blk[:, 3].sum())
        check("loss mask covers answer + EOS", nmask == len(a) + 1,
              f"{nmask} vs {len(a) + 1}")
        check("BOS/EOS placed", toks[0] == BOS and toks[1 + len(q) + len(a)] == EOS)

    # 6. the dimensionality claim, on the real corpus
    try:
        from datasets import load_dataset
        print("[clrs] verifying dimensionality on real CLRS-Text samples ...")
        need = {}
        for quad in ("2d_long", "1d_long"):
            for a_, (lo_, hi_) in QUADRANTS[quad].items():
                need[(a_, hi_)] = quad   # check the top of each band
        seen = {}
        ds = load_dataset(HF_TEST, split="test_1", streaming=True)
        for r in ds:
            k = (r["algo_name"], int(r["length"]))
            if k in need and k not in seen:
                seen[k] = coord_report(r["question"], r["answer"])
            if len(seen) == len(need):
                break
        for k, rep in sorted(seen.items()):
            quad = need[k]
            is2d = quad.startswith("2d")
            # The question's OUTER axis is the discriminator. A matrix sweeps it
            # over all n rows; a flat-array problem only ever reaches the small
            # count of top-level fields ('key: [...], initial_trace: [...]'),
            # so the 2-D address space is ~n x n and the 1-D one is ~1 x n.
            good = (rep["q_outer_max"] >= k[1] - 1) if is2d \
                else (rep["q_outer_max"] <= 3)
            check(f"{k[0]} n={k[1]} ({quad}) question outer axis "
                  f"{'sweeps rows' if is2d else 'stays small'}", good, str(rep))
    except Exception as e:  # network/dataset unavailable -> report, do not fake
        print(f"  [SKIP] real-corpus check unavailable: {type(e).__name__}: {e}")

    print(f"[clrs] self-test {'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


def main():
    """Usage:
      python clrs_data.py selftest
      python clrs_data.py build <quadrant> [seq_len] [val_blocks]
    """
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if cmd == "selftest":
        raise SystemExit(selftest())
    if cmd == "build":
        quad = sys.argv[2] if len(sys.argv) > 2 else "2d_long"
        seq_len = int(sys.argv[3]) if len(sys.argv) > 3 else SEQ_LEN
        n_val = int(sys.argv[4]) if len(sys.argv) > 4 else 512
        ensure_train_blocks(quad, seq_len)
        v = get_or_build_clrs_val_set(n_val, val_cache_path(quad, seq_len),
                                      quad, seq_len)
        print(f"[clrs] {quad} ready (val {tuple(v.shape)}).", flush=True)
        return
    raise SystemExit(f"unknown command: {cmd}")


if __name__ == "__main__":
    main()
