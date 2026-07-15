# -*- coding: utf-8 -*-
"""Synthetic exact-offset-copy task for train_small.py (--synthetic copy).

Diagnostic probe (Q16): in natural language the hidden-site rotary never adds
on top of the input rotary (ga_honly/LEDGER.md). Hypothesis: the blocker is
the TASK (LM barely rewards precise positional retrieval), not the 1D
coordinate. This task makes exact relative position the ONLY signal, and the
copy offset (2560) exceeds 2x the attention window (1024) and crosses lact
chunk boundaries (1024), so the fast-weight update->apply path must carry the
retrieval.

Sequence layout (seq_len = 4096):
  [0 .. 510]      noise                     (random ids in [10, 1010))
  [511]           SRC_MARKER (id 4)
  [512 .. 767]    SOURCE span (256 random noise-vocab tokens)
  [768 .. 3070]   noise
  [3071]          RECALL_MARKER (id 5)
  [3072 .. 3327]  COPY = exact repeat of the source span  <- the ONLY
                  supervised positions (labels = -100 everywhere else)
  [3328 .. 4095]  noise

Each supervised token at position t equals the token at t - 2560. The model's
forward shifts labels internally (logits at t score labels[t+1]), so the
first supervised prediction is made at position 3071 (the RECALL_MARKER).

Marker ids 4 and 5 do not collide with the fla-hub/transformer-1.3B-100B
specials (bos=1, eos=2) and never appear in the noise vocab [10, 1010).

Determinism: every sequence is a pure function of (data_seed, sample_index)
via a per-sample torch.Generator — identical batch stream for every variant
with the same data_seed, and exact resume by restoring the consumed-sample
counter. Validation sequences use sample indices >= VAL_INDEX_BASE (1e9),
disjoint from the training indices (0, 1, 2, ...).
"""

import torch

# --- layout constants (seq_len 4096) ---------------------------------------
SEQ_LEN = 4096
NOISE_LO, NOISE_HI = 10, 1010          # noise/content vocab: 1000 ids
SRC_MARKER_ID = 4
RECALL_MARKER_ID = 5
SRC_MARKER_POS = 511
SRC_START, SRC_END = 512, 768          # source span, length 256
RECALL_MARKER_POS = 3071
COPY_START, COPY_END = 3072, 3328      # supervised copy region, length 256
COPY_OFFSET = COPY_START - SRC_START   # 2560
VAL_INDEX_BASE = 10 ** 9               # val sample indices: VAL_INDEX_BASE + j

IGNORE_INDEX = -100


def make_sequence(data_seed, index, seq_len=SEQ_LEN):
    """Deterministic sequence for (data_seed, index); returns int64 [seq_len]."""
    assert seq_len == SEQ_LEN, f"copy task layout is defined for seq_len={SEQ_LEN}"
    g = torch.Generator()
    g.manual_seed((int(data_seed) * 1_000_003 + int(index)) % (2 ** 63 - 1))
    seq = torch.randint(NOISE_LO, NOISE_HI, (seq_len,), generator=g, dtype=torch.int64)
    seq[SRC_MARKER_POS] = SRC_MARKER_ID
    seq[RECALL_MARKER_POS] = RECALL_MARKER_ID
    seq[COPY_START:COPY_END] = seq[SRC_START:SRC_END]
    return seq


def make_labels(input_ids):
    """Labels for a [.., seq_len] batch: input ids on the copy region,
    IGNORE_INDEX (-100) everywhere else. The model shifts internally."""
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    labels[..., COPY_START:COPY_END] = input_ids[..., COPY_START:COPY_END]
    return labels


def build_val_set(data_seed, n_seqs=64, seq_len=SEQ_LEN):
    """Fixed held-out val set: int64 [n_seqs, seq_len], indices >= VAL_INDEX_BASE."""
    return torch.stack(
        [make_sequence(data_seed, VAL_INDEX_BASE + j, seq_len) for j in range(n_seqs)]
    )


class SyntheticCopyStream:
    """Resumable training stream, interface-compatible with
    data_utils.PackedBlockStream (yields python lists; state()/restore()).

    state() reuses the PackedBlockStream schema {"n_raw_consumed", "buf"} so
    train_small.save_checkpoint works unchanged: n_raw_consumed = number of
    sequences emitted so far (the next sample index), buf always empty (the
    stream is a pure function of the sample index — resume is exact).
    """

    def __init__(self, data_seed, seq_len=SEQ_LEN):
        self.data_seed = data_seed
        self.seq_len = seq_len
        self.n_raw_consumed = 0

    def __iter__(self):
        return self

    def __next__(self):
        seq = make_sequence(self.data_seed, self.n_raw_consumed, self.seq_len)
        self.n_raw_consumed += 1
        assert self.n_raw_consumed < VAL_INDEX_BASE, \
            "training stream ran into the val index range"
        return seq.tolist()

    def state(self):
        return {
            "n_raw_consumed": self.n_raw_consumed,
            "buf": torch.tensor([], dtype=torch.int64),
        }

    def restore(self, state, log_every=None):
        assert self.n_raw_consumed == 0, \
            "restore() must be called on a fresh SyntheticCopyStream"
        assert len(state["buf"]) == 0, \
            "checkpoint stream state has carry-over tokens; not a synthetic-copy checkpoint"
        self.n_raw_consumed = int(state["n_raw_consumed"])
        print(f"[data] synthetic copy stream restored at sample index "
              f"{self.n_raw_consumed:,} (exact; pure function of index)", flush=True)
