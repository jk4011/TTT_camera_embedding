# -*- coding: utf-8 -*-
"""hg38 char-level DNA data path for train_small.py (cross-task validation #2).

Same interface as data_utils.PackedBlockStream so the trainer's checkpoint/
resume machinery (state()/restore(), batch_generator with_state) works unchanged.

Design (fixed by NODE_QUEUE.md P0 — do not change):
  * char tokenizer, vocab_size=8: {PAD:0, BOS:1, EOS:2, A:3, C:4, G:5, T:6, N:7}
    soft-mask lowercase -> uppercase; any non-ACGT char -> N.
  * chromosomes chr1..chr22, chrX (scaffolds / chrM / chrY excluded).
    val = chr20 (held out entirely); train = the rest.
  * seq_len 4096 contiguous blocks, NO document boundary / EOS insertion
    (pure continuous ACGTN stream). Blocks whose N ratio > 50% are dropped
    (telomere / centromere gaps).
  * train reads blocks in a data_seed-shuffled order, resumable exactly via
    state()/restore() ({"n_raw_consumed", "buf"} format).
  * val cache: first 488 kept blocks of chr20 (~2M tokens) ->
    val_cache_hg38_4096.pt (same tensor format as get_or_build_val_set).
"""

import os
import gzip
import time

import numpy as np
import torch

LACT_LLM_DIR = os.path.dirname(os.path.abspath(__file__))
DNA_DIR = "/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/datasets/dna"
FA_GZ = os.path.join(DNA_DIR, "hg38.fa.gz")

SEQ_LEN = 4096  # default block length (WAVE 1); WAVE 2 uses 32768


def train_blocks_path(seq_len=SEQ_LEN):
    return os.path.join(DNA_DIR, f"hg38_train_blocks_{seq_len}.npy")


def val_cache_path(seq_len=SEQ_LEN):
    return os.path.join(LACT_LLM_DIR, f"val_cache_hg38_{seq_len}.pt")


# Back-compat aliases for the WAVE 1 (4096) artifacts.
TRAIN_BLOCKS = train_blocks_path(SEQ_LEN)
VAL_CACHE = val_cache_path(SEQ_LEN)
VOCAB_SIZE = 8
PAD, BOS, EOS, A, C, G, T, N = range(8)  # 0..7; data only ever contains 3..7
N_CODE = 7

VAL_CHROM = "chr20"
TRAIN_CHROMS = [f"chr{i}" for i in range(1, 23) if i != 20] + ["chrX"]
N_HEAVY_MAX = 0.5  # drop a block if its N fraction is strictly greater than this


class DnaCharTokenizer:
    """Minimal stub with the attributes train_small.build_config reads."""
    bos_token_id = BOS
    eos_token_id = EOS
    vocab_size = VOCAB_SIZE

    def __len__(self):
        return VOCAB_SIZE


def _build_lut():
    """256-entry byte -> code table; everything not ACGT (either case) -> N."""
    lut = np.full(256, N_CODE, dtype=np.uint8)
    for ch, code in (("A", A), ("C", C), ("G", G), ("T", T)):
        lut[ord(ch)] = code
        lut[ord(ch.lower())] = code
    return lut


_LUT = _build_lut()


def _encode(seq_bytes):
    """ASCII bytes -> uint8 code array (A/C/G/T/N -> 3/4/5/6/7)."""
    return _LUT[np.frombuffer(seq_bytes, dtype=np.uint8)]


def _iter_fasta(path, wanted):
    """Yield (chrom_name, concatenated_sequence_bytes) for chroms in `wanted`,
    in file order. Sequence of unwanted chroms is not accumulated (memory)."""
    wanted = set(wanted)
    name = None
    parts = None
    with gzip.open(path, "rb") as f:
        for line in f:
            if line[:1] == b">":
                if name in wanted and parts:
                    yield name, b"".join(parts)
                name = line[1:].split()[0].decode()
                parts = [] if name in wanted else None
            elif parts is not None:
                parts.append(line.strip())
        if name in wanted and parts:
            yield name, b"".join(parts)


def _blocks_from_arr(arr, seq_len):
    """Chunk a 1-D code array into [n, seq_len] blocks, dropping the tail
    partial block and any block whose N fraction > 0.5."""
    n = len(arr) // seq_len
    if n == 0:
        return np.empty((0, seq_len), dtype=np.uint8)
    b = arr[: n * seq_len].reshape(n, seq_len)
    n_frac = (b == N_CODE).mean(axis=1)
    return np.ascontiguousarray(b[n_frac <= N_HEAVY_MAX])


def ensure_train_blocks(seq_len=SEQ_LEN):
    """Build (once, cached) the shuffled-order source array of train blocks:
    uint8 [n_blocks, seq_len] concatenated over TRAIN_CHROMS in file order.
    Returns the .npy path (loaded mmap by DnaBlockStream)."""
    path = train_blocks_path(seq_len)
    if os.path.exists(path):
        return path
    assert os.path.exists(FA_GZ), (
        f"hg38 not downloaded: {FA_GZ} missing "
        f"(wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz)")
    t0 = time.time()
    chunks = []
    total = 0
    for name, seq in _iter_fasta(FA_GZ, TRAIN_CHROMS):
        blk = _blocks_from_arr(_encode(seq), seq_len)
        chunks.append(blk)
        total += blk.shape[0]
        print(f"[dna] {name}: {blk.shape[0]:,} blocks kept "
              f"({total:,} total, {time.time() - t0:.0f}s)", flush=True)
    allb = np.concatenate(chunks, axis=0)
    tmp = path + ".tmp.npy"
    np.save(tmp, allb)
    os.replace(tmp, path)
    print(f"[dna] train blocks: {allb.shape[0]:,} x {seq_len} "
          f"({allb.nbytes / 1e9:.1f} GB) -> {path} in {time.time() - t0:.0f}s",
          flush=True)
    return path


def get_or_build_dna_val_set(n_blocks, cache_path, seq_len=SEQ_LEN):
    """First `n_blocks` kept blocks of chr20 as the fixed val set (int64
    [n_blocks, seq_len]). Cached; independent of the training stream position."""
    if os.path.exists(cache_path):
        v = torch.load(cache_path, map_location="cpu")
        if v.dim() == 2 and v.shape[0] >= n_blocks and v.shape[1] == seq_len:
            print(f"[dna] reusing cached val set {cache_path} "
                  f"({v.numel()} tokens)", flush=True)
            return v[:n_blocks]
        print(f"[dna] WARNING: cached val {cache_path} shape {tuple(v.shape)} "
              f"insufficient for {n_blocks} x {seq_len}; rebuilding.", flush=True)
    blk = None
    for _, seq in _iter_fasta(FA_GZ, [VAL_CHROM]):
        blk = _blocks_from_arr(_encode(seq), seq_len)
    assert blk is not None and blk.shape[0] >= n_blocks, (
        f"chr20 yielded {0 if blk is None else blk.shape[0]} kept blocks "
        f"(< {n_blocks})")
    val = torch.from_numpy(blk[:n_blocks].astype(np.int64))
    tmp = cache_path + ".tmp"
    torch.save(val, tmp)
    os.replace(tmp, cache_path)
    print(f"[dna] saved val set to {cache_path} ({val.numel()} tokens)", flush=True)
    return val


def _epoch_seed(data_seed, epoch):
    """Deterministic per-epoch shuffle seed (stable across processes)."""
    return (int(data_seed) * 1_000_003 + int(epoch) * 2_654_435_761) & 0x7FFFFFFF


class DnaBlockStream:
    """Resumable shuffled-block stream over the pre-chunked train blocks.

    Emits blocks (python int lists of length seq_len) in a data_seed-derived
    permutation; on exhausting an epoch it reshuffles with a fresh per-epoch
    seed and continues (>1 epoch for a 3B-token budget). Exact resume via
    state()/restore(): n_emitted alone determines epoch and within-epoch index,
    so restore is O(1) (no stream fast-forward)."""

    def __init__(self, blocks_path, data_seed, seq_len):
        self.blocks = np.load(blocks_path, mmap_mode="r")
        assert self.blocks.shape[1] == seq_len, (
            f"train blocks seq_len {self.blocks.shape[1]} != {seq_len}")
        self.N = int(self.blocks.shape[0])
        assert self.N > 0, "no train blocks"
        self.seq_len = seq_len
        self.data_seed = int(data_seed)
        self.n_emitted = 0          # == n_raw_consumed in state()
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


def main():
    """Standalone preprocessing: build the train-block array and val cache.

    Usage: python dna_data.py [seq_len] [val_tokens]
    (val block count = val_tokens // seq_len, matching train_small.py)"""
    import sys
    seq_len = int(sys.argv[1]) if len(sys.argv) > 1 else SEQ_LEN
    val_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 2_000_000
    n_val = val_tokens // seq_len
    ensure_train_blocks(seq_len)
    v = get_or_build_dna_val_set(n_val, val_cache_path(seq_len), seq_len)
    print(f"[dna] preprocessing complete (seq_len={seq_len}, "
          f"val {tuple(v.shape)} = {v.numel():,} tokens).", flush=True)


if __name__ == "__main__":
    main()
