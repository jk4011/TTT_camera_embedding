# -*- coding: utf-8 -*-
"""Lakh MIDI (LMD-full) REMI symbolic-music data path for train_small.py
(cross-task validation #3).

Same interface as data_utils.PackedBlockStream / dna_data.DnaBlockStream, so the
trainer's checkpoint/resume machinery (state()/restore(), batch_generator
with_state) works unchanged.

Design (mirrors dna_data.py):
  * miditok REMI tokenizer, config pinned in TOKENIZER_PARAMS below
    (multi-track single stream: use_programs + one_token_stream_for_programs).
    PAD/BOS/EOS/MASK = 0/1/2/3; vocab_size is recorded in the meta JSON so the
    trainer never needs miditok at runtime.
  * file-level train/val split (NOT random blocks): a fixed SPLIT_SEED
    permutation holds out VAL_FILES MIDI files entirely for validation.
  * every usable piece is tokenized and concatenated into ONE contiguous uint16
    stream with EOS between pieces (exactly like the fineweb packer), then cut
    into seq_len blocks (tail partial block dropped).
  * train reads blocks in a data_seed-shuffled order, resumable exactly via
    state()/restore() ({"n_raw_consumed", "buf"} format).
  * val cache: first `val_tokens // seq_len` blocks of the held-out files ->
    val_cache_music_4096.pt (same tensor format as get_or_build_val_set).

Artifacts (all on lustre, MUSIC_DIR):
  lmd_full.tar.gz                     source archive (1,768,163,879 bytes)
  remi_tokenizer.json                 miditok save_params() of the exact tokenizer
  lmd_remi_meta.json                  vocab/special ids, file counts, token totals
  lmd_remi_train_stream_u16.npy       flat train token stream (uint16)
  lmd_remi_val_stream_u16.npy         flat val token stream (uint16, disjoint files)
  lmd_remi_train_blocks_<S>.npy       uint16 [n_blocks, S] (what the stream reads)

Preprocessing (CPU only, ~30 min on 48 workers):
  ../.venv_llm/bin/python music_data.py [seq_len] [val_tokens]
"""

import json
import os
import sys
import time

import numpy as np
import torch

LACT_LLM_DIR = os.path.dirname(os.path.abspath(__file__))
MUSIC_DIR = "/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/datasets/music"
LMD_URL = "http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz"
LMD_TARBALL = os.path.join(MUSIC_DIR, "lmd_full.tar.gz")
# Extracted MIDI tree (176,581 files). Node-local by default: it is only needed
# to BUILD the corpus; the .npy artifacts on lustre are the durable product.
MIDI_ROOT = os.environ.get("LMD_MIDI_DIR", "/tmp/lmd_stage/lmd_full")

SEQ_LEN = 4096

# ---- tokenizer -----------------------------------------------------------
# Standard miditok REMI config (the documented example config) + the two flags
# that make a multi-track General-MIDI corpus a single token stream.
TOKENIZER_PARAMS = dict(
    pitch_range=(21, 109),
    beat_res={(0, 4): 8, (4, 12): 4},
    num_velocities=24,
    special_tokens=["PAD", "BOS", "EOS", "MASK"],
    use_chords=True,
    use_rests=False,
    use_tempos=True,
    use_time_signatures=False,
    use_programs=True,               # 128 GM programs + drums
    one_token_stream_for_programs=True,   # all tracks -> one interleaved stream
    num_tempos=32,
    tempo_range=(40, 250),
)
PAD, BOS, EOS, MASK = 0, 1, 2, 3      # special_tokens order above
EOS_ID = EOS

# ---- corpus construction knobs (fixed; changing them invalidates artifacts) --
SPLIT_SEED = 20260803    # file-level train/val split permutation (NOT data_seed)
VAL_FILES = 1024         # files held out entirely for validation
MIN_TOKENS = 16          # pieces shorter than this are dropped as empty
FILE_TIMEOUT_S = 20      # per-file parse/tokenize watchdog (pathological MIDI)


def tokenizer_params_path():
    return os.path.join(MUSIC_DIR, "remi_tokenizer.json")


def meta_path():
    return os.path.join(MUSIC_DIR, "lmd_remi_meta.json")


def train_stream_path():
    return os.path.join(MUSIC_DIR, "lmd_remi_train_stream_u16.npy")


def val_stream_path():
    return os.path.join(MUSIC_DIR, "lmd_remi_val_stream_u16.npy")


def train_blocks_path(seq_len=SEQ_LEN):
    return os.path.join(MUSIC_DIR, f"lmd_remi_train_blocks_{seq_len}.npy")


def val_cache_path(seq_len=SEQ_LEN):
    return os.path.join(LACT_LLM_DIR, f"val_cache_music_{seq_len}.pt")


# Back-compat aliases for the 4096 artifacts.
TRAIN_BLOCKS = train_blocks_path(SEQ_LEN)
VAL_CACHE = val_cache_path(SEQ_LEN)


def build_tokenizer():
    """The pinned REMI tokenizer (needs miditok; only used when BUILDING)."""
    from miditok import REMI, TokenizerConfig
    return REMI(TokenizerConfig(**TOKENIZER_PARAMS))


def load_meta():
    """Corpus metadata written by the build (vocab size, token counts, ...)."""
    path = meta_path()
    assert os.path.exists(path), (
        f"music corpus metadata missing: {path} "
        f"(run: ../.venv_llm/bin/python music_data.py)")
    with open(path) as f:
        return json.load(f)


def vocab_size():
    return int(load_meta()["vocab_size"])


class MusicRemiTokenizer:
    """Minimal stub with the attributes train_small.build_config reads.

    Reads the vocab size from the corpus meta JSON, so training does not import
    miditok (and cannot silently disagree with the tokenized corpus)."""

    def __init__(self):
        meta = load_meta()
        self.vocab_size = int(meta["vocab_size"])
        self.bos_token_id = int(meta["bos_token_id"])
        self.eos_token_id = int(meta["eos_token_id"])
        self.pad_token_id = int(meta["pad_token_id"])

    def __len__(self):
        return self.vocab_size


# ---- file enumeration / split -------------------------------------------
def list_midi_files(midi_root=None):
    """All MIDI files under the LMD tree, in a deterministic (sorted) order."""
    midi_root = midi_root or MIDI_ROOT
    assert os.path.isdir(midi_root), (
        f"extracted LMD tree not found at {midi_root} "
        f"(tar -xzf {LMD_TARBALL} -C $(dirname {midi_root}); "
        f"or set LMD_MIDI_DIR)")
    files = []
    for dirpath, dirnames, filenames in os.walk(midi_root):
        dirnames.sort()
        for fn in sorted(filenames):
            if fn.lower().endswith((".mid", ".midi")):
                files.append(os.path.join(dirpath, fn))
    files.sort()
    return files


def split_files(files):
    """Fixed file-level split: (train_files, val_files). Independent of
    data_seed — the val set must be the same for every run/seed."""
    perm = np.random.RandomState(SPLIT_SEED).permutation(len(files))
    val = [files[i] for i in perm[:VAL_FILES]]
    train = [files[i] for i in perm[VAL_FILES:]]
    return train, val


# ---- parallel tokenization ----------------------------------------------
_TOK = None


class _FileTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _FileTimeout()


def _init_worker():
    global _TOK
    import signal
    _TOK = build_tokenizer()
    signal.signal(signal.SIGALRM, _alarm_handler)


def _tokenize_chunk(paths):
    """Tokenize a chunk of files -> (uint16 token array with EOS separators,
    n_ok, n_fail, n_empty, {error_name: count}). Order-preserving."""
    import signal
    from pathlib import Path
    global _TOK
    if _TOK is None:
        _init_worker()
    out = []
    n_ok = n_fail = n_empty = 0
    errs = {}
    eos_arr = np.array([EOS_ID], dtype=np.uint16)
    for p in paths:
        try:
            signal.setitimer(signal.ITIMER_REAL, FILE_TIMEOUT_S)
            seq = _TOK(Path(p))
            ids = seq.ids if hasattr(seq, "ids") else None
            signal.setitimer(signal.ITIMER_REAL, 0)
        except BaseException as e:   # corrupt MIDI, timeouts, miditok asserts
            signal.setitimer(signal.ITIMER_REAL, 0)
            n_fail += 1
            k = type(e).__name__
            errs[k] = errs.get(k, 0) + 1
            continue
        if not ids or len(ids) < MIN_TOKENS:
            n_empty += 1
            continue
        out.append(np.asarray(ids, dtype=np.uint16))
        out.append(eos_arr)
        n_ok += 1
    arr = np.concatenate(out) if out else np.empty(0, dtype=np.uint16)
    return arr, n_ok, n_fail, n_empty, errs


def tokenize_files(files, out_path, n_workers=None, chunk=128, tag=""):
    """Tokenize `files` in order into one flat uint16 .npy stream.

    Deterministic: chunks are consumed in order (imap), so the concatenated
    stream does not depend on worker scheduling. Returns a stats dict."""
    import multiprocessing as mp

    n_workers = n_workers or max(1, min(48, (os.cpu_count() or 8) - 4))
    chunks = [files[i:i + chunk] for i in range(0, len(files), chunk)]
    parts, n_ok, n_fail, n_empty = [], 0, 0, 0
    errs = {}
    t0 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(n_workers, initializer=_init_worker) as pool:
        for i, (arr, ok, fail, empty, e) in enumerate(
                pool.imap(_tokenize_chunk, chunks), start=1):
            parts.append(arr)
            n_ok += ok
            n_fail += fail
            n_empty += empty
            for k, v in e.items():
                errs[k] = errs.get(k, 0) + v
            if i % 50 == 0 or i == len(chunks):
                tot = sum(len(a) for a in parts)
                dt = time.time() - t0
                done = min(i * chunk, len(files))
                print(f"[music]{tag} {done:,}/{len(files):,} files "
                      f"({n_ok:,} ok / {n_fail:,} fail / {n_empty:,} empty) "
                      f"{tot:,} tokens {dt:.0f}s "
                      f"eta {dt / max(1, i) * (len(chunks) - i):.0f}s", flush=True)
    stream = np.concatenate(parts) if parts else np.empty(0, dtype=np.uint16)
    del parts
    tmp = out_path + ".tmp.npy"
    np.save(tmp, stream)
    os.replace(tmp, out_path)
    print(f"[music]{tag} wrote {len(stream):,} tokens -> {out_path} "
          f"({stream.nbytes / 1e9:.2f} GB) in {time.time() - t0:.0f}s", flush=True)
    return {"n_files": len(files), "n_ok": n_ok, "n_fail": n_fail,
            "n_empty": n_empty, "n_tokens": int(len(stream)), "errors": errs}


def _pack_blocks(stream_path, seq_len, out_path):
    """Cut a flat uint16 stream into [n, seq_len] blocks (drop tail partial)."""
    flat = np.load(stream_path, mmap_mode="r")
    n = int(len(flat) // seq_len)
    assert n > 0, f"{stream_path} has {len(flat)} tokens (< seq_len {seq_len})"
    tmp = out_path + ".tmp.npy"
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint16,
                                    shape=(n, seq_len))
    step = max(1, 1_000_000 // seq_len)
    for i in range(0, n, step):
        j = min(n, i + step)
        out[i:j] = np.asarray(flat[i * seq_len:j * seq_len]).reshape(j - i, seq_len)
    out.flush()
    del out
    os.replace(tmp, out_path)
    print(f"[music] packed {n:,} x {seq_len} blocks -> {out_path}", flush=True)
    return n


def ensure_train_blocks(seq_len=SEQ_LEN):
    """Build (once, cached) the uint8/uint16 [n_blocks, seq_len] train array.
    Returns the .npy path (loaded mmap by MusicBlockStream)."""
    path = train_blocks_path(seq_len)
    if os.path.exists(path):
        return path
    assert os.path.exists(train_stream_path()), (
        f"music train stream missing: {train_stream_path()} "
        f"(run: ../.venv_llm/bin/python music_data.py)")
    _pack_blocks(train_stream_path(), seq_len, path)
    return path


def get_or_build_music_val_set(n_blocks, cache_path, seq_len=SEQ_LEN):
    """First `n_blocks` blocks of the held-out FILES as the fixed val set
    (int64 [n_blocks, seq_len]). Cached; independent of the training stream."""
    if os.path.exists(cache_path):
        v = torch.load(cache_path, map_location="cpu")
        if v.dim() == 2 and v.shape[0] >= n_blocks and v.shape[1] == seq_len:
            print(f"[music] reusing cached val set {cache_path} "
                  f"({v.numel()} tokens)", flush=True)
            return v[:n_blocks]
        print(f"[music] WARNING: cached val {cache_path} shape {tuple(v.shape)} "
              f"insufficient for {n_blocks} x {seq_len}; rebuilding.", flush=True)
    assert os.path.exists(val_stream_path()), (
        f"music val stream missing: {val_stream_path()} "
        f"(run: ../.venv_llm/bin/python music_data.py)")
    flat = np.load(val_stream_path(), mmap_mode="r")
    have = int(len(flat) // seq_len)
    assert have >= n_blocks, (
        f"held-out files yield {have} blocks of {seq_len} (< {n_blocks}); "
        f"raise VAL_FILES and rebuild")
    val = torch.from_numpy(
        np.asarray(flat[:n_blocks * seq_len]).reshape(n_blocks, seq_len).astype(np.int64))
    tmp = cache_path + ".tmp"
    torch.save(val, tmp)
    os.replace(tmp, cache_path)
    print(f"[music] saved val set to {cache_path} ({val.numel()} tokens)", flush=True)
    return val


def _epoch_seed(data_seed, epoch):
    """Deterministic per-epoch shuffle seed (stable across processes)."""
    return (int(data_seed) * 1_000_003 + int(epoch) * 2_654_435_761) & 0x7FFFFFFF


class MusicBlockStream:
    """Resumable shuffled-block stream over the pre-chunked train blocks.

    Emits blocks (python int lists of length seq_len) in a data_seed-derived
    permutation; on exhausting an epoch it reshuffles with a fresh per-epoch
    seed and continues. Exact resume via state()/restore(): n_emitted alone
    determines epoch and within-epoch index, so restore is O(1)."""

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
    """Standalone preprocessing: tokenize LMD-full, build the train-block array,
    the val stream and the val cache.

    Usage: python music_data.py [seq_len] [val_tokens]
    (val block count = val_tokens // seq_len, matching train_small.py)"""
    seq_len = int(sys.argv[1]) if len(sys.argv) > 1 else SEQ_LEN
    val_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 2_000_000
    n_val = val_tokens // seq_len

    os.makedirs(MUSIC_DIR, exist_ok=True)
    tok = build_tokenizer()
    V = len(tok)
    assert V < 65536, f"vocab {V} does not fit uint16"
    assert (tok["PAD_None"], tok["BOS_None"], tok["EOS_None"]) == (PAD, BOS, EOS), \
        "special-token ids moved; update PAD/BOS/EOS in music_data.py"
    tok.save_params(tokenizer_params_path())
    print(f"[music] REMI tokenizer: vocab_size={V} -> {tokenizer_params_path()}",
          flush=True)

    need_streams = not (os.path.exists(train_stream_path())
                        and os.path.exists(val_stream_path()))
    if need_streams:
        files = list_midi_files()
        print(f"[music] {len(files):,} MIDI files under {MIDI_ROOT}", flush=True)
        train_files, val_files = split_files(files)
        print(f"[music] split: {len(train_files):,} train / {len(val_files):,} val "
              f"files (SPLIT_SEED={SPLIT_SEED})", flush=True)
        val_stats = tokenize_files(val_files, val_stream_path(), tag=" val")
        train_stats = tokenize_files(train_files, train_stream_path(), tag=" train")
        meta = {
            "source_url": LMD_URL,
            "tarball_bytes": os.path.getsize(LMD_TARBALL)
                             if os.path.exists(LMD_TARBALL) else None,
            "tokenizer": "REMI",
            "tokenizer_params": {k: (list(v) if isinstance(v, tuple) else v)
                                 for k, v in TOKENIZER_PARAMS.items()
                                 if k != "beat_res"},
            "beat_res": {str(k): v for k, v in TOKENIZER_PARAMS["beat_res"].items()},
            "vocab_size": V,
            "pad_token_id": PAD, "bos_token_id": BOS, "eos_token_id": EOS,
            "split_seed": SPLIT_SEED, "val_files": VAL_FILES,
            "n_files_total": len(files),
            "train": train_stats, "val": val_stats,
            "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        tmp = meta_path() + ".tmp"
        with open(tmp, "w") as f:
            json.dump(meta, f, indent=2)
        os.replace(tmp, meta_path())
        print(f"[music] meta -> {meta_path()}", flush=True)

    n_blocks = None
    if not os.path.exists(train_blocks_path(seq_len)):
        n_blocks = _pack_blocks(train_stream_path(), seq_len,
                                train_blocks_path(seq_len))
    else:
        n_blocks = int(np.load(train_blocks_path(seq_len), mmap_mode="r").shape[0])
    v = get_or_build_music_val_set(n_val, val_cache_path(seq_len), seq_len)

    meta = load_meta()
    tot = meta["train"]["n_tokens"]
    print(f"\n[music] CORPUS: {tot:,} train tokens "
          f"({n_blocks:,} blocks of {seq_len}), val {tuple(v.shape)} = "
          f"{v.numel():,} tokens, vocab {meta['vocab_size']}", flush=True)
    for budget in (3_000_000_000,):
        print(f"[music] {budget / 1e9:.0f}B-token budget = "
              f"{budget / max(1, tot):.2f} epochs over the train corpus", flush=True)


if __name__ == "__main__":
    main()
