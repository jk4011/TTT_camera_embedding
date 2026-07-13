# -*- coding: utf-8 -*-
"""Data utilities for train_small.py: tokenizer loading with fallbacks,
streaming fineweb-edu with token-buffer packing, fixed validation split,
and a background-thread prefetching batch generator.

HF_HOME is expected to be set by the caller (train_small.py sets it to
/tmp/hf_cache before importing this module).
"""

import os
import queue
import threading
import time

import torch


TOKENIZER_CANDIDATES = [
    "fla-hub/transformer-1.3B-100B",
    "mistralai/Mistral-7B-v0.1",
    "gpt2",
]


def load_tokenizer(preferred=None):
    """Try tokenizers in order, return (tokenizer, name_used, vocab_size)."""
    from transformers import AutoTokenizer

    candidates = [preferred] + TOKENIZER_CANDIDATES if preferred else list(TOKENIZER_CANDIDATES)
    last_err = None
    for name in candidates:
        try:
            tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
            vocab_size = max(len(tok), getattr(tok, "vocab_size", 0) or 0)
            print(f"[data] tokenizer loaded: {name} (vocab_size={vocab_size})", flush=True)
            return tok, name, vocab_size
        except Exception as e:  # gated / network / missing
            print(f"[data] tokenizer {name} failed: {type(e).__name__}: {e}", flush=True)
            last_err = e
    raise RuntimeError(f"All tokenizer candidates failed; last error: {last_err}")


def build_shuffled_stream(data_seed, buffer_size=10000):
    """fineweb-edu sample-10BT streaming split, deterministically shuffled."""
    from datasets import load_dataset

    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    return ds.shuffle(buffer_size=buffer_size, seed=data_seed)


def packed_block_generator(ds, tokenizer, seq_len, eos_id, text_batch=64):
    """Tokenize the "text" field, append eos per document, and pack into
    contiguous blocks of exactly seq_len tokens (no padding).

    Yields python lists of length seq_len (int token ids)."""
    buf = []
    texts = []
    for ex in ds:
        texts.append(ex["text"])
        if len(texts) < text_batch:
            continue
        encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
        texts = []
        for ids in encoded:
            buf.extend(ids)
            buf.append(eos_id)
        while len(buf) >= seq_len:
            yield buf[:seq_len]
            del buf[:seq_len]
    # flush remaining texts at stream end
    if texts:
        encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
        for ids in encoded:
            buf.extend(ids)
            buf.append(eos_id)
        while len(buf) >= seq_len:
            yield buf[:seq_len]
            del buf[:seq_len]


class PackedBlockStream:
    """Stateful, resumable version of packed_block_generator.

    Yields the bit-identical block sequence (packing into fixed seq_len blocks
    is a pure function of the concatenated token stream, and batched
    tokenization of independent documents is grouping-invariant), but tracks
    the exact stream position so training can checkpoint/resume:

      state = (n_raw_consumed, buf)

    where n_raw_consumed is the number of raw examples pulled from the
    (deterministically shuffled) dataset iterator and buf is the leftover
    token carry-over that has been tokenized but not yet emitted as a block.
    Texts are tokenized immediately on pull, so there is never a pending
    untokenized text between __next__ calls — (n_raw_consumed, buf) is the
    complete packer state at any block boundary.
    """

    def __init__(self, ds, tokenizer, seq_len, eos_id, text_batch=64):
        self.ds_iter = iter(ds)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.eos_id = eos_id
        self.text_batch = text_batch
        self.n_raw_consumed = 0
        self.buf = []
        self._exhausted = False

    def __iter__(self):
        return self

    def _refill(self):
        texts = []
        while len(texts) < self.text_batch and not self._exhausted:
            try:
                texts.append(next(self.ds_iter)["text"])
                self.n_raw_consumed += 1
            except StopIteration:
                self._exhausted = True
        if texts:
            encoded = self.tokenizer(texts, add_special_tokens=False)["input_ids"]
            for ids in encoded:
                self.buf.extend(ids)
                self.buf.append(self.eos_id)

    def __next__(self):
        while len(self.buf) < self.seq_len:
            if self._exhausted:
                raise StopIteration  # drop final partial block (matches generator)
            self._refill()
        block = self.buf[:self.seq_len]
        del self.buf[:self.seq_len]
        return block

    def state(self):
        """Snapshot of the exact stream position (call at a block/batch boundary)."""
        return {
            "n_raw_consumed": self.n_raw_consumed,
            "buf": torch.tensor(self.buf, dtype=torch.int64),
        }

    def restore(self, state, log_every=100_000):
        """Fast-forward a FRESH stream (same data_seed) to a saved position:
        skip the consumed raw examples, then restore the token carry-over."""
        assert self.n_raw_consumed == 0 and not self.buf, \
            "restore() must be called on a freshly constructed PackedBlockStream"
        n = int(state["n_raw_consumed"])
        t0 = time.time()
        for i in range(n):
            try:
                next(self.ds_iter)
            except StopIteration:
                raise RuntimeError(f"stream exhausted at {i}/{n} while fast-forwarding")
            if log_every and (i + 1) % log_every == 0:
                dt = time.time() - t0
                print(f"[data] fast-forward {i + 1:,}/{n:,} raw examples "
                      f"({(i + 1) / dt:,.0f} ex/s)", flush=True)
        self.n_raw_consumed = n
        buf = state["buf"]
        self.buf = buf.tolist() if torch.is_tensor(buf) else list(buf)
        print(f"[data] stream fast-forward done: skipped {n:,} raw examples in "
              f"{time.time() - t0:.1f}s, restored {len(self.buf)} carry-over tokens",
              flush=True)


def get_or_build_val_set(block_gen, n_val_blocks, cache_path):
    """Pull the FIRST n_val_blocks from block_gen as the fixed validation set.

    IMPORTANT: the first n_val_blocks are always consumed from block_gen even
    when a cache file exists, so the training stream position (and therefore
    the training data order) is identical across all runs."""
    pulled = []
    for _ in range(n_val_blocks):
        pulled.append(next(block_gen))
    pulled = torch.tensor(pulled, dtype=torch.int64)

    if os.path.exists(cache_path):
        val = torch.load(cache_path, map_location="cpu")
        if val.shape != pulled.shape or not torch.equal(val[0], pulled[0]):
            print(f"[data] WARNING: cached val set at {cache_path} does not match the "
                  f"current stream (shape {tuple(val.shape)} vs {tuple(pulled.shape)}); "
                  f"overwriting with freshly pulled blocks.", flush=True)
            val = pulled
            torch.save(val, cache_path)
        else:
            print(f"[data] reusing cached val set {cache_path} ({val.numel()} tokens)", flush=True)
    else:
        val = pulled
        torch.save(val, cache_path)
        print(f"[data] saved val set to {cache_path} ({val.numel()} tokens)", flush=True)
    return val


def batch_generator(block_gen, batch_size, prefetch=4, with_state=False):
    """Group packed blocks into [batch_size, seq_len] int64 tensors, produced
    by a background thread for prefetching.

    with_state=True (requires block_gen to be a PackedBlockStream): yield
    (tensor, state) tuples where state is the packer position snapshotted in
    the worker thread right after the batch was assembled — i.e. exactly the
    position to resume from so the NEXT batch after this one is reproduced.
    The snapshot happens in the worker before it pulls further blocks, so it
    is race-free despite prefetching."""
    q = queue.Queue(maxsize=prefetch)
    _SENTINEL = object()

    def _worker():
        try:
            batch = []
            for block in block_gen:
                batch.append(block)
                if len(batch) == batch_size:
                    t = torch.tensor(batch, dtype=torch.int64)
                    q.put((t, block_gen.state()) if with_state else t)
                    batch = []
        except Exception as e:  # propagate errors to consumer
            q.put(e)
            return
        q.put(_SENTINEL)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

    while True:
        item = q.get()
        if item is _SENTINEL:
            return
        if isinstance(item, Exception):
            raise item
        yield item
