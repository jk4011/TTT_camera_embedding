# -*- coding: utf-8 -*-
"""Sanity checks for the LMD-full REMI symbolic-music data path (music_data.py).

Mirrors sanity_dna.py and adds the CPU training smoke test:
  [val]        val cache format / token range / file-level disjointness
  [shuffle]    same data_seed identical, different seed differs
  [resume]     state()/restore() -> next batches bit-identical (gold-test hash)
  [epoch-wrap] resume across an epoch boundary
  [train]      ~20 steps of a TINY LaCT model on CPU: loss starts near
               ln(vocab) and decreases, no NaN
  [ckpt]       full save/load round trip through train_small's checkpoint
               helpers: the batch after resume matches the uninterrupted run

CPU ONLY (no GPU is touched): run with CUDA_VISIBLE_DEVICES="".
Run: CUDA_VISIBLE_DEVICES="" ../.venv_llm/bin/python sanity_music.py [--no-train]
"""
import hashlib
import math
import os
import sys

# /tmp is noexec on this machine: inductor/triton must compile into an
# exec-allowed filesystem (same repo-local caches train_small.py uses).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_REPO_ROOT, ".cache_triton"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(_REPO_ROOT, ".cache_inductor"))
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import music_data


def _hash_blocks(blocks):
    h = hashlib.sha256()
    for b in blocks:
        h.update(torch.tensor(b, dtype=torch.int64).numpy().tobytes())
    return h.hexdigest()


class _CpuRotary(torch.nn.Module):
    """Pure-PyTorch NeoX rotary standing in for fla's triton RotaryEmbedding
    (which has no CPU backend). Same call signature; CPU smoke test only."""

    def __init__(self, dim, base=10000.0, **kw):
        super().__init__()
        self.dim, self.base = int(dim), float(base)

    def forward(self, q, k, seqlen_offset=0, max_seqlen=None, cu_seqlens=None):
        half = self.dim // 2
        s = q.shape[1]
        inv = 1.0 / (self.base ** (torch.arange(0, half, dtype=torch.float32,
                                                device=q.device) / half))
        t = torch.arange(s, dtype=torch.float32, device=q.device) + int(seqlen_offset)
        f = torch.outer(t, inv)
        cos, sin = f.cos()[None, :, None, :], f.sin()[None, :, None, :]

        def rot(x):
            x1, x2 = x[..., :half].float(), x[..., half:2 * half].float()
            out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos,
                             x[..., 2 * half:].float()], dim=-1)
            return out.to(x.dtype)
        return rot(q), rot(k)


def _cpu_flash_attn(q, k, v, causal=True, window_size=(-1, -1), **kw):
    """Sliding-window causal attention via SDPA, standing in for flash_attn_func
    (CUDA-only). q/k/v: [b, s, h, d]. CPU smoke test only."""
    import torch.nn.functional as F
    s = q.shape[1]
    i = torch.arange(s, device=q.device)
    mask = i[:, None] >= i[None, :] if causal else torch.ones(s, s, dtype=torch.bool,
                                                              device=q.device)
    if window_size[0] is not None and window_size[0] >= 0:
        mask = mask & ((i[:, None] - i[None, :]) <= window_size[0])
    o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                       v.transpose(1, 2),
                                       attn_mask=mask[None, None])
    return o.transpose(1, 2)


def patch_for_cpu():
    """Swap the two CUDA-only kernels (fla triton rotary, flash-attn) for
    pure-PyTorch equivalents so the model can run on CPU. Only ever called from
    this sanity script — train_small.py is untouched."""
    from lact_model import layer_lact_swiglu as L
    L.RotaryEmbedding = _CpuRotary
    L.flash_attn_func = _cpu_flash_attn
    L.RMSNorm = torch.nn.RMSNorm      # fla's RMSNorm is triton-only
    import torch.nn.functional as F
    import fla.modules.mlp as FM      # fla's fused SwiGLU MLP is triton-only
    FM.SwiGLULinear.forward = (
        lambda self, x, y, weight, bias: F.linear(F.silu(x) * y, weight, bias))


def _tiny_model(vocab_size, seq_len):
    """Smallest LaCT config that still exercises the real TTT path."""
    import json
    patch_for_cpu()
    from lact_model import LaCTForCausalLM, LaCTSWIGLUConfig
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "configs/760M_lact_swiglu_nh4_fwlow_rank_momentum_muon.json")) as f:
        cfg = json.load(f)
    cfg.pop("model_type", None)
    cfg.update(dict(hidden_size=64, num_hidden_layers=2, num_attn_heads=2,
                    num_lact_heads=1, lact_chunk_size=256, window_size=128,
                    max_position_embeddings=seq_len, vocab_size=vocab_size,
                    use_fused_kernel=False, w0_w2_low_rank=-1,
                    fuse_cross_entropy=False, fuse_norm=False,
                    last_layer_fuse_norm=False,
                    bos_token_id=music_data.BOS, eos_token_id=music_data.EOS))
    return LaCTForCausalLM(LaCTSWIGLUConfig(**cfg))


def train_smoke(seq_len, n_steps=20, bs=2, sub_len=512):
    """Tiny CPU training run on the music stream. Returns (ok, losses).

    sub_len: each 4096-block is truncated to sub_len tokens so 20 CPU steps
    finish in ~a minute; the DATA PATH (stream, packing, ids) is the real one."""
    import data_utils
    torch.manual_seed(0)
    V = music_data.vocab_size()
    model = _tiny_model(V, sub_len)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    stream = music_data.MusicBlockStream(
        music_data.train_blocks_path(seq_len), 42, seq_len)
    batches = data_utils.batch_generator(stream, bs, prefetch=2, with_state=True)
    losses = []
    for step in range(n_steps):
        x, _ = next(batches)
        x = x[:, :sub_len].contiguous()
        loss = model(input_ids=x, labels=x).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(float(loss))
        if step % 5 == 0 or step == n_steps - 1:
            print(f"  [train] step={step + 1} loss={losses[-1]:.4f}", flush=True)
    ln_v = math.log(V)
    finite = all(math.isfinite(l) for l in losses)
    near_lnv = abs(losses[0] - ln_v) < 1.0
    decreased = (sum(losses[-3:]) / 3) < (sum(losses[:3]) / 3) - 0.05
    print(f"  [train] ln(vocab)={ln_v:.3f} first={losses[0]:.4f} "
          f"last={losses[-1]:.4f} finite={finite} near_lnv={near_lnv} "
          f"decreasing={decreased}", flush=True)
    return (finite and near_lnv and decreased), losses


def ckpt_roundtrip(seq_len, sub_len=512, bs=2, n_pre=6, n_post=3):
    """End-to-end resume test through train_small's checkpoint helpers:
    run n_pre steps, snapshot (model+opt+stream), keep going -> hash the next
    n_post batches; then rebuild from the checkpoint and hash again."""
    import argparse
    import data_utils
    import train_small

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "outputs", "_sanity_music")
    os.makedirs(out_dir, exist_ok=True)
    V = music_data.vocab_size()

    def fresh():
        torch.manual_seed(0)
        model = _tiny_model(V, sub_len)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sched = train_small.build_scheduler(opt, 2, 100, 0.0)
        stream = music_data.MusicBlockStream(
            music_data.train_blocks_path(seq_len), 42, seq_len)
        return model, opt, sched, stream

    model, opt, sched, stream = fresh()
    batches = data_utils.batch_generator(stream, bs, prefetch=1, with_state=True)
    state = None
    for _ in range(n_pre):
        x, state = next(batches)
        loss = model(input_ids=x[:, :sub_len].contiguous(),
                     labels=x[:, :sub_len].contiguous()).loss
        loss.backward()
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
    # continue uninterrupted -> reference hash of the next n_post batches
    ref = [next(batches)[0] for _ in range(n_post)]
    h_ref = _hash_blocks([b.tolist() for b in ref])

    # write a real checkpoint, then resume from it into a fresh process state
    args = argparse.Namespace(out_dir=out_dir, keep_ckpts=1)
    for k, v in dict(data="music", data_seed=42, seq_len=seq_len, bs=bs,
                     grad_accum=1, val_tokens=2_000_000, lr=1e-3, warmup=2,
                     min_lr_ratio=0.0, steps=100, token_budget=0,
                     synthetic="none", input_rope_dropout_p0=0.0,
                     input_rope_dropout_anneal=30000,
                     input_rope_warmup="none").items():
        setattr(args, k, v)
    ckpt = {"step": n_pre, "tokens_seen": 0, "model": model.state_dict(),
            "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
            "stream": state,
            "args": {k: getattr(args, k) for k in train_small._RESUME_CRITICAL_ARGS}}
    path = os.path.join(out_dir, "ckpt_roundtrip.pt")
    torch.save(ckpt, path)

    model2, opt2, sched2, stream2 = fresh()
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    for k in train_small._RESUME_CRITICAL_ARGS:
        assert loaded["args"].get(k, train_small._RESUME_ARG_DEFAULTS.get(k)) == \
            getattr(args, k), f"resume-critical arg {k} mismatch"
    model2.load_state_dict(loaded["model"])
    opt2.load_state_dict(loaded["optimizer"])
    sched2.load_state_dict(loaded["scheduler"])
    stream2.restore(loaded["stream"])
    batches2 = data_utils.batch_generator(stream2, bs, prefetch=1, with_state=True)
    got = [next(batches2)[0] for _ in range(n_post)]
    h_got = _hash_blocks([b.tolist() for b in got])

    same_weights = all(torch.equal(a, b) for a, b in
                       zip(model.state_dict().values(), model2.state_dict().values()))
    os.remove(path)
    print(f"  [ckpt] stream pos={loaded['stream']['n_raw_consumed']} "
          f"batch_hash_match={h_ref == h_got} weights_match={same_weights}", flush=True)
    return (h_ref == h_got) and same_weights


def main():
    S = music_data.SEQ_LEN
    do_train = "--no-train" not in sys.argv
    n_val = 2_000_000 // S
    print(f"=== sanity_music seq_len={S} (val blocks={n_val}) ===")
    path = music_data.ensure_train_blocks(S)
    meta = music_data.load_meta()
    V = int(meta["vocab_size"])
    ok = True

    # --- val set format ---
    val = music_data.get_or_build_music_val_set(n_val, music_data.val_cache_path(S), S)
    vmin, vmax = int(val.min()), int(val.max())
    val_ok = (tuple(val.shape) == (n_val, S) and val.dtype == torch.int64
              and vmin >= 0 and vmax < V)
    print(f"[val] shape={tuple(val.shape)} dtype={val.dtype} min={vmin} max={vmax} "
          f"vocab={V} -> {'OK' if val_ok else 'FAIL'}")
    ok &= val_ok

    # --- train/val file disjointness (the split is by FILE, not by block) ---
    tr, va = music_data.split_files(sorted(set(
        music_data.list_midi_files()))) if os.path.isdir(music_data.MIDI_ROOT) \
        else (None, None)
    if tr is not None:
        disj = len(set(tr) & set(va)) == 0 and len(va) == music_data.VAL_FILES
        print(f"[split] train={len(tr):,} val={len(va):,} overlap=0 -> "
              f"{'OK' if disj else 'FAIL'}")
        ok &= disj
    else:
        print(f"[split] SKIP (MIDI tree {music_data.MIDI_ROOT} not mounted; "
              f"split recorded in meta: {meta['val_files']} val files)")

    # --- shuffle determinism: same seed identical, different seed differs ---
    s_a = music_data.MusicBlockStream(path, 42, S)
    s_b = music_data.MusicBlockStream(path, 42, S)
    s_c = music_data.MusicBlockStream(path, 43, S)
    h_a = _hash_blocks([next(s_a) for _ in range(20)])
    h_b = _hash_blocks([next(s_b) for _ in range(20)])
    h_c = _hash_blocks([next(s_c) for _ in range(20)])
    det_ok = (h_a == h_b) and (h_a != h_c)
    print(f"[shuffle] seed42==seed42:{h_a == h_b} seed42!=seed43:{h_a != h_c} -> "
          f"{'OK' if det_ok else 'FAIL'}")
    ok &= det_ok

    # --- resume exactness (gold-test hash compare) ---
    s1 = music_data.MusicBlockStream(path, 42, S)
    for _ in range(137):
        next(s1)
    st = s1.state()
    h_cont = _hash_blocks([next(s1) for _ in range(11)])
    s2 = music_data.MusicBlockStream(path, 42, S)
    s2.restore(st)
    h_cont2 = _hash_blocks([next(s2) for _ in range(11)])
    resume_ok = (h_cont == h_cont2) and (int(st["n_raw_consumed"]) == 137)
    print(f"[resume] n_raw_consumed={int(st['n_raw_consumed'])} "
          f"cont_hash_match={h_cont == h_cont2} -> {'OK' if resume_ok else 'FAIL'}")
    ok &= resume_ok

    # --- epoch wrap resume ---
    Ntot = s1.N
    s3 = music_data.MusicBlockStream(path, 42, S)
    s3.restore({"n_raw_consumed": Ntot - 3, "buf": torch.empty(0, dtype=torch.int64)})
    h_wrap = _hash_blocks([next(s3) for _ in range(6)])
    s4 = music_data.MusicBlockStream(path, 42, S)
    s4.restore({"n_raw_consumed": Ntot - 3, "buf": torch.empty(0, dtype=torch.int64)})
    h_wrap2 = _hash_blocks([next(s4) for _ in range(6)])
    wrap_ok = h_wrap == h_wrap2
    print(f"[epoch-wrap] N={Ntot:,} cross-boundary resume match={wrap_ok} -> "
          f"{'OK' if wrap_ok else 'FAIL'}")
    ok &= wrap_ok

    # --- token budget report ---
    tot = int(meta["train"]["n_tokens"])
    print(f"[corpus] {tot:,} train tokens, {Ntot:,} blocks x {S} = "
          f"{Ntot * S:,} packed tokens; 3B budget = {3e9 / tot:.2f} epochs")

    if do_train:
        assert not torch.cuda.is_available() or os.environ.get("CUDA_VISIBLE_DEVICES") == "", \
            "run with CUDA_VISIBLE_DEVICES='' (this node's GPUs are busy)"
        print("[train] tiny CPU training smoke test (20 steps)...", flush=True)
        train_ok, _ = train_smoke(S)
        print(f"[train] -> {'OK' if train_ok else 'FAIL'}")
        ok &= train_ok
        print("[ckpt] checkpoint save/resume round trip...", flush=True)
        ck_ok = ckpt_roundtrip(S)
        print(f"[ckpt] -> {'OK' if ck_ok else 'FAIL'}")
        ok &= ck_ok

    print(f"\nSANITY_MUSIC: {'ALL PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
