# -*- coding: utf-8 -*-
"""Q21 LieRE-on-hidden-rope sanity suite (GPU-6 dev).

Modes:
  s5ref   : compute + save the FIXED honly path reference loss (run BEFORE the edit)
  s5check : recompute the fixed-path loss and compare bitwise vs the saved reference
  s1s2    : rope-init reduction checks (b=2 and b=4) vs the fixed cos/sin path
  s3s4    : grad-flow + orthogonality-after-updates checks
  bench   : fwd+bwd sec/step, fixed vs liere-b2 vs liere-b8

Production 200M arch throughout: 768/12L/12ah/4fwh, chunk 1024, window 128,
seq 4096, honly (ttt_nope + ttt_hidden_rope, gain 1.0).
"""

import argparse
import json
import math
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
_REPO_ROOT = os.path.dirname(SCRIPT_DIR)
os.environ.setdefault("HF_HOME", "/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/datasets/hf_cache")
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_REPO_ROOT, ".cache_triton"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(_REPO_ROOT, ".cache_inductor"))
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import torch  # noqa: E402

from lact_model import LaCTForCausalLM, LaCTSWIGLUConfig  # noqa: E402

BASE_JSON = os.path.join(SCRIPT_DIR, "configs/760M_lact_swiglu_nh4_fwlow_rank_momentum_muon.json")
VAL_CACHE = os.path.join(SCRIPT_DIR, "val_cache_fla-hub_transformer-1.3B-100B_4096_ds42.pt")
REF_PATH = os.path.join(SCRIPT_DIR, "outputs/q21_liere_dev/s5_ref.pt")

HONLY = dict(ttt_nope=True, ttt_hidden_rope=True, ttt_hrope_gain=1.0)


def build_config(**extra):
    with open(BASE_JSON) as f:
        cfg = json.load(f)
    cfg.pop("model_type", None)
    cfg.update(dict(
        hidden_size=768, num_hidden_layers=12, num_attn_heads=12,
        num_lact_heads=4, lact_chunk_size=1024, window_size=128,
        max_position_embeddings=4096, vocab_size=32000,
        use_fused_kernel=False, bos_token_id=1, eos_token_id=2,
    ))
    cfg.update(HONLY)
    cfg.update(extra)
    return LaCTSWIGLUConfig(**cfg)


def build_model(seed=42, **extra):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = LaCTForCausalLM(build_config(**extra)).to("cuda")
    return model


def get_batch(bs=2):
    val = torch.load(VAL_CACHE, map_location="cpu")
    return val[:bs].to("cuda")


def fwd_loss(model, x, train=True):
    model.train(train)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(input_ids=x, labels=x)
    return out.loss


# ---------------------------------------------------------------- s5


def s5ref(args):
    x = get_batch(args.bs)
    model = build_model(seed=42)
    with torch.no_grad():
        l1 = fwd_loss(model, x).float().item()
        l2 = fwd_loss(model, x).float().item()  # in-process determinism probe
    os.makedirs(os.path.dirname(REF_PATH), exist_ok=True)
    torch.save({"loss1": l1, "loss2": l2, "bs": args.bs}, REF_PATH)
    print(f"[s5ref] fixed honly loss = {l1!r} (repeat {l2!r}, "
          f"deterministic={'YES' if l1 == l2 else 'NO'}) -> {REF_PATH}")


def s5check(args):
    ref = torch.load(REF_PATH)
    x = get_batch(ref["bs"])
    model = build_model(seed=42)
    with torch.no_grad():
        l1 = fwd_loss(model, x).float().item()
    same = l1 == ref["loss1"]
    print(f"[s5check] ref={ref['loss1']!r} new={l1!r} bit_identical={'PASS' if same else 'FAIL'}"
          f" (abs diff {abs(l1 - ref['loss1']):.3e})")


# ---------------------------------------------------------------- s1/s2


def rot_ref_from_ladder(layer, pos, fp64=False):
    """Reference per-position rotation blocks built directly from cos/sin.

    fp64=False mirrors the production fixed path (fp32 angle product + fp32
    cos/sin); fp64=True is the mathematically exact rotation for the ladder."""
    b = layer.ttt_liere
    inv = layer.h_inv_freq.cuda()
    inv = inv.double() if fp64 else inv.float()
    ang = pos.to(inv.dtype)[:, None] * inv[None]  # [s, P_h]
    c, s_ = ang.cos().float(), ang.sin().float()
    P_h = inv.shape[0]
    nb = (2 * P_h) // b
    R = torch.zeros(pos.shape[0], nb, b, b, device="cuda")
    for j in range(nb):
        for m in range(b // 2):
            p = j * (b // 2) + m
            R[:, j, 2 * m, 2 * m] = c[:, p]
            R[:, j, 2 * m, 2 * m + 1] = -s_[:, p]
            R[:, j, 2 * m + 1, 2 * m] = s_[:, p]
            R[:, j, 2 * m + 1, 2 * m + 1] = c[:, p]
    return R


def liere_R(layer, pos):
    """R for integer positions 0..len(pos)-1 -> gathered at pos (uses the
    layer's own production code path)."""
    n = int(pos.max().item()) + 1
    R_all = layer.liere_rotations(n, 0, device=pos.device)
    return R_all[pos.long()]


def s1s2(args):
    x = get_batch(args.bs)
    with torch.no_grad():
        m_fix = build_model(seed=42)
        l_fix = fwd_loss(m_fix, x).float().item()
        del m_fix
        torch.cuda.empty_cache()
        for b in (2, 4):
            m = build_model(seed=42, ttt_liere=b, ttt_liere_init="rope")
            # matrix agreement over all positions, layer 0
            layer = m.model.layers[0].attn
            pos = torch.arange(4096, device="cuda", dtype=torch.float32)
            R = liere_R(layer, pos)
            mdiff = (R - rot_ref_from_ladder(layer, pos)).abs().max().item()
            mdiff64 = (R - rot_ref_from_ladder(layer, pos, fp64=True)).abs().max().item()
            l_lie = fwd_loss(m, x).float().item()
            print(f"[s{1 if b == 2 else 2}] b={b} rope-init: loss fixed={l_fix!r} "
                  f"liere={l_lie!r} |dloss|={abs(l_lie - l_fix):.3e} "
                  f"max|R - R_fp32ladder|={mdiff:.3e} max|R - R_exact|={mdiff64:.3e}")
            del m
            torch.cuda.empty_cache()


# ---------------------------------------------------------------- s3/s4


def s3s4(args):
    x = get_batch(args.bs)
    m = build_model(seed=42, ttt_liere=args.b, ttt_liere_init=args.init)
    opt = torch.optim.AdamW([p for p in m.parameters()], lr=3e-4, weight_decay=0.1)
    for step in range(3):
        opt.zero_grad(set_to_none=True)
        loss = fwd_loss(m, x)
        loss.backward()
        if step == 0:
            ok_all, per_layer = True, []
            for i, lyr in enumerate(m.model.layers):
                g = lyr.attn.liere_delta.grad
                finite = g is not None and torch.isfinite(g).all().item()
                nrm = g.norm().item() if g is not None else float("nan")
                blk = g.flatten(1).norm(dim=1) if g is not None else None
                distinct = blk is not None and (blk.std() / (blk.mean() + 1e-12)).item() > 1e-3
                ok_all &= finite and nrm > 0 and distinct
                per_layer.append((i, finite, nrm, distinct))
            print("[s3] liere_delta grads (layer, finite, norm, per-block distinct):")
            for row in per_layer:
                print(f"     L{row[0]:02d} finite={row[1]} norm={row[2]:.3e} distinct={row[3]}")
            print(f"[s3] {'PASS' if ok_all else 'FAIL'}")
        opt.step()
    # S4: orthogonality after 3 real AdamW steps
    worst = 0.0
    pos = torch.randint(0, 4096, (256,), device="cuda").float()
    for lyr in m.model.layers:
        # measure the fp32 R actually handed to the kernel, but accumulate the
        # R^T R check in fp64 (a fp32/TF32-matmul check measures the CHECK's
        # rounding, not R's orthogonality)
        R = liere_R(lyr.attn, pos).double()
        eye = torch.eye(R.shape[-1], device="cuda", dtype=torch.float64).expand_as(R)
        worst = max(worst, (R.transpose(-1, -2) @ R - eye).abs().max().item())
    print(f"[s4] max |R^T R - I| over 256 random t x 12 layers (after 3 AdamW steps) "
          f"= {worst:.3e} -> {'PASS' if worst < 1e-5 else 'FAIL'} (target < 1e-5)")


# ---------------------------------------------------------------- bench


def bench_one(model, x, n_warm=8, n_meas=20):
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for _ in range(n_warm):
        opt.zero_grad(set_to_none=True)
        fwd_loss(model, x).backward()
        opt.step()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_meas):
        opt.zero_grad(set_to_none=True)
        fwd_loss(model, x).backward()
        opt.step()
    torch.cuda.synchronize()
    return (time.time() - t0) / n_meas


def bench(args):
    x = get_batch(8)
    rows = []
    for name, extra in [("fixed", {}),
                        ("liere_b2", dict(ttt_liere=2, ttt_liere_init="rope")),
                        ("liere_b8", dict(ttt_liere=8, ttt_liere_init="rope"))]:
        m = build_model(seed=42, **extra)
        spb = bench_one(m, x)
        rows.append((name, spb))
        print(f"[bench] {name}: {spb * 1000:.1f} ms/step (bs8 seq4096 fwd+bwd+opt)")
        del m
        torch.cuda.empty_cache()
    base = rows[0][1]
    for name, spb in rows[1:]:
        print(f"[bench] {name} overhead vs fixed: {100 * (spb - base) / base:+.1f}%")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["s5ref", "s5check", "s1s2", "s3s4", "bench"])
    p.add_argument("--bs", type=int, default=2)
    p.add_argument("--b", type=int, default=2)
    p.add_argument("--init", type=str, default="rope")
    args = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    globals()[args.mode](args)
