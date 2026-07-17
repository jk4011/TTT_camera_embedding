# -*- coding: utf-8 -*-
"""Q23 VaPE (value-path rotary) sanity suite (GPU-2 dev).

Modes:
  a : zero-phase equivalence — ttt_vrope_gain=0.0 must reproduce the baseline
      (no-vrope) logits/loss BIT-EXACTLY (same seed + state_dict copy), for
      both the delta_only (default) and full apply variants.
  b : grad check — backward runs, grads finite, a few AdamW steps on a fixed
      batch decrease the loss (gain=1.0 delta_only).
  c : relative-phase check — (c-mini) fp64 standalone reproduction of the
      design identity: with w0/w2 frozen and no weight-norm, the counter-
      rotated DELTA readout sum_t lr_t <h_t,h_s> R_{t-s} v_t is EXACTLY
      invariant to a constant shift of all positions (relative code), while
      the full-variant readout is not; (c-kernel) the actual kernel with
      lr0=lr2=0: delta_only output shift-violation is small (only the
      weight-norm row-coupling survives) vs the full variant's O(1)
      absolute-phase violation.

Production 200M arch for a/b: 768/12L/12ah/4fwh, chunk 1024, window 128,
seq 4096, input rope ON (ttt_nope=False) — the target rope+VaPE config.
"""

import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
_REPO_ROOT = os.path.dirname(SCRIPT_DIR)
os.environ.setdefault("HF_HOME", "/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/datasets/hf_cache")
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_REPO_ROOT, ".cache_triton"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(_REPO_ROOT, ".cache_inductor"))
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from lact_model import LaCTForCausalLM, LaCTSWIGLUConfig  # noqa: E402
from lact_model.ttt_operation import (  # noqa: E402
    prenorm_block_causal_lact_swiglu_value_rope,
)

BASE_JSON = os.path.join(SCRIPT_DIR, "configs/760M_lact_swiglu_nh4_fwlow_rank_momentum_muon.json")
VAL_CACHE = os.path.join(SCRIPT_DIR, "val_cache_fla-hub_transformer-1.3B-100B_4096_ds42.pt")


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
    cfg.update(extra)
    return LaCTSWIGLUConfig(**cfg)


def build_model(seed=42, **extra):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return LaCTForCausalLM(build_config(**extra)).to("cuda")


def get_batch(bs=2):
    return torch.load(VAL_CACHE, map_location="cpu")[:bs].to("cuda")


def fwd(model, x, train=True):
    model.train(train)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model(input_ids=x, labels=x)


# ---------------------------------------------------------------- a


def _loss_and_logits(model, x):
    """The fused linear+CE path (train mode) never materializes logits ->
    loss in train mode (the training code path), logits in eval mode."""
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model.train(True)
        loss = model(input_ids=x, labels=x).loss.float().item()
        model.eval()
        logits = model(input_ids=x).logits.float()
    return loss, logits


def a(args):
    x = get_batch(args.bs)
    base = build_model(seed=42)
    loss_b, logits_b = _loss_and_logits(base, x)
    sd = base.state_dict()
    del base
    torch.cuda.empty_cache()
    ok_all = True
    for name, extra in [
        ("delta_only", dict(ttt_value_rope=True, ttt_vrope_gain=0.0)),
        ("full", dict(ttt_value_rope=True, ttt_vrope_gain=0.0,
                      ttt_vrope_delta_only=False)),
    ]:
        m = build_model(seed=42, **extra)
        missing, unexpected = m.load_state_dict(sd, strict=False)
        assert not missing and not unexpected, (missing, unexpected)
        loss_v, logits_v = _loss_and_logits(m, x)
        same_loss = loss_v == loss_b
        max_logit_diff = (logits_v - logits_b).abs().max().item()
        same_logits = max_logit_diff == 0.0
        ok = same_loss and same_logits
        ok_all &= ok
        print(f"[a:{name}] gain=0.0: loss base={loss_b!r} vape={loss_v!r} "
              f"bit_identical_loss={same_loss} max|dlogit|={max_logit_diff:.3e} "
              f"-> {'PASS' if ok else 'FAIL'}")
        del m
        torch.cuda.empty_cache()
    print(f"[a] {'PASS' if ok_all else 'FAIL'}")


# ---------------------------------------------------------------- b


def b(args):
    x = get_batch(args.bs)
    m = build_model(seed=42, ttt_value_rope=True)  # gain 1.0, delta_only default
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.1)
    losses = []
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        loss = fwd(m, x).loss
        loss.backward()
        if step == 0:
            n_grads, n_bad = 0, 0
            for name, p in m.named_parameters():
                if p.grad is None:
                    continue
                n_grads += 1
                if not torch.isfinite(p.grad).all().item():
                    n_bad += 1
                    print(f"[b] NON-FINITE grad: {name}")
            print(f"[b] step0 grads: {n_grads} params with grads, "
                  f"{n_bad} non-finite -> {'PASS' if n_bad == 0 else 'FAIL'}")
        opt.step()
        losses.append(loss.float().item())
    dec = losses[-1] < losses[0]
    print(f"[b] losses over {args.steps} steps on a fixed batch: "
          + " ".join(f"{v:.4f}" for v in losses))
    print(f"[b] loss decreases ({losses[0]:.4f} -> {losses[-1]:.4f}) "
          f"-> {'PASS' if dec else 'FAIL'}")


# ---------------------------------------------------------------- c


def _rot(x_cols, cos, sin):
    """fp64 reference rotation of the first 2P rows of x [d, l] (NeoX pairs,
    same pairing as apply_rotary_cols)."""
    P = cos.shape[0]
    x = x_cols.clone()
    x1 = x_cols[0:2 * P:2, :]
    x2 = x_cols[1:2 * P + 1:2, :]
    x[0:2 * P:2, :] = x1 * cos - x2 * sin
    x[1:2 * P + 1:2, :] = x1 * sin + x2 * cos
    return x


def c_mini(shift=937.0):
    """fp64 design-identity check: single update chunk + later-chunk apply,
    NO weight-norm, w0/w2 frozen. Delta readout must be exactly shift-
    invariant; the full readout (init part counter-rotated too) must not."""
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    d_in, d_h, d_out, C = 16, 32, 16, 64
    P_v = d_out // 4  # frac 0.5
    w0 = torch.randn(d_h, d_in, generator=g, dtype=torch.float64) / d_in ** 0.5
    w2 = torch.randn(d_h, d_in, generator=g, dtype=torch.float64) / d_in ** 0.5
    w1 = torch.randn(d_out, d_h, generator=g, dtype=torch.float64) / d_h ** 0.5
    k = torch.randn(d_in, C, generator=g, dtype=torch.float64)
    v = torch.randn(d_out, C, generator=g, dtype=torch.float64)
    q = torch.randn(d_in, C, generator=g, dtype=torch.float64)
    lr1 = 1e-3 * torch.rand(1, C, generator=g, dtype=torch.float64)
    inv = 1.0 / (500000.0 ** (torch.arange(P_v, dtype=torch.float64) / P_v))

    def readout(pos0):
        pos_k = torch.arange(C, dtype=torch.float64) + pos0        # update chunk
        pos_q = torch.arange(C, dtype=torch.float64) + pos0 + C    # apply chunk
        ck, sk = (inv[:, None] * pos_k).cos(), (inv[:, None] * pos_k).sin()
        cq, sq = (inv[:, None] * pos_q).cos(), (inv[:, None] * pos_q).sin()
        hidden = F.silu(w0 @ k) * (w2 @ k)                          # [d_h, C]
        v_rot = _rot(v, ck, sk)                                     # R_t v_t
        dw1 = v_rot @ (hidden * lr1).T                              # [d_out, d_h]
        hq = F.silu(w0 @ q) * (w2 @ q)
        delta = _rot(dw1 @ hq, cq, -sq)                             # R_s^{-1} dW1 h_s
        full = _rot((w1 + dw1) @ hq, cq, -sq)                       # full variant
        return delta, full

    d0, f0 = readout(0.0)
    d1, f1 = readout(shift)
    rel_d = (d1 - d0).abs().max().item() / d0.abs().max().item()
    rel_f = (f1 - f0).abs().max().item() / f0.abs().max().item()
    ok = rel_d < 1e-10 and rel_f > 1e-2
    print(f"[c-mini] fp64, shift={shift}: delta-readout rel shift-violation "
          f"= {rel_d:.3e} (target < 1e-10); full-readout = {rel_f:.3e} "
          f"(expected O(1)) -> {'PASS' if ok else 'FAIL'}")
    return ok


def c_kernel(shift=937.0):
    """Actual kernel with lr0=lr2=0 (w0/w2 exactly frozen): the delta_only
    output's shift-violation (weight-norm row-coupling only) must be small
    relative to the delta magnitude; the full variant's must be O(1)."""
    torch.manual_seed(0)
    bh, d, d_h, seq, chunk = 2, 128, 512, 768, 256
    P_v = d // 4
    inv = 1.0 / (500000.0 ** (torch.arange(P_v, dtype=torch.float32) / P_v)).cuda()
    w0 = (torch.randn(bh, d_h, d) / d ** 0.5).cuda()
    w2 = (torch.randn(bh, d_h, d) / d ** 0.5).cuda()
    w1 = (torch.randn(bh, d, d_h) / d_h ** 0.5).cuda()
    q = torch.randn(bh, seq, d).cuda()
    k = torch.randn(bh, seq, d).cuda()
    v = torch.randn(bh, seq, d).cuda()
    lr_z = torch.zeros(bh, seq, 1).cuda()

    def run(pos0, lr1_scale, delta_only):
        pos = torch.arange(seq, device="cuda", dtype=torch.float32) + pos0
        ang = inv[:, None] * pos[None, :]
        lr1 = torch.full((bh, seq, 1), lr1_scale, device="cuda")
        with torch.no_grad():
            return prenorm_block_causal_lact_swiglu_value_rope(
                w0.clone(), w1.clone(), w2.clone(), q, k, v,
                lr_z, lr1, lr_z, ang.cos(), ang.sin(),
                chunk_size=chunk, use_muon=False, momentum=None,
                delta_only=delta_only,
            ).float()

    o_frozen = run(0.0, 0.0, True)  # lr1=0 -> pure init readout (pos-free)
    rows = []
    for name, delta_only in [("delta_only", True), ("full", False)]:
        o0 = run(0.0, 3e-4, delta_only)
        o1 = run(shift, 3e-4, delta_only)
        # violation normalized by the update's contribution to the output
        delta_mag = (o0 - o_frozen).abs().max().item()
        viol = (o1 - o0).abs().max().item()
        rows.append((name, viol, delta_mag, viol / delta_mag))
        print(f"[c-kernel] {name}: max|o(shift)-o(0)| = {viol:.3e}, "
              f"update-contribution mag = {delta_mag:.3e}, "
              f"ratio = {viol / delta_mag:.3f}")
    ok = rows[0][3] < 0.2 and rows[1][3] > 5 * rows[0][3]
    print(f"[c-kernel] delta_only ratio {rows[0][3]:.3f} (target < 0.2, "
          f"residual = weight-norm row-coupling + bf16) vs full ratio "
          f"{rows[1][3]:.3f} -> {'PASS' if ok else 'FAIL'}")
    return ok


def c(args):
    ok = c_mini() and c_kernel()
    print(f"[c] {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["a", "b", "c"])
    p.add_argument("--bs", type=int, default=2)
    p.add_argument("--steps", type=int, default=8)
    args = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    globals()[args.mode](args)
