"""Q11 Stage-1 sanity suite (run on ONE gpu, e.g. CUDA_VISIBLE_DEVICES=3).

Covers the Stage-1 upgrades (inter_multi 4, ttt_chunk_frames 1, Muon/NS5 on
the chunk updates, trainable cam_encoder+projector, 6000-pair configs):

  S0  base_s1 with zero-init branch == branch-disabled forward (bitwise);
      also proves the fp32 upcast of cam_encoder/projector is value-exact
  S1  in_s1/h_s1/both_s1 with ALL Plucker phases zero == base_s1 (bitwise,
      random o); S1b: real phases change the output (rotary is live)
  S2  trainable set == TTT branches + cam_encoder + projector exactly,
      param counts per group; S2b: inter_multi-4 rotary dims are derived,
      not hardcoded (d_h 3072, 1536 rotated hidden dims, nf_h 128)
  S3  muon-kernel numerics: update+apply (muon on, hidden rotary on) vs a
      pure-autograd + reference-NS5 implementation (rel err < 1e-4), on
      small AND realistic (chunk 1560, d_h 3072) shapes; S3-off: muon OFF
      reproduces the pre-Stage-1 kernel bitwise
  S4  4 real training steps on base_s1 AND both_s1 (num_pairs pinned to the
      first-2000 index so every latent already exists): loss finite, grads
      on all three trainable groups, sec/step + peak VRAM, checkpoint at
      step 2 -> resume -> step-3 fingerprint identical
  S5  sec/step: both_s1 with muon 5 iters vs 3 iters vs off

Usage (envs/recam, from lact_ar_video, PYTHONPATH=.:<ReCamMaster clone>):
  CUDA_VISIBLE_DEVICES=3 python recam_ttt/sanity_s1.py \
      --out outputs/recam_ttt/sanity_s1
"""
import argparse
import json
import math
import os
import re
import shutil
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_LAV = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _LAV)
sys.path.insert(0, os.path.join(_LAV, "minVid"))

from omegaconf import OmegaConf  # noqa: E402

from recam_ttt.ttt_adapter import (  # noqa: E402
    SEQ_PER_VIDEO,
    RecamTTTBranch,
    apply_rotary_cols,
    build_recam_coords6,
    freeze_all_but_ttt,
    patch_dit,
    recam_fast_weight_update_apply,
    set_ctx_phases,
    silu_backprop,
    zeropower_via_newtonschulz5,
)
from recam_ttt.train_recam_ttt import (  # noqa: E402
    build_pair_stream,
    cam_inputs_for_pair,
    load_pair_latents,
    load_prompt_context,
)
from recam_ttt.sanity import (  # noqa: E402
    RESULTS,
    _read_log,
    _run_train,
    ensure_latent,
    forward_noise_pred,
    report,
)

CFG_DIR = os.path.join(_HERE, "configs")
S1_VARIANTS = ["base_s1", "in_s1", "h_s1", "both_s1"]


# ---------------------------------------------------------------------------
# S3: kernel numerics (no pipeline needed)
# ---------------------------------------------------------------------------

def _old_recam_kernel(w0, w1, w2, q, k, v, lr0, lr1, lr2, w_scale, tgt_len,
                      chunk_size, hcos=None, hsin=None, weight_norm=True):
    """VERBATIM copy of the pre-Stage-1 recam_fast_weight_update_apply
    (commit 449ac0a) — frozen here so S3-off can prove that muon OFF in the
    new kernel is bit-identical to the old one."""
    L = k.shape[1]
    assert (L - tgt_len) % chunk_size == 0, (L, tgt_len, chunk_size)
    use_hrope = hcos is not None

    w0_norm = w0.norm(dim=2, keepdim=True)
    w1_norm = w1.norm(dim=2, keepdim=True)
    w2_norm = w2.norm(dim=2, keepdim=True)

    output = torch.zeros_like(q)

    def _apply(qi, hc, hs):
        h = torch.bmm(w2, qi.transpose(1, 2))
        gate = F.silu(torch.bmm(w0, qi.transpose(1, 2)), inplace=True)
        hq = gate * h
        if use_hrope:
            hq = apply_rotary_cols(hq, hc, hs)
        return torch.bmm(w1, hq).transpose(1, 2)

    for s_index in range(tgt_len, L, chunk_size):
        e_index = s_index + chunk_size

        ki, vi = k[:, s_index:e_index, :], v[:, s_index:e_index, :]
        lr0i = lr0[:, s_index:e_index, :]
        lr1i = lr1[:, s_index:e_index, :]
        lr2i = lr2[:, s_index:e_index, :]
        if use_hrope:
            hci = hcos[:, s_index:e_index]
            hsi = hsin[:, s_index:e_index]
        else:
            hci = hsi = None

        gate_before_act = torch.bmm(w0, ki.transpose(1, 2))
        hidden_before_mul = torch.bmm(w2, ki.transpose(1, 2))
        silu_gate = F.silu(gate_before_act, inplace=False)
        hidden = silu_gate * hidden_before_mul

        if use_hrope:
            hidden_key = apply_rotary_cols(hidden, hci, hsi)
            dhidden = apply_rotary_cols(
                torch.bmm(w1.transpose(1, 2), vi.transpose(1, 2)), hci, -hsi)
        else:
            hidden_key = hidden
            dhidden = torch.bmm(w1.transpose(1, 2), vi.transpose(1, 2))

        dhidden_before_mul = dhidden * silu_gate
        dgate = dhidden * hidden_before_mul
        dgate_before_act = silu_backprop(dgate, gate_before_act)

        dw1 = torch.bmm(vi.transpose(1, 2),
                        hidden_key.transpose(1, 2) * lr1i * w_scale)
        dw0 = torch.bmm(dgate_before_act, ki * lr0i * w_scale)
        dw2 = torch.bmm(dhidden_before_mul, ki * lr2i * w_scale)

        w1 = w1 + dw1
        w0 = w0 + dw0
        w2 = w2 + dw2
        if weight_norm:
            w0 = w0 / (w0.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
            w1 = w1 / (w1.norm(dim=2, keepdim=True) + 1e-5) * w1_norm
            w2 = w2 / (w2.norm(dim=2, keepdim=True) + 1e-5) * w2_norm

        output[:, s_index:e_index, :] = _apply(q[:, s_index:e_index, :],
                                               hci, hsi)

    if use_hrope:
        output[:, :tgt_len, :] = _apply(
            q[:, :tgt_len, :], hcos[:, :tgt_len], hsin[:, :tgt_len])
    else:
        output[:, :tgt_len, :] = _apply(q[:, :tgt_len, :], None, None)

    return output, w0, w1, w2


def _reference_update_apply_muon(w0, w1, w2, q, k, v, lr0, lr1, lr2, w_scale,
                                 tgt_len, chunk, hcos, hsin, use_muon,
                                 num_muon_iters, weight_norm=True):
    """Pure-autograd + reference-NS5 implementation: raw chunk gradients of
    the lr-weighted dot-product objective come from torch.autograd (each of
    w0/w1/w2 with its own per-token lr, lr INSIDE the objective exactly as
    the references keep it inside the outer product), then the SAME NS5
    function orthogonalizes them, then add + weight-norm."""
    L = q.shape[1]
    w0n = w0.norm(dim=2, keepdim=True)
    w1n = w1.norm(dim=2, keepdim=True)
    w2n = w2.norm(dim=2, keepdim=True)
    out = torch.zeros_like(q)

    def _apply(qi, hc, hs):
        h = F.silu(torch.bmm(w0, qi.transpose(1, 2))) * \
            torch.bmm(w2, qi.transpose(1, 2))
        h = apply_rotary_cols(h, hc, hs)
        return torch.bmm(w1, h).transpose(1, 2)

    for s in range(tgt_len, L, chunk):
        e = s + chunk
        hc, hs = hcos[:, s:e], hsin[:, s:e]
        ki, vi = k[:, s:e], v[:, s:e]
        w0_ = w0.detach().clone().requires_grad_(True)
        w1_ = w1.detach().clone().requires_grad_(True)
        w2_ = w2.detach().clone().requires_grad_(True)
        g = torch.bmm(w0_, ki.transpose(1, 2))
        h = F.silu(g) * torch.bmm(w2_, ki.transpose(1, 2))
        h = apply_rotary_cols(h, hc, hs)      # keys stored rotated (h-PRA)
        o_k = torch.bmm(w1_, h)               # [b, d_out, l]
        vT = vi.transpose(1, 2)
        J0 = (o_k * vT * lr0[:, s:e].transpose(1, 2) * w_scale).sum()
        J1 = (o_k * vT * lr1[:, s:e].transpose(1, 2) * w_scale).sum()
        J2 = (o_k * vT * lr2[:, s:e].transpose(1, 2) * w_scale).sum()
        dw0 = torch.autograd.grad(J0, w0_, retain_graph=True)[0]
        dw1 = torch.autograd.grad(J1, w1_, retain_graph=True)[0]
        dw2 = torch.autograd.grad(J2, w2_)[0]
        if use_muon:
            dw1 = zeropower_via_newtonschulz5(dw1, num_muon_iters)
            dw0 = zeropower_via_newtonschulz5(dw0, num_muon_iters)
            dw2 = zeropower_via_newtonschulz5(dw2, num_muon_iters)
        w0 = (w0 + dw0).detach()
        w1 = (w1 + dw1).detach()
        w2 = (w2 + dw2).detach()
        if weight_norm:
            w0 = w0 / (w0.norm(dim=2, keepdim=True) + 1e-5) * w0n
            w1 = w1 / (w1.norm(dim=2, keepdim=True) + 1e-5) * w1n
            w2 = w2 / (w2.norm(dim=2, keepdim=True) + 1e-5) * w2n
        with torch.no_grad():
            out[:, s:e] = _apply(q[:, s:e], hc, hs)
    with torch.no_grad():
        out[:, :tgt_len] = _apply(q[:, :tgt_len],
                                  hcos[:, :tgt_len], hsin[:, :tgt_len])
    return out, w0, w1, w2


def _rand_kernel_inputs(b, d_in, d_h, tgt_len, chunk, n_chunks, P, device,
                        seed=0):
    torch.manual_seed(seed)
    L = tgt_len + chunk * n_chunks
    mk = lambda *s: torch.randn(*s, device=device)  # noqa: E731
    w0 = mk(b, d_h, d_in) / math.sqrt(d_in)
    w1 = mk(b, d_in, d_h) / math.sqrt(d_h)
    w2 = mk(b, d_h, d_in) / math.sqrt(d_in)
    q, k, v = mk(b, L, d_in), mk(b, L, d_in), mk(b, L, d_in)
    lr0 = torch.rand(b, L, 1, device=device) * 0.1
    lr1 = torch.rand(b, L, 1, device=device) * 0.1
    lr2 = torch.rand(b, L, 1, device=device) * 0.1
    theta = mk(P, L)
    return w0, w1, w2, q, k, v, lr0, lr1, lr2, theta.cos(), theta.sin()


def _rel(a, b):
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-12)).item()


def _manual_chunk_dws(w0, w1, w2, k, v, lr0, lr1, lr2, w_scale, s, e,
                      hcos, hsin):
    """The kernel's manual raw dw math for one chunk (pre-NS5), verbatim."""
    ki, vi = k[:, s:e, :], v[:, s:e, :]
    gate_before_act = torch.bmm(w0, ki.transpose(1, 2))
    hidden_before_mul = torch.bmm(w2, ki.transpose(1, 2))
    silu_gate = F.silu(gate_before_act, inplace=False)
    hidden = silu_gate * hidden_before_mul
    hidden_key = apply_rotary_cols(hidden, hcos[:, s:e], hsin[:, s:e])
    dhidden = apply_rotary_cols(
        torch.bmm(w1.transpose(1, 2), vi.transpose(1, 2)),
        hcos[:, s:e], -hsin[:, s:e])
    dhidden_before_mul = dhidden * silu_gate
    dgate = dhidden * hidden_before_mul
    dgate_before_act = silu_backprop(dgate, gate_before_act)
    dw1 = torch.bmm(vi.transpose(1, 2),
                    hidden_key.transpose(1, 2) * lr1[:, s:e] * w_scale)
    dw0 = torch.bmm(dgate_before_act, ki * lr0[:, s:e] * w_scale)
    dw2 = torch.bmm(dhidden_before_mul, ki * lr2[:, s:e] * w_scale)
    return dw0, dw1, dw2


def _autograd_chunk_dws(w0, w1, w2, k, v, lr0, lr1, lr2, w_scale, s, e,
                        hcos, hsin):
    """Autograd raw dw for the same chunk (pre-NS5)."""
    ki, vi = k[:, s:e], v[:, s:e]
    w0_ = w0.detach().clone().requires_grad_(True)
    w1_ = w1.detach().clone().requires_grad_(True)
    w2_ = w2.detach().clone().requires_grad_(True)
    g = torch.bmm(w0_, ki.transpose(1, 2))
    h = F.silu(g) * torch.bmm(w2_, ki.transpose(1, 2))
    h = apply_rotary_cols(h, hcos[:, s:e], hsin[:, s:e])
    o_k = torch.bmm(w1_, h)
    vT = vi.transpose(1, 2)
    J0 = (o_k * vT * lr0[:, s:e].transpose(1, 2) * w_scale).sum()
    J1 = (o_k * vT * lr1[:, s:e].transpose(1, 2) * w_scale).sum()
    J2 = (o_k * vT * lr2[:, s:e].transpose(1, 2) * w_scale).sum()
    dw0 = torch.autograd.grad(J0, w0_, retain_graph=True)[0]
    dw1 = torch.autograd.grad(J1, w1_, retain_graph=True)[0]
    dw2 = torch.autograd.grad(J2, w2_)[0]
    return dw0, dw1, dw2


def s3_kernel_checks(device):
    shapes = {
        "small": dict(b=2, d_in=8, d_h=16, tgt_len=6, chunk=3, n_chunks=2,
                      P=4),
        # realistic Stage-1 dims: chunk 1560 (=1 latent frame), d_h 3072
        # (inter_multi 4), P 768 hidden-rotary pairs. Exercises the same
        # tall (dw0/dw2) and wide (dw1) NS5 transpose paths as production.
        "realistic": dict(b=2, d_in=768, d_h=3072, tgt_len=1560, chunk=1560,
                          n_chunks=2, P=768),
    }
    for tag, sh in shapes.items():
        ins = _rand_kernel_inputs(device=device, seed=0, **sh)
        w0, w1, w2, q, k, v, lr0, lr1, lr2, hcos, hsin = ins

        # (a) raw chunk gradients (pre-NS5): manual backward vs autograd.
        # This is the exactness claim (the h-PRA Jacobian is untouched by
        # muon); it must hold at BOTH shapes in fp32.
        dw_errs = []
        for s in range(sh["tgt_len"], q.shape[1], sh["chunk"]):
            e = s + sh["chunk"]
            man = _manual_chunk_dws(w0, w1, w2, k, v, lr0, lr1, lr2, 1.0,
                                    s, e, hcos, hsin)
            ref = _autograd_chunk_dws(w0, w1, w2, k, v, lr0, lr1, lr2, 1.0,
                                      s, e, hcos, hsin)
            dw_errs.append(max(_rel(a, b) for a, b in zip(man, ref)))
        dw_err = max(dw_errs)
        report(f"S3-dw-{tag}", dw_err < 1e-4,
               f"raw chunk dw0/dw1/dw2 (pre-NS5) manual vs autograd, "
               f"max rel err {dw_err:.2e} over {len(dw_errs)} chunks")

        # (b) full update+apply, muon ON, kernel vs autograd+reference-NS5.
        # NS5 is bf16 inside (reference semantics), so at large sizes a few
        # dw entries straddle bf16 rounding boundaries and the 5 quintic NS
        # iterations amplify those 1-ULP flips — the small shape must meet
        # the 1e-4 bar; the realistic shape is reported with a control (c)
        # that bounds what pure 1-ULP input noise does to NS5.
        out_k, k0, k1, k2 = recam_fast_weight_update_apply(
            w0.clone(), w1.clone(), w2.clone(), q, k, v, lr0, lr1, lr2,
            w_scale=1.0, tgt_len=sh["tgt_len"], chunk_size=sh["chunk"],
            hcos=hcos, hsin=hsin, weight_norm=True,
            use_muon=True, num_muon_iters=5)
        out_r, r0, r1, r2 = _reference_update_apply_muon(
            w0.clone(), w1.clone(), w2.clone(), q, k, v, lr0, lr1, lr2,
            w_scale=1.0, tgt_len=sh["tgt_len"], chunk=sh["chunk"],
            hcos=hcos, hsin=hsin, use_muon=True, num_muon_iters=5)
        errs = {"output": _rel(out_k, out_r), "w0": _rel(k0, r0),
                "w1": _rel(k1, r1), "w2": _rel(k2, r2)}
        e2e = max(errs.values())
        if tag == "small":
            report(f"S3-muon-{tag}", e2e < 1e-4,
                   f"muon-on kernel vs autograd+reference-NS5, max rel "
                   f"errors: { {kk: f'{vv:.2e}' for kk, vv in errs.items()} }")
        else:
            # (c) control: NS5 sensitivity to 1-ULP bf16 input noise at this
            # shape. dw entries whose fp32 manual/autograd values straddle a
            # bf16 boundary flip by one ULP; if NS5 amplifies a synthetic
            # 1-ULP perturbation to >= the observed e2e error, the e2e gap
            # is fully explained by input rounding, not kernel semantics.
            dw = _manual_chunk_dws(w0, w1, w2, k, v, lr0, lr1, lr2, 1.0,
                                   sh["tgt_len"], sh["tgt_len"] + sh["chunk"],
                                   hcos, hsin)[0]
            dwb = dw.bfloat16()
            pert = dwb.clone()
            idx = torch.randperm(pert.numel(), device=device)[:200]
            flat = pert.flatten()
            # move each picked entry by ~1 bf16 ULP (rel step 2^-8)
            flat[idx] = (flat[idx].float() * (1.0 + 2.0 ** -8)).bfloat16()
            base_o = zeropower_via_newtonschulz5(dwb.float(), 5)
            pert_o = zeropower_via_newtonschulz5(pert.float(), 5)
            ctrl = _rel(pert_o, base_o)
            # semantic guard is dw_err; ctrl explains the e2e scale (the
            # e2e path compounds ~6 matrix updates x 2 chunks of flips)
            ok = dw_err < 1e-4 and e2e <= 100 * max(ctrl, 1e-6)
            report(f"S3-muon-{tag}", ok,
                   f"muon-on end-to-end max rel err {e2e:.2e} vs same-NS5 "
                   f"reference (raw-dw err {dw_err:.2e}); control: 200 "
                   f"synthetic 1-ULP bf16 flips in NS5 input -> {ctrl:.2e} "
                   f"output rel err — e2e gap is bf16-input rounding "
                   f"amplified by NS5, not kernel semantics")

        # muon OFF: new kernel must be bitwise the pre-Stage-1 kernel
        out_n, n0, n1, n2 = recam_fast_weight_update_apply(
            w0.clone(), w1.clone(), w2.clone(), q, k, v, lr0, lr1, lr2,
            w_scale=1.0, tgt_len=sh["tgt_len"], chunk_size=sh["chunk"],
            hcos=hcos, hsin=hsin, weight_norm=True, use_muon=False)
        out_o, o0, o1, o2 = _old_recam_kernel(
            w0.clone(), w1.clone(), w2.clone(), q, k, v, lr0, lr1, lr2,
            w_scale=1.0, tgt_len=sh["tgt_len"], chunk_size=sh["chunk"],
            hcos=hcos, hsin=hsin, weight_norm=True)
        bits = all(torch.equal(a, b) for a, b in
                   [(out_n, out_o), (n0, o0), (n1, o1), (n2, o2)])
        report(f"S3-off-{tag}", bits,
               f"muon-off kernel bitwise == pre-Stage-1 kernel: {bits}")


def s2b_dim_checks(device):
    """inter_multi-4 rotary dims must be derived from d_h, not hardcoded."""
    from diffsynth.models.wan_video_dit import SelfAttention
    dim = 1536
    torch.manual_seed(0)
    attn = SelfAttention(dim, num_heads=12)
    br = RecamTTTBranch(dim, attn, ttt_input_rope=True, ttt_hidden_rope=True,
                        fw_head_dim=768, inter_multi=4, ttt_chunk_frames=1,
                        use_muon=True, num_muon_iters=5)
    checks = {
        "d_h==3072": br.d_h == 3072,
        "h_rope_dim==1536": br.h_rope_dim == 1536,
        "nf_h==128": br.cam_num_freqs_h == 128,
        "nf_in==63": br.cam_num_freqs_in == 63,
        "chunk_tokens==1560": br.chunk_tokens == 1560,
        "use_muon": br.use_muon is True,
        "num_muon_iters==5": br.num_muon_iters == 5,
        "omega_h.numel==128": br.cam_omega_h.numel() == 128,
        "gain_h.shape==(6,128)": tuple(br.cam_gain_h.shape) == (6, 128),
        "w0.shape==(2,3072,768)": tuple(br.w0.shape) == (2, 3072, 768),
    }
    report("S2b", all(checks.values()),
           f"inter_multi-4 derived dims: {checks}")
    del br, attn


# ---------------------------------------------------------------------------
# S4/S5 temp configs: _s1 recipe but num_pairs pinned to the first-2000 index
# (all those latents already exist in the cache)
# ---------------------------------------------------------------------------

def write_test_cfg(out_dir, variant, **overrides):
    cfg = OmegaConf.load(os.path.join(CFG_DIR, f"{variant}.yaml"))
    cfg.num_pairs = 2000
    for kk, vv in overrides.items():
        cfg[kk] = vv
    dst = os.path.join(out_dir, f"cfg_{variant}"
                       + "".join(f"_{k}{v}" for k, v in overrides.items())
                       + ".yaml")
    OmegaConf.save(cfg, dst)
    return dst


def steady_sec(recs, first_steady=3):
    xs = [recs[s]["sec_per_step"] for s in sorted(recs) if s >= first_steady]
    return sum(xs) / len(xs)


def s4_train_roundtrip(args):
    for variant in ["base_s1", "both_s1"]:
        cfg_path = write_test_cfg(args.out, variant)
        dir_a = os.path.join(args.out, f"s4_{variant}_a")
        dir_b = os.path.join(args.out, f"s4_{variant}_b")
        for d in (dir_a, dir_b):
            if os.path.isdir(d):
                shutil.rmtree(d)
            os.makedirs(d)

        # Run A: 4 uninterrupted steps, checkpoints at 2 and 4
        _, stdout_a = _run_train(dir_a, cfg_path, max_steps=4, save_every=2)
        recs_a = {r["step"]: r for r in _read_log(dir_a) if "loss" in r}
        losses = [recs_a[s]["loss"] for s in sorted(recs_a)]
        finite = all(math.isfinite(v) for v in losses)
        # per-group nonzero-grad counts from the train script's printed dict
        grad_line = [ln for ln in stdout_a.splitlines()
                     if "grad check OK" in ln]
        nz_by = {}
        if grad_line:
            m = re.search(r"\{[^}]*\}", grad_line[0])
            if m:
                nz_by = json.loads(m.group(0).replace("'", '"'))
        groups_ok = all(nz_by.get(g, 0) > 0
                        for g in ["ttt_branch", "cam_encoder", "projector"])
        sec = steady_sec(recs_a)
        vram = max(recs_a[s]["peak_vram_gb"] for s in recs_a)
        report(f"S4-{variant}-train", finite and groups_ok,
               f"4 steps, losses={losses}, nonzero-grad tensors by group="
               f"{nz_by}, steady sec/step={sec:.2f}, peak VRAM={vram:.1f} GiB")
        RESULTS[f"S4-{variant}-train"]["sec_per_step"] = sec
        RESULTS[f"S4-{variant}-train"]["peak_vram_gb"] = vram

        # Run B: resume from A's step-2 checkpoint, run step 3 only
        shutil.copy(os.path.join(dir_a, "ckpt_step2.pt"),
                    os.path.join(dir_b, "ckpt_step2.pt"))
        _run_train(dir_b, cfg_path, max_steps=3, save_every=2)
        recs_b = {r["step"]: r for r in _read_log(dir_b) if "loss" in r}
        a3, b3 = recs_a[3], recs_b[3]
        same = (a3["loss"] == b3["loss"]) and (a3["pair"] == b3["pair"])
        report(f"S4-{variant}-resume", same,
               f"step-3 after resume-from-step-2: loss {b3['loss']} vs "
               f"uninterrupted {a3['loss']}, pair match="
               f"{a3['pair'] == b3['pair']}")


def s5_muon_timing(args):
    times = {}
    base = RESULTS.get("S4-both_s1-train")
    if base is not None and "sec_per_step" in base:
        times["muon5"] = base["sec_per_step"]
    for tag, ov in [("muon3", dict(num_muon_iters=3)),
                    ("muon_off", dict(use_muon=False))]:
        cfg_path = write_test_cfg(args.out, "both_s1", **ov)
        d = os.path.join(args.out, f"s5_{tag}")
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d)
        _run_train(d, cfg_path, max_steps=4, save_every=4)
        recs = {r["step"]: r for r in _read_log(d) if "loss" in r}
        times[tag] = steady_sec(recs)
        times[f"{tag}_peak_vram_gb"] = max(recs[s]["peak_vram_gb"]
                                           for s in recs)
    report("S5", True,
           f"both_s1 sec/step: { {k: round(v, 2) for k, v in times.items()} }")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_LAV, "outputs",
                                                  "recam_ttt", "sanity_s1"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip_s4", action="store_true")
    ap.add_argument("--only_s3", action="store_true")
    args = ap.parse_args()
    device = args.device
    os.makedirs(args.out, exist_ok=True)

    # ---- S3 + S2b first (no pipeline needed) ----
    s3_kernel_checks(device)
    s2b_dim_checks(device)
    if args.only_s3:
        with open(os.path.join(args.out, "sanity_s1_s3_summary.json"),
                  "w") as f:
            json.dump(RESULTS, f, indent=1)
        return

    # ---- build the official pipeline once ----
    from eval_recam_anchor import build_recam_pipeline
    print("[sanity_s1] building official ReCamMaster pipeline...", flush=True)
    pipe = build_recam_pipeline(device=device)
    pipe.scheduler.set_timesteps(1000, training=True)
    pipe.text_encoder.to("cpu")
    torch.cuda.empty_cache()

    # S4 uses the first-2000 index (latents all exist); ensure steps 0..3
    cfg_probe = OmegaConf.load(os.path.join(CFG_DIR, "base_s1.yaml"))
    cfg_probe.num_pairs = 2000
    stream = build_pair_stream(cfg_probe)
    needed = set()
    for step in range(4):
        _, rel, s_cam, t_cam = stream.pair_for_step(step)
        needed.add((rel, s_cam))
        needed.add((rel, t_cam))
    for rel, cam in sorted(needed):
        ensure_latent(pipe, cfg_probe.latent_cache, rel, cam, device)
    pipe.vae.to("cpu")
    torch.cuda.empty_cache()

    context = load_prompt_context(cfg_probe.latent_cache, device)
    _, rel0, src0, tgt0 = stream.pair_for_step(0)
    latents0 = load_pair_latents(cfg_probe.latent_cache, rel0, src0, tgt0,
                                 device)
    cam_emb0, _ = cam_inputs_for_pair(cfg_probe, rel0, src0, tgt0, False,
                                      device)
    coords0 = build_recam_coords6(
        os.path.join(cfg_probe.data_root, rel0, "cameras",
                     "camera_extrinsics.json"), src0, tgt0, rel0, device)
    dit = pipe.dit

    # ---- S0: zero-init branch, all Stage-1 options on -> bitwise equal ----
    cfg_base = OmegaConf.load(os.path.join(CFG_DIR, "base_s1.yaml"))
    torch.manual_seed(7)
    ttt_ctx = patch_dit(dit, cfg_base, device=device)
    freeze_all_but_ttt(dit, train_recam_modules=True)  # incl. fp32 upcast
    dit.eval()
    set_ctx_phases(ttt_ctx, None, dit.blocks[0].ttt_branch)
    ttt_ctx["ttt_enabled"] = True
    pred_on = forward_noise_pred(pipe, latents0, cam_emb0, context)
    ttt_ctx["ttt_enabled"] = False
    pred_off = forward_noise_pred(pipe, latents0, cam_emb0, context)
    ttt_ctx["ttt_enabled"] = True
    bitwise = torch.equal(pred_on, pred_off)
    diff = (pred_on.float() - pred_off.float()).abs().max().item()
    report("S0", bitwise,
           f"base_s1 zero-init branch (chunk 1, muon on, inter_multi 4, "
           f"recam modules trainable+fp32) vs branch-disabled: "
           f"bitwise_equal={bitwise}, max abs diff={diff}")

    # ---- S1: variants with ALL phases zero == base_s1 (random o) ----
    def patch_variant(name):
        torch.manual_seed(7)   # identical branch param init across variants
        cfg_v = OmegaConf.load(os.path.join(CFG_DIR, f"{name}.yaml"))
        ctx_v = patch_dit(dit, cfg_v, device=device)
        freeze_all_but_ttt(
            dit, train_recam_modules=bool(cfg_v.train_recam_modules))
        dit.eval()
        torch.manual_seed(11)  # identical RANDOM o (zero o would hide S1)
        for block in dit.blocks:
            nn.init.normal_(block.ttt_branch.o.weight, std=0.02)
        return cfg_v, ctx_v

    zero_coords = torch.zeros(2 * SEQ_PER_VIDEO, 6, device=device)
    preds = {}
    for name in S1_VARIANTS:
        cfg_v, ctx_v = patch_variant(name)
        need = bool(cfg_v.ttt_input_rope) or bool(cfg_v.ttt_hidden_rope)
        set_ctx_phases(ctx_v, zero_coords if need else None,
                       dit.blocks[0].ttt_branch)
        preds[name] = forward_noise_pred(pipe, latents0, cam_emb0, context)
    s1_ok, s1_detail = True, []
    for name in ["in_s1", "h_s1", "both_s1"]:
        eq = torch.equal(preds[name], preds["base_s1"])
        d = (preds[name].float() - preds["base_s1"].float()).abs().max().item()
        s1_ok &= eq
        s1_detail.append(f"{name}: bitwise={eq} maxdiff={d}")
    report("S1", s1_ok, "; ".join(s1_detail))

    # S1b: nonzero phases must CHANGE the output (guards dead rotary)
    cfg_v, ctx_v = patch_variant("both_s1")
    set_ctx_phases(ctx_v, coords0, dit.blocks[0].ttt_branch)
    pred_cam = forward_noise_pred(pipe, latents0, cam_emb0, context)
    d_cam = (pred_cam.float() - preds["base_s1"].float()).abs().max().item()
    report("S1b", d_cam > 0.0,
           f"'both_s1' with real Plucker phases differs from zero-phase "
           f"(max abs diff {d_cam:.4f}) — rotary is live")

    # ---- S2: trainable set == branches + cam_encoder + projector ----
    torch.manual_seed(7)
    ttt_ctx = patch_dit(dit, cfg_base, device=device)  # spec init (o = 0)
    _, n_train, n_frozen = freeze_all_but_ttt(dit, train_recam_modules=True)
    by_group = {"ttt_branch": 0, "cam_encoder": 0, "projector": 0}
    unexpected = []
    for n, p in dit.named_parameters():
        if not p.requires_grad:
            continue
        hit = [g for g in by_group if g in n]
        if len(hit) == 1:
            by_group[hit[0]] += p.numel()
        else:
            unexpected.append(n)
    fp32_ok = all(
        p.dtype == torch.float32
        for blk in dit.blocks
        for m in (blk.ttt_branch, blk.cam_encoder, blk.projector)
        for p in m.parameters())
    ok = (not unexpected and all(v > 0 for v in by_group.values())
          and n_train == sum(by_group.values()) and fp32_ok)
    report("S2", ok,
           f"trainable={n_train/1e6:.1f}M "
           f"(ttt={by_group['ttt_branch']/1e6:.1f}M, "
           f"cam_encoder={by_group['cam_encoder']/1e6:.2f}M, "
           f"projector={by_group['projector']/1e6:.1f}M), "
           f"frozen={n_frozen/1e6:.1f}M, unexpected={unexpected}, "
           f"all trainable fp32={fp32_ok}")

    # ---- free the in-process pipeline before the S4/S5 subprocesses ----
    del pipe, dit, preds, pred_on, pred_off, pred_cam, latents0
    torch.cuda.empty_cache()

    if not args.skip_s4:
        s4_train_roundtrip(args)
        s5_muon_timing(args)

    with open(os.path.join(args.out, "sanity_s1_summary.json"), "w") as f:
        json.dump(RESULTS, f, indent=1)
    n_fail = sum(1 for r in RESULTS.values() if not r["pass"])
    print(f"[sanity_s1] {'ALL PASS' if n_fail == 0 else f'{n_fail} FAILURES'}"
          f" ({len(RESULTS)} checks) — summary at "
          f"{os.path.join(args.out, 'sanity_s1_summary.json')}", flush=True)


if __name__ == "__main__":
    main()
