# -*- coding: utf-8 -*-
"""Q25b GbR (gate-branch-only input rope) sanity suite (GPU-0 dev).

The SwiGLU fast weight f(x) = w1(silu(w0 x) * (w2 x)) has two input
branches: the silu GATE branch (w0) and the linear CONTENT branch (w2).
GbR routes the ROTATED q/k copy to one branch and the plain (unrotated,
post-l2norm) copy to the other, localizing where the input rope's value
lives.

Modes:
  a : kernel slot-equivalence — the branch kernel with the SAME tensors in
      both (gate, content) slots must reproduce the baseline prenorm kernel
      on those tensors BIT-EXACTLY (tested with a "rotated" and a "plain"
      tensor set, with and without muon+momentum).
  b : model-level 4-way — nope vs rope vs gate-only vs content-only on the
      same weights + batch: four finite, pairwise-distinct losses; backward
      runs with all-finite grads for gate and content.
  c : 100-step training smoke of gate-only (w128, bs 8) via train_small.py
      subprocess; loss must decrease and stay finite; smoke dir deleted.

Production 200M arch for b/c: 768/12L/12ah/4fwh, chunk 1024, window 128,
seq 4096, input rope ON — the target config for q25b_{gate,content}_w128.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
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

from lact_model import LaCTForCausalLM, LaCTSWIGLUConfig  # noqa: E402
from lact_model.ttt_operation import (  # noqa: E402
    l2_norm,
    prenorm_block_causal_lact_swiglu,
    prenorm_block_causal_lact_swiglu_branch_rope,
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
        ttt_prenorm=True,
    ))
    cfg.update(extra)
    return LaCTSWIGLUConfig(**cfg)


def build_model(seed=42, **extra):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return LaCTForCausalLM(build_config(**extra)).to("cuda")


def get_batch(bs=2):
    return torch.load(VAL_CACHE, map_location="cpu")[:bs].to("cuda")


# ---------------------------------------------------------------- a


def a(args):
    """Same tensors in both branch slots == baseline kernel, bit-exact."""
    torch.manual_seed(0)
    bh, d, d_h, seq, chunk = 8, 192, 192, 4096, 1024
    P = d // 2
    inv = 1.0 / (500000.0 ** (torch.arange(P, dtype=torch.float32) / P)).cuda()
    pos = torch.arange(seq, device="cuda", dtype=torch.float32)
    ang = pos[:, None] * inv[None, :]  # [s, P]
    cos = torch.cat([ang.cos(), ang.cos()], dim=-1)[None]  # [1, s, d]
    sin = torch.cat([ang.sin(), ang.sin()], dim=-1)[None]

    def rot(x):  # NeoX rotate_half, matching the layer's manual path
        x1, x2 = x.chunk(2, dim=-1)
        return (x * cos + torch.cat([-x2, x1], dim=-1) * sin).type_as(x)

    w0 = (torch.randn(bh, d_h, d) / d ** 0.5).cuda()
    w2 = (torch.randn(bh, d_h, d) / d ** 0.5).cuda()
    w1 = (torch.randn(bh, d, d_h) / d_h ** 0.5).cuda()
    q_plain = l2_norm(torch.randn(bh, seq, d).cuda())
    k_plain = l2_norm(torch.randn(bh, seq, d).cuda())
    v = torch.randn(bh, seq, d).cuda()
    q_rot, k_rot = rot(q_plain), rot(k_plain)
    lr0 = 1e-3 * torch.rand(bh, seq, 1).cuda()
    lr1 = 1e-3 * torch.rand(bh, seq, 1).cuda()
    lr2 = 1e-3 * torch.rand(bh, seq, 1).cuda()
    mom = torch.rand(bh, seq, 1).cuda()

    ok_all = True
    for setting, use_muon, momentum in [
        ("muon+mom", True, mom), ("plainsgd", False, None),
    ]:
        for name, (qq, kk) in [("rotated", (q_rot, k_rot)),
                               ("plain", (q_plain, k_plain))]:
            with torch.no_grad():
                o_base = prenorm_block_causal_lact_swiglu(
                    w0.clone(), w1.clone(), w2.clone(), qq, kk, v,
                    lr0, lr1, lr2, chunk_size=chunk,
                    use_muon=use_muon, momentum=momentum)
                o_br = prenorm_block_causal_lact_swiglu_branch_rope(
                    w0.clone(), w1.clone(), w2.clone(), qq, kk, qq, kk, v,
                    lr0, lr1, lr2, chunk_size=chunk,
                    use_muon=use_muon, momentum=momentum)
            diff = (o_base.float() - o_br.float()).abs().max().item()
            ok = diff == 0.0
            ok_all &= ok
            print(f"[a:{setting}:{name}] max|baseline - branch(same,same)| "
                  f"= {diff:.3e} -> {'PASS (bit-exact)' if ok else 'FAIL'}")
    print(f"[a] {'PASS' if ok_all else 'FAIL'}")
    return ok_all


# ---------------------------------------------------------------- b


def b(args):
    """nope / rope / gate / content: 4 finite pairwise-distinct losses,
    finite grads for the two new variants."""
    x = get_batch(args.bs)
    ref = build_model(seed=42)  # rope baseline; no variant adds parameters
    sd = ref.state_dict()
    del ref
    torch.cuda.empty_cache()

    variants = [
        ("nope", dict(ttt_nope=True)),
        ("rope", dict()),
        ("gate", dict(ttt_branch_rope="gate")),
        ("content", dict(ttt_branch_rope="content")),
    ]
    losses = {}
    ok_all = True
    for name, extra in variants:
        m = build_model(seed=42, **extra)
        missing, unexpected = m.load_state_dict(sd, strict=False)
        assert not missing and not unexpected, (name, missing, unexpected)
        m.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = m(input_ids=x, labels=x).loss
        finite = torch.isfinite(loss).item()
        ok_all &= finite
        if name in ("gate", "content"):
            loss.backward()
            n_grads, n_bad = 0, 0
            for pname, p in m.named_parameters():
                if p.grad is None:
                    continue
                n_grads += 1
                if not torch.isfinite(p.grad).all().item():
                    n_bad += 1
                    print(f"[b:{name}] NON-FINITE grad: {pname}")
            ok_all &= n_bad == 0
            print(f"[b:{name}] backward: {n_grads} params with grads, "
                  f"{n_bad} non-finite -> {'PASS' if n_bad == 0 else 'FAIL'}")
        losses[name] = loss.float().item()
        print(f"[b:{name}] loss = {losses[name]:.6f} finite={finite}")
        del m, loss
        torch.cuda.empty_cache()

    names = [n for n, _ in variants]
    distinct = all(losses[a_] != losses[b_]
                   for i, a_ in enumerate(names) for b_ in names[i + 1:])
    ok_all &= distinct
    print(f"[b] pairwise distinct: {distinct}")
    print(f"[b] {'PASS' if ok_all else 'FAIL'}")
    return ok_all


# ---------------------------------------------------------------- c


def c(args):
    """100-step gate-only training smoke (subprocess, killed by PID on
    timeout, smoke dir deleted afterwards)."""
    out_dir = os.path.join(SCRIPT_DIR, "outputs", "q25b_smoke_gate_w128")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "train_small.py"),
        "--out_dir", out_dir,
        "--window_size", "128",
        "--bs", "8",
        "--steps", "100",
        "--log_every", "10",
        "--val_every", "1000000",
        "--save_every", "1000000",
        "--extra_json", json.dumps({"ttt_branch_rope": "gate"}),
    ]
    log_path = os.path.join(out_dir, "smoke.log")
    print(f"[c] launching: {' '.join(cmd)}")
    with open(log_path, "w") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                cwd=SCRIPT_DIR)
        print(f"[c] PID {proc.pid}")
        try:
            proc.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            print(f"[c] TIMEOUT: killing PID {proc.pid}")
            proc.kill()
            proc.wait()
    with open(log_path) as f:
        log = f.read()
    steps = re.findall(r"step=(\d+) loss=([\d.]+)", log)
    losses = [(int(s), float(v)) for s, v in steps]
    print("[c] logged losses: " + " ".join(f"{s}:{v:.4f}" for s, v in losses))
    finite = all(torch.isfinite(torch.tensor(v)).item() for _, v in losses)
    nan_warn = "non-finite loss" in log
    ok = (proc.returncode == 0 and len(losses) >= 2 and finite
          and not nan_warn and losses[-1][1] < losses[0][1])
    if not ok:
        print(f"[c] rc={proc.returncode} tail of log:\n" + log[-2000:])
    else:
        shutil.rmtree(out_dir)
        print(f"[c] loss {losses[0][1]:.4f} -> {losses[-1][1]:.4f}, "
              f"smoke dir deleted")
    print(f"[c] {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["a", "b", "c"])
    p.add_argument("--bs", type=int, default=2)
    args = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    globals()[args.mode](args)
