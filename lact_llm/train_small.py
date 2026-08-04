# -*- coding: utf-8 -*-
"""Minimal single-GPU LM training script for controlled LaCT ablations.

RUNTIME ENVIRONMENT (required):
  python : /NHNHOME/WORKSPACE/26msit001_T_B/POSTECH-CGLAB/TTT_camera_embedding/.venv_llm/bin/python
  env    : TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
           TRITON_CUOBJDUMP_PATH=/usr/local/cuda/bin/cuobjdump
           TRITON_NVDISASM_PATH=/usr/local/cuda/bin/nvdisasm
           C_INCLUDE_PATH=/usr/local/cuda/include
           PATH=/usr/local/cuda/bin:$PATH
           HF_HOME=/tmp/hf_cache   (set automatically below if unset)
           TRITON_CACHE_DIR / TORCHINDUCTOR_CACHE_DIR -> repo-local
           .cache_triton / .cache_inductor (set automatically below; /tmp and
           /dev/shm are mounted noexec on this machine, so triton cannot load
           compiled launchers from there)
  cwd    : run from lact_llm/ (script also adds its own dir to sys.path)

Example:
  CUDA_VISIBLE_DEVICES=0 python train_small.py \
      --config configs/760M_lact_swiglu_nh4_fwlow_rank_momentum_muon.json \
      --out_dir outputs/base_small

Or use the wrapper:  ./run_llm.sh 0 base_small [extra args...]
"""

import os
import sys

os.environ.setdefault("HF_HOME", "/tmp/hf_cache")
# /tmp is noexec on this machine; triton/inductor must compile into an
# exec-allowed filesystem (repo-local cache dirs).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_REPO_ROOT, ".cache_triton"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(_REPO_ROOT, ".cache_inductor"))

import argparse
import glob
import json
import math
import re
import time

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from lact_model import LaCTForCausalLM, LaCTSWIGLUConfig  # noqa: E402
import data_utils  # noqa: E402
import synthetic_copy  # noqa: E402


def str2bool(v):
    return str(v).lower() in ("1", "true", "yes", "y")


def parse_args():
    p = argparse.ArgumentParser(description="Minimal single-GPU LaCT LM trainer")
    # config / model
    p.add_argument("--config", type=str,
                   default=os.path.join(SCRIPT_DIR, "configs/760M_lact_swiglu_nh4_fwlow_rank_momentum_muon.json"),
                   help="Base JSON config; small-model CLI overrides applied on top.")
    p.add_argument("--hidden_size", type=int, default=768)
    p.add_argument("--num_hidden_layers", type=int, default=12)
    p.add_argument("--num_attn_heads", type=int, default=12)
    p.add_argument("--num_lact_heads", type=int, default=4)
    p.add_argument("--lact_chunk_size", type=int, default=1024)
    p.add_argument("--window_size", type=int, default=1024)
    p.add_argument("--max_position_embeddings", type=int, default=4096)
    p.add_argument("--use_fused_kernel", type=str2bool, default=False,
                   help="False = non-fused pure-PyTorch TTT path (default).")
    p.add_argument("--extra_json", type=str, default="{}",
                   help="JSON dict merged into the config dict LAST.")
    # data
    p.add_argument("--data", type=str, default="fineweb",
                   choices=["fineweb", "dna", "music", "clrs", "grid"],
                   help="'fineweb': fineweb-edu char/BPE stream (default, unchanged). "
                        "'dna': hg38 char-level LM (vocab_size=8, chr20 held out for "
                        "val); see dna_data.py. "
                        "'music': Lakh MIDI (LMD-full) REMI symbolic music "
                        "(vocab_size=451, held-out FILES for val); see music_data.py. "
                        "'clrs': CLRS-Text algorithmic traces, char-level, with a "
                        "per-token 2-D address parsed from the serialization "
                        "(Q29 dimensionality ablation); see clrs_data.py.")
    p.add_argument("--grid_rows", type=int, default=32,
                   help="--data grid: grid height R (a column query is R tokens).")
    p.add_argument("--grid_cols", type=int, default=32,
                   help="--data grid: grid width C (a column query gathers at stride C).")
    p.add_argument("--grid_query", type=str, default="col",
                   choices=["col", "row", "mix"],
                   help="--data grid: 'col' = stride-C gather (the hard case that a "
                        "1-D address must hold R distinct offsets for, and a (row,col) "
                        "address holds as one constant axis); 'row' = contiguous "
                        "control; 'mix' = both, chosen per sample.")
    p.add_argument("--clrs_quadrant", type=str, default="2d_long",
                   choices=["2d_long", "1d_long", "2d_short", "1d_short"],
                   help="--data clrs: which (dimensionality x memory-load) cell to "
                        "train on; see clrs_data.QUADRANTS.")
    p.add_argument("--clrs_coord_mode", type=str, default="2d", choices=["2d", "1d"],
                   help="--data clrs: address fed to BOTH rotary sites. '2d' = the "
                        "parsed (outer, inner) coordinate, e.g. (row, col) of the "
                        "adjacency matrix. '1d' = (t, t), which recombines the split "
                        "ladder into inv_freq*t and is therefore BIT-IDENTICAL to the "
                        "stock rotary (verified 0.000e+00 by sanity_clrs_coords.py "
                        "mode a) -- so 2d-vs-1d isolates address DIMENSIONALITY and "
                        "nothing else.")
    p.add_argument("--synthetic", type=str, default="none", choices=["none", "copy"],
                   help="'copy': exact-offset-copy diagnostic task (synthetic_copy.py) "
                        "instead of fineweb-edu; loss/val on the copy region only.")
    p.add_argument("--tokenizer", type=str, default=None,
                   help="Optional preferred tokenizer; falls back through the standard chain.")
    p.add_argument("--seq_len", type=int, default=4096)
    p.add_argument("--data_seed", type=int, default=42)
    p.add_argument("--val_tokens", type=int, default=2_000_000,
                   help="First N packed tokens held out as the fixed val set.")
    # optimization
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=256)
    p.add_argument("--steps", type=int, default=None,
                   help="If unset: token_budget // (bs * seq_len * grad_accum).")
    p.add_argument("--token_budget", type=int, default=2_000_000_000)
    p.add_argument("--bs", type=int, default=24)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--min_lr_ratio", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42, help="Model init / torch seed.")
    # Q24 training-dynamics interventions (perturb the ACQUISITION ORDER early
    # in training, then anneal away; the final architecture is EXACTLY standard
    # hpra). Both require the manual input-rotary path: ttt_input_chunkq >= 1.
    p.add_argument("--input_rope_dropout_p0", type=float, default=0.0,
                   help="Design (a): each sequence independently DROPS its input "
                        "rope (unrotated fast q/k) with prob p = p0 * max(0, 1 - "
                        "step/anneal); the hidden rotary is then its only relative "
                        "code. 0.0 disables. Eval always runs with p=0.")
    p.add_argument("--input_rope_dropout_anneal", type=int, default=30000,
                   help="Design (a): steps over which p anneals linearly to 0.")
    p.add_argument("--input_rope_warmup", type=str, default="none",
                   choices=["none", "cosine_4k_12k"],
                   help="Design (b): hidden-first curriculum. A global scalar "
                        "s(step) multiplies the input-rope ANGLES: s=0 for "
                        "step<4000, cosine ramp 0->1 by step 12000, then 1. Eval "
                        "uses the CURRENT s (the deployable model mid-ramp is the "
                        "current-s model; s anneals to 1 anyway).")
    # logging / io
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--val_every", type=int, default=1000)
    p.add_argument("--val_bs", type=int, default=8)
    p.add_argument("--out_dir", type=str, required=True)
    # checkpointing / resume
    p.add_argument("--save_every", type=int, default=2000,
                   help="Save a full resume checkpoint every N steps (0 disables).")
    p.add_argument("--keep_ckpts", type=int, default=2,
                   help="Keep only the newest N periodic checkpoints.")
    p.add_argument("--auto_resume", type=str2bool, default=True,
                   help="Resume from the newest ckpt_step*.pt in out_dir if present.")
    return p.parse_args()


def build_config(args, vocab_size, tokenizer):
    with open(args.config) as f:
        cfg = json.load(f)
    cfg.pop("model_type", None)

    # small-model overrides (CLI flags)
    cfg.update(dict(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attn_heads=args.num_attn_heads,
        num_lact_heads=args.num_lact_heads,
        lact_chunk_size=args.lact_chunk_size,
        window_size=args.window_size,
        max_position_embeddings=args.max_position_embeddings,
        vocab_size=vocab_size,
        use_fused_kernel=args.use_fused_kernel,
        # fp32_states: left at config/class default on purpose
    ))
    # keep special token ids consistent with the tokenizer actually used
    if tokenizer.bos_token_id is not None:
        cfg["bos_token_id"] = tokenizer.bos_token_id
    if tokenizer.eos_token_id is not None:
        cfg["eos_token_id"] = tokenizer.eos_token_id

    # custom experiment flags merged LAST
    extra = json.loads(args.extra_json)
    if not isinstance(extra, dict):
        raise ValueError("--extra_json must be a JSON object")
    cfg.update(extra)

    return LaCTSWIGLUConfig(**cfg)


def build_optimizer(model, args):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (decay if param.dim() >= 2 else no_decay).append(param)
    groups = [
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95))


def build_scheduler(optimizer, warmup, total_steps, min_lr_ratio):
    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        progress = min(1.0, progress)
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cos
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, val_set, val_bs, device):
    """Mean per-token val loss over the whole cached val set."""
    was_training = model.training
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for i in range(0, val_set.shape[0], val_bs):
        x = val_set[i:i + val_bs].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=x, labels=x)
        n_tok = x.shape[0] * (x.shape[1] - 1)  # last position per row is ignored
        total_loss += out.loss.float().item() * n_tok
        total_tokens += n_tok
    if was_training:
        model.train()
    return total_loss / max(1, total_tokens)


@torch.no_grad()
def evaluate_copy(model, val_set, val_bs, device):
    """Copy-region mean loss + argmax accuracy for --synthetic copy.

    Loss is computed OUTSIDE the model (fp32 cross-entropy on the copy-region
    logits) so it cannot depend on the fused-CE masking path; the model
    forward shifts labels internally (logits at t score the token at t+1), so
    the logits scoring copy positions [COPY_START, COPY_END) live at
    [COPY_START-1, COPY_END-1)."""
    sc = synthetic_copy
    was_training = model.training
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    for i in range(0, val_set.shape[0], val_bs):
        x = val_set[i:i + val_bs].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=x).logits
        pred = logits[:, sc.COPY_START - 1:sc.COPY_END - 1, :].float()
        tgt = x[:, sc.COPY_START:sc.COPY_END]
        total_loss += torch.nn.functional.cross_entropy(
            pred.reshape(-1, pred.shape[-1]), tgt.reshape(-1), reduction="sum").item()
        total_correct += (pred.argmax(-1) == tgt).sum().item()
        total += tgt.numel()
    if was_training:
        model.train()
    return total_loss / max(1, total), total_correct / max(1, total)


# ---- Q24 training-dynamics interventions ---------------------------------
def input_rope_dropout_p(p0, anneal_steps, step):
    """Design (a) dropout prob at this step: p0 * max(0, 1 - step/anneal)."""
    return p0 * max(0.0, 1.0 - step / max(1, anneal_steps))


def input_rope_dropout_keep(data_seed, step, bs, p):
    """Per-sequence keep mask [bs] (1.0 = keep input rope, 0.0 = drop).

    STATELESS reproducibility: a pure function of (data_seed, step) via a
    dedicated CPU Generator — no dependence on global RNG state, so auto-resume
    reproduces the exact same masks. (CPython's hash of an int tuple is
    unsalted, hence stable across processes.)"""
    g = torch.Generator(device="cpu")
    g.manual_seed(hash((int(data_seed), int(step))) & 0x7FFFFFFF)
    u = torch.rand(bs, generator=g)
    return (u >= p).float()


def input_rope_warmup_scale(kind, step):
    """Design (b) global input-rope angle scale s(step)."""
    if kind == "cosine_4k_12k":
        if step < 4000:
            return 0.0
        if step >= 12000:
            return 1.0
        return 0.5 * (1.0 - math.cos(math.pi * (step - 4000) / 8000.0))
    raise ValueError(f"unknown input_rope_warmup: {kind}")


def clrs_split(batch, coord_mode):
    """Split a CLRS block batch [b, s, 4] into (tokens, coords, labels).

    Channels are (token, c_outer, c_inner, loss_mask); see clrs_data.make_block.
    Labels are -100 outside the answer region, so the loss scores only what the
    model must GENERATE -- the same discipline as evaluate_copy's copy region.
    coord_mode '1d' feeds (t, t), which is bit-identical to the stock rotary.
    """
    assert batch.dim() == 3 and batch.shape[-1] == 4, \
        f"--data clrs expects blocks [b, s, 4], got {tuple(batch.shape)}"
    tokens = batch[..., 0].contiguous()
    labels = tokens.masked_fill(batch[..., 3] == 0, -100)
    if coord_mode == "1d":
        t = torch.arange(batch.shape[1], device=batch.device, dtype=torch.float32)
        coords = t[None, :, None].expand(batch.shape[0], batch.shape[1], 2)
    else:
        coords = batch[..., 1:3].float()
    return tokens, coords.contiguous(), labels


def set_ext_coords(ttt_layers, coords):
    """Install the per-token external address on every TTT layer (None = off)."""
    for lyr in ttt_layers:
        lyr._ext_coords = coords


@torch.no_grad()
def evaluate_clrs(model, val_set, val_bs, device, coord_mode, ttt_layers):
    """Answer-region mean loss + teacher-forced argmax accuracy, plus exact-match
    (a whole answer counts only if EVERY supervised token is right).

    Accuracy is the primary metric: a from-scratch 200M char model may sit at 0%
    exact match, which would make an exact-match-only readout indistinguishable
    from 'the wiring is broken'."""
    was_training = model.training
    model.eval()
    tot_loss, tot_correct, tot, exact_ok, exact_n = 0.0, 0, 0, 0, 0
    for i in range(0, val_set.shape[0], val_bs):
        blk = val_set[i:i + val_bs].to(device, non_blocking=True)
        tokens, coords, labels = clrs_split(blk, coord_mode)
        set_ext_coords(ttt_layers, coords)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=tokens).logits
        # the model shifts internally: logits at t score the token at t+1
        pred = logits[:, :-1, :].float()
        tgt = labels[:, 1:]
        sel = tgt != -100
        if not bool(sel.any()):
            continue
        flat_p = pred[sel]
        flat_t = tgt[sel]
        tot_loss += torch.nn.functional.cross_entropy(
            flat_p, flat_t, reduction="sum").item()
        hit = (flat_p.argmax(-1) == flat_t)
        tot_correct += int(hit.sum())
        tot += int(flat_t.numel())
        row_ok = ((pred.argmax(-1) == tgt) | ~sel).all(dim=1)
        exact_ok += int(row_ok.sum())
        exact_n += int(row_ok.numel())
    set_ext_coords(ttt_layers, None)
    if was_training:
        model.train()
    return (tot_loss / max(1, tot), tot_correct / max(1, tot),
            exact_ok / max(1, exact_n))


@torch.no_grad()
def verify_clrs_coords_active(model, val_set, args, device, ttt_layers):
    """Positive startup guard: prove the 2-D address actually reaches the rotary.

    Q29's first grid silently produced bit-identical 2d and 1d results because
    `ttt_layers` was empty, so set_ext_coords looped over nothing. Nothing errored
    -- the arms just ran the stock rotary, which looks like a real (null) result.
    A no-op is indistinguishable from a null unless something asserts otherwise,
    so this runs one forward per coordinate mode on the same batch and REQUIRES
    them to differ whenever a rotary site is enabled (and to match exactly when
    none is, which is the other half of the control).
    """
    assert ttt_layers, "clrs: ttt_layers is empty -- the address can never be applied"
    has_rotary = (not getattr(model.config, "ttt_nope", False)) or \
        bool(getattr(model.config, "ttt_hidden_rope", False))
    blk = val_set[:2].to(device)
    losses = {}
    for mode in ("2d", "1d"):
        tok, coords, labels = clrs_split(blk, mode)
        set_ext_coords(ttt_layers, coords)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            losses[mode] = float(model(input_ids=tok, labels=labels).loss)
    set_ext_coords(ttt_layers, None)
    d = abs(losses["2d"] - losses["1d"])
    tag = f"2d={losses['2d']:.6f} 1d={losses['1d']:.6f} |d|={d:.3e}"
    if has_rotary:
        assert d > 1e-5, (
            f"clrs: COORD INERT ({tag}) on {len(ttt_layers)} layers with a rotary "
            f"enabled -- the 2-D address is not reaching the rotary, so 2d and 1d "
            f"would be the same experiment. Refusing to burn the budget.")
        print(f"[clrs] COORD VERIFIED ACTIVE on {len(ttt_layers)} layers: {tag}",
              flush=True)
    else:
        assert d == 0.0, (
            f"clrs: no rotary is enabled but the address changed the loss ({tag}) "
            f"-- it is leaking through some other path.")
        print(f"[clrs] no rotary (ttt_nope): address correctly inert: {tag}",
              flush=True)


def set_input_rope_scale(ttt_layers, val):
    """val: None (standard hpra) | float (curriculum s) | tensor [b] (dropout
    mask). Plain attribute — never enters state_dict."""
    for lyr in ttt_layers:
        lyr._input_rope_scale = val


def run_validation(model, val_set, args, step, tokens_seen, device, val_log_path,
                   ttt_layers=()):
    t0 = time.time()
    entry = {"step": step}
    if args.data in ("clrs", "grid"):
        val_loss, acc, exact = evaluate_clrs(
            model, val_set, args.val_bs, device, args.clrs_coord_mode, ttt_layers)
        ppl = math.exp(min(20.0, val_loss))
        entry.update({"val_loss": val_loss, "ppl": ppl,
                      "answer_acc": acc, "exact_match": exact})
        print(f"VAL step={step} answer_loss={val_loss:.4f} ppl={ppl:.2f} "
              f"answer_acc={acc:.4f} exact={exact:.4f} "
              f"(eval took {time.time() - t0:.1f}s)", flush=True)
    elif args.synthetic == "copy":
        val_loss, acc = evaluate_copy(model, val_set, args.val_bs, device)
        ppl = math.exp(min(20.0, val_loss))
        entry.update({"val_loss": val_loss, "ppl": ppl, "copy_acc": acc})
        print(f"VAL step={step} copy_loss={val_loss:.4f} ppl={ppl:.2f} "
              f"copy_acc={acc:.4f} (eval took {time.time() - t0:.1f}s)", flush=True)
    else:
        val_loss = evaluate(model, val_set, args.val_bs, device)
        ppl = math.exp(min(20.0, val_loss))
        entry.update({"val_loss": val_loss, "ppl": ppl})
        print(f"VAL step={step} loss={val_loss:.4f} ppl={ppl:.2f} "
              f"(eval took {time.time() - t0:.1f}s)", flush=True)
    entry.update({"tokens_seen": tokens_seen, "time": time.time()})
    with open(val_log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return val_loss


# Args that must match between the checkpoint and the resuming run for the
# resumed run to reproduce an uninterrupted run (data stream + lr schedule).
_RESUME_CRITICAL_ARGS = ("data", "data_seed", "seq_len", "bs", "grad_accum", "val_tokens",
                         "lr", "warmup", "min_lr_ratio", "steps", "token_budget",
                         "synthetic", "input_rope_dropout_p0",
                         "input_rope_dropout_anneal", "input_rope_warmup")
# Defaults for critical args missing from OLD checkpoints (saved before the
# arg existed) so they stay resumable.
_RESUME_ARG_DEFAULTS = {"data": "fineweb", "synthetic": "none",
                        "input_rope_dropout_p0": 0.0,
                        "input_rope_dropout_anneal": 30000,
                        "input_rope_warmup": "none"}


def find_latest_ckpt(out_dir):
    """Newest ckpt_step*.pt in out_dir by step number, or None."""
    best, best_step = None, -1
    for path in glob.glob(os.path.join(out_dir, "ckpt_step*.pt")):
        m = re.fullmatch(r"ckpt_step(\d+)\.pt", os.path.basename(path))
        if m and int(m.group(1)) > best_step:
            best, best_step = path, int(m.group(1))
    return best


def save_checkpoint(args, step, tokens_seen, model, optimizer, scheduler, stream_state):
    """Atomic (tmp+rename) full resume checkpoint; keeps the newest keep_ckpts."""
    t0 = time.time()
    ckpt = {
        "step": step,
        "tokens_seen": tokens_seen,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(),
        "stream": stream_state,  # {"n_raw_consumed", "buf"} from PackedBlockStream
        "args": {k: getattr(args, k) for k in _RESUME_CRITICAL_ARGS},
    }
    path = os.path.join(args.out_dir, f"ckpt_step{step}.pt")
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)
    # rotate: keep only the newest keep_ckpts periodic checkpoints
    ckpts = sorted(
        (p for p in glob.glob(os.path.join(args.out_dir, "ckpt_step*.pt"))
         if re.fullmatch(r"ckpt_step(\d+)\.pt", os.path.basename(p))),
        key=lambda p: int(re.fullmatch(r"ckpt_step(\d+)\.pt", os.path.basename(p)).group(1)),
    )
    for old in ckpts[:-max(1, args.keep_ckpts)]:
        try:
            os.remove(old)
        except OSError:
            pass
    print(f"[ckpt] saved {path} (stream at {stream_state['n_raw_consumed']:,} raw examples, "
          f"{len(stream_state['buf'])} carry-over tokens) in {time.time() - t0:.1f}s", flush=True)


def load_checkpoint(path, args, model, optimizer, scheduler, device):
    """Restore model/optimizer/scheduler/RNG; returns (step, tokens_seen, stream_state)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    for k in _RESUME_CRITICAL_ARGS:
        old = ckpt["args"].get(k, _RESUME_ARG_DEFAULTS.get(k))
        new = getattr(args, k)
        if old != new:
            raise RuntimeError(
                f"--auto_resume arg mismatch: checkpoint has {k}={old!r} but this run "
                f"has {k}={new!r}; resumed run would not reproduce the original stream/schedule. "
                f"Use a fresh --out_dir or --auto_resume false.")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])  # moves state to param devices
    scheduler.load_state_dict(ckpt["scheduler"])
    torch.set_rng_state(ckpt["torch_rng"].cpu())
    torch.cuda.set_rng_state(ckpt["cuda_rng"].cpu(), device=torch.device(device).index or 0)
    print(f"[ckpt] resumed from {path}: step={ckpt['step']} "
          f"tokens_seen={ckpt['tokens_seen']:,}", flush=True)
    return ckpt["step"], ckpt["tokens_seen"], ckpt["stream"]


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda"
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    tokens_per_step = args.bs * args.seq_len * args.grad_accum
    total_steps = args.steps if args.steps is not None else args.token_budget // tokens_per_step
    assert total_steps > 0, "token_budget too small for bs * seq_len * grad_accum"

    # ---- tokenizer -----------------------------------------------------
    if args.data == "dna":
        import dna_data
        tokenizer = dna_data.DnaCharTokenizer()
        tok_name, vocab_size = "hg38", dna_data.VOCAB_SIZE
        print(f"[data] dna char tokenizer (vocab_size={vocab_size})", flush=True)
    elif args.data == "music":
        import music_data
        tokenizer = music_data.MusicRemiTokenizer()
        tok_name, vocab_size = "music", tokenizer.vocab_size
        print(f"[data] music REMI tokenizer (vocab_size={vocab_size}, "
              f"bos={tokenizer.bos_token_id} eos={tokenizer.eos_token_id})", flush=True)
    elif args.data == "grid":
        import synthetic_grid
        tokenizer = synthetic_grid.GridCharTokenizer()
        tok_name, vocab_size = "grid", synthetic_grid.VOCAB_SIZE
        print(f"[data] grid-recall tokenizer (vocab_size={vocab_size})", flush=True)
    elif args.data == "clrs":
        import clrs_data
        tokenizer = clrs_data.ClrsCharTokenizer()
        tok_name, vocab_size = "clrs", clrs_data.VOCAB_SIZE
        print(f"[data] clrs char tokenizer (vocab_size={vocab_size})", flush=True)
    else:
        tokenizer, tok_name, vocab_size = data_utils.load_tokenizer(args.tokenizer)
    eos_id = tokenizer.eos_token_id
    assert eos_id is not None, "tokenizer has no eos token"

    # ---- model ---------------------------------------------------------
    config = build_config(args, vocab_size, tokenizer)
    print(f"[cfg] {config}", flush=True)
    model = LaCTForCausalLM(config).to(device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] param count: {n_params:,} ({n_params / 1e6:.1f}M)", flush=True)

    # Q24 interventions: resolve mode and grab the TTT layers once.
    use_iropedrop = args.input_rope_dropout_p0 > 0.0
    use_iropewarm = args.input_rope_warmup != "none"
    # Populate UNCONDITIONALLY. This used to be filled only inside the Q24 branch
    # below, which silently reduced every consumer to a no-op loop over an empty
    # list -- Q29's first grid ran that way and produced 2d/1d results that were
    # bit-identical because the coordinate never reached a layer.
    ttt_layers = [blk.attn for blk in model.model.layers]
    if use_iropedrop or use_iropewarm:
        assert not (use_iropedrop and use_iropewarm), \
            "pick ONE intervention: --input_rope_dropout_p0 OR --input_rope_warmup"
        assert getattr(config, "ttt_input_chunkq", 0) > 0, \
            "Q24 interventions scale angles in the MANUAL input-rotary path; " \
            "add \"ttt_input_chunkq\": 1 to --extra_json"
        assert not getattr(config, "ttt_nope", False), \
            "Q24 interventions modulate the INPUT rope; ttt_nope must be False"
        print(f"[q24] intervention="
              f"{'input_rope_dropout' if use_iropedrop else args.input_rope_warmup} "
              f"p0={args.input_rope_dropout_p0} "
              f"anneal={args.input_rope_dropout_anneal} "
              f"layers={len(ttt_layers)}", flush=True)

    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args.warmup, total_steps, args.min_lr_ratio)

    # ---- auto-resume -----------------------------------------------------
    start_step, start_tokens, resume_stream_state = 0, 0, None
    resume_path = find_latest_ckpt(args.out_dir) if args.auto_resume else None
    if resume_path is not None:
        start_step, start_tokens, resume_stream_state = load_checkpoint(
            resume_path, args, model, optimizer, scheduler, device)

    # ---- data ----------------------------------------------------------
    if args.synthetic == "copy":
        # Synthetic exact-offset-copy task: pure function of (data_seed,
        # sample index) -> no val cache needed (rebuilt identically each run),
        # val indices disjoint from the training stream by construction.
        block_gen = synthetic_copy.SyntheticCopyStream(args.data_seed, args.seq_len)
        if resume_stream_state is not None:
            block_gen.restore(resume_stream_state)
        val_set = synthetic_copy.build_val_set(args.data_seed, n_seqs=64,
                                               seq_len=args.seq_len)
        print(f"[data] synthetic copy task: val set {tuple(val_set.shape)}, "
              f"copy region [{synthetic_copy.COPY_START}, {synthetic_copy.COPY_END}) "
              f"offset {synthetic_copy.COPY_OFFSET}", flush=True)
    elif args.data == "dna":
        # hg38 char-LM: contiguous seq_len blocks in a data_seed-shuffled order.
        # val (chr20) is held out entirely, so it is built independently of the
        # training stream position (no first-n-blocks consumption like fineweb).
        import dna_data
        block_gen = dna_data.DnaBlockStream(
            dna_data.ensure_train_blocks(args.seq_len), args.data_seed, args.seq_len)
        n_val_blocks = args.val_tokens // args.seq_len
        val_cache = dna_data.val_cache_path(args.seq_len)
        val_set = dna_data.get_or_build_dna_val_set(
            n_val_blocks, val_cache, args.seq_len)
        if resume_stream_state is not None:
            block_gen.restore(resume_stream_state)
        print(f"[data] dna hg38: {block_gen.N:,} train blocks, val set "
              f"{tuple(val_set.shape)} (chr20)", flush=True)
    elif args.data == "music":
        # LMD-full REMI symbolic music: contiguous seq_len blocks (pieces joined
        # by EOS) in a data_seed-shuffled order. val files are held out entirely,
        # so the val set is built independently of the training stream position.
        import music_data
        block_gen = music_data.MusicBlockStream(
            music_data.ensure_train_blocks(args.seq_len), args.data_seed, args.seq_len)
        n_val_blocks = args.val_tokens // args.seq_len
        val_set = music_data.get_or_build_music_val_set(
            n_val_blocks, music_data.val_cache_path(args.seq_len), args.seq_len)
        if resume_stream_state is not None:
            block_gen.restore(resume_stream_state)
        print(f"[data] music LMD-full REMI: {block_gen.N:,} train blocks, val set "
              f"{tuple(val_set.shape)} (held-out files)", flush=True)
    elif args.data == "grid":
        # Grid recall: deterministic in (data_seed, index), so no cache is needed
        # and the val indices are disjoint from training by construction.
        import synthetic_grid
        block_gen = synthetic_grid.GridStream(
            args.data_seed, args.seq_len, args.grid_rows, args.grid_cols,
            args.grid_query)
        if resume_stream_state is not None:
            block_gen.restore(resume_stream_state)
        val_set = synthetic_grid.build_val_set(
            args.data_seed, 64, args.seq_len, args.grid_rows, args.grid_cols,
            args.grid_query)
        print(f"[data] grid recall {args.grid_rows}x{args.grid_cols} "
              f"query={args.grid_query}: val set {tuple(val_set.shape)}", flush=True)
    elif args.data == "clrs":
        # CLRS-Text algorithmic traces: one problem per block, carrying a
        # per-token 2-D address. val comes from the HELD-OUT test repo, so it is
        # independent of the training stream position.
        import clrs_data
        block_gen = clrs_data.ClrsBlockStream(
            clrs_data.ensure_train_blocks(args.clrs_quadrant, args.seq_len),
            args.data_seed, args.seq_len)
        n_val_blocks = max(1, args.val_tokens // args.seq_len)
        val_set = clrs_data.get_or_build_clrs_val_set(
            n_val_blocks, clrs_data.val_cache_path(args.clrs_quadrant, args.seq_len),
            args.clrs_quadrant, args.seq_len)
        if resume_stream_state is not None:
            block_gen.restore(resume_stream_state)
        print(f"[data] clrs {args.clrs_quadrant} coord_mode={args.clrs_coord_mode}: "
              f"{block_gen.N:,} train blocks, val set {tuple(val_set.shape)} "
              f"(held-out test seeds)", flush=True)
    else:
        # Identical shuffled stream for every run with the same data_seed.
        stream = data_utils.build_shuffled_stream(args.data_seed, buffer_size=10000)
        block_gen = data_utils.PackedBlockStream(stream, tokenizer, args.seq_len, eos_id)

        n_val_blocks = args.val_tokens // args.seq_len
        # data_seed in the filename: the val set is the head of the seed's stream,
        # so caches from different seeds must never share a file (a seed-43 run
        # once clobbered the seed-42 cache through the mismatch-overwrite guard).
        val_cache = os.path.join(SCRIPT_DIR,
                                 f"val_cache_{tok_name.replace('/', '_')}_{args.seq_len}_ds{args.data_seed}.pt")
        if resume_stream_state is None:
            val_set = data_utils.get_or_build_val_set(block_gen, n_val_blocks, val_cache)
        else:
            # The saved stream position already accounts for the val-set blocks
            # (they are the head of the stream), so do NOT consume them again.
            if os.path.exists(val_cache):
                val_set = torch.load(val_cache, map_location="cpu")
                print(f"[data] resume: reusing cached val set {val_cache} "
                      f"({val_set.numel()} tokens)", flush=True)
            else:
                # cache lost: rebuild from a throwaway fresh stream (same head)
                tmp_gen = data_utils.PackedBlockStream(
                    data_utils.build_shuffled_stream(args.data_seed, buffer_size=10000),
                    tokenizer, args.seq_len, eos_id)
                val_set = data_utils.get_or_build_val_set(tmp_gen, n_val_blocks, val_cache)
                del tmp_gen
            # fast-forward the training stream to the exact checkpointed position
            block_gen.restore(resume_stream_state)

    batches = data_utils.batch_generator(block_gen, args.bs, with_state=True)

    val_log_path = os.path.join(args.out_dir, "val_log.jsonl")
    print(f"[train] steps={total_steps} bs={args.bs} grad_accum={args.grad_accum} "
          f"seq_len={args.seq_len} tokens/step={tokens_per_step} "
          f"token_budget~{total_steps * tokens_per_step:,}", flush=True)

    # Debug: LLM_BATCH_FP=1 prints a data fingerprint (token-id sum) for the
    # batches of every 100th step — used by the crash-resume gold test.
    batch_fp = str2bool(os.environ.get("LLM_BATCH_FP", "0"))

    if args.data in ("clrs", "grid"):
        verify_clrs_coords_active(model, val_set, args, device, ttt_layers)

    # ---- training loop -------------------------------------------------
    step = start_step
    tokens_seen = start_tokens
    running_loss, running_count = 0.0, 0
    t_last = time.time()
    tokens_last = tokens_seen
    exhausted = False
    last_stream_state = resume_stream_state  # position after the last consumed batch

    while step < total_steps and not exhausted:
        # ---- Q24 intervention schedule for THIS optimizer step (`step` is the
        # 0-based index of the upcoming step; pure function of it -> resume-safe).
        keep_mask, p_drop, s_cur = None, 0.0, 1.0
        if use_iropedrop:
            p_drop = input_rope_dropout_p(
                args.input_rope_dropout_p0, args.input_rope_dropout_anneal, step)
            if p_drop > 0.0:
                keep_mask = input_rope_dropout_keep(
                    args.data_seed, step, args.bs, p_drop)
            set_input_rope_scale(ttt_layers, None)  # set per-micro below
        elif use_iropewarm:
            s_cur = input_rope_warmup_scale(args.input_rope_warmup, step)
            # s == 1.0 -> None: pristine (unscaled) code path, same math.
            set_input_rope_scale(ttt_layers, None if s_cur >= 1.0 else float(s_cur))

        optimizer.zero_grad(set_to_none=True)
        micro_losses = []
        for micro in range(args.grad_accum):
            try:
                x, last_stream_state = next(batches)
            except StopIteration:
                exhausted = True
                break
            if keep_mask is not None:
                # slice: the stream's last batch can be short; grad_accum>1
                # reuses the same per-step mask across micro-batches.
                set_input_rope_scale(
                    ttt_layers, keep_mask[:x.shape[0]].to(device))
            if batch_fp and (step + 1) % 100 == 0:
                print(f"[fp] step={step + 1} micro={micro} tok_sum={int(x.sum().item())}",
                      flush=True)
            x = x.to(device, non_blocking=True)
            # copy task: supervise ONLY the copy region (-100 elsewhere);
            # clrs: supervise the answer region and install the 2-D address;
            # otherwise plain LM (labels = inputs, shifted inside the model).
            if args.data in ("clrs", "grid"):
                x, _coords, labels = clrs_split(x, args.clrs_coord_mode)
                set_ext_coords(ttt_layers, _coords)
            elif args.synthetic == "copy":
                labels = synthetic_copy.make_labels(x)
            else:
                labels = x
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=x, labels=labels).loss
            (loss / args.grad_accum).backward()
            micro_losses.append(loss.float().item())
            tokens_seen += x.numel()
        if not micro_losses:
            break
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        step += 1

        step_loss = sum(micro_losses) / len(micro_losses)
        if not math.isfinite(step_loss):
            print(f"[train] WARNING: non-finite loss at step {step}: {step_loss}", flush=True)
        running_loss += step_loss
        running_count += 1

        if step % args.log_every == 0:
            dt = time.time() - t_last
            tps = (tokens_seen - tokens_last) / max(1e-9, dt)
            q24_log = ""
            if use_iropedrop:
                q24_log = f" ropedrop_p={p_drop:.3f}"
            elif use_iropewarm:
                q24_log = f" irope_s={s_cur:.3f}"
            print(f"step={step} loss={running_loss / running_count:.4f} "
                  f"tokens/sec={tps:,.0f} lr={scheduler.get_last_lr()[0]:.3e} "
                  f"tokens_seen={tokens_seen:,}{q24_log}", flush=True)
            running_loss, running_count = 0.0, 0
            t_last = time.time()
            tokens_last = tokens_seen

        if step % args.val_every == 0:
            # Design (a): eval = standard hpra (p=0, no mask). Design (b): keep
            # the CURRENT s — mid-ramp the deployable model is the current-s
            # model (and s anneals to 1 anyway); the next step re-sets it.
            if use_iropedrop:
                set_input_rope_scale(ttt_layers, None)
            run_validation(model, val_set, args, step, tokens_seen, device, val_log_path,
                           ttt_layers)
            t_last = time.time()  # don't count eval time in tokens/sec
            tokens_last = tokens_seen

        # Periodic resume checkpoint (after val, so a resumed run continues at
        # the next val point; vals between the last ckpt and a crash re-run and
        # append duplicate-step entries to val_log.jsonl — accepted tradeoff).
        if args.save_every > 0 and step % args.save_every == 0 and last_stream_state is not None:
            save_checkpoint(args, step, tokens_seen, model, optimizer, scheduler,
                            last_stream_state)
            t_last = time.time()  # don't count ckpt time in tokens/sec
            tokens_last = tokens_seen

    # ---- final val + checkpoint ----------------------------------------
    if use_iropedrop:
        set_input_rope_scale(ttt_layers, None)  # eval/final = standard hpra
    if step % args.val_every != 0 or step == 0:
        run_validation(model, val_set, args, step, tokens_seen, device, val_log_path,
                           ttt_layers)
    ckpt_path = os.path.join(args.out_dir, "final.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"[train] done at step {step} ({tokens_seen:,} tokens); "
          f"saved model state_dict to {ckpt_path}", flush=True)

    # The hf datasets streaming stack leaves ~100+ live threads that prevent a
    # clean interpreter shutdown (observed: process lingers after "done",
    # holding all GPU memory). Everything is saved at this point, so exit hard.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
