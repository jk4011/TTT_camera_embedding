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


def set_input_rope_scale(ttt_layers, val):
    """val: None (standard hpra) | float (curriculum s) | tensor [b] (dropout
    mask). Plain attribute — never enters state_dict."""
    for lyr in ttt_layers:
        lyr._input_rope_scale = val


def run_validation(model, val_set, args, step, tokens_seen, device, val_log_path):
    t0 = time.time()
    entry = {"step": step}
    if args.synthetic == "copy":
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
_RESUME_CRITICAL_ARGS = ("data_seed", "seq_len", "bs", "grad_accum", "val_tokens",
                         "lr", "warmup", "min_lr_ratio", "steps", "token_budget",
                         "synthetic", "input_rope_dropout_p0",
                         "input_rope_dropout_anneal", "input_rope_warmup")
# Defaults for critical args missing from OLD checkpoints (saved before the
# arg existed) so they stay resumable.
_RESUME_ARG_DEFAULTS = {"synthetic": "none", "input_rope_dropout_p0": 0.0,
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
    ttt_layers = []
    if use_iropedrop or use_iropewarm:
        assert not (use_iropedrop and use_iropewarm), \
            "pick ONE intervention: --input_rope_dropout_p0 OR --input_rope_warmup"
        assert getattr(config, "ttt_input_chunkq", 0) > 0, \
            "Q24 interventions scale angles in the MANUAL input-rotary path; " \
            "add \"ttt_input_chunkq\": 1 to --extra_json"
        assert not getattr(config, "ttt_nope", False), \
            "Q24 interventions modulate the INPUT rope; ttt_nope must be False"
        ttt_layers = [blk.attn for blk in model.model.layers]
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
            # otherwise plain LM (labels = inputs, shifted inside the model).
            labels = synthetic_copy.make_labels(x) if args.synthetic == "copy" else x
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
            run_validation(model, val_set, args, step, tokens_seen, device, val_log_path)
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
        run_validation(model, val_set, args, step, tokens_seen, device, val_log_path)
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
