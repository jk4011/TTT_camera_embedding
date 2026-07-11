"""Q11 training: ReCamMaster-frozen + TTT-adapter, single GPU, bs1.

Loads the official ReCamMaster pipeline (released step20000.ckpt, strict=True),
patches every DiT block with per-video attention + a trainable TTT fast-weight
branch (recam_ttt.ttt_adapter), freezes everything else, and trains the
branches with the OFFICIAL ReCamMaster training-step math (copied verbatim
from train_recammaster.py: shared uniform timestep, noise on the TGT half
only, flow-matching target, MSE on the TGT half, scheduler training weight).

Determinism: torch.manual_seed(loss_seed_base + step) immediately before the
timestep+noise draws, and a seeded epoch-reshuffled pair stream — so all 4
variants see IDENTICAL data + noise (paired analysis), and resume replays the
shuffle exactly (no timestamp seeds).

Usage (envs/recam, from lact_ar_video, PYTHONPATH=.:<ReCamMaster clone>):
  python recam_ttt/train_recam_ttt.py --config recam_ttt/configs/in.yaml \
      --out outputs/recam_ttt/in --max_steps 9000 --save_every 250
"""
import argparse
import glob
import json
import math
import os
import re
import sys
import time

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "minVid"))

from recam_ttt.ttt_adapter import (  # noqa: E402
    build_recam_coords6,
    freeze_all_but_ttt,
    patch_dit,
    set_ctx_phases,
)

HOLDOUT_JSON = os.path.join(_HERE, "..", "minVid", "configs", "ar",
                            "ccv_holdout_pairs_64.json")


# ---------------------------------------------------------------------------
# data plumbing
# ---------------------------------------------------------------------------

def latent_path(cache_dir, rel, cam):
    return os.path.join(cache_dir, f"{rel.replace(os.sep, '_')}_cam{cam:02d}.pt")


def wait_for_file(path, poll_s=60, max_wait_s=4 * 3600):
    """The latent cache may still be filling (precompute shards on other
    GPUs); wait for a missing file instead of crashing."""
    waited = 0
    while not os.path.exists(path):
        if waited == 0:
            print(f"[train] waiting for latent cache file: {path}", flush=True)
        time.sleep(poll_s)
        waited += poll_s
        if waited >= max_wait_s:
            raise FileNotFoundError(f"gave up waiting for {path}")
    return path


def load_pair_latents(cache_dir, rel, src_cam, tgt_cam, device):
    """[1, 16, 42, 60, 104] bf16: concat([TARGET 21f, SOURCE 21f]) — TGT first
    (ReCamMaster training layout)."""
    tgt = torch.load(wait_for_file(latent_path(cache_dir, rel, tgt_cam)),
                     weights_only=True, map_location="cpu")["latent"]
    src = torch.load(wait_for_file(latent_path(cache_dir, rel, src_cam)),
                     weights_only=True, map_location="cpu")["latent"]
    lat = torch.cat([tgt, src], dim=1)[None]
    return lat.to(device=device, dtype=torch.bfloat16)


class PairStream:
    """Deterministic epoch-reshuffled stream over ds.pairs (seed data_seed).
    pair_for_step(step) is a pure function of step — resume just recomputes
    the permutations (replaying the shuffle; no timestamp seeds)."""

    def __init__(self, pairs, seed):
        self.pairs = pairs
        self.seed = int(seed)
        self._perms = {}
        self._gen_epochs = 0
        self._gen = torch.Generator()
        self._gen.manual_seed(self.seed)

    def _perm(self, epoch):
        while self._gen_epochs <= epoch:
            self._perms[self._gen_epochs] = torch.randperm(
                len(self.pairs), generator=self._gen)
            self._gen_epochs += 1
        return self._perms[epoch]

    def pair_for_step(self, step):
        n = len(self.pairs)
        epoch, idx = divmod(step, n)
        return self.pairs[self._perm(epoch)[idx].item()]


def build_pair_stream(cfg):
    from minVid.data.multicam_pair_dataset import build_pair_index
    pairs = build_pair_index(cfg.data_root, int(cfg.num_pairs),
                             int(cfg.index_seed))
    return PairStream(pairs, cfg.data_seed)


def load_prompt_context(cache_dir, device):
    payload = torch.load(os.path.join(cache_dir, "prompt_emb.pt"),
                         weights_only=True, map_location="cpu")
    ctx = payload["prompt_emb"]["context"]
    if isinstance(ctx, (list, tuple)):
        ctx = ctx[0]
    if ctx.dim() == 2:
        ctx = ctx[None]
    return ctx.to(device=device, dtype=torch.bfloat16)


def cam_inputs_for_pair(cfg, rel, src_cam, tgt_cam, need_coords, device):
    """(cam_emb [1,21,12] bf16, coords6 [L,6] fp32 or None) — official
    conventions: cam_emb = tgt trajectory rel. to the SOURCE (condition)
    camera's frame-0; Plucker coords in the same anchor."""
    from eval_recam_anchor import build_cam_embedding
    cam_json = os.path.join(cfg.data_root, rel, "cameras",
                            "camera_extrinsics.json")
    cam_emb = build_cam_embedding(cam_json, cond_cam=src_cam,
                                  tgt_cam=tgt_cam)[None].to(device)
    coords6 = None
    if need_coords:
        coords6 = build_recam_coords6(cam_json, src_cam, tgt_cam, rel,
                                      device=device)
    return cam_emb, coords6


# ---------------------------------------------------------------------------
# model + loss
# ---------------------------------------------------------------------------

def build_patched_pipe(cfg, device):
    """Official pipeline (ckpt loads strict=True inside), then patch blocks,
    then freeze. Text encoder / VAE are moved off-GPU (latents + prompt are
    cached)."""
    from eval_recam_anchor import build_recam_pipeline
    pipe = build_recam_pipeline(device=device)
    pipe.text_encoder.to("cpu")
    pipe.vae.to("cpu")
    torch.cuda.empty_cache()
    pipe.scheduler.set_timesteps(1000, training=True)

    ttt_ctx = patch_dit(pipe.dit, cfg, device=device)
    trainable, n_train, n_frozen = freeze_all_but_ttt(pipe.dit)
    pipe.dit.train()
    print(f"[train] variant={cfg.variant}  trainable={n_train/1e6:.1f}M  "
          f"frozen={n_frozen/1e6:.1f}M", flush=True)
    return pipe, ttt_ctx, trainable


def training_loss(pipe, latents, cam_emb, context, step_seed,
                  use_gradient_checkpointing=True):
    """Verbatim port of ReCamMaster train_recammaster.py training_step math.
    step_seed fixes the timestep + noise draws (identical across variants)."""
    scheduler = pipe.scheduler
    torch.manual_seed(step_seed)
    noise = torch.randn_like(latents)
    timestep_id = torch.randint(0, scheduler.num_train_timesteps, (1,))
    timestep = scheduler.timesteps[timestep_id].to(
        dtype=torch.bfloat16, device=latents.device)

    origin_latents = latents.clone()
    noisy_latents = scheduler.add_noise(latents, noise, timestep)
    tgt_latent_len = noisy_latents.shape[2] // 2
    # the SECOND half (source video) stays clean
    noisy_latents[:, :, tgt_latent_len:, ...] = \
        origin_latents[:, :, tgt_latent_len:, ...]
    training_target = scheduler.training_target(latents, noise, timestep)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        noise_pred = pipe.dit(
            noisy_latents, timestep=timestep, cam_emb=cam_emb,
            context=context,
            use_gradient_checkpointing=use_gradient_checkpointing)
    loss = F.mse_loss(noise_pred[:, :, :tgt_latent_len, ...].float(),
                      training_target[:, :, :tgt_latent_len, ...].float())
    loss = loss * scheduler.training_weight(timestep)
    return loss


@torch.no_grad()
def probe_val(pipe, ttt_ctx, cfg, context, device):
    """Deterministic probe: first probe_pairs holdout pairs, one fixed
    timestep (id probe_timestep_id) + fixed per-pair noise seed. Returns the
    mean UNWEIGHTED MSE on the target half (the fixed timestep makes the
    training weight a shared constant)."""
    with open(HOLDOUT_JSON) as f:
        holdout = json.load(f)["pairs"][: int(cfg.probe_pairs)]
    scheduler = pipe.scheduler
    # keep the timestep 1-D ([1], like the training path) — the DiT's
    # sinusoidal_embedding_1d rejects 0-D input
    timestep = scheduler.timesteps[[int(cfg.probe_timestep_id)]].to(
        dtype=torch.bfloat16, device=device)
    need_coords = bool(cfg.ttt_input_rope) or bool(cfg.ttt_hidden_rope)

    was_training = pipe.dit.training
    pipe.dit.eval()
    losses = []
    for i, p in enumerate(holdout):
        rel, src_cam, tgt_cam = p["relpath"], int(p["src_cam"]), int(p["tgt_cam"])
        latents = load_pair_latents(cfg.latent_cache, rel, src_cam, tgt_cam,
                                    device)
        cam_emb, coords6 = cam_inputs_for_pair(cfg, rel, src_cam, tgt_cam,
                                               need_coords, device)
        set_ctx_phases(ttt_ctx, coords6, pipe.dit.blocks[0].ttt_branch)

        g = torch.Generator()
        g.manual_seed(int(cfg.probe_noise_seed_base) + i)
        noise = torch.randn(latents.shape, generator=g, dtype=torch.float32)
        noise = noise.to(device=device, dtype=latents.dtype)

        noisy = scheduler.add_noise(latents, noise, timestep)
        tgt_len = noisy.shape[2] // 2
        noisy[:, :, tgt_len:, ...] = latents[:, :, tgt_len:, ...]
        target = scheduler.training_target(latents, noise, timestep)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = pipe.dit(noisy, timestep=timestep, cam_emb=cam_emb,
                            context=context, use_gradient_checkpointing=False)
        losses.append(F.mse_loss(pred[:, :, :tgt_len, ...].float(),
                                 target[:, :, :tgt_len, ...].float()).item())
    if was_training:
        pipe.dit.train()
    return sum(losses) / len(losses)


# ---------------------------------------------------------------------------
# checkpointing (user rule: periodic checkpointing on every training run)
# ---------------------------------------------------------------------------

def ckpt_path(out_dir, step):
    return os.path.join(out_dir, f"ckpt_step{step}.pt")


def save_ckpt(out_dir, step, dit, optimizer, keep_last=3, keep_mult=2000):
    state = {
        "step": step,  # number of COMPLETED optimizer steps
        "model_trainable": {n: p.detach().cpu()
                            for n, p in dit.named_parameters()
                            if p.requires_grad},
        "optimizer": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(),
        "data_stream": {"next_step": step},
    }
    dst = ckpt_path(out_dir, step)
    tmp = dst + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, dst)
    print(f"[train] saved {dst}", flush=True)
    # prune: keep the newest keep_last + every keep_mult-th
    steps = sorted(
        int(re.search(r"ckpt_step(\d+)\.pt$", f).group(1))
        for f in glob.glob(os.path.join(out_dir, "ckpt_step*.pt")))
    for s in steps[:-keep_last]:
        if keep_mult and s % keep_mult == 0:
            continue
        os.remove(ckpt_path(out_dir, s))


def latest_ckpt(out_dir):
    cands = glob.glob(os.path.join(out_dir, "ckpt_step*.pt"))
    if not cands:
        return None
    return max(cands, key=lambda f: int(re.search(r"ckpt_step(\d+)\.pt$", f)
                                        .group(1)))


def load_ckpt(path, dit, optimizer):
    state = torch.load(path, weights_only=True, map_location="cpu")
    named = dict(dit.named_parameters())
    loaded = 0
    for n, t in state["model_trainable"].items():
        p = named[n]
        assert p.requires_grad, n
        p.data.copy_(t.to(p.device))
        loaded += 1
    n_trainable = sum(1 for p in dit.parameters() if p.requires_grad)
    assert loaded == n_trainable, (loaded, n_trainable)
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state(state["cuda_rng"].cpu())
    print(f"[train] resumed from {path} (completed steps: {state['step']})",
          flush=True)
    return int(state["step"])


# ---------------------------------------------------------------------------
# schedule
# ---------------------------------------------------------------------------

def lr_at(step, cfg, max_steps):
    """0-based step; linear warmup then cosine to final_lr_frac * peak."""
    peak = float(cfg.lr)
    warmup = int(cfg.warmup_steps)
    final = peak * float(cfg.final_lr_frac)
    if step < warmup:
        return peak * (step + 1) / warmup
    t = (step - warmup) / max(1, max_steps - warmup)
    return final + 0.5 * (peak - final) * (1.0 + math.cos(math.pi * t))


# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------

def run_training(args):
    cfg = OmegaConf.load(args.config)
    device = args.device
    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "train_log.jsonl")

    pipe, ttt_ctx, trainable = build_patched_pipe(cfg, device)
    optimizer = torch.optim.AdamW(
        trainable, lr=float(cfg.lr), betas=tuple(cfg.betas),
        weight_decay=float(cfg.weight_decay))

    start_step = 0
    if args.auto_resume:
        ck = latest_ckpt(args.out)
        if ck is not None:
            start_step = load_ckpt(ck, pipe.dit, optimizer)
    if start_step >= args.max_steps:
        print(f"[train] nothing to do (completed {start_step} >= "
              f"max_steps {args.max_steps})", flush=True)
        return

    stream = build_pair_stream(cfg)
    context = load_prompt_context(cfg.latent_cache, device)
    need_coords = bool(cfg.ttt_input_rope) or bool(cfg.ttt_hidden_rope)

    grad_checked = False
    torch.cuda.reset_peak_memory_stats()
    for step in range(start_step, args.max_steps):
        tic = time.time()
        _, rel, src_cam, tgt_cam = stream.pair_for_step(step)
        latents = load_pair_latents(cfg.latent_cache, rel, src_cam, tgt_cam,
                                    device)
        cam_emb, coords6 = cam_inputs_for_pair(cfg, rel, src_cam, tgt_cam,
                                               need_coords, device)
        set_ctx_phases(ttt_ctx, coords6, pipe.dit.blocks[0].ttt_branch)

        lr_now = lr_at(step, cfg, args.max_steps)
        for grp in optimizer.param_groups:
            grp["lr"] = lr_now

        loss = training_loss(pipe, latents, cam_emb, context,
                             step_seed=int(cfg.loss_seed_base) + step)
        assert torch.isfinite(loss), f"non-finite loss at step {step}"
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if not grad_checked:  # one-time: grads land ONLY on TTT params
            grad_checked = True
            for n, p in pipe.dit.named_parameters():
                if not p.requires_grad:
                    assert p.grad is None, f"frozen param got grad: {n}"
            nz = sum(1 for p in trainable
                     if p.grad is not None and p.grad.abs().sum() > 0)
            print(f"[train] grad check OK: {nz} trainable tensors with "
                  f"nonzero grad, frozen params untouched", flush=True)
            assert nz > 0

        torch.nn.utils.clip_grad_norm_(trainable, float(cfg.grad_clip))
        optimizer.step()

        completed = step + 1
        sec = time.time() - tic
        if completed % args.log_every == 0 or completed == args.max_steps:
            rec = {"step": completed, "loss": round(loss.item(), 6),
                   "lr": lr_now, "sec_per_step": round(sec, 3),
                   "pair": f"{rel}:{src_cam}->{tgt_cam}",
                   "peak_vram_gb": round(
                       torch.cuda.max_memory_allocated() / 2**30, 2)}
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[train] {rec}", flush=True)

        if completed % int(cfg.probe_every) == 0:
            pv = probe_val(pipe, ttt_ctx, cfg, context, device)
            with open(log_path, "a") as f:
                f.write(json.dumps({"step": completed, "probe_val": pv}) + "\n")
            print(f"[train] step {completed} probe_val {pv:.6f}", flush=True)

        if completed % args.save_every == 0 or completed == args.max_steps:
            save_ckpt(args.out, completed, pipe.dit, optimizer)

    print("[train] DONE", flush=True)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Q11 recam_ttt training")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_steps", type=int, required=True)
    ap.add_argument("--save_every", type=int, default=250)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--auto_resume", type=int, default=1)
    return ap.parse_args(argv)


if __name__ == "__main__":
    run_training(parse_args())
