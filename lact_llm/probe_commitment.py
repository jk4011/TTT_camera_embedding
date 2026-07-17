"""Q24 commitment probe: does a trained hpra model actually USE its hidden rotation?

For a run dir, evaluate every ckpt_step*.pt (+ final.pt) TWICE on the fixed ds42
val cache: once normal, once with the hidden rotation zeroed (angles scaled by
0.0 via the plain `_hrope_scale` attribute; cos(0)=1/sin(0)=0 is an exact
identity, so "zeroed" = the rotation surgically removed, F27b-style).

  C(step) = loss(hidden zeroed) - loss(normal)

This is the kill-switch instrument for the Q24 training-dynamics interventions:
C(final) > 0.1 = the final weights depend on the hidden channel (success
signature); C(final) ~ 0.003 = the model ignored the channel again (F27b).

Eval-only; never touches weights or state_dict (plain attribute surgery, same
pattern as surgery_chunkq.py).

Usage:
  python probe_commitment.py --run_dir outputs/q17_hpra_g1_w128
  python probe_commitment.py --run_dir outputs/q24a_hpra_iropedrop_w128 --ckpts final
"""
import argparse
import glob
import json
import math
import os
import re

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from lact_model.configuration_lact_swiglu import LaCTSWIGLUConfig  # noqa: E402
from lact_model.modeling_lact import LaCTForCausalLM  # noqa: E402
from lact_model.layer_lact_swiglu import LaCTSWIGLULayer  # noqa: E402


def config_from_train_log(log_path):
    """Recover the exact LaCTSWIGLUConfig from the '[cfg] ...' block in train.log."""
    with open(log_path) as f:
        text = f.read()
    m = re.search(r"\[cfg\] LaCTSWIGLUConfig (\{.*?\n\})", text, re.S)
    assert m, f"no [cfg] block in {log_path}"
    cfg = json.loads(m.group(1))
    for k in ("model_type", "transformers_version", "architectures"):
        cfg.pop(k, None)
    return LaCTSWIGLUConfig(**cfg)


def set_hrope_scale(model, val):
    """val: None = normal; 0.0 = hidden rotation removed. Returns #layers touched."""
    n = 0
    for mod in model.modules():
        if isinstance(mod, LaCTSWIGLULayer) and getattr(mod, "ttt_hidden_rope", False):
            mod._hrope_scale = val
            n += 1
    return n


@torch.no_grad()
def evaluate(model, val_set, val_bs, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for i in range(0, val_set.shape[0], val_bs):
        x = val_set[i:i + val_bs].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=x, labels=x)
        n_tok = x.shape[0] * (x.shape[1] - 1)
        total_loss += out.loss.float().item() * n_tok
        total_tokens += n_tok
    return total_loss / max(1, total_tokens)


def list_ckpts(run_dir, which):
    """[(step_or_None, path)] sorted by step; final.pt last (step None)."""
    if which != "all":
        out = []
        for name in which.split(","):
            name = name.strip()
            if name == "final":
                name = "final.pt"
            m = re.fullmatch(r"ckpt_step(\d+)\.pt", name)
            out.append((int(m.group(1)) if m else None, os.path.join(run_dir, name)))
        return out
    ckpts = []
    for path in glob.glob(os.path.join(run_dir, "ckpt_step*.pt")):
        m = re.fullmatch(r"ckpt_step(\d+)\.pt", os.path.basename(path))
        if m:
            ckpts.append((int(m.group(1)), path))
    ckpts.sort()
    final = os.path.join(run_dir, "final.pt")
    if os.path.exists(final):
        ckpts.append((None, final))
    return ckpts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--ckpts", type=str, default="all",
                   help='"all" (every ckpt_step*.pt + final.pt) or a comma list '
                        'of file names ("final" = final.pt).')
    p.add_argument("--val_cache", type=str, default=os.path.join(
        SCRIPT_DIR, "val_cache_fla-hub_transformer-1.3B-100B_4096_ds42.pt"))
    p.add_argument("--val_bs", type=int, default=8)
    args = p.parse_args()

    device = "cuda"
    config = config_from_train_log(os.path.join(args.run_dir, "train.log"))
    assert getattr(config, "ttt_hidden_rope", False), \
        f"{args.run_dir} has no hidden rotary; nothing to probe"
    model = LaCTForCausalLM(config).to(device)

    val_set = torch.load(args.val_cache, map_location="cpu")
    ckpts = list_ckpts(args.run_dir, args.ckpts)
    assert ckpts, f"no checkpoints found in {args.run_dir}"
    print(f"[probe] run={args.run_dir} val={tuple(val_set.shape)} "
          f"ckpts={[os.path.basename(c[1]) for c in ckpts]}", flush=True)

    for step, path in ckpts:
        sd = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "model" in sd and "step" in sd:
            step = sd["step"]  # full resume checkpoint
            sd = sd["model"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert not missing and not unexpected, (missing, unexpected)

        n = set_hrope_scale(model, None)
        loss_normal = evaluate(model, val_set, args.val_bs, device)
        set_hrope_scale(model, 0.0)
        loss_zeroed = evaluate(model, val_set, args.val_bs, device)
        set_hrope_scale(model, None)

        tag = "final" if step is None else str(step)
        print(f"PROBE step={tag} (layers={n}) "
              f"loss_normal={loss_normal:.4f} (ppl {math.exp(min(20.0, loss_normal)):.3f}) "
              f"loss_zeroed={loss_zeroed:.4f} (ppl {math.exp(min(20.0, loss_zeroed)):.3f}) "
              f"C={loss_zeroed - loss_normal:+.4f}", flush=True)


if __name__ == "__main__":
    main()
