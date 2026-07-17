"""Q22 P0 surgeries: chunk-quantized rotary phases on TRAINED w128 checkpoints.

Eval-only. For each requested mode we set the chunkq attributes on the loaded
model's TTT layers and re-run the standard val evaluation (fixed 2M-token ds42
cache), asking: how much of the trained model's rotary use is intra-chunk-fine
vs chunk-granular?

  hidden surgery (on honly-g1.0):  ttt_hrope_chunkq=1024
  input  surgery (on rope):        ttt_input_chunkq=1024
  input  manual-path sanity:       ttt_input_chunkq=1  (must match baseline)

Usage:
  python surgery_chunkq.py --run_dir outputs/q17_honly_g1_w128 \
      --modes base,hidden1024
  python surgery_chunkq.py --run_dir outputs/q17_rope_w128 \
      --modes base,input1,input1024
"""
import argparse
import json
import math
import os
import re

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from lact_model.configuration_lact_swiglu import LaCTSWIGLUConfig  # noqa: E402
from lact_model.modeling_lact import LaCTForCausalLM  # noqa: E402


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


def set_chunkq(model, hidden=0, inp=0):
    n = 0
    for mod in model.modules():
        if hasattr(mod, "ttt_hrope_chunkq"):
            mod.ttt_hrope_chunkq = int(hidden)
            mod.ttt_input_chunkq = int(inp)
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--ckpt", type=str, default="final.pt")
    p.add_argument("--modes", type=str, default="base",
                   help="comma list: base | hidden<C> | input<C>")
    p.add_argument("--val_cache", type=str, default=os.path.join(
        SCRIPT_DIR, "val_cache_fla-hub_transformer-1.3B-100B_4096_ds42.pt"))
    p.add_argument("--val_bs", type=int, default=8)
    args = p.parse_args()

    device = "cuda"
    config = config_from_train_log(os.path.join(args.run_dir, "train.log"))
    model = LaCTForCausalLM(config)
    sd = torch.load(os.path.join(args.run_dir, args.ckpt),
                    map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    model = model.to(device)

    val_set = torch.load(args.val_cache, map_location="cpu")
    print(f"[surgery] run={args.run_dir} val={val_set.shape} "
          f"chunk={config.lact_chunk_size} window={config.window_size}")

    for mode in args.modes.split(","):
        mode = mode.strip()
        if mode == "base":
            h, i = 0, 0
        elif mode.startswith("hidden"):
            h, i = int(mode[len("hidden"):]), 0
        elif mode.startswith("input"):
            h, i = 0, int(mode[len("input"):])
        else:
            raise ValueError(mode)
        n = set_chunkq(model, hidden=h, inp=i)
        loss = evaluate(model, val_set, args.val_bs, device)
        print(f"SURGERY mode={mode} (layers={n}) loss={loss:.4f} "
              f"ppl={math.exp(min(20.0, loss)):.3f}", flush=True)


if __name__ == "__main__":
    main()
