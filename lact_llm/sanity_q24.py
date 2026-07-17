"""Q24 sanity suite for the training-dynamics interventions (input-rope dropout /
hidden-first curriculum) and the commitment probe.

Tests (select with --tests, default A,B,C):
  A. _input_rope_scale=1.0 (float) and all-ones mask (tensor) through the manual
     input-rotary path (ttt_input_chunkq=1) are BIT-EXACT vs no scale; and the
     manual path ~equals the fla fast_rotary path (small bf16 tolerance).
  B. scale=0.0 (and an all-zeros mask) == a ttt_nope=True model with the same
     weights, bit-exact (zeroed angles -> cos=1/sin=0 -> exact identity). Plus a
     mixed mask [1,0,1,0]: kept rows match the rotated logits, dropped rows match
     the nope logits, row for row.
  C. Mask reproducibility: same (data_seed, step) -> same mask across two fresh
     python processes (stateless torch.Generator draw, no global RNG).
  D. 100-step smoke of design (a) with p0=0.5 (hpra w128 + ttt_input_chunkq=1):
     loss decreases and is finite. Run on a FREE gpu only (lock gpu6 first).
  E. probe_commitment.py on outputs/q17_hpra_g1_w128: C(final) should be small
     (~+0.003..0.01 in loss terms; the model ignored the hidden channel, F27b).

Usage (from lact_llm/, gpu6 locked via ../lact_nvs/outputs/.gpu_locks/gpu6):
  CUDA_VISIBLE_DEVICES=6 ../.venv_llm/bin/python sanity_q24.py --tests A,B,C
  CUDA_VISIBLE_DEVICES=6 ../.venv_llm/bin/python sanity_q24.py --tests D,E
"""
import argparse
import os
import re
import shutil
import subprocess
import sys

os.environ.setdefault("HF_HOME", "/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/datasets/hf_cache")
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_REPO_ROOT, ".cache_triton"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(_REPO_ROOT, ".cache_inductor"))
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from lact_model.configuration_lact_swiglu import LaCTSWIGLUConfig  # noqa: E402
from lact_model.modeling_lact import LaCTForCausalLM  # noqa: E402
from lact_model.layer_lact_swiglu import LaCTSWIGLULayer  # noqa: E402

PYTHON = os.path.join(_REPO_ROOT, ".venv_llm", "bin", "python")


def tiny_hpra_model(device):
    """Small hpra model routed through the manual input-rotary path."""
    torch.manual_seed(1234)
    cfg = LaCTSWIGLUConfig(
        hidden_size=256, num_hidden_layers=2, num_attn_heads=4, num_lact_heads=2,
        lact_chunk_size=128, window_size=64, max_position_embeddings=512,
        vocab_size=512, rope_theta=1000000, use_muon=True, use_momentum=True,
        learnable_ttt_scale=True, ttt_prenorm=True, ttt_hidden_rope=True,
        ttt_hrope_gain=1.0, ttt_input_chunkq=1, use_fused_kernel=False,
    )
    model = LaCTForCausalLM(cfg).to(device).eval()
    return model


def ttt_layers(model):
    return [m for m in model.modules() if isinstance(m, LaCTSWIGLULayer)]


@torch.no_grad()
def logits_of(model, x):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model(input_ids=x).logits.float().clone()


def set_scale(model, val):
    for lyr in ttt_layers(model):
        lyr._input_rope_scale = val


def set_chunkq(model, val):
    for lyr in ttt_layers(model):
        lyr.ttt_input_chunkq = int(val)


def set_nope(model, val):
    for lyr in ttt_layers(model):
        lyr.ttt_nope = bool(val)


def maxdiff(a, b):
    return (a - b).abs().max().item()


def test_A(device):
    print("=== A: scale=1.0 / ones-mask bit-exactness; manual vs fla path ===")
    model = tiny_hpra_model(device)
    torch.manual_seed(7)
    x = torch.randint(0, 512, (4, 512), device=device)

    set_scale(model, None)
    ref = logits_of(model, x)  # manual path, no scale

    set_scale(model, 1.0)
    d_float = maxdiff(logits_of(model, x), ref)
    set_scale(model, torch.ones(4, device=device))
    d_ones = maxdiff(logits_of(model, x), ref)
    set_scale(model, None)

    set_chunkq(model, 0)  # fla fast_rotary path
    d_fla = maxdiff(logits_of(model, x), ref)
    set_chunkq(model, 1)

    print(f"A1 scale=1.0 (float)  max|dlogits| = {d_float:.3e}  (want 0)")
    print(f"A2 mask=ones (tensor) max|dlogits| = {d_ones:.3e}  (want 0)")
    print(f"A3 manual vs fla path max|dlogits| = {d_fla:.3e}  (want small, bf16)")
    ok = d_float == 0.0 and d_ones == 0.0 and d_fla < 0.1
    print(f"A {'PASS' if ok else 'FAIL'}")
    return ok


def test_B(device):
    print("=== B: scale=0.0 == ttt_nope; mixed mask row-exactness ===")
    model = tiny_hpra_model(device)
    torch.manual_seed(7)
    x = torch.randint(0, 512, (4, 512), device=device)

    set_scale(model, None)
    ref_rope = logits_of(model, x)  # rotated (manual path)

    set_nope(model, True)  # skips the whole rotary block
    ref_nope = logits_of(model, x)
    set_nope(model, False)

    set_scale(model, 0.0)
    d_zero = maxdiff(logits_of(model, x), ref_nope)
    set_scale(model, torch.zeros(4, device=device))
    d_zeros_mask = maxdiff(logits_of(model, x), ref_nope)

    mask = torch.tensor([1.0, 0.0, 1.0, 0.0], device=device)
    set_scale(model, mask)
    mix = logits_of(model, x)
    d_kept = max(maxdiff(mix[0], ref_rope[0]), maxdiff(mix[2], ref_rope[2]))
    d_drop = max(maxdiff(mix[1], ref_nope[1]), maxdiff(mix[3], ref_nope[3]))
    set_scale(model, None)

    print(f"B1 scale=0.0 vs nope       max|dlogits| = {d_zero:.3e}  (want 0)")
    print(f"B2 mask=zeros vs nope      max|dlogits| = {d_zeros_mask:.3e}  (want 0)")
    print(f"B3 mask=[1,0,1,0] kept rows vs rope max|d| = {d_kept:.3e}  (want 0)")
    print(f"B4 mask=[1,0,1,0] drop rows vs nope max|d| = {d_drop:.3e}  (want 0)")
    ok = d_zero == 0.0 and d_zeros_mask == 0.0 and d_kept == 0.0 and d_drop == 0.0
    print(f"B {'PASS' if ok else 'FAIL'}")
    return ok


_C_SNIPPET = (
    "import sys; sys.path.insert(0, %r)\n"
    "from train_small import input_rope_dropout_keep\n"
    "out = []\n"
    "for seed, step in [(42, 0), (42, 1), (42, 100), (42, 29999), (137, 5000)]:\n"
    "    m = input_rope_dropout_keep(seed, step, 8, 0.5)\n"
    "    out.append((seed, step, tuple(int(v) for v in m)))\n"
    "print(out)\n" % SCRIPT_DIR
)


def test_C():
    print("=== C: mask reproducibility across two fresh processes ===")
    outs = []
    for i in range(2):
        r = subprocess.run([PYTHON, "-c", _C_SNIPPET], capture_output=True,
                           text=True, cwd=SCRIPT_DIR)
        assert r.returncode == 0, r.stderr[-2000:]
        outs.append(r.stdout.strip().splitlines()[-1])
    same = outs[0] == outs[1]
    print(f"proc1: {outs[0]}")
    print(f"proc2: {outs[1]}")
    print(f"C {'PASS' if same else 'FAIL'}")
    return same


def test_D(smoke_dir):
    print("=== D: 100-step design-(a) smoke (hpra w128 + chunkq1 + p0=0.5) ===")
    if os.path.exists(smoke_dir):
        shutil.rmtree(smoke_dir)
    os.makedirs(smoke_dir, exist_ok=True)
    cmd = [
        PYTHON, "train_small.py", "--out_dir", smoke_dir, "--bs", "8",
        "--steps", "100", "--log_every", "20", "--val_every", "100",
        "--save_every", "0", "--auto_resume", "false",
        "--input_rope_dropout_p0", "0.5", "--input_rope_dropout_anneal", "30000",
        "--extra_json",
        '{"ttt_hidden_rope": true, "ttt_hrope_gain": 1.0, "window_size": 128, '
        '"ttt_input_chunkq": 1}',
    ]
    log_path = os.path.join(smoke_dir, "train.log")
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                cwd=SCRIPT_DIR)
        print(f"[D] smoke pid={proc.pid} log={log_path}")
        try:
            ret = proc.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            proc.kill()  # our own PID only
            print("D FAIL (timeout, killed by pid)")
            return False
    text = open(log_path).read()
    losses = [float(m.group(1)) for m in
              re.finditer(r"^step=\d+ loss=([\d.]+|nan|inf)", text, re.M)]
    ps = re.findall(r"ropedrop_p=([\d.]+)", text)
    val = re.search(r"VAL step=100 loss=([\d.]+)", text)
    print(f"[D] ret={ret} step losses={losses} ropedrop_p={ps[:2]}... "
          f"val100={val.group(1) if val else 'MISSING'}")
    ok = (ret == 0 and len(losses) >= 3 and all(l == l and l < 1e4 for l in losses)
          and losses[-1] < losses[0] and len(ps) > 0 and val is not None)
    print(f"D {'PASS' if ok else 'FAIL'}"
          f"{'' if ok else ' (see ' + log_path + ')'}")
    if ok:
        shutil.rmtree(smoke_dir)
        print(f"[D] deleted {smoke_dir}")
    return ok


def test_E():
    print("=== E: commitment probe on trained q17_hpra_g1_w128 ===")
    r = subprocess.run(
        [PYTHON, "probe_commitment.py", "--run_dir", "outputs/q17_hpra_g1_w128"],
        capture_output=True, text=True, cwd=SCRIPT_DIR)
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr[-3000:])
        print("E FAIL")
        return False
    cs = [float(c) for c in re.findall(r"C=([+-][\d.]+)", r.stdout)]
    # expected: small positive C (~+0.003..0.01 loss) -- the trained hpra model
    # ignores its hidden rotation (F27b)
    ok = len(cs) >= 1 and all(-0.01 < c < 0.1 for c in cs)
    print(f"E C values = {cs} -> {'PASS (small, F27b signature)' if ok else 'FAIL'}")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tests", type=str, default="A,B,C")
    p.add_argument("--smoke_dir", type=str,
                   default=os.path.join(SCRIPT_DIR, "outputs", "q24_smoke_tmp"))
    args = p.parse_args()
    tests = [t.strip().upper() for t in args.tests.split(",")]

    device = "cuda"
    results = {}
    if "A" in tests:
        results["A"] = test_A(device)
    if "B" in tests:
        results["B"] = test_B(device)
    if "C" in tests:
        results["C"] = test_C()
    if "D" in tests:
        results["D"] = test_D(args.smoke_dir)
    if "E" in tests:
        results["E"] = test_E()

    print("\n=== SUMMARY ===")
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
