# Sanity checks for the synthetic exact-offset-copy task.
# Part A (CPU): layout / determinism / labels / stream resume.
# Part B (GPU, --gpu): untrained-model loss through the REAL training path
#   (model.train() + bf16 autocast + fused linear CE with -100 labels) vs a
#   manual fp32 CE on the copy region from eval logits.
import argparse
import math
import sys

sys.path.insert(0, "/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm")

import torch
import synthetic_copy as sc

OK = []


def check(name, cond):
    OK.append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def part_a():
    print("== Part A: layout / determinism / labels / resume ==")
    ds = 42
    for i in range(3):
        s = sc.make_sequence(ds, i)
        check(f"seq{i}: len 4096", s.shape == (4096,))
        check(f"seq{i}: SRC_MARKER id 4 at pos 511", s[511].item() == 4)
        check(f"seq{i}: RECALL_MARKER id 5 at pos 3071", s[3071].item() == 5)
        check(f"seq{i}: copy == source (offset 2560)",
              torch.equal(s[3072:3328], s[512:768]))
        check(f"seq{i}: exactly one occurrence of each marker",
              (s == 4).sum().item() == 1 and (s == 5).sum().item() == 1)
        noise_mask = torch.ones(4096, dtype=torch.bool)
        noise_mask[511] = noise_mask[3071] = False
        nz = s[noise_mask]
        check(f"seq{i}: all non-marker tokens in [10,1010)",
              (nz >= 10).all().item() and (nz < 1010).all().item())
        # source span should not accidentally equal any other aligned window
        check(f"seq{i}: source span is non-trivial (not constant)",
              s[512:768].unique().numel() > 100)
    # determinism
    check("determinism: regenerate identical",
          torch.equal(sc.make_sequence(ds, 1), sc.make_sequence(ds, 1)))
    check("different index -> different seq",
          not torch.equal(sc.make_sequence(ds, 0), sc.make_sequence(ds, 1)))
    check("different data_seed -> different seq",
          not torch.equal(sc.make_sequence(42, 0), sc.make_sequence(43, 0)))
    # labels
    x = torch.stack([sc.make_sequence(ds, i) for i in range(3)])
    lab = sc.make_labels(x)
    check("labels: -100 everywhere except copy region",
          (lab[:, :3072] == -100).all().item() and (lab[:, 3328:] == -100).all().item())
    check("labels: copy region equals inputs",
          torch.equal(lab[:, 3072:3328], x[:, 3072:3328]))
    check("labels: 256 supervised positions per row",
          ((lab != -100).sum(1) == 256).all().item())
    # val set
    val = sc.build_val_set(ds, n_seqs=64)
    check("val set shape [64,4096]", val.shape == (64, 4096))
    check("val rebuild identical", torch.equal(val, sc.build_val_set(ds, n_seqs=64)))
    check("val[0] uses index 1e9 (disjoint from train)",
          torch.equal(val[0], sc.make_sequence(ds, 10**9))
          and not torch.equal(val[0], sc.make_sequence(ds, 0)))
    # stream + resume
    st = sc.SyntheticCopyStream(ds)
    first6 = [next(st) for _ in range(6)]
    st2 = sc.SyntheticCopyStream(ds)
    for _ in range(5):
        next(st2)
    snap = st2.state()
    st3 = sc.SyntheticCopyStream(ds)
    st3.restore(snap)
    check("stream: restore(state after 5) -> 6th sequence identical",
          next(st3) == first6[5])
    check("stream: matches make_sequence stream",
          first6[0] == sc.make_sequence(ds, 0).tolist())
    # batch_generator integration
    import data_utils
    st4 = sc.SyntheticCopyStream(ds)
    bg = data_utils.batch_generator(st4, 2, with_state=True)
    b0, s0 = next(bg)
    b1, s1 = next(bg)
    check("batch_generator: [2,4096] int64 batches",
          b0.shape == (2, 4096) and b0.dtype == torch.int64)
    check("batch_generator: batch0 rows = seqs 0,1",
          torch.equal(b0[0], sc.make_sequence(ds, 0)) and torch.equal(b0[1], sc.make_sequence(ds, 1)))
    check("batch_generator: state after batch0 counts 2 consumed",
          int(s0["n_raw_consumed"]) == 2 and len(s0["buf"]) == 0)


def part_b():
    print("== Part B (GPU): untrained-model loss, fused-CE -100 masking ==")
    sys.path.insert(0, "/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm")
    import json
    from lact_model import LaCTForCausalLM, LaCTSWIGLUConfig

    cfg_path = ("/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/TTT_camera_embedding/lact_llm/"
                "configs/760M_lact_swiglu_nh4_fwlow_rank_momentum_muon.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg.pop("model_type", None)
    # production 200M CLI overrides (train_small.py defaults)
    cfg.update(dict(hidden_size=768, num_hidden_layers=12, num_attn_heads=12,
                    num_lact_heads=4, lact_chunk_size=1024, window_size=1024,
                    max_position_embeddings=4096, vocab_size=32000,
                    use_fused_kernel=False, bos_token_id=1, eos_token_id=2))
    config = LaCTSWIGLUConfig(**cfg)
    torch.manual_seed(42)
    model = LaCTForCausalLM(config).cuda()

    x = sc.build_val_set(42, n_seqs=4).cuda()
    labels = sc.make_labels(x)

    # (1) training path: fused linear CE with -100 labels
    model.train()
    with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
        loss_train = model(input_ids=x, labels=labels).loss.float().item()

    # (2) manual fp32 CE on copy-region logits (eval path, no labels)
    model.eval()
    with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
        logits = model(input_ids=x).logits
    pred = logits[:, sc.COPY_START - 1:sc.COPY_END - 1, :].float()
    tgt = x[:, sc.COPY_START:sc.COPY_END]
    loss_manual = torch.nn.functional.cross_entropy(
        pred.reshape(-1, 32000), tgt.reshape(-1)).item()
    acc = (pred.argmax(-1) == tgt).float().mean().item()

    # (3) eval path WITH masked labels through the model's own (fused) CE
    with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
        loss_eval_fused = model(input_ids=x, labels=labels).loss.float().item()

    ln_v, ln_1000 = math.log(32000), math.log(1000)
    print(f"  train-path fused-linear-CE loss (masked labels): {loss_train:.4f}")
    print(f"  manual fp32 copy-region CE:                      {loss_manual:.4f}")
    print(f"  eval-path fused-CE loss (masked labels):         {loss_eval_fused:.4f}")
    print(f"  untrained argmax copy accuracy:                  {acc:.5f}")
    print(f"  ln(vocab=32000) = {ln_v:.4f}  (untrained expectation)")
    print(f"  ln(1000) = {ln_1000:.4f}  (random floor once the 1000-id noise "
          f"vocab marginal is learned; the smoke run must reach ~this fast)")
    check("masking honored: train fused loss == manual copy-region CE (tol 0.05)",
          abs(loss_train - loss_manual) < 0.05)
    check("masking honored: eval fused loss == manual (tol 0.05)",
          abs(loss_eval_fused - loss_manual) < 0.05)
    check("untrained loss ~ ln(vocab) (within 1.0)", abs(loss_manual - ln_v) < 1.0)
    check("untrained accuracy ~ 1/32000", acc < 0.005)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    a = ap.parse_args()
    part_a()
    if a.gpu:
        part_b()
    n_fail = sum(1 for _, ok in OK if not ok)
    print(f"\n{len(OK) - n_fail}/{len(OK)} checks passed")
    sys.exit(1 if n_fail else 0)
