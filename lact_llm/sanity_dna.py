# -*- coding: utf-8 -*-
"""Sanity checks for the hg38 DNA data path (dna_data.py).

Covers P0 sanity (c) at the stream level: exact resume via state()/restore()
(batch-tensor hash equality), plus shuffle determinism and val-set format.
Run: ../.venv_llm/bin/python sanity_dna.py
"""
import hashlib
import sys

import torch

import dna_data


def _hash_blocks(blocks):
    h = hashlib.sha256()
    for b in blocks:
        h.update(torch.tensor(b, dtype=torch.int64).numpy().tobytes())
    return h.hexdigest()


def main():
    S = int(sys.argv[1]) if len(sys.argv) > 1 else dna_data.SEQ_LEN
    n_val = 2_000_000 // S
    print(f"=== sanity_dna seq_len={S} (val blocks={n_val}) ===")
    path = dna_data.ensure_train_blocks(S)
    ok = True

    # --- val set format ---
    val = dna_data.get_or_build_dna_val_set(n_val, dna_data.val_cache_path(S), S)
    vmin, vmax = int(val.min()), int(val.max())
    val_ok = tuple(val.shape) == (n_val, S) and vmin >= 3 and vmax <= 7
    print(f"[val] shape={tuple(val.shape)} min={vmin} max={vmax} -> "
          f"{'OK' if val_ok else 'FAIL'}")
    ok &= val_ok

    # --- shuffle determinism: same seed identical, different seed differs ---
    s_a = dna_data.DnaBlockStream(path, 42, S)
    s_b = dna_data.DnaBlockStream(path, 42, S)
    s_c = dna_data.DnaBlockStream(path, 43, S)
    h_a = _hash_blocks([next(s_a) for _ in range(20)])
    h_b = _hash_blocks([next(s_b) for _ in range(20)])
    h_c = _hash_blocks([next(s_c) for _ in range(20)])
    det_ok = (h_a == h_b) and (h_a != h_c)
    print(f"[shuffle] seed42==seed42:{h_a == h_b} seed42!=seed43:{h_a != h_c} -> "
          f"{'OK' if det_ok else 'FAIL'}")
    ok &= det_ok

    # --- resume exactness: snapshot mid-stream, restore into a fresh stream,
    #     next M batches must be bit-identical (the gold-test hash compare) ---
    s1 = dna_data.DnaBlockStream(path, 42, S)
    for _ in range(137):          # advance to an arbitrary mid-stream position
        next(s1)
    st = s1.state()
    cont = [next(s1) for _ in range(11)]
    h_cont = _hash_blocks(cont)

    s2 = dna_data.DnaBlockStream(path, 42, S)
    s2.restore(st)
    cont2 = [next(s2) for _ in range(11)]
    h_cont2 = _hash_blocks(cont2)
    resume_ok = (h_cont == h_cont2) and (int(st["n_raw_consumed"]) == 137)
    print(f"[resume] n_raw_consumed={int(st['n_raw_consumed'])} "
          f"cont_hash_match={h_cont == h_cont2} -> {'OK' if resume_ok else 'FAIL'}")
    ok &= resume_ok

    # --- epoch wrap resume: restore across an epoch boundary ---
    Ntot = s1.N
    s3 = dna_data.DnaBlockStream(path, 42, S)
    s3.restore({"n_raw_consumed": Ntot - 3, "buf": torch.empty(0, dtype=torch.int64)})
    wrap = [next(s3) for _ in range(6)]        # crosses into epoch 1
    h_wrap = _hash_blocks(wrap)
    s4 = dna_data.DnaBlockStream(path, 42, S)
    s4.restore({"n_raw_consumed": Ntot - 3, "buf": torch.empty(0, dtype=torch.int64)})
    h_wrap2 = _hash_blocks([next(s4) for _ in range(6)])
    wrap_ok = h_wrap == h_wrap2
    print(f"[epoch-wrap] N={Ntot:,} cross-boundary resume match={wrap_ok} -> "
          f"{'OK' if wrap_ok else 'FAIL'}")
    ok &= wrap_ok

    print(f"\nSANITY_DNA: {'ALL PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
