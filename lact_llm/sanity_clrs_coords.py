# -*- coding: utf-8 -*-
"""Q29 external-coordinate rotary sanity suite (CLRS-Text 2-D addressing).

What is under test: `_ext_coords` on LaCTSWIGLULayer, which replaces the scalar
token position with a per-token 2-D address [b, s, 2] at BOTH rotary sites, by
splitting each frequency ladder in half (low half <- axis 0, high half <- axis 1).

The suite exists to close two specific traps this project has already been
bitten by:

  * the SILENT NULL (non-persistent buffers / unpatched apply sites): a wiring
    that is quietly ignored produces a plausible baseline-looking number instead
    of an error. Mode `b` is the POSITIVE control -- a 2-D address must change
    the loss, or the grid is measuring nothing.
  * an ablation that is not actually an ablation: the 1-D arm of the grid must
    be the SAME encoding the baseline already uses, not a degraded one. Mode `a`
    proves it exactly: feeding coords (t, t) makes the split ladder recombine
    into inv_freq * t, i.e. the stock rotary, so `1d` vs `2d` differs in the
    address and in nothing else.

Modes:
  a : equivalence -- coords (t, t) must reproduce the stock rotary to ~0, at the
      input site (vs the manual chunkq C=1 path AND vs the default fast_rotary
      path) and at the hidden site (vs the plain `pos` ladder).
  b : positive control -- coords (row, col) must give a loss DIFFERENT from
      (t, t) at each site, and the difference must vanish when the rotary is
      switched off (ttt_nope / no hidden rope), proving the effect enters
      through the rotary and not some other path.
  c : per-sequence addressing -- changing row 1's coordinates must leave row 0's
      logits untouched (the address is per sequence, not global).
  d : backward is finite through both sites with 2-D coords.

Device: reuses sanity_gbr_hidden's GPU-lock discipline and CPU shims (it picks
a free GPU, else runs tiny on CPU). Import side effects are intentional.
"""

import argparse
import sys

import torch

import sanity_gbr_hidden as S  # device pick, CPU shims, model builders

DEVICE = S.DEVICE
SEQ = S.SEQ_LEN

TOL = 1e-5


def _coords_1d(bs, seq, device):
    """(t, t): both ladder halves driven by the token index -> exactly the
    stock 1-D rotary."""
    t = torch.arange(seq, device=device, dtype=torch.float32)
    return t[None, :, None].expand(bs, seq, 2).contiguous()


def _coords_2d(bs, seq, device, n=41):
    """A synthetic (row, col) address with the shape CLRS produces: row grows
    every n tokens, col cycles 0..n-1."""
    t = torch.arange(seq, device=device, dtype=torch.float32)
    row = torch.div(t, n, rounding_mode="floor")
    col = t % n
    return torch.stack([row, col], dim=-1)[None].expand(bs, seq, 2).contiguous()


def _ttt_layers(m):
    out = []
    for mod in m.modules():
        if hasattr(mod, "_ext_coords") and hasattr(mod, "h_inv_freq_or_none"):
            out.append(mod)
    if not out:  # fall back to duck-typing on the attribute we added
        out = [mod for mod in m.modules() if hasattr(mod, "_ext_coords")]
    return out


def _set_coords(m, coords):
    n = 0
    for lyr in _ttt_layers(m):
        lyr._ext_coords = coords
        n += 1
    assert n > 0, "no layer exposed _ext_coords -- wiring missing"
    return n


def _loss(m, x):
    return float(S.model_loss(m, x).detach())


def a(args):
    """Equivalence: coords (t, t) == stock rotary."""
    ok = True
    x = S.get_batch(bs=2, seq=SEQ)
    c1d = _coords_1d(x.shape[0], x.shape[1], DEVICE)

    # ---- input site ----
    m_ref = S.build_model(seed=7, ttt_input_chunkq=1)          # manual path, C=1
    m_ext = S.build_model(seed=7)                              # manual path, ext coords
    m_ext.load_state_dict(m_ref.state_dict())
    l_ref = _loss(m_ref, x)
    n = _set_coords(m_ext, c1d)
    l_ext = _loss(m_ext, x)
    d = abs(l_ref - l_ext)
    ok &= d < TOL
    print(f"  [{'PASS' if d < TOL else 'FAIL'}] input site: ext(t,t) vs chunkq C=1 "
          f"|d|={d:.3e} ({n} layers patched)")

    # the stock (fla fast_rotary) path too -- catches a NeoX/GPT-J style mismatch
    m_stock = S.build_model(seed=7)
    m_stock.load_state_dict(m_ref.state_dict())
    l_stock = _loss(m_stock, x)
    d2 = abs(l_stock - l_ext)
    ok &= d2 < TOL
    print(f"  [{'PASS' if d2 < TOL else 'FAIL'}] input site: ext(t,t) vs stock "
          f"fast_rotary |d|={d2:.3e}")

    # ---- hidden site ----
    m_href = S.build_model(seed=11, ttt_hidden_rope=True, ttt_hrope_gain=1.0)
    m_hext = S.build_model(seed=11, ttt_hidden_rope=True, ttt_hrope_gain=1.0)
    m_hext.load_state_dict(m_href.state_dict())
    lh_ref = _loss(m_href, x)
    _set_coords(m_hext, c1d)
    lh_ext = _loss(m_hext, x)
    d3 = abs(lh_ref - lh_ext)
    ok &= d3 < TOL
    print(f"  [{'PASS' if d3 < TOL else 'FAIL'}] hidden site: ext(t,t) vs plain "
          f"pos ladder |d|={d3:.3e}")
    return ok


def b(args):
    """Positive control: a 2-D address must actually change the loss."""
    ok = True
    x = S.get_batch(bs=2, seq=SEQ)
    c1d = _coords_1d(x.shape[0], x.shape[1], DEVICE)
    c2d = _coords_2d(x.shape[0], x.shape[1], DEVICE)

    for name, extra in (("input", {}),
                        ("hidden", dict(ttt_hidden_rope=True, ttt_hrope_gain=1.0))):
        m = S.build_model(seed=13, **extra)
        _set_coords(m, c1d)
        l1 = _loss(m, x)
        _set_coords(m, c2d)
        l2 = _loss(m, x)
        d = abs(l1 - l2)
        good = d > 1e-4
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {name} site: 2-D address changes "
              f"the loss |d|={d:.3e}  (1d={l1:.6f} 2d={l2:.6f})")

    # negative half of the control: with the rotary OFF the address must be inert
    m = S.build_model(seed=13, ttt_nope=True)
    _set_coords(m, c1d)
    l1 = _loss(m, x)
    _set_coords(m, c2d)
    l2 = _loss(m, x)
    d = abs(l1 - l2)
    good = d < TOL
    ok &= good
    print(f"  [{'PASS' if good else 'FAIL'}] ttt_nope: address is inert "
          f"|d|={d:.3e}")
    return ok


def c(args):
    """The address is per sequence: perturbing row 1 must not move row 0."""
    x = S.get_batch(bs=2, seq=SEQ)
    m = S.build_model(seed=17, ttt_hidden_rope=True, ttt_hrope_gain=1.0)
    c_a = _coords_2d(2, x.shape[1], DEVICE, n=41).clone()
    c_b = c_a.clone()
    c_b[1] = _coords_2d(1, x.shape[1], DEVICE, n=7)[0]   # row 1 only
    _set_coords(m, c_a)
    with torch.no_grad():
        lg_a = m(input_ids=x).logits.float()
    _set_coords(m, c_b)
    with torch.no_grad():
        lg_b = m(input_ids=x).logits.float()
    d0 = float((lg_a[0] - lg_b[0]).abs().max())
    d1 = float((lg_a[1] - lg_b[1]).abs().max())
    ok = d0 < TOL and d1 > 1e-4
    print(f"  [{'PASS' if d0 < TOL else 'FAIL'}] row 0 unchanged  max|d|={d0:.3e}")
    print(f"  [{'PASS' if d1 > 1e-4 else 'FAIL'}] row 1 changed    max|d|={d1:.3e}")
    return ok


def d_(args):
    """Backward is finite through both sites with a 2-D address."""
    ok = True
    x = S.get_batch(bs=2, seq=SEQ)
    c2d = _coords_2d(2, x.shape[1], DEVICE)
    for name, extra in (("input", {}),
                        ("hidden", dict(ttt_hidden_rope=True, ttt_hrope_gain=1.0))):
        m = S.build_model(seed=19, **extra)
        _set_coords(m, c2d)
        loss = S.model_loss(m, x, backward=True)
        gs = [p.grad for p in m.parameters() if p.grad is not None]
        finite = bool(torch.isfinite(loss)) and all(
            bool(torch.isfinite(g).all()) for g in gs)
        ok &= finite
        print(f"  [{'PASS' if finite else 'FAIL'}] {name} site: loss "
              f"{float(loss):.6f}, {len(gs)} grads finite")
    return ok


MODES = {"a": a, "b": b, "c": c, "d": d_}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("modes", nargs="*", default=list(MODES),
                    help="subset of a b c d (default: all)")
    args = ap.parse_args()
    modes = args.modes or list(MODES)
    print(f"[q29] device={DEVICE} seq={SEQ}")
    ok = True
    for mname in modes:
        print(f"[q29] mode {mname}: {MODES[mname].__doc__.splitlines()[0]}")
        ok &= bool(MODES[mname](args))
    print(f"[q29] {'ALL PASSED' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
