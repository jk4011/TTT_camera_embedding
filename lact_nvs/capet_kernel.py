"""Fused phase kernel for CaPET (2026-09-19).

The phase build -- cos/sin of the point (and direction) ladder -- was the single most
expensive thing CaPET added to a tttLRM layer: 3.09 ms of the +14 ms the hidden site
cost, plus 1.57 ms of bf16->fp32 casts on the backward of the output `.to(bf16)` and
0.89 ms for the gain reduction. It is not bandwidth bound; 200M fp32 sin/cos pairs is a
special-function-unit problem.

This module does the whole build in one Triton kernel:
  * the coordinate x = o + clamp(t) d is formed inside the kernel, so neither it nor the
    raw phase ever reaches DRAM (the eager path materialised an [R, P] fp32 phase tensor
    of 800 MB at the hidden site, wrote it, and read it twice);
  * cos/sin use the hardware SFU (`fast_cosf`/`fast_sinf`). The result is stored to
    bf16 regardless, whose 8-bit mantissa is ~1e-3 relative -- two orders coarser than
    the intrinsics' error -- so the accurate libm path bought nothing here;
  * the backward is hand-written, so the gradient stays bf16 end to end and reduces
    straight into the [3, F] gains and the [R, 1] depth.

The learnable gains and the predicted depth keep their gradients; only the arithmetic
changed. `TTTROPE_NO_TRITON=1` falls back to the eager implementation.
"""
import os

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
    _HAVE = os.environ.get("TTTROPE_NO_TRITON", "0") != "1"
except Exception:                                    # pragma: no cover
    _HAVE = False

if _HAVE:

    @triton.jit
    def _phase_fwd_kernel(O, D, T, WPT, WDIR, COS, SIN,
                          R, F: tl.constexpr, P: tl.constexpr, HAS_DIR: tl.constexpr,
                          BR: tl.constexpr, BF: tl.constexpr):
        pid_r = tl.program_id(0)
        pid_f = tl.program_id(1)
        r = pid_r * BR + tl.arange(0, BR)
        f = pid_f * BF + tl.arange(0, BF)
        rm = r < R
        fm = f < F
        t = tl.load(T + r, mask=rm, other=0.0)
        t = tl.maximum(t, 0.02)
        for a in tl.static_range(3):
            o_a = tl.load(O + r * 3 + a, mask=rm, other=0.0)
            d_a = tl.load(D + r * 3 + a, mask=rm, other=0.0)
            x_a = o_a + t * d_a
            w = tl.load(WPT + a * F + f, mask=fm, other=0.0)
            th = x_a[:, None] * w[None, :]
            c = libdevice.fast_cosf(th)
            s = libdevice.fast_sinf(th)
            off = r[:, None] * P + (a * F + f)[None, :]
            m = rm[:, None] & fm[None, :]
            tl.store(COS + off, c.to(COS.dtype.element_ty), mask=m)
            tl.store(SIN + off, s.to(SIN.dtype.element_ty), mask=m)
            if HAS_DIR:
                w2 = tl.load(WDIR + a * F + f, mask=fm, other=0.0)
                th2 = d_a[:, None] * w2[None, :]
                c2 = libdevice.fast_cosf(th2)
                s2 = libdevice.fast_sinf(th2)
                off2 = r[:, None] * P + (3 * F + a * F + f)[None, :]
                tl.store(COS + off2, c2.to(COS.dtype.element_ty), mask=m)
                tl.store(SIN + off2, s2.to(SIN.dtype.element_ty), mask=m)

    @triton.jit
    def _phase_bwd_kernel(O, D, T, WPT, WDIR, GCOS, GSIN, GWPT, GWDIR, GT,
                          R, F: tl.constexpr, P: tl.constexpr, HAS_DIR: tl.constexpr,
                          BR: tl.constexpr, BF: tl.constexpr):
        pid_r = tl.program_id(0)
        pid_f = tl.program_id(1)
        r = pid_r * BR + tl.arange(0, BR)
        f = pid_f * BF + tl.arange(0, BF)
        rm = r < R
        fm = f < F
        m = rm[:, None] & fm[None, :]
        t = tl.load(T + r, mask=rm, other=0.0)
        tc = tl.maximum(t, 0.02)
        gt = tl.zeros([BR], dtype=tl.float32)
        for a in tl.static_range(3):
            o_a = tl.load(O + r * 3 + a, mask=rm, other=0.0)
            d_a = tl.load(D + r * 3 + a, mask=rm, other=0.0)
            x_a = o_a + tc * d_a
            w = tl.load(WPT + a * F + f, mask=fm, other=0.0)
            th = x_a[:, None] * w[None, :]
            c = libdevice.fast_cosf(th)
            s = libdevice.fast_sinf(th)
            off = r[:, None] * P + (a * F + f)[None, :]
            gc = tl.load(GCOS + off, mask=m, other=0.0).to(tl.float32)
            gs = tl.load(GSIN + off, mask=m, other=0.0).to(tl.float32)
            gth = gs * c - gc * s                       # d/dth (gc cos + gs sin)
            tl.atomic_add(GWPT + a * F + f, tl.sum(gth * x_a[:, None], 0), mask=fm)
            gx = tl.sum(gth * w[None, :], 1)            # d/dx_a
            gt += gx * d_a
            if HAS_DIR:
                w2 = tl.load(WDIR + a * F + f, mask=fm, other=0.0)
                th2 = d_a[:, None] * w2[None, :]
                c2 = libdevice.fast_cosf(th2)
                s2 = libdevice.fast_sinf(th2)
                off2 = r[:, None] * P + (3 * F + a * F + f)[None, :]
                gc2 = tl.load(GCOS + off2, mask=m, other=0.0).to(tl.float32)
                gs2 = tl.load(GSIN + off2, mask=m, other=0.0).to(tl.float32)
                gth2 = gs2 * c2 - gc2 * s2
                tl.atomic_add(GWDIR + a * F + f, tl.sum(gth2 * d_a[:, None], 0), mask=fm)
        gt = tl.where(t > 0.02, gt, 0.0)                # clamp_min backward
        tl.atomic_add(GT + r, gt, mask=rm)


class _PhaseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, o, d, t, w_pt, w_dir, out_dtype):
        R = o.shape[0]
        F = w_pt.shape[1]
        P = 3 * F * (2 if w_dir is not None else 1)
        cos = torch.empty((R, P), device=o.device, dtype=out_dtype)
        sin = torch.empty((R, P), device=o.device, dtype=out_dtype)
        BR, BF = 64, 64
        grid = (triton.cdiv(R, BR), triton.cdiv(F, BF))
        _phase_fwd_kernel[grid](o, d, t, w_pt, w_dir if w_dir is not None else w_pt,
                                cos, sin, R, F, P, w_dir is not None, BR, BF,
                                num_warps=4, num_stages=2)
        ctx.save_for_backward(o, d, t, w_pt, w_dir)
        ctx.shape = (R, F, P, BR, BF)
        return cos, sin

    @staticmethod
    def backward(ctx, gcos, gsin):
        o, d, t, w_pt, w_dir = ctx.saved_tensors
        R, F, P, BR, BF = ctx.shape
        gwpt = torch.zeros_like(w_pt, dtype=torch.float32)
        gwdir = torch.zeros_like(w_dir, dtype=torch.float32) if w_dir is not None else gwpt
        gt = torch.zeros_like(t, dtype=torch.float32)
        grid = (triton.cdiv(R, BR), triton.cdiv(F, BF))
        _phase_bwd_kernel[grid](o, d, t, w_pt, w_dir if w_dir is not None else w_pt,
                                gcos.contiguous(), gsin.contiguous(), gwpt, gwdir, gt,
                                R, F, P, w_dir is not None, BR, BF,
                                num_warps=4, num_stages=2)
        return (None, None, gt.to(t.dtype),
                gwpt.to(w_pt.dtype), gwdir.to(w_pt.dtype) if w_dir is not None else None,
                None)


def phase_triton(o, d, t, w_pt, w_dir, out_dtype):
    """(cos, sin) [B, L, P] for the point (+direction) ladder.

    o, d: [B, L, 3] geometry (no grad). t: [B, L, 1] depth (grad flows to the depth
    channel). w_pt, w_dir: [3, F] = omega * gains (grad flows to the gains).
    """
    B, L, _ = o.shape
    c, s = _PhaseFn.apply(o.reshape(-1, 3).contiguous(), d.reshape(-1, 3).contiguous(),
                          t.reshape(-1).contiguous(), w_pt.contiguous(),
                          None if w_dir is None else w_dir.contiguous(), out_dtype)
    P = c.shape[-1]
    return c.view(B, L, P), s.view(B, L, P)


def available(o, d, t, w_pt):
    return (_HAVE and o.is_cuda and o.dtype == torch.float32
            and not o.requires_grad and not d.requires_grad)


# ---------------------------------------------------------------------------
# Fused rotary: phase + rotation in one kernel, no coefficient tensors.
#
# Measured on the hidden site (B=4, 16320 tokens, d_h=3072): precomputing the phases
# and rotating with them cost +10.4 ms/layer, of which only +2.7 ms was the rotation.
# The rest was autograd bookkeeping for the [B, L, 1536] cos/sin tensors: they feed
# three call sites (the stored note, its backward, and the query), so their gradient
# needs a zeroed buffer per site plus accumulation, and the coefficients are
# differentiable because both the gains and the predicted depth are learned. Building
# the phase inside the rotation deletes all of it -- the gradient that leaves this
# kernel is [3, F] per gain and [R] per depth. The phase is rebuilt on every call and
# again in the backward, which the hardware sin/cos makes cheap.
# ---------------------------------------------------------------------------

if _HAVE:

    @triton.jit
    def _rope_fwd_kernel(X, Y, O, D, T, WPT, WDIR,
                         R, Lc, HLc, Dx, Px, F, Plad, SGN,
                         HAS_DIR: tl.constexpr, BR: tl.constexpr, BF: tl.constexpr):
        pid_r = tl.program_id(0)
        pid_j = tl.program_id(1)
        r = pid_r * BR + tl.arange(0, BR)
        rm = r < R
        j0 = pid_j * BF
        j = j0 + tl.arange(0, BF)
        jm = j < Px
        off1 = r[:, None] * Dx + 2 * j[None, :]
        m = rm[:, None] & jm[None, :]
        x1 = tl.load(X + off1, mask=m, other=0.0).to(tl.float32)
        x2 = tl.load(X + off1 + 1, mask=m, other=0.0).to(tl.float32)
        if j0 < Plad:
            geo = (r // HLc) * Lc + (r % Lc)
            is_dir = HAS_DIR and (j0 >= 3 * F)
            jj = tl.where(is_dir, j - 3 * F, j)
            a = j0 // F - tl.where(is_dir, 3, 0)
            fcol = jj - a * F
            d_a = tl.load(D + geo * 3 + a, mask=rm, other=0.0)
            if is_dir:
                coord = d_a
                w = tl.load(WDIR + a * F + fcol, mask=jm, other=0.0)
            else:
                o_a = tl.load(O + geo * 3 + a, mask=rm, other=0.0)
                t = tl.maximum(tl.load(T + geo, mask=rm, other=0.0), 0.02)
                coord = o_a + t * d_a
                w = tl.load(WPT + a * F + fcol, mask=jm, other=0.0)
            th = coord[:, None] * w[None, :]
            c = libdevice.fast_cosf(th)
            s = SGN * libdevice.fast_sinf(th)
            y1 = x1 * c - x2 * s
            y2 = x1 * s + x2 * c
        else:
            y1 = x1
            y2 = x2
        tl.store(Y + off1, y1.to(Y.dtype.element_ty), mask=m)
        tl.store(Y + off1 + 1, y2.to(Y.dtype.element_ty), mask=m)

    @triton.jit
    def _rope_bwd_kernel(X, GY, GX, O, D, T, WPT, WDIR, GWPT, GWDIR, GT,
                         R, Lc, HLc, Dx, Px, F, Plad, SGN,
                         HAS_DIR: tl.constexpr, BR: tl.constexpr, BF: tl.constexpr):
        pid_r = tl.program_id(0)
        pid_j = tl.program_id(1)
        r = pid_r * BR + tl.arange(0, BR)
        rm = r < R
        j0 = pid_j * BF
        j = j0 + tl.arange(0, BF)
        jm = j < Px
        off1 = r[:, None] * Dx + 2 * j[None, :]
        m = rm[:, None] & jm[None, :]
        gy1 = tl.load(GY + off1, mask=m, other=0.0).to(tl.float32)
        gy2 = tl.load(GY + off1 + 1, mask=m, other=0.0).to(tl.float32)
        if j0 < Plad:
            x1 = tl.load(X + off1, mask=m, other=0.0).to(tl.float32)
            x2 = tl.load(X + off1 + 1, mask=m, other=0.0).to(tl.float32)
            geo = (r // HLc) * Lc + (r % Lc)
            is_dir = HAS_DIR and (j0 >= 3 * F)
            jj = tl.where(is_dir, j - 3 * F, j)
            a = j0 // F - tl.where(is_dir, 3, 0)
            fcol = jj - a * F
            d_a = tl.load(D + geo * 3 + a, mask=rm, other=0.0)
            traw = tl.load(T + geo, mask=rm, other=0.0)
            if is_dir:
                coord = d_a
                w = tl.load(WDIR + a * F + fcol, mask=jm, other=0.0)
            else:
                o_a = tl.load(O + geo * 3 + a, mask=rm, other=0.0)
                coord = o_a + tl.maximum(traw, 0.02) * d_a
                w = tl.load(WPT + a * F + fcol, mask=jm, other=0.0)
            th = coord[:, None] * w[None, :]
            c = libdevice.fast_cosf(th)
            sr = libdevice.fast_sinf(th)
            s = SGN * sr
            tl.store(GX + off1, (gy1 * c + gy2 * s).to(GX.dtype.element_ty), mask=m)
            tl.store(GX + off1 + 1, (gy2 * c - gy1 * s).to(GX.dtype.element_ty), mask=m)
            # d/dth of (y1, y2) with s = SGN sin(th), c = cos(th)
            gth = gy1 * (-x1 * sr - x2 * SGN * c) + gy2 * (x1 * SGN * c - x2 * sr)
            gth = tl.where(m, gth, 0.0)
            if is_dir:
                tl.atomic_add(GWDIR + a * F + fcol, tl.sum(gth * coord[:, None], 0), mask=jm)
            else:
                tl.atomic_add(GWPT + a * F + fcol, tl.sum(gth * coord[:, None], 0), mask=jm)
                gt = tl.sum(gth * w[None, :], 1) * d_a
                tl.atomic_add(GT + geo, tl.where(traw > 0.02, gt, 0.0), mask=rm)
        else:
            tl.store(GX + off1, gy1.to(GX.dtype.element_ty), mask=m)
            tl.store(GX + off1 + 1, gy2.to(GX.dtype.element_ty), mask=m)


class _RopeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, o, d, t, w_pt, w_dir, inverse, num_heads):
        xf = x.reshape(-1, x.shape[-1])
        R, Dx = xf.shape
        Px = Dx // 2
        F = w_pt.shape[1]
        Plad = 3 * F * (2 if w_dir is not None else 1)
        assert Plad <= Px, (Plad, Px)
        Lc = o.shape[1]
        y = torch.empty_like(xf)
        BR, BF = 32, (64 if F % 64 == 0 else 32)
        sgn = -1.0 if inverse else 1.0
        grid = (triton.cdiv(R, BR), triton.cdiv(Px, BF))
        _rope_fwd_kernel[grid](xf, y, o, d, t, w_pt, w_dir if w_dir is not None else w_pt,
                               R, Lc, num_heads * Lc, Dx, Px, F, Plad, sgn,
                               w_dir is not None, BR, BF, num_warps=4, num_stages=2)
        ctx.save_for_backward(xf, o, d, t, w_pt, w_dir)
        ctx.meta = (R, Lc, Dx, Px, F, Plad, sgn, num_heads, BR, BF)
        return y.view_as(x)

    @staticmethod
    def backward(ctx, gy):
        xf, o, d, t, w_pt, w_dir = ctx.saved_tensors
        R, Lc, Dx, Px, F, Plad, sgn, num_heads, BR, BF = ctx.meta
        gyf = gy.reshape(-1, Dx).contiguous()
        gx = torch.empty_like(gyf)
        gwpt = torch.zeros_like(w_pt, dtype=torch.float32)
        gwdir = torch.zeros_like(w_dir, dtype=torch.float32) if w_dir is not None else gwpt
        gt = torch.zeros(t.numel(), device=t.device, dtype=torch.float32)
        grid = (triton.cdiv(R, BR), triton.cdiv(Px, BF))
        _rope_bwd_kernel[grid](xf, gyf, gx, o, d, t, w_pt,
                               w_dir if w_dir is not None else w_pt,
                               gwpt, gwdir, gt,
                               R, Lc, num_heads * Lc, Dx, Px, F, Plad, sgn,
                               w_dir is not None, BR, BF, num_warps=4, num_stages=2)
        return (gx.view_as(gy), None, None, gt.view_as(t).to(t.dtype),
                gwpt.to(w_pt.dtype), gwdir.to(w_pt.dtype) if w_dir is not None else None,
                None, None)


def rope_fused(x, o, d, t, w_pt, w_dir, inverse=False, num_heads=1):
    """Rotate x by the point (+direction) ladder, building the phases inside the kernel.

    x: [B*heads, L, D] (interleaved pairs 2j / 2j+1, the layout the checkpoints use).
    o, d: [B, L, 3] geometry, no grad. t: [B, L, 1] depth. w_pt, w_dir: [3, F].
    """
    return _RopeFn.apply(x.contiguous(), o.contiguous(), d.contiguous(),
                         t.reshape(-1).contiguous(), w_pt.contiguous(),
                         None if w_dir is None else w_dir.contiguous(),
                         inverse, num_heads)


# ---------------------------------------------------------------------------
# Rotation against PRECOMPUTED phases, with the gain/depth gradient accumulated
# inside the kernel.
#
# The phase tables are built once per layer and read by three sites (the stored note,
# its backward, the query). Left to autograd that sharing is what costs: each site
# produces a full [rows, P] gradient for cos and for sin, which have to be zeroed,
# cast and summed -- measured at +7.6 ms of the hidden site's +10.4 ms, against +2.7 ms
# for the rotation itself. Building the phase inside the rotation instead (rope_fused
# above) removes the tables but recomputes 200M sin/cos six times per layer, which is
# worse. This path keeps the single build and takes the tables as NON-differentiable
# inputs: each site's backward reduces straight into the [3, F] gains and the [R] depth,
# so no [rows, P] gradient ever exists.
# ---------------------------------------------------------------------------

if _HAVE:

    @triton.jit
    def _rotpre_fwd_kernel(X, Y, C, S, R, Lc, HLc, Lfull, start, Dx, P, SGN,
                           BR: tl.constexpr, BF: tl.constexpr):
        pid_r = tl.program_id(0)
        pid_j = tl.program_id(1)
        r = pid_r * BR + tl.arange(0, BR)
        rm = r < R
        j0 = pid_j * BF
        j = j0 + tl.arange(0, BF)
        # the pair (2j, 2j+1) is read as one contiguous 2-element run and split in
        # registers: a stride-2 gather halves the useful bytes per memory transaction.
        e = 2 * j0 + tl.arange(0, 2 * BF)
        off = r[:, None] * Dx + e[None, :]
        me = rm[:, None] & (e < Dx)[None, :]
        v = tl.reshape(tl.load(X + off, mask=me, other=0.0), (BR, BF, 2))
        x1, x2 = tl.split(v)
        x1 = x1.to(tl.float32)
        x2 = x2.to(tl.float32)
        grow = (r // HLc) * Lfull + start + (r % Lc)
        poff = grow[:, None] * P + j[None, :]
        mp = rm[:, None] & (j < P)[None, :]
        c = tl.load(C + poff, mask=mp, other=1.0).to(tl.float32)
        s = SGN * tl.load(S + poff, mask=mp, other=0.0).to(tl.float32)
        y = tl.join((x1 * c - x2 * s).to(Y.dtype.element_ty),
                    (x1 * s + x2 * c).to(Y.dtype.element_ty))
        tl.store(Y + off, tl.reshape(y, (BR, 2 * BF)), mask=me)

    @triton.jit
    def _rotpre_bwd_kernel(X, GY, GX, C, S, O, D, T, WPT, WDIR, GWPT, GWDIR, GT,
                           R, Lc, HLc, Lfull, start, Dx, P, F, SGN,
                           HAS_DIR: tl.constexpr, BR: tl.constexpr, BF: tl.constexpr):
        pid_r = tl.program_id(0)
        pid_j = tl.program_id(1)
        r = pid_r * BR + tl.arange(0, BR)
        rm = r < R
        j0 = pid_j * BF
        j = j0 + tl.arange(0, BF)
        e = 2 * j0 + tl.arange(0, 2 * BF)
        off = r[:, None] * Dx + e[None, :]
        me = rm[:, None] & (e < Dx)[None, :]
        gy1, gy2 = tl.split(tl.reshape(tl.load(GY + off, mask=me, other=0.0), (BR, BF, 2)))
        gy1 = gy1.to(tl.float32)
        gy2 = gy2.to(tl.float32)
        grow = (r // HLc) * Lfull + start + (r % Lc)
        poff = grow[:, None] * P + j[None, :]
        mp = rm[:, None] & (j < P)[None, :]
        c = tl.load(C + poff, mask=mp, other=1.0).to(tl.float32)
        sr = tl.load(S + poff, mask=mp, other=0.0).to(tl.float32)
        s = SGN * sr
        gx = tl.join((gy1 * c + gy2 * s).to(GX.dtype.element_ty),
                     (gy2 * c - gy1 * s).to(GX.dtype.element_ty))
        tl.store(GX + off, tl.reshape(gx, (BR, 2 * BF)), mask=me)
        if j0 < P:
            x1, x2 = tl.split(tl.reshape(tl.load(X + off, mask=me, other=0.0), (BR, BF, 2)))
            x1 = x1.to(tl.float32)
            x2 = x2.to(tl.float32)
            gth = gy1 * (-x1 * sr - x2 * SGN * c) + gy2 * (x1 * SGN * c - x2 * sr)
            gth = tl.where(mp, gth, 0.0)
            jm = j < P
            is_dir = HAS_DIR and (j0 >= 3 * F)
            a = j0 // F - tl.where(is_dir, 3, 0)
            fcol = tl.where(is_dir, j - 3 * F, j) - a * F
            d_a = tl.load(D + grow * 3 + a, mask=rm, other=0.0)
            if is_dir:
                tl.atomic_add(GWDIR + a * F + fcol, tl.sum(gth * d_a[:, None], 0), mask=jm)
            else:
                o_a = tl.load(O + grow * 3 + a, mask=rm, other=0.0)
                traw = tl.load(T + grow, mask=rm, other=0.0)
                coord = o_a + tl.maximum(traw, 0.02) * d_a
                w = tl.load(WPT + a * F + fcol, mask=jm, other=0.0)
                tl.atomic_add(GWPT + a * F + fcol, tl.sum(gth * coord[:, None], 0), mask=jm)
                gt = tl.sum(gth * w[None, :], 1) * d_a
                tl.atomic_add(GT + grow, tl.where(traw > 0.02, gt, 0.0), mask=rm)


class _RotPreFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, c, s, o, d, t, w_pt, w_dir, inverse, num_heads, start, Lc):
        xf = x.reshape(-1, x.shape[-1])
        R, Dx = xf.shape
        Px = Dx // 2
        P = c.shape[-1]
        F = w_pt.shape[1]
        Lfull = o.shape[1]
        y = torch.empty_like(xf)
        # the backward reduces into the [3, F] gains, so a pair block must stay inside
        # one (axis, point/direction) segment of the ladder
        BR, BF = 16, (64 if F % 64 == 0 else 32)
        assert F % BF == 0, (F, BF)
        sgn = -1.0 if inverse else 1.0
        grid = (triton.cdiv(R, BR), triton.cdiv(Px, BF))
        _rotpre_fwd_kernel[grid](xf, y, c, s, R, Lc, num_heads * Lc, Lfull, start,
                                 Dx, P, sgn, BR, BF, num_warps=4, num_stages=3)
        ctx.save_for_backward(xf, c, s, o, d, t, w_pt, w_dir)
        ctx.meta = (R, Lc, Lfull, start, Dx, Px, P, F, sgn, num_heads, BR, BF)
        return y.view_as(x)

    @staticmethod
    def backward(ctx, gy):
        xf, c, s, o, d, t, w_pt, w_dir = ctx.saved_tensors
        R, Lc, Lfull, start, Dx, Px, P, F, sgn, num_heads, BR, BF = ctx.meta
        gyf = gy.reshape(-1, Dx).contiguous()
        gx = torch.empty_like(gyf)
        gwpt = torch.zeros_like(w_pt, dtype=torch.float32)
        gwdir = torch.zeros_like(w_dir, dtype=torch.float32) if w_dir is not None else gwpt
        gt = torch.zeros(t.numel(), device=t.device, dtype=torch.float32)
        grid = (triton.cdiv(R, BR), triton.cdiv(Px, BF))
        _rotpre_bwd_kernel[grid](xf, gyf, gx, c, s, o, d, t, w_pt,
                                 w_dir if w_dir is not None else w_pt, gwpt, gwdir, gt,
                                 R, Lc, num_heads * Lc, Lfull, start, Dx, P, F, sgn,
                                 w_dir is not None, BR, BF, num_warps=4, num_stages=3)
        return (gx.view_as(gy), None, None, None, None, gt.view_as(t).to(t.dtype),
                gwpt.to(w_pt.dtype), gwdir.to(w_pt.dtype) if w_dir is not None else None,
                None, None, None, None)


def rot_pre(x, c, s, o, d, t, w_pt, w_dir, inverse=False, num_heads=1, start=0, Lc=None):
    """Rotate x by precomputed phases; the gains and the depth get their gradient here.

    x: [B*heads, Lc, D] interleaved pairs. c, s: [B, Lfull, P] (never differentiated).
    o, d, t: the FULL-sequence geometry; `start` selects this call's token window.
    """
    return _RotPreFn.apply(x.contiguous(), c, s, o, d, t.reshape(-1), w_pt,
                           w_dir, inverse, num_heads, start,
                           x.shape[1] if Lc is None else Lc)


def phase_nograd(o, d, t, w_pt, w_dir, out_dtype):
    """(cos, sin) with no autograd graph -- the gradient comes back through `rot_pre`."""
    with torch.no_grad():
        B, L, _ = o.shape
        R = B * L
        F = w_pt.shape[1]
        P = 3 * F * (2 if w_dir is not None else 1)
        cos = torch.empty((R, P), device=o.device, dtype=out_dtype)
        sin = torch.empty((R, P), device=o.device, dtype=out_dtype)
        BR, BF = 64, 64
        grid = (triton.cdiv(R, BR), triton.cdiv(F, BF))
        _phase_fwd_kernel[grid](o.reshape(-1, 3).contiguous(), d.reshape(-1, 3).contiguous(),
                                t.reshape(-1).contiguous(), w_pt, w_dir if w_dir is not None else w_pt,
                                cos, sin, R, F, P, w_dir is not None, BR, BF,
                                num_warps=4, num_stages=2)
    return cos.view(B, L, P), sin.view(B, L, P)


# ---------------------------------------------------------------------------
# Forward-only entry points for the NVS cost measurement (table:cost). Inference runs
# under no_grad, so none of the backward kernels above are needed: the phase is built
# once per site with the SFU kernel and each rotation is one kernel instead of the
# five-to-ten small ops the eager path launches. At d=256 the backbone's own matmuls
# take ~4 ms of GPU time, so what the camera code costs is almost entirely the NUMBER
# of kernels it launches, not the bytes it moves.
# ---------------------------------------------------------------------------

def phase_fwd(o, d, t, w_pt, w_dir, out_dtype=torch.bfloat16):
    """(cos, sin) [B, L, P], P = 3F (+3F with a direction half). No autograd graph."""
    return phase_nograd(o, d, t, w_pt, w_dir, out_dtype)


def rot_fwd(x, c, s, inverse=False, start=0, Lc=None):
    """Rotate x [rows, L, D] by precomputed tables c/s [rows, Lfull, P], pairs (2j, 2j+1).

    `start`/`Lc` select a token window of the tables, for the hidden site's chunked kernel.
    """
    assert _HAVE, "triton unavailable"
    xf = x.reshape(-1, x.shape[-1])
    R, Dx = xf.shape
    P = c.shape[-1]
    Lfull = c.shape[1]
    Lc = x.shape[1] if Lc is None else Lc
    y = torch.empty_like(xf)
    BR, BF = 16, 64
    grid = (triton.cdiv(R, BR), triton.cdiv(Dx // 2, BF))
    _rotpre_fwd_kernel[grid](xf, y, c, s, R, Lc, Lc, Lfull, start, Dx, P,
                             -1.0 if inverse else 1.0, BR, BF, num_warps=4, num_stages=3)
    return y.view_as(x)
