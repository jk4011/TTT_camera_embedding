"""Camera-conditioned variants of the LaCT fast-weight layer.

All variants subclass FastWeightGluMLPMultihead and reuse its (compiled)
update/apply kernel unchanged. Camera tensors are computed once per forward in
model.compute_camera_info and threaded through the `info` dict.

Modes (see IDEAS.md):
  vo_rel        write v in world frame (block-diag c2w rotation), read back with
                the target view's inverse rotation -> exact relative value
                transport through the linear w1 pathway.
  qk_rope_cam   orthogonal per-token rotary on q/k after L2 norm; phases from
                Plucker (ray dir + moment) -> relative update-induced kernel.
  prope_ttt     full projective PRoPE transplant (q: P^T, k/v: P^-1, o: P) on
                the first half of head dims, with re-L2-norm of q/k.
  plucker_sinc  ray-segment integrated 3D rotary, closed-form sinc envelope;
                depth-free; fast weight becomes a field over 3D space.
  point_rope    depth-head-lifted 3D point rotary with uncertainty shrinkage.
  cam_lr        camera-conditioned per-token write lr (zero-init).
  adaln_cam     per-layer zero-init FiLM on x from pose features (control).
  q_reinject    query-side-only zero-init pose bias (read-path control).
  cam_registers per-input-view camera KV registers joining the update only.
  hyper_init    camera-set-conditioned low-rank delta on initial fast weights.
  fw3l          depth-3 inner net W_c silu(W_b (silu(W1 x) * (W3 x))), no rotary.
  fw3l_rot2     fw3l + input rotary (stock qk_rope_cam) + s2-site rotary.
  fw3l_rot3     fw3l + rotaries at all three address spaces (input, h1, s2).
  fw4l          depth-4 inner net W_d silu(W_c silu(W_b (silu(W1 x) * (W3 x)))),
                no rotary (4L depth control).
  fw4l_rot4     fw4l + rotaries at all four address spaces (input, h1, s2, s3).
  hnrot         (+ h_pra-family hidden rotary) RMS-normalize the hidden's
                ROTATED dims per token immediately before the hidden rotation,
                on both update and apply; the update backward passes through
                the exact RMSNorm Jacobian (LLM 'ttt_hrope_hnorm=rms_rot').
  gate_rope     (+ qk_rope_cam) GbR: only the silu GATE branch (w0) consumes
                the rotated q/k; the CONTENT branch (w2) gets the plain
                post-l2norm copy (LLM Q25b single-branch input rotary port).
  content_rope  mirror of gate_rope: only the CONTENT branch (w2) rotated.
"""
import math

import os
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from lact_ttt import (
    FastWeightGluMLPMultihead,
    TTTOperator,
    fast_weight_swish_glu_weight_norm_mini_batch_apply,
    inv_softplus,
    silu_backprop,
    zeropower_via_newtonschulz5,
)


def zero_init(linear):
    nn.init.zeros_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)
    return linear


def to_heads(t, num_heads):
    """[b, L, *] camera tensor -> [(b h), L, *] matching the qkv layout."""
    if num_heads == 1:
        return t
    return t.repeat_interleave(num_heads, dim=0)


def _apply_rotary_pairs_impl(x, coeff_cos, coeff_sin, inverse: bool = False):
    """Rotate adjacent feature pairs. coeff_*: [B, L, P]; acts on x[..., :2P]."""
    P = coeff_cos.size(-1)
    # index the pair dim and cast each half explicitly. Upcasting the whole slice
    # and unbind()-ing views lets inductor reorder the cast and land 1 ULP off in
    # bf16, which would break seed-matched bit-exact comparisons.
    x_rot = x[..., : 2 * P].reshape(*x.shape[:-1], P, 2)
    x1 = x_rot[..., 0].float()
    x2 = x_rot[..., 1].float()
    # inverse=True is the backward's transpose rotation. Negating the sin tensor at
    # the call site allocated a full copy per chunk; folding the sign in costs nothing.
    if inverse:
        y1 = x1 * coeff_cos + x2 * coeff_sin
        y2 = -x1 * coeff_sin + x2 * coeff_cos
    else:
        y1 = x1 * coeff_cos - x2 * coeff_sin
        y2 = x1 * coeff_sin + x2 * coeff_cos
    y = torch.stack([y1, y2], dim=-1).reshape(*x.shape[:-1], 2 * P)
    return torch.cat([y.to(x.dtype), x[..., 2 * P :]], dim=-1)



# Fused: the arithmetic is trivial but each elementwise step was its own kernel
# launch, so the tensor traffic was paid six or seven times over. Measured at the
# tttLRM hidden shape [4, 4096, 3072] on a B200: 1.892 ms eager against a 0.038 ms
# bandwidth floor, 0.160 ms fused, bit-identical. Compiling only this function is
# safe even where the surrounding TTT kernel cannot be compiled, because it holds no
# collectives. TTTROPE_NO_COMPILE=1 falls back to eager.
_ROTARY_FUSED = True
apply_rotary_pairs = _apply_rotary_pairs_impl
if hasattr(torch, "compile") and os.environ.get("TTTROPE_NO_COMPILE", "0") != "1":
    try:
        apply_rotary_pairs = torch.compile(_apply_rotary_pairs_impl, dynamic=True)
    except Exception:
        apply_rotary_pairs = _apply_rotary_pairs_impl

def apply_block_rot(x, R, transpose=False):
    """Apply per-token 3x3 rotation to consecutive 3-dim blocks of x.

    x: [B, L, D]; R: [B, L, 3, 3]. Leftover D % 3 dims are left unchanged.
    """
    D = x.size(-1)
    nb = D // 3
    blocks = x[..., : nb * 3].float().reshape(*x.shape[:-1], nb, 3)
    eq = "blji,blkj->blki" if transpose else "blij,blkj->blki"
    rotated = torch.einsum(eq, R.float(), blocks).reshape(*x.shape[:-1], nb * 3)
    return torch.cat([rotated.to(x.dtype), x[..., nb * 3 :]], dim=-1)



def _prope_rope_coeffs(positions, feat_dim, device):
    """Official PRoPE image-coordinate RoPE coefficients (freq_base 100,
    split pairing): positions [L] -> (cos, sin) [L, feat_dim//2]."""
    num_freqs = feat_dim // 2
    freqs = 100.0 ** (-torch.arange(num_freqs, device=device, dtype=torch.float32)
                      / num_freqs)
    ang = positions.float()[:, None] * freqs[None]
    return ang.cos(), ang.sin()


def _prope_rope_apply(x, cos, sin, inverse=False):
    """Official split-convention rope on the LAST dim of x [(bh), L, feat_dim]."""
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h].float(), x[..., h:].float()
    if not inverse:
        out = torch.cat([cos * x1 + sin * x2, -sin * x1 + cos * x2], dim=-1)
    else:
        out = torch.cat([cos * x1 - sin * x2, sin * x1 + cos * x2], dim=-1)
    return out.to(x.dtype)


def apply_tiled_mat4(x, M, tokens_per_view, num_dims):
    """Apply per-view 4x4 matrix to 4-dim blocks of x[..., :num_dims].

    x: [B, L, D]; M: [B, V, 4, 4]; L = V * tokens_per_view.
    """
    B, L, _ = x.shape
    V = M.size(1)
    nb = num_dims // 4
    blocks = x[..., : nb * 4].float().reshape(B, V, tokens_per_view, nb, 4)
    out = torch.einsum("bvij,bvtkj->bvtki", M.float(), blocks).reshape(B, L, nb * 4)
    return torch.cat([out.to(x.dtype), x[..., nb * 4 :]], dim=-1)


def lift_K4(K_norm):
    """[b, v, 4] normalized (fx, fy, cx, cy) -> [b, v, 4, 4] lifted intrinsics."""
    b, v, _ = K_norm.shape
    M = torch.eye(4, device=K_norm.device, dtype=K_norm.dtype).expand(b, v, 4, 4).clone()
    M[..., 0, 0] = K_norm[..., 0]
    M[..., 1, 1] = K_norm[..., 1]
    M[..., 0, 2] = K_norm[..., 2]
    M[..., 1, 2] = K_norm[..., 3]
    return M


def lift_K4_inv(K_norm):
    b, v, _ = K_norm.shape
    M = torch.eye(4, device=K_norm.device, dtype=K_norm.dtype).expand(b, v, 4, 4).clone()
    M[..., 0, 0] = 1.0 / K_norm[..., 0]
    M[..., 1, 1] = 1.0 / K_norm[..., 1]
    M[..., 0, 2] = -K_norm[..., 2] / K_norm[..., 0]
    M[..., 1, 2] = -K_norm[..., 3] / K_norm[..., 1]
    return M


def sinc(x):
    safe = torch.where(x.abs() < 1e-4, torch.ones_like(x), x)
    return torch.where(x.abs() < 1e-4, torch.ones_like(x), torch.sin(safe) / safe)


@torch.compile
def fast_weight_swish_glu_branch_input_rotary_apply(
    w0, w1, w2,
    q_gate, k_gate, q_cont, k_cont,
    v, lr0, lr1, lr2,
    ttt_ua_order: list,
    muon_update_steps: int = 0,
):
    """Baseline LaCT kernel with the two SwiGLU input branches fed by
    INDEPENDENT q/k copies (GbR single-branch input rotary, LLM Q25b port).

    The SwiGLU fast weight f(x) = (silu(x w0) * (x w2)) w1 has two input
    branches -- the silu GATE branch (w0) and the linear CONTENT branch (w2).
    The stock input rotary (qk_rope_cam) rotates the q/k feeding both; this
    kernel takes two q/k pairs so the caller can route the ROTATED copy to one
    branch and the PLAIN (post-l2norm) copy to the other, localizing where the
    input rotary's effect lives.

    Exact function: the update trains on
        f(k) = (silu(k_gate w0) * (k_cont w2)) w1,
    so every inner-loop gradient follows from that function -- dw0's input row
    is k_gate, dw2's is k_cont, and the shared upstream terms
    (dhidden_before_mul's silu gating, dgate's hidden_before_mul) each use
    their own branch's pre-activation. The kernel never backprops to k inside
    the inner loop (only dw0/dw1/dw2 are formed), so no dk split is needed;
    outer-loop autograd differentiates through both copies automatically. The
    apply path computes the same two matmuls from q_gate / q_cont. Muon /
    weight-norm are identical to the baseline.

    With q_gate is q_cont and k_gate is k_cont (same tensors in both slots),
    every op matches fast_weight_swish_glu_weight_norm_mini_batch_apply.

    Shapes match the baseline kernel: w0/w2 [b, d, dh], w1 [b, dh, d],
    q_*/k_*/v [b, l, d], lr* [b, l, 1-or-d].
    """
    from lact_ttt import silu_backprop, zeropower_via_newtonschulz5

    w0_norm = w0.detach().norm(dim=1, keepdim=True)
    w1_norm = w1.detach().norm(dim=1, keepdim=True)
    w2_norm = w2.detach().norm(dim=1, keepdim=True)

    output = []
    for start, end, update, apply in ttt_ua_order:
        w0_now, w1_now, w2_now = w0, w1, w2

        if update:
            kgi, vi = k_gate[:, start:end, :], v[:, start:end, :]
            kci = k_cont[:, start:end, :]
            lr0i = lr0[:, start:end, :]
            lr1i = lr1[:, start:end, :]
            lr2i = lr2[:, start:end, :]

            # gate branch consumes k_gate, content branch k_cont
            gate_before_act = kgi @ w0_now       # [b, l, dh]
            hidden_before_mul = kci @ w2_now     # [b, l, dh]
            # silu(gate) is used twice; it was recomputed over the full
            # [.., d_h] tensor. These kernels are not compiled (sp_all_reduce),
            # so nothing else removes it. Bit-exact.
            gate_act = F.silu(gate_before_act, inplace=False)
            hidden = gate_act * hidden_before_mul

            dhidden = vi @ w1_now.transpose(-1, -2)
            dhidden_before_mul = dhidden * gate_act
            dgate = dhidden * hidden_before_mul
            dgate_before_act = silu_backprop(dgate, gate_before_act)

            w1_grad = zeropower_via_newtonschulz5(
                (hidden * lr1i).transpose(-1, -2) @ vi, muon_update_steps
            )
            # dw0's input row is the GATE branch's k; dw2's is the CONTENT's
            w0_grad = zeropower_via_newtonschulz5(
                (kgi * lr0i).transpose(-1, -2) @ dgate_before_act, muon_update_steps
            )
            w2_grad = zeropower_via_newtonschulz5(
                (kci * lr2i).transpose(-1, -2) @ dhidden_before_mul, muon_update_steps
            )
            w1_now = w1_now + w1_grad
            w0_now = w0_now + w0_grad
            w2_now = w2_now + w2_grad

            w0_now = w0_now / (w0_now.norm(dim=1, keepdim=True) + 1e-5) * w0_norm
            w1_now = w1_now / (w1_now.norm(dim=1, keepdim=True) + 1e-5) * w1_norm
            w2_now = w2_now / (w2_now.norm(dim=1, keepdim=True) + 1e-5) * w2_norm

            w0, w1, w2 = w0_now, w1_now, w2_now

        if apply:
            qgi = q_gate[:, start:end, :]
            qci = q_cont[:, start:end, :]
            oi = (F.silu(qgi @ w0_now, inplace=True) * (qci @ w2_now)) @ w1_now
            output.append(oi)

    output = torch.cat(output, dim=1)
    return output, w0, w1, w2



def _mat4_tok(x, M_tok):
    """Per-TOKEN 4x4 block transform over the full width.

    x: [B, L, D] with D % 4 == 0; M_tok: [B, L, 4, 4]. Every 4-dim block of every
    token is multiplied by that token's matrix. fp32 math, cast back.
    """
    B, L, D = x.shape
    # fp32 with TF32 off: under bf16 autocast einsum would run in bf16, and even in
    # fp32 TF32 rounds the mantissa -- the identity-matrix audit then fails bit-
    # exactness against the stock kernel. These are 4x4 blocks; precision > speed.
    with torch.autocast(device_type=x.device.type, enabled=False):
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            blocks = x.float().reshape(B, L, D // 4, 4)
            out = torch.einsum("blij,blkj->blki", M_tok.float(), blocks)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32
    return out.reshape(B, L, D).to(x.dtype)


def fast_weight_swish_glu_hidden_mat4_apply(
    w0, w1, w2, q, k, v, lr0, lr1, lr2,
    Mu_tok, Ma_tok,
    ttt_ua_order: list,
    muon_update_steps: int = 0,
):
    """h-GA (Q41): the hidden-site GROUP ACTION. Identical to
    fast_weight_swish_glu_hidden_rotary_apply except the per-token PHASE rotation is
    replaced by a per-token tiled 4x4 MATRIX on the full hidden width:

        update:  dW1 ~ (M_u(i) h(k_i))^T v_i      apply:  o_j = (M_a(j) h(q_j)) W1

    With M_u = P^-1 (view of the update token) and M_a = P^T (view of the apply
    token), the retrieval channel becomes <P_j^T h_q, P_i^-1 h_k> = h_q^T (P_j
    P_i^-1) h_k: the relative projective transform, exactly prope's cancellation,
    but in HIDDEN space. Motivation F56: at wide baselines matrix actions win at
    the input site while every phase code loses; this asks whether the same holds
    at our site.

    The update backward passes through the exact transpose (dL/dh = M^T dL/dh'),
    which for a general matrix is NOT its inverse -- using the inverse here (the
    rotary habit) would silently corrupt the update. Muon and weight-norm are
    untouched. Mu_tok/Ma_tok: [B, L, 4, 4], sliced per op like hcos/hsin.
    """
    w0_norm = w0.detach().norm(dim=1, keepdim=True)
    w1_norm = w1.detach().norm(dim=1, keepdim=True)
    w2_norm = w2.detach().norm(dim=1, keepdim=True)
    output = []
    for start, end, update, apply in ttt_ua_order:
        w0_now, w1_now, w2_now = w0, w1, w2

        if update:
            ki, vi = k[:, start:end, :], v[:, start:end, :]
            lr0i = lr0[:, start:end, :]
            lr1i = lr1[:, start:end, :]
            lr2i = lr2[:, start:end, :]
            Mi = Mu_tok[:, start:end]

            gate_before_act = ki @ w0_now
            hidden_before_mul = ki @ w2_now
            gate_act = F.silu(gate_before_act, inplace=False)
            hidden = gate_act * hidden_before_mul
            hidden_mat = _mat4_tok(hidden, Mi)

            # dL/dh = M^T (dL/dh'): exact transpose, not the inverse.
            dhidden_mat = vi @ w1_now.transpose(-1, -2)
            dhidden = _mat4_tok(dhidden_mat, Mi.transpose(-1, -2))
            dhidden_before_mul = dhidden * gate_act
            dgate = dhidden * hidden_before_mul
            dgate_before_act = silu_backprop(dgate, gate_before_act)

            w1_grad = zeropower_via_newtonschulz5(
                (hidden_mat * lr1i).transpose(-1, -2) @ vi, muon_update_steps
            )
            w0_grad = zeropower_via_newtonschulz5(
                (ki * lr0i).transpose(-1, -2) @ dgate_before_act, muon_update_steps
            )
            w2_grad = zeropower_via_newtonschulz5(
                (ki * lr2i).transpose(-1, -2) @ dhidden_before_mul, muon_update_steps
            )
            w1_now = w1_now + w1_grad
            w0_now = w0_now + w0_grad
            w2_now = w2_now + w2_grad

            w0_now = w0_now / (w0_now.norm(dim=1, keepdim=True) + 1e-5) * w0_norm
            w1_now = w1_now / (w1_now.norm(dim=1, keepdim=True) + 1e-5) * w1_norm
            w2_now = w2_now / (w2_now.norm(dim=1, keepdim=True) + 1e-5) * w2_norm

            w0, w1, w2 = w0_now, w1_now, w2_now

        if apply:
            qi = q[:, start:end, :]
            hq = F.silu(qi @ w0_now, inplace=True) * (qi @ w2_now)
            hq = _mat4_tok(hq, Ma_tok[:, start:end])
            output.append(hq @ w1_now)

    output = torch.cat(output, dim=1)
    return output, w0, w1, w2


@torch.compile
def fast_weight_swish_glu_hidden_rotary_apply(
    w0, w1, w2, q, k, v, lr0, lr1, lr2,
    hcos, hsin,
    ttt_ua_order: list,
    muon_update_steps: int = 0,
    hnorm: bool = False,
):
    """Baseline LaCT kernel + per-token rotary applied to the SwiGLU *hidden*
    activation (h-PRA). The hidden layer h(x) is rotated by the token's phases
    before meeting w1, in both the update and apply paths:

        write:  dW1 ~ (R_i h(k_i))^T v_i         read:  o_j = (R_j h(q_j)) W1

    so the value-retrieval channel <R_j h(q_j), R_i h(k_i)> becomes relative in
    hidden space -- a second, independent relative channel with no attention
    analogue. Backprop through the rotation is its inverse (negated sin).

    hcos/hsin: [B, L, P] with 2P <= d_h; sliced per op like k (update) / q (apply).

    hnorm (cam_mode flag 'hnrot', mirrors LLM ttt_hrope_hnorm='rms_rot'):
    RMS-normalize the hidden's ROTATED dims h[..., :2P] per token immediately
    before the rotation, on both update and apply, leaving the position-free
    content dims' magnitudes intact. The update backward passes through the
    exact RMSNorm Jacobian, so the stored notes remain the exact gradient of
    the recall objective. hnorm=False is bit-identical to the stock kernel.
    """
    from lact_ttt import silu_backprop, zeropower_via_newtonschulz5

    P_rot = hcos.shape[-1]

    def _hn_fwd(x):
        """RMS-normalize the rotated dims x[..., :2P] per token (rms_rot).
        Returns (y, y_rot, rms); y_rot/rms cover only the normalized dims."""
        xr = x[..., : 2 * P_rot]
        rms = (xr.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6).sqrt()
        yr = (xr.float() / rms).type_as(x)
        return torch.cat([yr, x[..., 2 * P_rot :]], dim=-1), yr, rms

    def _hn_bwd(dy, y_bwd, rms):
        """Exact RMSNorm backward on the rotated dims:
        dx = (dy - y * mean(dy*y)) / rms; identity on the rest."""
        dyr = dy[..., : 2 * P_rot]
        m = (dyr.float() * y_bwd.float()).mean(dim=-1, keepdim=True)
        dxr = ((dyr.float() - y_bwd.float() * m) / rms).type_as(dy)
        return torch.cat([dxr, dy[..., 2 * P_rot :]], dim=-1)

    w0_norm = w0.detach().norm(dim=1, keepdim=True)
    w1_norm = w1.detach().norm(dim=1, keepdim=True)
    w2_norm = w2.detach().norm(dim=1, keepdim=True)

    output = []
    for start, end, update, apply in ttt_ua_order:
        w0_now, w1_now, w2_now = w0, w1, w2

        if update:
            ki, vi = k[:, start:end, :], v[:, start:end, :]
            lr0i = lr0[:, start:end, :]
            lr1i = lr1[:, start:end, :]
            lr2i = lr2[:, start:end, :]
            hci, hsi = hcos[:, start:end, :], hsin[:, start:end, :]

            gate_before_act = ki @ w0_now
            hidden_before_mul = ki @ w2_now
            # silu(gate) is used twice; it was recomputed over the full
            # [.., d_h] tensor. These kernels are not compiled (sp_all_reduce),
            # so nothing else removes it. Bit-exact.
            gate_act = F.silu(gate_before_act, inplace=False)
            hidden = gate_act * hidden_before_mul
            if hnorm:
                hidden_n, hn_y, hn_rms = _hn_fwd(hidden)
            else:
                hidden_n = hidden
            hidden_rot = apply_rotary_pairs(hidden_n, hci, hsi)

            # Backprop: dL/dh = R^T (dL/dh'), R^T = rotary with negated sin.
            dhidden_rot = vi @ w1_now.transpose(-1, -2)
            dhidden = apply_rotary_pairs(dhidden_rot, hci, hsi, inverse=True)
            if hnorm:
                # exact RMSNorm Jacobian back to the pre-norm hidden
                dhidden = _hn_bwd(dhidden, hn_y, hn_rms)
            dhidden_before_mul = dhidden * gate_act
            dgate = dhidden * hidden_before_mul
            dgate_before_act = silu_backprop(dgate, gate_before_act)

            w1_grad = zeropower_via_newtonschulz5(
                (hidden_rot * lr1i).transpose(-1, -2) @ vi, muon_update_steps
            )
            w0_grad = zeropower_via_newtonschulz5(
                (ki * lr0i).transpose(-1, -2) @ dgate_before_act, muon_update_steps
            )
            w2_grad = zeropower_via_newtonschulz5(
                (ki * lr2i).transpose(-1, -2) @ dhidden_before_mul, muon_update_steps
            )
            w1_now = w1_now + w1_grad
            w0_now = w0_now + w0_grad
            w2_now = w2_now + w2_grad

            w0_now = w0_now / (w0_now.norm(dim=1, keepdim=True) + 1e-5) * w0_norm
            w1_now = w1_now / (w1_now.norm(dim=1, keepdim=True) + 1e-5) * w1_norm
            w2_now = w2_now / (w2_now.norm(dim=1, keepdim=True) + 1e-5) * w2_norm

            w0, w1, w2 = w0_now, w1_now, w2_now

        if apply:
            qi = q[:, start:end, :]
            hq = F.silu(qi @ w0_now, inplace=True) * (qi @ w2_now)
            if hnorm:
                hq, _, _ = _hn_fwd(hq)
            hq = apply_rotary_pairs(hq, hcos[:, start:end, :], hsin[:, start:end, :])
            output.append(hq @ w1_now)

    output = torch.cat(output, dim=1)
    return output, w0, w1, w2


@torch.compile
def fast_weight_swish_glu_hidden_rotary_delta_apply(
    w0, w1, w2, q, k, v, lr0, lr1, lr2,
    hcos, hsin,
    ttt_ua_order: list,
    muon_update_steps: int = 0,
):
    """h-PRA with the hidden rotation applied ONLY to the Delta-W1 pathway.

    Write path identical to fast_weight_swish_glu_hidden_rotary_apply (rotated
    hidden writes into w1). Read path splits the last layer:

        o_j = h(q_j) @ w1_init  +  (R_j h(q_j)) @ (w1_now - w1_init)

    so the init readout T0 = h(q)W1^0 stays phase-free (slow weights need not
    hide in unrotated dims) while the retrieval channel through Delta-W1 stays
    fully relative. Unlocks large F_h.
    """
    from lact_ttt import silu_backprop, zeropower_via_newtonschulz5

    w0_norm = w0.detach().norm(dim=1, keepdim=True)
    w1_norm = w1.detach().norm(dim=1, keepdim=True)
    w2_norm = w2.detach().norm(dim=1, keepdim=True)
    w1_init = w1

    output = []
    for start, end, update, apply in ttt_ua_order:
        w0_now, w1_now, w2_now = w0, w1, w2

        if update:
            ki, vi = k[:, start:end, :], v[:, start:end, :]
            lr0i = lr0[:, start:end, :]
            lr1i = lr1[:, start:end, :]
            lr2i = lr2[:, start:end, :]
            hci, hsi = hcos[:, start:end, :], hsin[:, start:end, :]

            gate_before_act = ki @ w0_now
            hidden_before_mul = ki @ w2_now
            # silu(gate) is used twice; it was recomputed over the full
            # [.., d_h] tensor. These kernels are not compiled (sp_all_reduce),
            # so nothing else removes it. Bit-exact.
            gate_act = F.silu(gate_before_act, inplace=False)
            hidden = gate_act * hidden_before_mul
            hidden_rot = apply_rotary_pairs(hidden, hci, hsi)

            dhidden_rot = vi @ w1_now.transpose(-1, -2)
            dhidden = apply_rotary_pairs(dhidden_rot, hci, hsi, inverse=True)
            dhidden_before_mul = dhidden * gate_act
            dgate = dhidden * hidden_before_mul
            dgate_before_act = silu_backprop(dgate, gate_before_act)

            w1_grad = zeropower_via_newtonschulz5(
                (hidden_rot * lr1i).transpose(-1, -2) @ vi, muon_update_steps
            )
            w0_grad = zeropower_via_newtonschulz5(
                (ki * lr0i).transpose(-1, -2) @ dgate_before_act, muon_update_steps
            )
            w2_grad = zeropower_via_newtonschulz5(
                (ki * lr2i).transpose(-1, -2) @ dhidden_before_mul, muon_update_steps
            )
            w1_now = w1_now + w1_grad
            w0_now = w0_now + w0_grad
            w2_now = w2_now + w2_grad

            w0_now = w0_now / (w0_now.norm(dim=1, keepdim=True) + 1e-5) * w0_norm
            w1_now = w1_now / (w1_now.norm(dim=1, keepdim=True) + 1e-5) * w1_norm
            w2_now = w2_now / (w2_now.norm(dim=1, keepdim=True) + 1e-5) * w2_norm

            w0, w1, w2 = w0_now, w1_now, w2_now

        if apply:
            qi = q[:, start:end, :]
            hq = F.silu(qi @ w0_now, inplace=False) * (qi @ w2_now)
            hq_rot = apply_rotary_pairs(hq, hcos[:, start:end, :], hsin[:, start:end, :])
            output.append(hq @ w1_init + hq_rot @ (w1_now - w1_init))

    output = torch.cat(output, dim=1)
    return output, w0, w1, w2


@torch.compile
def fast_weight_swish_glu_hidden_rotary_multistep_apply(
    w0, w1, w2, q, k, v, lr0, lr1, lr2,
    hcos, hsin,
    step_gains,          # [n_steps, 3] learnable post-Muon write scales
    ttt_ua_order: list,
    muon_update_steps: int = 0,
):
    """h-PRA kernel with multiple full-chunk update steps.

    Differences from the single-step kernel: (i) each orthogonalized gradient
    is scaled by a learnable per-step, per-matrix gain (Muon's Frobenius
    pre-norm pins write magnitude; this restores a magnitude knob), and
    (ii) weight-norm runs ONCE after the final update step (per-step renorm
    would rescale step-1 content before step 2 sees it).
    """
    from lact_ttt import silu_backprop, zeropower_via_newtonschulz5

    w0_norm = w0.detach().norm(dim=1, keepdim=True)
    w1_norm = w1.detach().norm(dim=1, keepdim=True)
    w2_norm = w2.detach().norm(dim=1, keepdim=True)

    n_updates = sum(1 for op in ttt_ua_order if op.update)
    u_idx = 0
    output = []
    for start, end, update, apply in ttt_ua_order:
        w0_now, w1_now, w2_now = w0, w1, w2

        if update:
            ki, vi = k[:, start:end, :], v[:, start:end, :]
            lr0i = lr0[:, start:end, :]
            lr1i = lr1[:, start:end, :]
            lr2i = lr2[:, start:end, :]
            hci, hsi = hcos[:, start:end, :], hsin[:, start:end, :]

            gate_before_act = ki @ w0_now
            hidden_before_mul = ki @ w2_now
            # silu(gate) is used twice; it was recomputed over the full
            # [.., d_h] tensor. These kernels are not compiled (sp_all_reduce),
            # so nothing else removes it. Bit-exact.
            gate_act = F.silu(gate_before_act, inplace=False)
            hidden = gate_act * hidden_before_mul
            hidden_rot = apply_rotary_pairs(hidden, hci, hsi)

            dhidden_rot = vi @ w1_now.transpose(-1, -2)
            dhidden = apply_rotary_pairs(dhidden_rot, hci, hsi, inverse=True)
            dhidden_before_mul = dhidden * gate_act
            dgate = dhidden * hidden_before_mul
            dgate_before_act = silu_backprop(dgate, gate_before_act)

            w1_grad = zeropower_via_newtonschulz5(
                (hidden_rot * lr1i).transpose(-1, -2) @ vi, muon_update_steps
            )
            w0_grad = zeropower_via_newtonschulz5(
                (ki * lr0i).transpose(-1, -2) @ dgate_before_act, muon_update_steps
            )
            w2_grad = zeropower_via_newtonschulz5(
                (ki * lr2i).transpose(-1, -2) @ dhidden_before_mul, muon_update_steps
            )
            w0_now = w0_now + step_gains[u_idx, 0] * w0_grad
            w1_now = w1_now + step_gains[u_idx, 1] * w1_grad
            w2_now = w2_now + step_gains[u_idx, 2] * w2_grad

            if u_idx == n_updates - 1:
                w0_now = w0_now / (w0_now.norm(dim=1, keepdim=True) + 1e-5) * w0_norm
                w1_now = w1_now / (w1_now.norm(dim=1, keepdim=True) + 1e-5) * w1_norm
                w2_now = w2_now / (w2_now.norm(dim=1, keepdim=True) + 1e-5) * w2_norm
            u_idx = u_idx + 1

            w0, w1, w2 = w0_now, w1_now, w2_now

        if apply:
            qi = q[:, start:end, :]
            hq = F.silu(qi @ w0_now, inplace=True) * (qi @ w2_now)
            hq = apply_rotary_pairs(hq, hcos[:, start:end, :], hsin[:, start:end, :])
            output.append(hq @ w1_now)

    output = torch.cat(output, dim=1)
    return output, w0, w1, w2


@torch.compile
def fast_weight_swish_glu_hidden_rotary_res2_apply(
    w0, w1, w2, q, k, v, lr0, lr1, lr2,
    hcos, hsin,
    alpha,               # scalar: residual-correction strength (zero-init)
    step_gains,          # [2, 3] learnable post-Muon write scales
    ttt_ua_order: list,
    muon_update_steps: int = 0,
):
    """h-PRA kernel + delta-rule corrective second step (error-correcting write).

    Step 1 writes (k, v) as usual. Step 2 re-reads the keys against the
    updated memory, forms the residual target v' = v - alpha * f_{W'}(k), and
    writes (k, v'): the memory stores what it cannot yet retrieve. At alpha=0
    this reduces to a plain second step; with step_gains[1]=0 it is exactly
    the single-step kernel.
    """
    from lact_ttt import silu_backprop, zeropower_via_newtonschulz5

    w0_norm = w0.detach().norm(dim=1, keepdim=True)
    w1_norm = w1.detach().norm(dim=1, keepdim=True)
    w2_norm = w2.detach().norm(dim=1, keepdim=True)

    def one_update(w0_now, w1_now, w2_now, ki, vi, lr0i, lr1i, lr2i, hci, hsi, gains):
        gate_before_act = ki @ w0_now
        hidden_before_mul = ki @ w2_now
        # silu(gate) is used twice; it was recomputed over the full
        # [.., d_h] tensor. These kernels are not compiled (sp_all_reduce),
        # so nothing else removes it. Bit-exact.
        gate_act = F.silu(gate_before_act, inplace=False)
        hidden = gate_act * hidden_before_mul
        hidden_rot = apply_rotary_pairs(hidden, hci, hsi)

        dhidden_rot = vi @ w1_now.transpose(-1, -2)
        dhidden = apply_rotary_pairs(dhidden_rot, hci, hsi, inverse=True)
        dhidden_before_mul = dhidden * gate_act
        dgate = dhidden * hidden_before_mul
        dgate_before_act = silu_backprop(dgate, gate_before_act)

        w1_grad = zeropower_via_newtonschulz5(
            (hidden_rot * lr1i).transpose(-1, -2) @ vi, muon_update_steps
        )
        w0_grad = zeropower_via_newtonschulz5(
            (ki * lr0i).transpose(-1, -2) @ dgate_before_act, muon_update_steps
        )
        w2_grad = zeropower_via_newtonschulz5(
            (ki * lr2i).transpose(-1, -2) @ dhidden_before_mul, muon_update_steps
        )
        w0_now = w0_now + gains[0] * w0_grad
        w1_now = w1_now + gains[1] * w1_grad
        w2_now = w2_now + gains[2] * w2_grad
        w0_now = w0_now / (w0_now.norm(dim=1, keepdim=True) + 1e-5) * w0_norm
        w1_now = w1_now / (w1_now.norm(dim=1, keepdim=True) + 1e-5) * w1_norm
        w2_now = w2_now / (w2_now.norm(dim=1, keepdim=True) + 1e-5) * w2_norm
        return w0_now, w1_now, w2_now

    output = []
    for start, end, update, apply in ttt_ua_order:
        w0_now, w1_now, w2_now = w0, w1, w2

        if update:
            ki, vi = k[:, start:end, :], v[:, start:end, :]
            lr0i = lr0[:, start:end, :]
            lr1i = lr1[:, start:end, :]
            lr2i = lr2[:, start:end, :]
            hci, hsi = hcos[:, start:end, :], hsin[:, start:end, :]

            w0, w1, w2 = one_update(
                w0_now, w1_now, w2_now, ki, vi, lr0i, lr1i, lr2i, hci, hsi,
                step_gains[0],
            )

            # Corrective step: read keys against the updated memory, write
            # the residual target.
            gate2 = ki @ w0
            h2 = F.silu(gate2, inplace=False) * (ki @ w2)
            o_k = apply_rotary_pairs(h2, hci, hsi) @ w1
            v2 = vi - alpha * o_k
            w0, w1, w2 = one_update(
                w0, w1, w2, ki, v2, lr0i, lr1i, lr2i, hci, hsi, step_gains[1]
            )

        if apply:
            qi = q[:, start:end, :]
            hq = F.silu(qi @ w0, inplace=True) * (qi @ w2)
            hq = apply_rotary_pairs(hq, hcos[:, start:end, :], hsin[:, start:end, :])
            output.append(hq @ w1)

    output = torch.cat(output, dim=1)
    return output, w0, w1, w2


@torch.compile
def fast_weight_swiglu3l_weight_norm_apply(
    w0, w2, wb, w1, q, k, v,
    lr0, lr2, lrb, lr1,
    h1cos, h1sin, s2cos, s2sin,
    ttt_ua_order: list,
    muon_update_steps: int = 0,
):
    """Depth-3 fast-weight net (Q2: one rotary per address space).

        h1(x) = silu(x @ w0) * (x @ w2)          [d -> d_h]   (stock SwiGLU layer)
        s2(x) = silu(rot_h1(h1(x)) @ wb)         [d_h -> d_h2] (new hidden)
        f(x)  = rot_s2(s2(x)) @ w1               [d_h2 -> d]   (w1 plays W_c)

    One gradient step on -sum_i lr <v_i, f(k_i)> (same ascent-direction sign
    convention as the stock kernel), hand-derived backward, Muon
    orthogonalization + per-column weight renorm on ALL FOUR matrices.
    Addresses carry the lrs: k_i for w0/w2, rot(h1(k_i)) for wb,
    rot(s2(k_i)) for w1. Rotations backprop as their inverses (negated sin),
    mirroring fast_weight_swish_glu_hidden_rotary_apply.

    h1cos/h1sin: [B, L, P1] with 2*P1 <= d_h, or None (site-h1 disabled).
    s2cos/s2sin: [B, L, P2] with 2*P2 <= d_h2, or None (site-s2 disabled).
    The input (q/k) rotary site lives outside this kernel.
    """
    from lact_ttt import silu_backprop, zeropower_via_newtonschulz5

    w0_norm = w0.detach().norm(dim=1, keepdim=True)
    w2_norm = w2.detach().norm(dim=1, keepdim=True)
    wb_norm = wb.detach().norm(dim=1, keepdim=True)
    w1_norm = w1.detach().norm(dim=1, keepdim=True)

    output = []
    for start, end, update, apply in ttt_ua_order:
        w0_now, w2_now, wb_now, w1_now = w0, w2, wb, w1

        if update:
            ki, vi = k[:, start:end, :], v[:, start:end, :]
            lr0i = lr0[:, start:end, :]
            lr2i = lr2[:, start:end, :]
            lrbi = lrb[:, start:end, :]
            lr1i = lr1[:, start:end, :]

            gate_before_act = ki @ w0_now
            hidden_before_mul = ki @ w2_now
            # silu(gate) is used twice; it was recomputed over the full
            # [.., d_h] tensor. These kernels are not compiled (sp_all_reduce),
            # so nothing else removes it. Bit-exact.
            gate_act = F.silu(gate_before_act, inplace=False)
            h1 = gate_act * hidden_before_mul
            if h1cos is not None:
                h1r = apply_rotary_pairs(h1, h1cos[:, start:end, :], h1sin[:, start:end, :])
            else:
                h1r = h1
            z = h1r @ wb_now                    # second-hidden pre-activation
            s2 = F.silu(z, inplace=False)
            if s2cos is not None:
                s2r = apply_rotary_pairs(s2, s2cos[:, start:end, :], s2sin[:, start:end, :])
            else:
                s2r = s2

            # Backward of +<v, f(k)>; rotations invert with negated sin.
            ds2r = vi @ w1_now.transpose(-1, -2)
            if s2cos is not None:
                ds2 = apply_rotary_pairs(ds2r, s2cos[:, start:end, :], -s2sin[:, start:end, :])
            else:
                ds2 = ds2r
            dz = silu_backprop(ds2, z)
            dh1r = dz @ wb_now.transpose(-1, -2)
            if h1cos is not None:
                dh1 = apply_rotary_pairs(dh1r, h1cos[:, start:end, :], -h1sin[:, start:end, :])
            else:
                dh1 = dh1r
            dhidden_before_mul = dh1 * gate_act
            dgate = dh1 * hidden_before_mul
            dgate_before_act = silu_backprop(dgate, gate_before_act)

            w1_grad = zeropower_via_newtonschulz5(
                (s2r * lr1i).transpose(-1, -2) @ vi, muon_update_steps
            )
            wb_grad = zeropower_via_newtonschulz5(
                (h1r * lrbi).transpose(-1, -2) @ dz, muon_update_steps
            )
            w0_grad = zeropower_via_newtonschulz5(
                (ki * lr0i).transpose(-1, -2) @ dgate_before_act, muon_update_steps
            )
            w2_grad = zeropower_via_newtonschulz5(
                (ki * lr2i).transpose(-1, -2) @ dhidden_before_mul, muon_update_steps
            )
            w0_now = w0_now + w0_grad
            w2_now = w2_now + w2_grad
            wb_now = wb_now + wb_grad
            w1_now = w1_now + w1_grad

            w0_now = w0_now / (w0_now.norm(dim=1, keepdim=True) + 1e-5) * w0_norm
            w2_now = w2_now / (w2_now.norm(dim=1, keepdim=True) + 1e-5) * w2_norm
            wb_now = wb_now / (wb_now.norm(dim=1, keepdim=True) + 1e-5) * wb_norm
            w1_now = w1_now / (w1_now.norm(dim=1, keepdim=True) + 1e-5) * w1_norm

            w0, w2, wb, w1 = w0_now, w2_now, wb_now, w1_now

        if apply:
            qi = q[:, start:end, :]
            hq = F.silu(qi @ w0_now, inplace=True) * (qi @ w2_now)
            if h1cos is not None:
                hq = apply_rotary_pairs(hq, h1cos[:, start:end, :], h1sin[:, start:end, :])
            sq = F.silu(hq @ wb_now, inplace=False)
            if s2cos is not None:
                sq = apply_rotary_pairs(sq, s2cos[:, start:end, :], s2sin[:, start:end, :])
            output.append(sq @ w1_now)

    output = torch.cat(output, dim=1)
    return output, w0, w1, w2, wb


@torch.compile
def fast_weight_swiglu4l_weight_norm_apply(
    w0, w2, wb, wc, w1, q, k, v,
    lr0, lr2, lrb, lrc, lr1,
    h1cos, h1sin, s2cos, s2sin, s3cos, s3sin,
    ttt_ua_order: list,
    muon_update_steps: int = 0,
):
    """Depth-4 fast-weight net (Q2 depth-3 point extended by one layer).

        h1(x) = silu(x @ w0) * (x @ w2)          [d    -> d_h ]  (stock SwiGLU layer)
        s2(x) = silu(rot_h1(h1(x)) @ wb)         [d_h  -> d_h2]  (hidden 2)
        s3(x) = silu(rot_s2(s2(x)) @ wc)         [d_h2 -> d_h3]  (hidden 3, NEW vs fw3l)
        f(x)  = rot_s3(s3(x)) @ w1               [d_h3 -> d   ]  (w1 plays W_d)

    Identical in every other respect to fast_weight_swiglu3l_weight_norm_apply:
    one gradient step on -sum_i lr <v_i, f(k_i)> (ascent-direction sign
    convention of the stock kernel), hand-derived backward, Muon
    orthogonalization + per-column weight renorm on ALL FIVE matrices.
    Addresses carry the lrs: k_i for w0/w2, rot(h1(k_i)) for wb,
    rot(s2(k_i)) for wc, rot(s3(k_i)) for w1. Rotations backprop as their
    inverses (negated sin), mirroring the depth-3 kernel.

    h1cos/h1sin: [B, L, P1] with 2*P1 <= d_h, or None (site-h1 disabled).
    s2cos/s2sin: [B, L, P2] with 2*P2 <= d_h2, or None (site-s2 disabled).
    s3cos/s3sin: [B, L, P3] with 2*P3 <= d_h3, or None (site-s3 disabled).
    The input (q/k) rotary site lives outside this kernel.
    """
    from lact_ttt import silu_backprop, zeropower_via_newtonschulz5

    w0_norm = w0.detach().norm(dim=1, keepdim=True)
    w2_norm = w2.detach().norm(dim=1, keepdim=True)
    wb_norm = wb.detach().norm(dim=1, keepdim=True)
    wc_norm = wc.detach().norm(dim=1, keepdim=True)
    w1_norm = w1.detach().norm(dim=1, keepdim=True)

    output = []
    for start, end, update, apply in ttt_ua_order:
        w0_now, w2_now, wb_now, wc_now, w1_now = w0, w2, wb, wc, w1

        if update:
            ki, vi = k[:, start:end, :], v[:, start:end, :]
            lr0i = lr0[:, start:end, :]
            lr2i = lr2[:, start:end, :]
            lrbi = lrb[:, start:end, :]
            lrci = lrc[:, start:end, :]
            lr1i = lr1[:, start:end, :]

            gate_before_act = ki @ w0_now
            hidden_before_mul = ki @ w2_now
            h1 = F.silu(gate_before_act, inplace=False) * hidden_before_mul
            if h1cos is not None:
                h1r = apply_rotary_pairs(h1, h1cos[:, start:end, :], h1sin[:, start:end, :])
            else:
                h1r = h1
            z2 = h1r @ wb_now                   # second-hidden pre-activation
            s2 = F.silu(z2, inplace=False)
            if s2cos is not None:
                s2r = apply_rotary_pairs(s2, s2cos[:, start:end, :], s2sin[:, start:end, :])
            else:
                s2r = s2
            z3 = s2r @ wc_now                   # third-hidden pre-activation
            s3 = F.silu(z3, inplace=False)
            if s3cos is not None:
                s3r = apply_rotary_pairs(s3, s3cos[:, start:end, :], s3sin[:, start:end, :])
            else:
                s3r = s3

            # Backward of +<v, f(k)>; rotations invert with negated sin.
            ds3r = vi @ w1_now.transpose(-1, -2)
            if s3cos is not None:
                ds3 = apply_rotary_pairs(ds3r, s3cos[:, start:end, :], -s3sin[:, start:end, :])
            else:
                ds3 = ds3r
            dz3 = silu_backprop(ds3, z3)
            ds2r = dz3 @ wc_now.transpose(-1, -2)
            if s2cos is not None:
                ds2 = apply_rotary_pairs(ds2r, s2cos[:, start:end, :], -s2sin[:, start:end, :])
            else:
                ds2 = ds2r
            dz2 = silu_backprop(ds2, z2)
            dh1r = dz2 @ wb_now.transpose(-1, -2)
            if h1cos is not None:
                dh1 = apply_rotary_pairs(dh1r, h1cos[:, start:end, :], -h1sin[:, start:end, :])
            else:
                dh1 = dh1r
            dhidden_before_mul = dh1 * F.silu(gate_before_act, inplace=False)
            dgate = dh1 * hidden_before_mul
            dgate_before_act = silu_backprop(dgate, gate_before_act)

            w1_grad = zeropower_via_newtonschulz5(
                (s3r * lr1i).transpose(-1, -2) @ vi, muon_update_steps
            )
            wc_grad = zeropower_via_newtonschulz5(
                (s2r * lrci).transpose(-1, -2) @ dz3, muon_update_steps
            )
            wb_grad = zeropower_via_newtonschulz5(
                (h1r * lrbi).transpose(-1, -2) @ dz2, muon_update_steps
            )
            w0_grad = zeropower_via_newtonschulz5(
                (ki * lr0i).transpose(-1, -2) @ dgate_before_act, muon_update_steps
            )
            w2_grad = zeropower_via_newtonschulz5(
                (ki * lr2i).transpose(-1, -2) @ dhidden_before_mul, muon_update_steps
            )
            w0_now = w0_now + w0_grad
            w2_now = w2_now + w2_grad
            wb_now = wb_now + wb_grad
            wc_now = wc_now + wc_grad
            w1_now = w1_now + w1_grad

            w0_now = w0_now / (w0_now.norm(dim=1, keepdim=True) + 1e-5) * w0_norm
            w2_now = w2_now / (w2_now.norm(dim=1, keepdim=True) + 1e-5) * w2_norm
            wb_now = wb_now / (wb_now.norm(dim=1, keepdim=True) + 1e-5) * wb_norm
            wc_now = wc_now / (wc_now.norm(dim=1, keepdim=True) + 1e-5) * wc_norm
            w1_now = w1_now / (w1_now.norm(dim=1, keepdim=True) + 1e-5) * w1_norm

            w0, w2, wb, wc, w1 = w0_now, w2_now, wb_now, wc_now, w1_now

        if apply:
            qi = q[:, start:end, :]
            hq = F.silu(qi @ w0_now, inplace=True) * (qi @ w2_now)
            if h1cos is not None:
                hq = apply_rotary_pairs(hq, h1cos[:, start:end, :], h1sin[:, start:end, :])
            sq = F.silu(hq @ wb_now, inplace=False)
            if s2cos is not None:
                sq = apply_rotary_pairs(sq, s2cos[:, start:end, :], s2sin[:, start:end, :])
            tq = F.silu(sq @ wc_now, inplace=False)
            if s3cos is not None:
                tq = apply_rotary_pairs(tq, s3cos[:, start:end, :], s3sin[:, start:end, :])
            output.append(tq @ w1_now)

    output = torch.cat(output, dim=1)
    return output, w0, w1, w2, wb, wc


@torch.compile
def fast_weight_mlp2_weight_norm_apply(
    w0, w1, q, k, v, lr0, lr1,
    hcos, hsin,
    ttt_ua_order: list,
    muon_update_steps: int = 0,
):
    """Gateless 2-layer-MLP fast weights (inner-model generality control):

        f(x) = rot_h(silu(x @ w0)) @ w1

    Identical recipe to the stock SwiGLU kernel — one ascent step on
    sum_i lr <v_i, f(k_i)> with hand-derived backward, Muon
    orthogonalization, per-column weight-norm — but with the gate branch
    (w2) removed, so the inner model is a plain 2-layer MLP. The optional
    hidden rotary rotates the single hidden activation in both update and
    apply; backprop uses the inverse rotation (negated sin), mirroring
    fast_weight_swish_glu_hidden_rotary_apply. hcos/hsin: [B, L, P] with
    2P <= d_h, or None (rotary disabled).
    """
    from lact_ttt import silu_backprop, zeropower_via_newtonschulz5

    w0_norm = w0.detach().norm(dim=1, keepdim=True)
    w1_norm = w1.detach().norm(dim=1, keepdim=True)

    output = []
    for start, end, update, apply in ttt_ua_order:
        w0_now, w1_now = w0, w1

        if update:
            ki, vi = k[:, start:end, :], v[:, start:end, :]
            lr0i = lr0[:, start:end, :]
            lr1i = lr1[:, start:end, :]

            h_pre = ki @ w0_now
            h = F.silu(h_pre, inplace=False)
            if hcos is not None:
                h_rot = apply_rotary_pairs(h, hcos[:, start:end, :], hsin[:, start:end, :])
            else:
                h_rot = h

            # Backward of +<v, f(k)>; rotation inverts with negated sin.
            dh_rot = vi @ w1_now.transpose(-1, -2)
            if hcos is not None:
                dh = apply_rotary_pairs(dh_rot, hcos[:, start:end, :], -hsin[:, start:end, :])
            else:
                dh = dh_rot
            dh_pre = silu_backprop(dh, h_pre)

            w1_grad = zeropower_via_newtonschulz5(
                (h_rot * lr1i).transpose(-1, -2) @ vi, muon_update_steps
            )
            w0_grad = zeropower_via_newtonschulz5(
                (ki * lr0i).transpose(-1, -2) @ dh_pre, muon_update_steps
            )
            w0_now = w0_now + w0_grad
            w1_now = w1_now + w1_grad

            w0_now = w0_now / (w0_now.norm(dim=1, keepdim=True) + 1e-5) * w0_norm
            w1_now = w1_now / (w1_now.norm(dim=1, keepdim=True) + 1e-5) * w1_norm

            w0, w1 = w0_now, w1_now

        if apply:
            qi = q[:, start:end, :]
            hq = F.silu(qi @ w0_now, inplace=True)
            if hcos is not None:
                hq = apply_rotary_pairs(hq, hcos[:, start:end, :], hsin[:, start:end, :])
            output.append(hq @ w1_now)

    output = torch.cat(output, dim=1)
    return output, w0, w1


class CamFastWeightGluMLPMultihead(FastWeightGluMLPMultihead):
    # 'sharedf' cam_mode: one learnable gain tensor shared across all layers
    _sharedf_registry = {}
    _layer_counter = 0  # construction-order depth index (mip staggering)

    def __init__(
        self,
        dim: int,
        head_dim: int,
        cam_mode: str,
        inter_multi: int = 1,
        bias: bool = False,
        base_lr=0.01,
        muon_update_steps=0,
        num_freqs: int = 16,
        num_freqs_seg: int = 10,
        num_freqs_h: int = 21,
        num_registers: int = 4,
        rank: int = 8,
        omega_tilt: float = 0.0,
        prope_proj_frac: float = 0.5,
        omega_scale: float = 1.0,
        freeze_freqs: bool = False,
        phase_bias: bool = False,
        t_near: float = 0.05,
        t_far: float = 4.0,
        shell_r: float = 0.35,
        n_dirs: int = 6,
        num_freqs_hseg: int = 84,
        sweep_k: int = 4,
        asym_key: str = "chord",
        asym_query: str = "anchor",
        asym_k: int = 3,
        omega_scale_h: float = 1.0,
        oracle_noise: float = 0.0,
        cfr_gamma: float = 3.1,
        qh_kappa: float = 1.0,
        env_gamma: float = 1.0,
        num_freqs_epi: int = 21,
        num_freqs_hepi: int = 42,
        bf_alpha_h: int = 4,
        bf_coord: str = "alpha",
        lam_pairs: int = 8,
        epi_env: bool = True,
        vo_coords: str = "6d",
        fejer_h: bool = False,
        fejer_omega0: float = 0.5,
        bump_p: int = 96,
        bump_kappa: float = 2.0,
    ):
        super().__init__(dim, head_dim, inter_multi, bias, base_lr, muon_update_steps)
        self.cam_mode = cam_mode
        self.cam_modes = set(cam_mode.split("+"))
        known = {"qk_rope_cam", "qk_rope_camimg", "plucker_sinc", "point_rope", "pra_sinc", "vo_rel", "ogta", "h_ga", "rot_raw",
                 "prope_ttt", "prope_in", "gta_in", "prope_in_raw", "prope_raw", "prope_orig", "prope_imgrope", "cam_lr", "adaln_cam", "q_reinject", "cam_registers",
                 "hyper_init", "h_pra", "h_dpra", "cone_pra", "ms2",
                 "w0_mask", "omega_map", "m_scale", "res2", "mip", "h_strat",
                 "fw3l", "fw3l_rot2", "fw3l_rot3", "fw4l", "fw4l_rot4",
                 "mlp2", "mlp2_rot2",
                 "hnrot", "sharedf", "gate_rope", "content_rope",
                 # gObjaverse program (2026-08-31): 3D-point / object-shell addressing,
                 # oracle-depth diagnostics, hidden image ropes, hidden rotation action.
                 "shell_sinc", "shell_iso", "pt_gt", "pt_gt_in", "anchor_in", "foot_in", "sweep_in", "head_anchor", "asym_in", "gate_shell_rot",
                 "h_shell", "h_bump", "h_shell_iso", "h_pt_gt", "h_pt_gt_in", "h_anchor", "h_foot", "h_img", "h_rot",
                 "raygta", "rot_content", "od_coords", "vo_rope", "iso",
                 "hh_in", "hh_vo", "layer_pt", "h_layer_pt", "near_in", "h_near", "ff_vo",
                 "cfr_in", "vo_store", "h_qh", "epi_in", "h_epi", "bf_in", "h_bf", "h_lam"}
        unknown = self.cam_modes - known
        if unknown:
            raise ValueError(f"unknown cam_mode(s) {unknown}")

        def _gain(name, *shape):
            """Learnable ladder gain. With the 'sharedf' flag ONE parameter
            (per name/shape) is created by the first layer and reused by every
            later layer, so all layers train the same frequency spectrum.
            Registry is class-level: valid for the usual one-model-per-process
            train/eval scripts, not for building two different models in one
            process."""
            if "sharedf" not in self.cam_modes:
                return nn.Parameter(torch.ones(*shape))
            key = (name, tuple(shape))
            reg = CamFastWeightGluMLPMultihead._sharedf_registry
            if key not in reg:
                reg[key] = nn.Parameter(torch.ones(*shape))
            return reg[key]
        # Q2 depth-3 fast weights: standalone modes; rot2/rot3 reuse the stock
        # qk_rope_cam machinery for the input rotary site.
        self.fw3l = bool(self.cam_modes & {"fw3l", "fw3l_rot2", "fw3l_rot3"})
        if self.fw3l:
            assert len(self.cam_modes) == 1, "fw3l modes are standalone (no '+' combos)"
            if self.cam_modes & {"fw3l_rot2", "fw3l_rot3"}:
                self.cam_modes.add("qk_rope_cam")
        # Depth-4 fast weights (third point of the depth x rotary interaction):
        # standalone modes; fw4l_rot4 reuses the stock qk_rope_cam machinery for
        # the input site and adds ladders at h1 / s2 / s3.
        self.fw4l = bool(self.cam_modes & {"fw4l", "fw4l_rot4"})
        if self.fw4l:
            assert len(self.cam_modes) == 1, "fw4l modes are standalone (no '+' combos)"
            if "fw4l_rot4" in self.cam_modes:
                self.cam_modes.add("qk_rope_cam")
        # Gateless 2-layer-MLP fast weights (inner-model generality control):
        # standalone modes; mlp2_rot2 = input rotary (stock qk_rope_cam
        # machinery) + hidden rotary on the single hidden activation.
        self.mlp2 = bool(self.cam_modes & {"mlp2", "mlp2_rot2"})
        if self.mlp2:
            assert len(self.cam_modes) == 1, "mlp2 modes are standalone (no '+' combos)"
            if "mlp2_rot2" in self.cam_modes:
                self.cam_modes.add("qk_rope_cam")
        rotary_fams = {"qk_rope_cam", "plucker_sinc", "point_rope", "pra_sinc", "cone_pra",
                       "shell_sinc", "shell_iso", "pt_gt", "pt_gt_in", "anchor_in", "foot_in", "sweep_in", "head_anchor", "asym_in"}
        assert len(rotary_fams & self.cam_modes) <= 1, "only one rotary family at a time"
        matrix_fams = {"prope_raw", "prope_in_raw", "rot_raw", "prope_orig", "prope_imgrope",
                       "prope_ttt", "prope_in", "gta_in", "ogta", "raygta", "rot_content", "gate_shell_rot"}
        assert len(matrix_fams & self.cam_modes) <= 1, "only one matrix/transport family at a time"
        self.seg_in_modes = {"shell_sinc", "shell_iso", "pt_gt", "pt_gt_in", "anchor_in", "foot_in", "sweep_in", "head_anchor", "asym_in", "gate_shell_rot", "layer_pt", "near_in"}
        if "asym_in" in self.cam_modes:
            # ASYMMETRIC store/read codes (TTT-native, 2026-08-31): the KEY (stored) and the
            # QUERY (read) get DIFFERENT phase coordinates on the same K x 3 x F pair budget.
            #   'chord'  : sinc-integrated chord (same coefficients in every block)
            #   'foot'   : single closest-approach point (every block)
            #   'anchor' : block b = point at chord fraction (b+0.5)/K
            # e.g. key=chord, query=anchor => sum_b int_chord_i cos(w (p_{j,b} - x_i(t))) dt:
            # "does any of the query's K depth hypotheses lie on the key's chord" -- the
            # K-hypothesis read folded into the dimension budget, one apply, zero cost.
            # Attention cannot do this (q and k must share one code under softmax); under
            # the un-normalised Hebbian readout a broad/multi-point query INTEGRATES stored
            # mass instead of acting as a temperature.
            assert asym_key in ("chord", "foot", "anchor") and asym_query in ("chord", "foot", "anchor")
            self.asym_key, self.asym_query, self.asym_k = asym_key, asym_query, asym_k
        if "head_anchor" in self.cam_modes:
            # LAYERED MEMORY (2026-08-31): with H fast-weight heads, head k addresses the
            # 3D point at chord fraction (k+0.5)/H -- each head's fast weight becomes a
            # test-time scene memory for ONE depth layer (an MPI-like stack); the slow
            # weights after the layer (c_proj, next blocks) choose which depth to trust.
            # Zero extra compute, no depth prediction anywhere.
            assert self.num_heads >= 2, "head_anchor needs >= 2 fast-weight heads"
        self.seg_h_modes = {"h_shell", "h_shell_iso", "h_pt_gt", "h_pt_gt_in", "h_anchor", "h_foot", "h_layer_pt", "h_near"}
        self.n_anchor = 3   # fixed chord fractions 0.25 / 0.5 / 0.75 (H3b, no learned depth)
        hidden_fams = {"h_pra", "h_dpra", "h_strat", "h_img", "h_rot", "h_ga", "h_bump", "h_qh", "h_epi", "h_bf"} | self.seg_h_modes
        if "h_lam" in self.cam_modes:
            assert self.cam_modes & {"h_epi", "h_bf"}, "h_lam is a modifier of h_epi / h_bf"
        if "h_qh" in self.cam_modes:
            # QH -- quaternion half-angle hidden code (algebra agent P1): per-token unit
            # quaternion u_i for the geodesic rotation e -> n_i (foot direction), applied as
            # LEFT quaternion multiplication L_u on the 4-blocks of h (update and apply, via
            # the mat4 kernel). Coefficient multiplier = cos(Delta/2) with Delta ~ angle
            # between foot directions: NON-NEGATIVE (never subtracts a matched value),
            # monotone, wrap-free -- what the sign-sensitive linear hidden slot demands.
            # (spin-1/2 = the double cover, NOT a Wigner l>=2 irrep; flag for the user.)
            self.qh_kappa = nn.Parameter(torch.tensor(float(qh_kappa)))
        # env_gamma (algebra agent P3): raise the sinc envelope of segment codes to a learnable
        # power -- Muon re-amplifies mildly suppressed directions, so only deep nulls stick.
        self.env_gamma = nn.Parameter(torch.tensor(float(env_gamma))) if env_gamma != 1.0 else None
        # ---- Epipolar-plane codes (P2 program, 2026-09-01). Integer harmonics of phi (the ray's
        # epipolar-plane angle about the input baseline, see model.compute_camera_info): phi is
        # 2pi-periodic, so integer harmonics are wrap-free and need no scene-unit ladder. 'bf'
        # adds harmonics of the along-line angle (alpha, or the vergence-corrected psi_c) --
        # sharp at the input site (squared kernel), only a few low harmonics at the hidden site
        # (linear kernel: parallax must stay inside the first half-period). 'h_lam' adds a few
        # hidden pairs rotated by the camera's position along the baseline (nearer-view weighting).
        self.bf_coord = bf_coord; self.epi_env = epi_env
        if self.cam_modes & {"epi_in", "bf_in"}:
            P = num_freqs_epi
            assert 2 * P <= head_dim, (P, head_dim)
            if "bf_in" in self.cam_modes:
                Pp = P // 2; Pa = P - Pp
                self.register_buffer("m_alp_in", torch.arange(1, Pa + 1).float(), persistent=False)
                self.gain_alp_in = nn.Parameter(torch.ones(Pa))
            else:
                Pp = P
            self.register_buffer("m_epi_in", torch.arange(1, Pp + 1).float(), persistent=False)
            self.gain_epi_in = nn.Parameter(torch.ones(Pp))
        if self.cam_modes & {"h_epi", "h_bf"}:
            P = num_freqs_hepi
            self.register_buffer("m_epi_h", torch.arange(1, P + 1).float(), persistent=False)
            self.gain_epi_h = nn.Parameter(torch.ones(P))
            if "h_bf" in self.cam_modes:
                self.register_buffer("m_alp_h", torch.arange(1, bf_alpha_h + 1).float(), persistent=False)
                self.gain_alp_h = nn.Parameter(torch.ones(bf_alpha_h))
            if "h_lam" in self.cam_modes:
                self.gain_lam = nn.Parameter(torch.full((lam_pairs,), math.pi / 2))
        if "cfr_in" in self.cam_modes:
            # CFR -- Cayley Foot Rotation (overnight): per-token orthogonal rotation about the
            # foot DIRECTION u by the bounded angle 2 atan(gamma*rho/2) (rho = |x_c - p*|).
            # Relative product R_j^T R_i = I iff the foot points coincide (matched pairs at any
            # view separation), angle < pi always (wrap-free), one learnable scalar. The
            # matched-identity MATRIX that merges L1's coordinate with L4's input-site matrix
            # preference; applied to q/k 3-blocks (and v/o when combined with vo_rel).
            assert not (self.cam_modes & {"prope_raw", "rot_raw", "prope_in_raw", "prope_orig",
                                          "prope_imgrope", "raygta", "rot_content", "gate_shell_rot",
                                          "hh_in", "qk_rope_cam"})
            self.cfr_gamma = nn.Parameter(torch.tensor(float(cfr_gamma)))
        if "vo_store" in self.cam_modes:
            # STORE-ONLY carrier: v <- R_i v on update, NO o-side map (values canonicalised into
            # the world frame; the slow c_proj reads world-frame outputs). Tests whether the
            # transport must be closed (both-sided) -- only TTT lets the o-map differ from the
            # v-map at all.
            assert not (self.cam_modes & {"vo_rel", "vo_rope", "ff_vo", "hh_vo"})
        if self.cam_modes & {"hh_in", "hh_vo"}:
            # HOUSEHOLDER PE (overnight 2026-09-01, non-rotary): per-token orthogonal reflection
            # H = I - 2 n n^T on every 3-block, n = unit(x_c - p*) (the FOOT DIRECTION -- matched
            # pairs share x_c, hence n, hence H_j H_i ~ I; ray directions would not). hh_in:
            # address (q,k); hh_vo: carrier (v on update, o on apply; H is its own inverse).
            # Degeneracy: central rays have x_c ~ p* -> n noisy; blended toward the camera-from-
            # focus direction below a radius eps. Parameter-free.
            assert not (self.cam_modes & ({"prope_raw", "rot_raw", "prope_in_raw", "prope_orig",
                                           "prope_imgrope", "raygta", "rot_content",
                                           "gate_shell_rot"} | ({"vo_rel", "vo_rope"} if "hh_vo" in self.cam_modes else set())))
        assert len(hidden_fams & self.cam_modes) <= 1, "one hidden-site mechanism at a time"
        if self.cam_modes & {"ms2", "res2"}:
            assert {"h_pra", "h_dpra"} & self.cam_modes, "ms2/res2 require a hidden rotary mode"
        assert not ({"ms2", "res2"} <= self.cam_modes), "ms2 and res2 are exclusive"
        if "hnrot" in self.cam_modes:
            # hnrot = RMS-normalize the hidden's rotated dims before the hidden
            # rotary; implemented only in the plain single-step hidden-rotary
            # kernel (mirrors LLM ttt_hrope_hnorm='rms_rot', which excludes the
            # delta path).
            assert {"h_pra", "h_strat"} & self.cam_modes, \
                "hnrot requires an h_pra-family hidden rotary (h_pra/h_strat)"
            assert not (self.cam_modes & {"ms2", "res2"}), \
                "hnrot is only implemented in the plain hidden-rotary kernel (no ms2/res2)"
        # GbR single-branch input rotary (LLM Q25b port): route the rotated
        # q/k to only ONE SwiGLU input branch; the other gets the plain copy.
        self.branch_rope = bool(self.cam_modes & {"gate_rope", "content_rope"})
        if self.branch_rope:
            assert "qk_rope_cam" in self.cam_modes, \
                "gate_rope/content_rope are modifiers of the input rotary (require qk_rope_cam)"
            assert not ({"gate_rope", "content_rope"} <= self.cam_modes), \
                "gate_rope and content_rope are mutually exclusive"
            assert not (self.fw3l or self.fw4l or self.mlp2), \
                "gate_rope/content_rope not implemented for fw3l/fw4l/mlp2 kernels"
            assert not (self.cam_modes & {"h_pra", "h_dpra", "h_strat", "ms2",
                                          "res2", "cam_registers"}), \
                "gate_rope/content_rope only implemented in the plain-kernel path"
        self.head_dim = head_dim
        self.num_freqs = num_freqs
        # oracle_noise (calibration, 2026-09-01): Gaussian noise (std, canonical units) added to
        # the GT ray parameter of the oracle modes -- "what is an estimator with error sigma worth?"
        self.oracle_noise = float(oracle_noise)
        d_h = int(head_dim * inter_multi)

        if "ogta" in self.cam_modes:
            # Q38 (user proposal 2026-08-07): fully ORTHOGONAL group action for the
            # fast-weight address space. The projective/SE(3) transforms are affine,
            # not orthogonal, so they distort address norms -- and the whole TTT
            # stack (q/k L2-norm, Muon orthogonalisation, per-column weight-norm)
            # lives on spheres. Here the per-view matrix is block-diagonal
            # orthogonal: [R (exact c2w rotation, 3x3)] + [SO(2)(w_u t_x)] +
            # [SO(2)(w_u t_y)] + [SO(2)(w_u t_z)] = one 9-dim unit, tiled
            # head_dim//9 times (28 units = 252 of 256 dims, mirroring F21).
            # Rotation enters EXACTLY (the unbounded, wrap-prone part at wide
            # baselines); translation enters as phases, but scene normalisation
            # bounds |t| <= 1, so with the ladder capped at pi/2 the phase
            # difference |w (t_i - t_j)| <= pi NEVER wraps, by construction.
            n_units = head_dim // 9
            assert n_units >= 1
            omega_t = torch.logspace(
                math.log2(math.pi / 32), math.log2(math.pi / 2), n_units, base=2.0)
            self.register_buffer("ogta_omega_t", omega_t, persistent=False)

        if "qk_rope_camimg" in self.cam_modes:
            # PRoPE-style split: half the rotary budget on the 6 Plucker coords
            # (which view a token came from), half on the 2 in-view patch
            # coordinates (where inside that view it sits). F34 found PRoPE's
            # entire gain in this stack came from its image-coordinate ropes
            # (+0.379) while its projective transform cost -0.294, yet our own
            # ladder spends 100% on camera and nothing on image position.
            # Pair count is held EQUAL to qk_rope_cam so the two are
            # budget-matched: 6*F_cam + 2*F_img = 6*num_freqs.
            assert "qk_rope_cam" not in self.cam_modes, \
                "qk_rope_camimg replaces qk_rope_cam; enable only one"
            F_cam = max(1, num_freqs // 2)
            F_img = (6 * num_freqs - 6 * F_cam) // 2
            assert F_img >= 1 and 2 * (6 * F_cam + 2 * F_img) <= head_dim, \
                f"split budget 6*{F_cam}+2*{F_img} does not fit head_dim {head_dim}"
            self.n_freqs_cam, self.n_freqs_img = F_cam, F_img
            omega_ci = math.pi * torch.logspace(
                math.log2(0.5), math.log2(16.0), F_cam, base=2.0)
            omega_im = math.pi * torch.logspace(
                math.log2(0.5), math.log2(16.0), F_img, base=2.0)
            self.register_buffer("omega_ci", omega_ci, persistent=False)
            self.register_buffer("omega_im", omega_im, persistent=False)
            self.gain_ci = _gain("gain_ci", 6, F_cam)
            self.gain_im = _gain("gain_im", 2, F_img)

        if "vo_rope" in self.cam_modes:
            # PHASE-FORM CARRIER TRANSPORT (user request 2026-08-31): rotate v by the KEY token's
            # 6-coordinate phases on update and un-rotate o by the QUERY token's phases on apply,
            # so the retrieved value carries e^{i(theta_i - theta_j)} -- RayRoPE's v/o wiring,
            # with our coordinates ((d, o x d) or, with od_coords, (o, d)). Same ladder/budget as
            # the input rotary (6 x num_freqs pairs on the 256-d value), learnable gains.
            assert not (self.cam_modes & {"vo_rel", "prope_raw", "rot_raw", "prope_orig",
                                          "prope_imgrope", "prope_ttt", "raygta", "rot_content",
                                          "gate_shell_rot"}), "vo_rope excludes matrix carriers"
            # vo_coords "6d": the 6 coordinates ((d, m) or (o, d)); "d": the ray DIRECTION only
            # (3 coords, twice the rungs -> same 252-dim budget) -- the user's "camera ray only"
            # carrier, the phase analogue of the rotation-matrix transport (d transforms with R).
            assert vo_coords in ("6d", "d", "foot")
            self.vo_coords = vo_coords
            n_c, F_vo = (6, num_freqs) if vo_coords == "6d" else (3, 2 * num_freqs)   # d/foot: 3 coords
            assert 2 * n_c * F_vo <= head_dim
            self.register_buffer("omega_vo", math.pi * torch.logspace(
                math.log2(0.5), math.log2(16.0), F_vo, base=2.0) * omega_scale, persistent=False)
            self.gain_vo = _gain("gain_vo", n_c, F_vo)

        if "qk_rope_cam" in self.cam_modes:
            # 6 Plucker coords x num_freqs pairs.
            assert 2 * 6 * num_freqs <= head_dim
            omega = math.pi * torch.logspace(
                math.log2(0.5), math.log2(16.0), num_freqs, base=2.0
            ) * omega_scale
            # omega_scale (Q39, user hypothesis 2026-08-07): frequencies do not LEARN
            # their way out of wrapping (F20/F29/F37 init lottery; F55-corr: learnable
            # gains sat at -0.41 on gObjaverse), so wrap-freedom must come from the
            # INIT. 1/32 puts the band at [pi/64, pi/2]: with |dc| <= 2 between views
            # no rung can wrap, by construction -- the ladder analogue of ogta's cap.
            self.register_buffer("omega", omega, persistent=False)
            if freeze_freqs:
                # user decision 2026-08-07: FREQUENCY LEARNING OFF -- the init is the
                # spectrum. (The learnable gain never rescued a bad band anyway.)
                self.register_buffer("freq_gain", torch.ones(6, num_freqs),
                                     persistent=False)
            else:
                self.freq_gain = _gain("freq_gain", 6, num_freqs)

        if "pra_sinc" in self.cam_modes:
            # Split rotary budget: Plucker line identity (6 x num_freqs pairs)
            # + sinc-integrated ray segment (3 x num_freqs_seg pairs).
            assert 2 * (6 * num_freqs + 3 * num_freqs_seg) <= head_dim
            omega_line = math.pi * torch.logspace(
                math.log2(0.5), math.log2(16.0), num_freqs, base=2.0
            )
            omega_seg = math.pi * torch.logspace(
                math.log2(0.5), math.log2(16.0), num_freqs_seg, base=2.0
            )
            self.register_buffer("omega_line", omega_line, persistent=False)
            self.register_buffer("omega_seg", omega_seg, persistent=False)
            self.gain_line = nn.Parameter(torch.ones(6, num_freqs))
            self.gain_seg = nn.Parameter(torch.ones(3, num_freqs_seg))
            self.t_near, self.t_far = t_near, t_far

        # ---- 3D-point / object-shell addressing (gObjaverse program, 2026-08-31) ----
        # Phase coordinate = a 3D POINT on the token's ray instead of the ray's Plucker
        # line: two rays that see the same surface point from 90 deg apart have a zero
        # phase difference there (the line coordinates differ by O(1)). Depth is unknown,
        # so the phase is INTEGRATED over the chord of the ray through a sphere of radius
        # r (learnable) around the scene focus point p* (LS intersection of the input
        # optical axes = the object centre on look-at renders): closed form
        #   int cos(w u.x(t)) dt over the chord = sinc(w u.(h d)) cos(w u.x_mid)
        # per unit direction u; rays missing the sphere use [t_c - r, t_c + r].
        #   shell_sinc / h_shell : u in {x, y, z} (axis-separable kernel, as plucker_sinc)
        #   shell_iso / h_shell_iso : u = the 6 icosahedral axes (isotropic 3D kernel)
        #   pt_gt / h_pt_gt : ORACLE -- t1 = t2 = GT surface depth (env = 1) where a
        #       surface exists, chord otherwise; *_in = GT for input tokens only.
        # Input-site variants re-L2-normalise q/k after the rotary (the envelope shrinks
        # norms token-dependently; F3). Hidden variants reuse the h-PRA kernel.
        def _dirs(n):
            if n == 3:
                return torch.eye(3)
            if n == 6:
                phi = (1.0 + 5 ** 0.5) / 2.0
                d = torch.tensor([[0, 1, phi], [0, 1, -phi], [1, phi, 0],
                                  [1, -phi, 0], [phi, 0, 1], [phi, 0, -1]], dtype=torch.float32)
                return d / d.norm(dim=-1, keepdim=True)
            i = torch.arange(n, dtype=torch.float32) + 0.5
            th = torch.acos(1 - i / n); ph = math.pi * (1 + 5 ** 0.5) * i
            return torch.stack([th.sin() * ph.cos(), th.sin() * ph.sin(), th.cos()], -1)
        if self.cam_modes & ((self.seg_in_modes | self.seg_h_modes) - {"foot_in", "h_foot"}):  # layer_pt keeps the chord radius
            # (foot modes use no chord -> no radius; an unused parameter would trip DDP)
            self.shell_r_raw = nn.Parameter(torch.tensor(float(shell_r)))
        if self.cam_modes & self.seg_in_modes:
            nd = n_dirs if (self.cam_modes & {"shell_iso", "iso"}) else 3   # 'iso' = icosahedral dirs for any seg mode
            mult = self.n_anchor if "anchor_in" in self.cam_modes else (asym_k if "asym_in" in self.cam_modes else 1)
            assert 2 * nd * num_freqs_seg * mult <= head_dim, (nd, num_freqs_seg, mult, head_dim)
            self.register_buffer("dirs_in", _dirs(nd), persistent=False)
            self.register_buffer("omega_seg3", math.pi * torch.logspace(
                math.log2(0.5), math.log2(16.0), num_freqs_seg, base=2.0) * omega_scale,
                persistent=False)
            self.gain_seg3 = _gain("gain_seg3", nd, num_freqs_seg)
        if self.cam_modes & self.seg_h_modes:
            nd = n_dirs if (self.cam_modes & {"h_shell_iso", "iso"}) else 3
            mult = self.n_anchor if "h_anchor" in self.cam_modes else 1
            if fejer_h:
                # FEJER ladder (2026-08-31): harmonic rungs n*w0 (n = 0..N0-1) with triangular
                # multiplicity (N0 - n) => the per-direction kernel is the Fejer kernel,
                # pointwise NON-NEGATIVE. Only a linear (Hebbian) readout cares about the sign
                # of its address kernel -- a wrapped hidden phase SUBTRACTS another view's
                # value -- so this is a hidden-site-specific fix with no attention analogue.
                # n = 0 rungs are unrotated (content-pure) pairs.
                N0 = int((math.sqrt(8 * num_freqs_hseg + 1) - 1) / 2)
                lad = [n * fejer_omega0 * math.pi for n in range(N0) for _ in range(N0 - n)]
                num_freqs_hseg = len(lad)
                omega_h_l = torch.tensor(lad, dtype=torch.float32)
            else:
                omega_h_l = math.pi * torch.logspace(
                    math.log2(0.5), math.log2(16.0), num_freqs_hseg, base=2.0) * omega_scale_h
            assert 2 * nd * num_freqs_hseg * mult <= d_h, (nd, num_freqs_hseg, mult, d_h)
            self.register_buffer("dirs_h", _dirs(nd), persistent=False)
            self.register_buffer("omega_hseg", omega_h_l, persistent=False)
            self.gain_hseg = _gain("gain_hseg", nd, num_freqs_hseg)
        if "h_bump" in self.cam_modes:
            # HIDDEN BUMP CODE (2026-08-31): amplitude, not phase. Hidden pair p is scaled by
            # a_p(u) = exp(-kappa (1 - u . c_p)), u = unit direction focus -> token's camera,
            # c_p fixed Fibonacci centres on the sphere. <a_j h_j, a_i h_i> ~ vMF(angle_ij)
            # x <h_j, h_i>: a positive, monotone, wrap-free view-proximity kernel on the
            # dominant channel (a soft partition of hidden units by viewing direction).
            # Runs on the stock hidden-rotary kernel with hsin = 0 (diagonal map; its
            # backward transpose is exact). q/k are L2-normalised so amplitude codes are
            # squashed there; h is not -- hidden-site specific.
            assert 2 * bump_p <= d_h
            self.register_buffer("bump_centres", _dirs(bump_p), persistent=False)
            self.bump_kappa = nn.Parameter(torch.tensor(float(bump_kappa)))
        if "h_img" in self.cam_modes:
            # Hidden-site IMAGE-coordinate rotary: 2 in-view patch coords x num_freqs_h
            # pairs. Tax-free across views by construction (same coordinates in every
            # view); F34/F56: image ropes are the one phase code positive at both ends
            # of the baseline axis, but they were only ever placed at the input site.
            assert 2 * 2 * num_freqs_h <= d_h
            self.register_buffer("omega_himg", math.pi * torch.logspace(
                math.log2(0.5), math.log2(16.0), num_freqs_h, base=2.0), persistent=False)
            self.gain_himg = _gain("gain_himg", 2, num_freqs_h)

        if "sweep_in" in self.cam_modes:
            # DEPTH-SWEEP READOUT (TTT-native, 2026-08-31). The fast weight is a function
            # over the address space, so a TARGET token can query it at several 3D points
            # along its own ray: K point-coded copies of q_j at chord fractions (k+0.5)/K,
            # plus the chord-integrated read (k = 0), are all applied, and the K+1 readouts
            # are mixed by a softmax over a zero-init linear probe of each readout ("let the
            # memory's own retrieval pick the depth"). Keys/values are stored with the
            # chord code as in shell_sinc; input-token reads are unchanged. Attention has
            # no equivalent short of K extra full passes; here it costs K extra MLP
            # evaluations on the target tokens only. Zero-init probe => starts as the
            # uniform mixture (close to shell_sinc).
            assert not (self.cam_modes & (hidden_fams | {"prope_raw", "rot_raw", "prope_in_raw",
                                                          "prope_orig", "prope_imgrope", "raygta",
                                                          "rot_content", "gate_rope", "content_rope"})), \
                "sweep_in: first version supports the plain kernel (+ vo_rel) only"
            self.sweep_k = sweep_k
            self.sweep_probe = nn.Linear(head_dim, 1)
            nn.init.zeros_(self.sweep_probe.weight); nn.init.zeros_(self.sweep_probe.bias)

        if {"h_pra", "h_dpra"} & self.cam_modes:
            # Hidden-space Plucker rotary: 6 coords x num_freqs_h pairs in d_h.
            assert 2 * 6 * num_freqs_h <= d_h
            omega_h = math.pi * torch.logspace(
                math.log2(0.5), math.log2(16.0), num_freqs_h, base=2.0
            )
            self.register_buffer("omega_h", omega_h, persistent=False)
            self.gain_h = _gain("gain_h", 6, num_freqs_h)

        if self.fw3l:
            # Depth-3 inner net: f(x) = w1( rot_s2( silu( wb( rot_h1(
            # silu(x w0) * (x w2) ))))). wb is the new [d_h -> d_h2] fast
            # weight (d_h2 = d_h); w1 keeps its [d_h, d] shape and becomes
            # the output matrix W_c reading the second hidden s2.
            d_h2 = d_h
            self.d_h2 = d_h2
            self.wb = nn.Parameter(
                torch.randn(self.num_heads, d_h, d_h2) * math.sqrt(2) / math.sqrt(d_h)
            )
            # 4th per-token lr channel for wb (same softplus/base_lr machinery).
            self.lr_fc = nn.Linear(dim, self.lr_dim * 4)
            if "fw3l_rot3" in self.cam_modes:
                # Site-h1 Plucker ladder: 6 coords x num_freqs_h pairs in d_h.
                assert 2 * 6 * num_freqs_h <= d_h
                omega_h1 = math.pi * torch.logspace(
                    math.log2(0.5), math.log2(16.0), num_freqs_h, base=2.0
                )
                self.register_buffer("omega_h1", omega_h1, persistent=False)
                self.gain_h1 = nn.Parameter(torch.ones(6, num_freqs_h))
            if self.cam_modes & {"fw3l_rot2", "fw3l_rot3"}:
                # Site-s2 Plucker ladder: 6 coords x num_freqs_h pairs in d_h2.
                assert 2 * 6 * num_freqs_h <= d_h2
                omega_s2 = math.pi * torch.logspace(
                    math.log2(0.5), math.log2(16.0), num_freqs_h, base=2.0
                )
                self.register_buffer("omega_s2", omega_s2, persistent=False)
                self.gain_s2 = nn.Parameter(torch.ones(6, num_freqs_h))

        if self.fw4l:
            # Depth-4 inner net: f(x) = w1( rot_s3( silu( wc( rot_s2( silu( wb(
            # rot_h1( silu(x w0) * (x w2) )))))))). One more hidden matrix (wc)
            # than fw3l; w1 stays the [d_h, d] output matrix (W_d).
            #
            # WIDTH: every internal width is kept at d_h = head_dim*inter_multi
            # (d_h2 = d_h3 = d_h), exactly the rule fw3l uses (d_h2 = d_h).
            # Depth is therefore the ONLY thing that changes across the
            # 2L/3L/4L ladder -- no width is re-tuned, and w1 keeps its stock
            # [d_h -> d] shape at every depth.
            d_h2 = d_h3 = d_h
            self.d_h2, self.d_h3 = d_h2, d_h3
            self.wb = nn.Parameter(
                torch.randn(self.num_heads, d_h, d_h2) * math.sqrt(2) / math.sqrt(d_h)
            )
            self.wc = nn.Parameter(
                torch.randn(self.num_heads, d_h2, d_h3) * math.sqrt(2) / math.sqrt(d_h2)
            )
            # 5 per-token lr channels (w0, w2, wb, wc, w1); same softplus/base_lr.
            self.lr_fc = nn.Linear(dim, self.lr_dim * 5)
            if "fw4l_rot4" in self.cam_modes:
                # One Plucker ladder per internal address space (h1, s2, s3);
                # the input site is the stock qk_rope_cam ladder.
                assert 2 * 6 * num_freqs_h <= d_h
                assert 2 * 6 * num_freqs_h <= d_h2
                assert 2 * 6 * num_freqs_h <= d_h3
                for name in ("omega_h1", "omega_s2", "omega_s3"):
                    self.register_buffer(
                        name,
                        math.pi * torch.logspace(
                            math.log2(0.5), math.log2(16.0), num_freqs_h, base=2.0
                        ),
                        persistent=False,
                    )
                self.gain_h1 = nn.Parameter(torch.ones(6, num_freqs_h))
                self.gain_s2 = nn.Parameter(torch.ones(6, num_freqs_h))
                self.gain_s3 = nn.Parameter(torch.ones(6, num_freqs_h))

        if self.mlp2:
            # Remove the gate branch: f(x) = silu(x w0) w1. Param parity with
            # the SwiGLU x2 layer at inter_multi=3 (2 x d x 3d = 3 x d x 2d).
            # Two per-token lr channels instead of three.
            del self.w2
            self.lr_fc = nn.Linear(dim, self.lr_dim * 2)
            if "mlp2_rot2" in self.cam_modes:
                # Hidden Plucker ladder fills the d_h budget (6 coords x F_h pairs).
                assert 2 * 6 * num_freqs_h <= d_h
                omega_mh = math.pi * torch.logspace(
                    math.log2(0.5), math.log2(16.0), num_freqs_h, base=2.0
                )
                self.register_buffer("omega_mh", omega_mh, persistent=False)
                self.gain_mh = nn.Parameter(torch.ones(6, num_freqs_h))

        self.layer_idx = CamFastWeightGluMLPMultihead._layer_counter
        CamFastWeightGluMLPMultihead._layer_counter += 1
        if "mip" in self.cam_modes:
            # Half-octave per-layer stagger of both ladders: union spectral
            # support of the 6-layer stack becomes ~1/3-octave spaced.
            stag = 2.0 ** (((self.layer_idx % 3) - 1) / 3.0)
            with torch.no_grad():
                if hasattr(self, "omega"):
                    self.omega.mul_(stag)
                if hasattr(self, "omega_h"):
                    self.omega_h.mul_(stag)

        if "h_strat" in self.cam_modes:
            # Depth-stratified orthogonal 3D-point rotary in hidden space:
            # 6 log-spaced depth slices x 3 axes x 7 freqs = 126 pairs (252/512).
            # Sum of per-slice point kernels: rays crossing near slice t_s keep
            # that slice coherent; others decorrelate. point_rope without the
            # depth head; plucker_sinc without the (F3-unsafe) envelope.
            n_sl, n_f = 6, 7
            assert 2 * n_sl * 3 * n_f <= d_h
            t_sl = torch.logspace(math.log10(0.05), math.log10(4.0), n_sl)
            base = math.pi * torch.logspace(0.0, 4.0, n_f, base=2.0)  # pi*[1,16]
            om = torch.zeros(n_sl * 3, n_f)
            for si in range(n_sl):
                om[si * 3 : (si + 1) * 3] = base[None] / (0.5 + t_sl[si])
            self.register_buffer("t_strat", t_sl, persistent=False)
            self.register_buffer("omega_strat", om, persistent=False)
            self.gain_strat = nn.Parameter(torch.ones(n_sl * 3, n_f))

        if "cone_pra" in self.cam_modes:
            # Ray-cone anti-aliased line rotary: extended ladder, sinc envelope
            # from the patch's per-coordinate footprint, post-rotary re-norm.
            assert 2 * 6 * num_freqs <= head_dim
            omega_c = math.pi * torch.logspace(
                math.log2(0.5), math.log2(64.0), num_freqs, base=2.0
            )
            self.register_buffer("omega_cone", omega_c, persistent=False)
            self.gain_cone = nn.Parameter(torch.ones(6, num_freqs))

        if self.cam_modes & {"ms2", "res2"}:
            # Per-step, per-matrix post-Muon write gains (step 2 starts small).
            self.step_gains = nn.Parameter(
                torch.tensor([[1.0, 1.0, 1.0], [0.3, 0.3, 0.3]])
            )
            if "res2" in self.cam_modes:
                self.res_alpha = nn.Parameter(torch.zeros(1))

        if "w0_mask" in self.cam_modes:
            # Content-only W^0: zero the rotated input rows of w0/w2 and the
            # rotated hidden rows of w1 -> exact all-orders phase invariance
            # of every W^0 pathway (leaks L0/L1/L2), stock kernels unchanged.
            # Alive rows are rescaled to compensate the lost input power.
            assert {"h_pra", "h_dpra"} & self.cam_modes and                 ("qk_rope_cam" in self.cam_modes or "cone_pra" in self.cam_modes)
            n_rot_in = 2 * 6 * num_freqs
            n_rot_h = 2 * 6 * num_freqs_h
            m_in = torch.full((1, head_dim, 1), 2.0)
            m_in[:, :n_rot_in] = 0.0
            m_h = torch.full((1, d_h, 1), math.sqrt(d_h / max(d_h - n_rot_h, 1)))
            m_h[:, :n_rot_h] = 0.0
            self.register_buffer("w0_mask_in", m_in, persistent=False)
            self.register_buffer("w1_mask_h", m_h, persistent=False)

        if "omega_map" in self.cam_modes:
            # Learnable 6->P linear phase maps (zero-init delta): atoms may
            # leave the coordinate axes; relativity exact by construction.
            # omega_tilt > 0: random off-axis init, scaled per-row by the
            # base atom radius (breaks axis alignment immediately).
            def d_omega(P_rows, omega, gain):
                d = torch.zeros(P_rows, 6)
                if omega_tilt > 0:
                    radius = (omega[None, :] * gain).reshape(-1, 1)  # [P, 1]
                    d = torch.randn(P_rows, 6) * omega_tilt * radius
                return nn.Parameter(d)
            if "qk_rope_cam" in self.cam_modes:
                self.dOmega = d_omega(6 * num_freqs, self.omega, torch.ones(6, num_freqs))
            if {"h_pra", "h_dpra"} & self.cam_modes:
                self.dOmega_h = d_omega(6 * num_freqs_h, self.omega_h, torch.ones(6, num_freqs_h))
            if phase_bias:
                # Constant per-pair offsets: cancel exactly in phase
                # differences (relative kernel untouched); only re-frame the
                # functional absolute-phase interaction with W^0 (F12).
                if "qk_rope_cam" in self.cam_modes:
                    self.phase_b = nn.Parameter(torch.zeros(6 * num_freqs))
                if {"h_pra", "h_dpra"} & self.cam_modes:
                    self.phase_b_h = nn.Parameter(torch.zeros(6 * num_freqs_h))

        if self.cam_modes & {"plucker_sinc", "point_rope"}:
            # 3 spatial coords x num_freqs pairs, sinc-enveloped segment rotary.
            assert 2 * 3 * num_freqs <= head_dim
            omega = math.pi * torch.logspace(
                math.log2(0.5), math.log2(16.0), num_freqs, base=2.0
            )
            self.register_buffer("omega", omega, persistent=False)
            self.freq_gain = nn.Parameter(torch.ones(3, num_freqs))
            self.t_near, self.t_far = t_near, t_far
            if "point_rope" in self.cam_modes:
                self.depth_head = nn.Linear(dim, 2)
                nn.init.zeros_(self.depth_head.weight)
                with torch.no_grad():
                    # bias -> t_mid = 1.0, log-space half-width sigma ~ 1.5
                    self.depth_head.bias.copy_(torch.tensor([0.0, 1.5]))

        if "vo_rel" in self.cam_modes:
            pass  # parameter-free

        self.prope_proj_frac = prope_proj_frac
        if self.cam_modes & {"prope_ttt", "prope_in", "gta_in", "prope_in_raw", "prope_raw", "prope_orig", "prope_imgrope"}:
            assert head_dim % 8 == 0
            assert 0.0 < prope_proj_frac < 1.0 and (int(head_dim * prope_proj_frac) // 8 * 8) % 4 == 0

        if "cam_lr" in self.cam_modes:
            self.lr_cam = zero_init(nn.Linear(12, 3 * self.num_heads))

        if "adaln_cam" in self.cam_modes:
            self.cam_mlp = nn.Sequential(nn.Linear(11, 64), nn.SiLU())
            self.film_g = zero_init(nn.Linear(64, dim))
            self.film_b = zero_init(nn.Linear(64, dim))

        if "q_reinject" in self.cam_modes:
            self.cam_mlp = nn.Sequential(nn.Linear(11, 64), nn.SiLU())
            self.q_cam = zero_init(nn.Linear(64, dim))

        if "cam_registers" in self.cam_modes:
            self.num_registers = num_registers
            self.reg_mlp = nn.Sequential(nn.Linear(11, 64), nn.SiLU())
            self.reg_k = nn.Linear(64, num_registers * dim)
            self.reg_v = zero_init(nn.Linear(64, num_registers * dim))
            self.reg_lr = nn.Parameter(torch.zeros(3))

        if "hyper_init" in self.cam_modes:
            self.rank = rank
            self.set_enc = nn.Sequential(nn.Linear(11, 64), nn.SiLU(), nn.Linear(64, 64))
            self.s_head = zero_init(nn.Linear(128, 3 * rank))
            def lowrank(d_in, d_out):
                return nn.Parameter(torch.randn(d_in, rank) * 0.02), nn.Parameter(
                    torch.randn(rank, d_out) * 0.02
                )
            self.U0, self.V0 = lowrank(head_dim, d_h)
            self.U1, self.V1 = lowrank(d_h, head_dim)
            self.U2, self.V2 = lowrank(head_dim, d_h)



    # ---------- helpers ----------

    def _coords6(self, info):
        """[b, L, 6] Plucker coords; optionally per-scene moment whitening.
        od_coords (user question 2026-08-31): use (camera origin o, direction d) instead
        of the Plucker (d, o x d) -- a control for 'is it the moment specifically?'."""
        if "od_coords" in self.cam_modes:
            return torch.cat([info["tok_o"], info["tok_d"]], dim=-1)
        tok_m = info["tok_m"]
        if "m_scale" in self.cam_modes:
            if "_m_scale" not in info:
                n_in = info["ttt_op_order"][0].end
                s_m = info["tok_m"][:, :n_in].norm(dim=-1).median(dim=1).values
                info["_m_scale"] = (s_m.clamp_min(0.05))[:, None, None].detach()
            tok_m = tok_m / info["_m_scale"]
        return torch.cat([info["tok_d"], tok_m], dim=-1)

    def _rope_coeffs_camimg(self, info):
        """cos/sin for the camera+image split rotary. Returns heads-shaped
        coeffs over 6*F_cam + 2*F_img pairs: the Plucker block first, then the
        in-view patch block."""
        cam = self._coords6(info)                        # [b, L, 6]
        img = info["tok_uv"].to(cam.dtype)               # [b, L, 2] in [-1, 1]
        th_c = (cam[..., None] * (self.omega_ci[None, None, None]
                                  * self.gain_ci[None, None])).flatten(2)
        th_i = (img[..., None] * (self.omega_im[None, None, None]
                                  * self.gain_im[None, None])).flatten(2)
        theta = torch.cat([th_c, th_i], dim=-1)          # [b, L, 6F_cam+2F_img]
        return (to_heads(theta.cos(), self.num_heads),
                to_heads(theta.sin(), self.num_heads))

    def _rope_coeffs(self, info, omega=None, gain=None, dOmega=None, bias=None):
        """cos/sin for Plucker line rotary. Returns [B, L, 6F]."""
        omega = self.omega if omega is None else omega
        gain = self.freq_gain if gain is None else gain
        coords = self._coords6(info)  # [b, L, 6]
        theta = coords[..., None] * (omega[None, None, None] * gain[None, None])
        theta = theta.flatten(2)  # [b, L, 6F]
        if dOmega is not None:
            theta = theta + coords @ dOmega.T
        if bias is not None:
            theta = theta + bias[None, None]
        return to_heads(theta.cos(), self.num_heads), to_heads(theta.sin(), self.num_heads)

    def _segment_coeffs(self, info, t1, t2, omega=None, gain=None):
        """Sinc-enveloped rotary coeffs for a ray segment [t1, t2].

        t1/t2: [b, L, 1]. Returns cos/sin-like coeffs [b, L, 3F].
        """
        omega = self.omega if omega is None else omega
        gain = self.freq_gain if gain is None else gain
        o, d = info["tok_o"], info["tok_d"]
        mid = o + 0.5 * (t1 + t2) * d      # [b, L, 3]
        half = 0.5 * (t2 - t1) * d
        wg = omega[None, None, None] * gain[None, None]  # [1,1,3,F]
        phase = (mid[..., None] * wg).flatten(2)
        halfphase = (half[..., None] * wg).flatten(2)
        env = sinc(halfphase)
        return env * phase.cos(), env * phase.sin()

    def _chord_t(self, info):
        """Per-token chord [t1, t2] of the ray through the focus sphere. [b, L, 1] each."""
        r = self.shell_r_raw.clamp(0.05, 1.0)
        tc, b2 = info["tok_tc"].float(), info["tok_b2"].float()
        hit = b2 < r * r
        # sqrt(0) has an infinite backward (0 * inf = NaN through the where); floor it.
        hlf = torch.sqrt((r * r - b2).clamp_min(0.0) + 1e-4)
        half = torch.where(hit, hlf, torch.ones_like(hlf) * r)
        t1 = (tc - half).clamp_min(0.02)
        t2 = torch.maximum(tc + half, t1 + 1e-3)
        return t1, t2

    def _seg_dirs_coeffs(self, info, t1, t2, dirs, omega, gain):
        """Sinc-enveloped rotary coeffs of the ray segment [t1, t2] projected on unit
        directions `dirs` [n, 3]; omega [F], gain [n, F]. Returns cos/sin [b, L, n*F]."""
        o, d = info["tok_o"], info["tok_d"]
        mid = o + 0.5 * (t1 + t2) * d                    # [b, L, 3]
        half = 0.5 * (t2 - t1) * d
        pm = mid @ dirs.t()                              # [b, L, n]
        ph = half @ dirs.t()
        wg = omega[None, None, None] * gain[None, None]  # [1, 1, n, F]
        phase = (pm[..., None] * wg).flatten(2)
        env = sinc((ph[..., None] * wg).flatten(2))
        if getattr(self, "env_gamma", None) is not None:
            env = env.abs().clamp_min(1e-4).pow(self.env_gamma.clamp(0.5, 4.0)) * env.sign()
        return env * phase.cos(), env * phase.sin()

    def _epi_coeffs(self, info, site):
        """Epipolar-plane angle codes, [(b h), L, P]. phi harmonics (optionally enveloped near the
        epipole, where phi is undefined), + along-line-angle harmonics (bf), + baseline-position
        pairs (h_lam, hidden only)."""
        phi = info["tok_phi"]                                                # [b, L, 1]
        a = info["tok_psic"] if self.bf_coord == "psic" else info["tok_alpha"]
        env = info["tok_epi_env"] if self.epi_env else torch.ones_like(phi)
        modes = self.cam_modes
        cos_parts, sin_parts = [], []
        def add(theta, e=None):
            c, sn = theta.cos(), theta.sin()
            if e is not None:
                c, sn = c * e, sn * e
            cos_parts.append(c); sin_parts.append(sn)
        if site == "in":
            add(phi * (self.m_epi_in * self.gain_epi_in), env)
            if "bf_in" in modes:
                add(a * (self.m_alp_in * self.gain_alp_in))
        else:
            if modes & {"h_epi", "h_bf"}:
                add(phi * (self.m_epi_h * self.gain_epi_h), env)
            if "h_bf" in modes:
                add(a * (self.m_alp_h * self.gain_alp_h))
            if "h_lam" in modes:
                add(info["tok_u"] * self.gain_lam)
        c = torch.cat(cos_parts, -1); sn = torch.cat(sin_parts, -1)
        return to_heads(c, self.num_heads), to_heads(sn, self.num_heads)

    def _asym_coeffs(self, info, code, t1, t2):
        """Coefficients [b, L, K*n*F] for one role of asym_in ('chord'/'foot'/'anchor')."""
        dirs, om, gn = self.dirs_in, self.omega_seg3, self.gain_seg3
        K = self.asym_k
        if code == "anchor":
            cs, ss = [], []
            for kf in range(K):
                ta = t1 + ((kf + 0.5) / K) * (t2 - t1)
                c, sn = self._seg_dirs_coeffs(info, ta, ta, dirs, om, gn)
                cs.append(c); ss.append(sn)
            return torch.cat(cs, -1), torch.cat(ss, -1)
        if code == "foot":
            tc = info["tok_tc"].clamp_min(0.02)
            c, sn = self._seg_dirs_coeffs(info, tc, tc, dirs, om, gn)
        else:
            c, sn = self._seg_dirs_coeffs(info, t1, t2, dirs, om, gn)
        return c.repeat(1, 1, K), sn.repeat(1, 1, K)

    def _point_site_coeffs(self, info, site):
        """cos/sin for the shell / oracle modes at the input ('in') or hidden ('h') site."""
        modes = self.cam_modes
        foot_here = ("foot_in" in modes) if site == "in" else ("h_foot" in modes)
        layer_here = ("layer_pt" in modes) if site == "in" else ("h_layer_pt" in modes)
        foot_here = foot_here or layer_here
        if foot_here:
            t1 = t2 = None          # (layer_pt computes its own chord inside)
        else:
            t1, t2 = self._chord_t(info)
        oracle = {"pt_gt", "pt_gt_in"} if site == "in" else {"h_pt_gt", "h_pt_gt_in"}
        if modes & oracle:
            assert "tok_t_gt" in info, "oracle modes need --depth_dir (GT patch depth)"
            tg = info["tok_t_gt"]
            use = tg > 0
            if self.oracle_noise > 0:
                key = "_tgt_noise_%d" % id(self)
                if key not in info:   # one draw per forward per layer, shared by both sites
                    info[key] = torch.randn_like(tg) * self.oracle_noise
                tg = (tg + info[key]).clamp_min(0.02)
            if modes & {"pt_gt_in", "h_pt_gt_in"}:
                n_in = info["ttt_op_order"][0].end
                idx = torch.arange(tg.shape[1], device=tg.device)[None, :, None]
                use = use & (idx < n_in)
            t1 = torch.where(use, tg, t1)
            t2 = torch.where(use, tg, t2)
        dirs, om, gn = ((self.dirs_in, self.omega_seg3, self.gain_seg3) if site == "in"
                        else (self.dirs_h, self.omega_hseg, self.gain_hseg))
        if "head_anchor" in modes and site == "in":
            # per-HEAD anchor: head k -> point at chord fraction (k+0.5)/H; layout (b h)
            cs, ss = [], []
            H = self.num_heads
            for kf in range(H):
                ta = t1 + ((kf + 0.5) / H) * (t2 - t1)
                c, sn = self._seg_dirs_coeffs(info, ta, ta, dirs, om, gn)   # [b, L, nF]
                cs.append(c); ss.append(sn)
            c = torch.stack(cs, dim=1).flatten(0, 1)                        # [(b h), L, nF]
            sn = torch.stack(ss, dim=1).flatten(0, 1)
            return c, sn
        near_here = ("near_in" in modes) if site == "in" else ("h_near" in modes)
        if near_here:
            # NEAR-SHELL POINT (opacity prior, overnight): the visible surface of an opaque
            # object is the NEAR chord crossing; use x_near = o + t1 d as the sharp coordinate.
            c, sn = self._seg_dirs_coeffs(info, t1, t1, dirs, om, gn)
            return to_heads(c, self.num_heads), to_heads(sn, self.num_heads)
        if foot_here:
            if layer_here:
                # LAYER-INDEXED PLANE SWEEP (free): layer l reads/stores at chord fraction
                # (l%6 + 0.5)/6 -- the six memories cover six depth slices; the residual
                # stream integrates. Needs no estimate anywhere.
                t1l, t2l = self._chord_t(info)
                fr = ((self.layer_idx % 6) + 0.5) / 6.0
                tc = (t1l + fr * (t2l - t1l)).clamp_min(0.02)
            else:
                # Simplest 3D-point coordinate: the ray's closest-approach point to the focus
                # point, x_c = o + t_c d (no integral, no radius; env = 1).
                tc = info["tok_tc"].clamp_min(0.02)
            c, sn = self._seg_dirs_coeffs(info, tc, tc, dirs, om, gn)
        elif modes & {"anchor_in", "h_anchor"}:
            # H3b: K FIXED anchor points along the chord (plane-sweep phases, env = 1 each).
            cs, ss = [], []
            for kf in range(self.n_anchor):
                f = (kf + 0.5) / self.n_anchor
                ta = t1 + f * (t2 - t1)
                c, sn = self._seg_dirs_coeffs(info, ta, ta, dirs, om, gn)
                cs.append(c); ss.append(sn)
            c, sn = torch.cat(cs, -1), torch.cat(ss, -1)
        else:
            c, sn = self._seg_dirs_coeffs(info, t1, t2, dirs, om, gn)
        return to_heads(c, self.num_heads), to_heads(sn, self.num_heads)

    def _prope_mats(self, info):
        K, w2c = info["view_K_norm"].float(), info["view_w2c"].float()
        P = lift_K4(K) @ w2c
        P_inv = info["view_c2w"].float() @ lift_K4_inv(K)
        return P, P_inv

    # ---------- forward ----------

    def forward(self, x: torch.Tensor, info={}, *args):
        modes = self.cam_modes
        nh = self.num_heads
        tpv = info["tokens_per_view"]

        if "adaln_cam" in modes:
            c = self.cam_mlp(info["cam_feat"].to(x.dtype))
            x = x * (1 + self.film_g(c)) + self.film_b(c)

        qkv = F.silu(self.to_qkv(x), inplace=True)
        q, k, v = rearrange(
            qkv, "b l (qkv h d) -> qkv (b h) l d", qkv=3, h=nh
        )

        if "q_reinject" in modes:
            q_bias = self.q_cam(self.cam_mlp(info["cam_feat"].to(x.dtype)))
            q = q + rearrange(q_bias, "b l (h d) -> (b h) l d", h=nh)

        if "prope_ttt" in modes:
            P, P_inv = self._prope_mats(info)
            half = self.head_dim // 2
            P_h = to_heads(P, nh)
            P_inv_h = to_heads(P_inv, nh)
            q = apply_tiled_mat4(q, P_h.transpose(-1, -2), tpv, half)
            k = apply_tiled_mat4(k, P_inv_h, tpv, half)
            v = apply_tiled_mat4(v, P_inv_h, tpv, half)
        elif modes & {"prope_in", "gta_in"}:
            # Q15: INPUT-ONLY ports (PRoPE's original form) — transform the fast
            # q/k only; v and the output stay untouched. prope_in keeps the full
            # projective P = lift(K) @ w2c; gta_in drops the intrinsics lift
            # (rigid 4x4 rep only), the closer-to-orthogonal control.
            if "prope_in" in modes:
                P, P_inv = self._prope_mats(info)
            else:
                P = info["view_w2c"].float()
                P_inv = info["view_c2w"].float()
            half = self.head_dim // 2
            P_h = to_heads(P, nh)
            P_inv_h = to_heads(P_inv, nh)
            q = apply_tiled_mat4(q, P_h.transpose(-1, -2), tpv, half)
            k = apply_tiled_mat4(k, P_inv_h, tpv, half)

        q = q / (q.norm(dim=2, keepdim=True) + 1e-5).to(x.dtype)
        k = k / (k.norm(dim=2, keepdim=True) + 1e-5).to(x.dtype)

        if "ogta" in modes:
            with torch.autocast(device_type=x.device.type, enabled=False):
                c2w = info["view_c2w"].float()                    # [b, V, 4, 4]
                Rm, tv = c2w[..., :3, :3], c2w[..., :3, 3]        # [b,V,3,3],[b,V,3]
                V = Rm.shape[1]
                nu = self.ogta_omega_t.numel(); used = nu * 9
                # per-token view assignment, then to heads
                Rh = to_heads(Rm.reshape(Rm.shape[0], V, 9), nh).reshape(-1, V, 3, 3)
                th = to_heads(tv, nh)                              # [(b nh), V, 3]
                vidx = torch.arange(q.shape[1], device=q.device) // tpv
                Rt = Rh[:, vidx]                                   # [(b nh), L, 3, 3]
                tt = th[:, vidx]                                   # [(b nh), L, 3]
                ang = tt[..., None, :] * self.ogta_omega_t[None, None, :, None]
                ca, sa = ang.cos(), ang.sin()                      # [(b nh), L, nu, 3]

                def _ogta(t_in):
                    t9 = t_in[..., :used].float().reshape(*t_in.shape[:-1], nu, 9)
                    r = torch.einsum("blij,blukj->bluki", Rt, t9[..., :3][..., None, :]
                                     ).squeeze(-2) if False else                         torch.einsum("blij,bluj->blui", Rt, t9[..., :3])
                    outs = [r]
                    for a in range(3):                             # SO(2) per axis
                        x1 = t9[..., 3 + 2 * a]; x2 = t9[..., 4 + 2 * a]
                        c = ca[..., a]; sn = sa[..., a]
                        outs.append(torch.stack(
                            [x1 * c - x2 * sn, x1 * sn + x2 * c], -1))
                    y = torch.cat([outs[0], outs[1], outs[2], outs[3]], -1)
                    y = y.reshape(*t_in.shape[:-1], used).to(t_in.dtype)
                    return torch.cat([y, t_in[..., used:]], -1)                         if used < t_in.shape[-1] else y

                q = _ogta(q)
                k = _ogta(k)

        prope_raw_P_h = None
        prope_orig_state = None
        if modes & {"prope_orig", "prope_imgrope"}:
            # Q15: FAITHFUL original PRoPE (prope/prope/torch.py): on q/k/v/o,
            # [head_dim/2 = tiled projective | head_dim/4 = image-x RoPE |
            #  head_dim/4 = image-y RoPE], freq_base 100, split pairing,
            # inverse rotations on the output. Applied after the q/k L2-norm
            # (the official code has no q/k normalization of its own).
            P, P_inv = self._prope_mats(info)
            if "prope_imgrope" in modes:
                # attribution cell: the ORTHOGONAL part of PRoPE only — the
                # projective half replaced by identity, image x/y ropes kept
                eye = torch.eye(4, device=P.device, dtype=P.dtype)[None, None]
                P = eye.expand_as(P).contiguous()
                P_inv = P
            hd = self.head_dim
            # prope_proj_frac (Q40, 2026-08-07): budget split between the projective
            # block and the two image ropes. F56: on gObjaverse projective-alone
            # (+0.48) beats the 50/50 prope_orig (+0.32) while imgrope-alone is also
            # positive (+0.34) -- both components help alone and interfere at 50/50,
            # so the split is a live axis. Default 0.5 = the faithful original.
            half = int(hd * self.prope_proj_frac) // 8 * 8
            quart = (hd - half) // 2
            P_h = to_heads(P, nh)
            P_inv_h = to_heads(P_inv, nh)
            import math as _m
            px = int(_m.sqrt(tpv)); assert px * px == tpv, tpv
            pos = torch.arange(tpv, device=q.device)
            cx, sx = _prope_rope_coeffs(pos % px, quart, q.device)
            cy, sy = _prope_rope_coeffs(pos // px, quart, q.device)
            V = P.shape[1]
            cx, sx = cx.repeat(V, 1), sx.repeat(V, 1)
            cy, sy = cy.repeat(V, 1), sy.repeat(V, 1)

            def _prope_apply(t, mat, inv=False):
                t2 = apply_tiled_mat4(t, mat, tpv, half)
                a = _prope_rope_apply(t2[..., half:half + quart], cx, sx, inv)
                b = _prope_rope_apply(t2[..., half + quart:], cy, sy, inv)
                return torch.cat([t2[..., :half], a, b], dim=-1)

            q = _prope_apply(q, P_h.transpose(-1, -2))
            k = _prope_apply(k, P_inv_h)
            v = _prope_apply(v, P_inv_h)
            prope_orig_state = (P_h, _prope_apply)

        if modes & {"prope_in_raw", "prope_raw", "rot_raw"}:
            # Q15: as-is PRoPE port — projective P on the L2-NORMALIZED fast
            # q/k, no re-normalization afterward (original PRoPE order). The
            # score cancellation <P^T q, P^-1 k> = <q, k> is exact; the norm
            # distortion of the addresses flows into update strength.
            # prope_raw additionally carries the v/o transforms (full PRoPE).
            if "rot_raw" in modes:
                # Q42 (user, 2026-08-07): decompose prope_raw's +0.48 further --
                # ROTATION ONLY. P = [[R_w2c, 0],[0,1]]: no translation, no
                # intrinsics lift. Orthogonal by construction (P_inv = P^T), so
                # unlike the projective P it cannot distort address norms; and
                # unlike ogta it keeps prope_raw's v transport + o inverse.
                w2c = info["view_w2c"].float()
                P = torch.zeros_like(w2c)
                P[..., :3, :3] = w2c[..., :3, :3]
                P[..., 3, 3] = 1.0
                P_inv = P.transpose(-1, -2)
            else:
                P, P_inv = self._prope_mats(info)
            # prope_raw/rot_raw follow the ORIGINAL PRoPE: tile over the FULL head
            # dim (all 4-dim blocks); prope_in_raw keeps the half-dim tiling
            # of the earlier port for comparability.
            span = self.head_dim if modes & {"prope_raw", "rot_raw"} else self.head_dim // 2
            P_h = to_heads(P, nh)
            P_inv_h = to_heads(P_inv, nh)
            q = apply_tiled_mat4(q, P_h.transpose(-1, -2), tpv, span)
            k = apply_tiled_mat4(k, P_inv_h, tpv, span)
            if modes & {"prope_raw", "rot_raw"}:
                v = apply_tiled_mat4(v, P_inv_h, tpv, span)
                prope_raw_P_h = P_h

        cfr_R = None
        if "cfr_in" in modes:
            with torch.autocast(device_type=x.device.type, enabled=False):
                xc_t = info["tok_o"] + info["tok_tc"].clamp_min(0.02) * info["tok_d"]
                rel_t = xc_t - info["focus"][:, None, :]
                rho = rel_t.norm(dim=-1, keepdim=True)
                u = rel_t / (rho + 1e-8)
                gam = self.cfr_gamma.clamp(0.1, 30.0)
                theta = 2.0 * torch.atan(gam * rho / 2.0)                     # [b, L, 1]
                ct, st = theta.cos()[..., None], theta.sin()[..., None]
                K = torch.zeros(*u.shape[:2], 3, 3, device=x.device)
                K[..., 0, 1] = -u[..., 2]; K[..., 0, 2] = u[..., 1]
                K[..., 1, 0] = u[..., 2];  K[..., 1, 2] = -u[..., 0]
                K[..., 2, 0] = -u[..., 1]; K[..., 2, 1] = u[..., 0]
                eye3 = torch.eye(3, device=x.device)[None, None]
                R_t = eye3 + st * K + (1.0 - ct) * (K @ K)
                cfr_R = to_heads(R_t, nh)
            q = apply_block_rot(q, cfr_R)
            k = apply_block_rot(k, cfr_R)

        hh_H = None
        if modes & {"hh_in", "hh_vo"}:
            with torch.autocast(device_type=x.device.type, enabled=False):
                xc_t = info["tok_o"] + info["tok_tc"].clamp_min(0.02) * info["tok_d"]
                rel_t = xc_t - info["focus"][:, None, :]
                rn = rel_t.norm(dim=-1, keepdim=True)
                # central-ray degeneracy: blend toward the camera-from-focus direction
                cam_dir = info["tok_o"] - info["focus"][:, None, :]
                cam_dir = cam_dir / (cam_dir.norm(dim=-1, keepdim=True) + 1e-8)
                wgt = (rn / 0.05).clamp(0.0, 1.0)
                n_t = rel_t / (rn + 1e-8) * wgt + cam_dir * (1.0 - wgt)
                n_t = n_t / (n_t.norm(dim=-1, keepdim=True) + 1e-8)
                eye3 = torch.eye(3, device=x.device)
                H_t = eye3[None, None] - 2.0 * n_t[..., :, None] * n_t[..., None, :]
                hh_H = to_heads(H_t, nh)                                  # [(b nh), L, 3, 3]
            if "hh_in" in modes:
                q = apply_block_rot(q, hh_H)
                k = apply_block_rot(k, hh_H)
                q = q / (q.norm(dim=2, keepdim=True) + 1e-5).to(x.dtype)
                k = k / (k.norm(dim=2, keepdim=True) + 1e-5).to(x.dtype)
            if "hh_vo" in modes:
                v = apply_block_rot(v, hh_H)

        ff_F = None
        if "ff_vo" in modes:
            # FOOT-GEOGRAPHIC CARRIER FRAME (overnight): per-token frame F = [r^, e^, n^] built
            # at the foot direction (r^ = blended unit(x_c - p*), e^ = unit(z x r^) with x-axis
            # fallback near the pole, n^ = r^ x e^). Store F_i^T v (canonical surface frame),
            # read F_j o: the relative product F_j F_i^T ~ I exactly for MATCHED pairs (shared
            # foot direction) -- a canonicalizing carrier, vs rot_raw's view-rotation transport.
            assert not (self.cam_modes & {"vo_rel", "vo_rope", "hh_vo", "prope_raw", "rot_raw",
                                          "prope_orig", "prope_imgrope", "raygta", "rot_content",
                                          "gate_shell_rot"})
            with torch.autocast(device_type=x.device.type, enabled=False):
                xc_t = info["tok_o"] + info["tok_tc"].clamp_min(0.02) * info["tok_d"]
                rel_t = xc_t - info["focus"][:, None, :]
                rn = rel_t.norm(dim=-1, keepdim=True)
                cam_dir = info["tok_o"] - info["focus"][:, None, :]
                cam_dir = cam_dir / (cam_dir.norm(dim=-1, keepdim=True) + 1e-8)
                wgt = (rn / 0.05).clamp(0.0, 1.0)
                r_hat = rel_t / (rn + 1e-8) * wgt + cam_dir * (1.0 - wgt)
                r_hat = r_hat / (r_hat.norm(dim=-1, keepdim=True) + 1e-8)
                zax = torch.zeros_like(r_hat); zax[..., 2] = 1.0
                xax = torch.zeros_like(r_hat); xax[..., 0] = 1.0
                e1 = torch.cross(zax, r_hat, dim=-1); e2 = torch.cross(xax, r_hat, dim=-1)
                use2 = (e1.norm(dim=-1, keepdim=True) < 0.2).float()
                e_hat = e1 * (1 - use2) + e2 * use2
                e_hat = e_hat / (e_hat.norm(dim=-1, keepdim=True) + 1e-8)
                n_hat = torch.cross(r_hat, e_hat, dim=-1)
                F_t = torch.stack([r_hat, e_hat, n_hat], dim=-1)          # columns = frame axes
                ff_F = to_heads(F_t, nh)
            v = apply_block_rot(v, ff_F, transpose=True)                   # F^T v : to canonical

        raygta_M = None
        if "raygta" in modes:
            # H6 RayGTA: per-TOKEN ray-frame rotation R_tok = R_c2w * R_pix(u,v), where R_pix
            # rotates the optical axis onto the pixel ray (camera y = roll reference). q/k/v
            # are brought from the token's RAY frame to the world frame, the output back to
            # the query's ray frame: address <R_tok,j q, R_tok,i k> and carrier R_tok,j^T
            # R_tok,i v depend only on the relative rotation between the two RAYS -- the
            # matrix (wrap-free) fusion of image-coordinate ropes and rot_raw's camera rotation.
            with torch.autocast(device_type=x.device.type, enabled=False):
                Rc = info["view_rot"].float()                                   # [b, V, 3, 3] c2w
                bsz, L = Rc.shape[0], q.shape[1]
                vidx = torch.arange(L, device=q.device) // tpv
                R_cam = Rc[:, vidx]                                             # [b, L, 3, 3]
                d_w = info["tok_d"].float()                                     # [b, L, 3]
                z = torch.einsum("blji,blj->bli", R_cam, d_w)                   # cam-frame ray dir
                z = z / (z.norm(dim=-1, keepdim=True) + 1e-8)
                ey = torch.zeros_like(z); ey[..., 1] = 1.0
                xa = torch.cross(ey, z, dim=-1)
                xa = xa / (xa.norm(dim=-1, keepdim=True) + 1e-8)
                ya = torch.cross(z, xa, dim=-1)
                R_pix = torch.stack([xa, ya, z], dim=-1)                        # ray -> cam
                R_tok = R_cam @ R_pix                                           # ray -> world
                M = torch.zeros(bsz, L, 4, 4, device=q.device, dtype=torch.float32)
                M[..., :3, :3] = R_tok
                M[..., 3, 3] = 1.0
                raygta_M = to_heads(M.reshape(bsz, L, 16), nh).reshape(-1, L, 4, 4)
            q = _mat4_tok(q, raygta_M)
            k = _mat4_tok(k, raygta_M)
            v = _mat4_tok(v, raygta_M)

        q_plain = k_plain = None
        sweep_state = None
        if "rot_content" in modes:
            # H8 stage 1 (gate-invariant / content-relative SwiGLU): the rot_raw transform
            # (R_c2w on q/k tiled over 4-blocks, v transport, output back) is routed ONLY to
            # the CONTENT branch (w2); the GATE branch (w0) reads the plain post-L2-norm q/k.
            # The gate silu(q W0) is then pose-free (no absolute leak through W0^0), while
            # the content path and the W1 carrier are relative.
            w2c = info["view_w2c"].float()
            P = torch.zeros_like(w2c)
            P[..., :3, :3] = w2c[..., :3, :3]
            P[..., 3, 3] = 1.0
            P_inv = P.transpose(-1, -2)
            P_h = to_heads(P, nh)
            P_inv_h = to_heads(P_inv, nh)
            q_plain, k_plain = q, k
            q = apply_tiled_mat4(q, P_h.transpose(-1, -2), tpv, self.head_dim)
            k = apply_tiled_mat4(k, P_inv_h, tpv, self.head_dim)
            v = apply_tiled_mat4(v, P_inv_h, tpv, self.head_dim)
            prope_raw_P_h = P_h
        if "qk_rope_camimg" in modes:
            ccos, csin = self._rope_coeffs_camimg(info)
            q = apply_rotary_pairs(q, ccos, csin)
            k = apply_rotary_pairs(k, ccos, csin)
        elif "qk_rope_cam" in modes:
            ccos, csin = self._rope_coeffs(
                info, dOmega=getattr(self, "dOmega", None),
                bias=getattr(self, "phase_b", None),
            )
            if self.branch_rope:
                # GbR: keep the plain post-l2norm copies for the other branch.
                q_plain, k_plain = q, k
            q = apply_rotary_pairs(q, ccos, csin)
            k = apply_rotary_pairs(k, ccos, csin)
        elif "cone_pra" in modes:
            coords = torch.cat([info["tok_d"], info["tok_m"]], dim=-1)
            deltas = torch.cat([info["tok_d_delta"], info["tok_m_delta"]], dim=-1)
            wg = self.omega_cone[None, None, None] * self.gain_cone[None, None]
            theta = (coords[..., None] * wg).flatten(2)
            env = sinc((deltas[..., None] * wg).flatten(2))
            ccos = to_heads(env * theta.cos(), nh)
            csin = to_heads(env * theta.sin(), nh)
            q = apply_rotary_pairs(q, ccos, csin)
            k = apply_rotary_pairs(k, ccos, csin)
            # Envelope shrinks norms token-dependently; restore calibration.
            q = q / (q.norm(dim=2, keepdim=True) + 1e-5).to(x.dtype)
            k = k / (k.norm(dim=2, keepdim=True) + 1e-5).to(x.dtype)
        elif "pra_sinc" in modes:
            lcos, lsin = self._rope_coeffs(info, self.omega_line, self.gain_line)
            ones = torch.ones_like(info["tok_o"][..., :1])
            ec, es = self._segment_coeffs(
                info, self.t_near * ones, self.t_far * ones, self.omega_seg, self.gain_seg
            )
            ec, es = to_heads(ec, nh), to_heads(es, nh)
            ccos = torch.cat([lcos, ec], dim=-1)
            csin = torch.cat([lsin, es], dim=-1)
            q = apply_rotary_pairs(q, ccos, csin)
            k = apply_rotary_pairs(k, ccos, csin)
        elif "plucker_sinc" in modes:
            ones = torch.ones_like(info["tok_o"][..., :1])
            ec, es = self._segment_coeffs(info, self.t_near * ones, self.t_far * ones)
            ec, es = to_heads(ec, nh), to_heads(es, nh)
            q = apply_rotary_pairs(q, ec, es)
            k = apply_rotary_pairs(k, ec, es)
        elif "gate_shell_rot" in modes:
            # BRANCH-SPLIT PRODUCT KERNEL (2026-08-31): the SwiGLU fast weight has two input
            # branches; the GATE branch gets the chord 3D-point code, the CONTENT branch the
            # rotation tiles (rot_raw's input half) -- the hidden coefficient becomes an AND of
            # "near the same 3D point" x "rotation-compatible content", multiplicatively, with
            # no dim competition (attention's single bilinear score can only ADD block kernels).
            # v/o keep rot_raw's rotation transport.
            ec, es = self._point_site_coeffs(info, "in")
            qg = apply_rotary_pairs(q, ec, es); kg = apply_rotary_pairs(k, ec, es)
            qg = qg / (qg.norm(dim=2, keepdim=True) + 1e-5).to(x.dtype)
            kg = kg / (kg.norm(dim=2, keepdim=True) + 1e-5).to(x.dtype)
            w2c = info["view_w2c"].float()
            P = torch.zeros_like(w2c); P[..., :3, :3] = w2c[..., :3, :3]; P[..., 3, 3] = 1.0
            P_h = to_heads(P, nh); P_inv_h = to_heads(P.transpose(-1, -2), nh)
            q = apply_tiled_mat4(q, P_h.transpose(-1, -2), tpv, self.head_dim)
            k = apply_tiled_mat4(k, P_inv_h, tpv, self.head_dim)
            v = apply_tiled_mat4(v, P_inv_h, tpv, self.head_dim)
            prope_raw_P_h = P_h
            q_plain, k_plain = qg, kg          # gate-branch inputs
        elif "asym_in" in modes:
            t1, t2 = self._chord_t(info)
            kc, ks = self._asym_coeffs(info, self.asym_key, t1, t2)
            qc, qs = self._asym_coeffs(info, self.asym_query, t1, t2)
            q = apply_rotary_pairs(q, to_heads(qc, nh), to_heads(qs, nh))
            k = apply_rotary_pairs(k, to_heads(kc, nh), to_heads(ks, nh))
            q = q / (q.norm(dim=2, keepdim=True) + 1e-5).to(x.dtype)
            k = k / (k.norm(dim=2, keepdim=True) + 1e-5).to(x.dtype)
        elif modes & {"epi_in", "bf_in"}:
            ec, es = self._epi_coeffs(info, "in")
            q = apply_rotary_pairs(q, ec, es)
            k = apply_rotary_pairs(k, ec, es)
            q = q / (q.norm(dim=2, keepdim=True) + 1e-5).to(x.dtype)
            k = k / (k.norm(dim=2, keepdim=True) + 1e-5).to(x.dtype)
        elif modes & self.seg_in_modes:
            ec, es = self._point_site_coeffs(info, "in")
            q_pre = q
            q = apply_rotary_pairs(q, ec, es)
            k = apply_rotary_pairs(k, ec, es)
            q = q / (q.norm(dim=2, keepdim=True) + 1e-5).to(x.dtype)
            k = k / (k.norm(dim=2, keepdim=True) + 1e-5).to(x.dtype)
            if "sweep_in" in modes:
                ops = info["ttt_op_order"]
                assert len(ops) == 2 and ops[0].update and not ops[0].apply and ops[1].apply \
                    and ops[0].start == 0 and ops[1].start == 0, "sweep_in expects [update(inputs), apply(all)]"
                n_in, n_all = ops[0].end, ops[1].end
                n_tgt = n_all - n_in
                t1, t2 = self._chord_t(info)
                q_ks = []
                for kk in range(self.sweep_k):
                    fr = (kk + 0.5) / self.sweep_k
                    ta = t1 + fr * (t2 - t1)
                    c, sn = self._seg_dirs_coeffs(info, ta, ta, self.dirs_in, self.omega_seg3, self.gain_seg3)
                    c, sn = to_heads(c, nh)[:, n_in:n_all], to_heads(sn, nh)[:, n_in:n_all]
                    qk_ = apply_rotary_pairs(q_pre[:, n_in:n_all], c, sn)
                    q_ks.append(qk_ / (qk_.norm(dim=2, keepdim=True) + 1e-5).to(x.dtype))
                q = torch.cat([q] + q_ks, dim=1)                  # [B, n_all + K*n_tgt, d]
                sweep_state = (n_in, n_tgt, self.sweep_k)
        elif "point_rope" in modes:
            with torch.autocast(device_type="cuda", enabled=False):
                depth_raw = self.depth_head(x.float())
            t_mid = depth_raw[..., 0:1].clamp(-3.0, 2.0).exp()
            sigma = F.softplus(depth_raw[..., 1:2]) + 0.05
            t1 = (t_mid * torch.exp(-sigma)).clamp(0.02, 8.0)
            t2 = (t_mid * torch.exp(sigma)).clamp(0.02, 8.0)
            ec, es = self._segment_coeffs(info, t1, t2)
            ec, es = to_heads(ec, nh), to_heads(es, nh)
            q = apply_rotary_pairs(q, ec, es)
            k = apply_rotary_pairs(k, ec, es)

        if "vo_rel" in modes:
            R_tok = info["view_rot"].repeat_interleave(tpv, dim=1)  # [b, L, 3, 3]
            R_tok = to_heads(R_tok, nh)
            v = apply_block_rot(v, R_tok, transpose=False)
        if "vo_store" in modes:
            R_st = to_heads(info["view_rot"].repeat_interleave(tpv, dim=1), nh)
            v = apply_block_rot(v, R_st, transpose=False)   # store canonical; no o-side map
        vo_rope_coeffs = None
        if "vo_rope" in modes:
            if self.vo_coords == "foot":
                # phase carrier on the FOOT POINT: matched pairs have delta x_c ~ 0, so the
                # carrier phase is near-identity exactly where it matters (unlike ray coords).
                xc_tok = info["tok_o"] + info["tok_tc"].clamp_min(0.02) * info["tok_d"]
                th = (xc_tok[..., None] * (self.omega_vo[None, None, None]
                                           * self.gain_vo[None, None])).flatten(2)
                vcos, vsin = to_heads(th.cos(), nh), to_heads(th.sin(), nh)
            elif self.vo_coords == "d":
                th = (info["tok_d"][..., None] * (self.omega_vo[None, None, None]
                                                  * self.gain_vo[None, None])).flatten(2)
                vcos, vsin = to_heads(th.cos(), nh), to_heads(th.sin(), nh)
            else:
                vcos, vsin = self._rope_coeffs(info, self.omega_vo, self.gain_vo)
            v = apply_rotary_pairs(v, vcos, vsin)
            vo_rope_coeffs = (vcos, vsin)

        with torch.autocast(device_type="cuda", enabled=False):
            lr = self.lr_fc(x.float())  # [b, l, lr_dim]
            if "cam_lr" in modes:
                lr = lr + self.lr_cam(info["cam_feat_lr"].float())

        lr = torch.nn.functional.softplus(lr.float() + self.base_lr_inv)
        if self.fw3l:
            lr0, lr2, lrb, lr1 = rearrange(
                lr, "b l (lrs h d) -> lrs (b h) l d", lrs=4, h=nh
            )
        elif self.fw4l:
            lr0, lr2, lrb, lrc, lr1 = rearrange(
                lr, "b l (lrs h d) -> lrs (b h) l d", lrs=5, h=nh
            )
        elif self.mlp2:
            lr0, lr1 = rearrange(
                lr, "b l (lrs h d) -> lrs (b h) l d", lrs=2, h=nh
            )
            lr2 = None
        else:
            lr0, lr1, lr2 = rearrange(
                lr, "b l (lrs h d) -> lrs (b h) l d", lrs=3, h=nh
            )

        if "w0" in info:
            w0, w1, w2 = info["w0"], info["w1"], info.get("w2")
        else:
            if "w0_mask" in modes:
                w0 = (self.w0 * self.w0_mask_in).repeat(x.shape[0], 1, 1)
                w2 = (self.w2 * self.w0_mask_in).repeat(x.shape[0], 1, 1)
                w1 = (self.w1 * self.w1_mask_h).repeat(x.shape[0], 1, 1)
            else:
                w0 = self.w0.repeat(x.shape[0], 1, 1)
                w1 = self.w1.repeat(x.shape[0], 1, 1)
                w2 = None if self.mlp2 else self.w2.repeat(x.shape[0], 1, 1)
            if "hyper_init" in modes:
                pose = info["view_pose11"][:, : info["num_input_views"]].to(x.dtype)
                h_enc = self.set_enc(pose)  # [b, v_in, 64]
                pooled = torch.cat([h_enc.mean(1), h_enc.amax(1)], dim=-1)
                s = self.s_head(pooled).float().reshape(x.shape[0], 3, self.rank)
                w0 = w0 + to_heads(torch.einsum("dr,br,rh->bdh", self.U0, s[:, 0], self.V0), nh)
                w1 = w1 + to_heads(torch.einsum("dr,br,rh->bdh", self.U1, s[:, 1], self.V1), nh)
                w2 = w2 + to_heads(torch.einsum("dr,br,rh->bdh", self.U2, s[:, 2], self.V2), nh)

        ttt_op_order = info["ttt_op_order"]
        if sweep_state is not None:
            n_in_, n_tgt_, K_ = sweep_state
            ttt_op_order = [ttt_op_order[0],
                            TTTOperator(0, n_in_ + n_tgt_ + K_ * n_tgt_, False, True)]
        if "cam_registers" in modes:
            ops = ttt_op_order
            assert len(ops) == 2 and ops[0].update and not ops[0].apply \
                and not ops[1].update and ops[1].apply and ops[0].start == 0, \
                "cam_registers only supports the [update(inputs), apply(all)] pattern"
            v_in = info["num_input_views"]
            R = self.num_registers
            h_reg = self.reg_mlp(info["view_pose11"][:, :v_in].to(x.dtype))  # [b, v_in, 64]
            k_reg = self.reg_k(h_reg).reshape(x.shape[0], v_in * R, self.dim)
            v_reg = self.reg_v(h_reg).reshape(x.shape[0], v_in * R, self.dim)
            k_reg = rearrange(k_reg, "b l (h d) -> (b h) l d", h=nh)
            v_reg = rearrange(v_reg, "b l (h d) -> (b h) l d", h=nh)
            k_reg = k_reg / (k_reg.norm(dim=2, keepdim=True) + 1e-5).to(x.dtype)
            n_reg = v_in * R

            k = torch.cat([k_reg, k], dim=1)
            v = torch.cat([v_reg, v], dim=1)
            lr_reg = F.softplus(self.reg_lr.float() + self.base_lr_inv)
            reg_fill = lr0.new_ones(lr0.size(0), n_reg, lr0.size(2))
            lr0 = torch.cat([reg_fill * lr_reg[0], lr0], dim=1)
            lr1 = torch.cat([reg_fill * lr_reg[1], lr1], dim=1)
            lr2 = torch.cat([reg_fill * lr_reg[2], lr2], dim=1)
            ttt_op_order = [
                TTTOperator(0, ops[0].end + n_reg, True, False),
                TTTOperator(ops[1].start, ops[1].end, False, True),
            ]

        fw_state_extra = {}
        if self.mlp2:
            hcos = hsin = None
            if "mlp2_rot2" in modes:
                hcos, hsin = self._rope_coeffs(info, self.omega_mh, self.gain_mh)
            output, w0, w1 = fast_weight_mlp2_weight_norm_apply(
                w0, w1, q, k, v, lr0, lr1, hcos, hsin, ttt_op_order,
                muon_update_steps=self.muon_update_steps,
            )
        elif self.fw3l:
            wb = info["wb"] if "wb" in info else self.wb.repeat(x.shape[0], 1, 1)
            h1cos = h1sin = s2cos = s2sin = None
            if "fw3l_rot3" in modes:
                h1cos, h1sin = self._rope_coeffs(info, self.omega_h1, self.gain_h1)
            if modes & {"fw3l_rot2", "fw3l_rot3"}:
                s2cos, s2sin = self._rope_coeffs(info, self.omega_s2, self.gain_s2)
            output, w0, w1, w2, wb = fast_weight_swiglu3l_weight_norm_apply(
                w0, w2, wb, w1, q, k, v, lr0, lr2, lrb, lr1,
                h1cos, h1sin, s2cos, s2sin, ttt_op_order,
                muon_update_steps=self.muon_update_steps,
            )
            fw_state_extra["wb"] = wb
        elif self.fw4l:
            wb = info["wb"] if "wb" in info else self.wb.repeat(x.shape[0], 1, 1)
            wc = info["wc"] if "wc" in info else self.wc.repeat(x.shape[0], 1, 1)
            h1cos = h1sin = s2cos = s2sin = s3cos = s3sin = None
            if "fw4l_rot4" in modes:
                h1cos, h1sin = self._rope_coeffs(info, self.omega_h1, self.gain_h1)
                s2cos, s2sin = self._rope_coeffs(info, self.omega_s2, self.gain_s2)
                s3cos, s3sin = self._rope_coeffs(info, self.omega_s3, self.gain_s3)
            output, w0, w1, w2, wb, wc = fast_weight_swiglu4l_weight_norm_apply(
                w0, w2, wb, wc, w1, q, k, v, lr0, lr2, lrb, lrc, lr1,
                h1cos, h1sin, s2cos, s2sin, s3cos, s3sin, ttt_op_order,
                muon_update_steps=self.muon_update_steps,
            )
            fw_state_extra["wb"] = wb
            fw_state_extra["wc"] = wc
        elif ({"h_pra", "h_dpra", "h_strat", "h_img", "h_bump", "h_epi", "h_bf"} | self.seg_h_modes) & modes:
            if "h_bump" in modes:
                with torch.autocast(device_type=x.device.type, enabled=False):
                    centre = info["view_c2w"][..., :3, 3].float()                # [b, V, 3]
                    u = centre - info["focus"][:, None, :].float()
                    u = u / (u.norm(dim=-1, keepdim=True) + 1e-8)
                    vidx = torch.arange(q.shape[1], device=q.device) // tpv
                    u_tok = u[:, vidx]                                          # [b, L, 3]
                    amp = torch.exp(-self.bump_kappa.clamp(0.0, 20.0)
                                    * (1.0 - u_tok @ self.bump_centres.t()))     # [b, L, P]
                hcos = to_heads(amp, nh)
                hsin = torch.zeros_like(hcos)
            elif modes & {"h_epi", "h_bf"}:
                hcos, hsin = self._epi_coeffs(info, "h")
            elif modes & self.seg_h_modes:
                hcos, hsin = self._point_site_coeffs(info, "h")
            elif "h_img" in modes:
                img = info["tok_uv"].to(info["tok_d"].dtype)          # [b, L, 2]
                th = (img[..., None] * (self.omega_himg[None, None, None]
                                        * self.gain_himg[None, None])).flatten(2)
                hcos = to_heads(th.cos(), nh)
                hsin = to_heads(th.sin(), nh)
            elif "h_strat" in modes:
                xs = info["tok_o"][:, :, None, :] + \
                    self.t_strat[None, None, :, None] * info["tok_d"][:, :, None, :]
                coords18 = xs.flatten(2)  # [b, L, 18]
                wg = (self.omega_strat * self.gain_strat)[None, None]
                theta = (coords18[..., None] * wg).flatten(2)
                hcos = to_heads(theta.cos(), nh)
                hsin = to_heads(theta.sin(), nh)
            else:
                hcos, hsin = self._rope_coeffs(
                    info, self.omega_h, self.gain_h,
                    dOmega=getattr(self, "dOmega_h", None),
                    bias=getattr(self, "phase_b_h", None),
                )
            assert "cam_registers" not in modes, "hidden rotary + cam_registers unsupported"
            if "res2" in modes:
                output, w0, w1, w2 = fast_weight_swish_glu_hidden_rotary_res2_apply(
                    w0, w1, w2, q, k, v, lr0, lr1, lr2, hcos, hsin,
                    self.res_alpha, self.step_gains, ttt_op_order,
                    muon_update_steps=self.muon_update_steps,
                )
            elif "ms2" in modes:
                ops = ttt_op_order
                assert len(ops) == 2 and ops[0].update and ops[1].apply, \
                    "ms2 expects the [update, apply] pattern"
                ms_order = [ops[0], ops[0], ops[1]]
                output, w0, w1, w2 = fast_weight_swish_glu_hidden_rotary_multistep_apply(
                    w0, w1, w2, q, k, v, lr0, lr1, lr2, hcos, hsin,
                    self.step_gains, ms_order,
                    muon_update_steps=self.muon_update_steps,
                )
            elif "h_dpra" in modes:
                output, w0, w1, w2 = fast_weight_swish_glu_hidden_rotary_delta_apply(
                    w0, w1, w2, q, k, v, lr0, lr1, lr2, hcos, hsin, ttt_op_order,
                    muon_update_steps=self.muon_update_steps,
                )
            else:
                output, w0, w1, w2 = fast_weight_swish_glu_hidden_rotary_apply(
                    w0, w1, w2, q, k, v, lr0, lr1, lr2, hcos, hsin, ttt_op_order,
                    muon_update_steps=self.muon_update_steps,
                    hnorm="hnrot" in modes,
                )
        elif "h_qh" in modes:
            with torch.autocast(device_type=x.device.type, enabled=False):
                xc_t = info["tok_o"] + info["tok_tc"].clamp_min(0.02) * info["tok_d"]
                rel_t = xc_t - info["focus"][:, None, :]
                rn = rel_t.norm(dim=-1, keepdim=True)
                cam_dir = info["tok_o"] - info["focus"][:, None, :]
                cam_dir = cam_dir / (cam_dir.norm(dim=-1, keepdim=True) + 1e-8)
                wgt_c = (rn / 0.05).clamp(0.0, 1.0)
                n_t = rel_t / (rn + 1e-8) * wgt_c + cam_dir * (1.0 - wgt_c)
                n_t = n_t / (n_t.norm(dim=-1, keepdim=True) + 1e-8)
                # scene axis e = mean input-camera direction from the focus
                n_in_tok = info["ttt_op_order"][0].end
                e_ax = cam_dir[:, :n_in_tok].mean(1, keepdim=True)
                e_ax = e_ax / (e_ax.norm(dim=-1, keepdim=True) + 1e-8)
                cosang = (n_t * e_ax).sum(-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
                ang = torch.acos(cosang) * self.qh_kappa.clamp(0.1, 4.0)
                axis = torch.cross(e_ax.expand_as(n_t), n_t, dim=-1)
                axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-8)
                half = 0.5 * ang
                w_q = half.cos()
                xyz = half.sin() * axis
                xq, yq, zq = xyz[..., 0:1], xyz[..., 1:2], xyz[..., 2:3]
                rowa = torch.cat([w_q, -xq, -yq, -zq], -1)
                rowb = torch.cat([xq, w_q, -zq, yq], -1)
                rowc = torch.cat([yq, zq, w_q, -xq], -1)
                rowd = torch.cat([zq, -yq, xq, w_q], -1)
                Lq = torch.stack([rowa, rowb, rowc, rowd], -2)          # [b, L, 4, 4]
                M_tok = to_heads(Lq.flatten(2), nh).reshape(-1, Lq.shape[1], 4, 4)
            output, w0, w1, w2 = fast_weight_swish_glu_hidden_mat4_apply(
                w0, w1, w2, q, k, v, lr0, lr1, lr2, M_tok, M_tok, ttt_op_order,
                muon_update_steps=self.muon_update_steps,
            )
        elif "h_rot" in modes:
            # Hidden-site ROTATION action (2026-08-31): the orthogonal cousin of h_ga.
            # Every 4-block of the hidden activation is rotated by the token's c2w
            # rotation (l=1 on 3 dims + l=0 on the 4th) on update AND apply, so the
            # dominant retrieval coefficient becomes <R_j h_j, R_i h_i> = h_j^T R_j^T R_i h_i
            # -- the relative rotation, norm-preserving (no F3 distortion), Muon and
            # weight-norm untouched. Composable with rot_raw (input-site R + v/o
            # transport) = "one matrix action per address space" at wide baseline.
            with torch.autocast(device_type=x.device.type, enabled=False):
                c2w = info["view_c2w"].float()
                V = c2w.shape[1]
                M = torch.zeros_like(c2w)
                M[..., :3, :3] = c2w[..., :3, :3]
                M[..., 3, 3] = 1.0
                Mh = to_heads(M.reshape(M.shape[0], V, 16), nh).reshape(-1, V, 4, 4)
                vidx = torch.arange(q.shape[1], device=q.device) // tpv
                M_tok = Mh[:, vidx]                              # [(b nh), L, 4, 4]
            output, w0, w1, w2 = fast_weight_swish_glu_hidden_mat4_apply(
                w0, w1, w2, q, k, v, lr0, lr1, lr2, M_tok, M_tok, ttt_op_order,
                muon_update_steps=self.muon_update_steps,
            )
        elif "h_ga" in modes:
            # Q41: hidden-site group action. Per-view projective mats expanded to
            # per-token; update side P^-1, apply side P^T (prope's convention moved
            # to hidden space). d_h must be divisible by 4 (512 = 128 blocks).
            with torch.autocast(device_type=x.device.type, enabled=False):
                P, P_inv = self._prope_mats(info)                # [b, V, 4, 4]
                V = P.shape[1]
                Pt = to_heads(P.reshape(P.shape[0], V, 16), nh).reshape(-1, V, 4, 4)
                Pi = to_heads(P_inv.reshape(P.shape[0], V, 16), nh).reshape(-1, V, 4, 4)
                vidx = torch.arange(q.shape[1], device=q.device) // tpv
                Ma_tok = Pt[:, vidx].transpose(-1, -2)           # P^T per apply token
                Mu_tok = Pi[:, vidx]                             # P^-1 per update token
            output, w0, w1, w2 = fast_weight_swish_glu_hidden_mat4_apply(
                w0, w1, w2, q, k, v, lr0, lr1, lr2, Mu_tok, Ma_tok, ttt_op_order,
                muon_update_steps=self.muon_update_steps,
            )
        elif "gate_shell_rot" in modes:
            output, w0, w1, w2 = fast_weight_swish_glu_branch_input_rotary_apply(
                w0, w1, w2, q_plain, k_plain, q, k, v, lr0, lr1, lr2, ttt_op_order,
                muon_update_steps=self.muon_update_steps,
            )
        elif "rot_content" in modes:
            output, w0, w1, w2 = fast_weight_swish_glu_branch_input_rotary_apply(
                w0, w1, w2, q_plain, k_plain, q, k, v, lr0, lr1, lr2, ttt_op_order,
                muon_update_steps=self.muon_update_steps,
            )
        elif self.branch_rope:
            # gate_rope: gate branch (w0) reads the rotated q/k, content
            # branch (w2) the plain copy; content_rope is the mirror.
            if "gate_rope" in modes:
                qg, kg, qc, kc = q, k, q_plain, k_plain
            else:
                qg, kg, qc, kc = q_plain, k_plain, q, k
            output, w0, w1, w2 = fast_weight_swish_glu_branch_input_rotary_apply(
                w0, w1, w2, qg, kg, qc, kc, v, lr0, lr1, lr2, ttt_op_order,
                muon_update_steps=self.muon_update_steps,
            )
        else:
            output, w0, w1, w2 = fast_weight_swish_glu_weight_norm_mini_batch_apply(
                w0, w1, w2, q, k, v, lr0, lr1, lr2, ttt_op_order,
                muon_update_steps=self.muon_update_steps,
            )

        if sweep_state is not None:
            n_in_, n_tgt_, K_ = sweep_state
            Bq = output.shape[0]
            o_main = output[:, : n_in_ + n_tgt_]
            o_t0 = o_main[:, n_in_:]
            o_ks = output[:, n_in_ + n_tgt_ :].reshape(Bq, K_, n_tgt_, output.shape[-1])
            cands = torch.cat([o_t0[:, None], o_ks], dim=1)          # [B, K+1, n_tgt, d]
            logits = self.sweep_probe(cands.float()).squeeze(-1)      # [B, K+1, n_tgt]
            wts = torch.softmax(logits, dim=1)
            o_tgt = (wts[..., None] * cands.float()).sum(1).to(output.dtype)
            output = torch.cat([o_main[:, :n_in_], o_tgt], dim=1)
        if ff_F is not None:
            output = apply_block_rot(output, ff_F, transpose=False)        # F o : to the query frame
        if hh_H is not None and "hh_vo" in modes:
            output = apply_block_rot(output, hh_H)     # H is its own inverse
        if vo_rope_coeffs is not None:
            output = apply_rotary_pairs(output, vo_rope_coeffs[0], vo_rope_coeffs[1], inverse=True)
        if "vo_rel" in modes:
            output = apply_block_rot(output, R_tok, transpose=True)
        if "prope_ttt" in modes:
            output = apply_tiled_mat4(output, P_h, tpv, self.head_dim // 2)
        if prope_raw_P_h is not None:
            output = apply_tiled_mat4(output, prope_raw_P_h, tpv, self.head_dim)
        if raygta_M is not None:
            output = _mat4_tok(output, raygta_M.transpose(-1, -2))
        if prope_orig_state is not None:
            P_h, _prope_apply = prope_orig_state
            output = _prope_apply(output, P_h, inv=True)

        output = self.o_norm(output)
        output = rearrange(
            output, "(b h) l d -> b l (h d)", h=nh, b=x.shape[0]
        )
        output = self.c_proj(output)
        state = {"w0": w0, "w1": w1, **fw_state_extra}
        if w2 is not None:
            state["w2"] = w2
        return output, state

    def extra_repr(self) -> str:
        if self.mlp2:
            return (f"cam_mode: {self.cam_mode}, w0: {tuple(self.w0.shape)}, "
                    f"w1: {tuple(self.w1.shape)} (gateless 2-layer MLP), "
                    f"Muon update steps: {self.muon_update_steps}")
        return f"cam_mode: {self.cam_mode}, " + super().extra_repr()
