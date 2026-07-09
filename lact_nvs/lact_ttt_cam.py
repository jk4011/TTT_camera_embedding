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
"""
import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from lact_ttt import (
    FastWeightGluMLPMultihead,
    TTTOperator,
    fast_weight_swish_glu_weight_norm_mini_batch_apply,
    inv_softplus,
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


def apply_rotary_pairs(x, coeff_cos, coeff_sin):
    """Rotate adjacent feature pairs. coeff_*: [B, L, P]; acts on x[..., :2P]."""
    P = coeff_cos.size(-1)
    x_rot = x[..., : 2 * P].float().reshape(*x.shape[:-1], P, 2)
    x1, x2 = x_rot.unbind(-1)
    y1 = x1 * coeff_cos - x2 * coeff_sin
    y2 = x1 * coeff_sin + x2 * coeff_cos
    y = torch.stack([y1, y2], dim=-1).reshape(*x.shape[:-1], 2 * P)
    return torch.cat([y.to(x.dtype), x[..., 2 * P :]], dim=-1)


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
def fast_weight_swish_glu_hidden_rotary_apply(
    w0, w1, w2, q, k, v, lr0, lr1, lr2,
    hcos, hsin,
    ttt_ua_order: list,
    muon_update_steps: int = 0,
):
    """Baseline LaCT kernel + per-token rotary applied to the SwiGLU *hidden*
    activation (h-PRA). The hidden layer h(x) is rotated by the token's phases
    before meeting w1, in both the update and apply paths:

        write:  dW1 ~ (R_i h(k_i))^T v_i         read:  o_j = (R_j h(q_j)) W1

    so the value-retrieval channel <R_j h(q_j), R_i h(k_i)> becomes relative in
    hidden space -- a second, independent relative channel with no attention
    analogue. Backprop through the rotation is its inverse (negated sin).

    hcos/hsin: [B, L, P] with 2P <= d_h; sliced per op like k (update) / q (apply).
    """
    from lact_ttt import silu_backprop, zeropower_via_newtonschulz5

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
            hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul
            hidden_rot = apply_rotary_pairs(hidden, hci, hsi)

            # Backprop: dL/dh = R^T (dL/dh'), R^T = rotary with negated sin.
            dhidden_rot = vi @ w1_now.transpose(-1, -2)
            dhidden = apply_rotary_pairs(dhidden_rot, hci, -hsi)
            dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)
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
            hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul
            hidden_rot = apply_rotary_pairs(hidden, hci, hsi)

            dhidden_rot = vi @ w1_now.transpose(-1, -2)
            dhidden = apply_rotary_pairs(dhidden_rot, hci, -hsi)
            dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)
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
            hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul
            hidden_rot = apply_rotary_pairs(hidden, hci, hsi)

            dhidden_rot = vi @ w1_now.transpose(-1, -2)
            dhidden = apply_rotary_pairs(dhidden_rot, hci, -hsi)
            dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)
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
        hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul
        hidden_rot = apply_rotary_pairs(hidden, hci, hsi)

        dhidden_rot = vi @ w1_now.transpose(-1, -2)
        dhidden = apply_rotary_pairs(dhidden_rot, hci, -hsi)
        dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)
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
            h1 = F.silu(gate_before_act, inplace=False) * hidden_before_mul
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
            dhidden_before_mul = dh1 * F.silu(gate_before_act, inplace=False)
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
        phase_bias: bool = False,
        t_near: float = 0.05,
        t_far: float = 4.0,
    ):
        super().__init__(dim, head_dim, inter_multi, bias, base_lr, muon_update_steps)
        self.cam_mode = cam_mode
        self.cam_modes = set(cam_mode.split("+"))
        known = {"qk_rope_cam", "plucker_sinc", "point_rope", "pra_sinc", "vo_rel",
                 "prope_ttt", "cam_lr", "adaln_cam", "q_reinject", "cam_registers",
                 "hyper_init", "h_pra", "h_dpra", "cone_pra", "ms2",
                 "w0_mask", "omega_map", "m_scale", "res2", "mip", "h_strat",
                 "fw3l", "fw3l_rot2", "fw3l_rot3", "mlp2", "mlp2_rot2"}
        unknown = self.cam_modes - known
        if unknown:
            raise ValueError(f"unknown cam_mode(s) {unknown}")
        # Q2 depth-3 fast weights: standalone modes; rot2/rot3 reuse the stock
        # qk_rope_cam machinery for the input rotary site.
        self.fw3l = bool(self.cam_modes & {"fw3l", "fw3l_rot2", "fw3l_rot3"})
        if self.fw3l:
            assert len(self.cam_modes) == 1, "fw3l modes are standalone (no '+' combos)"
            if self.cam_modes & {"fw3l_rot2", "fw3l_rot3"}:
                self.cam_modes.add("qk_rope_cam")
        # Gateless 2-layer-MLP fast weights (inner-model generality control):
        # standalone modes; mlp2_rot2 = input rotary (stock qk_rope_cam
        # machinery) + hidden rotary on the single hidden activation.
        self.mlp2 = bool(self.cam_modes & {"mlp2", "mlp2_rot2"})
        if self.mlp2:
            assert len(self.cam_modes) == 1, "mlp2 modes are standalone (no '+' combos)"
            if "mlp2_rot2" in self.cam_modes:
                self.cam_modes.add("qk_rope_cam")
        rotary_fams = {"qk_rope_cam", "plucker_sinc", "point_rope", "pra_sinc", "cone_pra"}
        assert len(rotary_fams & self.cam_modes) <= 1, "only one rotary family at a time"
        assert len({"h_pra", "h_dpra", "h_strat"} & self.cam_modes) <= 1, "one hidden rotary at a time"
        if self.cam_modes & {"ms2", "res2"}:
            assert {"h_pra", "h_dpra"} & self.cam_modes, "ms2/res2 require a hidden rotary mode"
        assert not ({"ms2", "res2"} <= self.cam_modes), "ms2 and res2 are exclusive"
        self.head_dim = head_dim
        self.num_freqs = num_freqs
        d_h = int(head_dim * inter_multi)

        if "qk_rope_cam" in self.cam_modes:
            # 6 Plucker coords x num_freqs pairs.
            assert 2 * 6 * num_freqs <= head_dim
            omega = math.pi * torch.logspace(
                math.log2(0.5), math.log2(16.0), num_freqs, base=2.0
            )
            self.register_buffer("omega", omega, persistent=False)
            self.freq_gain = nn.Parameter(torch.ones(6, num_freqs))

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

        if {"h_pra", "h_dpra"} & self.cam_modes:
            # Hidden-space Plucker rotary: 6 coords x num_freqs_h pairs in d_h.
            assert 2 * 6 * num_freqs_h <= d_h
            omega_h = math.pi * torch.logspace(
                math.log2(0.5), math.log2(16.0), num_freqs_h, base=2.0
            )
            self.register_buffer("omega_h", omega_h, persistent=False)
            self.gain_h = nn.Parameter(torch.ones(6, num_freqs_h))

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

        if "prope_ttt" in self.cam_modes:
            assert head_dim % 8 == 0

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
        """[b, L, 6] Plucker coords; optionally per-scene moment whitening."""
        tok_m = info["tok_m"]
        if "m_scale" in self.cam_modes:
            if "_m_scale" not in info:
                n_in = info["ttt_op_order"][0].end
                s_m = info["tok_m"][:, :n_in].norm(dim=-1).median(dim=1).values
                info["_m_scale"] = (s_m.clamp_min(0.05))[:, None, None].detach()
            tok_m = tok_m / info["_m_scale"]
        return torch.cat([info["tok_d"], tok_m], dim=-1)

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

        q = q / (q.norm(dim=2, keepdim=True) + 1e-5).to(x.dtype)
        k = k / (k.norm(dim=2, keepdim=True) + 1e-5).to(x.dtype)

        if "qk_rope_cam" in modes:
            ccos, csin = self._rope_coeffs(
                info, dOmega=getattr(self, "dOmega", None),
                bias=getattr(self, "phase_b", None),
            )
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

        with torch.autocast(device_type="cuda", enabled=False):
            lr = self.lr_fc(x.float())  # [b, l, lr_dim]
            if "cam_lr" in modes:
                lr = lr + self.lr_cam(info["cam_feat_lr"].float())

        lr = torch.nn.functional.softplus(lr.float() + self.base_lr_inv)
        if self.fw3l:
            lr0, lr2, lrb, lr1 = rearrange(
                lr, "b l (lrs h d) -> lrs (b h) l d", lrs=4, h=nh
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
        elif {"h_pra", "h_dpra", "h_strat"} & modes:
            if "h_strat" in modes:
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
                )
        else:
            output, w0, w1, w2 = fast_weight_swish_glu_weight_norm_mini_batch_apply(
                w0, w1, w2, q, k, v, lr0, lr1, lr2, ttt_op_order,
                muon_update_steps=self.muon_update_steps,
            )

        if "vo_rel" in modes:
            output = apply_block_rot(output, R_tok, transpose=True)
        if "prope_ttt" in modes:
            output = apply_tiled_mat4(output, P_h, tpv, self.head_dim // 2)

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
