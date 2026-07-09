import torch
import torch.nn as nn
from einops import rearrange, repeat
from minVid.models.wan.wan_base.modules.model import (
    WanRMSNorm,
    rope_apply,
    rope_apply_ar,
    rope_apply_ar_inference,
    rope_apply_ar_with_repeat,
    precompute_freqs_time_for_repeat
)
import math
import torch.nn.functional as F

from minVid.models.blocks.functions import silu_backprop, l2_norm, inv_softplus, zeropower_via_newtonschulz5, rope_apply_same_time

from minVid.models.wan.wan_base.modules.attention import flash_attention

from minVid.models.blocks.cam_phase_builder import make_cam_ladder, cam_phase_tables

from torch.nn import init


@torch.compile()
def ar_fast_weight_swish_glu_weight_norm_mini_batch(
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lr0: torch.Tensor,
    lr1: torch.Tensor,
    lr2: torch.Tensor,
    w_scale: float,
    num_repeat: int = 1,  # gradient descent for num_repeat times
    mini_batch_size: int = -1,
    update_length: int = -1,
    update_every: int = -1,
    use_moun: bool = False,
    num_moun_iters: int = 3,
    weight_norm: bool = True,
    src_prefix_len: int = 0,
):
    """
    Note:
    Forward:
    w1 @ (silu(w0 @ x) * (w2 @ x))
    w0, w2: [b, dh, d]
    w1:     [b, d, dh]
    x:      [b, l, d]

    Fast linear layer with global learning rate.
    w0: initial weight of shape (b, dh, dk) or called (d1, d0)
    w1: initial weight of shape (b, dv, dh) or called (d2, d1)
    k: key of shape (batch_size, seq_len, dk)
    v: value of shape (batch_size, seq_len, dv)
    lr1: scalar learning rate. of shape [b, l, dk]
    lr0: scalar learning rate. of shape [b, l, dh]
    w0_scale: scalar weight for normalizing the update terms!
    w1_scale: scalar weight for normalizing the update terms!

    src_prefix_len (ccv): tokens [0, src_prefix_len) are a clean SRC-video
    prefix, consumed in mini_batch_size chunks: write each chunk into the fast
    weights, then apply the post-write weights to the same chunk. The existing
    AR-interleave logic then runs on [src_prefix_len, L) unchanged (the first
    tgt noisy chunk applies with post-src weights instead of initial weights).
    src_prefix_len=0 reproduces the original kernel bit-exactly.

    FLOPS:
    Let B = batch_size, L = seq_len, D = input_dim, H = hidden_dim
    Note, B-dim is already merged with the head dim.

    Forward pass:
    4 * D * H * L * B

    Weight Update:
    8 * D * H * L * B

    Final forward:
    6 * D * H * L * B

    Total FLOPs = 18 * D * H * L * B
        9 * D * H * L * B if only count multiplications.
    """
    L = k.shape[1]
    if update_length == -1:
        update_length = L

    if mini_batch_size == -1:
        mini_batch_size = update_length

    if update_every == -1:
        update_every = mini_batch_size * 2

    # w0_norm = w0.detach().norm(dim=2, keepdim=True)
    # w1_norm = w1.detach().norm(dim=2, keepdim=True)
    # w2_norm = w2.detach().norm(dim=2, keepdim=True)
    w0_norm = w0.norm(dim=2, keepdim=True)
    w1_norm = w1.norm(dim=2, keepdim=True)
    w2_norm = w2.norm(dim=2, keepdim=True)

    output = torch.zeros_like(q)

    #### ccv SRC prefix: write chunk -> apply same chunk with post-write weights.
    for s_index in range(0, src_prefix_len, mini_batch_size):
        e_index = s_index + mini_batch_size

        ki, vi = k[:, s_index:e_index, :], v[:, s_index:e_index, :]
        lr0i = lr0[:, s_index:e_index, :]
        lr1i = lr1[:, s_index:e_index, :]
        lr2i = lr2[:, s_index:e_index, :]

        gate_before_act = torch.bmm(w0, ki.transpose(1, 2))
        hidden_before_mul = torch.bmm(w2, ki.transpose(1, 2))
        silu_gate = F.silu(gate_before_act, inplace=False)
        hidden = silu_gate * hidden_before_mul

        dhidden = torch.bmm(w1.transpose(1, 2), vi.transpose(1, 2))
        dhidden_before_mul = dhidden * silu_gate
        dgate = dhidden * hidden_before_mul
        dgate_before_act = silu_backprop(dgate, gate_before_act)

        dw1 = torch.bmm(vi.transpose(1, 2), hidden.transpose(1, 2) * lr1i * w_scale)
        dw0 = torch.bmm(dgate_before_act, ki * lr0i * w_scale)
        dw2 = torch.bmm(dhidden_before_mul, ki * lr2i * w_scale)

        if use_moun:
            dw1 = zeropower_via_newtonschulz5(dw1, num_moun_iters)
            dw0 = zeropower_via_newtonschulz5(dw0, num_moun_iters)
            dw2 = zeropower_via_newtonschulz5(dw2, num_moun_iters)

        w1 = w1 + dw1
        w0 = w0 + dw0
        w2 = w2 + dw2
        if weight_norm:
            w0 = w0 / (w0.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
            w1 = w1 / (w1.norm(dim=2, keepdim=True) + 1e-5) * w1_norm
            w2 = w2 / (w2.norm(dim=2, keepdim=True) + 1e-5) * w2_norm

        qi = q[:, s_index:e_index, :]
        h = torch.bmm(w2, qi.transpose(1, 2))
        gate = F.silu(torch.bmm(w0, qi.transpose(1, 2)), inplace=True)
        output[:, s_index:e_index, :] = torch.bmm(w1, gate * h).transpose(1, 2)

    # first_noise_chunk_size sometimes is greater than mini_batch_size.
    # for example, we want to have the next ar chunk, repeated with multiple noise levels.
    first_noise_chunk_size = update_every - mini_batch_size
    tgt_start = src_prefix_len
    qi = q[:, tgt_start : tgt_start + first_noise_chunk_size, :]
    h = torch.bmm(w2, qi.transpose(1, 2))
    gate = F.silu(torch.bmm(w0, qi.transpose(1, 2)), inplace=True)
    # [b, d_2, d_1] @ [b, d_1, l] -> [b, d_2, l] -> [b, l, d_2]
    output[:, tgt_start : tgt_start + first_noise_chunk_size, :] = torch.bmm(w1, gate * h).transpose(1, 2)
    # output.append(torch.bmm(w1, gate * h).transpose(1, 2))
    for _ in range(num_repeat):
        for i in range(tgt_start + first_noise_chunk_size, update_length, update_every):
            s_index = i
            e_index = s_index + mini_batch_size

            # begin to update fast weight. 

            ki, vi = k[:, s_index:e_index, :], v[:, s_index:e_index, :]  # bf16
            lr0i = lr0[:, s_index:e_index, :]  # [b, l, d/1] fp32
            lr1i = lr1[:, s_index:e_index, :]  # [b, l, d/1] fp32
            lr2i = lr2[:, s_index:e_index, :]  # [b, l, d/1] fp32

            # [b, d, l]
            gate_before_act = torch.bmm(w0, ki.transpose(1, 2))
            hidden_before_mul = torch.bmm(w2, ki.transpose(1, 2))
            silu_gate = F.silu(gate_before_act, inplace=False)
            hidden = silu_gate * hidden_before_mul

            # pred_v = torch.bmm(w1, hidden)

            # [b, d, l]
            dhidden = torch.bmm(w1.transpose(1, 2), vi.transpose(1, 2))

            dhidden_before_mul = dhidden * silu_gate

            dgate = dhidden * hidden_before_mul
            dgate_before_act = silu_backprop(dgate, gate_before_act)

            # [b, d_2, l] @ [b, l, d_1] -> [b, d_2, d_1]
            # in bmm two mat is fp32, but the result is bf16.
            dw1 = torch.bmm(
                vi.transpose(1, 2), hidden.transpose(1, 2) * lr1i * w_scale
            )  # [b, d, d]
            # [b, d_1, l] @ [b, l, d_0] -> [b, d_1, d_0]
            dw0 = torch.bmm(dgate_before_act, ki * lr0i * w_scale)
            dw2 = torch.bmm(dhidden_before_mul, ki * lr2i * w_scale)


            if use_moun:
                dw1 = zeropower_via_newtonschulz5(dw1, num_moun_iters)
                dw0 = zeropower_via_newtonschulz5(dw0, num_moun_iters)
                dw2 = zeropower_via_newtonschulz5(dw2, num_moun_iters)

            w1 = w1 + dw1
            w0 = w0 + dw0
            w2 = w2 + dw2
            if weight_norm:
                # do weight norm here
                w0 = w0 / (w0.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
                w1 = w1 / (w1.norm(dim=2, keepdim=True) + 1e-5) * w1_norm
                w2 = w2 / (w2.norm(dim=2, keepdim=True) + 1e-5) * w2_norm
            

            # e_index = e_index + mini_batch_size
            e_index = s_index + update_every
            qi = q[:, s_index:e_index, :]

            # use updated w0 and w1 to get the final output
            # [b, d_1, d_0] @ [b, d_0, l] -> [b, d_1, l]
            h = torch.bmm(w2, qi.transpose(1, 2))
            gate = F.silu(torch.bmm(w0, qi.transpose(1, 2)), inplace=True)
            # [b, d_2, d_1] @ [b, d_1, l] -> [b, d_2, l] -> [b, l, d_2]
            output[:, s_index:e_index, :] = torch.bmm(w1, gate * h).transpose(1, 2)
            

    return output, w0, w1, w2


def apply_rotary_pairs(x, coeff_cos, coeff_sin):
    """Rotate adjacent feature pairs of row-major tokens (port of
    lact_nvs/lact_ttt_cam.apply_rotary_pairs).

    x: [b, l, d]; coeff_*: [l, P] or [b', l, P] (b' broadcastable to b);
    acts on x[..., :2P], remaining dims untouched.
    """
    P = coeff_cos.shape[-1]
    if coeff_cos.dim() == 2:
        coeff_cos = coeff_cos[None]
        coeff_sin = coeff_sin[None]
    x_rot = x[..., : 2 * P].float().reshape(*x.shape[:-1], P, 2)
    x1, x2 = x_rot.unbind(-1)
    y1 = x1 * coeff_cos - x2 * coeff_sin
    y2 = x1 * coeff_sin + x2 * coeff_cos
    y = torch.stack([y1, y2], dim=-1).reshape(*x.shape[:-1], 2 * P)
    if 2 * P == x.shape[-1]:
        return y.to(x.dtype)
    return torch.cat([y.to(x.dtype), x[..., 2 * P :]], dim=-1)


def apply_rotary_cols(x, cos, sin):
    """Rotate adjacent row-pairs of column-major tokens.

    x: [b, d, l]; cos/sin: [P, l] with 2P <= d. Rotates rows [0:2P].
    """
    P = cos.shape[0]
    x_rot = x[:, : 2 * P, :].float().reshape(x.shape[0], P, 2, x.shape[2])
    x1, x2 = x_rot[:, :, 0], x_rot[:, :, 1]
    c, s_ = cos[None], sin[None]
    y = torch.stack((x1 * c - x2 * s_, x1 * s_ + x2 * c), dim=2)
    y = y.reshape(x.shape[0], 2 * P, x.shape[2]).type_as(x)
    if 2 * P == x.shape[1]:
        return y
    return torch.cat([y, x[:, 2 * P :, :]], dim=1)


@torch.compile()
def ar_fast_weight_swish_glu_weight_norm_mini_batch_hidden_rope(
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lr0: torch.Tensor,
    lr1: torch.Tensor,
    lr2: torch.Tensor,
    hcos: torch.Tensor,  # [P, L] hidden-rotary coeffs (3D grid phases)
    hsin: torch.Tensor,
    w_scale: float,
    num_repeat: int = 1,
    mini_batch_size: int = -1,
    update_length: int = -1,
    update_every: int = -1,
    use_moun: bool = False,
    num_moun_iters: int = 3,
    weight_norm: bool = True,
    src_prefix_len: int = 0,
    delta_only: bool = False,
):
    """ar_fast_weight_swish_glu_weight_norm_mini_batch + h-PRA (hidden rotary).

    The SwiGLU hidden activation is rotated by per-token phases (grid-carrier
    or camera Plucker) before meeting w1 on both the apply (queries) and update
    (keys) paths; the update backprop applies the inverse rotation. Value
    retrieval becomes relative in hidden space (see the NVS/LLM studies).

    src_prefix_len (ccv): see the plain kernel; SRC chunks are updated and
    self-applied first, then the AR-interleave logic runs offset by the prefix.

    delta_only (Q9 gene, port of the LLM version): the apply splits the output
    matrix into its initial part and the accumulated fast-weight delta, and
    rotates the hidden only on the delta path:
        o = w1_init @ h(q) + (w1 - w1_init) @ R(phi_q) h(q).
    Update->apply recall stays relative (stored addresses are rotated, and the
    delta path probes with the rotated query hidden), but the initial readout
    sees the UNROTATED hidden — removing the constant absolute-phase tax on
    the initial readout. With all phases equal (zero angles) it reduces
    exactly to the baseline kernel.
    """
    L = k.shape[1]
    if update_length == -1:
        update_length = L
    if mini_batch_size == -1:
        mini_batch_size = update_length
    if update_every == -1:
        update_every = mini_batch_size * 2

    w0_norm = w0.norm(dim=2, keepdim=True)
    w1_norm = w1.norm(dim=2, keepdim=True)
    w2_norm = w2.norm(dim=2, keepdim=True)

    if delta_only:
        w1_init = w1.clone()

    output = torch.zeros_like(q)

    #### ccv SRC prefix: write chunk -> apply same chunk with post-write weights.
    for s_index in range(0, src_prefix_len, mini_batch_size):
        e_index = s_index + mini_batch_size

        ki, vi = k[:, s_index:e_index, :], v[:, s_index:e_index, :]
        lr0i = lr0[:, s_index:e_index, :]
        lr1i = lr1[:, s_index:e_index, :]
        lr2i = lr2[:, s_index:e_index, :]
        hci = hcos[:, s_index:e_index]
        hsi = hsin[:, s_index:e_index]

        gate_before_act = torch.bmm(w0, ki.transpose(1, 2))
        hidden_before_mul = torch.bmm(w2, ki.transpose(1, 2))
        silu_gate = F.silu(gate_before_act, inplace=False)
        hidden = silu_gate * hidden_before_mul
        hidden_rot = apply_rotary_cols(hidden, hci, hsi)

        dhidden_rot = torch.bmm(w1.transpose(1, 2), vi.transpose(1, 2))
        dhidden = apply_rotary_cols(dhidden_rot, hci, -hsi)

        dhidden_before_mul = dhidden * silu_gate
        dgate = dhidden * hidden_before_mul
        dgate_before_act = silu_backprop(dgate, gate_before_act)

        dw1 = torch.bmm(vi.transpose(1, 2), hidden_rot.transpose(1, 2) * lr1i * w_scale)
        dw0 = torch.bmm(dgate_before_act, ki * lr0i * w_scale)
        dw2 = torch.bmm(dhidden_before_mul, ki * lr2i * w_scale)

        if use_moun:
            dw1 = zeropower_via_newtonschulz5(dw1, num_moun_iters)
            dw0 = zeropower_via_newtonschulz5(dw0, num_moun_iters)
            dw2 = zeropower_via_newtonschulz5(dw2, num_moun_iters)

        w1 = w1 + dw1
        w0 = w0 + dw0
        w2 = w2 + dw2
        if weight_norm:
            w0 = w0 / (w0.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
            w1 = w1 / (w1.norm(dim=2, keepdim=True) + 1e-5) * w1_norm
            w2 = w2 / (w2.norm(dim=2, keepdim=True) + 1e-5) * w2_norm

        qi = q[:, s_index:e_index, :]
        h = torch.bmm(w2, qi.transpose(1, 2))
        gate = F.silu(torch.bmm(w0, qi.transpose(1, 2)), inplace=True)
        hq = gate * h
        hq_rot = apply_rotary_cols(hq, hci, hsi)
        if delta_only:
            # initial readout unrotated; only the accumulated delta probes rotated
            output[:, s_index:e_index, :] = (
                torch.bmm(w1_init, hq) + torch.bmm(w1 - w1_init, hq_rot)
            ).transpose(1, 2)
        else:
            output[:, s_index:e_index, :] = torch.bmm(w1, hq_rot).transpose(1, 2)

    first_noise_chunk_size = update_every - mini_batch_size
    tgt_start = src_prefix_len
    qi = q[:, tgt_start : tgt_start + first_noise_chunk_size, :]
    h = torch.bmm(w2, qi.transpose(1, 2))
    gate = F.silu(torch.bmm(w0, qi.transpose(1, 2)), inplace=True)
    hq = gate * h
    hq_rot = apply_rotary_cols(hq, hcos[:, tgt_start : tgt_start + first_noise_chunk_size], hsin[:, tgt_start : tgt_start + first_noise_chunk_size])
    if delta_only:
        output[:, tgt_start : tgt_start + first_noise_chunk_size, :] = (
            torch.bmm(w1_init, hq) + torch.bmm(w1 - w1_init, hq_rot)
        ).transpose(1, 2)
    else:
        output[:, tgt_start : tgt_start + first_noise_chunk_size, :] = torch.bmm(w1, hq_rot).transpose(1, 2)
    for _ in range(num_repeat):
        for i in range(tgt_start + first_noise_chunk_size, update_length, update_every):
            s_index = i
            e_index = s_index + mini_batch_size

            ki, vi = k[:, s_index:e_index, :], v[:, s_index:e_index, :]
            lr0i = lr0[:, s_index:e_index, :]
            lr1i = lr1[:, s_index:e_index, :]
            lr2i = lr2[:, s_index:e_index, :]
            hci = hcos[:, s_index:e_index]
            hsi = hsin[:, s_index:e_index]

            gate_before_act = torch.bmm(w0, ki.transpose(1, 2))
            hidden_before_mul = torch.bmm(w2, ki.transpose(1, 2))
            silu_gate = F.silu(gate_before_act, inplace=False)
            hidden = silu_gate * hidden_before_mul
            hidden_rot = apply_rotary_cols(hidden, hci, hsi)

            # backprop through the rotation: R^T = rotary with negated sin
            dhidden_rot = torch.bmm(w1.transpose(1, 2), vi.transpose(1, 2))
            dhidden = apply_rotary_cols(dhidden_rot, hci, -hsi)

            dhidden_before_mul = dhidden * silu_gate
            dgate = dhidden * hidden_before_mul
            dgate_before_act = silu_backprop(dgate, gate_before_act)

            dw1 = torch.bmm(
                vi.transpose(1, 2), hidden_rot.transpose(1, 2) * lr1i * w_scale
            )
            dw0 = torch.bmm(dgate_before_act, ki * lr0i * w_scale)
            dw2 = torch.bmm(dhidden_before_mul, ki * lr2i * w_scale)

            if use_moun:
                dw1 = zeropower_via_newtonschulz5(dw1, num_moun_iters)
                dw0 = zeropower_via_newtonschulz5(dw0, num_moun_iters)
                dw2 = zeropower_via_newtonschulz5(dw2, num_moun_iters)

            w1 = w1 + dw1
            w0 = w0 + dw0
            w2 = w2 + dw2
            if weight_norm:
                w0 = w0 / (w0.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
                w1 = w1 / (w1.norm(dim=2, keepdim=True) + 1e-5) * w1_norm
                w2 = w2 / (w2.norm(dim=2, keepdim=True) + 1e-5) * w2_norm

            e_index = s_index + update_every
            qi = q[:, s_index:e_index, :]
            h = torch.bmm(w2, qi.transpose(1, 2))
            gate = F.silu(torch.bmm(w0, qi.transpose(1, 2)), inplace=True)
            hq = gate * h
            hq_rot = apply_rotary_cols(hq, hcos[:, s_index:e_index], hsin[:, s_index:e_index])
            if delta_only:
                output[:, s_index:e_index, :] = (
                    torch.bmm(w1_init, hq) + torch.bmm(w1 - w1_init, hq_rot)
                ).transpose(1, 2)
            else:
                output[:, s_index:e_index, :] = torch.bmm(w1, hq_rot).transpose(1, 2)

    return output, w0, w1, w2


@torch.compile()
def ar_fast_weight_swish_glu_weight_norm_mini_batch_inference(
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lr0: torch.Tensor,
    lr1: torch.Tensor,
    lr2: torch.Tensor,
    w_scale: float,
    num_repeat: int = 1,  # gradient descent for num_repeat times
    mini_batch_size: int = -1,
    do_update: bool = False,
    use_moun: bool = False,
    num_moun_iters: int = 3,
    weight_norm: bool = True,
):
    """
    Note:
    Forward:
    w1 @ (silu(w0 @ x) * (w2 @ x))
    w0, w2: [b, dh, d]
    w1:     [b, d, dh]
    x:      [b, l, d]

    Fast linear layer with global learning rate.
    w0: initial weight of shape (b, dh, dk) or called (d1, d0)
    w1: initial weight of shape (b, dv, dh) or called (d2, d1)
    k: key of shape (batch_size, seq_len, dk)
    v: value of shape (batch_size, seq_len, dv)
    lr1: scalar learning rate. of shape [b, l, dk]
    lr0: scalar learning rate. of shape [b, l, dh]
    w0_scale: scalar weight for normalizing the update terms!
    w1_scale: scalar weight for normalizing the update terms!
    FLOPS:
    Let B = batch_size, L = seq_len, D = input_dim, H = hidden_dim
    Note, B-dim is already merged with the head dim.

    Forward pass:
    4 * D * H * L * B

    Weight Update:
    8 * D * H * L * B

    Final forward:
    6 * D * H * L * B

    Total FLOPs = 18 * D * H * L * B
        9 * D * H * L * B if only count multiplications.
    """
    L = k.shape[1]

    if mini_batch_size == -1:
        mini_batch_size = L


    # w0_norm = w0.detach().norm(dim=2, keepdim=True)
    # w1_norm = w1.detach().norm(dim=2, keepdim=True)
    # w2_norm = w2.detach().norm(dim=2, keepdim=True)
    w0_norm = w0.norm(dim=2, keepdim=True)
    w1_norm = w1.norm(dim=2, keepdim=True)
    w2_norm = w2.norm(dim=2, keepdim=True)


    if do_update:
        for _ in range(num_repeat):
            s_index = 0
            e_index = s_index + mini_batch_size

            # begin to update fast weight. 

            ki, vi = k[:, s_index:e_index, :], v[:, s_index:e_index, :]  # bf16
            lr0i = lr0[:, s_index:e_index, :]  # [b, l, d/1] fp32
            lr1i = lr1[:, s_index:e_index, :]  # [b, l, d/1] fp32
            lr2i = lr2[:, s_index:e_index, :]  # [b, l, d/1] fp32

            # [b, d, l]
            gate_before_act = torch.bmm(w0, ki.transpose(1, 2))
            hidden_before_mul = torch.bmm(w2, ki.transpose(1, 2))

            hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul

            # pred_v = torch.bmm(w1, hidden)

            # [b, d, l]
            dhidden = torch.bmm(w1.transpose(1, 2), vi.transpose(1, 2))

            dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)

            dgate = dhidden * hidden_before_mul
            dgate_before_act = silu_backprop(dgate, gate_before_act)

            # [b, d_2, l] @ [b, l, d_1] -> [b, d_2, d_1]
            # in bmm two mat is fp32, but the result is bf16.
            dw1 = torch.bmm(
                vi.transpose(1, 2), hidden.transpose(1, 2) * lr1i * w_scale
            )  # [b, d, d]
            # [b, d_1, l] @ [b, l, d_0] -> [b, d_1, d_0]
            dw0 = torch.bmm(dgate_before_act, ki * lr0i * w_scale)
            dw2 = torch.bmm(dhidden_before_mul, ki * lr2i * w_scale)


            if use_moun:
                dw1 = zeropower_via_newtonschulz5(dw1, num_moun_iters)
                dw0 = zeropower_via_newtonschulz5(dw0, num_moun_iters)
                dw2 = zeropower_via_newtonschulz5(dw2, num_moun_iters)

            w1 = w1 + dw1
            w0 = w0 + dw0
            w2 = w2 + dw2
            if weight_norm:
                # do weight norm here
                w0 = w0 / (w0.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
                w1 = w1 / (w1.norm(dim=2, keepdim=True) + 1e-5) * w1_norm
                w2 = w2 / (w2.norm(dim=2, keepdim=True) + 1e-5) * w2_norm
         
    
    h = torch.bmm(w2, q.transpose(1, 2))
    gate = F.silu(torch.bmm(w0, q.transpose(1, 2)), inplace=True)
    # [b, d_2, d_1] @ [b, d_1, l] -> [b, d_2, l] -> [b, l, d_2]
    output = torch.bmm(w1, gate * h).transpose(1, 2)
    

    return output, w0, w1, w2


def rope_apply_ar_src_prefix(x, grid_sizes, freqs, ar_window_f, n_latent_f, src_latent_f):
    """3D rope for the ccv [SRC || TGT-interleave] sequence.

    SRC frames get their own time grid t = 0..src_latent_f-1; TGT frames keep
    the existing AR-interleave time mapping (both copies of a frame share its
    time index). Spatial freqs unchanged. Mirrors rope_apply_ar otherwise.
    x: [b, L, n, d] with L = (src_latent_f + 2*n_latent_f - ar_window_f)*h*w.
    """
    with torch.autocast(device_type="cuda", enabled=False):
        n, c = x.size(2), x.size(3) // 2
        freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

        tgt_total_f = n_latent_f * 2 - ar_window_f
        freq_f = freqs[0][:n_latent_f]
        interleave_freq_f = freq_f.repeat(2, 1)
        for i in range(n_latent_f // ar_window_f):
            start = i * ar_window_f * 2
            end = start + ar_window_f
            interleave_freq_f[start:end] = freq_f[i * ar_window_f : (i + 1) * ar_window_f]
            interleave_freq_f[start + ar_window_f : end + ar_window_f] = \
                freq_f[i * ar_window_f : (i + 1) * ar_window_f]
        interleave_freq_f = interleave_freq_f[:tgt_total_f]

        time_freq_f = torch.cat([freqs[0][:src_latent_f], interleave_freq_f], dim=0)
        total_f = src_latent_f + tgt_total_f
        time_freq_f = time_freq_f.view(total_f, 1, 1, -1)

        output = []
        for i, (_, h, w) in enumerate(grid_sizes.tolist()):
            seq_len = total_f * h * w
            x_i = torch.view_as_complex(
                x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
            freqs_i = torch.cat([
                time_freq_f.expand(total_f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(total_f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(total_f, h, w, -1)
            ], dim=-1).reshape(seq_len, 1, -1)
            x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
            x_i = torch.cat([x_i, x[i, seq_len:]])
            output.append(x_i)
        return torch.stack(output).float()


@torch.compile()
def src_prefix_window_attention(q, k, v, num_src_chunks, window_len):
    """Full attention inside each SRC window; no cross-window, no src<->tgt.

    q/k/v: [b, num_src_chunks*window_len, h, d].
    """
    qi = rearrange(q, "b (n_c sw) h d -> (b n_c) sw h d", n_c=num_src_chunks, sw=window_len)
    ki = rearrange(k, "b (n_c sw) h d -> (b n_c) sw h d", n_c=num_src_chunks, sw=window_len)
    vi = rearrange(v, "b (n_c sw) h d -> (b n_c) sw h d", n_c=num_src_chunks, sw=window_len)
    k_lens = torch.tensor([window_len] * qi.shape[0], dtype=torch.int32, device=q.device)
    oi = flash_attention(q=qi, k=ki, v=vi, k_lens=k_lens, window_size=(-1, -1))
    return rearrange(oi, "(b n_c) sw h d -> b (n_c sw) h d", n_c=num_src_chunks)


@torch.compile()
def batched_sliding_window_attention(q, k, v, mini_batch_size, update_every, num_chunks, kv_cache_size):
    """
    q: [b, l, h, d]
    k: [b, l, h, d]
    v: [b, l, h, d]
    l = update_every - mini_batch_size +  update_every * (num_chunks - 1)
    """
    L = q.shape[1]
    o = torch.zeros_like(q)

    # first_ar_chunk_size is the size of repeated AR chunk. 
    first_ar_chunk_size = update_every - mini_batch_size
    qi = q[:, :first_ar_chunk_size, :]
    ki, vi = k[:, :first_ar_chunk_size, :], v[:, :first_ar_chunk_size, :]

    k_lens = [ki.shape[1]] * (ki.shape[0]) # [b] int, then to torch tensor
    k_lens = torch.tensor(k_lens, dtype=torch.int32, device=q.device)
    oi = flash_attention(
            q=qi,
            k=ki,
            v=vi,
            k_lens=k_lens,
            window_size=(-1, -1))
    o[:, :first_ar_chunk_size, :] = oi

    num_chunks_minus_one = num_chunks - 1
    # now, let's handle the rest update_every * (num_chunks - 1) chunks. 

    q_rest = rearrange(q[:, first_ar_chunk_size:, :], "b (n_c sw) h d -> (b n_c) sw h d", sw=update_every)
    k_rest = rearrange(k[:, first_ar_chunk_size:, :], "b (n_c sw) h d -> (b n_c) sw h d", sw=update_every)
    v_rest = rearrange(v[:, first_ar_chunk_size:, :], "b (n_c sw) h d -> (b n_c) sw h d", sw=update_every)

    q_rest_clean = q_rest[:, :mini_batch_size, :]
    q_rest_noise = q_rest[:, mini_batch_size:, :]

    k_rest_clean = k_rest[:, :mini_batch_size, :]
    v_rest_clean = v_rest[:, :mini_batch_size, :] 

    clean_k_lens = [mini_batch_size] * (k_rest.shape[0]) # [b] int, then to torch tensor
    clean_k_lens = torch.tensor(clean_k_lens, dtype=torch.int32, device=q.device)


    o_clean = flash_attention(
        q=q_rest_clean,
        k=k_rest_clean,
        v=v_rest_clean,
        k_lens=clean_k_lens,
        window_size=(-1, -1))
        
    k_rest_interleave = k_rest[:, mini_batch_size-kv_cache_size:, :]
    v_rest_interleave = v_rest[:, mini_batch_size-kv_cache_size:, :]
    clean_noise_k_lens = [mini_batch_size + kv_cache_size] * (k_rest.shape[0]) # [b] int, then to torch tensor
    clean_noise_k_lens = torch.tensor(clean_noise_k_lens, dtype=torch.int32, device=q.device)


    o_noise = flash_attention(
        q=q_rest_noise,
        k=k_rest_interleave,
        v=v_rest_interleave,
        k_lens=clean_noise_k_lens,
        window_size=(-1, -1))

    
    o_clean_noise_interleave = torch.cat([o_clean, o_noise], dim=1)

    o_clean_noise_interleave = rearrange(o_clean_noise_interleave, "(b n_c) sw h d -> b (n_c sw) h d", n_c=num_chunks_minus_one)

    o[:, first_ar_chunk_size:, :] = o_clean_noise_interleave
    
    return o

@torch.compile()
def batched_sliding_window_attention_with_repeated_chunks(q, k, v, mini_batch_size, update_every, num_chunks, kv_cache_size):
    """
    q: [b, l, h, d]
    k: [b, l, h, d]
    v: [b, l, h, d]
    l = update_every - mini_batch_size +  update_every * (num_chunks - 1)
    num_repeat = update_every // mini_batch_size - 1
    """
    L = q.shape[1]
    o = torch.zeros_like(q)

    # first_ar_chunk_size is the size of repeated AR chunk. 
    first_ar_chunk_size = update_every - mini_batch_size
    num_chunks_minus_one = num_chunks - 1
    num_repeat = update_every // mini_batch_size - 1
    qi = q[:, :first_ar_chunk_size, :]
    ki, vi = k[:, :first_ar_chunk_size, :], v[:, :first_ar_chunk_size, :]

    # reshape 
    qi = rearrange(qi, "b (nr fw) h d -> (b nr) fw h d", nr=num_repeat, fw=mini_batch_size)
    ki = rearrange(ki, "b (nr fw) h d -> (b nr) fw h d", nr=num_repeat, fw=mini_batch_size)
    vi = rearrange(vi, "b (nr fw) h d -> (b nr) fw h d", nr=num_repeat, fw=mini_batch_size)

    k_lens = [ki.shape[1]] * (ki.shape[0]) # [b] int, then to torch tensor
    k_lens = torch.tensor(k_lens, dtype=torch.int32, device=q.device)

    oi = flash_attention(
            q=qi,
            k=ki,
            v=vi,
            k_lens=k_lens,
            window_size=(-1, -1))

    oi = rearrange(oi, "(b nr) fw h d -> b (nr fw) h d", nr=num_repeat, fw=mini_batch_size)
    o[:, :first_ar_chunk_size, :] = oi

    # now, let's handle the rest update_every * (num_chunks - 1) chunks. 
    # [b * (num_chunks - 1), update_every, h, d]
    q_rest = rearrange(q[:, first_ar_chunk_size:, :], "b (n_c sw) h d -> (b n_c) sw h d", sw=update_every)
    k_rest = rearrange(k[:, first_ar_chunk_size:, :], "b (n_c sw) h d -> (b n_c) sw h d", sw=update_every)
    v_rest = rearrange(v[:, first_ar_chunk_size:, :], "b (n_c sw) h d -> (b n_c) sw h d", sw=update_every)

    q_rest_clean = q_rest[:, :mini_batch_size, :]
    k_rest_clean = k_rest[:, :mini_batch_size, :]
    v_rest_clean = v_rest[:, :mini_batch_size, :] 

    clean_k_lens = [mini_batch_size] * (k_rest.shape[0]) # [b] int, then to torch tensor
    clean_k_lens = torch.tensor(clean_k_lens, dtype=torch.int32, device=q.device)
    # [b * (num_chunks - 1), mini_batch_size, h, d]
    o_clean = flash_attention(
        q=q_rest_clean,
        k=k_rest_clean,
        v=v_rest_clean,
        k_lens=clean_k_lens,
        window_size=(-1, -1))
    
    # perform attention for noise chunks. 
    q_rest_noise = q_rest[:, mini_batch_size:, :]
    k_rest_noise = k_rest[:, mini_batch_size:, :]
    v_rest_noise = v_rest[:, mini_batch_size:, :]
    k_rest_noise = rearrange(k_rest_noise, "b (n_repeat sw) h d -> (b n_repeat) sw h d", n_repeat=num_repeat, sw=mini_batch_size)
    v_rest_noise = rearrange(v_rest_noise, "b (n_repeat sw) h d -> (b n_repeat) sw h d", n_repeat=num_repeat, sw=mini_batch_size)
    q_rest_noise = rearrange(q_rest_noise, "b (n_repeat sw) h d -> (b n_repeat) sw h d", n_repeat=num_repeat, sw=mini_batch_size)

    k_clean_repeated = repeat(k_rest_clean[:, -kv_cache_size:, :], 'b sw h d -> (b n_repeat) sw h d', n_repeat=num_repeat)
    v_clean_repeated = repeat(v_rest_clean[:, -kv_cache_size:, :], 'b sw h d -> (b n_repeat) sw h d', n_repeat=num_repeat)
    
    k_clean_noise_interleave = torch.cat([k_clean_repeated, k_rest_noise], dim=1)
    v_clean_noise_interleave = torch.cat([v_clean_repeated, v_rest_noise], dim=1)
    
    
    clean_noise_k_lens = [mini_batch_size + kv_cache_size] * (k_clean_noise_interleave.shape[0]) # [b] int, then to torch tensor
    clean_noise_k_lens = torch.tensor(clean_noise_k_lens, dtype=torch.int32, device=q.device)

    # print("debug shape 2", q_rest_noise.shape, k_clean_noise_interleave.shape, v_clean_noise_interleave.shape, clean_noise_k_lens.shape)
    o_noise  = flash_attention(
        q=q_rest_noise,
        k=k_clean_noise_interleave,
        v=v_clean_noise_interleave,
        k_lens=clean_noise_k_lens,
        window_size=(-1, -1))
    
    o_noise = rearrange(o_noise, "(b n_repeat) sw h d -> b (n_repeat sw) h d", n_repeat=num_repeat)

    
    o_clean_noise_interleave = torch.cat([o_clean, o_noise], dim=1)

    o_clean_noise_interleave = rearrange(o_clean_noise_interleave, "(b n_c) sw h d -> b (n_c sw) h d", n_c=num_chunks_minus_one, sw=update_every)

    o[:, first_ar_chunk_size:, :] = o_clean_noise_interleave
    
    return o

@torch.compile()
def sliding_window_attention_inference(q, k, v, mini_batch_size, kv_cache=None, interleave=False, kv_cache_size=1560):
    """
    q: [b, l, h, d]
    k: [b, l, h, d]
    v: [b, l, h, d]

    kv_cache: [2, b, l, h, d]

    Three cases:
    1. first ar noise chunk, where kv_cache is None and interleave is False 
    2. interleaved chunks, where kv_cache is None and interleave is True
    3. interleaved chunks, where kv_cache is not None and interleave is False
    """
    k_lens = [k.shape[1]] * (k.shape[0]) # [b] int, then to torch tensor
    k_lens = torch.tensor(k_lens, dtype=torch.int32, device=q.device)

    minibatch_lens = [mini_batch_size] * (q.shape[0]) # [b] int, then to torch tensor
    minibatch_lens = torch.tensor(minibatch_lens, dtype=torch.int32, device=q.device)

    if kv_cache is None and (not interleave):
        # Case-1: first ar noise chunk.  Just do vanilla attention. 
        o = flash_attention(
            q=q,
            k=k,
            v=v,
            k_lens=k_lens,
            window_size=(-1, -1))
    elif interleave:
        # Case-2: interleaved chunks.
        #   where the first chunk is clean, do vanilla attention. 
        #   for second chunk, use full k-v.  Don't use kv_cache here. 
        o = torch.zeros_like(q)
        qi, ki, vi = q[:, :mini_batch_size, :], k[:, :mini_batch_size, :], v[:, :mini_batch_size, :]
        oi = flash_attention(
            q=qi,
            k=ki,
            v=vi,
            k_lens=minibatch_lens,
            window_size=(-1, -1))
        o[:, :mini_batch_size, :] = oi

        qi = q[:, mini_batch_size:, :]
        interleaved_k = k[:, mini_batch_size - kv_cache_size:, :]
        interleaved_v = v[:, mini_batch_size - kv_cache_size:, :]
        interleaved_k_lens = [interleaved_v.shape[1]] * (interleaved_k.shape[0]) # [b] int, then to torch tensor
        interleaved_k_lens = torch.tensor(interleaved_k_lens, dtype=torch.int32, device=q.device)
        oi = flash_attention(
            q=qi,
            k=interleaved_k,
            v=interleaved_v,
            k_lens=interleaved_k_lens,
            window_size=(-1, -1))
        o[:, mini_batch_size:, :] = oi
    else:
        # Case-3: Denoising current chunk with kv_cache. 
        # [2 * real_bs, s, n_h, d] -> [real_bs, s, n_h, d]
        k_cache, v_cache = kv_cache.chunk(2, dim=0)
        k_with_cache = torch.cat([k_cache, k], dim=1)
        v_with_cache = torch.cat([v_cache, v], dim=1)

        k_lens = [k_with_cache.shape[1]] * (k.shape[0]) # [b] int, then to torch tensor
        k_lens = torch.tensor(k_lens, dtype=torch.int32, device=q.device)
        o = flash_attention(
            q=q,
            k=k_with_cache,
            v=v_with_cache,
            k_lens=k_lens,
            window_size=(-1, -1))
    
    return o

@torch.compile()
def rescale_qk(q, k, qk_scale, qk_offset):
    """
    q: [b, s, n_h, d]
    k: [b, s, n_h, d]
    """
    _, _, nheads, head_dim = q.shape
    qk_scale = qk_scale.view(1, 1, nheads, head_dim, 2)
    qk_offset = qk_offset.view(1, 1, nheads, head_dim, 2)
    q = q * qk_scale[:, :, :, :, 0] + qk_offset[:, :, :, :, 0]
    k = k * qk_scale[:, :, :, :, 1] + qk_offset[:, :, :, :, 1]
    return q, k

class ARFastWeightSwiGLU(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6,
                 o_norm=True,
                 fw_head_dim=768, # same as Standard Attention
                 inter_multi: int = 2,
                 lr_dim=1,
                 local_window_size:int = 4680,
                 kv_cache_size:int = -1, # -1 means same as local_window_size
                 w_init="clean",  # clean, identity
                 lr_parameterization="mamba",
                 qk_l2_norm=False,
                 qkv_silu=False,
                 use_moun: bool = False,
                 num_moun_iters: int = 5,
                 no_time_rope: bool = False,
                 weight_norm: bool = True,
                 ttt_scale: float = 1.0,
                 learnable_ttt_scale: bool = True, 
                 batch_size: int = 1, # batch_size is used to reshape the input 
                 # from [real_batch_size * num_windows, seq_len_per_window, d]
                 # to [batch_size, num_windows * seq_len_per_window, d],
                 n_latent_f: int = 21, # used for correct rope implementation. 
                 ar_window_f: int = 3, # chunk size for AR Video Diffusion. 
                 update_every: int = -1, # used for TTT. 
                 ttt_hidden_rope: bool = False,  # h-PRA: rotary on the SwiGLU hidden
                 # --- Q9 GA genes for the hidden rotary (all no-ops unless ttt_hidden_rope;
                 # defaults exactly reproduce the pre-gene behavior) ---
                 ttt_hrope_frac: float = 0.5,  # fraction of hidden dims rotated (0..1]
                 ttt_hrope_gain: float = 1.0,  # global multiplier on the hidden freq ladder
                 ttt_hrope_theta: float = None,  # grid-carrier ladder base; None -> 10000.0
                 # (the Plucker ladder is geometric, not theta-based; theta is a no-op there)
                 ttt_hrope_delta_only: bool = False,  # rotate only the fast-weight DELTA
                 # path on apply: o = w1_init @ h + (w1 - w1_init) @ R(phi) h
                 ttt_learnable_freqs: bool = False,  # learnable ladder gains (both sites)
                 ttt_freq_tilt: float = 0.1,
                 ttt_input_rope: bool = False,   # PRA input site: rotary on fast q/k post-l2norm
                 cam_phase_mode: str = "none",   # none | plucker (camera phases for the rotary sites)
                 src_latent_f: int = 0,          # ccv: latent frames of the clean SRC prefix (0 = off)
                 ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.local_window_size = local_window_size
        self.mini_batch_size = local_window_size
        if kv_cache_size == -1:
            self.kv_cache_size = local_window_size
        else:
            self.kv_cache_size = kv_cache_size
        self.batch_size = batch_size
        self.ar_window_f = ar_window_f
        self.n_latent_f = n_latent_f
        self.qk_norm = qk_norm
        self.eps = eps

        assert cam_phase_mode in ("none", "plucker")
        self.cam_phase_mode = cam_phase_mode
        self.ttt_input_rope = ttt_input_rope
        self.src_latent_f = src_latent_f
        if src_latent_f > 0:
            assert src_latent_f % ar_window_f == 0

        #### PRA transplant: hidden rotary. Phases are either the 3D (t, y, x)
        #### grid carrier (cam_phase_mode="none", v20k behavior) or per-token
        #### camera Plucker phases (cam_phase_mode="plucker").
        self.ttt_hidden_rope = ttt_hidden_rope
        self.ttt_hrope_delta_only = ttt_hrope_delta_only
        self.ttt_learnable_freqs = ttt_learnable_freqs
        if ttt_hidden_rope:
            d_h_total = int(fw_head_dim * inter_multi)
            # Q9 GA genes: ttt_hrope_frac sets the fraction of hidden dims rotated
            # (0.5 reproduces the pre-gene setting h_rope_dim = (d_h//2)//2*2, i.e.
            # 2*floor(d_h/4) dims), ttt_hrope_gain scales the whole ladder, and
            # ttt_hrope_theta overrides the grid-carrier ladder base (default 10000).
            P_h = max(1, int(d_h_total * ttt_hrope_frac / 2.0))
            assert 2 * P_h <= d_h_total, f"ttt_hrope_frac too large: {ttt_hrope_frac}"
            self.h_rope_dim = 2 * P_h
            if cam_phase_mode == "plucker":
                # 6 Plucker coords x nf_h freqs = h_rope_dim/2 pairs (768 -> 6x64)
                nf_h = (self.h_rope_dim // 2) // 6
                self.cam_num_freqs_h = nf_h
                self.register_buffer(
                    "cam_omega_h", make_cam_ladder(nf_h) * ttt_hrope_gain, persistent=False
                )
                gain_h = torch.ones(6, nf_h)
                if ttt_learnable_freqs:
                    self.cam_gain_h = nn.Parameter(gain_h)
                else:
                    self.register_buffer("cam_gain_h", gain_h, persistent=False)
            else:
                # same 3-way (t, y, x) split as rope_params ladders
                h_theta = ttt_hrope_theta if ttt_hrope_theta is not None else 10000.0
                ladder = ttt_hrope_gain / torch.pow(
                    h_theta, torch.arange(0, self.h_rope_dim, 2).float().div(self.h_rope_dim)
                )
                if ttt_learnable_freqs:
                    ladder = ladder * (1.0 + ttt_freq_tilt * torch.randn_like(ladder))
                    self.h_inv_freq = nn.Parameter(ladder)
                else:
                    self.register_buffer("h_inv_freq", ladder, persistent=False)

        #### PRA input site: camera rotary on fast q/k after l2 norm.
        if ttt_input_rope:
            assert cam_phase_mode == "plucker", "ttt_input_rope requires camera phases"
            # 6 coords x nf freqs, leaving >= 2*6 dims untouched (768 -> 6x63 = 378 pairs)
            nf_in = (fw_head_dim - 2 * 6) // (2 * 6)
            self.cam_num_freqs_in = nf_in
            self.register_buffer("cam_omega_in", make_cam_ladder(nf_in), persistent=False)
            gain_in = torch.ones(6, nf_in)
            if ttt_learnable_freqs:
                self.cam_gain_in = nn.Parameter(gain_in)
            else:
                self.register_buffer("cam_gain_in", gain_in, persistent=False)
        
        # layersx
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        ##### setup new layers for fast weight with swiglu

        if fw_head_dim < 1:
            fw_head_dim = self.head_dim # default head dim, same as Standard Attention
        self.fw_head_dim = fw_head_dim
        self.num_fw_heads = self.dim // self.fw_head_dim

        self.lr_dim = int(lr_dim * self.num_fw_heads * 3)
        self.lr_proj = nn.Linear(dim, self.lr_dim)

        self.use_o_norm = o_norm
        if o_norm:
            self.output_norm = WanRMSNorm(fw_head_dim, eps=eps)
        else:
            self.output_norm = nn.Identity()

        d_in = self.fw_head_dim
        d_h = int(d_in * inter_multi)
        d_out = self.fw_head_dim
        self.dh_over_din = 1.0
        self.dh_over_dout = 1.0
        gain = 1.0
        self.w_scale = 1.0

        self.w0 = nn.Parameter(
            torch.randn(self.num_fw_heads, int(d_h), d_in)
            * gain
            / math.sqrt(d_in)
            * math.sqrt(self.dh_over_din)
        )  # [d_h * num_heads,  d_in]
        self.w1 = nn.Parameter(
            torch.randn(self.num_fw_heads, int(d_out), d_h)
            * gain
            / math.sqrt(d_h)
            / math.sqrt(self.dh_over_dout)
        )  # [d_in * num_heads,  d_h]
        self.w2 = nn.Parameter(
            torch.randn(self.num_fw_heads, int(d_h), d_in)
            * gain
            / math.sqrt(d_in)
            * math.sqrt(self.dh_over_din)
        )  # [d_h * num_heads,  d_in]
        
        if w_init == "identity":
            print("init the weight matrix to identity matrix!")
            if d_h == d_in: 
                id_mat = torch.eye(d_h, d_in).to(self.w0.dtype)
                # repeat the id_mat for num_heads times. 
                id_mat = id_mat.unsqueeze(0).repeat(self.num_fw_heads, 1, 1)
                self.w0 = nn.Parameter(id_mat.clone())
                self.w1 = nn.Parameter(id_mat.clone())
                self.w2 = nn.Parameter(id_mat.clone())
                
            elif d_h == 2 * d_in:
                # concat two identity matrices. 
                id_mat = torch.eye(d_h // 2, d_in).to(self.w0.dtype)
                id_mat = id_mat.unsqueeze(0).repeat(self.num_fw_heads, 1, 1)
                id_mat = torch.cat([id_mat.clone(), id_mat.clone()], dim=1) # [num_fw_heads, 2 * d_in, d_in]
                self.w0 = nn.Parameter(id_mat.clone())
                self.w1 = nn.Parameter(id_mat.transpose(1, 2).clone())
                self.w2 = nn.Parameter(id_mat.clone())
            else:
                raise ValueError(f"d_h == {d_h} != {d_in} or {2 * d_in}")
        self.d_in = d_in
        self.d_h = d_h
        self.d_out = d_out

        self.qk_l2_norm = qk_l2_norm
        self.use_moun = use_moun
        self.num_moun_iters = num_moun_iters
        self.no_time_rope = no_time_rope
        self.qkv_silu = qkv_silu

        base_lr = 0.001
        # Lr parameterization and initialization
        if lr_parameterization.lower() == "mamba":
            self.base_lr_inv = inv_softplus(base_lr)
        self.lr_parameterization = lr_parameterization

        # add scaling and offset for fast weight!
        self.qk_scale = nn.Parameter(torch.ones(self.dim, 2))
        self.qk_offset = nn.Parameter(torch.zeros(self.dim, 2))
        self.weight_norm = weight_norm
        self.ttt_scale = ttt_scale
        self.learnable_ttt_scale = learnable_ttt_scale
        if self.learnable_ttt_scale:
            self.ttt_scale_proj = nn.Linear(dim, 1)

            init.zeros_(self.ttt_scale_proj.weight)
            if hasattr(self, 'ttt_scale_proj') and hasattr(self.ttt_scale_proj, 'bias') and self.ttt_scale_proj.bias is not None:
                nn.init.zeros_(self.ttt_scale_proj.bias)
        
        self.inference_frame_offset = 0 # int, used for inference. 
        self.cur_w0 = None
        self.cfg_w0 = None
        self.cfg_seq = False

        self.kv_cache = None
        self.kv_cache_cfg = None

        if update_every == -1:
            self.update_every = self.mini_batch_size * 2
        else:
            self.update_every = update_every
            assert update_every >= self.mini_batch_size
        
        self.cached_freqs_time = None
        self.num_repeat = self.update_every // self.mini_batch_size - 1

    def _rescale_qk(self, q, k):
        """
        q: [b, s, n_h, d]
        k: [b, s, n_h, d]
        """
        qk_scale = self.qk_scale.view(1, 1, self.num_heads, self.head_dim, 2)
        qk_offset = self.qk_offset.view(1, 1, self.num_heads, self.head_dim, 2)
        q = q * qk_scale[:, :, :, :, 0] + qk_offset[:, :, :, :, 0]
        k = k * qk_scale[:, :, :, :, 1] + qk_offset[:, :, :, :, 1]
        return q, k


    def forward(self, x, seq_lens, grid_sizes, freqs, cam_coords6=None):
        r"""
        Args:
            x (Tensor): chunked_x, where the seq_len dimension is the seq_len for each ar chunk!
                Shape [real_batch_size * (num_ar_windows * 2 - 1), seq_len_per_window, num_heads, C]
            cam_coords6 (Tensor, optional): [b, L_total, 6] fp32 per-token
                Plucker coordinates for the camera rotary sites (ccv runs).
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        # minibatch style AR Video Diffusion. 
        # input x format in the L dimension. 
        # suppose F frames of the video, then L = 2F - 1
        # noisy_f0, clean_f0, noise_f1, clean_f1, ..., clean_fL-1, noise_fL

        # Note, during inference, the code is little bit harder to understsand. 
        # The reason is that we need to handle multiple cases:
        # Case-1: x as first ar noise chunk. of shape [real_batch_size, mini_batch_size, d]
        # Case-2: x as interleaved chunks. with one clean chunk and one noise chunk.  x of shape [real_batch_size * 2, mini_batch_size, d]
        #     Note, for case-2, we need to cache the fast weight or kv-caches for the clean chunk! 
        # Case-3: x as denoising chunks, where the fast weight is already computed or the kv-caches are already computed. 
        #     Note, for case-3, we need to use the kv-caches to compute the attention. 

        # During handlling of above cases, we also need to handle CFG and non-CFG cases. 
        # To do that, we has a flag cfg_seq to indicate if the current chunk is CFG.  
        # we flip the flag in every forward pass.  since we assume the denoising loops calls cfg at every denoise step! 
        """
        is_training = self.training
        fake_batch_size, s_per_window, n, d = *x.shape[:2], self.num_heads, self.head_dim
        b = self.batch_size
        x = rearrange(x, "(b nw) sw d -> b (nw sw) d", b=self.batch_size)
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        if self.cached_freqs_time is None:
            # [total_f, head_dim//3]
            self.cached_freqs_time = precompute_freqs_time_for_repeat(self.head_dim, freqs, self.ar_window_f, self.n_latent_f, self.num_repeat)

        # also need to slice the grid_sizes, since when we call rope later, the qkv is of shape [real_batch_size, seq_len, n_h, d]
        grid_sizes = grid_sizes[:b]

        #### ccv SRC-prefix bookkeeping (src_latent_f=0 -> everything off).
        n_src_chunks = 0
        src_prefix_len = 0
        if self.src_latent_f > 0:
            assert self.training or fake_batch_size > 2, \
                "src-prefix inference path not implemented"
            assert self.update_every == self.mini_batch_size * 2, \
                "src prefix requires num_repeat == 1"
            n_src_chunks = self.src_latent_f // self.ar_window_f
            src_prefix_len = n_src_chunks * s_per_window

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        # [b, s, n_h, d]
        q, k, v = qkv_fn(x)

        # fast_q, fast_k = self._rescale_qk(q, k)
        fast_q, fast_k = rescale_qk(q, k, self.qk_scale, self.qk_offset)

        num_repeat = self.num_repeat

        # when applying rope, need to tell if it is training or inference.
        if self.training or fake_batch_size > 2:
            if src_prefix_len > 0:
                fast_q = rope_apply_ar_src_prefix(fast_q, grid_sizes, freqs, self.ar_window_f, self.n_latent_f, self.src_latent_f)
                fast_k = rope_apply_ar_src_prefix(fast_k, grid_sizes, freqs, self.ar_window_f, self.n_latent_f, self.src_latent_f)
            elif self.update_every == self.mini_batch_size * 2: # no repeat!
                # TODO: add precompute freqs here!
                fast_q = rope_apply_ar(fast_q, grid_sizes, freqs, self.ar_window_f, self.n_latent_f)
                fast_k = rope_apply_ar(fast_k, grid_sizes, freqs, self.ar_window_f, self.n_latent_f)
            else:
                fast_q = rope_apply_ar_with_repeat(fast_q, grid_sizes, freqs, 
                                                   self.ar_window_f, self.n_latent_f, 
                                                   interleave_freqs_time=self.cached_freqs_time,
                                                   num_repeat=num_repeat)
                fast_k = rope_apply_ar_with_repeat(fast_k, grid_sizes, freqs, self.ar_window_f, self.n_latent_f, 
                                                   interleave_freqs_time=self.cached_freqs_time,
                                                   num_repeat=num_repeat)
        else:
            # Inference. 
            # only edit the inference_frame_offset when:
            # 1. fake_batch_size == 2
            # 2. not cfg_seq

            # set interleave to True if:
            # 1. fake_batch_size == 2

            if fake_batch_size == 2:
                interleave = True
                if not self.cfg_seq:
                    self.inference_frame_offset += self.ar_window_f
                    
            else:
                interleave = False    
                

            fast_q = rope_apply_ar_inference(
                fast_q, grid_sizes, freqs, self.ar_window_f, self.inference_frame_offset, interleave
            )
            fast_k = rope_apply_ar_inference(
                fast_k, grid_sizes, freqs, self.ar_window_f, self.inference_frame_offset, interleave
            )
        fast_v = v
        if self.num_fw_heads != self.num_heads:
            # from [b, s, n_h, d] to [b, s, self.num_fw_heads, self.fw_head_dim]
            fast_q = rearrange(fast_q, 'b s n_h d -> b s (n_h d)')
            fast_k = rearrange(fast_k, 'b s n_h d -> b s (n_h d)')
            fast_v = rearrange(v, 'b s n_h d -> b s (n_h d)')

            fast_q = rearrange(fast_q, 'b s (n_h d) -> (b n_h) s d', n_h=self.num_fw_heads)
            fast_k = rearrange(fast_k, 'b s (n_h d) -> (b n_h) s d', n_h=self.num_fw_heads)
            fast_v = rearrange(fast_v, 'b s (n_h d) -> (b n_h) s d', n_h=self.num_fw_heads)

            # fast_q = rearrange(fast_q, 'b s (n_h d) -> b s n_h d', n_h=self.num_fw_heads)
            # fast_k = rearrange(fast_k, 'b s (n_h d) -> b s n_h d', n_h=self.num_fw_heads)
            # fast_v = rearrange(fast_v, 'b s (n_h d) -> b s n_h d', n_h=self.num_fw_heads)
        else:
            fast_q = rearrange(fast_q, 'b s n_h d -> (b n_h) s d')
            fast_k = rearrange(fast_k, 'b s n_h d -> (b n_h) s d')
            fast_v = rearrange(fast_v, 'b s n_h d -> (b n_h) s d')

        # (b n_h) s d
        if self.qk_l2_norm:
            fast_q = l2_norm(fast_q)
            fast_k = l2_norm(fast_k)

        if self.qkv_silu:
            fast_q = F.silu(fast_q, inplace=False)
            fast_k = F.silu(fast_k, inplace=False)
            fast_v = F.silu(fast_v, inplace=False)

        #### PRA input site: camera Plucker rotary on fast q/k after l2 norm.
        cam_coords = None
        if cam_coords6 is not None and self.cam_phase_mode == "plucker":
            cam_coords = cam_coords6 if cam_coords6.dim() == 3 else cam_coords6[None]
            cam_coords = cam_coords.float()
            assert cam_coords.shape[0] == b and cam_coords.shape[1] == s

        if self.ttt_input_rope and cam_coords is not None:
            with torch.autocast(device_type="cuda", enabled=False):
                in_cos, in_sin = cam_phase_tables(cam_coords, self.cam_omega_in, self.cam_gain_in)
                # [b, L, 6F] -> [(b n_fw_h), L, 6F] matching fast_q layout
                in_cos = in_cos.repeat_interleave(self.num_fw_heads, dim=0)
                in_sin = in_sin.repeat_interleave(self.num_fw_heads, dim=0)
            fast_q = apply_rotary_pairs(fast_q, in_cos, in_sin)
            fast_k = apply_rotary_pairs(fast_k, in_cos, in_sin)

        # fw_q = rearrange(fast_q, 'b s n_h d -> (b n_h) s d')
        # fw_k = rearrange(fast_k, 'b s n_h d -> (b n_h) s d')
        # fw_v = rearrange(fast_v, 'b s n_h d -> (b n_h) s d')


        lr = self.lr_proj(x) # [b, s, num_heads * lr_dim_per_head]
        if self.lr_parameterization == "mamba":
            lr = torch.nn.functional.softplus(lr.float() + self.base_lr_inv)
        else:
            raise NotImplementedError(f"LR parameterization {self.lr_parameterization} not implemented")
        fw_lr = rearrange(lr, 'b s (n_h lr_dim) -> (b n_h) s lr_dim', n_h=self.num_fw_heads)

        fw_lr1, fw_lr2, fw_lr3 = fw_lr.chunk(3, dim=-1)

        if self.training or fake_batch_size > 2:
            fw_w0 = self.w0.repeat(b, 1, 1) # [nh, d_h, d_in] -> [b*nh, d_h, d_in]
            fw_w1 = self.w1.repeat(b, 1, 1) # [nh, d_out, d_h] -> [b*nh, d_out, d_h]
            fw_w2 = self.w2.repeat(b, 1, 1) # [nh, d_h, d_in] -> [b*nh, d_h, d_in]

            if self.ttt_hidden_rope:
                assert self.update_every == self.mini_batch_size * 2, \
                    "hidden-rope carrier currently mirrors the no-repeat rope path"
                if self.cam_phase_mode == "plucker" and cam_coords is not None:
                    # Camera Plucker phases replace the grid-carrier phases.
                    assert b == 1, "camera hidden phases assume batch_size 1"
                    with torch.autocast(device_type="cuda", enabled=False):
                        h_cos, h_sin = cam_phase_tables(
                            cam_coords[0], self.cam_omega_h, self.cam_gain_h)
                        hcos = h_cos.transpose(0, 1).contiguous()  # [P, L]
                        hsin = h_sin.transpose(0, 1).contiguous()
                else:
                    # Build per-token hidden phases with the SAME token->(t,y,x)
                    # mapping as the fast q/k rope: pass a (1, 0)-pair carrier
                    # through rope_apply_ar with a hidden-sized freq table; the
                    # output pairs are exactly (cos, sin).
                    with torch.autocast(device_type="cuda", enabled=False):
                        seq_l = fast_q.shape[1]
                        table = torch.polar(
                            torch.ones(1024, self.h_rope_dim // 2, device=x.device),
                            torch.outer(
                                torch.arange(1024, device=x.device, dtype=torch.float32),
                                self.h_inv_freq.float(),
                            ),
                        )
                        carrier = torch.zeros(
                            1, seq_l, 1, self.h_rope_dim, device=x.device, dtype=torch.float32
                        )
                        carrier[..., 0::2] = 1.0
                        if src_prefix_len > 0:
                            roped = rope_apply_ar_src_prefix(
                                carrier, grid_sizes, table, self.ar_window_f,
                                self.n_latent_f, self.src_latent_f
                            )
                        else:
                            roped = rope_apply_ar(
                                carrier, grid_sizes, table, self.ar_window_f, self.n_latent_f
                            )
                        hcos = roped[0, :, 0, 0::2].transpose(0, 1).contiguous()  # [P, L]
                        hsin = roped[0, :, 0, 1::2].transpose(0, 1).contiguous()

                fw_x, fw_w0, fw_w1, fw_w2 = ar_fast_weight_swish_glu_weight_norm_mini_batch_hidden_rope(
                    fw_w0, fw_w1, fw_w2, fast_q, fast_k, fast_v,
                    fw_lr1, fw_lr2, fw_lr3,
                    hcos, hsin,
                    w_scale=self.w_scale,
                    mini_batch_size=self.mini_batch_size,
                    update_length=-1,
                    use_moun=self.use_moun,
                    update_every=self.update_every,
                    num_moun_iters=self.num_moun_iters,
                    weight_norm=self.weight_norm,
                    src_prefix_len=src_prefix_len,
                    delta_only=self.ttt_hrope_delta_only,
                )
            else:
                fw_x, fw_w0, fw_w1, fw_w2 = ar_fast_weight_swish_glu_weight_norm_mini_batch(
                    fw_w0, fw_w1, fw_w2, fast_q, fast_k, fast_v,
                    fw_lr1, fw_lr2, fw_lr3,
                    w_scale=self.w_scale,
                    mini_batch_size=self.mini_batch_size,
                    update_length=-1,
                    use_moun=self.use_moun,
                    update_every=self.update_every,
                    num_moun_iters=self.num_moun_iters,
                    weight_norm=self.weight_norm,
                    src_prefix_len=src_prefix_len,
                )
        else:
            # inference only. 
            # if self.cur_w0 is None:
            if self.cur_w0 is None or self.inference_frame_offset == 0 or self.cfg_w0 is None:
                fw_w0 = self.w0.clone().repeat(b, 1, 1) # [nh, d_h, d_in] -> [b*nh, d_h, d_in]
                fw_w1 = self.w1.clone().repeat(b, 1, 1) # [nh, d_out, d_h] -> [b*nh, d_out, d_h]
                fw_w2 = self.w2.clone().repeat(b, 1, 1) # [nh, d_h, d_in] -> [b*nh, d_h, d_in]
            else:
                if self.cfg_seq:
                    fw_w0 = self.cfg_w0
                    fw_w1 = self.cfg_w1
                    fw_w2 = self.cfg_w2
                else:
                    fw_w0 = self.cur_w0
                    fw_w1 = self.cur_w1
                    fw_w2 = self.cur_w2

            fw_x, fw_w0, fw_w1, fw_w2 = ar_fast_weight_swish_glu_weight_norm_mini_batch_inference(
                fw_w0, fw_w1, fw_w2, fast_q, fast_k, fast_v,
                fw_lr1, fw_lr2, fw_lr3,
                w_scale=self.w_scale,
                mini_batch_size=self.mini_batch_size,
                do_update=interleave,
                use_moun=self.use_moun,
                num_moun_iters=self.num_moun_iters,
                weight_norm=self.weight_norm,
            )
            if interleave:
                if self.cfg_seq:
                    self.cfg_w0 = fw_w0
                    self.cfg_w1 = fw_w1
                    self.cfg_w2 = fw_w2
                else:
                    self.cur_w0 = fw_w0
                    self.cur_w1 = fw_w1
                    self.cur_w2 = fw_w2


        ttt_x = self.output_norm(fw_x)
        ttt_x = rearrange(ttt_x, '(b n_h) s d -> b s (n_h d)', n_h=self.num_fw_heads)

        # do window attention here. now, q, k has shape of [true_bs, seq_len, n_h, d]
        if not self.no_time_rope:
            if self.training or fake_batch_size > 2:
                if src_prefix_len > 0:
                    q=rope_apply_ar_src_prefix(q, grid_sizes, freqs, self.ar_window_f, self.n_latent_f, self.src_latent_f)
                    k=rope_apply_ar_src_prefix(k, grid_sizes, freqs, self.ar_window_f, self.n_latent_f, self.src_latent_f)
                elif self.update_every == self.mini_batch_size * 2:
                    q=rope_apply_ar(q, grid_sizes, freqs, self.ar_window_f, self.n_latent_f)
                    k=rope_apply_ar(k, grid_sizes, freqs, self.ar_window_f, self.n_latent_f)
                else:
                    q=rope_apply_ar_with_repeat(q, grid_sizes, freqs, self.ar_window_f, self.n_latent_f, 
                                                interleave_freqs_time=self.cached_freqs_time,
                                                num_repeat=num_repeat)
                    k=rope_apply_ar_with_repeat(k, grid_sizes, freqs, self.ar_window_f, self.n_latent_f, 
                                                interleave_freqs_time=self.cached_freqs_time,
                                                num_repeat=num_repeat)
            else:
                q=rope_apply_ar_inference(q, grid_sizes, freqs, self.ar_window_f, self.inference_frame_offset, interleave)
                k=rope_apply_ar_inference(k, grid_sizes, freqs, self.ar_window_f, self.inference_frame_offset, interleave)
                
        else:
            raise NotImplementedError("No time rope not implemented")
            q=rope_apply_same_time(q, grid_sizes, freqs, self.rope_f_window_size)
            k=rope_apply_same_time(k, grid_sizes, freqs, self.rope_f_window_size)

        window_size = self.local_window_size
        num_chunks = self.n_latent_f // self.ar_window_f
        if self.training or fake_batch_size > 2:
            # [b, s, n_h, d]
            # x_window = basic_sliding_window_attention(q, k, v, self.mini_batch_size, self.update_every)
            if src_prefix_len > 0:
                # SRC windows attend only within themselves; TGT runs the
                # existing path. NO attention across the src/tgt boundary.
                x_src = src_prefix_window_attention(
                    q[:, :src_prefix_len], k[:, :src_prefix_len], v[:, :src_prefix_len],
                    n_src_chunks, self.mini_batch_size)
                x_tgt = batched_sliding_window_attention(
                    q[:, src_prefix_len:], k[:, src_prefix_len:], v[:, src_prefix_len:],
                    self.mini_batch_size, self.update_every, num_chunks,
                    kv_cache_size=self.kv_cache_size)
                x_window = torch.cat([x_src, x_tgt], dim=1)
            elif self.update_every == self.mini_batch_size * 2:
                x_window = batched_sliding_window_attention(q, k, v, self.mini_batch_size, self.update_every, num_chunks, kv_cache_size=self.kv_cache_size)
            else:
                x_window = batched_sliding_window_attention_with_repeated_chunks(q, k, v, self.mini_batch_size, self.update_every, num_chunks, 
                                                                                 kv_cache_size=self.kv_cache_size)
        else:
            # inference
            # [b, s, n_h, d]
            if self.cfg_seq:
                kv_cache = self.kv_cache_cfg
            else:
                kv_cache = self.kv_cache
            x_window = sliding_window_attention_inference(q, k, v, self.mini_batch_size, kv_cache, interleave, 
                                                          kv_cache_size=self.kv_cache_size)

            if interleave:
                # update the kv cache here
                k_cache = k[:, self.mini_batch_size - self.kv_cache_size : self.mini_batch_size, :, :]
                v_cache = v[:, self.mini_batch_size - self.kv_cache_size : self.mini_batch_size, :, :]
                # Stack is not memory efficient, but it's fine for now. TODO. 
                kv_cache = torch.concat([k_cache, v_cache], dim=0) # [2 * b, s, n_h, d]
                if self.cfg_seq:
                    self.kv_cache_cfg = kv_cache
                else:
                    self.kv_cache = kv_cache
        # output
        x_window = x_window.flatten(2)

        #### Merge ttt_x and x_window here.

        if self.learnable_ttt_scale:
            # TODO: added fused norm-gated (scalar gated)
            ttt_scale = F.silu(self.ttt_scale_proj(x), inplace=True)
        else:
            ttt_scale = self.ttt_scale
    
        
        x = ttt_x * ttt_scale + x_window
        

        extra_info = {}
        with torch.no_grad():
            # record the norm of some vectors.
            extra_info["statistics/fw_w0_norm"] = fw_w0.norm(dim=-1).mean().item()
            extra_info["statistics/fw_w1_norm"] = fw_w1.norm(dim=-1).mean().item()
            extra_info["statistics/fw_w2_norm"] = fw_w2.norm(dim=-1).mean().item()
            extra_info["statistics/fw_lr_norm"] = fw_lr.norm(dim=-1).mean().item()
            extra_info["statistics/fw_v_norm"] = fast_v.norm(dim=-1).mean().item()
            extra_info["statistics/tttx_norm"] = fw_x.norm(dim=-1).mean().item()
            extra_info["statistics/window_x_norm"] = x_window.norm(dim=-1).mean().item()
            extra_info["statistics/output_x_norm"] = x.norm(dim=-1).mean().item()
            if self.use_o_norm:
                extra_info["statistics/rms_norm_weight"] = self.output_norm.weight.detach().norm().item()
            if self.learnable_ttt_scale:
                extra_info["statistics/ttt_scale"] = ttt_scale.norm(dim=-1).mean().item()

        x = self.o(x)

        # reshape the output to the original shape
        x = rearrange(x, "b (nw sw) d -> (b nw) sw d", sw=s_per_window)

        # flip the cfg_seq flag. 
        self.cfg_seq = not self.cfg_seq
        return x, extra_info

    def get_trainable_params(self, **kwargs):
        return self.parameters()
        
