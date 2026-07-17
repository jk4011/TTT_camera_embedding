import torch.nn.functional as F
import torch


@torch.compile()
def silu_backprop(dy: torch.Tensor, x: torch.Tensor):
    """
    Args:
        dy: [b, d, l], gradient of the outer loss wrt the y
        x: [b, d, l], input of the silu activation
    outs:
        dx: [b, d, l], gradient of the outer loss wrt the x
        dx = dy * sigma * (1 + x * (1 - sigma))
    """
    sigma = torch.sigmoid(x)
    dx = dy * sigma * (1 + x * (1 - sigma))
    return dx


@torch.compile()
def l2_norm(x: torch.Tensor):
    """
    x: [b, l, d]
    """
    x_type = x.dtype
    ret = x / (x.norm(dim=-1, keepdim=True) + 1e-5)  # norm will upcast to float32
    return ret.type(x_type)


@torch.compile()
def zeropower_via_newtonschulz5(G):
    """
    This is an updated version of the zeropower_via_newtonschulz5 function in here:
    https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt_medium.py#L26
    The code is modified from https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py#L49, which contains the original muon implementation.
    Major change: G is [b, d, d] rather than [d, d]
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    Args:
        G: [b, d, d']
    Returns:
        X: [b, d, d']
    FLOPS:  When d=d', Total FLOPS=30 * b * d^3
    """
    assert len(G.shape) == 3
    X = G.bfloat16()
    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(1, 2), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for a, b, c in [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]:
        A = X @ X.transpose(1, 2)
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    return X


@torch.compile()
@torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16)
def block_causal_lact_swiglu(
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lr0: torch.Tensor,
    lr1: torch.Tensor,
    lr2: torch.Tensor,
    chunk_size: int = 2048,  # test-time training chunk size
    use_muon: bool = False,
    momentum: torch.Tensor = None,  # [b, s, 1]
):
    """
    Block causal LaCT with SwiGLU fast weight function.
        Apply then Update => Shifted Block Causal LaCT
    w0, w1, w2 are the fast weights. f(x) =  w1 @ (silu(w0 @ x) * (w2 @ x))

    About precision:
        w0, w1, w2 are mostly likely fp32.
        q, k, v are fp16.
        lr0, lr1, lr2 are fp32.
        The forward, backward produce bf16 gradients, updated fast weights are fp32.
        The final output are bf16.

    FLOPS:
        (assume dk=dv denoted as D, hidden dimension of swiglu-mlp is H, ignore muon, ignore last chunk)
        Forward pass with key: 4 * D * H * L * B
        Backward pass: 8 * D * H * L * B
        Forward with Query: 6 * D * H * L * B
        Total: 18 * D * H * L * B
    Outputs:
        o: [b, l, dv]
    """

    # adding detach here sometimes improves stability.
    w0_norm = w0.norm(dim=2, keepdim=True)
    w1_norm = w1.norm(dim=2, keepdim=True)
    w2_norm = w2.norm(dim=2, keepdim=True)

    if momentum is not None:
        dw1_momentum = torch.zeros_like(w1)
        dw0_momentum = torch.zeros_like(w0)
        dw2_momentum = torch.zeros_like(w2)

    q = q.transpose(1, 2)  # [b, dk, l]
    v = v.transpose(1, 2)

    output = torch.zeros_like(v)

    e_index = 0
    seq_len = k.shape[1]
    for i in range(0, seq_len - chunk_size, chunk_size):
        s_index = i
        e_index = s_index + chunk_size

        # [b, l, dk]
        ki = k[:, s_index:e_index, :]  # bf16
        # [b, dv, l]
        vi = v[:, :, s_index:e_index]  # bf16
        # [b, dh, l]
        qi = q[:, :, s_index:e_index]
        # [b, l, d/1] fp32
        lr1i = lr1[:, s_index:e_index, :]  # [b, l, d/1] fp32
        lr2i = lr2[:, s_index:e_index, :]  # [b, l, d/1] fp32
        lr0i = lr0[:, s_index:e_index, :]  # [b, l, d/1] fp32

        # use previous w0 and w1 to get the final output
        # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
        h = torch.bmm(w2, qi)
        gate = F.silu(torch.bmm(w0, qi), inplace=True)
        # [b, dv, dh] @ [b, dh, l] -> [b, dv, l] -> [b, l, dv]
        output[:, :, s_index:e_index] = torch.bmm(w1, gate * h)

        # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
        gate_before_act = torch.bmm(w0, ki.transpose(1, 2))
        hidden_before_mul = torch.bmm(w2, ki.transpose(1, 2))

        hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul

        # [b, dh, dv] @ [b, dv, l] -> [b, dh, l]
        dhidden = torch.bmm(w1.transpose(1, 2), vi)

        dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)

        dgate = dhidden * hidden_before_mul
        dgate_before_act = silu_backprop(dgate, gate_before_act)

        # [b, d_2, l] @ [b, l, d_1] -> [b, d_2, d_1]
        # in bmm two mat is fp32, but the result is bf16.
        # it's better to cast the mat to bf16 before bmm.
        # [b, dv, l] @ [b, l, dh] -> [b, dv, dh]
        # it's better to cast the mat to bf16 before bmm.
        dw1 = torch.bmm(vi, (hidden.transpose(1, 2) * lr1i).type_as(vi))  # [b, d, d]
        # [b, dh, l] @ [b, l, dk] -> [b, dh, dk]
        dw0 = torch.bmm(dgate_before_act, (ki * lr0i).type_as(dgate_before_act))
        dw2 = torch.bmm(dhidden_before_mul, (ki * lr2i).type_as(dhidden_before_mul))

        if momentum is not None:
            m_i = momentum[:, s_index:e_index, :]
            m_i = m_i.mean(dim=1, keepdim=True)

            dw0 = dw0 + dw0_momentum * m_i
            dw1 = dw1 + dw1_momentum * m_i
            dw2 = dw2 + dw2_momentum * m_i
            dw0_momentum = dw0
            dw1_momentum = dw1
            dw2_momentum = dw2

        if use_muon:
            dw1 = zeropower_via_newtonschulz5(dw1)
            dw0 = zeropower_via_newtonschulz5(dw0)
            dw2 = zeropower_via_newtonschulz5(dw2)
            # legacy code for different global lr for muon. Conclusion: 1.0 is good
            # if muon_w0_lr is not None:
            #     # lr is fp32 (after softplus)
            #     # in future version, we can cast it before input. TODO
            #     dw1 = (dw1 * muon_w1_lr).type_as(w1)
            #     dw0 = (dw0 * muon_w0_lr).type_as(w0)
            #     dw2 = (dw2 * muon_w2_lr).type_as(w2)

        w1 = w1 + dw1
        w0 = w0 + dw0
        w2 = w2 + dw2

        # Do channel-wise l2 norm.  conceptually like post-norm.
        w0 = w0 / (w0.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
        w1 = w1 / (w1.norm(dim=2, keepdim=True) + 1e-5) * w1_norm
        w2 = w2 / (w2.norm(dim=2, keepdim=True) + 1e-5) * w2_norm

    # for the last chunk, don't update the fast weights, directly apply the fast weights to the query.
    s_index = e_index
    e_index = seq_len

    qi = q[:, :, s_index:e_index]
    # use the last w0 and w1 to get the final output
    # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
    h = torch.bmm(w2, qi)
    gate = F.silu(torch.bmm(w0, qi), inplace=True)
    # [b, dv, dh] @ [b, dh, l] -> [b, dv, l] -> [b, l, dv]
    output[:, :, s_index:e_index] = torch.bmm(w1, gate * h)

    return output.transpose(1, 2)


@torch.compile()
@torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16)
def prenorm_block_causal_lact_swiglu(
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lr0: torch.Tensor,
    lr1: torch.Tensor,
    lr2: torch.Tensor,
    chunk_size: int = 2048,  # test-time training chunk size
    use_muon: bool = False,
    momentum: torch.Tensor = None,  # [b, s, 1]
    precess_cos: torch.Tensor = None,  # [P] Q25a chunk precession: at every chunk
    precess_sin: torch.Tensor = None,  # boundary, rotate the first 2P hidden-dim
    # pairs of the accumulated w1 DELTA by a fixed angle ladder. Stored addresses
    # precess with age -> a fixed recency kernel sum_p cos(delta_p * chunk_age) on
    # the update->apply inner product. The init readout w1_init is excluded (no
    # absolute-phase tax); right-rotation on the hidden axis preserves every w1
    # row norm, so the weight-norm step is unaffected. None = exact baseline.
):
    """
    Block causal LaCT with SwiGLU fast weight function.
        Apply then Update => Shifted Block Causal LaCT
    w0, w1, w2 are the fast weights. f(x) =  w1 @ (silu(w0 @ x) * (w2 @ x))

    About precision:
        w0, w1, w2 are mostly likely fp32.
        q, k, v are fp16.
        lr0, lr1, lr2 are fp32.
        The forward, backward produce bf16 gradients, updated fast weights are fp32.
        The final output are bf16.

    FLOPS:
        (assume dk=dv denoted as D, hidden dimension of swiglu-mlp is H, ignore muon, ignore last chunk)
        Forward pass with key: 4 * D * H * L * B
        Backward pass: 8 * D * H * L * B
        Forward with Query: 6 * D * H * L * B
        Total: 18 * D * H * L * B
    Outputs:
        o: [b, l, dv]
    """

    # adding detach here sometimes improves stability.
    w0_norm = w0.norm(dim=2, keepdim=True)
    w1_norm = w1.norm(dim=2, keepdim=True)
    w2_norm = w2.norm(dim=2, keepdim=True)

    w0_main, w1_main, w2_main = w0, w1, w2
    if precess_cos is not None:
        w1_init = w1.clone()
        P_pr = precess_cos.shape[0]

    if momentum is not None:
        dw1_momentum = torch.zeros_like(w1)
        dw0_momentum = torch.zeros_like(w0)
        dw2_momentum = torch.zeros_like(w2)

    q = q.transpose(1, 2)  # [b, dk, l]
    v = v.transpose(1, 2)

    output = torch.zeros_like(v)

    e_index = 0
    seq_len = k.shape[1]
    for i in range(0, seq_len - chunk_size, chunk_size):
        s_index = i
        e_index = s_index + chunk_size

        # [b, l, dk]
        ki = k[:, s_index:e_index, :]  # bf16
        # [b, dv, l]
        vi = v[:, :, s_index:e_index]  # bf16
        # [b, dh, l]
        qi = q[:, :, s_index:e_index]
        # [b, l, d/1] fp32
        lr1i = lr1[:, s_index:e_index, :]  # [b, l, d/1] fp32
        lr2i = lr2[:, s_index:e_index, :]  # [b, l, d/1] fp32
        lr0i = lr0[:, s_index:e_index, :]  # [b, l, d/1] fp32

        # use previous w0 and w1 to get the final output
        # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
        h = torch.bmm(w2, qi)
        gate = F.silu(torch.bmm(w0, qi), inplace=True)
        # [b, dv, dh] @ [b, dh, l] -> [b, dv, l] -> [b, l, dv]
        output[:, :, s_index:e_index] = torch.bmm(w1, gate * h)

        # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
        gate_before_act = torch.bmm(w0, ki.transpose(1, 2))
        hidden_before_mul = torch.bmm(w2, ki.transpose(1, 2))

        hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul

        # [b, dh, dv] @ [b, dv, l] -> [b, dh, l]
        dhidden = torch.bmm(w1.transpose(1, 2), vi)

        dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)

        dgate = dhidden * hidden_before_mul
        dgate_before_act = silu_backprop(dgate, gate_before_act)

        # [b, d_2, l] @ [b, l, d_1] -> [b, d_2, d_1]
        # in bmm two mat is fp32, but the result is bf16.
        # it's better to cast the mat to bf16 before bmm.
        # [b, dv, l] @ [b, l, dh] -> [b, dv, dh]
        # it's better to cast the mat to bf16 before bmm.
        dw1 = torch.bmm(vi, (hidden.transpose(1, 2) * lr1i).type_as(vi))  # [b, d, d]
        # [b, dh, l] @ [b, l, dk] -> [b, dh, dk]
        dw0 = torch.bmm(dgate_before_act, (ki * lr0i).type_as(dgate_before_act))
        dw2 = torch.bmm(dhidden_before_mul, (ki * lr2i).type_as(dhidden_before_mul))

        if momentum is not None:
            m_i = momentum[:, s_index:e_index, :]
            m_i = m_i.mean(dim=1, keepdim=True)

            dw0 = dw0 + dw0_momentum * m_i
            dw1 = dw1 + dw1_momentum * m_i
            dw2 = dw2 + dw2_momentum * m_i
            dw0_momentum = dw0
            dw1_momentum = dw1
            dw2_momentum = dw2

        if use_muon:
            dw1 = zeropower_via_newtonschulz5(dw1)
            dw0 = zeropower_via_newtonschulz5(dw0)
            dw2 = zeropower_via_newtonschulz5(dw2)
            # legacy code for different global lr for muon. Conclusion: 1.0 is good
            # if muon_w0_lr is not None:
            #     # lr is fp32 (after softplus)
            #     # in future version, we can cast it before input. TODO
            #     dw1 = (dw1 * muon_w1_lr).type_as(w1)
            #     dw0 = (dw0 * muon_w0_lr).type_as(w0)
            #     dw2 = (dw2 * muon_w2_lr).type_as(w2)

        w1_main = w1_main + dw1
        w0_main = w0_main + dw0
        w2_main = w2_main + dw2

        if precess_cos is not None:
            # Q25a: one fixed rotation step per chunk boundary on the delta's
            # hidden-dim pairs (an age-k update accumulates k rotations). fp32.
            d1 = w1_main - w1_init
            dr = d1[:, :, : 2 * P_pr].float().reshape(
                d1.shape[0], d1.shape[1], P_pr, 2)
            x1, x2 = dr[..., 0], dr[..., 1]
            pc, ps = precess_cos.float(), precess_sin.float()
            rot = torch.stack(
                [x1 * pc - x2 * ps, x1 * ps + x2 * pc], dim=-1
            ).reshape(d1.shape[0], d1.shape[1], 2 * P_pr).type_as(d1)
            w1_main = w1_init + torch.cat([rot, d1[:, :, 2 * P_pr:]], dim=2)

        # Do channel-wise l2 norm.  conceptually like post-norm.
        w0 = w0_main / (w0_main.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
        w1 = w1_main / (w1_main.norm(dim=2, keepdim=True) + 1e-5) * w1_norm
        w2 = w2_main / (w2_main.norm(dim=2, keepdim=True) + 1e-5) * w2_norm

    # for the last chunk, don't update the fast weights, directly apply the fast weights to the query.
    s_index = e_index
    e_index = seq_len

    qi = q[:, :, s_index:e_index]
    # use the last w0 and w1 to get the final output
    # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
    h = torch.bmm(w2, qi)
    gate = F.silu(torch.bmm(w0, qi), inplace=True)
    # [b, dv, dh] @ [b, dh, l] -> [b, dv, l] -> [b, l, dv]
    output[:, :, s_index:e_index] = torch.bmm(w1, gate * h)

    return output.transpose(1, 2)

def apply_rotary_cols(x, cos, sin):
    """Rotate adjacent row-pairs of column-major tokens.

    x: [b, d, l]; cos/sin: [P, l] (shared) or [b, P, l] (per-row, e.g.
    per-fw-head AdaFreq ladders) with 2P <= d. Rotates rows [0:2P].
    """
    P = cos.shape[-2]
    x_rot = x[:, : 2 * P, :].float().reshape(x.shape[0], P, 2, x.shape[2])
    x1, x2 = x_rot[:, :, 0], x_rot[:, :, 1]
    if cos.dim() == 3:
        c, s_ = cos, sin
    else:
        c, s_ = cos[None], sin[None]
    y = torch.stack((x1 * c - x2 * s_, x1 * s_ + x2 * c), dim=2)
    y = y.reshape(x.shape[0], 2 * P, x.shape[2]).type_as(x)
    if 2 * P == x.shape[1]:
        return y
    return torch.cat([y, x[:, 2 * P :, :]], dim=1)


def apply_block_rot(x, r):
    """LieRE: rotate the leading nb*bb rows of column-major tokens by per-token
    block-diagonal rotation matrices.

    x: [B, d, l]; r: [nb, bb, bb, l] fp32 (r[..., t] block-diagonal rotation
    for token t, shared across the B batch*head rows; last-dim token layout so
    chunk slicing is r[..., s:e] like hcos/hsin). Rotates rows [0:nb*bb]; the
    remaining rows pass through, mirroring apply_rotary_cols. Pure elementwise
    multiply + sum so the math stays fp32 under the kernel's bf16 autocast
    (elementwise ops are not autocast-downcast; einsum/bmm would be).

    Inverse rotation (R^T, since R is orthogonal): pass r.transpose(1, 2).
    """
    nb, bb = r.shape[0], r.shape[1]
    d_rot = nb * bb
    # y[B, n, i, l] = sum_j r[n, i, j, l] * x[B, n, j, l]
    xr = x[:, :d_rot, :].float().reshape(x.shape[0], nb, 1, bb, x.shape[2])
    y = (r[None] * xr).sum(dim=3)
    y = y.reshape(x.shape[0], d_rot, x.shape[2]).type_as(x)
    if d_rot == x.shape[1]:
        return y
    return torch.cat([y, x[:, d_rot:, :]], dim=1)


@torch.compile()
@torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16)
def prenorm_block_causal_lact_swiglu_hidden_rope(
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lr0: torch.Tensor,
    lr1: torch.Tensor,
    lr2: torch.Tensor,
    hcos: torch.Tensor,  # [P, seq_len] fp32 hidden-rotary coeffs (None iff r_blocks given)
    hsin: torch.Tensor,
    chunk_size: int = 2048,
    use_muon: bool = False,
    momentum: torch.Tensor = None,
    delta_only: bool = False,
    hnorm: str = "none",  # "none" | "rms" (whole hidden) | "rms_rot" (rotated dims only)
    h_basis: torch.Tensor = None,  # [d_h, d_h] learned rotation basis U (slow weight,
    # init I): hidden ADDRESS = R(theta) @ (U @ h) on both update and apply; the
    # score h_q^T U^T R(dtheta) U h_k stays purely relative, U=I reduces exactly
    # to the plain hidden rotary. Gives the hidden site the same freedom the
    # input site has via to_qkv (organize a rotation-friendly subspace).
    r_blocks: torch.Tensor = None,  # LieRE: [nb, b, b, seq_len] fp32 per-token
    # block-diagonal rotation matrices (matrix_exp of learnable skew generators,
    # orthogonal by construction). Mutually exclusive with hcos/hsin: when given,
    # the address rotation on update (keys) and apply (queries) uses these blocks
    # and the inner-loop backward uses the TRANSPOSE blocks (= inverse rotation).
):
    """prenorm_block_causal_lact_swiglu + h-PRA (hidden rotary).

    hnorm (Q9-EXT, user 2026-07-11): RMS-normalize the hidden activation BEFORE
    the rotation, on both update and apply, making the hidden code spherical
    like the L2-normalized input q/k (the F27c geometry hypothesis for why the
    input site pays a smaller absolute-phase tax). "rms" normalizes the whole
    hidden vector; "rms_rot" normalizes only the rotated dims, leaving the
    position-free content half's magnitudes intact. The update backward passes
    through the exact RMSNorm Jacobian, so the stored notes remain the exact
    gradient of the recall objective.

    The SwiGLU hidden activation is rotated by per-token position phases
    before meeting w1, on both the apply path (queries) and the update path
    (keys); the update backprop applies the inverse rotation. The value
    retrieval channel <R_t_q h(q), R_t_k h(k)> becomes relative in hidden
    space (see the NVS study; no attention analogue).

    delta_only (Q9): the apply splits the output matrix into its initial part
    and the accumulated fast-weight delta, and rotates the hidden only on the
    delta path:  o = w1_init @ h(q) + (w1 - w1_init) @ R(phi_q) h(q).
    Update->apply recall stays relative (stored addresses are rotated, and the
    delta path probes with the rotated query hidden), but the initial readout
    sees the UNROTATED hidden — removing the constant absolute-phase tax
    identified in F27b. With all phases equal it still reduces exactly to the
    baseline kernel.
    """
    assert hnorm in ("none", "rms", "rms_rot")
    assert not (delta_only and hnorm != "none"), "delta_only + hnorm not combined (keep variants separable)"
    assert (r_blocks is None) != (hcos is None), \
        "exactly one of hcos/hsin (fixed ladder) or r_blocks (LieRE) must be given"
    if r_blocks is not None:
        # keep the LieRE variant separable from the other hidden-rotary mutations
        assert not delta_only and hnorm == "none" and h_basis is None
    P_rot = hcos.shape[-2] if hcos is not None else None

    def _hn_fwd(x):
        """RMS-normalize columns of x [B, d, L] over the feature dim (mode-aware).
        Returns (y, y_for_bwd, rms) — y_for_bwd/rms cover only the normalized rows."""
        if hnorm == "rms":
            rms = (x.float().pow(2).mean(dim=1, keepdim=True) + 1e-6).sqrt()
            y = (x.float() / rms).type_as(x)
            return y, y, rms
        # rms_rot: normalize only the rotated rows [0:2P]
        xr = x[:, : 2 * P_rot, :]
        rms = (xr.float().pow(2).mean(dim=1, keepdim=True) + 1e-6).sqrt()
        yr = (xr.float() / rms).type_as(x)
        y = torch.cat([yr, x[:, 2 * P_rot :, :]], dim=1)
        return y, yr, rms

    def _hn_bwd(dy, y_bwd, rms):
        """Exact RMSNorm backward: dx = (dy - y * mean(dy*y)) / rms on normalized rows."""
        if hnorm == "rms":
            m = (dy.float() * y_bwd.float()).mean(dim=1, keepdim=True)
            return ((dy.float() - y_bwd.float() * m) / rms).type_as(dy)
        dyr = dy[:, : 2 * P_rot, :]
        m = (dyr.float() * y_bwd.float()).mean(dim=1, keepdim=True)
        dxr = ((dyr.float() - y_bwd.float() * m) / rms).type_as(dy)
        return torch.cat([dxr, dy[:, 2 * P_rot :, :]], dim=1)

    w0_norm = w0.norm(dim=2, keepdim=True)
    w1_norm = w1.norm(dim=2, keepdim=True)
    w2_norm = w2.norm(dim=2, keepdim=True)

    w0_main, w1_main, w2_main = w0, w1, w2
    if delta_only:
        w1_init = w1.clone()

    if momentum is not None:
        dw1_momentum = torch.zeros_like(w1)
        dw0_momentum = torch.zeros_like(w0)
        dw2_momentum = torch.zeros_like(w2)

    q = q.transpose(1, 2)
    v = v.transpose(1, 2)
    output = torch.zeros_like(v)

    e_index = 0
    seq_len = k.shape[1]
    for i in range(0, seq_len - chunk_size, chunk_size):
        s_index = i
        e_index = s_index + chunk_size

        ki = k[:, s_index:e_index, :]
        vi = v[:, :, s_index:e_index]
        qi = q[:, :, s_index:e_index]
        lr1i = lr1[:, s_index:e_index, :]
        lr2i = lr2[:, s_index:e_index, :]
        lr0i = lr0[:, s_index:e_index, :]
        if r_blocks is not None:
            ri = r_blocks[..., s_index:e_index]
            hci = hsi = None
        else:
            hci = hcos[..., s_index:e_index]
            hsi = hsin[..., s_index:e_index]

        # apply with previous weights; hidden of queries rotated by their phases
        h = torch.bmm(w2, qi)
        gate = F.silu(torch.bmm(w0, qi), inplace=True)
        hq = gate * h
        if hnorm != "none":
            hq, _, _ = _hn_fwd(hq)
        hq_addr = torch.matmul(h_basis, hq) if h_basis is not None else hq
        hq_rot = (apply_block_rot(hq_addr, ri) if r_blocks is not None
                  else apply_rotary_cols(hq_addr, hci, hsi))
        if delta_only:
            # initial readout unrotated; only the accumulated delta probes rotated
            output[:, :, s_index:e_index] = torch.bmm(w1_init, hq) + torch.bmm(
                w1 - w1_init, hq_rot)
        else:
            output[:, :, s_index:e_index] = torch.bmm(w1, hq_rot)

        gate_before_act = torch.bmm(w0, ki.transpose(1, 2))
        hidden_before_mul = torch.bmm(w2, ki.transpose(1, 2))
        hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul
        if hnorm != "none":
            hidden_n, hn_y, hn_rms = _hn_fwd(hidden)
        else:
            hidden_n = hidden
        hidden_addr = (torch.matmul(h_basis, hidden_n)
                       if h_basis is not None else hidden_n)
        hidden_rot = (apply_block_rot(hidden_addr, ri) if r_blocks is not None
                      else apply_rotary_cols(hidden_addr, hci, hsi))

        # backprop through the rotation: R^T (rotary with negated sin /
        # transposed LieRE blocks — R is orthogonal since the generator is skew)
        dhidden_rot = torch.bmm(w1.transpose(1, 2), vi)
        dhidden = (apply_block_rot(dhidden_rot, ri.transpose(1, 2))
                   if r_blocks is not None
                   else apply_rotary_cols(dhidden_rot, hci, -hsi))
        if h_basis is not None:
            # inner-loop gradient back through the basis: d(pre-basis hidden)
            dhidden = torch.matmul(h_basis.transpose(0, 1), dhidden)
        if hnorm != "none":
            # exact RMSNorm Jacobian back to the pre-norm hidden
            dhidden = _hn_bwd(dhidden, hn_y, hn_rms)

        dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)
        dgate = dhidden * hidden_before_mul
        dgate_before_act = silu_backprop(dgate, gate_before_act)

        dw1 = torch.bmm(vi, (hidden_rot.transpose(1, 2) * lr1i).type_as(vi))
        dw0 = torch.bmm(dgate_before_act, (ki * lr0i).type_as(dgate_before_act))
        dw2 = torch.bmm(dhidden_before_mul, (ki * lr2i).type_as(dhidden_before_mul))

        if momentum is not None:
            m_i = momentum[:, s_index:e_index, :]
            m_i = m_i.mean(dim=1, keepdim=True)
            dw0 = dw0 + dw0_momentum * m_i
            dw1 = dw1 + dw1_momentum * m_i
            dw2 = dw2 + dw2_momentum * m_i
            dw0_momentum = dw0
            dw1_momentum = dw1
            dw2_momentum = dw2

        if use_muon:
            dw1 = zeropower_via_newtonschulz5(dw1)
            dw0 = zeropower_via_newtonschulz5(dw0)
            dw2 = zeropower_via_newtonschulz5(dw2)

        w1_main = w1_main + dw1
        w0_main = w0_main + dw0
        w2_main = w2_main + dw2

        w0 = w0_main / (w0_main.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
        w1 = w1_main / (w1_main.norm(dim=2, keepdim=True) + 1e-5) * w1_norm
        w2 = w2_main / (w2_main.norm(dim=2, keepdim=True) + 1e-5) * w2_norm

    s_index = e_index
    e_index = seq_len

    qi = q[:, :, s_index:e_index]
    h = torch.bmm(w2, qi)
    gate = F.silu(torch.bmm(w0, qi), inplace=True)
    hq = gate * h
    if hnorm != "none":
        hq, _, _ = _hn_fwd(hq)
    hq_addr = torch.matmul(h_basis, hq) if h_basis is not None else hq
    hq_rot = (apply_block_rot(hq_addr, r_blocks[..., s_index:e_index])
              if r_blocks is not None
              else apply_rotary_cols(hq_addr, hcos[..., s_index:e_index],
                                     hsin[..., s_index:e_index]))
    if delta_only:
        output[:, :, s_index:e_index] = torch.bmm(w1_init, hq) + torch.bmm(
            w1 - w1_init, hq_rot)
    else:
        output[:, :, s_index:e_index] = torch.bmm(w1, hq_rot)

    return output.transpose(1, 2)


@torch.compile()
@torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16)
def prenorm_block_causal_lact_swiglu_value_rope(
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lr0: torch.Tensor,
    lr1: torch.Tensor,
    lr2: torch.Tensor,
    vcos: torch.Tensor,  # [P_v, seq_len] fp32 value-rotary coeffs
    vsin: torch.Tensor,
    chunk_size: int = 2048,
    use_muon: bool = False,
    momentum: torch.Tensor = None,
    delta_only: bool = True,
):
    """prenorm_block_causal_lact_swiglu + VaPE (value-path rotary, Q23).

    Unlike the hidden rotary (which rotates the ADDRESS h before w1 and
    therefore needs an inverse rotation inside the inner-loop backward), VaPE
    rotates the regression TARGET: the update trains f_W toward R_t v_t
    instead of v_t. Nothing inside f_W is rotated, so every inner-loop
    gradient follows from the rotated target with NO inverse rotation
    anywhere in the chain:
        dhidden = w1^T (R_t v_t),   dw1 = (R_t v_t) (hidden * lr1)^T,
    and dw0/dw2 flow from dhidden exactly as in the baseline. Note the ROW
    index of w1 is the value dim, so R_t mixes w1 rows; the per-channel
    weight-norm (norm over dim=2, the hidden axis, one norm per output row)
    re-fixes row norms after every update exactly as it does for any dw1.

    On apply the readout is counter-rotated by the query position, so the
    update->apply retrieval carries only RELATIVE phases: to first order the
    delta contribution is sum_t lr_t <h_t, h_s> R_{t-s} v_t.

    delta_only (the default): only the accumulated fast-weight delta readout
    is counter-rotated; the initial readout is left untouched, removing the
    absolute-phase tax on the init readout (F27b):
        o = w1 @ h(q) + [ R_s^{-1}(dW1 h(q)) - dW1 h(q) ],  dW1 = w1 - w1_init
    (algebraically o = w1_init @ h + R_s^{-1}(dW1 h); the bracketed form is
    used because the bracket is exactly 0 when R = I, so zero phases reduce
    BIT-EXACTLY to the baseline kernel).
    Full variant (delta_only=False): o = R_s^{-1} (w1 @ h(q)).
    """
    w0_norm = w0.norm(dim=2, keepdim=True)
    w1_norm = w1.norm(dim=2, keepdim=True)
    w2_norm = w2.norm(dim=2, keepdim=True)

    w0_main, w1_main, w2_main = w0, w1, w2
    if delta_only:
        w1_init = w1.clone()

    if momentum is not None:
        dw1_momentum = torch.zeros_like(w1)
        dw0_momentum = torch.zeros_like(w0)
        dw2_momentum = torch.zeros_like(w2)

    q = q.transpose(1, 2)  # [b, dk, l]
    v = v.transpose(1, 2)  # [b, dv, l]
    output = torch.zeros_like(v)

    e_index = 0
    seq_len = k.shape[1]
    for i in range(0, seq_len - chunk_size, chunk_size):
        s_index = i
        e_index = s_index + chunk_size

        ki = k[:, s_index:e_index, :]
        vi = v[:, :, s_index:e_index]
        qi = q[:, :, s_index:e_index]
        lr1i = lr1[:, s_index:e_index, :]
        lr2i = lr2[:, s_index:e_index, :]
        lr0i = lr0[:, s_index:e_index, :]
        vci = vcos[..., s_index:e_index]
        vsi = vsin[..., s_index:e_index]

        # apply with previous weights; the readout (value dim) is
        # counter-rotated by the query position (inverse rotation = (cos, -sin))
        h = torch.bmm(w2, qi)
        gate = F.silu(torch.bmm(w0, qi), inplace=True)
        hq = gate * h
        if delta_only:
            o_delta = torch.bmm(w1 - w1_init, hq)
            output[:, :, s_index:e_index] = torch.bmm(w1, hq) + (
                apply_rotary_cols(o_delta, vci, -vsi) - o_delta)
        else:
            output[:, :, s_index:e_index] = apply_rotary_cols(
                torch.bmm(w1, hq), vci, -vsi)

        gate_before_act = torch.bmm(w0, ki.transpose(1, 2))
        hidden_before_mul = torch.bmm(w2, ki.transpose(1, 2))
        hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul

        # rotated regression target R_t v_t; the rest of the update backward
        # is IDENTICAL to the baseline (no inverse rotation needed: the
        # rotation sits on the target, not inside f_W)
        vi_rot = apply_rotary_cols(vi, vci, vsi)

        # [b, dh, dv] @ [b, dv, l] -> [b, dh, l]
        dhidden = torch.bmm(w1.transpose(1, 2), vi_rot)

        dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)
        dgate = dhidden * hidden_before_mul
        dgate_before_act = silu_backprop(dgate, gate_before_act)

        # dw1 = (R_t v_t) (hidden * lr1)^T
        dw1 = torch.bmm(vi_rot, (hidden.transpose(1, 2) * lr1i).type_as(vi_rot))
        dw0 = torch.bmm(dgate_before_act, (ki * lr0i).type_as(dgate_before_act))
        dw2 = torch.bmm(dhidden_before_mul, (ki * lr2i).type_as(dhidden_before_mul))

        if momentum is not None:
            m_i = momentum[:, s_index:e_index, :]
            m_i = m_i.mean(dim=1, keepdim=True)
            dw0 = dw0 + dw0_momentum * m_i
            dw1 = dw1 + dw1_momentum * m_i
            dw2 = dw2 + dw2_momentum * m_i
            dw0_momentum = dw0
            dw1_momentum = dw1
            dw2_momentum = dw2

        if use_muon:
            dw1 = zeropower_via_newtonschulz5(dw1)
            dw0 = zeropower_via_newtonschulz5(dw0)
            dw2 = zeropower_via_newtonschulz5(dw2)

        w1_main = w1_main + dw1
        w0_main = w0_main + dw0
        w2_main = w2_main + dw2

        # Do channel-wise l2 norm.  conceptually like post-norm.
        w0 = w0_main / (w0_main.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
        w1 = w1_main / (w1_main.norm(dim=2, keepdim=True) + 1e-5) * w1_norm
        w2 = w2_main / (w2_main.norm(dim=2, keepdim=True) + 1e-5) * w2_norm

    # for the last chunk, don't update the fast weights, directly apply the
    # fast weights to the query — the tail apply must counter-rotate too
    # (missing it here was a past bug class in this repo).
    s_index = e_index
    e_index = seq_len

    qi = q[:, :, s_index:e_index]
    vci = vcos[..., s_index:e_index]
    vsi = vsin[..., s_index:e_index]
    h = torch.bmm(w2, qi)
    gate = F.silu(torch.bmm(w0, qi), inplace=True)
    hq = gate * h
    if delta_only:
        o_delta = torch.bmm(w1 - w1_init, hq)
        output[:, :, s_index:e_index] = torch.bmm(w1, hq) + (
            apply_rotary_cols(o_delta, vci, -vsi) - o_delta)
    else:
        output[:, :, s_index:e_index] = apply_rotary_cols(
            torch.bmm(w1, hq), vci, -vsi)

    return output.transpose(1, 2)


@torch.compile()
@torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16)
def prenorm_block_causal_lact_swiglu_branch_rope(
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    q_gate: torch.Tensor,  # q copy feeding the silu GATE branch (w0)
    k_gate: torch.Tensor,  # k copy feeding the silu GATE branch (w0)
    q_cont: torch.Tensor,  # q copy feeding the linear CONTENT branch (w2)
    k_cont: torch.Tensor,  # k copy feeding the linear CONTENT branch (w2)
    v: torch.Tensor,
    lr0: torch.Tensor,
    lr1: torch.Tensor,
    lr2: torch.Tensor,
    chunk_size: int = 2048,
    use_muon: bool = False,
    momentum: torch.Tensor = None,  # [b, s, 1]
):
    """prenorm_block_causal_lact_swiglu + GbR (gate-branch-only input rope, Q25b).

    Diagnostic decomposition of the input fast-rope: the SwiGLU fast weight
    f(x) = w1 (silu(w0 x) * (w2 x)) has TWO input branches — the silu GATE
    branch (w0) and the linear CONTENT branch (w2). The standard input rope
    rotates the q/k feeding both. This kernel takes two independent q/k pairs:
    the caller passes (rotated, plain) for gate-only rope or (plain, rotated)
    for content-only rope, localizing WHERE the input rope's value lives.

    Exact function: the update trains on
        f(k) = w1 (silu(w0 k_gate) * (w2 k_cont)),
    so every inner-loop gradient follows from that function — dw0's input row
    is k_gate, dw2's is k_cont, and the shared upstream terms (silu gating in
    dhidden_before_mul, hidden_before_mul in dgate) each use their own
    branch's pre-activation. The kernel never backprops to k inside the inner
    loop (only dw0/dw1/dw2 are formed), so no dk split is needed; outer-loop
    autograd differentiates through both copies automatically. The apply
    blocks (chunk loop + tail) compute the same two matmuls from q_gate /
    q_cont. Momentum / muon / weight-norm are identical to the baseline.

    With q_gate == q_cont and k_gate == k_cont (same tensors in both slots),
    every op matches prenorm_block_causal_lact_swiglu bit-exactly.
    """
    w0_norm = w0.norm(dim=2, keepdim=True)
    w1_norm = w1.norm(dim=2, keepdim=True)
    w2_norm = w2.norm(dim=2, keepdim=True)

    w0_main, w1_main, w2_main = w0, w1, w2

    if momentum is not None:
        dw1_momentum = torch.zeros_like(w1)
        dw0_momentum = torch.zeros_like(w0)
        dw2_momentum = torch.zeros_like(w2)

    q_gate = q_gate.transpose(1, 2)  # [b, dk, l]
    q_cont = q_cont.transpose(1, 2)  # [b, dk, l]
    v = v.transpose(1, 2)

    output = torch.zeros_like(v)

    e_index = 0
    seq_len = k_gate.shape[1]
    for i in range(0, seq_len - chunk_size, chunk_size):
        s_index = i
        e_index = s_index + chunk_size

        # [b, l, dk]
        kgi = k_gate[:, s_index:e_index, :]  # bf16
        kci = k_cont[:, s_index:e_index, :]  # bf16
        # [b, dv, l]
        vi = v[:, :, s_index:e_index]  # bf16
        # [b, dk, l]
        qgi = q_gate[:, :, s_index:e_index]
        qci = q_cont[:, :, s_index:e_index]
        # [b, l, d/1] fp32
        lr1i = lr1[:, s_index:e_index, :]
        lr2i = lr2[:, s_index:e_index, :]
        lr0i = lr0[:, s_index:e_index, :]

        # apply with previous weights: gate branch reads q_gate, content q_cont
        # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
        h = torch.bmm(w2, qci)
        gate = F.silu(torch.bmm(w0, qgi), inplace=True)
        # [b, dv, dh] @ [b, dh, l] -> [b, dv, l]
        output[:, :, s_index:e_index] = torch.bmm(w1, gate * h)

        # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
        gate_before_act = torch.bmm(w0, kgi.transpose(1, 2))
        hidden_before_mul = torch.bmm(w2, kci.transpose(1, 2))

        hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul

        # [b, dh, dv] @ [b, dv, l] -> [b, dh, l]
        dhidden = torch.bmm(w1.transpose(1, 2), vi)

        dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)

        dgate = dhidden * hidden_before_mul
        dgate_before_act = silu_backprop(dgate, gate_before_act)

        # [b, dv, l] @ [b, l, dh] -> [b, dv, dh]
        dw1 = torch.bmm(vi, (hidden.transpose(1, 2) * lr1i).type_as(vi))
        # dw0's input row is the GATE branch's k; dw2's is the CONTENT branch's
        # [b, dh, l] @ [b, l, dk] -> [b, dh, dk]
        dw0 = torch.bmm(dgate_before_act, (kgi * lr0i).type_as(dgate_before_act))
        dw2 = torch.bmm(dhidden_before_mul, (kci * lr2i).type_as(dhidden_before_mul))

        if momentum is not None:
            m_i = momentum[:, s_index:e_index, :]
            m_i = m_i.mean(dim=1, keepdim=True)

            dw0 = dw0 + dw0_momentum * m_i
            dw1 = dw1 + dw1_momentum * m_i
            dw2 = dw2 + dw2_momentum * m_i
            dw0_momentum = dw0
            dw1_momentum = dw1
            dw2_momentum = dw2

        if use_muon:
            dw1 = zeropower_via_newtonschulz5(dw1)
            dw0 = zeropower_via_newtonschulz5(dw0)
            dw2 = zeropower_via_newtonschulz5(dw2)

        w1_main = w1_main + dw1
        w0_main = w0_main + dw0
        w2_main = w2_main + dw2

        # Do channel-wise l2 norm.  conceptually like post-norm.
        w0 = w0_main / (w0_main.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
        w1 = w1_main / (w1_main.norm(dim=2, keepdim=True) + 1e-5) * w1_norm
        w2 = w2_main / (w2_main.norm(dim=2, keepdim=True) + 1e-5) * w2_norm

    # for the last chunk, don't update the fast weights, directly apply the
    # fast weights to the query — same branch split as the chunk-loop apply.
    s_index = e_index
    e_index = seq_len

    qgi = q_gate[:, :, s_index:e_index]
    qci = q_cont[:, :, s_index:e_index]
    # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
    h = torch.bmm(w2, qci)
    gate = F.silu(torch.bmm(w0, qgi), inplace=True)
    # [b, dv, dh] @ [b, dh, l] -> [b, dv, l]
    output[:, :, s_index:e_index] = torch.bmm(w1, gate * h)

    return output.transpose(1, 2)
