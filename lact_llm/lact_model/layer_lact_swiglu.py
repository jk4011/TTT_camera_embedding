# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from transformers.utils import logging

from fla.models.utils import Cache
from fla.modules import RMSNorm, RotaryEmbedding
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange, repeat

from .ttt_operation import (
    block_causal_lact_swiglu,
    prenorm_block_causal_lact_swiglu,
    prenorm_block_causal_lact_swiglu_hidden_rope,
    prenorm_block_causal_lact_swiglu_value_rope,
    prenorm_block_causal_lact_swiglu_branch_rope,
    l2_norm,
)

from .ttt_operation_fused_kernel import (
    postnorm_block_causal_lact_swiglu_fused_kernel_triton,
    prenorm_block_causal_lact_swiglu_fused_kernel_triton,
)

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
except ImportError:
    warnings.warn(
        "Flash Attention is not installed. Please install it via `pip install flash-attn --no-build-isolation`",
        category=ImportWarning,
    )
    flash_attn_func = None

logger = logging.get_logger(__name__)


def inv_softplus(x):
    if isinstance(x, torch.Tensor):
        y = x + torch.log(-torch.expm1(-x))
    else:
        y = x + math.log(-math.expm1(-x))
    return y


# ttt_sharedf: cross-layer shared learnable-frequency Parameters (Q13-LLM)
_SHAREDF_REGISTRY = {}


def _liere_doubling_scan(step, n_pos: int):
    """All integer powers step^0 .. step^(n_pos-1) of per-block rotation
    matrices step [nb, b, b] -> [n_pos, nb, b, b] (same dtype as step).

    Hierarchical doubling: r holds [R_0 .. R_{L-1}]; one level appends
    [R_0 P .. R_{L-1} P] with P = step^L, so each position is touched once
    overall. Block products are elementwise mul + sum (tiny batched matmuls
    are launch-bound in eager cuBLAS). Deliberately NOT torch.compile'd:
    inductor's fp64 triton codegen is ~3x slower than eager here (measured
    +132% total step overhead vs +44% eager for b=8)."""
    nb, b = step.shape[0], step.shape[1]
    r = torch.eye(b, dtype=step.dtype, device=step.device)[None, None].repeat(1, nb, 1, 1)
    pow_l = step  # step^L with L = r.shape[0]
    while r.shape[0] < n_pos:
        # upper[t, n, i, j] = sum_m r[t, n, i, m] * pow_l[n, m, j]
        upper = (r.unsqueeze(-1) * pow_l[None, :, None, :, :]).sum(dim=3)
        r = torch.cat([r, upper], dim=0)
        if r.shape[0] < n_pos:
            pow_l = (pow_l.unsqueeze(-1) * pow_l[:, None, :, :]).sum(dim=2)
    return r


class LowRankFastWeight(nn.Module):
    """
    Low rank fast weight. This is a compromise to keep the number of parameters low when comparing against baselines.
    Idealy, low-rank parameterization always hurts the performance.
    Args:
        num_heads: number of heads
        out_features: output features
        in_features: input features
        rank: rank of the low rank fast weight
        init_gain: initialization gain
        add_identity: whether to add identity matrix to the fast weight
    Returns:
        W: [num_heads, out_features, in_features]
    W = W_left @ W_right + I * 0.5
        where I is the identity matrix if add_identity is True.
    """

    def __init__(
        self,
        num_heads,
        out_features,
        in_features,
        rank=32,
        init_gain=0.5,
        add_identity=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        self.in_features = in_features
        self.rank = rank
        self.add_identity = add_identity

        self.w_left = nn.Parameter(torch.randn(num_heads, out_features, rank))
        self.w_right = nn.Parameter(torch.randn(num_heads, rank, in_features))
        self.init_gain = init_gain

        print("init low rank fast weight", num_heads, out_features, in_features, rank)

    def _init_weights(self):

        nn.init.normal_(self.w_left, std=1.0 / math.sqrt(self.rank) * self.init_gain)
        nn.init.normal_(
            self.w_right, std=1.0 / math.sqrt(self.in_features) * self.init_gain
        )

    def forward(
        self,
    ):
        """
        Returns:
            W: [num_heads, out_features, in_features]
            W = W_left @ W_right + I * 0.5
            where I is the identity matrix if add_identity is True.
        """

        W = self.w_left @ self.w_right

        if self.add_identity:
            W += (
                torch.eye(
                    self.out_features, self.in_features, device=W.device, dtype=W.dtype
                ).unsqueeze(0)
                * 0.5
            )

        return W


class LaCTSWIGLULayer(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_attn_heads: int,
        num_lact_heads: int,
        inter_multi: float,
        window_size: int,
        lact_chunk_size: int,
        qkv_bias: bool = False,
        attn_qk_norm: bool = True,
        qkv_silu: bool = True,
        no_v_silu: bool = False,
        lr_dim: int = 1,
        use_muon: bool = False,
        lr_parameterization: str = "mamba",
        learnable_ttt_scale: bool = False,
        ttt_prenorm: bool = False,
        ttt_nope: bool = False,
        ttt_hidden_rope: bool = False,
        ttt_hrope_frac: float = 0.5,
        ttt_hrope_gain: float = 1.0,
        ttt_hrope_theta: float = None,
        ttt_input_theta: float = None,
        ttt_hrope_interleave: bool = False,
        ttt_perhead_freqs: bool = False,
        ttt_liere: int = 0,
        ttt_liere_init: str = "rope",
        ttt_hrope_delta_only: bool = False,
        ttt_hrope_chunkq: int = 0,
        ttt_input_chunkq: int = 0,
        ttt_w1_precess: float = 0.0,
        ttt_hrope_conjpairs: bool = False,
        ttt_hrope_hnorm: str = "none",
        ttt_value_rope: bool = False,
        ttt_vrope_frac: float = 0.5,
        ttt_vrope_gain: float = 1.0,
        ttt_vrope_theta: float = None,
        ttt_vrope_delta_only: bool = True,
        ttt_branch_rope: str = "none",
        ttt_learnable_freqs: bool = False,
        ttt_sharedf: bool = False,
        ttt_learnable_input_freqs: bool = True,
        ttt_input_freq_tilt=None,
        ttt_hidden_basis: bool = False,
        ttt_freq_tilt: float = 0.1,
        rope_theta: float = 500000.0,
        layer_idx: int = None,
        max_position_embeddings: int = 2048,
        w0_w2_low_rank: int = -1,
        use_momentum: bool = False,
        ttt_loss_type: str = "dot_product",
        fw_init_gain: float = 0.5,  # init the fast weights
        use_fused_kernel: bool = False,
        fp32_states: bool = False,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_attn_heads  # num of heads for attention
        self.inter_multi = inter_multi
        self.window_size = window_size
        # head dim for attention
        self.head_dim = hidden_size // num_attn_heads

        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)

        self.attn_qk_norm = attn_qk_norm
        if self.attn_qk_norm:
            self.q_norm = RMSNorm(self.hidden_size)
            self.k_norm = RMSNorm(self.hidden_size)

        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.rope_theta = rope_theta
        self.rotary = RotaryEmbedding(dim=self.head_dim, base=self.rope_theta)
        # band-split: a separate, LOCAL-band input rope for the fast q/k
        self.ttt_input_theta = ttt_input_theta
        if ttt_input_theta is not None:
            self.fast_rotary = RotaryEmbedding(dim=self.head_dim, base=float(ttt_input_theta))
        else:
            self.fast_rotary = self.rotary
        self.layer_idx = layer_idx
        self.max_position_embeddings = max_position_embeddings

        ### Fast Weight init
        self.use_muon = use_muon
        self.lact_chunk_size = lact_chunk_size
        self.num_fw_heads = num_lact_heads
        self.fw_head_dim = self.hidden_size // self.num_fw_heads
        self.qkv_silu = qkv_silu
        self.no_v_silu = no_v_silu
        self.ttt_prenorm = ttt_prenorm
        self.ttt_nope = ttt_nope

        d_in, d_out = self.fw_head_dim, self.fw_head_dim
        d_h = int(d_in * inter_multi)

        self.d_h = d_h
        self.d_in = d_in
        self.d_out = d_out
        self.w0_w2_low_rank = w0_w2_low_rank
        self.fw_init_gain = fw_init_gain

        # Low Rank parameterization of the fast weights.
        # This is a compromise to keep the number of parameters low when comparing against baselines.
        # Idealy, low-rank parameterization always hurts the performance.
        if self.w0_w2_low_rank > 0:
            self.w0 = LowRankFastWeight(
                self.num_fw_heads,
                d_h,
                d_in,
                self.w0_w2_low_rank,
                init_gain=self.fw_init_gain,
                add_identity=True,
            )
            self.w2 = LowRankFastWeight(
                self.num_fw_heads,
                d_h,
                d_in,
                self.w0_w2_low_rank,
                init_gain=self.fw_init_gain,
                add_identity=True,
            )
        else:
            self.w0 = nn.Parameter(
                torch.randn(self.num_fw_heads, int(d_h), d_in) / math.sqrt(d_in)
            )  # [num_fw_heads, d_h, d_in]
            self.w2 = nn.Parameter(
                torch.randn(self.num_fw_heads, int(d_h), d_in) / math.sqrt(d_in)
            )  # [num_fw_heads, d_h, d_in]
        self.w1 = nn.Parameter(
            torch.randn(self.num_fw_heads, int(d_out), d_h) / math.sqrt(d_h)
        )  # [num_fw_heads, d_out, d_h]

        #### Per-Token LR parameterization.
        self.lr_dim = int(lr_dim * 3 * self.num_fw_heads)
        self.lr_proj = nn.Linear(self.hidden_size, self.lr_dim)
        base_lr = 0.001
        # Lr parameterization and initialization
        if lr_parameterization.lower() == "mamba":
            self.base_lr_inv = inv_softplus(base_lr)
        self.lr_parameterization = lr_parameterization

        #### per-channel scaling and offset for Q, and K.
        self.qk_scale = nn.Parameter(torch.ones(hidden_size, 2))
        self.qk_offset = nn.Parameter(torch.zeros(hidden_size, 2))
        self.learnable_ttt_scale = learnable_ttt_scale
        if self.learnable_ttt_scale:
            # per-head scaling.
            self.ttt_scale_proj = nn.Linear(hidden_size, self.num_fw_heads)

        # ttt output norm per head.
        self.ttt_norm = RMSNorm(self.fw_head_dim, elementwise_affine=True)

        self.use_momentum = use_momentum
        if self.use_momentum:
            self.momentum_proj = nn.Sequential(
                nn.Linear(hidden_size, self.num_fw_heads),
                nn.Sigmoid(),
            )

        self.ttt_loss_type = ttt_loss_type
        self.use_fused_kernel = use_fused_kernel
        self.fp32_states = fp32_states

        #### PRA transplant (1D): hidden rotary (h-PRA) + learnable freqs (omega_map).
        self.ttt_hidden_rope = ttt_hidden_rope
        self.ttt_hrope_delta_only = ttt_hrope_delta_only
        # Q22 chunk-quantized phases: positions floored to multiples of C before
        # building the rotary angles (0 = off). Behavioral flags, no parameters.
        self.ttt_hrope_chunkq = int(ttt_hrope_chunkq)
        self.ttt_input_chunkq = int(ttt_input_chunkq)
        # Q25a chunk precession of the w1 delta (gain of the per-chunk angle
        # ladder; 0 = off). Fixed angles delta_p = gain/theta^(p/P) * C over the
        # first half of the hidden dims; composes with the input rope (the
        # standard prenorm kernel path only).
        self.ttt_w1_precess = float(ttt_w1_precess)
        if self.ttt_w1_precess > 0.0:
            assert ttt_prenorm and not use_fused_kernel and not ttt_hidden_rope, \
                "w1 precession: prenorm baseline kernel path only"
            P_pr = max(1, self.d_h // 4)
            pr_theta = ttt_hrope_theta if ttt_hrope_theta is not None else rope_theta
            pr_ang = (self.ttt_w1_precess
                      / (pr_theta ** (torch.arange(P_pr).float() / P_pr))
                      ) * float(lact_chunk_size)
            self.register_buffer("precess_cos", pr_ang.cos(), persistent=False)
            self.register_buffer("precess_sin", pr_ang.sin(), persistent=False)
        self.ttt_hrope_hnorm = ttt_hrope_hnorm
        self.ttt_learnable_freqs = ttt_learnable_freqs
        self.ttt_sharedf = ttt_sharedf

        def _freq_param(name, tensor):
            """Learnable frequency Parameter. With ttt_sharedf ONE Parameter
            (per name/shape) is created by the first layer and shared by every
            later layer (named_parameters dedupes, so the optimizer sees it
            once). Registry is module-level: one model per process."""
            if not ttt_sharedf:
                return nn.Parameter(tensor)
            key = (name, tuple(tensor.shape))
            if key not in _SHAREDF_REGISTRY:
                _SHAREDF_REGISTRY[key] = nn.Parameter(tensor)
            return _SHAREDF_REGISTRY[key]

        if ttt_hidden_rope:
            # RoPE-style ladder over positions. Q9 GA genes: ttt_hrope_frac sets the
            # fraction of hidden dims rotated (0.5 reproduces the F27 setting of
            # P_h = (d_h//2)//2), ttt_hrope_gain scales the whole ladder, and
            # ttt_hrope_theta overrides the ladder base (default: attention rope_theta).
            P_h = max(1, int(self.d_h * ttt_hrope_frac / 2.0))
            assert 2 * P_h <= self.d_h, f"ttt_hrope_frac too large: {ttt_hrope_frac}"
            self.ttt_hidden_basis = ttt_hidden_basis
            if ttt_hidden_basis:
                assert not ttt_hrope_delta_only, "basis + delta_only not supported"
                # per-layer, head-shared learned basis U = I + V, V zero-init:
                # exact baseline at init, and weight decay pulls U toward the
                # IDENTITY (not toward zero, which would shrink the addresses)
                self.h_basis_delta = nn.Parameter(torch.zeros(self.d_h, self.d_h))
                self.register_buffer("h_basis_eye", torch.eye(self.d_h), persistent=False)
                # "orth" variant (Q14 wave 2): U = expm(A - A^T), always a
                # rotation matrix -> norm-preserving (the dense variant's +0.28
                # control tax was address-norm distortion, the F3 mechanism);
                # A = h_basis_delta, zero-init -> U = I exactly at init, and
                # weight decay pulls A -> 0, i.e., U -> I.
                self.ttt_basis_orth = str(ttt_hidden_basis) == "orth"
            h_theta = ttt_hrope_theta if ttt_hrope_theta is not None else rope_theta
            _off = 0.5 if ttt_hrope_interleave else 0.0
            h_inv = ttt_hrope_gain / (h_theta ** ((torch.arange(P_h).float() + _off) / max(P_h, 1)))
            if ttt_hrope_conjpairs:
                # Q25c conjugate-paired ladder: every frequency appears once at +omega
                # and once at -omega (P_h/2 distinct magnitudes). Tying the fast-weight
                # content across a +/- partner pair cancels the antisymmetric (offset-
                # code) term and leaves a pure even cos(omega*dt) recency envelope —
                # an orthogonal basis for keeping the envelope while dropping the
                # Delta-t copy the trained model ignores (F27b).
                assert P_h % 2 == 0, "conjpairs needs an even pair count"
                f_half = h_inv[: P_h // 2]
                h_inv = torch.cat([f_half, -f_half])
            self.ttt_perhead_freqs = ttt_perhead_freqs
            self.ttt_liere = int(ttt_liere)
            if self.ttt_liere > 0:
                # LieRE (ICML 2025) on the hidden address: per-layer, head-shared
                # learnable SKEW generators, one b x b block per contiguous group of
                # b rotated hidden rows (nb = 2*P_h/b blocks cover the same rotated
                # span [0:2*P_h] as the cos/sin ladder). Rotation per token t is
                # matrix_exp(A * t_scaled), orthogonal by construction, so planes
                # AND angles are learned jointly. Parameterized in the house delta
                # form raw = base(buffer) + delta(Parameter, zero-init) so (a) step 0
                # is exactly the chosen init and (b) weight decay pulls the generator
                # back toward its init, not toward the zero (identity) rotation —
                # a plain Parameter would have wd shrink every frequency ~4x over a
                # 3B-token run.
                b = self.ttt_liere
                assert not (ttt_perhead_freqs or ttt_learnable_freqs
                            or ttt_hidden_basis or ttt_hrope_delta_only
                            or ttt_hrope_hnorm != "none"), \
                    "ttt_liere is standalone (no other hidden-rotary mutation)"
                assert (2 * P_h) % b == 0, f"generator block size {b} must divide {2 * P_h}"
                nb = (2 * P_h) // b
                if ttt_liere_init == "rope":
                    # embed the fixed ladder's 2x2 rotations block-diagonally:
                    # skew A[2m, 2m+1] = -theta_p => matrix_exp(A*t) equals the
                    # apply_rotary_cols rotation for pair p = j*(b//2)+m exactly.
                    assert b % 2 == 0, "rope init needs an even generator block size"
                    raw = torch.zeros(nb, b, b)
                    for j in range(nb):
                        for m in range(b // 2):
                            raw[j, 2 * m, 2 * m + 1] = -h_inv[j * (b // 2) + m]
                    self.liere_pos_scale = 1.0
                elif ttt_liere_init == "random":
                    # LieRE convention: skew entries ~ U[0, 2pi] (RoPE-Mixed init),
                    # positions assumed O(1) -> normalize by max_position_embeddings.
                    raw = torch.rand(nb, b, b) * (2.0 * math.pi)
                    self.liere_pos_scale = 1.0 / float(max_position_embeddings)
                else:
                    raise ValueError(f"unknown ttt_liere_init: {ttt_liere_init}")
                self.register_buffer("liere_base", raw, persistent=True)
                self.liere_delta = nn.Parameter(torch.zeros(nb, b, b))
            if ttt_perhead_freqs:
                # AdaRoPE AdaFreq transplant, delta form: theta_h = h_inv *
                # exp(delta_h), delta [num_fw_heads, P_h] ZERO-init — exact
                # reduction to the fixed ladder at step 0 even under bf16
                # (exp(0)=1 exactly), log-space learning per head as in the
                # paper, and weight decay pulls back toward the fixed ladder.
                self.h_freq_delta_perhead = nn.Parameter(
                    torch.zeros(num_lact_heads, h_inv.shape[0]))
                self.register_buffer("h_inv_freq", h_inv, persistent=False)
            elif ttt_learnable_freqs:
                h_inv = h_inv * (1.0 + ttt_freq_tilt * torch.randn(P_h))
                self.h_inv_freq = _freq_param("h_inv_freq", h_inv)
            else:
                self.register_buffer("h_inv_freq", h_inv, persistent=False)
        #### VaPE (Q23): value-path rotary. The update trains f_W toward the
        # ROTATED target R_t v_t (fixed ladder on the first 2*P_v value dims);
        # on apply the (delta) readout is counter-rotated by the query
        # position, so retrieval carries only relative phases
        # sum_t lr_t <h_t, h_s> R_{t-s} v_t. Composes with the input rope
        # (ttt_nope=False); mutually exclusive with the hidden-address
        # rotaries for now.
        self.ttt_value_rope = ttt_value_rope
        self.ttt_vrope_delta_only = ttt_vrope_delta_only
        if ttt_value_rope:
            assert not ttt_hidden_rope and not ttt_liere, \
                "ttt_value_rope is mutually exclusive with ttt_hidden_rope / ttt_liere (for now)"
            P_v = max(1, int(self.d_out * ttt_vrope_frac / 2.0))
            assert 2 * P_v <= self.d_out, f"ttt_vrope_frac too large: {ttt_vrope_frac}"
            v_theta = ttt_vrope_theta if ttt_vrope_theta is not None else rope_theta
            v_inv = ttt_vrope_gain / (
                v_theta ** (torch.arange(P_v).float() / max(P_v, 1)))
            self.register_buffer("v_inv_freq", v_inv, persistent=False)

        #### GbR (Q25b): gate-branch-only input rope. The SwiGLU fast weight
        # has two input branches — silu GATE (w0) and linear CONTENT (w2).
        # "gate": only the q/k copy feeding w0 is rotated (w2 sees the plain,
        # unrotated post-l2norm q/k); "content": the mirror. Purely a routing
        # change: the standard input rotary path still produces the rotated
        # copy, and the branch kernel receives both copies. No new parameters;
        # zero-phase (or same tensors in both slots) reduces bit-exactly to
        # the baseline kernel.
        assert ttt_branch_rope in ("none", "gate", "content"), ttt_branch_rope
        self.ttt_branch_rope = ttt_branch_rope
        if ttt_branch_rope != "none":
            assert not ttt_nope, \
                "ttt_branch_rope needs the input rotary path (ttt_nope=False)"
            assert not (ttt_hidden_rope or ttt_liere or ttt_value_rope), \
                "ttt_branch_rope is mutually exclusive with the hidden/LieRE/value rotaries"
            assert not ttt_learnable_freqs and ttt_input_chunkq == 0, \
                "ttt_branch_rope only splits the standard fla fast_rotary path"

        if ttt_learnable_freqs and ttt_learnable_input_freqs:
            # additive learnable frequency deltas on the fast-weight q/k rotary
            # (composes with the base RoPE: total angle = base + t * dfreq).
            P_qk = self.head_dim // 2
            base_inv = 1.0 / (rope_theta ** (torch.arange(P_qk).float() / P_qk))
            in_tilt = ttt_freq_tilt if ttt_input_freq_tilt is None else ttt_input_freq_tilt
            self.fwqk_dfreq = _freq_param(
                "fwqk_dfreq", in_tilt * torch.randn(P_qk) * base_inv)

        assert self.ttt_loss_type in [
            "dot_product"
        ], f"Loss type {self.ttt_loss_type} not supported"

        #### Q24 training-dynamics interventions: runtime angle scales.
        # PLAIN python attributes (not buffers/Parameters) so setting them never
        # changes state_dict — checkpoints stay interchangeable with standard hpra.
        # _input_rope_scale: None (off) | python float (design b curriculum s(step))
        #   | tensor [b] of 0/1 (design a per-sequence input-rope dropout). Scales
        #   the INPUT-rope ANGLES in the manual (ttt_input_chunkq > 0) path only.
        # _hrope_scale: None | python float; scales the HIDDEN rotary angles
        #   (commitment probe: 0.0 = hidden rotation removed, cos(0)=1/sin(0)=0
        #   is an exact identity).
        self._input_rope_scale = None
        self._hrope_scale = None

    def liere_rotations(self, seq_len, seqlen_offset=0, device=None):
        """LieRE per-token rotation blocks R_t = matrix_exp(A * t * pos_scale),
        t = seqlen_offset .. seqlen_offset + seq_len - 1. Returns [s, nb, b, b] fp32.

        Computed once per forward (the generators change every optimizer step).
        Positions are integers, so exp(A*t) = exp(A)^t exactly: ONE tiny fp64
        matrix_exp gives the unit-step rotation (small generator norm -> no
        scaling-and-squaring error) and _liere_doubling_scan assembles all
        integer powers. fp32 one-shot matrix_exp over all positions violates
        orthogonality by ~3e-3 at t ~ 4e3 (squaring-error accumulation) and
        fp64 one-shot costs ~20 ms per layer per forward; the scan is
        machine-precision orthogonal (~1e-13) and sub-ms, with a cheap
        product-rule backward."""
        raw = (self.liere_base + self.liere_delta).double()
        u = torch.triu(raw, diagonal=1)
        skew = u - u.transpose(-1, -2)  # [nb, b, b], A = U - U^T
        step = torch.matrix_exp(skew * self.liere_pos_scale)  # exp(A * pos_scale)
        if not isinstance(seqlen_offset, int):
            # padding path (per-row offsets): fall back to one-shot fp64 exp
            t = (torch.arange(seq_len, device=device, dtype=torch.float64)
                 + seqlen_offset.double()) * self.liere_pos_scale
            return torch.matrix_exp(
                skew[None] * t[:, None, None, None]).float()
        r = _liere_doubling_scan(step, seq_len + seqlen_offset)
        return r[seqlen_offset:seqlen_offset + seq_len].float()

    def _rescale_qk(self, q, k):
        """
        Args:
            q: [b, s, d]
            k: [b, s, d]
        Returns:
            q: [b, s, d]
            k: [b, s, d]
        """
        qk_scale = self.qk_scale.view(1, 1, -1, 2)
        qk_offset = self.qk_offset.view(1, 1, -1, 2)
        q = q * qk_scale[:, :, :, 0] + qk_offset[:, :, :, 0]
        k = k * qk_scale[:, :, :, 1] + qk_offset[:, :, :, 1]
        return q, k

    def forward(
        self,
        hidden_states: torch.Tensor,  # [b, s, d]
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.size()

        q, k, v = self.qkv(hidden_states).chunk(3, dim=-1)
        #### compute window attention first, then do ttt. ####

        if self.attn_qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        # rescale and reshift the q, k for test-time training layer.
        fast_q, fast_k = self._rescale_qk(q, k)
        fast_v = v

        q = rearrange(q, "... (h d) -> ... h d", d=self.head_dim)
        k = rearrange(k, "... (h d) -> ... h d", d=self.head_dim)
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_dim)

        # WARNING: current implementation ignores cu_seqlens for ttt-layer.
        cu_seqlens = kwargs.get("cu_seqlens", None)

        seqlen_offset, max_seqlen = 0, q_len
        if past_key_values is not None:
            seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            max_seqlen = q.shape[1] + seqlen_offset

            if attention_mask is not None:
                # to deliminate the offsets of padding tokens
                seqlen_offset = (
                    seqlen_offset + attention_mask.sum(-1) - attention_mask.shape[-1]
                )
                max_seqlen = q.shape[1] + max(seqlen_offset)

        if self.max_position_embeddings is not None:
            max_seqlen = max(max_seqlen, self.max_position_embeddings)
        # [b, s, n_h, d]
        q, k = self.rotary(
            q,
            k,
            seqlen_offset=seqlen_offset,
            max_seqlen=max_seqlen,
            cu_seqlens=cu_seqlens,
        )

        if past_key_values is not None:
            cache_has_content = past_key_values.get_seq_length(self.layer_idx) > 0
            k_cached, v_cached = past_key_values.update(
                attn_state=(k.flatten(-2, -1), v.flatten(-2, -1)),
                layer_idx=self.layer_idx,
                offset=q_len,
                cache_kwargs=dict(window_size=self.window_size),
            )["attn_state"]
            if cache_has_content:
                k, v = k_cached, v_cached
                k = rearrange(k, "... (h d) -> ... h d", d=self.head_dim)
                v = rearrange(v, "... (h d) -> ... h d", d=self.head_dim)

        if flash_attn_func is None:
            raise ImportError(
                "Please install Flash Attention via `pip install flash-attn --no-build-isolation` first"
            )

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            q, k, v, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                q, k, v, attention_mask, q_len
            )
            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_q, max_seqlen_k = max_seq_lens
            o = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                causal=True,
                window_size=(
                    (-1, -1) if self.window_size is None else (self.window_size - 1, 0)
                ),
            )
            o = pad_input(o, indices_q, batch_size, q_len)
        elif cu_seqlens is not None:
            o = flash_attn_varlen_func(
                q.squeeze(0),
                k.squeeze(0),
                v.squeeze(0),
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=True,
                window_size=(
                    (-1, -1) if self.window_size is None else (self.window_size - 1, 0)
                ),
            ).unsqueeze(0)
        else:
            o = flash_attn_func(
                q,
                k,
                v,
                causal=True,
                window_size=(
                    (-1, -1) if self.window_size is None else (self.window_size - 1, 0)
                ),
            )
        o = o.reshape(batch_size, q_len, -1)

        ##### TTT starts here.
        # Split heads then merge it to batch dimension
        fast_q = rearrange(fast_q, "b s (n_h d) -> (b n_h) s d", n_h=self.num_fw_heads)
        fast_k = rearrange(fast_k, "b s (n_h d) -> (b n_h) s d", n_h=self.num_fw_heads)
        fast_v = rearrange(fast_v, "b s (n_h d) -> (b n_h) s d", n_h=self.num_fw_heads)

        if self.qkv_silu:
            if self.no_v_silu:
                fast_q = F.silu(fast_q)
                fast_k = F.silu(fast_k)
            else:
                fast_q = F.silu(fast_q)
                fast_k = F.silu(fast_k)
                fast_v = F.silu(fast_v)

        # per head l2 norm for fast_q, fast_k.
        fast_q = l2_norm(fast_q)
        fast_k = l2_norm(fast_k)

        # GbR (Q25b): keep PRE-rotary copies of the post-l2norm fast q/k for
        # the unrotated branch. Cloned because the rotary below may write
        # in-place through views of this storage. Shape is already
        # [(b n_h), s, d] — the rotary block's rearranges round-trip back to
        # exactly this layout, so the copies align with the rotated outputs.
        if getattr(self, "ttt_branch_rope", "none") != "none":
            plain_q = fast_q.clone()
            plain_k = fast_k.clone()

        if not self.ttt_nope:
            #### Apply rotary embedding.  Here we use the same rope as the attention layer.
            # I observed that using NoPE for ttt (No positional encoding) here also works.
            fast_q = rearrange(
                fast_q, "(b n_h) s d -> b s (n_h d)", n_h=self.num_fw_heads
            )
            fast_k = rearrange(
                fast_k, "(b n_h) s d -> b s (n_h d)", n_h=self.num_fw_heads
            )

            fast_q = rearrange(fast_q, "b s (n_h d) -> b s n_h d", n_h=self.num_heads)
            fast_k = rearrange(fast_k, "b s (n_h d) -> b s n_h d", n_h=self.num_heads)

            if self.ttt_input_chunkq > 0:
                # manual NeoX-style rotary at chunk-quantized positions
                # (same inv_freq as fast_rotary; C=1 reproduces the per-token
                # rotary through this code path for a clean surgery baseline)
                Cq = float(self.ttt_input_chunkq)
                pos_in = torch.arange(
                    fast_q.shape[1], device=fast_q.device, dtype=torch.float32
                ) + seqlen_offset
                pos_in = torch.floor(pos_in / Cq) * Cq
                ang_in = pos_in[:, None] * self.fast_rotary.inv_freq.float()[None]  # [s, P]
                # Q24: optional runtime angle scale (see __init__). float scales
                # globally (curriculum); tensor [b] scales per sequence (input-rope
                # dropout: 0 -> unrotated q/k, exactly identity). fast_q here is
                # [b, s, n_h, d] with b = TRUE batch size (rearranged back from
                # (b n_h) above), so a [b] scale broadcasts per batch row.
                _irs = getattr(self, "_input_rope_scale", None)
                if _irs is not None and not isinstance(_irs, torch.Tensor):
                    ang_in = ang_in * float(_irs)
                    _irs = None
                if _irs is not None:
                    assert _irs.numel() == fast_q.shape[0], \
                        f"_input_rope_scale has {_irs.numel()} entries, batch is {fast_q.shape[0]}"
                    ang_in = ang_in[None] * _irs.float().view(-1, 1, 1).to(ang_in.device)  # [b, s, P]
                    qcos = torch.cat([ang_in.cos(), ang_in.cos()], dim=-1)[:, :, None, :]  # [b, s, 1, d]
                    qsin = torch.cat([ang_in.sin(), ang_in.sin()], dim=-1)[:, :, None, :]
                else:
                    qcos = torch.cat([ang_in.cos(), ang_in.cos()], dim=-1)[None, :, None, :]
                    qsin = torch.cat([ang_in.sin(), ang_in.sin()], dim=-1)[None, :, None, :]

                def _rot_half_q(x):
                    x1, x2 = x.chunk(2, dim=-1)
                    return torch.cat([-x2, x1], dim=-1)

                fast_q = (fast_q * qcos + _rot_half_q(fast_q) * qsin).type_as(fast_q)
                fast_k = (fast_k * qcos + _rot_half_q(fast_k) * qsin).type_as(fast_k)
            else:
                fast_q, fast_k = self.fast_rotary(
                    fast_q,
                    fast_k,
                    seqlen_offset=seqlen_offset,
                    max_seqlen=max_seqlen,
                    cu_seqlens=cu_seqlens,
                )

            if self.ttt_learnable_freqs and hasattr(self, "fwqk_dfreq"):
                # omega_map (1D): extra rotary with learnable frequency deltas,
                # rotate_half convention to match the base RoPE pairing.
                pos = torch.arange(
                    fast_q.shape[1], device=fast_q.device, dtype=torch.float32
                ) + seqlen_offset
                ang = pos[:, None] * self.fwqk_dfreq.float()[None]  # [s, P_qk]
                dcos = torch.cat([ang.cos(), ang.cos()], dim=-1)[None, :, None, :]
                dsin = torch.cat([ang.sin(), ang.sin()], dim=-1)[None, :, None, :]

                def _rot_half(x):
                    x1, x2 = x.chunk(2, dim=-1)
                    return torch.cat([-x2, x1], dim=-1)

                fast_q = (fast_q * dcos + _rot_half(fast_q) * dsin).type_as(fast_q)
                fast_k = (fast_k * dcos + _rot_half(fast_k) * dsin).type_as(fast_k)

            fast_q = rearrange(fast_q, "b s n_h d -> b s (n_h d)", n_h=self.num_heads)
            fast_k = rearrange(fast_k, "b s n_h d -> b s (n_h d)", n_h=self.num_heads)

            fast_q = rearrange(
                fast_q, "b s (n_h d) -> (b n_h) s d", n_h=self.num_fw_heads
            )
            fast_k = rearrange(
                fast_k, "b s (n_h d) -> (b n_h) s d", n_h=self.num_fw_heads
            )
            #### RoPE done. ####

        if self.w0_w2_low_rank > 0:
            fw_w0 = self.w0().repeat(batch_size, 1, 1)
            fw_w2 = self.w2().repeat(batch_size, 1, 1)
        else:
            fw_w0 = self.w0.repeat(
                batch_size, 1, 1
            )  # [nh, d_h, d_in] -> [b*nh, d_h, d_in]
            fw_w2 = self.w2.repeat(
                batch_size, 1, 1
            )  # [nh, d_h, d_in] -> [b*nh, d_h, d_in]

        fw_w1 = self.w1.repeat(
            batch_size, 1, 1
        )  # [nh, d_out, d_h] -> [b*nh, d_out, d_h]

        lr = self.lr_proj(hidden_states)  # [b, s, num_heads * lr_dim_per_head]
        if self.lr_parameterization == "mamba":
            lr = torch.nn.functional.softplus(lr.float() + self.base_lr_inv)
        else:
            raise NotImplementedError(
                f"LR parameterization {self.lr_parameterization} not implemented"
            )
        fw_lr = rearrange(
            lr, "b s (n_h lr_dim) -> (b n_h) s lr_dim", n_h=self.num_fw_heads
        )
        fw_lr1, fw_lr2, fw_lr3 = fw_lr.chunk(3, dim=-1)

        if self.use_momentum:
            momentum = self.momentum_proj(hidden_states).float()  # [b, s, nh]
            momentum = rearrange(
                momentum, "b s (n_h d) -> (b n_h) s d", n_h=self.num_fw_heads
            )
        else:
            momentum = None

        if self.fp32_states:
            # here we cast the fast weights to fp32, but all matmuls are still in bf16
            # only fast weight updates are in fp32.  This is similar to bf16 training of slow weights.
            fw_w0 = fw_w0.to(torch.float32)
            fw_w1 = fw_w1.to(torch.float32)
            fw_w2 = fw_w2.to(torch.float32)

        # [b * nh, s, d_ttt_head]
        if getattr(self, "ttt_branch_rope", "none") != "none":
            # GbR path (prenorm, PyTorch ops only): fast_q/fast_k are the
            # ROTATED copies (standard input rotary above); plain_q/plain_k
            # the unrotated post-l2norm copies. "gate" routes the rotated
            # pair to the silu gate branch (w0), "content" to the linear
            # content branch (w2); the other branch sees the plain pair.
            assert self.ttt_prenorm and not self.use_fused_kernel, \
                "ttt_branch_rope requires ttt_prenorm=True, use_fused_kernel=False"
            if self.ttt_branch_rope == "gate":
                qg, kg, qc, kc = fast_q, fast_k, plain_q, plain_k
            else:  # "content"
                qg, kg, qc, kc = plain_q, plain_k, fast_q, fast_k
            fw_x = prenorm_block_causal_lact_swiglu_branch_rope(
                fw_w0, fw_w1, fw_w2,
                qg, kg, qc, kc, fast_v,
                fw_lr1, fw_lr2, fw_lr3,
                chunk_size=self.lact_chunk_size,
                use_muon=self.use_muon,
                momentum=momentum,
            )
        elif self.ttt_hidden_rope:
            # h-PRA path (prenorm, PyTorch ops only)
            assert self.ttt_prenorm and not self.use_fused_kernel, \
                "ttt_hidden_rope requires ttt_prenorm=True, use_fused_kernel=False"
            pos = torch.arange(
                fast_q.shape[1], device=fast_q.device, dtype=torch.float32
            ) + seqlen_offset
            if self.ttt_hrope_chunkq > 0:
                Ch = float(self.ttt_hrope_chunkq)
                pos = torch.floor(pos / Ch) * Ch
            # Q24 commitment probe: scale the hidden rotary ANGLES (h_ang =
            # h_inv_freq * pos, so scaling pos after the chunkq floor is
            # equivalent). 0.0 -> cos=1/sin=0 -> exact identity rotation in
            # the hidden kernel (forward AND inner-loop inverse rotation).
            _hrs = getattr(self, "_hrope_scale", None)
            if _hrs is not None:
                assert getattr(self, "ttt_liere", 0) == 0, \
                    "_hrope_scale not supported with ttt_liere (angles live in matrix_exp)"
                pos = pos * float(_hrs)
            r_blocks = None
            if getattr(self, "ttt_liere", 0) > 0:
                hcos_t = hsin_t = None
                with torch.autocast(device_type="cuda", enabled=False):
                    r_blocks = self.liere_rotations(
                        fast_q.shape[1], seqlen_offset, device=fast_q.device)
                    # kernel layout: [nb, b, b, s] so chunk slicing is [..., s:e]
                    r_blocks = r_blocks.permute(1, 2, 3, 0).contiguous()
            elif getattr(self, "ttt_perhead_freqs", False):
                theta_ph = (self.h_inv_freq.float()[None]
                            * self.h_freq_delta_perhead.float().exp())  # [nh, P_h]
                h_ang = theta_ph[:, :, None] * pos[None, None, :]  # [nh, P_h, s]
                b_true = fast_q.shape[0] // self.num_fw_heads
                h_ang = h_ang.repeat(b_true, 1, 1)  # [b*nh, P_h, s]
                hcos_t, hsin_t = h_ang.cos(), h_ang.sin()
            else:
                h_ang = self.h_inv_freq.float()[:, None] * pos[None, :]  # [P_h, s]
                hcos_t, hsin_t = h_ang.cos(), h_ang.sin()
            fw_x = prenorm_block_causal_lact_swiglu_hidden_rope(
                fw_w0, fw_w1, fw_w2,
                fast_q, fast_k, fast_v,
                fw_lr1, fw_lr2, fw_lr3,
                hcos_t, hsin_t,
                chunk_size=self.lact_chunk_size,
                use_muon=self.use_muon,
                momentum=momentum,
                delta_only=self.ttt_hrope_delta_only,
                hnorm=self.ttt_hrope_hnorm,
                h_basis=(
                    torch.matrix_exp(
                        (self.h_basis_delta - self.h_basis_delta.transpose(0, 1)).float()
                    ).to(self.h_basis_delta.dtype)
                    if getattr(self, "ttt_basis_orth", False)
                    else self.h_basis_eye + self.h_basis_delta)
                if getattr(self, "ttt_hidden_basis", False) else None,
                r_blocks=r_blocks,
            )
        elif self.ttt_value_rope:
            # VaPE path (prenorm, PyTorch ops only)
            assert self.ttt_prenorm and not self.use_fused_kernel, \
                "ttt_value_rope requires ttt_prenorm=True, use_fused_kernel=False"
            pos = torch.arange(
                fast_q.shape[1], device=fast_q.device, dtype=torch.float32
            ) + seqlen_offset
            v_ang = self.v_inv_freq.float()[:, None] * pos[None, :]  # [P_v, s]
            fw_x = prenorm_block_causal_lact_swiglu_value_rope(
                fw_w0, fw_w1, fw_w2,
                fast_q, fast_k, fast_v,
                fw_lr1, fw_lr2, fw_lr3,
                v_ang.cos(), v_ang.sin(),
                chunk_size=self.lact_chunk_size,
                use_muon=self.use_muon,
                momentum=momentum,
                delta_only=self.ttt_vrope_delta_only,
            )
        elif self.ttt_prenorm:
            # pre-norm version of ttt.   state = state + f(norm(state))
            if self.use_fused_kernel:
                fw_x = prenorm_block_causal_lact_swiglu_fused_kernel_triton(
                    fw_w0,
                    fw_w1,
                    fw_w2,
                    fast_q,
                    fast_k,
                    fast_v,
                    fw_lr1,
                    fw_lr2,
                    fw_lr3,
                    chunk_size=self.lact_chunk_size,
                    use_muon=self.use_muon,
                    momentum=momentum,
                )
            else:
                fw_x = prenorm_block_causal_lact_swiglu(
                    fw_w0,
                    fw_w1,
                    fw_w2,
                    fast_q,
                    fast_k,
                    fast_v,
                    fw_lr1,
                    fw_lr2,
                    fw_lr3,
                    chunk_size=self.lact_chunk_size,
                    use_muon=self.use_muon,
                    momentum=momentum,
                    precess_cos=getattr(self, "precess_cos", None),
                    precess_sin=getattr(self, "precess_sin", None),
                )
        else:
            # post-norm version of ttt.   state = norm(state + f(state))
            if self.use_fused_kernel:
                fw_x = postnorm_block_causal_lact_swiglu_fused_kernel_triton(
                    fw_w0,
                    fw_w1,
                    fw_w2,
                    fast_q,
                    fast_k,
                    fast_v,
                    fw_lr1,
                    fw_lr2,
                    fw_lr3,
                    chunk_size=self.lact_chunk_size,
                    use_muon=self.use_muon,
                    momentum=momentum,
                )
            else:
                fw_x = block_causal_lact_swiglu(
                    fw_w0,
                    fw_w1,
                    fw_w2,
                    fast_q,
                    fast_k,
                    fast_v,
                    fw_lr1,
                    fw_lr2,
                    fw_lr3,
                    chunk_size=self.lact_chunk_size,
                    use_muon=self.use_muon,
                    momentum=momentum,
                )

        # per-head output norm for ttt layer.
        ttt_x_normed = self.ttt_norm(fw_x)
        if self.learnable_ttt_scale:
            ttt_scale = F.silu(self.ttt_scale_proj(hidden_states), inplace=False)
            ttt_scale = rearrange(
                ttt_scale, "b s (n_h d) -> (b n_h) s d", n_h=self.num_fw_heads
            )
            ttt_x_normed = ttt_x_normed * ttt_scale

        ttt_x_normed = rearrange(
            ttt_x_normed, "(b n_h) s d -> b s (n_h d)", n_h=self.num_fw_heads
        )

        o = o + ttt_x_normed
        o = self.o_proj(o)

        if not output_attentions:
            attentions = None

        return o, attentions, past_key_values

    def _upad_input(self, q, k, v, attention_mask, q_len):
        batch_size, seq_len, num_key_value_heads, head_dim = k.shape
        cache_mask = attention_mask[:, -seq_len:]
        seqlens = cache_mask.sum(-1, dtype=torch.int32)
        indices_k = torch.nonzero(cache_mask.flatten(), as_tuple=False).flatten()
        max_seqlen_k = seqlens.max().item()
        cu_seqlens_k = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))

        k = index_first_axis(
            k.reshape(batch_size * seq_len, num_key_value_heads, head_dim), indices_k
        )
        v = index_first_axis(
            v.reshape(batch_size * seq_len, num_key_value_heads, head_dim), indices_k
        )
        if q_len == seq_len:
            q = index_first_axis(
                q.reshape(batch_size * seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_q = max_seqlen_k
            indices_q = indices_k
        elif q_len == 1:
            max_seqlen_q = 1
            # There is a memcpy here, that is very bad.
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=q.device
            )
            indices_q = cu_seqlens_q[:-1]
            q = q.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -q_len:]
            q, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(q, attention_mask)

        return (
            q,
            k,
            v,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_q, max_seqlen_k),
        )
