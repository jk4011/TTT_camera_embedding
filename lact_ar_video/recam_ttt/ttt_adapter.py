"""Q11: ReCamMaster-frozen + TTT-adapter.

Starts from the released ReCamMaster-Wan2.1 checkpoint, FREEZES every
pretrained weight, and replaces ReCamMaster's newly-introduced mechanism
(cross-video concat attention) with:
  1. per-video attention (frozen weights): the block's self_attn runs on each
     of the two videos separately with the single-video rope table, so no
     token ever attends across videos;
  2. a trainable TTT fast-weight branch that carries the cross-video
     information: LaCT SwiGLU fast weights are UPDATEd on the clean SRC half
     in sequential chunks of ttt_chunk_frames latent frames (default 3 = 7
     chunks; Stage 1 uses 1 = 21 chunks) and then APPLYed to the TGT half.
     Stage 1 additions: inter_multi 4 (2x fast-weight capacity), Muon (NS5)
     on the chunk updates, and train_recam_modules (each block's
     cam_encoder + projector — ReCamMaster's own additions, not Wan weights
     — train at a lower lr alongside the branches).

PRA (camera rotary) sites, FIXED frequency ladders only (no learnable gains):
  - input site: Plucker rotary on the fast q/k after l2-norm;
  - hidden site: Plucker rotary on the SwiGLU hidden pre-w1 (exact
    manual-backward Jacobian in the update path).

Token layout everywhere: [TGT f*h*w tokens || SRC f*h*w tokens] — TARGET
FIRST, matching ReCamMaster's latents = cat([target, source], frame dim).

Terminology follows the LaCT paper: "update" = fast-weight gradient step,
"apply" = using the fast weights on queries.

Dtype contract: the frozen DiT is bf16; branch params are fp32 and the DiT
forward must run under torch.autocast("cuda", torch.bfloat16) (matmuls in
bf16, pose/phase math fp32).
"""
import math
import os
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# minVid provides the PRA phase builders (import-only; do not modify)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
from minVid.models.blocks.cam_phase_builder import (  # noqa: E402
    cam_phase_tables,
    make_cam_ladder,
    plucker_per_token,
)

# ReCamMaster clone (read-only, on PYTHONPATH)
from diffsynth.models.wan_video_dit import RMSNorm, modulate  # noqa: E402

# Fixed ReCamMaster/MCV geometry (their DiTBlock hardcodes 30 x 52 too)
LATENT_F = 21          # latent frames per video
TOK_H, TOK_W = 30, 52  # token grid per frame
HW = TOK_H * TOK_W     # 1560 tokens per latent frame
# NOTE: the SRC chunk size is per-branch config now (ttt_chunk_frames in the
# variant yaml; default 3 = the original F31 recipe, Stage 1 uses 1). These
# module constants only give the default.
DEFAULT_CHUNK_FRAMES = 3   # SRC chunk = 3 latent frames (F31 recipe)
CHUNK_FRAMES = DEFAULT_CHUNK_FRAMES              # kept for old callers
CHUNK_TOKENS = DEFAULT_CHUNK_FRAMES * HW         # 4680, kept for old callers
SEQ_PER_VIDEO = LATENT_F * HW     # 32760
SEQ_TOTAL = 2 * SEQ_PER_VIDEO     # 65520


# ---------------------------------------------------------------------------
# small numerics helpers (local, un-compiled copies of
# minVid/models/blocks/functions.py — no torch.compile so no triton cache)
# ---------------------------------------------------------------------------

def silu_backprop(dy: torch.Tensor, x: torch.Tensor):
    """dy: grad wrt silu(x); returns grad wrt x."""
    sigma = torch.sigmoid(x)
    return dy * sigma * (1 + x * (1 - sigma))


def l2_norm(x: torch.Tensor):
    """x: [b, l, d]; L2-normalize the last dim (norm upcasts to fp32)."""
    x_type = x.dtype
    ret = x / (x.norm(dim=-1, keepdim=True) + 1e-5)
    return ret.type(x_type)


def inv_softplus(x):
    if isinstance(x, torch.Tensor):
        return x + torch.log(-torch.expm1(-x))
    return x + math.log(-math.expm1(-x))


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
    """Newton-Schulz orthogonalization of a batch of matrices (Muon).

    LOCAL UNCOMPILED copy of lact_nvs/lact_ttt.py::zeropower_via_newtonschulz5
    (== minVid/models/blocks/functions.py; both carry @torch.compile which we
    must not use on this noexec-/tmp node). Semantics identical to both
    references: cast to bf16, transpose tall matrices so rows <= cols,
    normalize by the Frobenius norm (spectral norm <= 1), run `steps` quintic
    NS iterations, transpose back. Returns ~US'V^T (S'_ii ~ U(0.5, 1.5)).

    G: [b, m, n]; returns [b, m, n] in bf16.
    """
    assert len(G.shape) == 3
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(1, 2), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.transpose(1, 2)
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    return X


def inv_silu(y: float) -> float:
    """Solve silu(b) = y by Newton iteration (used for the ttt_scale bias)."""
    b = torch.tensor(float(y))
    yt = torch.tensor(float(y))
    for _ in range(100):
        s = torch.sigmoid(b)
        f = b * s - yt
        fp = s * (1 + b * (1 - s))
        b = b - f / fp
    return float(b)


# ---------------------------------------------------------------------------
# rotary helpers (ports of minVid/models/blocks/ar_lact_swa_repeat.py)
# ---------------------------------------------------------------------------

def apply_rotary_pairs(x, coeff_cos, coeff_sin):
    """Rotate adjacent feature pairs of row-major tokens.

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
    return torch.cat([y.to(x.dtype), x[..., 2 * P:]], dim=-1)


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
    return torch.cat([y, x[:, 2 * P:, :]], dim=1)


# ---------------------------------------------------------------------------
# fast-weight kernel for the [TGT || SRC] recam layout
# (port of ar_fast_weight_swish_glu_weight_norm_mini_batch[_hidden_rope]'s
#  src-prefix loop; muon off, num_repeat 1, no AR interleave)
# ---------------------------------------------------------------------------

def recam_fast_weight_update_apply(
    w0: torch.Tensor,   # [b, d_h, d_in]
    w1: torch.Tensor,   # [b, d_out, d_h]
    w2: torch.Tensor,   # [b, d_h, d_in]
    q: torch.Tensor,    # [b, L, d_in]
    k: torch.Tensor,
    v: torch.Tensor,
    lr0: torch.Tensor,  # [b, L, 1] fp32 (per-token lr for the w0 update)
    lr1: torch.Tensor,
    lr2: torch.Tensor,
    w_scale: float,
    tgt_len: int,
    chunk_size: int,
    hcos: torch.Tensor = None,  # [P, L] hidden-rotary tables (h-PRA)
    hsin: torch.Tensor = None,
    weight_norm: bool = True,
    use_muon: bool = False,
    num_muon_iters: int = 5,
):
    """SwiGLU fast weights f(x) = w1 @ (silu(w0 x) * (w2 x)).

    Token layout: [0, tgt_len) = TGT tokens, [tgt_len, L) = clean SRC tokens.
    The SRC half is consumed in chunk_size chunks IN ORDER; each chunk first
    UPDATEs the fast weights (manual SwiGLU backward; with h-PRA the hidden of
    the keys is rotated before meeting w1 and the backward applies the inverse
    rotation — the exact Jacobian), then the same chunk is APPLYed with the
    post-update weights. Finally the whole TGT half is APPLYed with the final
    weights. With hcos=1/hsin=0 (or None) this reduces exactly to the plain
    LaCT update/apply.

    use_muon: orthogonalize each chunk's dw0/dw1/dw2 with Newton-Schulz
    (Muon) before adding. Reference order (minVid ar_lact_swa_repeat.py and
    lact_nvs lact_ttt.py agree): the raw chunk gradient — with the per-token
    lr and w_scale already INSIDE the outer product — is passed through NS5
    (bf16 internally), then added, then weight-norm re-normalizes. The h-PRA
    manual backward (inverse rotation on dhidden) is unchanged; only the
    final dw matrices are orthogonalized. With use_muon=False the kernel is
    bit-identical to the pre-Stage-1 version.
    """
    L = k.shape[1]
    assert (L - tgt_len) % chunk_size == 0, (L, tgt_len, chunk_size)
    use_hrope = hcos is not None

    w0_norm = w0.norm(dim=2, keepdim=True)
    w1_norm = w1.norm(dim=2, keepdim=True)
    w2_norm = w2.norm(dim=2, keepdim=True)

    output = torch.zeros_like(q)

    def _apply(qi, hc, hs):
        h = torch.bmm(w2, qi.transpose(1, 2))
        gate = F.silu(torch.bmm(w0, qi.transpose(1, 2)), inplace=True)
        hq = gate * h
        if use_hrope:
            hq = apply_rotary_cols(hq, hc, hs)
        return torch.bmm(w1, hq).transpose(1, 2)

    for s_index in range(tgt_len, L, chunk_size):
        e_index = s_index + chunk_size

        ki, vi = k[:, s_index:e_index, :], v[:, s_index:e_index, :]
        lr0i = lr0[:, s_index:e_index, :]
        lr1i = lr1[:, s_index:e_index, :]
        lr2i = lr2[:, s_index:e_index, :]
        if use_hrope:
            hci = hcos[:, s_index:e_index]
            hsi = hsin[:, s_index:e_index]
        else:
            hci = hsi = None

        # ---- UPDATE on the SRC chunk (manual SwiGLU backward) ----
        gate_before_act = torch.bmm(w0, ki.transpose(1, 2))
        hidden_before_mul = torch.bmm(w2, ki.transpose(1, 2))
        silu_gate = F.silu(gate_before_act, inplace=False)
        hidden = silu_gate * hidden_before_mul

        if use_hrope:
            # keys' hidden addresses are stored rotated; the backward through
            # the rotation is the inverse rotation R^T (negated sin)
            hidden_key = apply_rotary_cols(hidden, hci, hsi)
            dhidden = apply_rotary_cols(
                torch.bmm(w1.transpose(1, 2), vi.transpose(1, 2)), hci, -hsi)
        else:
            hidden_key = hidden
            dhidden = torch.bmm(w1.transpose(1, 2), vi.transpose(1, 2))

        dhidden_before_mul = dhidden * silu_gate
        dgate = dhidden * hidden_before_mul
        dgate_before_act = silu_backprop(dgate, gate_before_act)

        dw1 = torch.bmm(vi.transpose(1, 2),
                        hidden_key.transpose(1, 2) * lr1i * w_scale)
        dw0 = torch.bmm(dgate_before_act, ki * lr0i * w_scale)
        dw2 = torch.bmm(dhidden_before_mul, ki * lr2i * w_scale)

        if use_muon:
            dw1 = zeropower_via_newtonschulz5(dw1, num_muon_iters)
            dw0 = zeropower_via_newtonschulz5(dw0, num_muon_iters)
            dw2 = zeropower_via_newtonschulz5(dw2, num_muon_iters)

        w1 = w1 + dw1
        w0 = w0 + dw0
        w2 = w2 + dw2
        if weight_norm:
            w0 = w0 / (w0.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
            w1 = w1 / (w1.norm(dim=2, keepdim=True) + 1e-5) * w1_norm
            w2 = w2 / (w2.norm(dim=2, keepdim=True) + 1e-5) * w2_norm

        # ---- APPLY on the same chunk with the post-update weights ----
        output[:, s_index:e_index, :] = _apply(q[:, s_index:e_index, :], hci, hsi)

    # ---- APPLY on the whole TGT half with the final weights ----
    if use_hrope:
        output[:, :tgt_len, :] = _apply(
            q[:, :tgt_len, :], hcos[:, :tgt_len], hsin[:, :tgt_len])
    else:
        output[:, :tgt_len, :] = _apply(q[:, :tgt_len, :], None, None)

    return output, w0, w1, w2


# ---------------------------------------------------------------------------
# the trainable TTT branch
# ---------------------------------------------------------------------------

class RecamTTTBranch(nn.Module):
    """Trainable LaCT fast-weight branch for one (frozen) ReCamMaster block.

    Structure ported from minVid ARFastWeightSwiGLU with the ccv settings:
    q/k/v Linear copies of the block's self_attn (+ norm_q/norm_k RMSNorm
    copies), qk_l2_norm, qkv_silu off, fw_head_dim 768 (2 fast-weight heads),
    inter_multi 2 (Stage 1: 4), mamba lr (lr_dim 1), muon off (Stage 1: on,
    NS5 reference semantics), weight_norm on, o_norm on (fresh RMSNorm),
    learnable scalar ttt_scale, w_init "clean".

    Init deviation from minVid (documented): the output projection `o` is
    ZERO-initialized (spec: step-0 forward bit-identical to the per-video
    frozen model) and therefore ttt_scale_proj gets a NONZERO bias
    (silu(bias)=1) instead of minVid's all-zero init — with both zero the
    branch would receive exactly zero gradient forever.
    """

    def __init__(self, dim, self_attn, ttt_input_rope=False,
                 ttt_hidden_rope=False, fw_head_dim=768, inter_multi=2,
                 lr_dim=1, base_lr=0.001, hrope_frac=0.5, eps=1e-6,
                 ttt_chunk_frames=DEFAULT_CHUNK_FRAMES, use_muon=False,
                 num_muon_iters=5):
        super().__init__()
        assert dim % fw_head_dim == 0
        self.dim = dim
        self.fw_head_dim = fw_head_dim
        self.num_fw_heads = dim // fw_head_dim
        self.ttt_input_rope = ttt_input_rope
        self.ttt_hidden_rope = ttt_hidden_rope
        self.w_scale = 1.0
        self.weight_norm = True
        # SRC update granularity: chunk = ttt_chunk_frames latent frames
        assert LATENT_F % int(ttt_chunk_frames) == 0, ttt_chunk_frames
        self.chunk_tokens = int(ttt_chunk_frames) * HW
        # Muon on the chunk updates (NS5 on dw0/dw1/dw2 before adding)
        self.use_muon = bool(use_muon)
        self.num_muon_iters = int(num_muon_iters)

        # q/k/v (+ qk norms) initialized as COPIES of the frozen
        # ReCamMaster-trained self_attn, then trained in fp32
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        with torch.no_grad():
            for mine, theirs in ((self.q, self_attn.q), (self.k, self_attn.k),
                                 (self.v, self_attn.v)):
                mine.weight.copy_(theirs.weight)
                mine.bias.copy_(theirs.bias)
            self.norm_q.weight.copy_(self_attn.norm_q.weight)
            self.norm_k.weight.copy_(self_attn.norm_k.weight)

        # fast-weight scale/offset on q/k (identity init, as in minVid)
        self.qk_scale = nn.Parameter(torch.ones(dim, 2))
        self.qk_offset = nn.Parameter(torch.zeros(dim, 2))

        # SwiGLU fast weights, w_init="clean"
        d_in = fw_head_dim
        d_h = int(fw_head_dim * inter_multi)
        d_out = fw_head_dim
        self.d_h = d_h
        self.w0 = nn.Parameter(
            torch.randn(self.num_fw_heads, d_h, d_in) / math.sqrt(d_in))
        self.w1 = nn.Parameter(
            torch.randn(self.num_fw_heads, d_out, d_h) / math.sqrt(d_h))
        self.w2 = nn.Parameter(
            torch.randn(self.num_fw_heads, d_h, d_in) / math.sqrt(d_in))

        # per-token learning rates (mamba parameterization)
        self.lr_dim = int(lr_dim * self.num_fw_heads * 3)
        self.lr_proj = nn.Linear(dim, self.lr_dim)
        self.base_lr_inv = inv_softplus(base_lr)

        self.output_norm = RMSNorm(fw_head_dim, eps=eps)

        # learnable scalar ttt scale: silu(proj(x)); weight 0, bias silu^-1(1)
        self.ttt_scale_proj = nn.Linear(dim, 1)
        nn.init.zeros_(self.ttt_scale_proj.weight)
        with torch.no_grad():
            self.ttt_scale_proj.bias.fill_(inv_silu(1.0))

        # zero-init output projection: step-0 forward is bit-identical to the
        # per-video frozen model
        self.o = nn.Linear(dim, dim)
        nn.init.zeros_(self.o.weight)
        nn.init.zeros_(self.o.bias)

        # ---- PRA rotary sites, FIXED ladders (buffers, never learnable) ----
        if ttt_input_rope:
            # 6 Plucker coords x nf_in freqs, leaving >= 12 dims untouched
            nf_in = (fw_head_dim - 2 * 6) // (2 * 6)  # 768 -> 63
            self.cam_num_freqs_in = nf_in
            self.register_buffer("cam_omega_in", make_cam_ladder(nf_in),
                                 persistent=False)
            self.register_buffer("cam_gain_in", torch.ones(6, nf_in),
                                 persistent=False)
        if ttt_hidden_rope:
            # rotate hrope_frac of the SwiGLU hidden; all dims scale with
            # d_h = fw_head_dim * inter_multi (no hardcoded sizes):
            #   inter_multi 2: d_h 1536 -> 768 rotated dims, nf_h 64
            #   inter_multi 4: d_h 3072 -> 1536 rotated dims, nf_h 128
            P_h = max(1, int(d_h * hrope_frac / 2.0))
            self.h_rope_dim = 2 * P_h
            nf_h = (self.h_rope_dim // 2) // 6
            self.cam_num_freqs_h = nf_h
            self.register_buffer("cam_omega_h", make_cam_ladder(nf_h),
                                 persistent=False)
            self.register_buffer("cam_gain_h", torch.ones(6, nf_h),
                                 persistent=False)

    def forward(self, x, ctx):
        """x: [1, L, dim] = the same input_x the frozen attention sees
        (post norm1/modulate + frozen cam_encoder addition), layout
        [TGT fhw || SRC fhw]. ctx: shared per-step camera/phase context."""
        b, s, _ = x.shape
        assert b == 1, "recam_ttt assumes batch size 1"
        assert s % (2 * HW) == 0, s
        tgt_len = s // 2
        assert tgt_len % self.chunk_tokens == 0, (tgt_len, self.chunk_tokens)

        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = q * self.qk_scale[:, 0] + self.qk_offset[:, 0]
        k = k * self.qk_scale[:, 1] + self.qk_offset[:, 1]

        fast_q = rearrange(q, 'b s (h d) -> (b h) s d', h=self.num_fw_heads)
        fast_k = rearrange(k, 'b s (h d) -> (b h) s d', h=self.num_fw_heads)
        fast_v = rearrange(v, 'b s (h d) -> (b h) s d', h=self.num_fw_heads)

        fast_q = l2_norm(fast_q)
        fast_k = l2_norm(fast_k)

        # PRA input site: camera rotary on fast q/k after l2 norm
        if self.ttt_input_rope:
            in_cos, in_sin = ctx["in_cos"], ctx["in_sin"]  # [L, 6*nf_in] fp32
            assert in_cos.shape[0] == s
            fast_q = apply_rotary_pairs(fast_q, in_cos, in_sin)
            fast_k = apply_rotary_pairs(fast_k, in_cos, in_sin)

        lr = self.lr_proj(x)
        lr = F.softplus(lr.float() + self.base_lr_inv)
        fw_lr = rearrange(lr, 'b s (h l) -> (b h) s l', h=self.num_fw_heads)
        lr0, lr1, lr2 = fw_lr.chunk(3, dim=-1)

        w0 = self.w0.repeat(b, 1, 1)
        w1 = self.w1.repeat(b, 1, 1)
        w2 = self.w2.repeat(b, 1, 1)

        if self.ttt_hidden_rope:
            hcos, hsin = ctx["h_cos"], ctx["h_sin"]  # [P, L] fp32
            assert hcos.shape[1] == s
        else:
            hcos = hsin = None

        fw_x, _, _, _ = recam_fast_weight_update_apply(
            w0, w1, w2, fast_q, fast_k, fast_v, lr0, lr1, lr2,
            w_scale=self.w_scale, tgt_len=tgt_len,
            chunk_size=self.chunk_tokens,
            hcos=hcos, hsin=hsin, weight_norm=self.weight_norm,
            use_muon=self.use_muon, num_muon_iters=self.num_muon_iters,
        )

        ttt_x = self.output_norm(fw_x)
        ttt_x = rearrange(ttt_x, '(b h) s d -> b s (h d)', h=self.num_fw_heads)
        ttt_scale = F.silu(self.ttt_scale_proj(x), inplace=True)
        return self.o(ttt_x * ttt_scale)


# ---------------------------------------------------------------------------
# per-step camera/phase context
# ---------------------------------------------------------------------------

@torch.no_grad()
def set_ctx_phases(ctx, coords6, ref_branch):
    """Precompute the (cos, sin) phase tables ONCE per step (shared by all 30
    blocks; the ladders are fixed buffers identical across blocks).

    coords6: [L_total, 6] fp32 per-token Plucker coordinates, or None for the
    base variant (no rotary)."""
    ctx.pop("in_cos", None), ctx.pop("in_sin", None)
    ctx.pop("h_cos", None), ctx.pop("h_sin", None)
    if coords6 is None:
        return
    with torch.autocast(device_type="cuda", enabled=False):
        coords6 = coords6.float()
        if ref_branch.ttt_input_rope:
            c, s = cam_phase_tables(coords6, ref_branch.cam_omega_in,
                                    ref_branch.cam_gain_in)
            ctx["in_cos"], ctx["in_sin"] = c, s
        if ref_branch.ttt_hidden_rope:
            c, s = cam_phase_tables(coords6, ref_branch.cam_omega_h,
                                    ref_branch.cam_gain_h)
            ctx["h_cos"] = c.transpose(0, 1).contiguous()
            ctx["h_sin"] = s.transpose(0, 1).contiguous()


# ---------------------------------------------------------------------------
# block patching
# ---------------------------------------------------------------------------

def per_video_self_attn(self_attn, x, freqs, num_videos=2):
    """Per-video attention with frozen weights: rearrange the concat sequence
    [b, n*l, d] to [b*n, l, d], run self_attn with the single-video freqs
    table (identical positional tables for both halves), rearrange back.
    No token ever attends across videos."""
    x2 = rearrange(x, 'b (n l) d -> (b n) l d', n=num_videos)
    out = self_attn(x2, freqs)
    return rearrange(out, '(b n) l d -> b (n l) d', n=num_videos)


def _patched_block_forward(self, x, context, cam_emb, t_mod, freqs):
    """Replaces DiTBlock.forward (diffsynth wan_video_dit.py). Identical to
    the original except: (a) self-attention runs per video with the
    single-video freqs table; (b) the trainable TTT branch output is added
    inside the gate: x + gate * (projector(attn) + ttt)."""
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
        self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
    ).chunk(6, dim=1)
    input_x = modulate(self.norm1(x), shift_msa, scale_msa)

    # frozen camera encoder, verbatim (their hardcoded 30 x 52 grid)
    cam = self.cam_encoder(cam_emb)
    cam = cam.repeat(1, 2, 1)
    cam = cam.unsqueeze(2).unsqueeze(3).repeat(1, 1, 30, 52, 1)
    cam = rearrange(cam, 'b f h w d -> b (f h w) d')
    input_x = input_x + cam

    # per-video attention: the first half of the concat freqs table is
    # exactly the single-video (frames 0..20) table, f being the outermost
    # grid dim (verified in sanity T0b)
    half = input_x.shape[1] // 2
    attn = per_video_self_attn(self.self_attn, input_x, freqs[:half])

    ctx = self._ttt_ctx
    if ctx.get("ttt_enabled", True):
        ttt_out = self.ttt_branch(input_x, ctx)
        x = x + gate_msa * (self.projector(attn) + ttt_out)
    else:
        x = x + gate_msa * self.projector(attn)

    x = x + self.cross_attn(self.norm3(x), context)
    input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
    x = x + gate_mlp * self.ffn(input_x)
    return x


def patch_dit(dit, variant_cfg, device="cuda"):
    """Attach a TTT branch to every DiT block and override the block forward.

    Must be called AFTER the ReCamMaster checkpoint has been loaded
    (strict=True) into `dit`. Returns the shared per-step context dict.
    variant_cfg: mapping with ttt_input_rope / ttt_hidden_rope (+ optional
    fw_head_dim / inter_multi / lr_dim / base_lr / ttt_chunk_frames /
    use_muon / num_muon_iters overrides)."""
    ctx = {"ttt_enabled": True}
    for block in dit.blocks:
        branch = RecamTTTBranch(
            dit.dim, block.self_attn,
            ttt_input_rope=bool(variant_cfg.get("ttt_input_rope", False)),
            ttt_hidden_rope=bool(variant_cfg.get("ttt_hidden_rope", False)),
            fw_head_dim=int(variant_cfg.get("fw_head_dim", 768)),
            inter_multi=int(variant_cfg.get("inter_multi", 2)),
            lr_dim=int(variant_cfg.get("lr_dim", 1)),
            base_lr=float(variant_cfg.get("base_lr", 0.001)),
            ttt_chunk_frames=int(variant_cfg.get("ttt_chunk_frames",
                                                 DEFAULT_CHUNK_FRAMES)),
            use_muon=bool(variant_cfg.get("use_muon", False)),
            num_muon_iters=int(variant_cfg.get("num_muon_iters", 5)),
        )
        branch.to(device=device, dtype=torch.float32)
        block.ttt_branch = branch          # registered submodule (trainable)
        block._ttt_ctx = ctx
        block.forward = types.MethodType(_patched_block_forward, block)
    return ctx


def freeze_all_but_ttt(dit, train_recam_modules=False):
    """Freeze every pretrained weight; the TTT branches train, and (Stage 1,
    train_recam_modules=True) so do each block's cam_encoder + projector —
    these are ReCamMaster's OWN new modules (added on top of Wan, trained for
    the concat-attention regime), NOT Wan weights. They are upcast to fp32
    (same dtype contract as the branches; values are preserved exactly, and
    under bf16 autocast the forward is bit-identical to the bf16 copies).
    Returns (trainable_params, n_trainable, n_frozen)."""
    dit.requires_grad_(False)
    for block in dit.blocks:
        block.ttt_branch.requires_grad_(True)
        if train_recam_modules:
            block.cam_encoder.float().requires_grad_(True)
            block.projector.float().requires_grad_(True)
    trainable = [p for p in dit.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_frozen = sum(p.numel() for p in dit.parameters() if not p.requires_grad)
    # trainable set must be exactly {ttt_branch} (+ {cam_encoder, projector})
    for name, p in dit.named_parameters():
        is_branch = "ttt_branch" in name
        is_recam = (".cam_encoder." in name) or (".projector." in name)
        expected = is_branch or (train_recam_modules and is_recam)
        assert p.requires_grad == expected, name
    return trainable, n_train, n_frozen


# ---------------------------------------------------------------------------
# camera math: per-token Plucker coordinates from the OFFICIAL UE-parsed c2ws
# ---------------------------------------------------------------------------

SENSOR_WIDTH_MM = 23.76
NATIVE_RES = 1280
TARGET_H, TARGET_W = 480, 832


def mcv_intrinsics(relpath):
    """Intrinsics of the decoded 480x832 frame, ported from
    minVid MultiCamPairDataset._decode_clip/_intrinsics geometry with the MCV
    native resolution (1280x1280): cover-resize scale 0.65 -> 832x832, center
    crop to 480x832. relpath like 'f24_aperture5/scene2937' (focal from dir)."""
    focal_mm = float(relpath.split(os.sep)[0].split("_")[0][1:])
    h0 = w0 = NATIVE_RES
    scale = max(TARGET_H / h0, TARGET_W / w0)
    out_h = max(TARGET_H, int(math.ceil(h0 * scale / 2) * 2))
    out_w = max(TARGET_W, int(math.ceil(w0 * scale / 2) * 2))
    sx, sy = out_w / w0, out_h / h0
    top = (out_h - TARGET_H) // 2
    left = (out_w - TARGET_W) // 2
    fx = focal_mm / SENSOR_WIDTH_MM * w0
    fy = focal_mm / SENSOR_WIDTH_MM * h0
    cx, cy = w0 / 2.0, h0 / 2.0
    return torch.tensor(
        [[fx * sx, 0.0, cx * sx - left],
         [0.0, fy * sy, cy * sy - top],
         [0.0, 0.0, 1.0]], dtype=torch.float32)


def build_recam_coords6(cam_json_path, src_cam, tgt_cam, relpath,
                        device="cuda"):
    """Per-token 6D Plucker coordinates for the [TGT || SRC] token layout.

    Uses the OFFICIAL UE-parsed c2ws (eval_recam_anchor.load_mcv_c2ws); ALL
    42 frames' poses are expressed relative to the SOURCE camera's frame-0
    pose (the same anchor as ReCamMaster's cam_emb): src frame0 = identity.
    Returns [2*21*1560, 6] fp32 — TGT tokens first, then SRC tokens.
    All pose math fp32."""
    sys.path.insert(0, os.path.join(_HERE, "..", "minVid"))
    from eval_recam_anchor import load_mcv_c2ws
    import numpy as np

    c2w_src = torch.tensor(np.stack(load_mcv_c2ws(cam_json_path, src_cam)),
                           dtype=torch.float32)
    c2w_tgt = torch.tensor(np.stack(load_mcv_c2ws(cam_json_path, tgt_cam)),
                           dtype=torch.float32)
    anchor_inv = torch.inverse(c2w_src[0])
    rel_src = anchor_inv @ c2w_src  # [21, 4, 4], frame0 == I
    rel_tgt = anchor_inv @ c2w_tgt

    K = mcv_intrinsics(relpath)
    pl_tgt = plucker_per_token(rel_tgt, K, latent_hw=(TOK_H, TOK_W))
    pl_src = plucker_per_token(rel_src, K, latent_hw=(TOK_H, TOK_W))
    coords6 = torch.cat([pl_tgt.reshape(-1, 6), pl_src.reshape(-1, 6)], dim=0)
    return coords6.to(device)
