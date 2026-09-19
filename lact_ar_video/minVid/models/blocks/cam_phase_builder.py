"""Plucker-ray phase builder for the camera-controlled video (ccv) runs.

Ports the NVS PRA conventions (lact_nvs/lact_ttt_cam.py) to the Wan latent
token grid:
  - per-token Plucker coordinates (d, o x d) on the 30x52 latent patch grid,
    token order (h, w) row-major to match Conv3d patchify (f, h, w) flatten;
  - geometric frequency ladders  omega = pi * logspace2(0.5, 16, F)  with a
    per-(coord, freq) gain (learnable when ttt_learnable_freqs);
  - sequence assembly for the [SRC 21 frames || TGT AR-interleave 39 frames]
    layout (both noisy+clean copies of a tgt frame share the same phases).

All camera math is fp32 (built under no_grad in the model; only the gain
parameters of the ladders may carry grad, inside the attention layer).
"""
import math

import torch


def make_cam_ladder(num_freqs: int) -> torch.Tensor:
    """Geometric frequency ladder, NVS convention: pi * 2^linspace(-1, 4)."""
    return math.pi * torch.logspace(
        math.log2(0.5), math.log2(16.0), num_freqs, base=2.0
    )


def cam_phase_tables(coords6: torch.Tensor, omega: torch.Tensor, gain: torch.Tensor):
    """Plucker phases -> (cos, sin) tables.

    coords6: [..., L, 6] fp32; omega: [F]; gain: [6, F].
    Returns cos/sin of shape [..., L, 6*F] (coord-major, freq-minor flatten,
    same as lact_nvs _rope_coeffs).
    """
    theta = coords6.float().unsqueeze(-1) * (
        omega.float()[None, None, :] * gain.float()[None, :, :]
    )  # [..., L, 6, F]
    theta = theta.flatten(-2)
    return theta.cos(), theta.sin()


def plucker_per_token(c2w: torch.Tensor, K: torch.Tensor,
                      latent_hw=(30, 52), pixels_per_token: int = 16):
    """Per-token Plucker coordinates on the transformer token grid.

    c2w: [F, 4, 4] fp32, canonical CV-convention camera-to-world.
    K:   [3, 3] fp32 intrinsics of the decoded pixel frame (480x832).
    Token (py, px) has pixel center (u, v) = ((px+0.5)*16, (py+0.5)*16).
    Returns [F, H*W, 6] fp32 = (d, o x d), token order row-major (h, w).
    """
    H, W = latent_hw
    device = c2w.device
    py, px = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    u = (px.reshape(-1) + 0.5) * pixels_per_token
    v = (py.reshape(-1) + 0.5) * pixels_per_token
    pix = torch.stack([u, v, torch.ones_like(u)], dim=0)  # [3, HW]
    dirs_cam = torch.inverse(K.float()) @ pix  # [3, HW]
    R = c2w[:, :3, :3].float()  # [F, 3, 3]
    d = torch.einsum("fij,jl->fli", R, dirs_cam)  # [F, HW, 3]
    d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    o = c2w[:, :3, 3].float()[:, None, :].expand_as(d)
    m = torch.cross(o, d, dim=-1)
    return torch.cat([d, m], dim=-1)  # [F, HW, 6]


def tgt_interleave_frame_order(n_latent_f: int = 21, ar_window_f: int = 3):
    """Latent-frame index per interleave slot: [n0 c0 n1 c1 ... c5 n6].

    Both copies (noisy + clean) of a frame get the same index, matching
    rope_apply_ar's time mapping. Returns a list of length 2*n - ar_window_f.
    """
    n_w = n_latent_f // ar_window_f
    order = list(range(ar_window_f))  # first noisy chunk: frames 0..ar-1
    for w in range(n_w - 1):
        clean = list(range(w * ar_window_f, (w + 1) * ar_window_f))
        noisy = list(range((w + 1) * ar_window_f, (w + 2) * ar_window_f))
        order += clean + noisy
    return order


def build_ccv_cam_inputs(c2w_src: torch.Tensor, c2w_tgt: torch.Tensor,
                         K: torch.Tensor, latent_hw=(30, 52),
                         n_latent_f: int = 21, ar_window_f: int = 3):
    """Camera conditioning tensors for one [SRC || TGT-interleave] sample.

    c2w_src / c2w_tgt: [F, 4, 4] canonical fp32; K: [3, 3].
    Returns:
      cam12_per_frame: [F_total, 12] fp32; SRC frames = identity 3x4,
        TGT frames = (inv(c2w_src[t]) @ c2w_tgt[t])[:3, :4] flattened
        (ReCamMaster gauge: relative to the condition camera).
      coords6: [L_total, 6] fp32 per-token Plucker, order
        [SRC 21 frames || TGT interleave frame order] x (H*W tokens).
    """
    order = tgt_interleave_frame_order(n_latent_f, ar_window_f)

    pl_src = plucker_per_token(c2w_src, K, latent_hw)  # [F, HW, 6]
    pl_tgt = plucker_per_token(c2w_tgt, K, latent_hw)
    coords6 = torch.cat(
        [pl_src.reshape(-1, 6), pl_tgt[order].reshape(-1, 6)], dim=0
    )

    rel = torch.inverse(c2w_src.float()) @ c2w_tgt.float()  # [F, 4, 4]
    rel12 = rel[:, :3, :4].reshape(rel.shape[0], 12)
    eye12 = torch.eye(4, device=rel.device, dtype=torch.float32)[:3, :4].reshape(1, 12)
    cam12_per_frame = torch.cat(
        [eye12.expand(c2w_src.shape[0], 12), rel12[order]], dim=0
    )
    return cam12_per_frame, coords6


# ---------------------------------------------------------------------------
# CaPET (final NVS recipe) inputs: the 3D point at a predicted depth needs the ray
# ORIGIN as well as its direction, which the Plucker pair (d, o x d) does not expose
# per token. `build_ccv_capet_inputs` returns (o, d, t_c) instead, t_c being the ray's
# closest approach to the scene focus -- the depth the layer's learned channel scales.
# ---------------------------------------------------------------------------

def point_ray_per_token(c2w: torch.Tensor, K: torch.Tensor,
                        latent_hw=(30, 52), pixels_per_token: int = 16):
    """Per-token ray origin and unit direction, same token order as plucker_per_token.

    Returns (o, d), each [F, H*W, 3] fp32.
    """
    H, W = latent_hw
    device = c2w.device
    py, px = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    u = (px.reshape(-1) + 0.5) * pixels_per_token
    v = (py.reshape(-1) + 0.5) * pixels_per_token
    pix = torch.stack([u, v, torch.ones_like(u)], dim=0)
    dirs_cam = torch.inverse(K.float()) @ pix
    R = c2w[:, :3, :3].float()
    d = torch.einsum("fij,jl->fli", R, dirs_cam)
    d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    o = c2w[:, :3, 3].float()[:, None, :].expand_as(d).contiguous()
    return o, d


def scene_focus(c2w: torch.Tensor, lam: float = 1e-2) -> torch.Tensor:
    """Least-squares intersection of a camera trajectory's optical axes ([3] fp32).

    Same estimator as the NVS layer (tttlrm_ref/model/capet.py set_point_info), fed
    here by the SRC trajectory -- the ccv analogue of the input views.
    """
    cen = c2w[:, :3, 3].float()                                   # [F, 3]
    fwd = c2w[:, :3, 2].float()                                   # +z optical axis
    fwd = fwd / fwd.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    eye3 = torch.eye(3, device=c2w.device, dtype=torch.float32)
    Pm = eye3[None] - fwd[:, :, None] * fwd[:, None, :]           # I - f f^T
    A = Pm.sum(0) + lam * eye3
    f_mean = fwd.mean(0)
    f_mean = f_mean / f_mean.norm().clamp_min(1e-8)
    prior = cen.mean(0) + f_mean
    b = torch.einsum("fij,fj->i", Pm, cen) + lam * prior
    return torch.linalg.solve(A, b)


def build_ccv_capet_inputs(c2w_src: torch.Tensor, c2w_tgt: torch.Tensor,
                           K: torch.Tensor, latent_hw=(30, 52),
                           n_latent_f: int = 21, ar_window_f: int = 3):
    """[L_total, 7] fp32 = (ray origin, unit direction, foot depth t_c) per token.

    Sequence order matches build_ccv_cam_inputs: [SRC frames || TGT interleave order].
    """
    order = tgt_interleave_frame_order(n_latent_f, ar_window_f)
    o_s, d_s = point_ray_per_token(c2w_src, K, latent_hw)
    o_t, d_t = point_ray_per_token(c2w_tgt, K, latent_hw)
    o = torch.cat([o_s.reshape(-1, 3), o_t[order].reshape(-1, 3)], dim=0)
    d = torch.cat([d_s.reshape(-1, 3), d_t[order].reshape(-1, 3)], dim=0)
    focus = scene_focus(c2w_src)
    tc = ((focus[None, :] - o) * d).sum(-1, keepdim=True)
    return torch.cat([o, d, tc], dim=-1)


# ---------------------------------------------------------------------------
# PRoPE baseline inputs: per-slot intrinsics and extrinsics.
#
# PRoPE codes the RELATIVE camera as a projective transform, so unlike the rotary
# recipes it needs the matrices themselves, not per-token coordinates. The block's
# only camera channel is the per-token `cam_coords6` argument, so the per-slot
# (K_norm, c2w) pair is broadcast along that slot's tokens and sliced back out in
# the block; at 93,600 tokens x 20 floats that is 7.5 MB, which is not worth a new
# argument through three wrapper signatures.
# ---------------------------------------------------------------------------

def build_ccv_prope_inputs(c2w_src: torch.Tensor, c2w_tgt: torch.Tensor,
                           K: torch.Tensor, latent_hw=(30, 52),
                           n_latent_f: int = 21, ar_window_f: int = 3,
                           pixels_per_token: int = 16):
    """[L_total, 20] fp32 = (K_norm 4 | c2w 16) repeated over each slot's tokens.

    K_norm is PRoPE's normalised (fx/W, fy/H, cx/W - 0.5, cy/H - 0.5) on the DECODED
    pixel frame, the same convention the NVS cell uses. Slot order matches
    build_ccv_cam_inputs: [SRC frames || TGT interleave order].
    """
    order = tgt_interleave_frame_order(n_latent_f, ar_window_f)
    H, W = latent_hw
    tpv = H * W
    Wpx, Hpx = W * pixels_per_token, H * pixels_per_token
    K = K.float()
    K_norm = torch.tensor(
        [K[0, 0] / Wpx, K[1, 1] / Hpx, K[0, 2] / Wpx - 0.5, K[1, 2] / Hpx - 0.5],
        device=K.device, dtype=torch.float32)                      # shared by all slots
    c2w = torch.cat([c2w_src.float(), c2w_tgt.float()[order]], dim=0)   # [V, 4, 4]
    V = c2w.shape[0]
    per_slot = torch.cat([K_norm[None].expand(V, 4), c2w.reshape(V, 16)], dim=-1)  # [V, 20]
    return per_slot[:, None, :].expand(V, tpv, 20).reshape(V * tpv, 20).contiguous()
