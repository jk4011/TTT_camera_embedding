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
