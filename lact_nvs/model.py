# LaCT

import math
from einops import einsum, rearrange, repeat
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn
from torch.nn import LayerNorm
from torch.nn import functional as F

from lact_ttt import TTTOperator

def get_class_by_name(name):
    parts = name.split(".")
    module_name = ".".join(parts[:-1])
    class_name = parts[-1]
    
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)


def _init_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.RMSNorm, nn.LayerNorm)):
        module.reset_parameters()
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


class SelfAttention(nn.Module):
    """
    Self-attention layer
    Reference: https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/vision_transformer.py#L68-L92
    """

    def __init__(
        self,
        dim,
        head_dim,
        use_qk_norm=True,
        causal=False,
        bias=False,
        block_causal=False,
        attn_mode="none",
        prope_proj_frac=0.5,
    ):
        super().__init__()
        assert dim % head_dim == 0
        self.dim = dim
        self.head_dim = head_dim

        self.to_qkv = nn.Linear(dim, 3 * dim, bias=bias)
        self.c_proj = nn.Linear(dim, dim, bias=bias)
        self.use_qk_norm = use_qk_norm

        if self.use_qk_norm:
            self.q_norm = nn.RMSNorm(head_dim)
            self.k_norm = nn.RMSNorm(head_dim)

        self.causal = causal
        # Attention CEILING controls for the camera-embedding program (2026-08-31), used
        # with length_dim "vl" in place of the TTT layer (LaCT's own full-attention
        # baseline: "block-wise causal attention -- bidirectional among input tokens,
        # cross-attention from novel views"):
        #   block_causal: input tokens attend to all input tokens; target tokens attend
        #                 to all input tokens + their own view's tokens only.
        #   attn_mode "prope": faithful PRoPE on q/k/v/o per head ([frac] tiled
        #                 projective P = lift(K) w2c | image-x RoPE | image-y RoPE, freq
        #                 base 100, inverse on the output), after the qk norm.
        assert attn_mode in ("none", "prope"), attn_mode
        self.block_causal = block_causal
        self.attn_mode = attn_mode
        self.prope_proj_frac = prope_proj_frac

    def forward(self, x, info=None, *args):
        """
        x: (b, l, d)
        """
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, "b l (qkv nh dh) -> qkv b nh l dh", qkv=3, dh=self.head_dim)
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        prope_state = None
        if self.attn_mode == "prope":
            from lact_ttt_cam import (_prope_rope_apply, _prope_rope_coeffs,
                                      apply_tiled_mat4, lift_K4, lift_K4_inv, to_heads)
            assert info is not None and "view_w2c" in info, "attn prope needs camera_info"
            b, nh, L, hd = q.shape
            tpv = info["tokens_per_view"]
            with torch.autocast(device_type=x.device.type, enabled=False):
                K, w2c = info["view_K_norm"].float(), info["view_w2c"].float()
                P = lift_K4(K) @ w2c
                P_inv = info["view_c2w"].float() @ lift_K4_inv(K)
            half = int(hd * self.prope_proj_frac) // 8 * 8
            quart = (hd - half) // 2
            P_h, P_inv_h = to_heads(P, nh), to_heads(P_inv, nh)
            px = int(math.sqrt(tpv)); assert px * px == tpv, tpv
            pos = torch.arange(tpv, device=q.device)
            cx, sx = _prope_rope_coeffs(pos % px, quart, q.device)
            cy, sy = _prope_rope_coeffs(pos // px, quart, q.device)
            V = P.shape[1]
            cx, sx, cy, sy = cx.repeat(V, 1), sx.repeat(V, 1), cy.repeat(V, 1), sy.repeat(V, 1)

            def _apply(t, mat, inv=False):
                t = rearrange(t, "b nh l dh -> (b nh) l dh")
                t2 = apply_tiled_mat4(t, mat, tpv, half)
                a = _prope_rope_apply(t2[..., half:half + quart], cx, sx, inv)
                c = _prope_rope_apply(t2[..., half + quart:], cy, sy, inv)
                t3 = torch.cat([t2[..., :half], a, c], dim=-1)
                return rearrange(t3, "(b nh) l dh -> b nh l dh", nh=nh)

            q = _apply(q, P_h.transpose(-1, -2))
            k = _apply(k, P_inv_h)
            v = _apply(v, P_inv_h)
            prope_state = (P_h, _apply)

        attn_mask = None
        if self.block_causal:
            assert info is not None and "num_input_views" in info, "block_causal needs camera_info"
            L = q.shape[2]
            tpv = info["tokens_per_view"]
            n_in = info["num_input_views"] * tpv
            vid = torch.arange(L, device=q.device) // tpv
            is_in = torch.arange(L, device=q.device) < n_in
            attn_mask = is_in[None, :] | (vid[:, None] == vid[None, :])   # [Lq, Lk] bool
            attn_mask = attn_mask[None, None]

        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=self.causal)
        if prope_state is not None:
            P_h, _apply = prope_state
            x = _apply(x, P_h, inv=True)
        x = rearrange(x, "b nh l dh -> b l (nh dh)")

        x = self.c_proj(x)
        return x, {}


class MLP(nn.Module):

    def __init__(self, dim, inter_multi=4, bias=False):
        super().__init__()
        intermediate_dim = int(dim * inter_multi)
        self.c_fc = nn.Linear(dim, intermediate_dim, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(intermediate_dim, dim, bias=bias)

    def forward(self, x, *args):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x, {}


class Block(nn.Module):
    def __init__(self, dim, bias, block_config):
        super().__init__()
        module_list = []
        self.length_dim_list = []

        for _, module_config in enumerate(block_config):
            CLASS = get_class_by_name(module_config["type"])
            module = nn.ModuleDict(
                {
                    "ln": LayerNorm(dim, bias=bias),
                    "f": CLASS(dim=dim, bias=bias, **module_config["params"]),
                }
            )

            module_list.append(module)
            self.length_dim_list.append(module_config.get("length_dim", "vl"))

        self.module_list = nn.ModuleList(module_list)

    def forward(self, x, info):
        results = {}
        for module, length_dim in zip(self.module_list, self.length_dim_list):
            residual = x
            x = module["ln"](x)

            if length_dim == "l":
                b, vl, d = x.shape
                l = info["num_img_tokens"]
                x = x.reshape(b * (vl // l), l, d)
                x, result = module["f"](x, info)
                x = x.reshape(b, vl, d)
            else:
                x, result = module["f"](x, info)

            x = residual + x
            results.update(result)
        return x, results



def compute_rays(fxfycxcy, c2w, h, w):
    """Transform target before computing loss
    Args:
        fxfycxcy (torch.tensor): [b, v, 4]
        c2w (torch.tensor): [b, v, 4, 4]
    Returns:
        ray_o: (b, v, 3, h, w)
        ray_d: (b, v, 3, h, w)
    """
    b, v = fxfycxcy.size(0), fxfycxcy.size(1)

    # Efficient meshgrid equivalent using broadcasting
    idx_x = torch.arange(w, device=c2w.device)[None, :].expand(h, -1)  # [h, w]
    idx_y = torch.arange(h, device=c2w.device)[:, None].expand(-1, w)  # [h, w]

    # Reshape for batched matrix multiplication
    idx_x = idx_x.flatten().expand(b * v, -1)           # [b*v, h*w]
    idx_y = idx_y.flatten().expand(b * v, -1)           # [b*v, h*w]

    fxfycxcy = fxfycxcy.reshape(b * v, 4)               # [b*v, 4]
    c2w = c2w.reshape(b * v, 4, 4)                      # [b*v, 4, 4]

    x = (idx_x + 0.5 - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]     # [b*v, h*w]
    y = (idx_y + 0.5 - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]     # [b*v, h*w]
    z = torch.ones_like(x)                                      # [b*v, h*w]

    ray_d = torch.stack([x, y, z], dim=1)                       # [b*v, 3, h*w]
    ray_d = torch.bmm(c2w[:, :3, :3], ray_d)                    # [b*v, 3, h*w]
    ray_d = ray_d / torch.norm(ray_d, dim=1, keepdim=True)      # [b*v, 3, h*w]

    ray_o = c2w[:, :3, 3:4].expand(b * v, -1, h*w)              # [b*v, 3, h*w]

    ray_o = ray_o.reshape(b, v, 3, h, w)                        # [b, v, 3, h, w]
    ray_d = ray_d.reshape(b, v, 3, h, w)                        # [b, v, 3, h, w]

    return ray_o, ray_d


def compute_camera_info(fxfycxcy, c2w, h, w, patch_size, ray_o, ray_d, num_input_views,
                        cam_scene_random=False, tok_t_gt=None, focus_mode="ls"):
    """Per-token / per-view camera tensors for camera-conditioned TTT layers.

    All views (input + target) are covered; token order matches the
    patchify rearrange "b v c (hh ph) (ww pw) -> b (v hh ww) ...".

    cam_scene_random (Q1 absolute-adaptation probe): replace the per-token
    Plucker phase coordinates (tok_d, tok_m) with ONE random 6-vector per
    batch item, broadcast to all tokens/views of that scene (d = random unit
    3-vector, m = 0.5 * randn(3), fresh each forward). All relative rotations
    become identity; only absolute stamps vary across scenes. Raymap INPUT
    features (compute_rays outputs) and cam_feat keep the true rays.

    Returns dict with:
        tok_o, tok_d, tok_m: [b, L, 3]  patch-center Plucker (canonical frame)
        cam_feat:            [b, L, 11] (o, d, m, fx/w, fy/h)
        cam_feat_lr:         [b, L, 12] cam_feat + view novelty
        view_rot:            [b, v, 3, 3] c2w rotations
        view_w2c:            [b, v, 4, 4]
        view_K_norm:         [b, v, 4]  (fx/w, fy/h, cx/w - 0.5, cy/h - 0.5)
        view_pose11:         [b, v, 11] per-view pose summary
        tokens_per_view, num_views, num_input_views
    """
    b, v = fxfycxcy.size(0), fxfycxcy.size(1)
    p = patch_size

    # Patch-center rays: pool pixel rays, renormalize the direction.
    tok_d = F.avg_pool2d(ray_d.flatten(0, 1), p).unflatten(0, (b, v))  # [b,v,3,hh,ww]
    tok_d = tok_d / (tok_d.norm(dim=2, keepdim=True) + 1e-8)
    tok_o = c2w[:, :, :3, 3][..., None, None].expand_as(tok_d)         # cam center
    tok_m = torch.cross(tok_o, tok_d, dim=2)

    hh, ww = tok_d.shape[-2], tok_d.shape[-1]
    tok_o = rearrange(tok_o, "b v c hh ww -> b (v hh ww) c")
    tok_d = rearrange(tok_d, "b v c hh ww -> b (v hh ww) c")
    tok_m = rearrange(tok_m, "b v c hh ww -> b (v hh ww) c")

    # Per-token 2-D patch coordinate WITHIN its view, normalised to [-1, 1] so it
    # shares the O(1) scale of the Plucker coordinates and the same omega ladder is
    # meaningful for both. Plucker says which view a token came from; this says where
    # inside that view it sits, which the Plucker coordinates alone do not encode at
    # patch resolution. Token order matches the rearrange above.
    _u = torch.linspace(-1.0, 1.0, hh, device=tok_d.device)
    _v = torch.linspace(-1.0, 1.0, ww, device=tok_d.device)
    _uu, _vv = torch.meshgrid(_u, _v, indexing="ij")
    tok_uv = torch.stack([_uu, _vv], dim=-1).reshape(1, 1, hh * ww, 2)
    tok_uv = tok_uv.expand(b, v, hh * ww, 2).reshape(b, v * hh * ww, 2).contiguous()

    # Per-token patch footprint (half-range of each Plucker coordinate over
    # the patch's pixel rays) for ray-cone anti-aliased phases.
    def patch_half_range(x):  # [b, v, 3, h, w] -> [b, L, 3]
        xf = x.flatten(0, 1)
        half = 0.5 * (F.max_pool2d(xf, p) + F.max_pool2d(-xf, p))
        return rearrange(half.unflatten(0, (b, v)), "b v c hh ww -> b (v hh ww) c")

    ray_d_n = ray_d / (ray_d.norm(dim=2, keepdim=True) + 1e-8)
    m_pix = torch.cross(ray_o, ray_d_n, dim=2)
    tok_d_delta = patch_half_range(ray_d_n)
    tok_m_delta = patch_half_range(m_pix)

    K_norm = torch.stack([
        fxfycxcy[..., 0] / w,
        fxfycxcy[..., 1] / h,
        fxfycxcy[..., 2] / w - 0.5,
        fxfycxcy[..., 3] / h - 0.5,
    ], dim=-1)  # [b, v, 4]

    tokens_per_view = tok_o.size(1) // v
    K_tok = K_norm[:, :, :2].repeat_interleave(tokens_per_view, dim=1)  # [b, L, 2]
    cam_feat = torch.cat([tok_o, tok_d, tok_m, K_tok], dim=-1)          # [b, L, 11]

    # Per-view pose summary: camera center, forward axis, its moment, fx, fy.
    rot = c2w[:, :, :3, :3]
    center = c2w[:, :, :3, 3]
    forward = rot[:, :, :, 2]
    view_pose11 = torch.cat(
        [center, forward, torch.cross(center, forward, dim=-1), K_norm[:, :, :2]], dim=-1
    )

    # View novelty: distance to nearest *input* view (self excluded).
    tdist = torch.cdist(center, center[:, :num_input_views])            # [b, v, v_in]
    rel = torch.einsum("bvij,bwik->bvwjk", rot, rot[:, :num_input_views])
    tr = rel.diagonal(dim1=-2, dim2=-1).sum(-1)
    rdist = torch.arccos(((tr - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6)) / math.pi
    pose_dist = tdist + rdist
    eye_mask = torch.zeros_like(pose_dist)
    eye_mask[:, :num_input_views] = torch.eye(num_input_views, device=c2w.device) * 1e6
    novelty = (pose_dist + eye_mask).min(dim=-1, keepdim=True).values   # [b, v, 1]
    novelty_tok = novelty.repeat_interleave(tokens_per_view, dim=1)
    cam_feat_lr = torch.cat([cam_feat, novelty_tok], dim=-1)            # [b, L, 12]

    # Analytic SE(3) inverse (avoids CUDA cusolver): w2c = [[R^T, -R^T t],[0,1]]
    w2c = torch.zeros_like(c2w)
    w2c[..., 3, 3] = 1.0
    w2c[..., :3, :3] = rot.transpose(-1, -2)
    w2c[..., :3, 3] = -torch.einsum("bvji,bvj->bvi", rot, center)

    if cam_scene_random:
        # Q1 probe: per-scene-constant random rotary phases (see docstring).
        # Only tok_d / tok_m (the 6D Plucker rotary-phase source) are
        # replaced; everything derived from the true rays above stays intact.
        L = tok_d.size(1)
        rand_d = torch.randn(b, 1, 3, device=tok_d.device, dtype=torch.float32)
        rand_d = rand_d / (rand_d.norm(dim=-1, keepdim=True) + 1e-8)
        rand_m = 0.5 * torch.randn(b, 1, 3, device=tok_d.device, dtype=torch.float32)
        tok_d = rand_d.expand(b, L, 3)
        tok_m = rand_m.expand(b, L, 3)

    # ---- Object-centric geometry for 3D-POINT addressing (gObjaverse program, 2026-08-31).
    # Scene focus point p*: least-squares intersection of the INPUT views' optical axes
    # (exact object centre for look-at renders; a regularised "point in front of the mean
    # camera" for forward-facing captures where the axes are nearly parallel). Per token,
    # the closest-approach ray parameter t_c = (p* - o).d and the squared impact parameter
    # b^2 = |p* - o|^2 - t_c^2 let a TTT layer form the chord of the ray through a sphere
    # of radius r around p* without any learned depth. All in the canonical frame.
    f_in = rot[:, :num_input_views, :, 2]                                 # [b, v_in, 3]
    c_in = center[:, :num_input_views]                                    # [b, v_in, 3]
    eye3 = torch.eye(3, device=c2w.device, dtype=c2w.dtype)
    Pm = eye3[None, None] - f_in[..., :, None] * f_in[..., None, :]       # I - f f^T
    lam = 1e-2
    A = Pm.sum(1) + lam * eye3[None]
    f_mean = f_in.mean(1)
    f_mean = f_mean / (f_mean.norm(dim=-1, keepdim=True) + 1e-8)
    prior = c_in.mean(1) + f_mean                                         # 1 unit ahead of the mean cam
    bvec = torch.einsum("bvij,bvj->bi", Pm, c_in) + lam * prior
    focus = torch.linalg.solve(A, bvec)                                   # [b, 3]

    # ---- Baseline (2-view) geometry for EPIPOLAR-PLANE codes (P2 program, 2026-09-01).
    # Line through the input camera centres: b_hat (exact baseline for 2 inputs; first principal
    # axis for more). For any two cameras on that line and any ray through a scene point X,
    # b x (X - c_i) is the same vector, so the ANGLE of the ray's epipolar plane about the line,
    #   phi = atan2(n.e2, n.e1),  n = b_hat x d,
    # is identical for matched pixels at every depth and every baseline width -- the only
    # depth-free exact per-token invariant (no scene focus point, no scale). The along-line angle
    # alpha = acos(b_hat.d) carries the parallax; u = position of the token's camera along the
    # line (0/1 at the extreme inputs); nu = vergence of the two extreme input axes;
    # psi_c = alpha + (1/2 - u) nu is the vergence-corrected along-line angle. All angles, so
    # the frequency ladder has no scene-unit knob.
    v_all = center.shape[1]; L_tok = tok_o.shape[1]; tpv = L_tok // v_all
    assert tpv * v_all == L_tok, (L_tok, v_all)
    cm = c_in.mean(1, keepdim=True)
    if num_input_views == 2:
        bvec_ = c_in[:, 1] - c_in[:, 0]
    else:
        Xc = c_in - cm
        _, evecs = torch.linalg.eigh(Xc.transpose(1, 2) @ Xc)
        bvec_ = evecs[..., -1]
        sgn = torch.sign(((c_in[:, -1] - c_in[:, 0]) * bvec_).sum(-1, keepdim=True))
        bvec_ = bvec_ * torch.where(sgn == 0, torch.ones_like(sgn), sgn)
    bhat = bvec_ / (bvec_.norm(dim=-1, keepdim=True) + 1e-8)                 # [b, 3]
    proj_in = ((c_in - cm) * bhat[:, None]).sum(-1)                           # [b, v_in]
    ar = torch.arange(c_in.shape[0], device=c_in.device)
    i_min, i_max = proj_in.argmin(1), proj_in.argmax(1)
    c1, c2 = c_in[ar, i_min], c_in[ar, i_max]
    blen = ((c2 - c1) * bhat).sum(-1, keepdim=True).clamp_min(1e-4)          # [b, 1]
    f1, f2 = f_in[ar, i_min], f_in[ar, i_max]
    nu = torch.acos((f1 * f2).sum(-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6))
    nu = nu.clamp(math.radians(2.0), math.radians(178.0))                    # [b, 1]
    e1 = f_mean - (f_mean * bhat).sum(-1, keepdim=True) * bhat
    e1n = e1.norm(dim=-1, keepdim=True)
    up = torch.tensor([0.0, 1.0, 0.0], device=c2w.device, dtype=bhat.dtype).expand_as(bhat)
    xax = torch.tensor([1.0, 0.0, 0.0], device=c2w.device, dtype=bhat.dtype).expand_as(bhat)
    alt = torch.cross(bhat, up, dim=-1)
    alt = torch.where(alt.norm(dim=-1, keepdim=True) > 0.2, alt, torch.cross(bhat, xax, dim=-1))
    e1 = torch.where(e1n > 0.2, e1 / (e1n + 1e-8), alt / (alt.norm(dim=-1, keepdim=True) + 1e-8))
    e2 = torch.cross(bhat, e1, dim=-1)
    u_view = ((center - c1[:, None]) * bhat[:, None]).sum(-1) / blen          # [b, v_all]
    tok_u = u_view.repeat_interleave(tpv, dim=1)[..., None]                   # [b, L, 1]
    n_tok = torch.cross(bhat[:, None].expand_as(tok_d), tok_d, dim=-1)        # [b, L, 3]
    tok_epi_env = (n_tok.norm(dim=-1, keepdim=True) / 0.2).clamp(0.0, 1.0)    # 0 at the epipole
    tok_phi = torch.atan2((n_tok * e2[:, None]).sum(-1, keepdim=True),
                          (n_tok * e1[:, None]).sum(-1, keepdim=True))        # [b, L, 1]
    tok_alpha = torch.acos((tok_d * bhat[:, None]).sum(-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6))
    tok_psic = tok_alpha + (0.5 - tok_u) * nu[:, None]
    if focus_mode == "vergence":
        # p_nu: the point on the mean axis where the baseline subtends the vergence angle
        # (isosceles construction). = p* for look-at rigs; a well-defined far point when the axes
        # diverge; clamped to [1/4, 16] baselines so forward walks get a focus "at infinity"
        # instead of an ill-conditioned LS solve.
        rho = (blen / (2.0 * torch.tan(0.5 * nu))).clamp(0.25 * blen, 16.0 * blen)
        focus = 0.5 * (c1 + c2) + rho * f_mean
    else:
        assert focus_mode == "ls", focus_mode
    rel = focus[:, None, :] - tok_o                                       # [b, L, 3]
    tok_tc = (rel * tok_d).sum(-1, keepdim=True)                          # [b, L, 1]
    tok_b2 = (rel.pow(2).sum(-1, keepdim=True) - tok_tc.pow(2)).clamp_min(0.0)
    view_fdist = (focus[:, None, :] - center).norm(dim=-1)                # [b, v]

    out = {
        "tok_o": tok_o, "tok_d": tok_d, "tok_m": tok_m, "tok_uv": tok_uv,
        "tok_d_delta": tok_d_delta, "tok_m_delta": tok_m_delta,
        "tok_tc": tok_tc, "tok_b2": tok_b2, "focus": focus, "view_fdist": view_fdist,
        "tok_phi": tok_phi, "tok_alpha": tok_alpha, "tok_psic": tok_psic, "tok_u": tok_u,
        "tok_epi_env": tok_epi_env, "bip_nu": nu, "bhat": bhat,
        "cam_feat": cam_feat, "cam_feat_lr": cam_feat_lr,
        "view_rot": rot, "view_w2c": w2c, "view_c2w": c2w,
        "view_K_norm": K_norm, "view_pose11": view_pose11,
        "tokens_per_view": tokens_per_view,
        "num_views": v, "num_input_views": num_input_views,
    }
    if tok_t_gt is not None:
        # Oracle diagnostics only: GT ray parameter of the patch-centre surface point
        # (0 = no surface / background), already in the canonical scale. [b, L, 1]
        out["tok_t_gt"] = tok_t_gt.reshape(b, -1, 1).to(tok_o.dtype)
    return out


class LaCTLVSM(nn.Module):
    def __init__(self, patch_size, dim, layers, block_config,
                 ttt_chunk_per_view=False, ttt_view_tour=False,
                 ttt_num_chunks=1,
                 cam_scene_random=False,
                 input_raymap="world", focus_mode="ls"):
        super().__init__()
        self.patch_size = patch_size
        self.dim = dim
        # input_raymap (gObjaverse program, 2026-08-31):
        #   "world"  -- stock: per-pixel Plucker rays (o, d, o x d) in the canonical scene
        #               frame are the token features, so every token carries ABSOLUTE pose.
        #   "camray" -- pose-free tokens (the PRoPE/RayRoPE/GTA regime): rays computed with
        #               IDENTITY extrinsics (o = 0, d = normalised K^-1 [u v 1], o x d = 0),
        #               i.e. intrinsics + pixel position only. Pose then reaches the network
        #               ONLY through the camera-conditioned TTT layer (its transforms use the
        #               true c2w via compute_camera_info), making the whole model relative.
        assert input_raymap in ("world", "camray"), input_raymap
        self.input_raymap = input_raymap
        # focus_mode (P2 program): "ls" = LS intersection of input optical axes (stock);
        # "vergence" = p_nu, the isosceles vergence focus (p*-free, well-conditioned on walks).
        assert focus_mode in ("ls", "vergence"), focus_mode
        self.focus_mode = focus_mode
        # Camera-scheduled TTT updates: one update chunk per input view
        # (multi-step inner optimization), optionally ordered far-from-target
        # -> near-target so that target-adjacent views are written last
        # (weight-norm recency works in our favor).
        self.ttt_chunk_per_view = ttt_chunk_per_view
        # ttt_num_chunks: split the input-token update into n sequential chunks.
        # The method is derived for a SINGLE update step; at scale the update is
        # chunked, so this exists to test that the learned phases survive that.
        # Evaluation-time knob: 1 reproduces the trained setting exactly.
        # int -> fixed n. list -> MULTI-CHUNK TRAINING: one n is drawn uniformly per
        # forward, so a single model learns to update under every chunk count instead
        # of training one model per n. Eval still passes a fixed int.
        # Normalized here because a YAML list arrives as an omegaconf ListConfig,
        # which is NOT a list/tuple subclass: the isinstance check below would miss
        # it and fall through to `ttt_num_chunks > 1`, which raises TypeError on a
        # ListConfig. Coerce any non-int sequence to a plain list of ints once.
        if not isinstance(ttt_num_chunks, int):
            ttt_num_chunks = [int(x) for x in ttt_num_chunks]
        self.ttt_num_chunks = ttt_num_chunks
        self.ttt_view_tour = ttt_view_tour
        # Q1 probe: per-scene-constant random rotary-phase coordinates
        # (see compute_camera_info). Default OFF.
        self.cam_scene_random = cam_scene_random

        self.pose_keys = ["ray_o", "ray_d", "o_cross_d"]
        self.posed_image_keys = self.pose_keys + ["normalized_image"]

        self.input_dim = len(self.posed_image_keys) * 3
        self.input_linear = nn.Linear(self.input_dim * (self.patch_size**2), self.dim, bias=False)
        self.input_layernorm = nn.LayerNorm(self.dim, bias=False)
        self.blocks = nn.ModuleList([
            Block(dim=self.dim, bias=False, block_config=block_config)
            for _ in range(layers)
        ])

        self.image_token_decoder = nn.Sequential(
            nn.LayerNorm(self.dim, bias=False),
            nn.Linear(self.dim, (self.patch_size**2) * 3, bias=False),
            nn.Sigmoid(),
        )

        # apply special scaled init to the residual projections, per GPT-2 paper
        self.apply(_init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(len(block_config) * layers))
    
    def forward(self, input_data_dict, target_data_dict):
            # Do not autocast during the data processing
        with torch.autocast(device_type="cuda", enabled=False), torch.no_grad():
            batch_size, num_input_views, _, h, w = input_data_dict["image"].size()
            num_target_views = target_data_dict["c2w"].size(1)

            if self.ttt_view_tour:
                # Order input views by decreasing pose distance to the target
                # camera centroid: target-adjacent views are written last.
                in_pos = input_data_dict["c2w"][:, :, :3, 3]
                tgt_center = target_data_dict["c2w"][:, :, :3, 3].mean(dim=1, keepdim=True)
                dist = (in_pos - tgt_center).norm(dim=-1)          # [b, v_in]
                perm = dist.argsort(dim=1, descending=True)        # far -> near
                bidx = torch.arange(batch_size, device=perm.device)[:, None]
                input_data_dict = {
                    key: value[bidx, perm] for key, value in input_data_dict.items()
                }

            for data_dict in [input_data_dict, target_data_dict]:
                fxfycxcy = data_dict["fxfycxcy"]
                c2w = data_dict["c2w"]

                # true (canonical-frame) rays: camera_info always uses these
                data_dict["ray_o"], data_dict["ray_d"] = compute_rays(fxfycxcy, c2w, h, w)
                if self.input_raymap == "camray":
                    eye = torch.eye(4, device=c2w.device, dtype=c2w.dtype).expand_as(c2w)
                    tok_o_, tok_d_ = compute_rays(fxfycxcy, eye, h, w)
                    data_dict["tok_ray_o"], data_dict["tok_ray_d"] = tok_o_, tok_d_
                else:
                    data_dict["tok_ray_o"], data_dict["tok_ray_d"] = data_dict["ray_o"], data_dict["ray_d"]
                data_dict["o_cross_d"] = torch.cross(data_dict["tok_ray_o"], data_dict["tok_ray_d"], dim=2)
                data_dict["pose_only"] = torch.concat(
                    [data_dict[key] for key in ("tok_ray_o", "tok_ray_d", "o_cross_d")], dim=2
                )

                if "image" in data_dict:
                    data_dict["normalized_image"] = data_dict["image"] * 2.0 - 1.0

                    # Compile the information for posed-image input, and pose-only input.
                    # (token rays, so that "camray" tokens are pose-free for inputs too)
                    data_dict["posed_image"] = torch.concat(
                        [data_dict[key] for key in
                         ("tok_ray_o", "tok_ray_d", "o_cross_d", "normalized_image")], dim=2
                    )

            transformer_input = input_data_dict["image"].new_zeros(
                batch_size, num_input_views + num_target_views, self.input_dim, h, w
            )
            transformer_input[:, :num_input_views, :, :, :] = input_data_dict["posed_image"]
            pose_only_dim = target_data_dict["pose_only"].size(2)
            transformer_input[:, num_input_views:, :pose_only_dim, :, :] = target_data_dict["pose_only"]

            # Camera tensors for camera-conditioned TTT layers (all views,
            # token order matches the patchify rearrange below).
            all_fxfycxcy = torch.cat([input_data_dict["fxfycxcy"], target_data_dict["fxfycxcy"]], dim=1)
            all_c2w = torch.cat([input_data_dict["c2w"], target_data_dict["c2w"]], dim=1)
            all_ray_o = torch.cat([input_data_dict["ray_o"], target_data_dict["ray_o"]], dim=1)
            all_ray_d = torch.cat([input_data_dict["ray_d"], target_data_dict["ray_d"]], dim=1)
            tok_t_gt = None
            if "depth_t" in input_data_dict and "depth_t" in target_data_dict:
                # [b, v, hh, ww] patch-grid GT ray parameters (oracle diagnostics)
                tok_t_gt = torch.cat([input_data_dict["depth_t"], target_data_dict["depth_t"]], dim=1)
            camera_info = compute_camera_info(
                all_fxfycxcy, all_c2w, h, w, self.patch_size,
                all_ray_o, all_ray_d, num_input_views,
                cam_scene_random=self.cam_scene_random, tok_t_gt=tok_t_gt,
                focus_mode=self.focus_mode,
            )

        # Running the model
        num_img_tokens = h * w // (self.patch_size**2)
        num_input_tokens = num_input_views * num_img_tokens
        num_target_tokens = num_target_views * num_img_tokens
        if self.ttt_chunk_per_view:
            # One inner-loop gradient step per input view (camera-scheduled).
            ttt_op_order = [
                TTTOperator(start=v * num_img_tokens, end=(v + 1) * num_img_tokens,
                            update=True, apply=False)
                for v in range(num_input_views)
            ] + [
                TTTOperator(start=0, end=num_input_tokens + num_target_tokens,
                            update=False, apply=True),
            ]
        elif (isinstance(self.ttt_num_chunks, (list, tuple))
              or self.ttt_num_chunks > 1):
            # n sequential update steps over equal spans of the input tokens, then one
            # apply over everything. Chunk size stays num_input_tokens / n, so keep n
            # small enough that it stays above Muon's ~427-token amortisation point
            # (F8: 256-token chunks lose 0.23 dB for that reason alone).
            n = self.ttt_num_chunks
            if isinstance(n, (list, tuple)):
                # Multi-chunk training: one n drawn per forward, so a single model
                # learns to update under every chunk count. Eval passes a fixed int
                # and takes the first entry if a list somehow reaches it.
                #
                # NOTE this draw is only safe because the NVS ablations run on ONE
                # GPU (launch_exp.sh uses --nproc_per_node=1). Under DDP each rank
                # would draw its own n, build a different ttt_op_order, and therefore
                # a different graph; averaging those gradients does not correspond to
                # any single loss, and mismatched collectives can deadlock. If this is
                # ever run multi-GPU, draw on rank 0 and broadcast.
                n = int(n[torch.randint(len(n), (1,)).item()]) if self.training else int(n[0])
            assert num_input_tokens % n == 0, \
                f"{num_input_tokens} input tokens do not divide into {n} chunks"
            step = num_input_tokens // n
            ttt_op_order = [
                TTTOperator(start=i * step, end=(i + 1) * step, update=True, apply=False)
                for i in range(n)
            ] + [
                TTTOperator(start=0, end=num_input_tokens + num_target_tokens,
                            update=False, apply=True),
            ]
        else:
            ttt_op_order = [
                TTTOperator(start=0, end=num_input_tokens, update=True, apply=False),
                TTTOperator(start=0, end=num_input_tokens + num_target_tokens, update=False, apply=True),
            ]
        info = {
            "ttt_op_order": ttt_op_order,
            "num_img_tokens": num_img_tokens,
        }
        info.update(camera_info)

        x = rearrange(
            transformer_input,
            "b v c (hh ph) (ww pw) -> b (v hh ww) (ph pw c)",
            ph=self.patch_size,
            pw=self.patch_size,
        )
        x = self.input_linear(x)
        x = self.input_layernorm(x)
        for block in self.blocks:
            x, _ = block(x, info)
        
        target_x = x[:, -num_target_tokens:]
        target_x = self.image_token_decoder(target_x)
        target_x = rearrange(
            target_x,
            "b (v hh ww) (ph pw c) -> b v c (hh ph) (ww pw)",
            v=num_target_views,
            hh=h // self.patch_size,
            ww=w // self.patch_size,
            ph=self.patch_size,
            pw=self.patch_size,
            c=3,
        )
        return target_x
    
    def reconstruct(self, input_data_dict):
        with torch.autocast(device_type="cuda", enabled=False), torch.no_grad():
            batch_size, num_input_views, _, h, w = input_data_dict["image"].size()

            fxfycxcy = input_data_dict["fxfycxcy"]
            c2w = input_data_dict["c2w"]

            assert self.input_raymap == "world", "reconstruct(): camray tokens need forward()"
            input_data_dict["ray_o"], input_data_dict["ray_d"] = compute_rays(fxfycxcy, c2w, h, w)
            input_data_dict["o_cross_d"] = torch.cross(input_data_dict["ray_o"], input_data_dict["ray_d"], dim=2)
            input_data_dict["pose_only"] = torch.concat(
                [input_data_dict[key] for key in self.pose_keys], dim=2
            )

            input_data_dict["normalized_image"] = input_data_dict["image"] * 2.0 - 1.0

            # Compile the information for posed-image input, and pose-only input.
            posed_image = torch.concat(
                [input_data_dict[key] for key in self.posed_image_keys], dim=2
            )
            
        # Running the model
        num_img_tokens = h * w // (self.patch_size**2)
        num_input_tokens = num_input_views * num_img_tokens
        ttt_op_order = [
            TTTOperator(start=0, end=num_input_tokens, update=True, apply=True),
        ]
        info = {
            "ttt_op_order": ttt_op_order,
            "num_img_tokens": num_img_tokens,
        }

        x = rearrange(
            posed_image,
            "b v c (hh ph) (ww pw) -> b (v hh ww) (ph pw c)",
            ph=self.patch_size,
            pw=self.patch_size,
        )
        x = self.input_linear(x)
        x = self.input_layernorm(x)
        states = []
        for block in self.blocks:
            x, state = block(x, info)
            states.append(state)
        return states
    
    def rendering(self, target_data_dict, states, h, w):
        with torch.autocast(device_type="cuda", enabled=False):
            batch_size, num_target_views, _, _ = target_data_dict["c2w"].size()

            fxfycxcy = target_data_dict["fxfycxcy"]
            c2w = target_data_dict["c2w"]

            target_data_dict["ray_o"], target_data_dict["ray_d"] = compute_rays(fxfycxcy, c2w, h, w)
            target_data_dict["o_cross_d"] = torch.cross(target_data_dict["ray_o"], target_data_dict["ray_d"], dim=2)
            target_data_dict["pose_only"] = torch.concat(
                [target_data_dict[key] for key in self.pose_keys], dim=2
            )

            pose_only = target_data_dict["pose_only"].new_zeros(
                batch_size, num_target_views, self.input_dim, h, w
            )  
            pose_only_dim = target_data_dict["pose_only"].size(2)
            pose_only[:, :, :pose_only_dim, :, :] = target_data_dict["pose_only"]
            
        # Running the model for rendering
        num_img_tokens = h * w // (self.patch_size**2)
        num_target_tokens = num_target_views * num_img_tokens
        ttt_op_order = [
            TTTOperator(start=0, end=num_target_tokens, update=False, apply=True),
        ]
        info = {
            "ttt_op_order": ttt_op_order,
            "num_img_tokens": num_img_tokens,
        }

        # Process each target view separately
        all_x = []
        for v in range(num_target_views):
            single_view_pose = pose_only[:, v:v+1]  # b, 1, c, h, w
            
            x = rearrange(
                single_view_pose,
                "b v c (hh ph) (ww pw) -> b (v hh ww) (ph pw c)",
                ph=self.patch_size,
                pw=self.patch_size,
            )
            x = self.input_linear(x)
            x = self.input_layernorm(x)
            
            # Apply the saved states from reconstruction
            for block, state in zip(self.blocks, states):
                info.update(state)
                x, _ = block(x, info)
            
            all_x.append(x)
        
        # Concatenate all processed views
        x = torch.cat(all_x, dim=1)
            
        # Generate target images
        target_x = self.image_token_decoder(x)
        target_x = rearrange(
            target_x,
            "b (v hh ww) (ph pw c) -> b v c (hh ph) (ww pw)",
            v=num_target_views,
            hh=h // self.patch_size,
            ww=w // self.patch_size,
            ph=self.patch_size,
            pw=self.patch_size,
            c=3,
        )
        
        return target_x

    