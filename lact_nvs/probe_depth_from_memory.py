"""Zero-training probe: can the fast-weight memory tell a TARGET token where along its ray to read?

On a trained hidden-chord checkpoint (orbit renders, GT depth available), for every TTT layer:
  (1) MASS PROFILE (plane sweep against the pre-summed memory):
        m = sum_i lr1_i h~_i            (512-vector, the "1-column" of the Hebbian update)
        Z_j(t_k) = < R(theta(x_j(t_k))) h_j , m >   for K depths t_k on the target's chord
      -> t_sweep = argmax_k Z_j(t_k)  (also soft-argmax)
  (2) HEBBIAN LEAST-SQUARES RAY TRIANGULATION:
        C = sum_i lr1_i h~_i (x) [vec(A_i) (6), b_i (3), 1],  A_i = I - d_i d_i^T, b_i = A_i o_i
        read with the target's own chord-coded address: [A^, b^, N] = h~_j C
        x^ = (A^/N + lam I)^-1 (b^/N + lam x_c,j),  t_ls = (x^ - o_j) . d_j
  compared with the depth-free foot t_c and GT t (from the EXR side files), on target tokens with a
  surface. Reports per layer: median |t - t_gt| for foot / sweep / LS, Pearson corr, win rates.

Usage (from lact_nvs):
  python probe_depth_from_memory.py --config config/gobj_shell_h.yaml \
     --ckpt outputs/gobj_shell_h_s95/model_0030000.pth --n_scenes 32 --K 32
"""
import argparse
import json

import omegaconf
import torch
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader

from data_re10k import Re10KDataset
from lact_ttt_cam import (CamFastWeightGluMLPMultihead, apply_rotary_pairs,
                          fast_weight_swish_glu_hidden_rotary_apply, to_heads)
from model import LaCTLVSM

DD = "/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/dataset/gobj_depth_patch/test"

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--data", default="/tmp/gobj/test_index.json")
ap.add_argument("--n_scenes", type=int, default=32)
ap.add_argument("--K", type=int, default=32)
ap.add_argument("--lam", type=float, default=0.3)
ap.add_argument("--out", default=None)
args = ap.parse_args()

cfg = omegaconf.OmegaConf.load(args.config)
model = LaCTLVSM(**cfg).cuda().eval()
sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
model.load_state_dict(sd["model"] if "model" in sd else sd)

ds = Re10KDataset(args.data, num_views=12, image_size=(256, 256), scene_pose_normalize=True,
                  window=128, min_frames=40, eval_mode=True, num_input_views=8, num_target_views=4,
                  max_scenes=args.n_scenes, depth_dir=DD)
loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=4)

# capture (x, info) at every TTT layer
records = []
_orig = CamFastWeightGluMLPMultihead.forward
def _hooked(self, x, info={}, *a):
    records.append((self, x.detach(), dict(info)))
    return _orig(self, x, info, *a)
CamFastWeightGluMLPMultihead.forward = _hooked

stats = {}
def acc(layer, key, val):
    stats.setdefault(layer, {}).setdefault(key, []).append(val.detach().float().cpu())

with torch.no_grad():
    for bi, data in enumerate(loader):
        data = {k: v.cuda() for k, v in data.items()}
        inp = {k: v[:, :8] for k, v in data.items()}
        tgt = {k: v[:, 8:] for k, v in data.items()}
        records.clear()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(inp, tgt)
        for li, (mod, x, info) in enumerate(records):
            assert "h_shell" in mod.cam_modes or "h_foot" in mod.cam_modes, mod.cam_modes
            nh = mod.num_heads; tpv = info["tokens_per_view"]
            ops = info["ttt_op_order"]; n_in, n_all = ops[0].end, ops[1].end
            with torch.autocast("cuda", enabled=False):
                x = x.float()
                qkv = F.silu(mod.to_qkv(x))
                q, k, v = rearrange(qkv, "b l (qkv h d) -> qkv (b h) l d", qkv=3, h=nh)
                q = q / (q.norm(dim=2, keepdim=True) + 1e-5)
                k = k / (k.norm(dim=2, keepdim=True) + 1e-5)
                lr = F.softplus(mod.lr_fc(x) + mod.base_lr_inv)
                lr0, lr1, lr2 = rearrange(lr, "b l (lrs h d) -> lrs (b h) l d", lrs=3, h=nh)
                w0 = mod.w0.repeat(x.shape[0], 1, 1); w1 = mod.w1.repeat(x.shape[0], 1, 1); w2 = mod.w2.repeat(x.shape[0], 1, 1)
                hcos, hsin = mod._point_site_coeffs(info, "h")          # chord code, [(b h), L, P]
                _, w0u, w1u, w2u = fast_weight_swish_glu_hidden_rotary_apply(
                    w0, w1, w2, q, k, v, lr0, lr1, lr2, hcos, hsin, ops, muon_update_steps=mod.muon_update_steps)
                # stored addresses (update side uses the initial weights)
                k_in = k[:, :n_in]
                h_in = F.silu(k_in @ w0) * (k_in @ w2)
                ht_in = apply_rotary_pairs(h_in, hcos[:, :n_in], hsin[:, :n_in])       # [B, n_in, dh]
                lr_in = lr1[:, :n_in]                                                  # [B, n_in, 1]
                m = (lr_in * ht_in).sum(1)                                             # [B, dh]
                # Hebbian LS columns
                o_i = info["tok_o"][:, :n_in].float(); d_i = info["tok_d"][:, :n_in].float()
                eye = torch.eye(3, device=x.device)
                A_i = eye - d_i[..., :, None] * d_i[..., None, :]                      # [b, n_in, 3, 3]
                b_i = torch.einsum("blij,blj->bli", A_i, o_i)
                iu = [(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]
                g_i = torch.cat([torch.stack([A_i[..., a, c] for a, c in iu], -1), b_i,
                                 torch.ones_like(b_i[..., :1])], -1)                   # [b, n_in, 10]
                g_i = to_heads(g_i, nh)
                C = torch.einsum("bld,blg->bdg", lr_in * ht_in, g_i)                   # [B, dh, 10]
                # query side: updated weights, target tokens
                q_t = q[:, n_in:n_all]
                hq = F.silu(q_t @ w0u) * (q_t @ w2u)                                   # [B, n_tgt, dh]
                # (a) LS triangulation at the model's own (chord) address
                ht_q = apply_rotary_pairs(hq, hcos[:, n_in:n_all], hsin[:, n_in:n_all])
                r = torch.einsum("bld,bdg->blg", ht_q, C)                              # [B, n_tgt, 10]
                N = r[..., 9:10]
                Ah = torch.zeros(*r.shape[:2], 3, 3, device=x.device)
                for gi, (a, c) in enumerate(iu):
                    Ah[..., a, c] = r[..., gi]; Ah[..., c, a] = r[..., gi]
                bh = r[..., 6:9]
                Nn = N.abs().clamp_min(1e-6) * torch.sign(N + 1e-12)
                o_j = info["tok_o"][:, n_in:n_all].float(); d_j = info["tok_d"][:, n_in:n_all].float()
                tc = info["tok_tc"][:, n_in:n_all].float()                              # [b, n_tgt, 1]
                xc = o_j + tc * d_j
                Areg = Ah / Nn[..., None] + args.lam * eye
                breg = bh / Nn + args.lam * xc
                xh = torch.linalg.solve(Areg, breg[..., None])[..., 0]
                t_ls = ((xh - o_j) * d_j).sum(-1, keepdim=True)
                # (b) mass-profile sweep along the chord
                t1, t2 = mod._chord_t(info)
                t1, t2 = t1[:, n_in:n_all], t2[:, n_in:n_all]
                Z = []
                ts = []
                for kk in range(args.K):
                    fr = (kk + 0.5) / args.K
                    ta = t1 + fr * (t2 - t1)
                    t_full = torch.cat([info["tok_tc"][:, :n_in].float(), ta], 1)      # [b, L, 1]
                    c_k, s_k = mod._seg_dirs_coeffs(info, t_full, t_full,
                                                    mod.dirs_h, mod.omega_hseg, mod.gain_hseg)
                    c_k, s_k = to_heads(c_k, nh)[:, n_in:n_all], to_heads(s_k, nh)[:, n_in:n_all]
                    htk = apply_rotary_pairs(hq, c_k, s_k)
                    Z.append(torch.einsum("bld,bd->bl", htk, m))
                    ts.append(ta[..., 0])
                Z = torch.stack(Z, -1); ts = torch.stack(ts, -1)                       # [B, n_tgt, K]
                t_arg = ts.gather(-1, Z.argmax(-1, keepdim=True))[..., 0]
                p = torch.softmax(Z / (Z.std(-1, keepdim=True) + 1e-6), -1)
                t_soft = (p * ts).sum(-1)
                # GT
                tg = info["tok_t_gt"][:, n_in:n_all, 0].float()
                valid = tg > 0
                for name, est in (("foot", tc[..., 0]), ("ls", t_ls[..., 0]), ("sweep_arg", t_arg), ("sweep_soft", t_soft)):
                    err = (est - tg).abs()[valid]
                    acc(li, name + "_err", err)
                    acc(li, name + "_est", est[valid]);
                acc(li, "gt", tg[valid])
                acc(li, "chord_half", (0.5 * (t2 - t1))[..., 0][valid])
        print(f"batch {bi} done", flush=True)

CamFastWeightGluMLPMultihead.forward = _orig
out = {}
for li in sorted(stats):
    s = {k: torch.cat(v) for k, v in stats[li].items()}
    gt = s["gt"]; row = {"n": int(gt.numel()), "chord_half_med": float(s["chord_half"].median())}
    for name in ("foot", "ls", "sweep_arg", "sweep_soft"):
        e = s[name + "_err"]; est = s[name + "_est"]
        corr = float(torch.corrcoef(torch.stack([est, gt]))[0, 1])
        row[name] = {"med_abs_err": float(e.median()), "mean_abs_err": float(e.mean()), "corr": corr,
                     "beats_foot": float((e < s["foot_err"]).float().mean()) if name != "foot" else None}
    out[li] = row
    print(f"layer {li}: n={row['n']} chord_half={row['chord_half_med']:.3f} | " +
          " | ".join(f"{k}: med {row[k]['med_abs_err']:.3f} corr {row[k]['corr']:.2f}" +
                     (f" beats_foot {row[k]['beats_foot']:.2f}" if row[k]['beats_foot'] is not None else "")
                     for k in ("foot", "ls", "sweep_arg", "sweep_soft")))
if args.out:
    json.dump(out, open(args.out, "w"), indent=1)
