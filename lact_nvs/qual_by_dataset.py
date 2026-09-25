"""Per-dataset qualitative figures (user, 2026-09-25): one figure per dataset, 10 scenes each.

Selection (--select):
  image_psnr (default, user 2026-09-25): rank every held-out TARGET IMAGE by CaPET's PSNR lead over ONE
      baseline (--against, default RayRoPE) on that image (per_scene_per_view_psnr of the eval.json
      files), keep at most one image per scene, show the top n.
  scene_ssim (the first version): rank scenes by the per-scene (4-target mean) SSIM lead over the best
      baseline, then show the target with the largest SSIM lead measured on the renders.
Columns: 4 of the 8 input views (inputs 0, 2, 4, 6) at half size in a 2x2 block, then No Encoding,
GTA, PRoPE, RayRoPE, CaPET, GT. No metric insets.
Alignment check: the renders' per-scene SSIM is compared with eval.json and printed; the dataset is
built exactly as in eval.py (eval_mode, same window / min_frames / num_scenes), so scene i is scene i.

  python qual_by_dataset.py --dataset re10k|dl3dvu|gobj [--n 10] [--out ../paper_overleaf/figs/qual_re10k.pdf]
"""
import argparse
import os

import numpy as np
import omegaconf
import torch
import torch.nn.functional as F
from PIL import Image

from data_re10k import Re10KDataset
from model import LaCTLVSM

D = "/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/dataset/reshard"
METHODS = [("No Encoding", "config/lact_l6_d256_p16.yaml", "base_s137_re"),
           ("GTA", "config/cam_gta_in.yaml", "{p}_gta_s137"),
           ("PRoPE", "config/cam_prope_orig.yaml", "{p}_prope_s137"),
           ("RayRoPE", "config/rayrope_ttt_world.yaml", "{p}_rayropew_s137"),
           ("CaPET (ours)", "config/dp_lin_pdir_vo_both.yaml", "{p}_dplin_pdir_vo_s137")]
DATA = {   # eval.py settings of each dataset's standard run (run_re10k.sh / run_dl3dv.sh / run_gobj.sh)
    "re10k": dict(prefix="re10k", base="base_s137_re", data_path=f"{D}/re10k/test_index.json",
                  num_scenes=256, min_frames=None, image_size=(256, 256)),
    "dl3dvu": dict(prefix="dl3dvu", base="dl3dvu_base_s137_re", data_path=f"{D}/dl3dv/test_index.json",
                   num_scenes=140, min_frames=None, image_size=(256, 448)),
    "gobj": dict(prefix="gobj", base="gobj_base_s137_re", data_path=f"{D}/gobj/test_index.json",
                 num_scenes=500, min_frames=40, image_size=(256, 256)),
}

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True, choices=list(DATA))
p.add_argument("--n", type=int, default=10)
p.add_argument("--out", required=True)
p.add_argument("--inputs", type=str, default="0,2,4,6", help="which of the 8 input views to show")
p.add_argument("--select", default="image_psnr", choices=["image_psnr", "scene_ssim"])
p.add_argument("--against", default="RayRoPE", help="baseline the image_psnr lead is measured against")
args = p.parse_args()
cfgd = DATA[args.dataset]
exps = [(lab, cfg, cfgd["base"] if "{p}" not in e else e.format(p=cfgd["prefix"])) for lab, cfg, e in METHODS]


def ssim_fn(a, b):   # eval.py's SSIM (11x11 Gaussian, sigma 1.5), [N, C, H, W] in [0, 1] -> [N]
    coords = torch.arange(11, dtype=torch.float32, device=a.device) - 5
    g = torch.exp(-(coords**2) / (2 * 1.5**2))
    g = (g / g.sum()).outer(g / g.sum())
    w = g.expand(a.size(1), 1, 11, 11)
    mu_a = F.conv2d(a, w, groups=a.size(1)); mu_b = F.conv2d(b, w, groups=a.size(1))
    var_a = F.conv2d(a * a, w, groups=a.size(1)) - mu_a**2
    var_b = F.conv2d(b * b, w, groups=a.size(1)) - mu_b**2
    cov = F.conv2d(a * b, w, groups=a.size(1)) - mu_a * mu_b
    c1, c2 = 0.01**2, 0.03**2
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2))
    return s.flatten(1).mean(dim=1)


# ---- selection from the metrics stored in every eval.json
import json
EV = {lab: json.load(open(f"outputs/{e}/eval.json")) for lab, _, e in exps}
ss = {lab: np.array(EV[lab]["per_scene_ssim"]) for lab, _, _ in exps}
pv = {lab: np.array(EV[lab]["per_scene_per_view_psnr"]) for lab, _, _ in exps}      # [scenes, 4]
if args.select == "image_psnr":
    lead_v = pv["CaPET (ours)"] - pv[args.against]              # per image, against one baseline
    best_v = lead_v.argmax(1)                                   # at most one image per scene: its best
    lead = lead_v[np.arange(len(best_v)), best_v]
    scene_ids = [int(i) for i in np.argsort(-lead)[:args.n]]
    tview_sel = [int(best_v[i]) for i in scene_ids]
    print("images (scene, target):", list(zip(scene_ids, tview_sel)))
    print(f"PSNR lead over {args.against} (dB):", " ".join(f"{lead[i]:+.2f}" for i in scene_ids))
else:
    lead = ss["CaPET (ours)"] - np.max(np.stack([ss[l] for l, _, _ in exps[:-1]]), axis=0)
    scene_ids = [int(i) for i in np.argsort(-lead)[:args.n]]
    print("scenes:", scene_ids)
    print("SSIM lead over the best baseline:", " ".join(f"{lead[i]:+.3f}" for i in scene_ids))

# ---- render
n_in, n_tg = 8, 4
ds = Re10KDataset(cfgd["data_path"], num_views=n_in + n_tg, image_size=cfgd["image_size"],
                  scene_pose_normalize=True, pose_norm_mode="mean", window=128,
                  min_frames=cfgd["min_frames"], eval_mode=True, num_input_views=n_in,
                  num_target_views=n_tg, max_scenes=cfgd["num_scenes"])
items = [ds[i] for i in scene_ids]
batch = {k: torch.stack([it[k] for it in items]).cuda() for k in items[0]}
inp = {k: v[:, :n_in] for k, v in batch.items()}
tgt = {k: v[:, n_in:] for k, v in batch.items()}
gt = tgt["image"].float().clamp(0, 1)
S = len(scene_ids)
renders, ssv = {}, {}
for lab, cfg, e in exps:
    model = LaCTLVSM(**omegaconf.OmegaConf.load(cfg)).cuda()
    sd = torch.load(f"outputs/{e}/model_0030000.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(sd["model"] if "model" in sd else sd); model.eval()
    outs = []
    with torch.no_grad(), torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
        for i in range(0, S, 5):   # small chunks keep the wide DL3DV frames within memory
            outs.append(model({k: v[i:i + 5] for k, v in inp.items()},
                              {k: v[i:i + 5] for k, v in tgt.items()}).float().clamp(0, 1))
    r = torch.cat(outs)
    renders[lab] = r.cpu()
    ssv[lab] = ssim_fn(r.flatten(0, 1), gt.flatten(0, 1)).reshape(S, n_tg).cpu()
    err = np.abs(ssv[lab].mean(1).numpy() - ss[lab][scene_ids]).max()
    psv = (-10.0 * torch.log10(((r - gt) ** 2).flatten(2).mean(2))).cpu().numpy()      # [S, 4]
    perr = np.abs(psv - pv[lab][scene_ids]).max()
    print(f"{lab:13s} render-vs-eval.json max |diff|: per-scene SSIM {err:.4f}, per-image PSNR {perr:.3f} dB")
    # a misaligned scene is off by several dB; on white-background objects the MSE is tiny and one image's
    # PSNR moves ~0.4 dB with the bf16 batch composition (Objaverse), so the PSNR bound is loose
    assert err < 0.01 and perr < 1.0, "renders do not match eval.json: scene indices are misaligned"
    del model; torch.cuda.empty_cache()

if args.select == "image_psnr":
    tview = tview_sel
else:
    base_best = torch.stack([ssv[l] for l, _, _ in exps[:-1]]).max(0).values        # [S, n_tg]
    tview = (ssv["CaPET (ours)"] - base_best).argmax(1).tolist()

# ---- compose: one raster grid, column titles drawn by matplotlib
H, W = cfgd["image_size"]
gap, gap_in = 4, 10
show_in = [int(x) for x in args.inputs.split(",")]


def u8(t):
    return (t.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


cols = ["Input"] + [l for l, _, _ in exps] + ["GT"]
cw = W
Wtot = len(cols) * cw + (len(cols) - 1) * gap + (gap_in - gap)
Htot = S * H + (S - 1) * gap
canvas = np.full((Htot, Wtot, 3), 255, np.uint8)
xs = []
x = 0
for ci in range(len(cols)):
    xs.append(x)
    x += cw + (gap_in if ci == 0 else gap)
png_dir = f"outputs/_fig_{args.select}/{args.dataset}"
os.makedirs(png_dir, exist_ok=True)
for si, sid in enumerate(scene_ids):
    y = si * (H + gap)
    # 2x2 block of half-size inputs
    hh, hw = (H - 1) // 2, (W - 1) // 2
    for k, vi in enumerate(show_in[:4]):
        im = Image.fromarray(u8(inp["image"][si, vi].float().clamp(0, 1).cpu())).resize((hw, hh), Image.BICUBIC)
        oy, ox = y + (k // 2) * (H - hh), xs[0] + (k % 2) * (W - hw)
        canvas[oy:oy + hh, ox:ox + hw] = np.asarray(im)
    t = tview[si]
    for ci, lab in enumerate(cols[1:], start=1):
        img = u8(gt[si, t].cpu()) if lab == "GT" else u8(renders[lab][si, t])
        canvas[y:y + H, xs[ci]:xs[ci] + W] = img
        Image.fromarray(img).save(f"{png_dir}/scene{sid:04d}_t{t}_{lab}.png")
    print(f"scene {sid:4d} target {t}: PSNR " + " ".join(f"{l.split()[0]} {pv[l][sid, t]:.2f}" for l, _, _ in exps))

# PDF via PIL, which stores RGB pages as JPEG (the matplotlib route embedded ~8.5 MB of raw pixels).
# Titles are sized to read as ~7 pt when the figure is set at the 5.5 in text width.
from PIL import ImageDraw, ImageFont
import matplotlib.font_manager as fm
fpx = max(12, int(round(Wtot / 5.5 * 7 / 72)))
font = ImageFont.truetype(fm.findfont("DejaVu Sans"), fpx)
top = int(fpx * 1.7)
page = Image.new("RGB", (Wtot, Htot + top), "white")
page.paste(Image.fromarray(canvas), (0, top))
dr = ImageDraw.Draw(page)
for ci, c in enumerate(cols):
    tw = dr.textlength(c, font=font)
    dr.text((xs[ci] + (cw - tw) / 2, (top - fpx) / 2 - fpx * 0.1), c, fill="black", font=font)
page.save(args.out, "PDF", resolution=Wtot / 5.5, quality=92)
print("saved ->", args.out, f"({os.path.getsize(args.out) / 1e6:.1f} MB)")
