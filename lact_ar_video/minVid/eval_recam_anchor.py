"""External quality anchor: released ReCamMaster-Wan2.1 (step20000.ckpt) on our
ccv held-out pairs (CAMCTRL_DESIGN.md Phase-2 "External anchor").

Runs the OFFICIAL ReCamMaster pipeline (KwaiVGI/ReCamMaster @ github, DiffSynth
WanVideoReCamMasterPipeline) with THEIR defaults (50 steps, sigma_shift 5.0,
CFG 5.0 with their Chinese negative prompt, seed 0, tiled VAE) on the first
N pairs of our fixed held-out list, then scores the generated target-camera
videos against OUR GT frames with OUR metric code (psnr_per_frame /
ssim_per_frame from eval_ccv_common + the same 16-frame-batched LPIPS-alex as
eval_ccv_generate.compute_metrics).

Faithfulness choices (documented in metrics.json "caveats"):
  - Source video + camera conditioning are prepared with the OFFICIAL code
    paths copied verbatim from their train/inference scripts (imageio decode,
    cover-resize + center-crop to 480x832, [-1,1]; UE-matrix parsing,
    transpose, UE->CV axis remap; per-latent-frame relative extrinsics
    rel_i = w2c_cond[frame0] @ c2w_tgt[frame 4i], [:3,:4] flattened -> [21,12]).
    Training convention (train_recammaster.py TensorDataset): the reference is
    the CONDITION camera's frame-0 pose -- our src_cam.
  - MCV ships no captions; every pair uses our fixed ccv caption (the same
    string all our ccv models trained/evaluated with).
  - GT frames for the metrics come from OUR loader (MultiCamPairDataset,
    decord decode, same cover-resize + center-crop) so the GT pixels are
    IDENTICAL to those every eval_ccv_generate run was scored against.

Usage (recam venv, from lact_ar_video/minVid):
  PYTHONPATH=<repo>/lact_ar_video:/...datasets/recam_ckpt/ReCamMaster \
  .../envs/recam/bin/python eval_recam_anchor.py --n_pairs 8 \
      --out ../outputs/eval_dev/gen_recam_anchor
"""
import argparse
import json
import os
import time

import eval_ccv_common as common  # noqa: F401  (sets env before torch use)
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from eval_ccv_common import (
    DEFAULT_PAIRS_JSON,
    load_config,
    make_pair_dataset,
    psnr_per_frame,
    ssim_per_frame,
)

RECAM_REPO = "/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/datasets/recam_ckpt/ReCamMaster"
RECAM_CKPT = "/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/datasets/recam_ckpt/step20000.ckpt"
WAN_DIR = "/NHNHOME/WORKSPACE/26msit001_A/jinhyeok/datasets/wan_ckpt"

# their inference_recammaster.py defaults
RECAM_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，"
    "低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"
    "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


# ---------------------------------------------------------------------------
# OFFICIAL ReCamMaster input preparation (copied from their repo)
# ---------------------------------------------------------------------------

class Camera(object):
    """Verbatim from ReCamMaster train/inference scripts."""

    def __init__(self, c2w):
        c2w_mat = np.array(c2w).reshape(4, 4)
        self.c2w_mat = c2w_mat
        self.w2c_mat = np.linalg.inv(c2w_mat)


def parse_matrix(matrix_str):
    """Verbatim from ReCamMaster (raw UE row-major 4x4 string -> np [4,4])."""
    rows = matrix_str.strip().split('] [')
    matrix = []
    for row in rows:
        row = row.replace('[', '').replace(']', '')
        matrix.append(list(map(float, row.split())))
    return np.array(matrix)


def get_relative_pose(cam_params):
    """Verbatim from ReCamMaster train_recammaster.py (identity reference)."""
    abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
    abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
    target_cam_c2w = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
    abs2rel = target_cam_c2w @ abs_w2cs[0]
    ret_poses = [target_cam_c2w, ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
    ret_poses = np.array(ret_poses, dtype=np.float32)
    return ret_poses


def load_mcv_c2ws(cam_json_path, view_idx, num_frames=81, stride=4):
    """Official parsing of MCV camera_extrinsics.json for one camera:
    parse -> transpose -> UE->CV remap (train_recammaster.py lines 238-248)."""
    with open(cam_json_path, "r") as f:
        cam_data = json.load(f)
    cam_idx = list(range(num_frames))[::stride]  # 0,4,...,80 -> 21
    traj = [parse_matrix(cam_data[f"frame{idx}"][f"cam{view_idx:02d}"])
            for idx in cam_idx]
    traj = np.stack(traj).transpose(0, 2, 1)
    c2ws = []
    for c2w in traj:
        c2w = c2w[:, [1, 2, 0, 3]]
        c2w[:3, 1] *= -1.0
        c2w[:3, 3] /= 100
        c2ws.append(c2w)
    return c2ws  # list of 21 np [4,4]


def build_cam_embedding(cam_json_path, cond_cam, tgt_cam):
    """Training-convention camera conditioning: target trajectory relative to
    the CONDITION camera's frame-0 pose -> [21, 12] (their pose_embedding)."""
    from einops import rearrange

    cond_c2ws = load_mcv_c2ws(cam_json_path, cond_cam)
    tgt_c2ws = load_mcv_c2ws(cam_json_path, tgt_cam)
    cond_cam_params = [Camera(c) for c in cond_c2ws]
    tgt_cam_params = [Camera(c) for c in tgt_c2ws]
    relative_poses = []
    for i in range(len(tgt_cam_params)):
        relative_pose = get_relative_pose([cond_cam_params[0], tgt_cam_params[i]])
        relative_poses.append(torch.as_tensor(relative_pose)[:, :3, :][1])
    pose_embedding = torch.stack(relative_poses, dim=0)  # 21x3x4
    pose_embedding = rearrange(pose_embedding, 'b c d -> b (c d)')
    return pose_embedding.to(torch.bfloat16)  # [21, 12]


def load_source_video_official(path, height=480, width=832, num_frames=81):
    """Their TextVideoCameraDataset loading: imageio decode, cover-resize
    (PIL bilinear), CenterCrop+Resize, normalize to [-1,1] -> [C,F,H,W]."""
    import imageio
    import torchvision
    from torchvision.transforms import v2
    from einops import rearrange

    frame_process = v2.Compose([
        v2.CenterCrop(size=(height, width)),
        v2.Resize(size=(height, width), antialias=True),
        v2.ToTensor(),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    def crop_and_resize(image):
        w, h = image.size
        scale = max(width / w, height / h)
        return torchvision.transforms.functional.resize(
            image, (round(h * scale), round(w * scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR)

    reader = imageio.get_reader(path)
    frames = []
    for frame_id in range(num_frames):
        frame = Image.fromarray(reader.get_data(frame_id))
        frame = crop_and_resize(frame)
        frames.append(frame_process(frame))
    reader.close()
    frames = torch.stack(frames, dim=0)
    return rearrange(frames, "T C H W -> C T H W")


# ---------------------------------------------------------------------------
# pipeline construction (official inference_recammaster.py steps 1-3)
# ---------------------------------------------------------------------------

def build_recam_pipeline(device="cuda"):
    from diffsynth import ModelManager, WanVideoReCamMasterPipeline

    model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    model_manager.load_models([
        os.path.join(WAN_DIR, "diffusion_pytorch_model.safetensors"),
        os.path.join(WAN_DIR, "models_t5_umt5-xxl-enc-bf16.pth"),
        os.path.join(WAN_DIR, "Wan2.1_VAE.pth"),
    ])
    pipe = WanVideoReCamMasterPipeline.from_model_manager(model_manager, device=device)

    dim = pipe.dit.blocks[0].self_attn.q.weight.shape[0]
    for block in pipe.dit.blocks:
        block.cam_encoder = nn.Linear(12, dim)
        block.projector = nn.Linear(dim, dim)
        block.cam_encoder.weight.data.zero_()
        block.cam_encoder.bias.data.zero_()
        block.projector.weight = nn.Parameter(torch.eye(dim))
        block.projector.bias = nn.Parameter(torch.zeros(dim))

    state_dict = torch.load(RECAM_CKPT, map_location="cpu", weights_only=True)
    pipe.dit.load_state_dict(state_dict, strict=True)
    pipe.to(device)
    pipe.to(dtype=torch.bfloat16)
    return pipe


# ---------------------------------------------------------------------------
# metrics (identical to eval_ccv_generate.compute_metrics)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_metrics(gen_frames, gt_frames, lpips_model, device):
    """gen/gt: [F, C, H, W] float in [0, 1] on device. Returns dict."""
    psnr = psnr_per_frame(gen_frames, gt_frames)
    ssim = ssim_per_frame(gen_frames, gt_frames)
    lp = []
    for s in range(0, gen_frames.shape[0], 16):
        a = gen_frames[s: s + 16].float() * 2.0 - 1.0
        b = gt_frames[s: s + 16].float() * 2.0 - 1.0
        lp.append(lpips_model(a, b).flatten())
    lp = torch.cat(lp)
    return {
        "psnr_mean": psnr.mean().item(),
        "ssim_mean": ssim.mean().item(),
        "lpips_mean": lp.mean().item(),
        "psnr_per_frame": [round(v, 4) for v in psnr.tolist()],
        "ssim_per_frame": [round(v, 5) for v in ssim.tolist()],
        "lpips_per_frame": [round(v, 5) for v in lp.tolist()],
    }


def to_uint8_video(frames):
    """[F, C, H, W] in [0,1] -> uint8 [F, C, H, W] on cpu."""
    return (frames.clamp(0, 1) * 255).round().to(torch.uint8).cpu()


def main():
    parser = argparse.ArgumentParser(description="ReCamMaster external anchor eval")
    parser.add_argument("--config", type=str, default="configs/ar/abl_ccv_base.yaml",
                        help="ccv config (only for the GT dataset params + caption)")
    parser.add_argument("--pairs", type=str, default=DEFAULT_PAIRS_JSON)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--n_pairs", type=int, default=8)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50, help="official default")
    parser.add_argument("--cfg_scale", type=float, default=5.0, help="official default")
    parser.add_argument("--seed", type=int, default=0, help="official default")
    parser.add_argument("--save_videos", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    config = load_config(args.config)
    caption = config.dataset_train.params.caption

    with open(args.pairs, "r") as f:
        payload = json.load(f)
    pairs = payload["pairs"][args.start: args.start + args.n_pairs]
    data_root = payload["data_root"]
    print(f"[recam] {len(pairs)} pairs from {args.pairs} (start {args.start})")

    pipe = build_recam_pipeline(device=args.device)
    print("[recam] pipeline ready (Wan2.1-T2V-1.3B + ReCamMaster step20000.ckpt)")

    import lpips
    lpips_model = lpips.LPIPS(net="alex").to(args.device).eval()

    # OUR GT loader (identical pixels to every eval_ccv_generate run)
    ds = make_pair_dataset(config, pairs)
    os.makedirs(args.out, exist_ok=True)

    results = []
    for i, pair in enumerate(pairs):
        rec = {
            "index": args.start + i,
            "relpath": pair["relpath"],
            "src_cam": pair["src_cam"],
            "tgt_cam": pair["tgt_cam"],
            "seed": args.seed,
        }
        print(f"[recam] pair {i}/{len(pairs)}: {rec['relpath']} "
              f"cams({rec['src_cam']},{rec['tgt_cam']})", flush=True)

        src_mp4 = os.path.join(pair["videos_dir"], f"cam{pair['src_cam']:02d}.mp4")
        cam_json = os.path.join(data_root, pair["relpath"], "cameras",
                                "camera_extrinsics.json")

        source_video = load_source_video_official(src_mp4)[None]  # [1,C,81,H,W]
        target_camera = build_cam_embedding(
            cam_json, cond_cam=pair["src_cam"], tgt_cam=pair["tgt_cam"])[None]

        tic = time.time()
        frames = pipe(
            prompt=[caption],
            negative_prompt=RECAM_NEGATIVE_PROMPT,
            source_video=source_video,
            target_camera=target_camera,
            cfg_scale=args.cfg_scale,
            num_inference_steps=args.steps,
            seed=args.seed, tiled=True,
        )  # list of 81 PIL images
        gen_seconds = time.time() - tic

        gen = torch.stack(
            [torch.from_numpy(np.array(fr)) for fr in frames]
        ).float().div_(255.0).permute(0, 3, 1, 2).to(args.device)  # [F,C,H,W]

        item = ds[i]
        gt_frames = item["frames_tgt"].permute(1, 0, 2, 3).to(args.device)  # [F,C,H,W]

        n_f = min(gen.shape[0], gt_frames.shape[0])
        if gen.shape[0] != gt_frames.shape[0]:
            rec["frame_count_mismatch"] = [gen.shape[0], gt_frames.shape[0]]
        gen, gt_frames = gen[:n_f], gt_frames[:n_f]
        assert gen.shape == gt_frames.shape, (gen.shape, gt_frames.shape)

        rec.update(compute_metrics(gen, gt_frames, lpips_model, args.device))
        rec["gen_seconds"] = gen_seconds
        results.append(rec)
        print(f"[recam]   PSNR {rec['psnr_mean']:.3f} dB  "
              f"SSIM {rec['ssim_mean']:.4f}  LPIPS {rec['lpips_mean']:.4f}  "
              f"({gen_seconds:.0f}s)", flush=True)

        if args.save_videos:
            from minVid.utils.io_utils import save_video
            tag = (f"pair{rec['index']:03d}_{rec['relpath'].replace(os.sep, '_')}"
                   f"_cam{rec['src_cam']:02d}to{rec['tgt_cam']:02d}")
            gen_u8 = to_uint8_video(gen)
            gt_u8 = to_uint8_video(gt_frames)
            src_u8 = to_uint8_video(
                item["frames_src"].permute(1, 0, 2, 3)[:n_f])
            save_video(gen_u8, os.path.join(args.out, f"{tag}_gen.mp4"), save_fps=15)
            sxs = torch.cat([src_u8, gen_u8, gt_u8], dim=3)
            save_video(sxs, os.path.join(args.out, f"{tag}_src_gen_gt.mp4"),
                       save_fps=15)

        # partial-summary flush so an interrupted run keeps its results
        _write_summary(args, results)

    _write_summary(args, results)
    s = results
    print(f"[recam] SUMMARY over {len(s)} pairs: "
          f"PSNR {sum(r['psnr_mean'] for r in s)/len(s):.3f} dB, "
          f"SSIM {sum(r['ssim_mean'] for r in s)/len(s):.4f}, "
          f"LPIPS {sum(r['lpips_mean'] for r in s)/len(s):.4f}")


def _write_summary(args, results):
    if not results:
        return
    summary = {
        "model": "ReCamMaster-Wan2.1 released checkpoint (external anchor)",
        "recam_repo": RECAM_REPO,
        "recam_ckpt": RECAM_CKPT,
        "wan_base": WAN_DIR,
        "pairs_file": os.path.abspath(args.pairs),
        "sampler": {"type": "official_recammaster_flowmatch",
                    "steps": args.steps, "sigma_shift": 5.0,
                    "cfg_scale": args.cfg_scale, "seed": args.seed,
                    "negative_prompt": "official (Chinese)",
                    "tiled_vae": True},
        "caveats": [
            "prompt: MCV has no captions; used our fixed ccv caption for all pairs",
            "camera conditioning: official training convention "
            "(rel to CONDITION cam frame-0), official UE parsing on raw "
            "camera_extrinsics.json (no mean-pose normalization)",
            "source video: official imageio/PIL cover-resize+center-crop to "
            "480x832 (ours uses decord; same geometry, sub-pixel resample "
            "differences only affect conditioning, GT is from OUR loader)",
            "sampler: their 50-step CFG-5 flow matching vs our 40-step "
            "CFG-off Euler -- intentionally their best-foot-forward defaults",
            "81 frames @ 480x832, same as our protocol; no resolution or "
            "frame-count mismatch",
        ],
        "n_pairs": len(results),
        "psnr_mean": sum(r["psnr_mean"] for r in results) / len(results),
        "ssim_mean": sum(r["ssim_mean"] for r in results) / len(results),
        "lpips_mean": sum(r["lpips_mean"] for r in results) / len(results),
        "per_pair": results,
    }
    out_json = os.path.join(args.out, "metrics.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
