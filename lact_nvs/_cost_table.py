"""table:cost re-measure (2026-09-25): params + profiler FLOPs of one eval forward
(8 input + 4 target views, 256x256, batch 1), torch.compile disabled (F98 measurement note)."""
import os, sys, torch, omegaconf
os.environ["TORCH_COMPILE_DISABLE"] = "1"
from model import LaCTLVSM
from torch.profiler import profile, ProfilerActivity
torch.manual_seed(0)
def cams(v):
    th = torch.linspace(0, 1.2, v)
    c2w = torch.eye(4).repeat(v, 1, 1)
    c2w[:, 0, 0], c2w[:, 0, 2], c2w[:, 2, 0], c2w[:, 2, 2] = th.cos(), th.sin(), -th.sin(), th.cos()
    c2w[:, 0, 3], c2w[:, 2, 3] = -th.sin() * 1.5, -th.cos() * 1.5 + 1.5
    return c2w[None].cuda()
for cfg in sys.argv[1:]:
    net = LaCTLVSM(**omegaconf.OmegaConf.load(cfg)).cuda().eval()
    n = sum(p.numel() for p in net.parameters())
    vi, vt = 8, 4
    fx = torch.tensor([[200., 200., 128., 128.]]).repeat(1, vi, 1).cuda()
    inp = {"image": torch.rand(1, vi, 3, 256, 256).cuda(), "c2w": cams(vi), "fxfycxcy": fx}
    tgt = {"image": torch.rand(1, vt, 3, 256, 256).cuda(), "c2w": cams(vt) + 0.01, "fxfycxcy": fx[:, :vt]}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        net(inp, tgt)
        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], with_flops=True) as prof:
            net(inp, tgt)
    fl = sum(e.flops for e in prof.key_averages() if e.flops)
    print(f"{cfg}: params {n:,}  FLOPs {fl/1e9:.2f} G", flush=True)
