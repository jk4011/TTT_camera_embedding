"""dpt_abs must not read the scene focus: poison t_c with NaN and require a finite loss + grads."""
import sys, torch, omegaconf
import model as M
from model import LaCTLVSM
torch.manual_seed(0)
cfg_path, poison = sys.argv[1], sys.argv[2] == "nan"
mc = omegaconf.OmegaConf.load(cfg_path)
net = LaCTLVSM(**mc).cuda()
_orig = M.compute_camera_info
def _poisoned(*a, **k):
    info = _orig(*a, **k)
    if poison:
        info["tok_tc"] = torch.full_like(info["tok_tc"], float("nan"))
    return info
M.compute_camera_info = _poisoned
b, vi, vt, h, w = 1, 8, 8, 256, 256
def cams(v):
    th = torch.linspace(0, 1.2, v)
    c2w = torch.eye(4).repeat(v, 1, 1)
    c2w[:, 0, 0], c2w[:, 0, 2], c2w[:, 2, 0], c2w[:, 2, 2] = th.cos(), th.sin(), -th.sin(), th.cos()
    c2w[:, 0, 3], c2w[:, 2, 3] = -th.sin() * 1.5, -th.cos() * 1.5 + 1.5
    return c2w[None].cuda()
fx = torch.tensor([[200., 200., 128., 128.]]).repeat(1, vi, 1).cuda()
inp = {"image": torch.rand(b, vi, 3, h, w).cuda(), "c2w": cams(vi), "fxfycxcy": fx}
tgt = {"image": torch.rand(b, vt, 3, h, w).cuda(), "c2w": cams(vt) + 0.01, "fxfycxcy": fx[:, :vt]}
with torch.autocast("cuda", dtype=torch.bfloat16):
    out = net(inp, tgt)
def _tensors(o):
    if torch.is_tensor(o): yield o
    elif isinstance(o, dict):
        for v in o.values(): yield from _tensors(v)
    elif isinstance(o, (tuple, list)):
        for v in o: yield from _tensors(v)
ts = [t for t in _tensors(out) if t.is_floating_point() and t.requires_grad]
print("outputs:", [tuple(t.shape) for t in ts][:4])
loss = sum(t.float().pow(2).mean() for t in ts)
loss.backward()
bad = [n for n, p in net.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
print(f"{cfg_path} poison={poison}: loss {float(loss):.4f} finite={bool(torch.isfinite(loss))}  non-finite grads: {len(bad)}")
