"""
Implementation of https://arxiv.org/abs/2410.17242
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor

from prope.torch import PropeDotProductAttention
from prope.utils.functional import (
    Camera,
    camera_to_raymap,
    patchify,
    raymap_to_plucker,
    unpatchify,
)
from prope.utils.transformer import (
    TransformerEncoderConfig,
    TransformerEncoderLayerConfig,
)


@dataclass
class LVSMDecoderOnlyModelConfig:

    ref_views: int
    tar_views: int = 1

    encoder: TransformerEncoderConfig = field(
        default_factory=lambda: TransformerEncoderConfig(
            layer=TransformerEncoderLayerConfig(
                d_model=768,
                nhead=16,
                dim_feedforward=3072,
                dropout=0.0,
                activation=F.relu,
                layer_norm_eps=1e-5,
                batch_first=True,
                norm_first=True,
                bias=False,
                elementwise_affine=True,
                norm_type="layer_norm",
                modulation_activation=None,
                qk_norm=False,
            ),
            num_layers=6,
            input_norm=True,
            output_norm=True,
            checkpointing=False,
        ),
    )

    img_shape: Tuple[int, ...] = (256, 256, 3)
    cam_shape: Tuple[int, ...] = (256, 256, 6)
    patch_size: int = 8

    # How the input rays are encoded.
    ray_encoding: Literal["plucker", "camray", "none", "raymap"] = "plucker"

    pos_enc: Literal["prope", "gta", "none"] = "prope"


class LVSMDecoderOnlyModel(nn.Module):
    def __init__(self, config: LVSMDecoderOnlyModelConfig):
        super().__init__()
        self.config = config

        self.attention = PropeDotProductAttention(
            head_dim=config.encoder.layer.d_model // config.encoder.layer.nhead,
            patches_x=config.img_shape[1] // config.patch_size,
            patches_y=config.img_shape[0] // config.patch_size,
            image_width=config.img_shape[1],
            image_height=config.img_shape[0],
        )

        assert (
            config.cam_shape[:2] == config.img_shape[:2]
        ), f"{config.cam_shape[:2]} != {config.img_shape[:2]}"

        if config.ray_encoding == "none":
            shared_rays = torch.randn(config.cam_shape)
            self.shared_rays = nn.Parameter(shared_rays, requires_grad=False)

        # query tokenizer encodes tar_cam
        self.query_tokenizer = nn.Linear(
            config.cam_shape[-1] * config.patch_size**2,
            config.encoder.layer.d_model,
            bias=config.encoder.layer.bias,
        )
        # input tokenizer encodes ref_img and ref_cam
        self.input_tokenizer = nn.Linear(
            (
                config.img_shape[-1] * config.patch_size**2
                + config.cam_shape[-1] * config.patch_size**2
            ),
            config.encoder.layer.d_model,
            bias=config.encoder.layer.bias,
        )

        self.encoder = self.config.encoder.setup()

        self.output_layer = nn.Linear(
            config.encoder.layer.d_model,
            config.img_shape[-1] * config.patch_size**2,
            bias=config.encoder.layer.bias,
        )
        self.init_weights()

    def init_weights(self):
        for idx, layer in enumerate(self.encoder.layers):
            layer.apply(self.init_layer_weights(idx))

    def init_layer_weights(self, idx):
        # LVMS Paper A.1:
        # "We initialize the model weights with a normal distribution of zero-mean
        # and standard deviation of 0.02/(2 * (idx+ 1)) ** 0.5, where idx means
        # transform layer index."
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.02 / (2 * (idx + 1)) ** 0.5)

        return _init_weights

    def create_rays(self, cams: Camera) -> Tensor:
        """Convert cameras to raymaps.

        Returns:
            rays: [B, V, H, W, C]
        """
        config = self.config
        batch_size, v = cams.camtoworld.shape[:2]
        cam_dtype = cams.camtoworld.dtype
        device = cams.camtoworld.device

        if config.ray_encoding == "none":
            rays = repeat(self.shared_rays, "h w c -> b v h w c", b=batch_size, v=v)
        else:
            # Preprocess cameras into rays.
            downscale = config.img_shape[0] // config.cam_shape[0]
            rays = camera_to_raymap(
                Ks=cams.K,
                camtoworlds=(
                    torch.eye(4, dtype=cam_dtype, device=device).broadcast_to(
                        cams.camtoworld.shape
                    )
                    if config.ray_encoding == "camray"
                    else cams.camtoworld
                ),
                height=cams.height,
                width=cams.width,
                downscale=downscale,
            )
            if config.ray_encoding in ["plucker", "camray"]:
                rays = raymap_to_plucker(rays)
            else:
                assert config.ray_encoding == "raymap"
        return rays

    def forward(
        self,
        ref_imgs: Tensor,
        ref_cams: Camera,
        tar_cams: Camera,
    ) -> Tensor:
        # ref_imgs: [B, V1, H, W, C]
        # tar_imgs: [B, V2, H, W, C]
        batch_size, v2 = tar_cams.camtoworld.shape[:2]
        config = self.config

        # Create rays.
        # ref_rays: [B, V1, H, W, C]
        # tar_rays: [B, V2, H, W, C]
        ref_rays = self.create_rays(ref_cams)
        tar_rays = self.create_rays(tar_cams)

        # ref_imgs: [B, V1, N1, DIM1]
        ref_imgs = patchify(ref_imgs, config.patch_size)
        # ref_rays: [B, V1, N2, DIM2]
        ref_rays = patchify(ref_rays, config.patch_size)
        # tar_rays: [B, V2, N2, DIM2]
        tar_rays = patchify(tar_rays, config.patch_size)

        # Tokenize into
        # x: [B*V2, V1*N1, DIM1]
        # q: [B*V2, N2, DIM2]
        x = self.input_tokenizer(torch.cat([ref_imgs, ref_rays], dim=-1))
        x = repeat(x, "b v1 n d -> (b v2) (v1 n) d", v2=v2)
        q = self.query_tokenizer(tar_rays)
        q = rearrange(q, "b v2 n d -> (b v2) n d")
        q_tokens = q.shape[1]

        # --- Prepare data for geomtry-aware self-attention ---
        ref_c2ws = repeat(ref_cams.camtoworld, "b v1 x y -> (b v2) v1 x y", v2=v2)
        ref_Ks = repeat(ref_cams.K, "b v1 x y -> (b v2) v1 x y", v2=v2)
        tar_c2ws = rearrange(tar_cams.camtoworld, "b v2 x y -> (b v2) 1 x y", v2=v2)
        tar_Ks = rearrange(tar_cams.K, "b v2 x y -> (b v2) 1 x y")
        c2ws = torch.cat([ref_c2ws, tar_c2ws], dim=1)  # [B, N, 4, 4] per camera
        Ks = torch.cat([ref_Ks, tar_Ks], dim=1)  # [B, N, 3, 3] per camera
        viewmats = torch.inverse(c2ws)

        def sdpa_fn(q, k, v, **kwargs):
            if config.pos_enc == "prope":
                return self.attention(q, k, v, viewmats=viewmats, Ks=Ks, **kwargs)
            elif config.pos_enc == "gta":
                # GTA is effectively PRoPE without intrinsics.
                return self.attention(q, k, v, viewmats=viewmats, Ks=None, **kwargs)
            elif config.pos_enc == "none":
                # Use the default attention.
                return F.scaled_dot_product_attention(q, k, v, **kwargs)
            else:
                raise ValueError(f"Invalid pos_enc: {config.pos_enc}")

        # run attentions
        xq = torch.cat([x, q], dim=1)
        xq = self.encoder(xq, sdpa_fn=sdpa_fn)
        q = xq[:, -q_tokens:, :]
        q = rearrange(q, "(b v) n d -> b v n d", b=batch_size, v=v2)

        # output layer
        o = self.output_layer(q)
        o = unpatchify(
            o,
            height=config.img_shape[0],
            width=config.img_shape[1],
            patch_size=config.patch_size,
        )
        return o


if __name__ == "__main__":
    # Test the model.
    import tqdm

    device = "cuda:0"
    ref_views = 2
    tar_views = 4
    batch_size = 1
    height = 256
    width = 256

    ref_imgs = torch.randn(batch_size, ref_views, height, width, 3).to(device)
    ref_cams = Camera(
        K=torch.randn(1, ref_views, 3, 3).to(device),
        camtoworld=torch.randn(1, ref_views, 4, 4).to(device),
        height=height,
        width=width,
    )
    tar_cams = Camera(
        K=torch.randn(1, tar_views, 3, 3).to(device),
        camtoworld=torch.randn(1, tar_views, 4, 4).to(device),
        height=height,
        width=width,
    )

    config = LVSMDecoderOnlyModelConfig(ref_views=2)
    model = LVSMDecoderOnlyModel(config).to(device)
    with torch.autocast("cuda"):
        for _ in tqdm.trange(100):
            y = model(ref_imgs, ref_cams, tar_cams)
        assert y.shape == (batch_size, tar_views, height, width, 3)
