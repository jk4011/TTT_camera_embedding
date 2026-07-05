# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
import torch.cuda.amp as amp
import torch.nn as nn
from typing import Optional
import omegaconf
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .attention import flash_attention

from .model import sinusoidal_embedding_1d, rope_params, rope_apply, WanRMSNorm, WanLayerNorm, WanSelfAttention, WanI2VCrossAttention, WanI2VCrossAttention, Head, MLPProj, WanT2VCrossAttention
from minVid.utils.config_utils import instantiate_from_config, ObjectParamConfig

WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
}


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 do_hybrid=False,
                 efficient_attn_config: Optional[ObjectParamConfig] = None,
                 use_cam_encoder=False):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # ReCamMaster-style per-block camera injection (ccv baseline):
        # Linear(12 -> dim) on per-frame relative extrinsics + identity-init
        # projector, ADDED to the block input before self-attn. Follows the
        # official ReCamMaster init (cam_encoder zero-init, projector = I) so
        # the injection is exactly zero at step 0 (cross-variant sanity).
        if use_cam_encoder:
            self.cam_encoder = nn.Linear(12, dim)
            self.cam_projector = nn.Linear(dim, dim)
            nn.init.zeros_(self.cam_encoder.weight)
            nn.init.zeros_(self.cam_encoder.bias)
            with torch.no_grad():
                self.cam_projector.weight.copy_(torch.eye(dim))
                self.cam_projector.bias.zero_()
        else:
            self.cam_encoder = None
            self.cam_projector = None

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        if do_hybrid:
            print(f"Init Efficient Attention with efficient_attn_config: {efficient_attn_config}")
            self.self_attn = instantiate_from_config(efficient_attn_config, split_config=True)
        else:
            self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        
        self.do_hybrid = do_hybrid
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        cam12=None,
        cam_coords6=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            cam12(Tensor, optional): [B, L, 12] per-token relative extrinsics
                (ccv cam_encoder path).
            cam_coords6(Tensor, optional): [b_true, L_total, 6] per-token
                Plucker coords for the hybrid attention's camera rotary.
        """
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e).chunk(6, dim=1)
        assert e[0].dtype == torch.float32

        # ccv: camera feature injection before self-attn (ReCamMaster recipe)
        if cam12 is not None and self.cam_encoder is not None:
            x = x + self.cam_projector(self.cam_encoder(cam12.to(x.dtype)))

        # self-attention
        extra_info = None
        attn_kwargs = {}
        if self.do_hybrid and cam_coords6 is not None:
            attn_kwargs["cam_coords6"] = cam_coords6
        y, extra_info = self.self_attn(
            self.norm1(x).float() * (1 + e[1]) + e[0], seq_lens, grid_sizes,
            freqs, **attn_kwargs)
        with amp.autocast(dtype=torch.float32):
            x = x + y * e[2]

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn(self.norm2(x).float() * (1 + e[4]) + e[3])
            with amp.autocast(dtype=torch.float32):
                x = x + y * e[5]
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x, extra_info
    
    def get_trainable_params(self, attn_only=True):
        if attn_only:
            # only return the parameters of the hybrid attention module
            # (+ the ccv cam_encoder/cam_projector when present)
            param_list = []
            if self.do_hybrid:
                # Check if the attention module has a get_trainable_params method
                if hasattr(self.self_attn, 'get_trainable_params'):
                    param_list.extend(self.self_attn.get_trainable_params(attn_only=attn_only))
                else:
                    param_list.extend(self.self_attn.parameters())
            if self.cam_encoder is not None:
                param_list.extend(self.cam_encoder.parameters())
                param_list.extend(self.cam_projector.parameters())
            return param_list
        else:
            # return all the parameters
            return self.parameters()



class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 efficient_attn_config: Optional[ObjectParamConfig] = None,
                 attn_every_n_layers: int = 1,
                 use_cam_encoder: bool = False):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        print(f"WanModel init with model_type: {model_type}")

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.efficient_attn_config = efficient_attn_config
        self.attn_every_n_layers = attn_every_n_layers

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        # self.blocks = nn.ModuleList([
        #     WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
        #                       window_size, qk_norm, cross_attn_norm, eps,)
        #     for _ in range(num_layers)
        # ])
        self.use_cam_encoder = use_cam_encoder
        self.blocks = nn.ModuleList()
        print("type of efficient_attn_config", type(efficient_attn_config))
        for i in range(num_layers):
            if i % attn_every_n_layers == attn_every_n_layers - 1:
                self.blocks.append(WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps, do_hybrid=False,
                              use_cam_encoder=use_cam_encoder))
            else:
                if isinstance(efficient_attn_config, omegaconf.listconfig.ListConfig):
                    print("efficient_attn_config is a ListConfig")
                    _attn_config = efficient_attn_config[i % len(efficient_attn_config)]
                else:
                    _attn_config = efficient_attn_config
                self.blocks.append(WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps, do_hybrid=True, efficient_attn_config=_attn_config,
                              use_cam_encoder=use_cam_encoder))

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
                               dim=1)

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        # REMOVE weight init, unnecessary for finetuning. 
        # Please handle the init for added weights! 
        # self.init_weights()

        self.gradient_checkpointing = False # might not use it.

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value


    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        cam12_per_frame=None,
        cam_coords6=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x
            cam12_per_frame (Tensor, *optional*):
                [b_true, F_total, 12] fp32 per-latent-frame relative extrinsics
                for the ccv cam_encoder path (broadcast over each frame's tokens).
            cam_coords6 (Tensor, *optional*):
                [b_true, L_total, 6] fp32 per-token Plucker coords for the
                hybrid attention's camera rotary sites (ccv PRA path).

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len


        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
        # For ease of coding, assume the seq_len is the same for batched videos. 
        # x = torch.cat(x, dim=0) # [b, s, d] now. 

        # time embeddings
        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # ccv: broadcast per-frame camera features to per-token, chunk layout.
        cam12_tokens = None
        if cam12_per_frame is not None and self.use_cam_encoder:
            b_true, F_total, _ = cam12_per_frame.shape
            f_per_chunk = int(grid_sizes[0][0])
            tokens_per_frame = int(grid_sizes[0][1]) * int(grid_sizes[0][2])
            # [b, F_total, 12] -> [b * n_chunks (= fake batch), f_per_chunk, 12]
            cam12_tokens = cam12_per_frame.reshape(-1, f_per_chunk, 12)
            # -> [fake_batch, f_per_chunk * tokens_per_frame, 12] (frame-major,
            # matching the (f, h, w) patchify flatten order)
            cam12_tokens = cam12_tokens.repeat_interleave(tokens_per_frame, dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            cam12=cam12_tokens,
            cam_coords6=cam_coords6)

        extra_info_list = []
        for block in self.blocks:
            x, extra_info = block(x, **kwargs)
            extra_info_list.append(extra_info)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x], extra_info_list

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)

    def get_trainable_params(self, attn_only=True, **kwargs):
        param_list = []

        if attn_only:
            for block in self.blocks:
                if block.do_hybrid:
                    # Check if the attention module has a get_trainable_params method
                    if hasattr(block.self_attn, 'get_trainable_params'):
                        print("block has get_trainable_params implemented, calling it")
                        param_list.extend(block.self_attn.get_trainable_params(attn_only=attn_only, **kwargs))
                    else:
                        print("block has no get_trainable_params implemented, calling parameters")
                        param_list.extend(block.self_attn.parameters())
                # ccv cam_encoder path (ReCamMaster recipe: cam modules trainable)
                if getattr(block, "cam_encoder", None) is not None:
                    param_list.extend(block.cam_encoder.parameters())
                    param_list.extend(block.cam_projector.parameters())
        else:
            param_list.extend(self.parameters())
        return param_list