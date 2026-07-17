# -*- coding: utf-8 -*-

from typing import Optional

from transformers.configuration_utils import PretrainedConfig


class LaCTSWIGLUConfig(PretrainedConfig):
    """
    Configuration for LaCT-SWIGLU model.
    It implements the LaCT-SWIGLU layer mixed with in-layer sliding window attention

    Args:
        hidden_size (int, optional): The hidden size of the model. Defaults to 2048.
        num_hidden_layers (int, optional): The number of hidden layers in the model. Defaults to 24.
        num_attn_heads (int, optional): The number of attention heads in the model. Defaults to 32.
        num_lact_heads (int, optional): The number of feed-forward heads in the model. Defaults to 4.
    """

    model_type = "lact_swiglu"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        hidden_size: int = 2048,
        num_hidden_layers: int = 24,
        num_attn_heads: int = 32,
        num_lact_heads: int = 4,
        inter_multi: int = 1,
        qkv_bias: bool = False,
        attn_qk_norm: bool = False,
        lact_chunk_size: int = 2048,
        use_muon: bool = False,
        lr_dim: int = 1,
        qkv_silu: bool = True,  # if True, apply silu to q, k, v.
        no_v_silu: bool = False,  # if True, don't apply silu to v, will overwrite qkv_silu.
        lr_parameterization: str = "mamba",
        learnable_ttt_scale: bool = True,
        use_momentum: bool = True,
        ttt_loss_type: str = "dot_product",  # "l2"
        ttt_prenorm: bool = True,  # pre-norm or post-norm for ttt.
        # prenorm ttt:  state = state + f(norm(state))
        # postnorm ttt:  state = norm(state + f(state)
        ttt_nope: bool = False,  # if True, no positional encoding for query and key used in ttt.
        ttt_hidden_rope: bool = False,  # h-PRA: rotary on the SwiGLU hidden activation
        # --- Q9 GA genes for the hidden rotary (all no-ops unless ttt_hidden_rope) ---
        ttt_hrope_frac: float = 0.5,  # fraction of hidden dims rotated (0..1]; 0.5 = F27 setting
        ttt_hrope_gain: float = 1.0,  # global multiplier on the hidden frequency ladder
        ttt_hrope_theta: Optional[float] = None,  # ladder base; None -> rope_theta
        ttt_input_theta: Optional[float] = None,  # base for the fast-q/k INPUT rope; None -> rope_theta. Small = local (high-freq) band, for band-split vs the hidden site.
        ttt_hrope_interleave: bool = False,  # offset the hidden ladder by half a log-step so its frequencies sit BETWEEN the input's (same band, complementary frequencies).
        ttt_perhead_freqs: bool = False,  # AdaRoPE-style AdaFreq: PER-FW-HEAD learnable log-frequencies xi [num_fw_heads, P_h], theta=exp(xi), fp32, init from the fixed ladder.
        ttt_liere: int = 0,  # LieRE (learnable Lie-group rotary, ICML 2025) on the HIDDEN
        # address: 0 = off; >0 = generator block size b. Per-layer, head-shared learnable
        # skew generators A [nb, b, b] (nb = 2*P_h/b covers the rotated span); rotation
        # per token = matrix_exp(A * t), orthogonal by construction. b=2 reduces to a
        # learnable-frequency 2D rotation; larger b learns the rotation PLANES too.
        # Requires ttt_hidden_rope; replaces the fixed cos/sin ladder.
        ttt_liere_init: str = "rope",  # "rope": init so matrix_exp(A*t) EXACTLY equals the
        # fixed h_inv_freq cos/sin ladder at step 0 (2x2 ladder rotations embedded
        # block-diagonally, positions unscaled). "random": LieRE convention, skew entries
        # ~U[0, 2pi] with positions normalized by max_position_embeddings.
        ttt_hrope_delta_only: bool = False,  # rotate only the fast-weight DELTA path on apply:
        # out = w_init @ h + (w - w_init) @ R(phi) h. Removes the absolute-phase tax on the
        # initial readout (F27b) while keeping relative update->apply recall.
        ttt_hrope_chunkq: int = 0,  # Q22: quantize the HIDDEN rotary positions to multiples of
        # this chunk size (floor(t/C)*C) so stored addresses within an update chunk share one
        # rotation (block-relative h-PRA). 0 = off; use lact_chunk_size (1024) normally.
        ttt_input_chunkq: int = 0,  # Q22: same quantization for the INPUT fast-q/k rope
        # (manual rotary path; 1 = per-token via the manual path, for surgery baselines).
        ttt_w1_precess: float = 0.0,  # Q25a chunk precession: gain of the fixed
        # per-chunk-boundary rotation of the w1 delta's hidden-dim pairs (recency
        # kernel in the STATE; init readout excluded; 0 = exact baseline).
        ttt_hrope_conjpairs: bool = False,  # Q25c: conjugate-paired hidden ladder
        # h_inv = cat([f, -f]) with f = P_h/2-point ladder — every frequency on one
        # +plane and one -plane. Gives the model an even/odd-separable basis: tying
        # content across partner planes keeps the cos(omega*dt) recency envelope and
        # cancels the antisymmetric offset code (the component F27b says gets ignored).
        ttt_hrope_hnorm: str = "none",  # "rms" | "rms_rot": RMS-normalize the hidden before
        # the rotation (make the hidden code spherical like the L2-normalized input q/k;
        # tests the F27c geometry hypothesis). "rms_rot" normalizes only the rotated dims.
        ttt_value_rope: bool = False,  # VaPE (Q23): value-path rotary. The update trains
        # f_W toward the ROTATED target R_t v_t (fixed ladder on the first 2*P_v value
        # dims); on apply the (delta) readout is counter-rotated by the query position so
        # retrieval carries only relative phases sum_t lr_t <h_t,h_s> R_{t-s} v_t. No
        # rotation inside f_W -> no inverse rotation in the inner-loop backward. Composes
        # with the input rope; mutually exclusive with ttt_hidden_rope/ttt_liere for now.
        ttt_vrope_frac: float = 0.5,  # fraction of value dims rotated (P_v = floor(frac*d_out/2))
        ttt_vrope_gain: float = 1.0,  # global multiplier on the value frequency ladder
        ttt_vrope_theta: Optional[float] = None,  # ladder base; None -> rope_theta
        ttt_vrope_delta_only: bool = True,  # counter-rotate only the fast-weight DELTA
        # readout on apply (init readout untouched -> no absolute-phase tax, F27b). The
        # DEFAULT variant; False = counter-rotate the whole readout.
        ttt_branch_rope: str = "none",  # GbR (Q25b): "none" | "gate" | "content".
        # Gate-branch-only input rope: the SwiGLU fast weight f(x) =
        # w1(silu(w0 x) * (w2 x)) has two input branches; the standard input rope
        # rotates the q/k feeding BOTH. "gate" rotates only the copy feeding the silu
        # gate branch (w0) and leaves the linear content branch (w2) unrotated;
        # "content" is the mirror. Diagnostic 2-cell readout localizing where the
        # input rope's +0.20 ppl lives. No new parameters; zero-phase reduces
        # bit-exactly to baseline. Mutually exclusive with ttt_nope /
        # ttt_hidden_rope / ttt_liere / ttt_value_rope / ttt_learnable_freqs.
        ttt_learnable_freqs: bool = False,  # omega_map(1D): learnable frequency deltas
        ttt_sharedf: bool = False,  # share the learnable frequency Parameters across ALL layers
        ttt_hrope_min_layer: int = 0,  # apply the hidden rotary only from this layer index on
        ttt_learnable_input_freqs: bool = True,  # fwqk_dfreq input deltas (off = hidden ladder only)
        ttt_input_freq_tilt: Optional[float] = None,  # input-delta init scale; None -> ttt_freq_tilt; 0.0 = start exactly at the fixed ladder
        ttt_hidden_basis: bool = False,  # learned rotation basis U (init I) on the hidden address
        ttt_freq_tilt: float = 0.1,  # random-tilt init scale for learnable freqs
        w0_w2_low_rank: int = -1,  # -1 means fully learnable.  > 1 means low rank parameterization of the initial learnable weights.
        window_size: int = 2048,
        rope_theta: Optional[float] = 10000.0,
        max_position_embeddings: int = 2048,
        hidden_ratio: Optional[int] = 4,
        intermediate_size: Optional[int] = None,
        hidden_act: str = "swish",
        initializer_range: float = 0.006,
        elementwise_affine: Optional[bool] = True,
        norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        fuse_norm: bool = True,
        last_layer_fuse_norm: bool = True,
        fuse_swiglu: bool = True,
        fuse_cross_entropy: bool = True,
        vocab_size: int = 32000,
        fw_init_gain: float = 0.5,
        use_fused_kernel: bool = False,  # use triton kernel for ttt implementation
        fp32_states: bool = False,  # whether to keep the fast weights in fp32
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attn_heads = num_attn_heads
        self.num_lact_heads = num_lact_heads
        self.inter_multi = inter_multi
        self.qkv_bias = qkv_bias
        self.attn_qk_norm = attn_qk_norm
        self.lact_chunk_size = lact_chunk_size
        self.use_muon = use_muon
        self.lr_dim = lr_dim
        self.qkv_silu = qkv_silu
        self.no_v_silu = no_v_silu
        self.window_size = window_size
        self.lr_parameterization = lr_parameterization
        self.learnable_ttt_scale = learnable_ttt_scale
        self.ttt_prenorm = ttt_prenorm
        self.ttt_nope = ttt_nope
        self.ttt_hidden_rope = ttt_hidden_rope
        self.ttt_hrope_frac = ttt_hrope_frac
        self.ttt_hrope_gain = ttt_hrope_gain
        self.ttt_hrope_theta = ttt_hrope_theta
        self.ttt_input_theta = ttt_input_theta
        self.ttt_hrope_interleave = ttt_hrope_interleave
        self.ttt_perhead_freqs = ttt_perhead_freqs
        self.ttt_liere = ttt_liere
        self.ttt_liere_init = ttt_liere_init
        if ttt_liere:
            assert ttt_hidden_rope, "ttt_liere requires ttt_hidden_rope=True"
            assert ttt_liere_init in ("rope", "random"), ttt_liere_init
        self.ttt_hrope_delta_only = ttt_hrope_delta_only
        self.ttt_hrope_chunkq = ttt_hrope_chunkq
        self.ttt_w1_precess = ttt_w1_precess
        self.ttt_hrope_conjpairs = ttt_hrope_conjpairs
        self.ttt_input_chunkq = ttt_input_chunkq
        self.ttt_hrope_hnorm = ttt_hrope_hnorm
        self.ttt_value_rope = ttt_value_rope
        self.ttt_vrope_frac = ttt_vrope_frac
        self.ttt_vrope_gain = ttt_vrope_gain
        self.ttt_vrope_theta = ttt_vrope_theta
        self.ttt_vrope_delta_only = ttt_vrope_delta_only
        if ttt_value_rope:
            assert not ttt_hidden_rope and not ttt_liere, \
                "ttt_value_rope is mutually exclusive with ttt_hidden_rope / ttt_liere (for now)"
        self.ttt_branch_rope = ttt_branch_rope
        assert ttt_branch_rope in ("none", "gate", "content"), ttt_branch_rope
        if ttt_branch_rope != "none":
            assert not (ttt_nope or ttt_hidden_rope or ttt_liere or ttt_value_rope
                        or ttt_learnable_freqs), \
                "ttt_branch_rope is mutually exclusive with ttt_nope / ttt_hidden_rope / " \
                "ttt_liere / ttt_value_rope / ttt_learnable_freqs"
        self.ttt_learnable_freqs = ttt_learnable_freqs
        self.ttt_sharedf = ttt_sharedf
        self.ttt_hrope_min_layer = ttt_hrope_min_layer
        self.ttt_learnable_input_freqs = ttt_learnable_input_freqs
        self.ttt_input_freq_tilt = ttt_input_freq_tilt
        self.ttt_hidden_basis = ttt_hidden_basis
        self.ttt_freq_tilt = ttt_freq_tilt
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act

        self.initializer_range = initializer_range
        self.elementwise_affine = elementwise_affine
        self.norm_eps = norm_eps
        self.use_cache = use_cache

        self.fuse_norm = fuse_norm
        self.last_layer_fuse_norm = last_layer_fuse_norm  # seems that you need to set this to False to use activation checkpointing for every layer.
        self.fuse_swiglu = fuse_swiglu
        self.fuse_cross_entropy = fuse_cross_entropy
        self.vocab_size = vocab_size

        self.use_momentum = use_momentum
        self.ttt_loss_type = ttt_loss_type
        self.w0_w2_low_rank = w0_w2_low_rank
        self.fw_init_gain = fw_init_gain
        self.use_fused_kernel = use_fused_kernel
        self.fp32_states = fp32_states
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
