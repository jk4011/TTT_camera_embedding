# Rotary Fast-Weight Addressing: Relative Camera Conditioning for Test-Time-Training Layers

## Abstract

Test-time-training (TTT) layers replace quadratic attention with fast-weight networks that are updated by gradient descent during the forward pass, giving linear cost in sequence length. Attention has a mature toolbox of relative camera encodings such as PRoPE, but all of these encodings exploit its bilinear query–key logits, and TTT layers—nonlinear SwiGLU networks trained on the fly—offer no such logits. We show that they offer something just as good: because every fast-weight update is a sum of outer products, everything a query reads from newly written memory passes through inner products. Expanding this readout reveals three channels—initial readout, value retrieval, and gate corrections—of which the dominant value-retrieval channel lives in the hidden layer of the fast weights and has no counterpart in attention. Two of the channels are created by test-time writes—value retrieval and gate correction—and we call these the *writable channels*. Rotating queries and keys (input rotary) and hidden activations (hidden rotary) by phases linear in Plücker ray coordinates makes both writable channels exactly relative; learnable phase maps let phases mix coordinates while preserving relativity. On RealEstate10K, our method, Plücker Rotary Addressing (PRA)—the camera instance of a general recipe we call rotary fast-weight addressing—improves LaCT-LVSM, a linear-time TTT view-synthesis model, by +1.226 dB PSNR at <0.1% FLOPs overhead; on a 200M LLM, adding the hidden rotary on top of the token-index RoPE that LaCT already applies to fast-weight queries and keys cuts perplexity from 19.32 to 19.13, matching the entire gap between no positional encoding (NoPE) and RoPE.

## 1 Introduction

Sequence models have become the dominant architecture for multi-view novel view synthesis: images are split into patch tokens, tokens from all views are concatenated, and a backbone exchanges information across views before target pixels are decoded. Attention is the standard backbone, but its cost grows quadratically with the number of tokens, which is punishing in vision, where a handful of images already produces thousands of tokens. Test-time-training (TTT) layers are a linear-time alternative: instead of comparing every token with every other token, a TTT layer maintains a small *fast-weight* network that is updated by gradient descent *during the forward pass*—input tokens are written into the weights, and later tokens read from them. Large-chunk TTT (LaCT)—which updates the fast weights once per large chunk of tokens rather than once per token—made this recipe practical at scale, and LaCT-based LVSM (Large View Synthesis Model) architectures match full attention on novel view synthesis at a fraction of the cost.

For attention, the question of how to tell the model *where each token's camera is* has a mature answer. CaPE (Kong et al., 2024), GTA (Miyato et al., 2024), PRoPE (Li et al., 2025), and RayRoPE (2025) all condition attention on camera pose, and the most effective of them are *relative* encodings in the spirit of RoPE: each token is transformed by a function of its own absolute pose, and the algebra of attention guarantees that only relative pose survives. Concretely, RoPE rotates queries and keys by position-dependent angles, and because the attention logit is a bilinear form $q^\top k$, the two rotations meet inside the inner product and collapse into a rotation by the *difference* of positions. PRoPE and its relatives generalize the same trick to camera transforms. Absolute inputs, relative interactions—this is the property that makes these encodings robust to reposing the scene.

TTT layers have no comparable machinery. In LaCT-LVSM, camera information enters the network exactly once, as a ray map concatenated to the RGB input; every TTT layer—the only module that exchanges information across views—is pose-blind. The obstacle to fixing this is structural: relative encodings for attention lean on the fact that every query–key interaction is *exactly* a bilinear logit, whereas a TTT layer is a nonlinear SwiGLU network whose weights are rewritten by gradient descent. There is no logit to rotate. And the difficulty is not hypothetical: in our experiments, direct transplants of the attention recipe—projective PRoPE-style transforms (applying projective camera matrices to queries and keys), value-side transport (mapping stored values into the reader's camera frame), feature injection (concatenating pose features to tokens), optimizer conditioning (making per-token learning rates pose-dependent), and per-view chunking (one write chunk per camera view)—consistently fail to help. The attention *recipe* does not transplant.

What does transplant, we show, is the *channel structure*. Our starting point is a simple algebraic fact: every fast-weight update in a TTT layer is a sum of outer products of key and value features, such as $k_i^\top v_i$—here $k_i$ and $v_i$ are the key and value that token $i$ writes into the fast weights—and a row vector meeting an outer product contracts through an inner product, $x(a^\top b) = \langle x,a\rangle\, b$. (We use row-vector convention throughout the paper, formalized in Section 2.1: $a^\top b$ is the $d\times d$ outer product of two row vectors, and $\langle x,a\rangle = x a^\top$ is their inner product.) In other words, an outer-product update can hand a later query nothing but an inner product with the stored key, times a stored vector. Consequently, *every* interaction between a query and content written at test time—through any weight matrix, at any order of expansion—passes through inner products between query-side and key-side features. Inner products are exactly the algebraic hook that RoPE-style encodings need. Expanding the readout of a LaCT layer—the output a query receives from the updated fast weights—makes the structure explicit: the output splits into three channels—an *initial-readout channel* (the pre-update weights), a *value-retrieval channel* whose inner product lives in the *hidden* space of the SwiGLU network, and a *gate-correction channel* whose inner product lives in the input space. The value-retrieval channel is the dominant one (empirically it carries most of the readout magnitude; our ablations bear this out), and it is the surprise of this analysis: it has no counterpart in attention, because attention has no hidden layer between its similarity scores and its value aggregation. A conditioning site that is invisible from the attention literature simply becomes visible once one looks at the fast-weight algebra.

This decomposition dictates the method, which we call *rotary fast-weight addressing*. Each of the two writable channels—value retrieval and gate correction, the two channels created by test-time writes—gets its own rotary encoding: an *input rotary* on queries and keys (which makes the gate-correction channel exactly relative, and coincides with the fast-weight RoPE already used by the LaCT authors in language modeling), and a *hidden rotary* that rotates the SwiGLU hidden activation between the hidden layer and the output weights, on both write and read, with the inner-loop (write-step) backward pass applying the inverse rotation. The hidden rotary makes the dominant value-retrieval channel exactly relative—and, we emphasize again, it is the component no attention method could have suggested, because the channel it conditions does not exist in attention. A third ingredient, *learnable phase maps*, replaces the fixed axis-aligned frequency ladder—the standard RoPE-style geometric sequence of frequencies, applied to one coordinate axis at a time—with a learnable linear map from coordinates to phases; linearity preserves exact relativity while letting phases mix coordinate axes. All rotations are orthogonal, so token norms, the $\ell_2$ normalization of $q$ and $k$, the scale of the inner-loop loss, Muon orthogonalization (LaCT's step that orthogonalizes each fast-weight update), and weight-column renormalization are untouched; the overhead is below 0.1% of FLOPs and 0.01% of parameters. Instantiated with 6D Plücker ray coordinates for cameras, we call the method PRA (Plücker Rotary Addressing).

Empirically, the channels behave as the theory predicts. On RealEstate10K with a LaCT-LVSM backbone (6 layers, width 256, 30k iterations, 256-scene evaluation), PRA improves PSNR from 21.745 ± 0.196 to 22.971 ± 0.088 (+1.226 dB over 3 seeds each) and LPIPS from 0.2929 to 0.2613. The channels are additive: in our ablation configuration, input rotary alone gives +0.41 dB and saturates (adding more frequencies at that site yields no further gain); hidden rotary alone gives +0.46 dB; together they give +0.77 dB, rising to +0.93 dB with wider frequency ladders and learnable phase maps. The headline number comes from the full recipe rather than the lighter ablation setting: training the full recipe under the full protocol closes the remaining gap to +1.226 dB. The construction is not camera-specific: on a 200M-parameter LLM trained on 3B tokens, adding the hidden rotary on top of the token-index fast-weight RoPE that LaCT already applies reduces perplexity from 19.32 to 19.13—a gain matching the entire gap between no positional encoding (NoPE) and RoPE (19.56 → 19.32), stacked on top of the RoPE gain. On an autoregressive video finetune the method is neutral, which we report honestly as a boundary condition: gains require the TTT memory to carry enough of the workload. Across three tasks, two improve and none is hurt.

**Contributions.**
- **An addressing lemma for TTT layers.** We prove that in a LaCT layer every interaction between a query and test-time-written content passes through inner products, and derive a three-channel decomposition of the readout (initial readout, value retrieval, gate corrections) that identifies exactly where relative encodings can act.
- **The hidden rotary site.** We identify the dominant value-retrieval channel inside the SwiGLU hidden layer—a conditioning site with no analogue in attention—and condition it with an orthogonal rotary applied on write and read, with inverse-rotation backpropagation.
- **Learnable phase maps.** A linear, learnable coordinate-to-phase map that provably preserves relativity while letting phases leave the coordinate axes, useful precisely when coordinates are multi-dimensional (6D rays, 3D grids) and provably degenerate in 1D.
- **Evidence and forensics.** +1.226 dB PSNR on RealEstate10K at <0.1% FLOPs overhead, −1.0% LLM perplexity, a documented neutral video result, and 26 runs of negative-result forensics showing that attention-style recipes fail while the channel-structured recipe succeeds.

## 2 Preliminaries

### 2.1 Setting and notation

We describe the method in its camera instance; Section 3.5 discusses other coordinates. Given a set of posed input images and target camera poses, an LVSM-style model splits every image into non-overlapping patches; each patch is one token. Input-view tokens carry patch RGB plus a per-pixel ray map (an encoding that stores, at every pixel, the origin and direction of that pixel's viewing ray). Target-view tokens carry only the ray map; a backbone processes the concatenated token sequence. Each token \(i\) is associated with the viewing ray of its patch center, expressed in a canonical scene frame: origin \(\mathbf{o}_i \in \mathbb{R}^3\) and unit direction \(\mathbf{d}_i \in \mathbb{R}^3\). The canonical frame is fixed per scene by convention—anchored to the first input camera, with the scene scale normalized—so that rays from all views share a single coordinate system.

We use row-vector convention throughout: a feature \(x \in \mathbb{R}^{1\times d}\) multiplies weight matrices from the right, and we write \(\langle x, y\rangle = x y^\top\) for the inner product of two row vectors.

### 2.2 The RoPE identity in attention

RoPE is the tool we want to transplant, so we recall the one identity that makes it work. For a phase vector \(\theta \in \mathbb{R}^{P}\), let \(R(\theta) \in \mathrm{SO}(2P)\) denote the block-diagonal matrix that rotates \(P\) independent 2D feature pairs (we call each pair a *phase plane*) by the angles \(\theta\). If token \(i\) carries phases \(\theta_i\) proportional to its position, then

$$
\big\langle\, q\,R(\theta_j),\; k\,R(\theta_i) \,\big\rangle \;=\; q\, R(\theta_j)\,R(\theta_i)^\top k^\top \;=\; \big\langle\, q,\; k\,R(\theta_i - \theta_j) \big\rangle .
$$

In words: two absolute rotations meet inside an inner product and cancel into a rotation by the phase *difference* \(\theta_i - \theta_j\). Every camera extension of RoPE (GTA, PRoPE) is a variation on this identity. The identity needs one thing to fire: the interaction must be an inner product. Attention supplies inner products by definition—its logits are bilinear. The entire question of this paper is where a *nonlinear* TTT layer supplies them.

### 2.3 The LaCT layer

A TTT layer replaces attention with a small fast-weight network that is updated during the forward pass. Two kinds of parameters are in play, and keeping them apart is essential for everything that follows. The *slow weights* are the ordinary parameters of the model—the query/key/value projections, the learning-rate predictors, and the initial fast-weight state introduced below—trained by standard backpropagation and shared across all sequences. The *fast weights* are a per-sequence state: at the start of every scene (more generally, every sequence), they are initialized to a learned initial state \(W^0\), which is itself part of the slow weights; the scene's tokens are then written into them by gradient steps, and the state is discarded when the scene ends. In other words, the fast weights are rebuilt from scratch for every scene—nothing written in one scene survives into the next.

The LaCT variant we build on maintains, per head, a SwiGLU fast-weight network \(W = (W_0, W_2, W_1)\), initialized to \(W^0 = (W_0^0, W_2^0, W_1^0)\), with \(W_0, W_2 \in \mathbb{R}^{d\times d_h}\), \(W_1 \in \mathbb{R}^{d_h \times d}\) (we keep LaCT's subscript convention: \(W_0\) and \(W_2\) are the two first-layer matrices, \(W_1\) is the output matrix), computing

$$
f_W(x) \;=\; h(x)\, W_1, \qquad h(x) \;=\; \mathrm{silu}(x W_0)\odot(x W_2),
$$

where \(\odot\) is the elementwise product, \(\mathrm{silu}(x W_0)\) is the *gate branch*, \(x W_2\) is the *linear branch*, and \(h(x)\in\mathbb{R}^{1\times d_h}\) is the *hidden activation*. Queries, keys, and values are produced by a learned projection, and \(q, k\) are \(\ell_2\)-normalized per token; we refer to this step as the *\(\ell_2\) normalization of \(q\) and \(k\)* throughout. The defining choice of LaCT is *large-chunk* test-time training: tokens are grouped into large chunks, and each chunk is written into the fast weights as a single gradient step, so a long sequence in general triggers a series of sequential updates, one per chunk. The layer runs in two phases:

**Write.** We describe one write step; in the LVSM setting, all input-view tokens form a single chunk, so there is exactly one such step. For the chunk of input-view tokens \(\{(k_i, v_i)\}_{i\in\mathcal{I}}\), the fast weights take one gradient step on the loss \(\mathcal{L}(W) = -\sum_i lr_i\, \langle v_i, f_W(k_i)\rangle\), where \(lr_i > 0\) are per-token learning rates predicted from token content. The resulting updates are sums of outer products:

$$
\Delta W_1 \propto \sum_i lr_i\, h(k_i)^\top v_i, \qquad
\Delta W_0 \propto \sum_i lr_i\, k_i^\top g_i, \qquad
\Delta W_2 \propto \sum_i lr_i\, k_i^\top u_i,
$$

where \(g_i, u_i \in \mathbb{R}^{1\times d_h}\) are the backpropagated signals arriving at the two branches: \(g_i\) at the gate branch (updating \(W_0\)) and \(u_i\) at the linear branch (updating \(W_2\)). LaCT additionally applies Muon orthogonalization (Newton–Schulz) to the updates and performs weight-column renormalization; both preserve the outer-product structure (the update remains a sum, over tokens, of rank-one terms in the key direction).

**Read.** Every token \(j\) queries the updated weights: \(o_j = f_{W + \Delta W}(q_j)\).

Note what is missing: neither the write nor the read has any notion of which view a token came from. The layer is pose-blind.

### 2.4 Plücker coordinates

A ray with origin \(\mathbf{o}\) and unit direction \(\mathbf{d}\) has Plücker coordinates \(\pi = (\mathbf{d}, \mathbf{m}) \in \mathbb{R}^6\) with moment \(\mathbf{m} = \mathbf{o}\times\mathbf{d}\). Plücker coordinates identify the underlying line irrespective of where the origin sits on it, and two rays through the same scene point \(\mathbf{p}\) satisfy \(\|\mathbf{m}_i - \mathbf{m}_j\| \le \|\mathbf{p}\|\,\|\mathbf{d}_i-\mathbf{d}_j\|\): rays observing the same point from nearby directions are provably close in Plücker space. This makes \(\pi_i\) a natural 6D "position" for token \(i\).

## 3 Method: Rotary Fast-Weight Addressing

### 3.1 The readout decomposes into three channels

Everything in this paper follows from writing out what a query actually reads after a chunk has been written. Consider one LaCT layer in the one-chunk regime: all input tokens form a single chunk, written in one gradient step at the pre-update weights—which here are exactly the learned initial state \(W^0 = (W_0^0, W_2^0, W_1^0)\) of Section 2.3—then every token reads. The one-chunk case is the LVSM configuration, and we present it for clarity; with multiple chunks the same expansion applies to each write step—every update is still a sum of outer products—so the decomposition below carries over chunk by chunk. Write \(h^0\) for the hidden activation under the pre-update weights. Expanding the read \(o_j = f_{W^0+\Delta W}(q_j)\) to first order in \(\Delta W\) gives the central display of this paper:

$$
\boxed{\;
o_j \;=\;
\underbrace{h^0(q_j)\, W_1^0}_{\substack{\textbf{initial-readout channel}\\ \text{(pre-update weights)}}}
\;+\;
\underbrace{\sum_{i\in\mathcal{I}} lr_i\, \big\langle h^0(q_j),\, h^0(k_i)\big\rangle\, v_i}_{\substack{\textbf{value-retrieval channel}\\ \text{(inner product in \emph{hidden} space; dominant)}}}
\;+\;
\underbrace{\sum_{i\in\mathcal{I}} \big\langle q_j,\, k_i\big\rangle\, c_{ij}}_{\substack{\textbf{gate-correction channel}\\ \text{(inner product in \emph{input} space)}}}
\;+\; O(\Delta W^2)
\;}
$$

Here \(c_{ij}\in\mathbb{R}^{1\times d}\) collects the token-dependent coefficients that arise when the first-layer updates \(\Delta W_0, \Delta W_2\) pass through the SwiGLU nonlinearity. One caution on the name: despite being called *gate-correction*, this channel collects the corrections from *both* first-layer updates—\(\Delta W_0\) (gate branch) and \(\Delta W_2\) (linear branch)—folded into \(c_{ij}\); there is no missing \(\Delta W_2\) term in the display. Before deriving this, read it as a sentence: *a TTT layer answers a query with (a) what the pre-update network would have said, plus (b) a sum of stored values \(v_i\), each weighted by a hidden-space similarity between query and key, plus (c) small corrections weighted by input-space similarity.* Term (b) is the layer doing the job attention does—retrieving values by similarity—except that the similarity is computed between *hidden activations*, one layer deep inside the fast-weight network. We label term (b) *dominant* because it empirically carries most of the readout magnitude; our ablations verify this.

The derivation rests on a single algebraic fact. A row vector meeting an outer product contracts through an inner product:

$$
x\,(a^\top b) \;=\; (x\,a^\top)\, b \;=\; \langle x, a\rangle\, b .
$$

An outer product cannot hand a row vector anything except an inner product—this is the whole mechanism. Since every update in Section 2.3 is a sum of outer products, we apply this contraction three times. Distributing \(W_1^0 + \Delta W_1\) in the read gives the value-retrieval sum exactly. Contracting \(q_j\) against \(\Delta W_0 = \sum_i lr_i\, k_i^\top g_i\) (and likewise \(\Delta W_2\)) shows that the query meets the first-layer updates only through the scalars \(\langle q_j, k_i\rangle\); linearizing the SwiGLU around its pre-update pre-activations—the only approximation in the display—turns those scalars into the gate-correction sum with coefficients \(c_{ij}\). The \(O(\Delta W^2)\) remainder collects terms with two or more update factors; crucially, those terms are themselves produced by further applications of the same contraction, so the query enters them, again, only through inner products with keys. The truncation is presentational, not load-bearing. From here on we drop the superscript and write plain \(h\) for \(h^0\); every hidden activation below is evaluated at the pre-update weights. We record the structural fact:

**Lemma (inner-product addressing).** *In the one-chunk regime, every interaction between a query \(q_j\) and the content written by a key \(k_i\)—through any of \(\Delta W_0, \Delta W_1, \Delta W_2\), at any order of the expansion—enters exclusively through the inner products \(\langle q_j, k_i\rangle\) or \(\langle h(q_j), h(k_i)\rangle\). The pre-update weights \(W^0\) are the only path by which \(q_j\) affects the output other than through an inner product with some key.*

The lemma says a TTT layer is, despite its nonlinearity, an inner-product-addressed memory—exactly the algebraic situation the RoPE identity of Section 2.2 was designed for. But it says more: there are *two distinct* inner products, living in two distinct spaces. This is where the TTT layer genuinely differs from attention. Attention has one similarity (the logit) followed immediately by value aggregation; a TTT layer has a second similarity computed one layer deep, between hidden activations, and that second similarity carries the dominant term. No positional encoding designed for attention ever conditioned such a channel, because in attention that channel does not exist.

The plan for the rest of the section follows the display term by term. Each writable channel gets its own rotary site (Sections 3.2 and 3.3); the initial-readout channel is the one absolute residue, and we explain why it is benign (Section 3.4); learnable phase maps generalize how phases are produced (Section 3.5); Section 3.6 assembles the recipe. Throughout, a running example helps: an input token whose patch sees the corner of a table in view 1 *writes*; a target token whose ray passes through the same corner, viewed from view 2, *reads* and should retrieve what was written.

### 3.2 The gate-correction channel: the input rotary

**What it is.** The gate-correction channel is the third term of the display: the first-layer updates \(\Delta W_0, \Delta W_2\) store rank-one payloads indexed by raw keys \(k_i\), and the query retrieves them with weight \(\langle q_j, k_i\rangle\). The inner product lives in the input space \(\mathbb{R}^d\), before any nonlinearity—structurally identical to an attention logit.

**How to rotate it.** Because the interaction is already an inner product on \(q\) and \(k\), the classical rotary applies verbatim. Give every token phases derived from its Plücker coordinates—for now, an axis-aligned frequency ladder with \(F\) frequencies per coordinate, hence \(6F\) phase planes in total: \(\theta_i^{(a,f)} = \omega_f\, \pi_i^{(a)}\) for coordinate \(a \in \{1,\dots,6\}\) and frequency \(f \in \{1,\dots,F\}\), where \(\omega_1, \dots, \omega_F\) is a geometric sequence of frequencies, as in RoPE (Section 3.5 generalizes this)—and rotate queries and keys after the \(\ell_2\) normalization of \(q\) and \(k\):

$$
\tilde q_j \;=\; q_j\, R(\theta_j), \qquad \tilde k_i \;=\; k_i\, R(\theta_i).
$$

The layer then proceeds *unchanged*: the write uses \(\tilde k_i\), the read uses \(\tilde q_j\); values, learning rates, and the update kernel are untouched. We call this the **input rotary**.

**Why it stays relative.** Substituting into the gate-correction weight and using orthogonality of \(R\),

$$
\big\langle \tilde q_j, \tilde k_i \big\rangle \;=\; q_j\, R(\theta_j)\,R(\theta_i)^\top k_i^\top \;=\; \big\langle q_j,\; k_i\, R(\theta_i - \theta_j) \big\rangle ,
$$

so this channel depends on camera geometry only through the phase differences \(\theta_i - \theta_j \propto \pi_i - \pi_j\): the relative Plücker offset between the writing ray and the reading ray. In the running example, the table-corner writer and the table-corner reader have small \(\pi_i - \pi_j\) (bounded by scene radius times their parallax, Section 2.4), so their phases nearly agree at every frequency simultaneously, and retrieval is strong—regardless of where in the world the scene happens to sit.

One honest caveat, which is also the bridge to the next subsection. The input rotary is not new as a mechanism: it coincides with the fast-weight RoPE the LaCT authors already use in language and video (our contribution at this site is the 6D Plücker instantiation and, more importantly, identifying *which channel* it actually relativizes). And it conditions only the *smaller* channel. The rotated inputs do also enter the value-retrieval term, but there they pass through the SwiGLU nonlinearity, and \(\langle h(\tilde q_j), h(\tilde k_i)\rangle\) admits no cancellation—the rotation cannot pass through \(\mathrm{silu}\). The dominant channel needs its own rotary, at its own site.

### 3.3 The value-retrieval channel: the hidden rotary

**What it is.** The value-retrieval channel is the second term of the display and the layer's main business: \(\Delta W_1\) stores each value \(v_i\) on the hidden direction \(h(k_i)^\top\), and the query retrieves \(v_i\) with weight \(\langle h(q_j), h(k_i)\rangle\). The inner product lives in the *hidden* space \(\mathbb{R}^{d_h}\), between the SwiGLU activation and the output matrix \(W_1\). We repeat the central point a third time, now at the mechanism level: this conditioning site is invisible from attention, because in attention the similarity score multiplies the value directly—there is no intermediate representation between logits and values that one could rotate. The site exists only because test-time training inserts a learned nonlinear network where attention has a bare softmax.

**How to rotate it.** The lemma tells us the cure need not fight the nonlinearity—it can act *after* it. Give every token a second set of phases \(\phi_i \in \mathbb{R}^{P_h}\), again linear in \(\pi_i\) (Section 3.5). Unlike the input rotary's per-coordinate ladder, the hidden phases are not organized per Plücker coordinate: \(P_h\) counts the *total* number of hidden phase planes, not a per-coordinate count. Insert an orthogonal rotation of the hidden activation between \(h\) and \(W_1\), on both write and read. The fast-weight network becomes

$$
f_W(x;\, \phi) \;=\; \big(\, h(x)\, R_h(\phi) \,\big)\, W_1 ,
$$

where \(R_h(\phi) \in \mathrm{SO}(d_h)\) rotates \(P_h\) phase planes (\(2P_h\) of the \(d_h\) hidden dimensions) and leaves the rest untouched. The write step differentiates through this rotation: since \(R_h\) is orthogonal and linear, the backward pass simply multiplies the gradient arriving at the hidden layer by the inverse (transpose) rotation—one extra line in LaCT's fused GPU kernel. The update and the read become

$$
\Delta W_1 \;\propto\; \sum_i lr_i\, \big(h(\tilde k_i)\, R_h(\phi_i)\big)^{\!\top} v_i ,
\qquad
o_j \;=\; \big(h(\tilde q_j)\, R_h(\phi_j)\big)\, \big(W_1^0 + \Delta W_1\big).
$$

In words: each value is now stored at a phase-tagged hidden address, and each reader looks it up with its own phase-tagged address. We call this the **hidden rotary**. It is a change to where the memory stores and looks things up—the hidden *address*—not to what it stores: values, learning rates, Muon orthogonalization, and weight-column renormalization are untouched.

**Why it stays relative.** Contracting the read against the update, the retrieval weight for value \(v_i\) becomes

$$
\big\langle\, h(\tilde q_j)\, R_h(\phi_j),\;\; h(\tilde k_i)\, R_h(\phi_i) \,\big\rangle
\;=\;
h(\tilde q_j)\, R_h(\phi_j)\, R_h(\phi_i)^\top h(\tilde k_i)^\top
\;=\;
\big\langle\, h(\tilde q_j),\; h(\tilde k_i)\, R_h(\phi_i - \phi_j) \,\big\rangle ,
$$

because the two orthogonal rotations meet inside the inner product—the same cancellation as the RoPE identity, in the same writer-minus-reader order as Sections 2.2 and 3.2, now firing one layer deep. The hidden phases enter the dominant channel *only* through the difference \(\phi_i - \phi_j \propto \pi_i - \pi_j\), no matter what the nonlinearity did upstream: the rotation is applied after \(h\), so \(h\) never gets the chance to scramble it. In the running example, the table corner's appearance is deposited on a phase-tagged hidden direction by view 1; the view-2 reader's own phase tag differs only by the small relative ray offset, so the deposited content is retrieved with nearly full weight. Note also that the two rotary sites are independent by construction—each relativizes its own channel and neither interferes with the other—which is exactly the additivity we later observe empirically.

### 3.4 The initial-readout channel: the absolute residue

**What it is.** By the lemma, the only path from the query to the output that does not pass through an inner product with a key is the pre-update weights: terms such as \(h(\tilde q_j)\, W_1^0\), where the absolute phases of token \(j\) meet the fixed slow weights (Section 2.3) and nothing cancels them.

**Why it is benign.** No rotation can relativize this channel, but two properties of the TTT setting defuse it. First, at the start of training the columns of the learned initial state \(W_0^0, W_2^0\) are isotropic Gaussian, a distribution invariant under orthogonal maps: turning the rotaries on leaves the network's forward statistics exactly unchanged at the start of training, and the slow weights simply co-adapt to the rotated input distribution thereafter. Second—and this argument is particular to test-time training—the fast weights are a *per-scene* state, rebuilt from scratch for every scene (Section 2.3), and all rays are expressed in a canonical scene frame (fixed per scene as described in Section 2.1). Across scenes there is no shared global frame whose absolute phases the slow weights could overfit; "absolute" here already means "relative to the per-scene frame choice." Attention enjoys no such protection, since its shared slow weights mediate all token interactions; the per-scene memory is one more way in which the fast-weight setting is not merely attention-with-extra-steps.

### 3.5 Learnable phase maps

So far both rotary sites derive phases from an axis-aligned ladder: each phase plane listens to exactly one Plücker coordinate at one frequency. There is no reason the useful phase directions in a 6D coordinate space should be the coordinate axes—direction and moment components of a ray are geometrically coupled. We therefore let the network learn the map from coordinates to phases, subject to the one property the theory needs: linearity.

For each rotary site, replace the fixed ladder by

$$
\theta_i \;=\; \pi_i\, (\Omega_0 + \Delta\Omega)^\top ,
$$

where, consistent with our row-vector convention, the coordinates \(\pi_i\) enter as a row vector and each row of the map produces one phase. Here \(\Omega_0\) is the fixed axis-aligned ladder written as a matrix (each row selects one coordinate at one frequency, in units of the canonical scene scale), and \(\Delta\Omega\) is a learnable matrix of the same shape, initialized with small random entries of scale 0.1 (the *tilt-0.1* initialization; zero initialization also works but tilting slightly off-axis helps symmetry breaking). Relativity is preserved *exactly*, because for any linear map,

$$
\theta_i - \theta_j \;=\; (\pi_i - \pi_j)\,(\Omega_0 + \Delta\Omega)^\top:
$$

phase differences remain a fixed linear function of coordinate differences, so every cancellation in Sections 3.2–3.3 goes through unchanged, at initialization and after any amount of training. Geometrically, each phase plane's frequency vector—a row of \(\Omega_0 + \Delta\Omega\)—becomes a learnable direction in Plücker space: a plane can learn to respond to, say, a combination of vertical parallax and one moment component, rather than to a single axis. One boundary condition follows directly from the formula and is worth stating: for a *one-dimensional* coordinate (e.g., a text token index), a linear map can only rescale the existing frequency ladder, so learnable phase maps are degenerate in 1D and can only help for multi-dimensional coordinates such as 6D camera rays or 3D video grids. Our LLM experiments confirm exactly this prediction.

### 3.6 Stability, overhead, and the full recipe

**Orthogonality is load-bearing.** Every operation we add is a rotation. Rotations preserve norms exactly, so the \(\ell_2\) normalization of \(q\) and \(k\), the scale of the inner-loop loss, the per-token learning rates, Muon orthogonalization (Newton–Schulz), and weight-column renormalization all behave identically to the baseline—the encodings commute with everything downstream of where they are inserted. This is not a nicety but a hard requirement: non-orthogonal alternatives (e.g., transplanting PRoPE's projective transforms) rescale tokens, act as uncontrolled perturbations of write strengths, and fail in our experiments. Norm distortion is an absolute effect that no read-time cancellation can remove.

**Overhead.** The input rotary is an elementwise operation on \(q\) and \(k\); the hidden rotary is an elementwise operation on the hidden activation plus an inverse rotation in the kernel's backward pass; the phase maps are computed once per forward pass from the per-token rays already produced by the input pipeline. Total cost: under 0.1% additional FLOPs and 0.01% additional parameters, with no new losses, schedules, or checkpoint-format changes. When all phases vanish, the layer reduces exactly to the pose-blind baseline, so the baseline is contained in the hypothesis class.

**The recipe (PRA).** Our final camera instantiation, *Plücker Rotary Addressing*, applies at every TTT layer: (i) the input rotary on post-normalization \(q\) and \(k\) with \(F = 21\) frequencies per Plücker coordinate—that is, \(6 \times 21 = 126\) phase planes, occupying 252 of the input dimensions; (ii) the hidden rotary with \(P_h = 42\) phase planes in total (\(2P_h = 84\) hidden dimensions), covering roughly half of the hidden dimensions; (iii) learnable phase maps with tilt-0.1 initialization at both sites. The general pattern—one rotary per inner-product channel of the fast-weight readout, phases linear in any coordinate system—is what we call rotary fast-weight addressing; text and video instances differ only in the choice of \(\pi\). The design is a direct transcription of the display in Section 3.1: one relative encoding per channel, including the one channel—hidden-space value retrieval—that attention never had to offer.