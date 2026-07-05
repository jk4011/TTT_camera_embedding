# Rotary Fast-Weight Addressing: Camera and Positional Conditioning for Test-Time-Training Layers

## Abstract

Test-time-training (TTT) layers replace attention with fast-weight networks—small MLPs written by gradient descent during the forward pass—and scale linearly in sequence length. In multi-view generation and novel view synthesis, every layer that exchanges information across views must know each token's camera. For attention, such camera conditioning is mature: a family of camera-conditioned attention variants (CaPE, GTA, PRoPE, RayRoPE) exploits attention's bilinear logits. TTT layers are nonlinear SwiGLU networks, so no conditioning method exists for them. We prove a simple lemma: because fast-weight updates are sums of outer products, every interaction between a query and update-written content passes through inner products—exactly the algebraic hook rotary encodings need. Expanding the layer's read operation exposes two such channels, hence two rotary sites: an input rotary on queries/keys, which matches what prior fast-weight models already do, and a hidden rotary between the SwiGLU activation and the output weights—the dominant site, as it carries the signal that performs the write and the larger share of the gain. The hidden rotary has no counterpart in attention, which has no hidden layer between its matching scores and its values. We call the general pattern rotary fast-weight addressing. Its camera instantiation, Plücker Rotary Addressing (PRA), adds learnable linear phase maps—which preserve the relative-encoding property exactly—and uses Plücker ray coordinates. PRA delivers +1.23 dB PSNR on RE10K (RealEstate10K) novel view synthesis (21.75→22.97). On a 200M LLM, the hidden rotary cuts perplexity 19.32→19.13—matching the entire NoPE (no positional encoding)→RoPE gap, on top of RoPE. The method is neutral on a video-diffusion finetune where attention layers still handle most cross-token interaction: it helps only when the TTT layer itself carries that traffic. Overhead: <0.1% FLOPs, +0.01% parameters.

---

## 1. Introduction

Attention is the workhorse of sequence modeling, but its cost grows quadratically with sequence length. Test-time-training (TTT) layers are a linear-cost alternative built on a different mechanism: instead of comparing every query against every key, a TTT layer maintains a small *fast-weight network*—an MLP whose weights are updated by gradient descent *during the forward pass*. Key–value pairs are *written* into the weights by a gradient step, and queries later *read* the weights by a plain forward pass. LaCT ("large-chunk" TTT) made this recipe practical at scale by writing many tokens per gradient step with a large SwiGLU fast-weight network, and showed that a LaCT-based novel-view-synthesis model matches full attention at a fraction of the cost.

For attention, injecting camera geometry is by now a mature craft. Camera-conditioned attention variants—CaPE, GTA, PRoPE, and RayRoPE—all exploit the same structural fact: every query–key interaction in attention is *exactly* a bilinear form. Rotate the query by its camera's transform and the key by its camera's transform, and the two rotations meet inside the attention logit and cancel into a function of the *relative* pose alone. TTT layers offer no such foothold. The fast-weight network is a SwiGLU MLP, its weights are moved by a gradient step, and there is no logit anywhere to intercept—the transfer of relative encodings to TTT is, on its face, blocked by nonlinearity. The gap is not hypothetical: in LaCT-based view synthesis, camera information enters the network exactly once, as a ray map (per-pixel camera-ray origins and directions) concatenated to the pixels at patchification, and the TTT layer—the only module that exchanges information across views—is completely pose-blind.

Our starting point is not a design but a computation. We take the LaCT update equations as given and simply expand, in four small steps, what a query actually touches when it reads the freshly written weights. The expansion rests on one algebraic fact: a fast-weight update is a *sum of outer products* (that is what a gradient of a linear layer is), and—with the row-vector conventions fixed in §2—a query can meet an outer-product update only through an inner product. Carrying this through the SwiGLU network yields a three-term readout. The first term is an *init readout*: the query passing through the pre-write weights. The second is a *value-retrieval channel*: each stored value is retrieved with a weight equal to an inner product of *hidden* activations. The third is an *input-correction channel*: a correction weighted by an inner product taken in input space, directly between query and key. We package this as a lemma: every interaction between a query and update-written content, at any order of the expansion, passes through one of these two inner products. A nonlinear TTT layer is, despite appearances, an inner-product-addressed memory.

Once the readout is written this way, the method designs itself: an inner product is precisely the object the RoPE identity relativizes—i.e., makes depend only on coordinate differences—and there are exactly two of them, so there are exactly two rotary sites. The first site, an *input rotary* on queries and keys after their \(\ell_2\)-normalization (LaCT \(\ell_2\)-normalizes queries and keys per token), relativizes the input-correction channel; it coincides with what prior work (including the LaCT authors' own fast-weight RoPE for language and video) already does, and we claim no novelty for it. The second site is new: a *hidden rotary* that rotates the SwiGLU hidden activation just before it meets the output projection, on write and on read, with the inverse rotation applied during the inner-loop backward pass (the backward pass of the fast-weight gradient step that implements the write). The hidden rotary relativizes the value-retrieval channel—the dominant one: it is the direct descendant of the write objective (§3.1), and it accounts for the larger share of the ablation gain (§3.2). And this channel simply does not exist in attention: attention routes its matching scores straight into a convex combination of values, with no intermediate representation in between, whereas a TTT layer stores its values behind a learned nonlinear encoder. The hidden rotary is therefore a conditioning channel *specific to fast weights*, not a port of an attention technique.

The two sites fix *where* to rotate; a third component improves *what angle* to rotate by. Standard RoPE assigns each rotation plane a frequency along one fixed coordinate axis. We replace this assignment with a *learnable linear phase map* \(\theta = \Omega\, p\) from the coordinate vector \(p\) to all phases. Because the map is linear, the relative-encoding property is preserved *exactly*—\(\theta_i - \theta_j = \Omega(p_i - p_j)\)—while the per-plane phases are freed to leave the coordinate axes and respond to learned combinations of the coordinates (for a camera ray, say, a screw: a combined rotation–translation pattern along the ray that mixes the ray's direction with its moment, both defined in §2.3). This helps precisely when coordinates are multi-dimensional (6D rays, 3D video grids) and is degenerate for 1D text, a boundary our experiments confirm.

We call the general pattern—input rotary, hidden rotary, and learnable phase maps, applied in every TTT layer—**rotary fast-weight addressing**; its camera instantiation is **Plücker Rotary Addressing (PRA)**. Instantiated with Plücker ray coordinates, PRA lifts LaCT-LVSM (LVSM is a feed-forward view-synthesis transformer; LaCT-LVSM replaces its attention with TTT layers) on RE10K from 21.745 ± 0.196 to 22.971 ± 0.088 PSNR (+1.226 dB; LPIPS 0.2929 → 0.2613) at the full training budget. The three components are complementary—each adds gain on top of the others—and we defer the per-component arithmetic to the ablations, quoted alongside the relevant components in §3.2–§3.3 and tabulated in the experiments section. On a 200M-parameter LLM the hidden rotary alone reduces perplexity from 19.32 to 19.13—a gain equal to the entire NoPE→RoPE gap (19.56 → 19.32), obtained *on top of* RoPE. On a video-diffusion finetune the method is neutral, which we report as an honest boundary: in that model the attention layers still handle most cross-token interaction, and gains require the TTT layer's fast-weight memory to carry enough of that traffic. Across 26 runs we also document what fails—projective transplants (porting PRoPE's projective, non-orthogonal matrices; §3.5), value transport (geometrically transporting stored values between views), feature injection (injecting pose features into the layer's inputs)—supporting our summary finding: *the attention recipe does not transplant; the channel structure does.* The overhead is below 0.1% of FLOPs and +0.01% of parameters.

**Contributions.**

1. **Inner-product addressing lemma.** We show, by a four-step expansion of the LaCT readout, that a nonlinear SwiGLU fast-weight layer reads everything written into it exclusively through inner products—the single algebraic hook rotary encodings require.
2. **Two rotary sites, one of them new.** The lemma exposes exactly two rotatable channels. The input rotary recovers known practice; the *hidden rotary*, which relativizes the dominant value-retrieval channel, is—to our knowledge—the first conditioning channel with no attention analogue.
3. **Learnable phase maps.** A linear, learnable coordinate-to-phase map that provably preserves the relative-encoding property, with a characterized multi-dimensional applicability boundary.
4. **PRA and cross-domain evidence.** A camera instantiation via Plücker coordinates yielding +1.226 dB PSNR on RE10K; a −1.0% perplexity gain on a 200M LLM equal to the full NoPE→RoPE gap; a documented neutral video boundary; and negative-result forensics showing that direct transplants of attention recipes fail while the channel-structured approach succeeds.

---

## 2. Preliminaries

We fix notation for the three ingredients the method combines: the LaCT layer (§2.1), the RoPE identity (§2.2), and Plücker ray coordinates (§2.3). Throughout, tokens are **row vectors**: \(x \in \mathbb{R}^{1 \times d}\) multiplies weight matrices from the right, and we write \(\langle a, b\rangle = a\, b^\top\) for the inner product of two row vectors.

### 2.1 The LaCT layer: writing and reading fast weights

A TTT layer replaces attention by a small MLP—the **fast-weight network**—whose parameters are updated during the forward pass. LaCT uses a SwiGLU fast-weight network with parameters \(W = (W_0, W_2, W_1)\), where \(W_0, W_2 \in \mathbb{R}^{d \times d_h}\) and \(W_1 \in \mathbb{R}^{d_h \times d}\) (note the ordering follows LaCT: \(W_0\) and \(W_2\) project *in*, \(W_1\) projects *out*):

$$
f_W(x) \;=\; h(x)\, W_1,
\qquad
h(x) \;=\; \mathrm{silu}(x W_0) \odot (x W_2),
\tag{1}
$$

where \(\odot\) is the elementwise product, \(\mathrm{silu}(a) = a\,\sigma(a)\), and we call \(h(x) \in \mathbb{R}^{1\times d_h}\) the **hidden activation**; following standard SwiGLU naming, we call \(xW_0\) the **gate branch** and \(xW_2\) the **up branch**. Queries, keys, and values are produced per token by a learned projection, and \(q, k\) are \(\ell_2\)-normalized per token. The layer runs in two phases.

**Write.** A **chunk** of tokens \(\{(k_i, v_i)\}_{i \in \mathcal{I}}\) (in view synthesis: all input-view tokens at once) is written by one gradient step on a dot-product write objective (LaCT's linearized regression loss). The objective is built so that descending it performs the write: one gradient step increases \(\langle v_i, f_W(k_i)\rangle\), i.e., pushes the network's output at \(k_i\) toward \(v_i\)—that is the write. Concretely, \(\mathcal{L}(W) = -\sum_{i} \eta_i \,\langle v_i, f_W(k_i)\rangle\), where \(\eta_i > 0\) is a per-token write strength predicted from token content. Because each weight matrix is linear in its input, the gradient step is a **sum of outer products**—this is the fact everything below builds on:

$$
\Delta W_0 \;\propto\; \sum_{i \in \mathcal{I}} \big(\eta_i\, k_i\big)^{\!\top} g_i,
\qquad
\Delta W_2 \;\propto\; \sum_{i \in \mathcal{I}} \big(\eta_i\, k_i\big)^{\!\top} u_i,
\qquad
\Delta W_1 \;\propto\; \sum_{i \in \mathcal{I}} \big(\eta_i\, h(k_i)\big)^{\!\top} v_i,
\tag{2}
$$

where \(g_i, u_i \in \mathbb{R}^{1 \times d_h}\) are the backpropagated signals into the gate and up branches (functions of \(k_i\), \(v_i\), and the pre-write weights). In words: the two input matrices store correction signals along the raw key direction \(k_i\); the output matrix stores each value \(v_i\) along the direction of its key's hidden activation \(h(k_i)\). LaCT additionally orthogonalizes updates with a few Newton–Schulz iterations (Muon, an orthogonalized update rule) and renormalizes weight columns; both operations reweight tokens but keep each update a sum of rank-one terms in the \(k_i\)- (resp. \(h(k_i)\)-) directions, which is all our analysis needs.

**Read.** Every token \(j\) then queries the written weights with an ordinary forward pass: \(o_j = f_{W + \Delta W}(q_j)\).

Note what is absent: nothing in (1)–(2) knows which camera, frame, or position a token came from. The layer is coordinate-blind.

### 2.2 The RoPE identity

Rotary position embeddings condition attention by rotating features in 2D planes. The entire mechanism is one identity, which we state per plane: in (3), \(q, k \in \mathbb{R}^{1 \times 2}\) are the components of a query and a key within a single rotation plane, and \(R(\theta) \in \mathrm{SO}(2)\) is a plane rotation:

$$
\big\langle\, q\,R(\theta_j),\; k\,R(\theta_i) \,\big\rangle
\;=\;
\big\langle\, q,\; k\,R(\theta_i - \theta_j) \,\big\rangle .
\tag{3}
$$

Rotations applied on the two sides of an **inner product** meet and collapse into a function of the phase *difference* alone: absolute inputs, relative interactions. The identity extends from one plane to the full feature dimension \(\mathbb{R}^d\) block-diagonally—many planes rotated at once—which is exactly the lift we construct in (10). And the identity says nothing about attention specifically: it applies to any inner product we can find. Our whole method consists of finding the inner products inside a TTT layer.

### 2.3 Plücker ray coordinates

For the camera setting, each token corresponds to an image patch, and we attach to it the viewing ray of the patch center: origin \(\mathbf{o}_i \in \mathbb{R}^3\) and unit direction \(\mathbf{d}_i \in \mathbb{R}^3\), expressed in a canonical scene frame (mean camera pose normalized to identity, positions scaled into the unit box). The ray's **Plücker coordinates** are

$$
p_i \;=\; (\mathbf{d}_i,\; \boldsymbol{\mu}_i) \in \mathbb{R}^6,
\qquad
\boldsymbol{\mu}_i \;=\; \mathbf{o}_i \times \mathbf{d}_i .
\tag{4}
$$

The moment \(\boldsymbol{\mu}_i\) is invariant to sliding the origin along the ray, so \(p_i\) identifies the 3D *line* itself. Crucially for multi-view synthesis, two rays through the same scene point \(\mathbf{x}\) satisfy \(\|\boldsymbol{\mu}_i - \boldsymbol{\mu}_j\| \le \|\mathbf{x}\|\,\|\mathbf{d}_i - \mathbf{d}_j\|\): rays observing the same content from nearby directions are provably close in Plücker space. A memory addressed by \(p_i\) can therefore associate content across views by geometry, not just by appearance.

---

## 3. Method

Our expository route is deliberately bottom-up. We do not begin from a design; we begin from the update equations (2) and expand, in four small steps, what the read \(o_j = f_{W+\Delta W}(q_j)\) actually computes (§3.1). The expansion reveals exactly two inner-product channels, which the RoPE identity (3) can relativize—so exactly two rotary sites follow, one known and one new (§3.2). We then make the phases learnable without breaking the relative-encoding property (§3.3), assemble the camera recipe (§3.4), and account for stability and cost (§3.5).

### 3.1 Reading the memory, four steps at a time

Everything below is a consequence of a one-line algebraic fact, which we state first so the reader can watch it do all the work. **A row vector meeting an outer product contracts through an inner product:**

$$
x\,\big(a^\top b\big) \;=\; \big(x\, a^\top\big)\, b \;=\; \langle x, a\rangle\; b .
\tag{5}
$$

An outer product cannot hand a row vector anything *except* an inner product times a fixed payload. Since every fast-weight update in (2) is a sum of outer products, every time the query touches an update, (5) fires. One notational reminder before we begin (the indexing is LaCT's, from §2.1): \(W_0\) and \(W_2\) are the *input* matrices—gate and up—and \(W_1\) is the *output* matrix.

We work in the **one-chunk regime**: all input tokens are written in a single gradient step taken at the pre-write weights \(W^0 = (W_0^0, W_2^0, W_1^0)\), then every token reads. (With several chunks, each chunk's write is still a sum of outer products, so the same argument applies chunk by chunk.) Write \(W' = W^0 + \Delta W\) for the post-write weights and \(h^0(x) = \mathrm{silu}(xW_0^0)\odot(xW_2^0)\) for the hidden activation under pre-write weights. The read is \(o_j = h'(q_j)\, W_1'\) with \(h'(q_j) = \mathrm{silu}(q_j W_0') \odot (q_j W_2')\).

**Step 1 — the output matrix \(W_1\) (exact).** Distribute \(W_1' = W_1^0 + \Delta W_1\) and apply (5) to \(\Delta W_1 = \sum_i (\eta_i h^0(k_i))^\top v_i\); here and below we absorb the inner-loop step size into \(\eta_i\), which turns the proportionalities of (2) into equalities:

$$
o_j \;=\; h'(q_j)\, W_1^0 \;+\; \sum_{i \in \mathcal{I}} \eta_i\, \big\langle h'(q_j),\, h^0(k_i)\big\rangle\; v_i .
\tag{6}
$$

No approximation was made. Read the second term aloud: the stored value \(v_i\) is retrieved with a weight equal to an inner product of **hidden activations**. This is the layer's analogue of an attention score—but it lives one layer deep, in hidden space, not at the input.

**Step 2 — the input matrices (exact).** The query also meets the updates to \(W_0\) and \(W_2\). Applying (5) to \(\Delta W_0 \propto \sum_i (\eta_i k_i)^\top g_i\), and likewise for \(\Delta W_2\):

$$
q_j W_0' \;=\; q_j W_0^0 + \sum_{i} \eta_i\, \langle q_j, k_i\rangle\, g_i,
\qquad
q_j W_2' \;=\; q_j W_2^0 + \sum_{i} \eta_i\, \langle q_j, k_i\rangle\, u_i .
\tag{7}
$$

Again exact, and again the same shape: the query couples to the first-layer writes only through the scalar \(\langle q_j, k_i\rangle\).

**Step 3 — linearize the nonlinearity (the only approximation).** The corrections in (7) sit inside \(\mathrm{silu}\). Taylor-expanding, \(\mathrm{silu}(a + \varepsilon) = \mathrm{silu}(a) + \mathrm{silu}'(a)\odot\varepsilon + O(\varepsilon^2)\), gives

$$
h'(q_j) \;=\; h^0(q_j) \;+\; \sum_{i} \eta_i\,\langle q_j, k_i\rangle\; e_{ij} \;+\; O(\Delta W^2),
\tag{8}
$$

where \(e_{ij} \in \mathbb{R}^{1\times d_h}\) collects \(\mathrm{silu}'\)-modulated combinations of \(g_i, u_i\) and the pre-write activations. The point is not the internal form of \(e_{ij}\)—it is that the query's coupling to it has *already* been reduced to \(\langle q_j, k_i\rangle\) by Step 2, and the nonlinearity cannot undo a scalar.

**Step 4 — assemble.** Substitute (8) into (6) and sort terms by their order in \(\Delta W\). Note what the substitution does to the value-retrieval term: it replaces \(h'(q_j)\) by \(h^0(q_j)\), because the cross terms—the update part of (8) meeting \(\Delta W_1\)—carry two update factors and join the \(O(\Delta W^2)\) remainder. Writing \(c_{ij} = e_{ij} W_1^0\), the readout is

$$
o_j
\;=\;
\underbrace{h^0(q_j)\, W_1^0}_{\text{init readout}}
\;+\;
\underbrace{\sum_{i} \eta_i\, \big\langle h^0(q_j),\, h^0(k_i)\big\rangle\, v_i}_{\text{value-retrieval channel (hidden space)}}
\;+\;
\underbrace{\sum_{i} \eta_i\,\langle q_j,\, k_i\rangle\; c_{ij}}_{\text{input-correction channel (input space)}}
\;+\; O(\Delta W^2).
\tag{9}
$$

Interpretation, term by term. The *init readout* is what the layer would output with no writes at all—it carries no cross-token information. The *value-retrieval channel* is the memory proper: values are recalled weighted by hidden-space similarity; it is the dominant channel, because it is the direct descendant of the write objective. The *input-correction channel* collects the corrections from both first-layer writes \(\Delta W_0\) and \(\Delta W_2\)—the gate and up projections alike—each weighted by an inner product taken directly in input space; it is a smaller refinement. The \(O(\Delta W^2)\) remainder collects terms with two or more update factors; it is small (one gradient step, weight normalization), and—more importantly—the truncation is presentational, not load-bearing: higher-order terms are produced by *repeated* applications of (5), so the query enters them, again, only through inner products with keys.

We record the structural conclusion:

> **Lemma 1 (inner-product addressing).** *In the one-chunk regime, every interaction between a query \(q_j\) and content written by a key \(k_i\)—through any of \(\Delta W_0, \Delta W_2, \Delta W_1\), at any order of the expansion—enters exclusively through one of the two inner products \(\langle q_j, k_i\rangle\) or \(\langle h'(q_j), h^0(k_i)\rangle\); the pre-write weights \(W^0\) are the only path by which \(q_j\) affects the output not through an inner product with some key.*

> **Remark (bookkeeping of \(h^0\) versus \(h'\)).** The hidden-space inner product pairs different weight versions on its two sides: the written key enters through the pre-write map \(h^0\), and the reading query through the post-write map \(h'\), exactly as in the exact expression (6). In the truncated readout (9), the substitution of Step 4 replaces \(h'\) by \(h^0\) on the query side at cost \(O(\Delta W^2)\). Nothing in the lemma depends on which version appears: both are inner products at the same hidden-space site, which is all the rotary construction of §3.2 needs.

The lemma says the TTT layer, though nonlinear, reads its written memory the way attention reads its values: through inner products. That is the one algebraic situation the RoPE identity (3) was built for. But note the asymmetry with attention already visible in (9): attention has *one* score, in input space; the fast-weight layer has *two*, and the dominant one sits in the hidden space of the fast-weight network—a space attention does not possess.

### 3.2 Two channels, two rotary sites

Equation (9) contains exactly two inner products, so identity (3) can be applied in exactly two places. Each application is a **rotary site**: attach to every token \(i\) a coordinate vector \(p_i\), map it to phases, and rotate the two sides of the inner product. From here on we let \(p_i \in \mathbb{R}^m\) denote a *generic* coordinate vector—a deliberate generalization of the symbol, of which the Plücker ray of §2.3 is the \(m=6\) instance (\(m=3\) for video grids, \(m=1\) for text positions). We take the sites in order of familiarity.

**Site 1: the input rotary.** For \(F\) rotation planes, let \(\theta_i \in \mathbb{R}^F\) be the phases of token \(i\) (their construction from \(p_i\) is the subject of §3.3), and collect them into the block-diagonal rotation

$$
G_i \;=\; \mathrm{blockdiag}\Big(R\big(\theta_i^{(1)}\big), \dots, R\big(\theta_i^{(F)}\big),\; I_{d - 2F}\Big) \;\in\; \mathrm{SO}(d),
\tag{10}
$$

which rotates the first \(2F\) feature dimensions and leaves the rest as a pure content channel. The **input rotary** applies, in every TTT layer, *after* the \(\ell_2\)-normalization of queries and keys,

$$
\tilde q_j \;=\; q_j\, G_j,
\qquad
\tilde k_i \;=\; k_i\, G_i,
\tag{11}
$$

and the layer proceeds unchanged: the write (2) stores \(\tilde k_i\) against \(v_i\), and reads use \(\tilde q_j\). By identity (3), the two rotations meet inside the inner product and cancel into a phase difference, making the input-correction channel relative:

$$
\langle \tilde q_j,\, \tilde k_i\rangle
\;=\;
q_j\, G_j\, G_i^\top\, k_i^\top ,
\qquad
G_j\, G_i^\top = \mathrm{blockdiag}\Big(R\big(\theta_j^{(f)} - \theta_i^{(f)}\big)\Big)_{f},
\tag{12}
$$

a function of the phase differences \(\theta_j - \theta_i\) alone. One might worry that rotated keys "contaminate" the stored weights—absolute phases really are present inside \(\Delta W\). That is the design, not a defect: \(\Delta W\) is where the phases are *stored*, and the read is the operation that *relativizes* them, because every read passes through the contraction (5) where \(G_j\) meets \(G_i^\top\). The input rotary coincides with the fast-weight RoPE the LaCT authors already use for language and video; we include it for completeness and claim no novelty.

But look at what the input rotary does *not* fix. The dominant value-retrieval channel now reads \(\langle h^0(\tilde q_j), h^0(\tilde k_i)\rangle\), and because \(h^0\) is nonlinear it does not commute with rotations: this weight is *not* a function of \(\theta_j - \theta_i\) alone. Rotating the inputs relativizes the minor channel and leaves the major one absolutely phased. This is exactly what our ablations show:[^1] the input rotary alone gains +0.41 dB and then saturates—further training does not grow the gain, because the dominant channel remains absolutely phased.

[^1]: All ablation numbers quoted in §3.2 are *configuration deltas*: each configuration is trained at a shorter budget than the headline run and measured against the no-rotary baseline, which is why the totals here sit below the headline +1.23 dB. The single phase-map number in §3.3 uses a different scheme, a *pairwise toggle*: the full recipe retrained with only the phase maps switched off and everything else held fixed. The two schemes give different estimates of the phase-map contribution—0.93 − 0.77 = 0.16 dB from configuration deltas versus +0.06 dB from the toggle—because they compare different pairs of independently trained runs, and run-to-run variance (compare the ± spreads on the headline numbers, ±0.196 and ±0.088) is of the same order as the discrepancy. The experiments section tabulates both schemes in full.

**Site 2: the hidden rotary.** Lemma 1 tells us where the remaining inner product lives: in hidden space, between the hidden activation and the output matrix \(W_1\). So we rotate *there*. We present the hidden rotary first in isolation—plain \(q_j\) and \(k_i\), no input rotary—both because the mechanism is cleanest there and because this is exactly the configuration of our hidden-rotary-alone ablation and of the LLM experiment. Let \(H_i \in \mathrm{SO}(d_h)\) be a block-diagonal rotation of the same form as (10), built from \(F_h\) planes with its own phases \(\theta_i^{h} \in \mathbb{R}^{F_h}\) derived from the same coordinates \(p_i\). The **hidden rotary** inserts \(H\) between \(h\) and \(W_1\), on write and on read; the next display shows both operations, with \(H\) appearing symmetrically on the two sides of the memory (superscripts as in §3.1: \(h^0\) under pre-write weights for the write, \(h'\) under post-write weights for the read):

$$
\textbf{write:}\quad
\Delta W_1 \;\propto\; \sum_{i} \big(\eta_i\, h^0(k_i)\, H_i\big)^{\!\top} v_i,
\qquad\quad
\textbf{read:}\quad
o_j \;=\; \big(h'(q_j)\, H_j\big)\, W_1' .
\tag{13}
$$

During the write's inner-loop backward pass, the gradient flowing from \(W_1\) back into the gate and up branches passes through \(H_i^\top\)—the inverse rotation—so the update remains the exact gradient of the modified network; this is a few-line change inside the update step. Now apply identity (3) at the new site: the two hidden rotations meet inside the retrieval weight and collapse into a relative factor,

$$
\big\langle h'(q_j)\, H_j,\; h^0(k_i)\, H_i \big\rangle
\;=\;
h'(q_j)\; H_j H_i^\top \;\, h^0(k_i)^\top ,
\tag{14}
$$

and \(H_j H_i^\top\) is block-diagonal in the phase differences \(\theta_j^h - \theta_i^h\): the dominant channel is now relative in the coordinates, *regardless* of the nonlinearity, because the rotation is applied downstream of \(h\), directly on the two sides of the inner product that Lemma 1 identified. Put differently: whatever the nonlinear map \(h\) does to its input, the hidden phases \(\theta^h\) can enter the retrieval weight only through the difference \(\theta_j^h - \theta_i^h\).

Composing the two sites is mechanical: with the input rotary of Site 1 also active, replace \(q_j, k_i\) by \(\tilde q_j, \tilde k_i\) throughout (13)–(14), and nothing in the derivation changes. Note that this does not contradict the Site-1 caveat that \(h\) fails to commute with rotations. The hidden phases \(\theta^h\) enter the retrieval weight *exactly* relatively, through the factor \(H_j H_i^\top\), no matter what \(h\) is; the input-rotary phases that sit inside \(h'(\tilde q_j)\) and \(h^0(\tilde k_i)\) are a separate, already-discussed effect—Site 1's absolute-phase remainder, which is benign for the same reasons given in the init-readout paragraph below.

This site is the core of the paper, and it is worth stating plainly why. In attention, the matching score multiplies the value immediately—softmax logits feed straight into a convex combination of \(v_i\); there is no representation *between* the score and the value that one could rotate. In a TTT layer there is: values are stored behind a learned nonlinear encoder, and retrieval happens by inner products of *encoded* (hidden) representations. The hidden rotary conditions that encoding. It is not a transplanted attention technique operating in a new venue; it is a conditioning channel that exists *only because* fast weights have a hidden layer—it has no attention analogue—and it took the bottom-up expansion (9) to see it. Consistently, our experiments find the two sites complementary and largely additive—+0.41 and +0.46 dB alone, +0.77 dB together, slightly below the sum—because they relativize different channels of (9); with the phase maps of §3.3 added, the full recipe reaches +0.93 dB at the same ablation budget.

**The init-readout residue.** By Lemma 1 the only non-relative path is through the pre-write weights: \(h'(\tilde q_j) H_j W_1^0\) and \(\tilde q_j W_0^0\) expose *absolute* phases. Two properties make this residue benign. First, at initialization the columns of \(W^0\) are isotropic Gaussian, whose distribution is invariant under orthogonal maps, so the encoding leaves forward statistics exactly unchanged at the start of training. Second, the fast weights are per-sequence state recomputed from scratch each scene, and rays are expressed in a canonical scene frame—so the "absolute" phases a scene writes are already in a per-scene canonical frame and carry no information that transfers across scenes; the shared slow weights only ever see the canonicalized distribution. There is no cross-scene pathway for memorizing absolute poses.

### 3.3 Learnable phase maps

It remains to specify how coordinates \(p_i \in \mathbb{R}^m\) become phases \(\theta_i \in \mathbb{R}^F\) (and \(\theta_i^h \in \mathbb{R}^{F_h}\)). Standard RoPE hard-wires each plane to one coordinate axis at one frequency: plane \(f\) gets \(\theta_i^{(f)} = \omega_f\, p_i^{(a_f)}\) for a fixed axis assignment \(a_f\) and a geometric frequency ladder \(\omega_f\). We keep the ladder as an initialization but make the whole coordinate-to-phase map a learnable **linear** map:

$$
\theta_i \;=\; \pi\,\big(\Omega^0 + \Delta\Omega\big)\, p_i,
\tag{15}
$$

where \(\Omega^0 \in \mathbb{R}^{F \times m}\) is the fixed axis-aligned ladder (each row a single frequency on a single coordinate) and \(\Delta\Omega\) is learnable. We initialize \(\Delta\Omega\) with small i.i.d. entries of scale 0.1 (**tilt-0.1 init**); zero init recovers standard RoPE exactly at the start of training.

Why linear, and why is this safe? Because the relative-encoding property in (12) and (14) needs only that phase *differences* depend on coordinate *differences*, and any linear map guarantees this identically:

$$
\theta_i - \theta_j \;=\; \pi\,\Omega\,(p_i - p_j)
\quad\text{for every } \Omega .
\tag{16}
$$

So the phase map can be trained freely—the relative-encoding property is preserved *exactly*, not approximately, throughout training. What the map buys is expressivity in the per-plane phases: a plane is no longer chained to a coordinate axis but can respond to a learned linear functional of the coordinates—for a Plücker ray, e.g., a screw-like combination of direction and moment. This can only matter when \(m > 1\): for 1D text positions, (15) reduces to a per-plane frequency rescale, which RoPE's ladder already spans. Our experiments confirm both sides of this prediction: as a pairwise toggle—the full recipe with only the phase maps switched off (see the footnote in §3.2)—phase maps add +0.06 dB on 6D camera rays, and they are neutral on 1D text.

### 3.4 The full recipe: Plücker Rotary Addressing

The general pattern—**rotary fast-weight addressing**—is: input rotary + hidden rotary + learnable phase maps, applied in every TTT layer. The camera instantiation, **PRA (Plücker Rotary Addressing)**, sets the coordinates to the Plücker ray of each token's patch center in the canonical scene frame, \(p_i = (\mathbf{d}_i, \boldsymbol{\mu}_i) \in \mathbb{R}^6\) as in (4), and uses:

- **input rotary**: \(F = 21\) planes on the \(\ell_2\)-normalized \(q, k\) (42 of the \(d\) input dimensions rotated, the rest content-only);
- **hidden rotary**: \(F_h = 42\) planes on the hidden activation (84 dimensions—roughly half the hidden width—rotated, the rest content-only);
- **phase maps**: separate maps (15) for the two sites, tilt-0.1 init.

By the moment bound of §2.3, two tokens in different views observing the same scene point \(\mathbf{x}\) have Plücker offsets bounded by \(\|\mathbf{x}\|\) times their directional parallax, and after the canonical normalization into the unit box, \(\|\mathbf{x}\|\) is at most the scene radius. By (16), the same bound then controls their phase offsets in *every* rotation plane simultaneously. The fast weights therefore behave as a memory over the scene's ray space: input views write appearance along their rays, and a target pixel reads along its own ray and retrieves content written by geometrically consistent rays—at every layer, without any change to the model's losses or training schedule.

### 3.5 Stability and overhead

**Orthogonality is load-bearing.** \(G_i\) and \(H_i\) are rotations, so they preserve the \(\ell_2\) norms of queries, keys, and hidden activations exactly. Consequently the calibration of the inner-loop loss, the per-token write strengths \(\eta_i\), the Muon orthogonalization, and the weight-norm constraint are numerically identical to the baseline—the encoding commutes with everything downstream. This is not a nicety: non-orthogonal alternatives (e.g., transplanting the camera matrices of the attention variant PRoPE, which are projective rather than orthogonal) rescale tokens, act as uncontrolled perturbations of write strengths, and fail in our experiments. The attention recipe does not transplant; the channel structure does.

**Placement.** The input rotary goes after \(\ell_2\)-normalization so norm preservation is exact. The hidden rotary is a rotation of \(h\) forward and its transpose on the returning inner-loop gradient—an exact-gradient, few-line modification of the update step. No new modules, losses, kernels, schedules, or checkpoint-format changes are introduced.

**Cost.** Each site is a fused elementwise rotation of \(2F\) (resp. \(2F_h\)) dimensions per token, computed from ray coordinates the input pipeline already produces. Total overhead is below 0.1% of layer FLOPs, and the phase maps add +0.01% parameters.