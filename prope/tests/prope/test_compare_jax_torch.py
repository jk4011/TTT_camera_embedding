# MIT License
#
# Copyright (c) Authors of
# "PRoPE: Projective Positional Encoding for Multiview Transformers"
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import jax
import numpy as np
import pytest
import torch

# Enable highest precision
jax.config.update("jax_enable_x64", True)
torch.set_default_dtype(torch.float64)

from prope.jax import PropeDotProductAttention as PropeDotProductAttentionJax
from prope.torch import PropeDotProductAttention as PropeDotProductAttentionTorch


@pytest.mark.parametrize("no_intrinics", [True, False])
def test_compare_jax_torch(no_intrinics: bool):
    torch.manual_seed(42)
    cameras = 3
    patches_x = 8
    patches_y = 8
    image_width = 128
    image_height = 128

    batch = 2
    seqlen = cameras * patches_x * patches_y
    num_heads = 4
    head_dim = 16

    q = torch.randn(batch, num_heads, seqlen, head_dim)
    k = torch.randn(batch, num_heads, seqlen, head_dim)
    v = torch.randn(batch, num_heads, seqlen, head_dim)

    viewmats = torch.eye(4).repeat(batch, cameras, 1, 1)

    if no_intrinics:
        Ks = None
    else:
        Ks = torch.rand(batch, cameras, 3, 3)

    out_jax = PropeDotProductAttentionJax(
        head_dim=head_dim,
        cameras=cameras,
        patches_x=patches_x,
        patches_y=patches_y,
        image_width=image_width,
        image_height=image_height,
    )(
        q.permute(0, 2, 1, 3).numpy(),
        k.permute(0, 2, 1, 3).numpy(),
        v.permute(0, 2, 1, 3).numpy(),
        viewmats=viewmats.numpy(),
        Ks=Ks.numpy() if Ks is not None else None,
    )
    out_torch = PropeDotProductAttentionTorch(
        head_dim=head_dim,
        cameras=cameras,
        patches_x=patches_x,
        patches_y=patches_y,
        image_width=image_width,
        image_height=image_height,
    )(
        q,
        k,
        v,
        viewmats=viewmats,
        Ks=Ks,
    )

    np.testing.assert_allclose(
        out_jax, out_torch.permute(0, 2, 1, 3).numpy(), atol=1e-4, rtol=1e-4
    )


if __name__ == "__main__":
    test_compare_jax_torch()
