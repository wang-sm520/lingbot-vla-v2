import pytest
import torch
from torch.nn import functional

from lingbotvla.models.vla.lingbot_vla.flex_attention import (
    build_block_mask,
    flex_attention_forward,
    flex_attention_with_block_mask,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA autocast regression")
@pytest.mark.parametrize("cached", [False, True])
def test_flex_backward_under_bfloat16_autocast_matches_float32_attention(cached):
    torch.manual_seed(42)
    sequence_length = 19
    query = torch.randn(1, sequence_length, 2, 16, device="cuda", requires_grad=True)
    key = torch.randn(1, sequence_length, 1, 16, device="cuda", requires_grad=True)
    value = torch.randn(1, sequence_length, 1, 16, device="cuda", requires_grad=True)
    mask = torch.ones(1, sequence_length, sequence_length, device="cuda", dtype=torch.bool).tril()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if cached:
            block_mask = build_block_mask(mask, 2, sequence_length, sequence_length)
            actual = flex_attention_with_block_mask(query, key, value, block_mask, sequence_length)
        else:
            actual = flex_attention_forward(query, key, value, mask)
    actual_gradients = torch.autograd.grad(actual.square().mean(), (query, key, value))
    expected = (
        functional.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=mask[:, None],
            enable_gqa=True,
        )
        .transpose(1, 2)
        .reshape(1, sequence_length, 32)
    )
    expected_gradients = torch.autograd.grad(expected.square().mean(), (query, key, value))
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-5, rtol=1e-4)
