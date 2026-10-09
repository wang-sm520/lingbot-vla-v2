import pytest
import torch

from lingbotvla.models.vla.lingbot_vla.qwen2_action_expert import Qwen2FusedExperts
from lingbotvla.ops.robby_moe import robby_moe_forward


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA MoE inference regression")
@pytest.mark.parametrize("num_tokens", [1, 51])
def test_repeated_moe_inference_with_reused_workspace_is_deterministic(num_tokens):
    torch.manual_seed(42)
    experts = Qwen2FusedExperts(32, 768, 512).cuda().bfloat16().eval()
    hidden = torch.randn(num_tokens, 768, device="cuda", dtype=torch.bfloat16)
    routes = torch.randn(num_tokens, 32, device="cuda").topk(4, dim=-1)
    routing_weights = routes.values.softmax(dim=-1).bfloat16()
    with torch.no_grad():
        workspace = experts._get_robby_moe_workspace(hidden, 4)
        arguments = (
            hidden,
            routing_weights,
            routes.indices,
            experts.gate_proj,
            experts.up_proj,
            experts.down_proj,
        )
        expected = robby_moe_forward(*arguments, workspace=workspace).clone()
        for _ in range(12):
            actual = robby_moe_forward(*arguments, workspace=workspace)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
