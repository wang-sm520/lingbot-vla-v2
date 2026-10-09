import copy
import hashlib
import json
import random
from itertools import islice
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from peft import get_peft_model_state_dict
from safetensors.torch import load_file
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel

from lingbotvla.airbot_lora import (
    ROOT,
    ReplayableSampler,
    add_adapters,
    build_optimizer,
    checkpoint_load_fingerprint,
    checkpoint_manifest,
    export_weights,
    load_checkpoint,
    read_configuration,
    record_batch_change,
    record_step_extension,
    save_checkpoint,
)


class TinyPolicy(PreTrainedModel):
    config_class = PretrainedConfig
    supports_gradient_checkpointing = True

    def __init__(self):
        super().__init__(PretrainedConfig())
        self.layer = nn.Module()
        self.layer.self_attn = nn.ModuleDict({name: nn.Linear(4, 4) for name in ("q_proj", "v_proj")})
        for name in ("state_proj", "action_in_proj", "action_out_proj", "action_time_mlp_in", "action_time_mlp_out"):
            setattr(self, name, nn.Linear(4, 4))

    def forward(self, values):
        hidden = self.layer.self_attn["q_proj"](values) + self.layer.self_attn["v_proj"](values)
        hidden = self.state_proj(hidden) + self.action_in_proj(values)
        hidden = self.action_time_mlp_out(torch.tanh(self.action_time_mlp_in(hidden)))
        return self.action_out_proj(hidden)


def make_model():
    recipe, _ = read_configuration(ROOT / "configs/vla/airbot_play/airbot_play_lora.yaml")
    recipe["optimizer"]["warmup_steps"] = 2
    model = add_adapters(TinyPolicy(), recipe)
    optimizer, scheduler = build_optimizer(model, recipe)
    return recipe, model, optimizer, scheduler


def update(model, optimizer, scheduler, inputs):
    optimizer.zero_grad(set_to_none=True)
    model(inputs).square().mean().backward()
    optimizer.step()
    scheduler.step()


def test_resume_reproduces_next_update_and_all_random_streams(tmp_path):
    torch.manual_seed(9)
    _, model, optimizer, scheduler = make_model()
    inputs = torch.randn(2, 4)
    update(model, optimizer, scheduler, inputs)
    checkpoint = save_checkpoint(tmp_path, model, optimizer, scheduler, 1, 8, "run")
    next_random = (random.random(), np.random.random(), torch.rand(3))
    update(model, optimizer, scheduler, inputs)
    expected = copy.deepcopy(get_peft_model_state_dict(model))
    expected_lr = scheduler.get_last_lr()
    load_checkpoint(checkpoint, model, optimizer, scheduler, "run")
    assert random.random() == next_random[0]
    assert np.random.random() == next_random[1]
    torch.testing.assert_close(torch.rand(3), next_random[2], rtol=0, atol=0)
    update(model, optimizer, scheduler, inputs)
    for name, actual in get_peft_model_state_dict(model).items():
        torch.testing.assert_close(actual, expected[name], rtol=0, atol=0)
    assert scheduler.get_last_lr() == expected_lr


def test_sampler_resume_ignores_worker_prefetch():
    original = iter(ReplayableSampler(11, seed=42))
    prefix = list(islice(original, 8))
    prefetched = list(islice(original, 30))
    resumed = list(islice(iter(ReplayableSampler(11, seed=42, consumed=8)), 30))
    assert prefetched == resumed
    assert sorted(prefix + resumed[:3]) == list(range(11))
    assert sorted(resumed[3:14]) == list(range(11))


def test_checkpoint_rejects_corruption_and_wrong_configuration(tmp_path):
    _, model, optimizer, scheduler = make_model()
    checkpoint = save_checkpoint(tmp_path, model, optimizer, scheduler, 1, 8, "correct")
    with pytest.raises(ValueError, match="fingerprint"):
        load_checkpoint(checkpoint, model, optimizer, scheduler, "wrong")
    with (checkpoint / "adapter.safetensors").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="verification"):
        checkpoint_manifest(checkpoint)


def batch_snapshot(micro_batch_size, accumulation_steps):
    return {
        "recipe": {
            "max_steps": 30000,
            "micro_batch_size": micro_batch_size,
            "gradient_accumulation_steps": accumulation_steps,
            "optimizer": {"learning_rate": 5e-5},
        },
        "configuration": {
            "train": {
                "max_steps": 30000,
                "micro_batch_size": micro_batch_size,
                "gradient_accumulation_steps": accumulation_steps,
                "global_batch_size": micro_batch_size * accumulation_steps,
            },
            "model": {"model_path": "base"},
        },
        "source_sha256": {"dataset": "unchanged"},
    }


def test_batch_change_preserves_checkpoint_and_resumes_optimizer_and_cursor(tmp_path):
    _, model, optimizer, scheduler = make_model()
    inputs = torch.randn(2, 4)
    update(model, optimizer, scheduler, inputs)
    previous = batch_snapshot(1, 8)
    previous["fingerprint"] = hashlib.sha256(json.dumps(previous, sort_keys=True).encode()).hexdigest()
    current = batch_snapshot(16, 2)
    new_fingerprint = hashlib.sha256(json.dumps(current, sort_keys=True).encode()).hexdigest()
    checkpoint = save_checkpoint(tmp_path / "checkpoints", model, optimizer, scheduler, 1, 8, previous["fingerprint"])
    original_manifest = checkpoint_manifest(checkpoint)
    expected_parameters = copy.deepcopy(get_peft_model_state_dict(model))
    expected_lr = scheduler.get_last_lr()
    change = record_batch_change(tmp_path, previous, current, checkpoint)
    (tmp_path / "run.json").write_text(json.dumps({**current, "fingerprint": new_fingerprint, "batch_change": change}))
    assert checkpoint_manifest(checkpoint) == original_manifest
    assert (
        checkpoint_manifest(tmp_path / "configuration_history" / previous["fingerprint"] / checkpoint.name)
        == original_manifest
    )
    update(model, optimizer, scheduler, inputs)
    restored = load_checkpoint(
        checkpoint, model, optimizer, scheduler, checkpoint_load_fingerprint(tmp_path, checkpoint, new_fingerprint)
    )
    assert restored["step"] == 1 and restored["consumed"] == 8
    assert scheduler.get_last_lr() == expected_lr
    for name, value in get_peft_model_state_dict(model).items():
        torch.testing.assert_close(value, expected_parameters[name], atol=0, rtol=0)
    update(model, optimizer, scheduler, inputs)
    next_checkpoint = save_checkpoint(tmp_path / "checkpoints", model, optimizer, scheduler, 2, 40, new_fingerprint)
    assert checkpoint_load_fingerprint(tmp_path, next_checkpoint, new_fingerprint) == new_fingerprint
    unrelated = save_checkpoint(tmp_path / "unrelated", model, optimizer, scheduler, 3, 24, previous["fingerprint"])
    with pytest.raises(ValueError, match="configuration migration"):
        checkpoint_load_fingerprint(tmp_path, unrelated, new_fingerprint)


@pytest.mark.parametrize("changed", ["optimizer", "model", "dataset"])
def test_batch_change_rejects_unrelated_configuration_changes(tmp_path, changed):
    previous = batch_snapshot(1, 8)
    previous["fingerprint"] = hashlib.sha256(json.dumps(previous, sort_keys=True).encode()).hexdigest()
    current = batch_snapshot(16, 2)
    if changed == "optimizer":
        current["recipe"]["optimizer"]["learning_rate"] = 1e-3
    elif changed == "model":
        current["configuration"]["model"]["model_path"] = "different"
    else:
        current["source_sha256"]["dataset"] = "different"
    with pytest.raises(ValueError, match="cannot change"):
        record_batch_change(tmp_path, previous, current, tmp_path / "checkpoint")
    assert not (tmp_path / "configuration_history").exists()


def test_step_extension_resumes_the_same_next_optimizer_update(tmp_path):
    _, model, optimizer, scheduler = make_model()
    inputs = torch.randn(2, 4)
    update(model, optimizer, scheduler, inputs)
    previous = batch_snapshot(16, 2)
    previous["recipe"]["max_steps"] = previous["configuration"]["train"]["max_steps"] = 1
    current = copy.deepcopy(previous)
    current["recipe"]["max_steps"] = current["configuration"]["train"]["max_steps"] = 3
    previous["fingerprint"] = hashlib.sha256(json.dumps(previous, sort_keys=True).encode()).hexdigest()
    new_fingerprint = hashlib.sha256(json.dumps(current, sort_keys=True).encode()).hexdigest()
    checkpoint = save_checkpoint(tmp_path / "checkpoints", model, optimizer, scheduler, 1, 32, previous["fingerprint"])
    change = record_step_extension(tmp_path, previous, current, checkpoint)
    (tmp_path / "run.json").write_text(
        json.dumps({**current, "fingerprint": new_fingerprint, "step_extension": change})
    )
    assert change["from_max_steps"] == 1 and change["to_max_steps"] == 3
    update(model, optimizer, scheduler, inputs)
    expected = copy.deepcopy(get_peft_model_state_dict(model))
    expected_lr = scheduler.get_last_lr()
    state = load_checkpoint(
        checkpoint, model, optimizer, scheduler, checkpoint_load_fingerprint(tmp_path, checkpoint, new_fingerprint)
    )
    assert state["step"] == 1 and state["consumed"] == 32
    update(model, optimizer, scheduler, inputs)
    for name, actual in get_peft_model_state_dict(model).items():
        torch.testing.assert_close(actual, expected[name], atol=0, rtol=0)
    assert scheduler.get_last_lr() == expected_lr
    next_checkpoint = save_checkpoint(tmp_path / "checkpoints", model, optimizer, scheduler, 2, 64, new_fingerprint)
    assert checkpoint_load_fingerprint(tmp_path, next_checkpoint, new_fingerprint) == new_fingerprint


@pytest.mark.parametrize("changed", ["batch", "optimizer", "model", "dataset", "shorter", "same"])
def test_step_extension_rejects_unrequested_changes(tmp_path, changed):
    previous = batch_snapshot(16, 2)
    current = copy.deepcopy(previous)
    previous["fingerprint"] = hashlib.sha256(json.dumps(previous, sort_keys=True).encode()).hexdigest()
    current["recipe"]["max_steps"] = current["configuration"]["train"]["max_steps"] = 50000
    if changed == "batch":
        current["recipe"]["micro_batch_size"] = 8
    elif changed == "optimizer":
        current["recipe"]["optimizer"]["learning_rate"] = 1e-3
    elif changed == "model":
        current["configuration"]["model"]["model_path"] = "different"
    elif changed == "dataset":
        current["source_sha256"]["dataset"] = "different"
    else:
        target = 20000 if changed == "shorter" else 30000
        current["recipe"]["max_steps"] = current["configuration"]["train"]["max_steps"] = target
    with pytest.raises(ValueError, match="cannot change|must increase"):
        record_step_extension(tmp_path, previous, current, tmp_path / "checkpoint")
    assert not (tmp_path / "configuration_history").exists()


def test_checkpoint_retention_preserves_exported_models_alongside_rolling_checkpoints(tmp_path):
    _, model, optimizer, scheduler = make_model()
    exported = save_checkpoint(tmp_path, model, optimizer, scheduler, 1, 32, "original")
    (exported / "hf_ckpt").mkdir()
    (exported / "hf_ckpt/model.safetensors.index.json").write_text("{}")
    for step in range(2, 5):
        save_checkpoint(tmp_path, model, optimizer, scheduler, step, step * 32, "extended", keep=2)
    assert (exported / "COMPLETED.json").is_file()
    assert (exported / "hf_ckpt/model.safetensors.index.json").is_file()
    assert not (tmp_path / "global_step_2").exists()
    assert (tmp_path / "global_step_3/COMPLETED.json").is_file()
    assert (tmp_path / "global_step_4/COMPLETED.json").is_file()


def test_checkpoint_retention_preserves_incomplete_and_unrelated_directories(tmp_path):
    _, model, optimizer, scheduler = make_model()
    incomplete = tmp_path / ".step_0_partial.tmp"
    incomplete.mkdir()
    unrelated = tmp_path / "global_step_notes"
    unrelated.mkdir()
    for step in range(1, 4):
        save_checkpoint(tmp_path, model, optimizer, scheduler, step, step * 8, "run", keep=2)
    assert not (tmp_path / "global_step_1").exists()
    assert (tmp_path / "global_step_2/COMPLETED.json").is_file()
    assert (tmp_path / "global_step_3/COMPLETED.json").is_file()
    assert incomplete.is_dir() and unrelated.is_dir()


def test_merged_export_loads_into_original_architecture(tmp_path):
    _, model, optimizer, scheduler = make_model()
    inputs = torch.randn(2, 4)
    update(model, optimizer, scheduler, inputs)
    model.eval()
    expected = model(inputs).detach()
    destination = tmp_path / "hf_ckpt"
    export_weights(model, destination, shard_limit=128)
    index = json.loads((destination / "model.safetensors.index.json").read_text())
    weights = {}
    for filename in set(index["weight_map"].values()):
        weights.update(load_file(destination / filename))
    native = TinyPolicy()
    native.load_state_dict(weights, strict=True)
    torch.testing.assert_close(native(inputs), expected, rtol=1e-5, atol=1e-6)
    assert not any("lora" in key or "modules_to_save" in key for key in weights)


def test_inference_probe_reuses_identical_noise_with_inplace_denoising():
    from tasks.vla.train_airbot_lora import inference_probe

    class InplaceNoisePolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(use_cache=False)
            self.model = SimpleNamespace(
                qwenvl_with_expert=SimpleNamespace(
                    config=SimpleNamespace(attention_implementation="flex_cached"),
                    get_attention_interface=lambda: None,
                )
            )

        def sample_actions(self, *, noise):
            assert self.config.use_cache
            assert self.model.qwenvl_with_expert.config.attention_implementation == "eager"
            noise.add_(1)
            return noise

    model = InplaceNoisePolicy()
    noise = torch.zeros(1, 50, 55)
    first = inference_probe(model, {}, noise)
    second = inference_probe(model, {}, noise)
    torch.testing.assert_close(noise, torch.zeros_like(noise), rtol=0, atol=0)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["vlm", "expert"])
def test_actual_decoder_split_forward_with_lora_and_checkpointing(kind):
    from transformers import Qwen2Config
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig

    from lingbotvla.models.vla.lingbot_vla.qwen2_action_expert import Qwen2DecoderLayer
    from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import Qwen3VLTextDecoderLayer

    class SplitPolicy(TinyPolicy):
        def __init__(self):
            super().__init__()
            configuration_class = Qwen3VLTextConfig if kind == "vlm" else Qwen2Config
            configuration = configuration_class(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                head_dim=16,
            )
            configuration._attn_implementation = "eager"
            layer_class = Qwen3VLTextDecoderLayer if kind == "vlm" else Qwen2DecoderLayer
            self.layer = layer_class(configuration, layer_idx=0)
            for name in (
                "state_proj",
                "action_in_proj",
                "action_out_proj",
                "action_time_mlp_in",
                "action_time_mlp_out",
            ):
                setattr(self, name, nn.Linear(32, 32))

        def forward(self, values):
            hidden = self.state_proj(values)
            query, key, value = self.layer(hidden, compute_kqv=True)
            attention = (query + key + value).flatten(-2)
            result = self.layer(hidden, attention, 0, values.shape[1], output_atten=True)
            hidden = result[0] if isinstance(result, tuple) else result
            return self.action_out_proj(hidden)

    recipe, _ = read_configuration(ROOT / "configs/vla/airbot_play/airbot_play_lora.yaml")
    model = add_adapters(SplitPolicy().bfloat16(), recipe)
    model.get_base_model().gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = model(torch.randn(1, 4, 32)).float().square().mean()
    loss.backward()
    gradients = [parameter.grad for name, parameter in model.named_parameters() if "lora_B" in name]
    assert gradients and all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) > 0 for gradient in gradients)
