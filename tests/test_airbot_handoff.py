import json

import yaml

from tools import run_airbot_lora_after_pi05 as handoff
from tools.run_airbot_lora_after_pi05 import checkpoint_committed, latest_checkpoint


def test_handoff_requires_both_manifests_and_writer_commit_marker(tmp_path):
    checkpoint = tmp_path / "30000"
    (checkpoint / "params").mkdir(parents=True)
    (checkpoint / "params/manifest.ocdbt").write_bytes(b"params")
    status = tmp_path / "watcher.status"
    status.write_text("2026-10-06 watcher_started\n")
    assert not checkpoint_committed(checkpoint, status)
    (checkpoint / "train_state").mkdir()
    (checkpoint / "train_state/manifest.ocdbt").write_bytes(b"state")
    assert not checkpoint_committed(checkpoint, status)
    status.write_text("2026-10-06 checkpoint_30000_complete\n")
    assert checkpoint_committed(checkpoint, status)


def test_resume_chooses_newest_committed_step_numerically(tmp_path):
    for name in ("global_step_9", "global_step_10", "global_step_11", "global_step_notes"):
        (tmp_path / "checkpoints" / name).mkdir(parents=True)
    for name in ("global_step_9", "global_step_10", "global_step_notes"):
        (tmp_path / "checkpoints" / name / "COMPLETED.json").write_text("{}")
    assert latest_checkpoint(tmp_path).name == "global_step_10"


def test_handoff_continues_past_previous_export_to_configured_target(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"output_dir": "output", "max_steps": 50000}))
    output = tmp_path / "output"
    previous = output / "checkpoints/global_step_30000"
    (previous / "hf_ckpt").mkdir(parents=True)
    (previous / "COMPLETED.json").write_text("{}")
    (previous / "hf_ckpt/model.safetensors.index.json").write_text("{}")
    smoke = tmp_path / "smoke"
    smoke.mkdir()
    (smoke / "export_verification.json").write_text(json.dumps({"status": "passed"}))
    monkeypatch.setattr(handoff, "ROOT", tmp_path)
    monkeypatch.setattr(handoff, "CONFIG", config)
    monkeypatch.setattr(handoff, "SMOKE", smoke)
    monkeypatch.setattr(handoff, "service_state", lambda: "inactive")
    monkeypatch.setattr(handoff, "checkpoint_committed", lambda *arguments: True)
    monkeypatch.setattr(handoff, "gpu_processes", lambda: [])
    statuses = []
    monkeypatch.setattr(handoff, "write_status", lambda stage, **details: statuses.append({"stage": stage, **details}))
    stages = []
    final = output / "checkpoints/global_step_50000"

    def run_stage(stage, module, *arguments):
        stages.append(stage)
        if stage == "formal_training":
            assert arguments == ("--config", str(config), "--resume", str(previous))
            (final / "hf_ckpt").mkdir(parents=True)
            (final / "hf_ckpt/model.safetensors.index.json").write_text("{}")
            (output / "status.json").write_text(json.dumps({"stage": "completed", "step": 50000}))
        elif stage == "final_export_check":
            assert arguments == ("--checkpoint", str(final))

    monkeypatch.setattr(handoff, "run_stage", run_stage)
    handoff.main()
    assert stages == ["formal_training", "final_export_check"]
    assert statuses[-1] == {
        "stage": "completed",
        "step": 50000,
        "checkpoint": str(final),
        "export": str(final / "hf_ckpt"),
    }
