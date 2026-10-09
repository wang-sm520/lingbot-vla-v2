from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
PI05_ROOT = Path(os.environ.get("OPENPI_ROOT", str(ROOT.parent / "pi0.5/openpi_v0.2.0")))
PI05_SERVICE = "pi05-multitask-50k-rerun-20261005.service"
PI05_RUN = "pi05_p6_can_paperbag_watermelon_multitask_50k_rerun_20261005"
PI05_CHECKPOINT = PI05_ROOT / "checkpoints/1-1_pick_can_paperbag_watermelon_multitask" / PI05_RUN / "30000"
PI05_STATUS = PI05_ROOT / "logs" / PI05_RUN / "stop_at_30000.status"
CONFIG = ROOT / "configs/vla/airbot_play/airbot_play_lora.yaml"
SMOKE = ROOT / "outputs/airbot_play_multitask_lora_smoke"
STATUS = ROOT / "logs/airbot_lora_handoff.json"


def checkpoint_committed(checkpoint: Path, status: Path) -> bool:
    return (
        (checkpoint / "params/manifest.ocdbt").is_file()
        and (checkpoint / "train_state/manifest.ocdbt").is_file()
        and status.is_file()
        and " checkpoint_30000_complete" in status.read_text()
    )


def write_status(stage: str, **details):
    record = {"time": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "stage": stage, **details}
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(STATUS)
    print(json.dumps(record), flush=True)


def service_state() -> str:
    return subprocess.check_output(
        ["systemctl", "--user", "show", PI05_SERVICE, "-p", "ActiveState", "--value"], text=True
    ).strip()


def gpu_processes() -> list[str]:
    text = subprocess.check_output(
        ["nvidia-smi", "--id=0", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True
    )
    return [line.strip() for line in text.splitlines() if line.strip()]


def run_stage(stage: str, module: str, *arguments: str):
    write_status(stage)
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONHASHSEED": "42",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "OMP_NUM_THREADS": "2",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
    }
    subprocess.run([sys.executable, "-m", module, *arguments], cwd=ROOT, env=environment, check=True)


def latest_checkpoint(output: Path) -> Path | None:
    candidates = [
        path
        for path in (output / "checkpoints").glob("global_step_*")
        if path.name.removeprefix("global_step_").isdigit() and (path / "COMPLETED.json").is_file()
    ]
    return max(candidates, key=lambda path: int(path.name.removeprefix("global_step_")), default=None)


def main(check_only: bool = False):
    if check_only:
        print(
            json.dumps(
                {
                    "pi05_state": service_state(),
                    "checkpoint_30000_committed": checkpoint_committed(PI05_CHECKPOINT, PI05_STATUS),
                    "gpu_pids": gpu_processes(),
                }
            )
        )
        return
    try:
        recipe = yaml.safe_load(CONFIG.read_text())
        output = ROOT / recipe["output_dir"]
        target_steps = recipe["max_steps"]
        write_status("waiting_for_pi05_checkpoint_30000")
        while True:
            state = service_state()
            committed = checkpoint_committed(PI05_CHECKPOINT, PI05_STATUS)
            if committed and state == "inactive":
                break
            if state in ("inactive", "failed") and not committed:
                raise RuntimeError(
                    "Pi05 stopped before the requested complete 30000 checkpoint; LingBot was not started."
                )
            if state == "failed":
                raise RuntimeError("Pi05 service failed; inspect its final checkpoint and exit status.")
            time.sleep(15)
        write_status("waiting_for_gpu_release")
        while gpu_processes():
            time.sleep(15)
        smoke_checkpoint = latest_checkpoint(SMOKE)
        if not (SMOKE / "export_verification.json").is_file():
            if smoke_checkpoint is None or not (smoke_checkpoint / "hf_ckpt/model.safetensors.index.json").is_file():
                resume = ["--resume", str(smoke_checkpoint)] if smoke_checkpoint else []
                run_stage(
                    "gpu_smoke_training", "tasks.vla.train_airbot_lora", "--smoke", "--output", str(SMOKE), *resume
                )
            smoke_checkpoint = latest_checkpoint(SMOKE)
            if smoke_checkpoint is None:
                raise RuntimeError("GPU smoke did not commit a checkpoint.")
            run_stage(
                "gpu_smoke_export_check",
                "tools.check_airbot_export",
                "--checkpoint",
                str(smoke_checkpoint),
                "--reference",
                str(SMOKE / "inference_reference.pt"),
            )
        verification = json.loads((SMOKE / "export_verification.json").read_text())
        if verification.get("status") != "passed":
            raise RuntimeError("GPU smoke export verification did not pass.")
        final = output / "checkpoints" / f"global_step_{target_steps}"
        if not (final / "hf_ckpt/model.safetensors.index.json").is_file():
            checkpoint = latest_checkpoint(output)
            resume = ["--resume", str(checkpoint)] if checkpoint else []
            run_stage("formal_training", "tasks.vla.train_airbot_lora", "--config", str(CONFIG), *resume)
            training_status = json.loads((output / "status.json").read_text())
            if training_status["stage"] == "paused":
                write_status("paused", step=training_status["step"], checkpoint=training_status["checkpoint"])
                return
        if not (final / "hf_ckpt/model.safetensors.index.json").is_file():
            raise RuntimeError(f"Formal training exited before exporting step {target_steps}.")
        run_stage("final_export_check", "tools.check_airbot_export", "--checkpoint", str(final))
        write_status("completed", step=target_steps, checkpoint=str(final), export=str(final / "hf_ckpt"))
    except Exception as error:
        write_status("failed", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    arguments = parser.parse_args()
    main(arguments.check_only)
