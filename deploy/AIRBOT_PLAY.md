# AIRBOT Play integration

The combined LeRobot v3 dataset has 7 dimensions: six arm joint positions followed by the G2 opening in meters. `configs/robot_configs/airbot_play.yaml` maps those dimensions to LingBot's `arm.position` and `effector.position`. Arm actions are trained as state-relative deltas; the G2 action remains absolute. The three task prompts remain in the dataset task table and are passed through unchanged.

Prepare the independent training environment and AIRBOT runtime:

```bash
bash tools/create_train_env.sh --env-name lingbotvla
bash tools/install_airbot_play_runtime.sh
```

Audit all 600 episodes and compute statistics on the CPU:

```bash
export AIRBOT_DATASET="$PWD/data/pick_can_paperbag_watermelon_multitask"
CUDA_VISIBLE_DEVICES='' conda run -n lingbotvla python -m tools.prepare_airbot_data \
  --dataset "$AIRBOT_DATASET"
CUDA_VISIBLE_DEVICES='' conda run -n lingbotvla python -m tools.check_airbot_data
```

The dataset contains 100 can, 200 red-paper-bag, and 300 watermelon episodes (105,882 frames). The audit reports source hashes, prompt counts, episode boundaries, and out-of-range values without changing the data. Statistics include 50-step action chunks padded with the last action of each episode. The CPU preflight checks task boundaries, both camera decoders, canonical active slots `0..5,28`, and normalization/denormalization round trips.

The user approved the single-5090 recipe in `configs/vla/airbot_play/airbot_play_lora.yaml`: LoRA rank 16/alpha 32, BF16 frozen weights, FP32 adapters/projections, micro-batch 16, accumulation 2, effective batch 32, seed 42 and 100,000 cumulative optimizer steps. It preserves the official depth/video architecture and teacher losses. LoRA covers 288 attention projections; 21,363,511 parameters train. Vision weights stay frozen. Video-teacher timing uses the dataset's 25 FPS and 49-frame offset, including preservation of the timing field through feature conversion.

`lingbot-airbot-lora-after-pi05.service` waits for the complete Pi05 checkpoint 30000 and process exit, then runs three isolated GPU optimizer steps, saves/reloads the adapter checkpoint, exports merged weights and tests offline policy inference for all three prompts. Those checks passed on 2026-10-06 and the formal run started. On 2026-10-07 the user approved changing 1×8 to 16×2 after GPU benchmarking. Training resumed from step 4753 with optimizer, scheduler, RNG and sample position preserved; the first new checkpoint is step 4754. The migration is recorded in the output's `run.json`, and the preceding configuration and checkpoint are archived in `configuration_history/`. See `BATCH_BENCHMARK_20261007.md` for measurements. Actual robot motion remains untested. Failed checks record the error in `logs/airbot_lora_handoff.json`; the recipe does not silently disable losses or switch precision.

The original 30,000-step run and its three-task export check completed on 2026-10-08. The user then requested continuation to 50,000 cumulative steps, adding 20,000 updates. Resume uses the full step-30000 checkpoint with 845,928 consumed samples and the original optimizer, scheduler and RNG; learning rate stays at 5e-5. Only the step target changes. `run.json` records the `step_extension`, including both configuration fingerprints and the source manifest hash. The preceding configuration, assets, completion report and full checkpoint are archived under `configuration_history/`. The existing step-30000 deployment export is retained at its original path. The output directory keeps its historical `_30k` name; the recipe and runtime status define the current target.

The 50,000-step continuation and its three-task export check completed on 2026-10-09. The user then requested 100,000 cumulative steps, adding 50,000 updates from `global_step_50000` with 1,485,928 consumed samples. The same migration procedure preserves the optimizer, scheduler, RNG, sample position, batch 16×2, learning rate 5e-5 and all 600 episodes. The completed step-50000 checkpoint, deployment export and configuration are archived before resuming; the original step-30000 and step-50000 exports remain available. Earlier extension records remain in the archived run configurations.

To launch directly when the GPU is available:

```bash
conda run -n lingbotvla python -m tasks.vla.train_airbot_lora
```

To resume a committed checkpoint, set `CHECKPOINT_DIR` to an existing complete checkpoint under `outputs/airbot_play_multitask_lora_30k/checkpoints/`. The systemd service selects the latest committed checkpoint automatically:

```bash
conda run -n lingbotvla python -m tasks.vla.train_airbot_lora \
  --resume "$CHECKPOINT_DIR"
```

The output contains `metrics.jsonl`, `status.json` and TensorBoard events. Every 2,500 optimizer steps a checksum-protected adapter/projection/optimizer/RNG/data-position checkpoint is atomically committed; the latest two ordinary checkpoints and previously exported checkpoints are retained. A recorded configuration migration also saves immediately after the first resumed update. Resume requires matching dataset/configuration fingerprints or an explicit, validated migration record and preserves sample order despite worker prefetch. SIGTERM requests saving after the current optimizer update. The service reads its final target from the recipe; this continuation will export `checkpoints/global_step_100000/hf_ckpt` and verify all three task prompts. The matching configuration and normalization remain in the run directory.

Qwen configuration/tokenizer files are sufficient: the LingBot checkpoint contains the VLM weights. `tools.download_airbot_assets` pins all model revisions and verifies LFS SHA-256 values. The installer uses `mlflow-skinny==3.1.4` and `plyfile==1.1.2` to retain the main project's PyArrow, packaging and NumPy pins. Optional LeRobot/MoGe application dependencies are intentionally not installed; `pip check` therefore reports upstream metadata conflicts. CPU dataset, camera-wrapper and depth-model imports are verified separately. The FlashAttention wheel must match Python 3.12, Torch 2.8, CUDA 12 and C++11 ABI.

The AIRBOT entrypoint reuses the Pi05 `RobotAH` and `airbot_arm` implementation with `airbot-hardware-py==0.2.9.2`. Commanded joints use the existing Pi05 ranges, interpolation uses 0.01 rad per joint and 0.005 m for the gripper, and rate defaults to 25 Hz. As authorized by the user, finite arm joint commands are clipped to their existing limits and G2 commands are clipped to [0, 0.072] m before interpolation and execution. Adjustments are logged and inference continues; the entire predicted chunk is clipped before any actions are sent. Nonfinite values and invalid action dimensions are rejected. Camera reception timestamps must be within two seconds; the RTSP source also checks its monotonic frame age. RealSense reception timestamps do not prove exposure freshness. Measured state is checked for shape and finite values without treating tiny sensor offsets as commanded motion. Shutdown follows the Pi05 operator; no new passive/gravity-compensation transition is added.

The following command describes the interface for a compatible, trained HF checkpoint. Live inference and robot motion have not been tested. Any project safety-gate changes require the user's opinion under the workspace `AGENTS.md`.

```bash
conda run -n lingbotvla python -m deploy.airbot_play_lingbot_inference \
  --model-path /path/to/output/checkpoints/global_step_30000/hf_ckpt \
  --robot-interface can0 \
  --prompt "Pick up the can, return to the starting position, and keep holding it." \
  --chunk-size-execute 5 --max-steps 25 \
  --reset-action 0 0 0 0 0 0 0.07
```

The command runs from the LingBot repository. Prediction horizon is 50; execution horizon defaults to 30. The example explicitly selects a 5-step execution horizon and 25 policy actions for an initial test. Interpolation may emit multiple motor commands per policy action. Reset poses are movement targets, not zero calibration. The runtime installer selects the hardware wheel using the actual Python, architecture and glibc compatibility tags; it imports wrappers without initializing hardware. The separate Orin inference environment, checkpoint 52500, task commands and verification records are documented in [ORIN_MULTITASK_20261009.md](ORIN_MULTITASK_20261009.md).
