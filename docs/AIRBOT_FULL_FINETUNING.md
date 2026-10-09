# AIRBOT Play 全参数微调

此流程从官方 LingBot-VLA 2.0 6B 基座开始，训练整个策略的视觉编码器、语言主干、动作专家和投影层，不注入 LoRA。独立的深度/视频教师只提供监督目标，不参与优化。

数据为同一套 600 条、105,882 帧 LeRobot v3 示范：易拉罐 100、红纸袋 200、西瓜皮 300。每条示范保留自己的 prompt。双相机输入 224×224，六关节采用相对当前状态的目标差分，夹爪采用绝对开度，预测窗口 50 步，数据频率 25 Hz。

## 服务器与环境

| 项目 | 设置 |
| --- | --- |
| 平台 | Linux x86_64、Conda、Python 3.12 |
| 框架 | PyTorch 2.8.0、CUDA 12、FlashAttention 2.8.3、Transformers 4.57.3 |
| 单卡方案 | 80GB 级 GPU，例如 H100 80GB；`bf16-kahan`，micro-batch 1 起步 |
| 多卡方案 | `fp32-master`，建议至少 2×80GB；FSDP2 分片 |
| 内存/磁盘 | 建议主机内存 128GB 或更多；NVMe 容量另计数据、预训练权重及多个全参 checkpoint |

`bf16-kahan` 使用仓库已有的 AnyPrecisionAdamW：BF16 参数、梯度、优化器状态和 Kahan 补偿缓冲。约 6.38B 策略的这些张量合计约 59.4 GiB，尚未计入激活、教师、CUDA 工作区及 checkpoint 导出开销。它是单卡 80GB 的候选配置，实际能否容纳、速度和数值效果需在目标服务器短跑确认。24GB/32GB 卡不能直接照搬此全参配置。

`fp32-master` 使用 AdamW 的 FP32 主权重/状态和 BF16 计算，优化器常驻张量更大，依靠多卡 FSDP2 分片。两套配置都训练全部策略参数，数值精度设置不同。

```bash
git clone https://github.com/wang-sm520/lingbot-vla-v2.git
cd lingbot-vla-v2
bash tools/create_train_env.sh --env-name lingbotvla
conda activate lingbotvla
```

安装脚本需要与 Torch CUDA 版本兼容的驱动；从源码构建 FlashAttention 时还需要 CUDA toolkit 和 C++ 编译器。已有匹配 Python 3.12 / Torch 2.8 / CUDA 12 / C++11 ABI 的 wheel，可给安装脚本追加 `--flash-attn-wheel /path/to/file.whl`。训练不需要 AIRBOT 硬件 SDK。LeRobot/MoGe 的可选应用依赖采用 `--no-deps` 安装以保留训练依赖版本，安装脚本会说明其元数据冲突。

## 准备资产

在仓库根目录设置新服务器实际路径：

```bash
export AIRBOT_DATASET=/mnt/data/pick_can_paperbag_watermelon_multitask
export AIRBOT_WEIGHTS=/mnt/data/lingbot_weights
export AIRBOT_RUN=/mnt/data/runs/airbot_full_100k

python -m tools.download_airbot_assets --weights-root "$AIRBOT_WEIGHTS"
python -m tools.prepare_airbot_data --dataset "$AIRBOT_DATASET"
```

先将已合并的数据集目录传到 `$AIRBOT_DATASET`，包括 `meta/`、`data/`、`videos/`；不要用单个原始采集会话替代。数据集和微调 checkpoint 不随 Git 分发。下载工具固定官方 LingBot、Qwen tokenizer、MoGe 的版本，并检查 LFS SHA-256；LingBot 仓库同时提供 depth 和 DINO-Video 教师。

归一化命令检查全部 600 条轨迹、任务表、七维状态/动作和视频引用，生成 `assets/norm_stats/airbot_play_multitask.json` 与 `logs/airbot_data_audit.json`。它只报告采集数据中的越界，不裁剪原始训练数据。

## 单卡 80GB：生成配置并训练

```bash
python -m tools.configure_airbot_full \
  --dataset "$AIRBOT_DATASET" --weights-root "$AIRBOT_WEIGHTS" \
  --output-dir "$AIRBOT_RUN" --output configs/local/airbot_full.yaml \
  --profile bf16-kahan --world-size 1 \
  --micro-batch-size 1 --global-batch-size 32 --steps 100000

CUDA_VISIBLE_DEVICES='' python -m tools.check_airbot_data \
  --config configs/local/airbot_full.yaml

CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 \
  bash train.sh tasks/vla/train_lingbotvla.py configs/local/airbot_full.yaml
```

生成器将实际路径写入独立配置；已有同名配置不会覆盖，修改时可编辑该文件或使用新的 `--output` 文件名。

默认训练参数：有效 batch 32、micro-batch 1、梯度累积 32、学习率 5e-5、预热 1000 步后恒定、weight decay 0.01、梯度范数上限 1.0、seed 42、workers 2、梯度检查点开启、`torch.compile` 关闭。保留当前/未来深度与视频监督，三项权重均为 0.004，保留 MoE 平衡损失。全参配置使用统一学习率，未启用 MoE 专家的额外学习率缩放。这些是迁移起点，不代表已经测得最优学习率或成功率。

先验证目标 GPU 的几个完整更新及保存路径，可把训练命令替换为：

```bash
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 \
  bash train.sh tasks/vla/train_lingbotvla.py configs/local/airbot_full.yaml \
  --train.max_steps 3 --train.save_steps 3 \
  --train.output_dir "${AIRBOT_RUN}_smoke"
```

短跑输出独立于正式实验。全参数训练的累计步数是优化器更新次数，不是 epoch。仓库原生全参训练入口不会沿用 LoRA 的“只保留两个 checkpoint”规则；每 2500 步保存一次全参状态，需按服务器磁盘预算调整 `save_steps` 或另行管理保存点。

## 两卡 FP32 主权重方案

```bash
python -m tools.configure_airbot_full \
  --dataset "$AIRBOT_DATASET" --weights-root "$AIRBOT_WEIGHTS" \
  --output-dir "${AIRBOT_RUN}_fp32master" \
  --output configs/local/airbot_full_2gpu.yaml \
  --profile fp32-master --world-size 2 \
  --micro-batch-size 1 --global-batch-size 32 --steps 100000

CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 \
  bash train.sh tasks/vla/train_lingbotvla.py configs/local/airbot_full_2gpu.yaml
```

这里是每卡 1 样本 × 2 卡 × 累积 16 次＝有效 batch 32。生成配置的 `--world-size` 必须与 torchrun 的总 GPU 进程数一致。多节点时按原生 torchrun 配置 `NNODES`、`NODE_RANK`、`MASTER_ADDR`、`MASTER_PORT`。

## 恢复与导出

保持原数据、归一化、配置及 GPU 拓扑，在同一输出目录恢复最新完整分布式 checkpoint：

```bash
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 \
  bash train.sh tasks/vla/train_lingbotvla.py configs/local/airbot_full.yaml \
  --train.enable_resume true
```

原生全参保存点采用 `checkpoints/global_step_<累计更新步>/`，同时导出 `hf_ckpt/` 供推理。等待日志确认保存/导出结束；推理目录本身不能替代恢复优化器所需的完整训练保存点。

部署还需要该运行的 `lingbotvla_cli.yaml`、归一化文件和 Qwen tokenizer/config。跨机时更新这些配置中的路径或把归一化文件一起打包。已有 LoRA 导出工具 `tools.export_airbot_checkpoint` 面向 LoRA 保存格式，全参训练直接使用原生 `hf_ckpt` 导出。

机器人接口见 [AIRBOT_PLAY.md](../deploy/AIRBOT_PLAY.md)。当前默认 execution horizon 30、prediction horizon 50，关节角和夹爪开度超界按已设定限位裁剪。

## 验证范围

迁移代码在现有 5090 主机进行配置解析、CPU 回归和完整模型的 meta 参数检查；不会占用正在训练的 GPU。全参 GPU 首步、长跑稳定性和目标服务器吞吐量需要在目标机器验证。当前 5090 LoRA 实验与这里新建的全参输出目录独立。

2026-10-09 验证：54 项测试通过，4 项依赖 CUDA 的测试跳过；三种物品共 12 条真实边界观测的解码、任务文字、动作窗口和归一化往返检查通过。meta 模型共有 6,375,906,359 个参数，全部 `requires_grad=True`，没有 LoRA 参数。检查时没有加载 GPU 模型或运行全参优化器步。
