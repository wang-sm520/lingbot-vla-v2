# AIRBOT Play / LingBot-VLA 2.0 准备记录

截至 2026-10-06 19:55（Asia/Shanghai），三步 GPU 试跑、checkpoint 保存恢复、合并导出及三种任务 prompt 的离线推理检查均已通过。LingBot 正式 30,000 步训练已启动，首步 loss 为 0.738177。真机运动尚未验证；实时进度读取下文的状态文件。

## 当前 Pi05

Pi05 已于 19:41:02 停止，最后记录到 35,800 步，loss 0.0025。用户指定的 30,000 checkpoint 于 15:14:02 完整提交并保留，另保留已有的 35,000 等完整保存点。

自动停训未按时触发：systemd 的 PATH 中没有 `rg`，脚本反复检查失败却仍处于运行状态。已将日志检查改为绝对路径 `/usr/bin/awk`，在 systemd 同等 PATH 下验证五种完整/不完整日志组合，再重启停训服务。服务确认 30000 的 params/train_state manifest、目录提交日志及异步保存完成日志后停止训练；未删除超出目标后产生的 checkpoint。

## 已完成

- 独立环境：`/home/user/miniconda3/envs/lingbotvla`，Python 3.12、Torch 2.8.0+cu128、FlashAttention 2.8.3、Transformers 4.57.3、LeRobot 0.4.2。
- AIRBOT SDK 0.2.9.2、Pi05 控制包装器、双相机包装器、MoGe 与 LingBot-Depth 导入成功，检查过程未初始化硬件。
- 官方 LingBot 资产 28,239,981,618 字节；Qwen 配置和 tokenizer 11,564,736 字节；MoGe 419,111,765 字节。三个版本固定，LFS 文件均与官方 SHA-256 比对一致。清单见 `logs/asset_*.json`。
- 全部 600 条、105,882 帧：易拉罐 100 条，红纸袋 200 条，西瓜皮 300 条；逐帧任务与 episode 元数据匹配，沿用各自原 prompt。
- 50 步窗口归一化已生成，关节动作相对当前状态，夹爪绝对米值；每条 episode 末尾仅重复自身最后动作。统计使用 5,294,100 个动作样本。
- 12 个跨任务和 episode 边界样本，通过真实当前/未来图像解码、动作 padding、有效维度 `0..5,28`、归一化及反归一化检查。最大动作还原误差约 `1.22e-7`。报告：`logs/airbot_data_preflight.json`。
- 17 项离线测试、相关 Python lint、shell 语法及差异空白检查通过。覆盖 LoRA 的真实解码层梯度重算、断点后下一次更新一致性、随机状态和数据顺序恢复、损坏文件拒绝、保存保留策略、合并导出与硬件适配。训练模块可在禁用 CUDA 的环境下导入。

## 已确认的单卡方案

官方全参配置不能直接作为 RTX 5090 32GB 的训练方案；官方文档的 4×A6000 示例标注约 49 GB/卡。检查全部 1,708 个 checkpoint tensor，共约 6.376B 元素，FP32 权重本身为 25.50 GB，尚不包括梯度、优化器与教师模型。

具体提议在 `configs/vla/airbot_play/rtx5090_lora_proposal.yaml`：LoRA rank 16/alpha 32，BF16，micro-batch 1，梯度累积 8，有效 batch 8，seed 42，30,000 个优化器更新。对视觉语言模型的语言注意力和动作专家注意力，共 288 个线性层设置 LoRA，并训练状态/动作投影；预计可训练约 21.36M 参数。保持原生深度/视频分支和教师损失。

用户已确认该方案。实现配置为 `configs/vla/airbot_play/airbot_play_lora.yaml`，入口为 `tasks/vla/train_airbot_lora.py`。LoRA 注入、断点恢复、预取情况下的数据顺序恢复、合并导出以及真实解码层的梯度重算已经通过 CPU 测试；可训练参数实际为 21,363,511。主配置中的全参设置仅作为结构参考。

后台接力单元为 `lingbot-airbot-lora-after-pi05.service`：Pi05 30000 完整保存并退出、GPU 释放后，先运行三步独立 GPU 验证，检查断点恢复、合并导出和三种 prompt 的离线策略输出，成功后从基座启动正式 30000 步训练。19:54:52 验证通过并启动正式进程，19:55:14 完成第一个优化器更新。三步试跑显存峰值约 15.28 GiB，后两步约 5.9–6.1 秒/步；长期速度以正式训练记录为准。状态位于 `logs/airbot_lora_handoff.json`，正式训练进度位于 `outputs/airbot_play_multitask_lora_30k/status.json`，逐步指标位于同目录的 `metrics.jsonl`。

GPU 验证期间修复了以下问题，未更改已批准的数据、训练步数、LoRA、batch、学习率或损失分支：

- Flex Attention 显式使用 FP32，但外层 BF16 autocast 导致未编译反向传播的类型不匹配；只在该 FP32 算子调用范围禁用 autocast。两种注意力路径的真实 GPU 输出和梯度与参考实现一致。
- 采样会原地修改噪声；一致性检查现在为每次推理复制同一初始噪声，并与部署一样启用 KV cache、使用 eager attention。
- 推理专用 MoE 对多个专家结果的原子累加顺序不固定，逐层 BF16 舍入会放大差异；改为分别写入各 route 的 FP32 结果，再固定顺序求和。1/51 token 的工作区复用测试及完整模型恢复检查通过。

12 项相关回归通过；完整模型的适配器恢复和导出回读检查也通过。合并至 BF16 后与未合并适配器的归一化动作最大差异为 0.0339355，不能解释为真机动作误差或成功率。试跑资产位于 `outputs/airbot_play_multitask_lora_smoke/`，导出验证结果见该目录的 `export_verification.json`。

## 部署适配与已知边界

部署复用 Pi05 控制器、相机映射、关节限位和插值方法；预测 50 步，默认执行 30 步、25 Hz。接口见 `AIRBOT_PLAY.md`。

用户已明确授权将有限值的 G2 指令裁剪至 [0, 0.072] 米，与 Pi05 对齐；插值和执行使用裁剪后的值，并记录裁剪日志。非有限值和越界的机械臂关节指令仍会被拒绝。接收时间超过两秒的相机数据会被拒绝；RealSense 接收时间不等价于曝光时间。后续项目安全门禁的决策遵照 `/home/user/wang-sm/AGENTS.md` 先征询用户意见。

原始动作中 2,508 帧的关节 2 高于 0.17 rad，上限达到 0.1768139；状态中 828 帧关节 2 超限，25,567 帧夹爪反馈存在微小边界偏差。数据未裁剪、限位未放宽；详见 `logs/airbot_data_audit.json`。真机可能拒绝相应预测，部署行为需要结合用户的设备判断。

上游多个子包的声明依赖与主项目版本冲突，`pip check` 仍有报告。安装器保留主项目 Torch/NumPy/PyArrow，采用兼容的 MLflow skinny 与 plyfile，按需验证运行路径；不宣称整个 LeRobot 或 MoGe 独立应用均可用。版本快照：`logs/environment-freeze.txt`。

ModelTrace 多次后台检测因无法 fork 当前执行中的对话而失败。它是监测缺口，未产生可用于本轮的模型不匹配结论。
