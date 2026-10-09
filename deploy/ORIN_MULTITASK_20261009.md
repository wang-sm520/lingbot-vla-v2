# Orin 多物体夹取部署

两套策略均使用 600 条示范、105,882 帧：易拉罐 100 条、红纸袋 200 条、西瓜皮 300 条。

| 模型 | 部署 checkpoint | Orin 目录 |
| --- | --- | --- |
| Pi05 + LoRA | 35000 | `/mnt/nvme/pi05/tasks/can-paperbag-watermelon/checkpoints/35000` |
| LingBot-VLA 2.0 6B + 合并 LoRA | 52500 | `/mnt/nvme/lingbot-vla-v2/deployments/multitask-52500` |

LingBot 是从正在续训的实验中固定的完整 52500 保存点；5090 上累计 100000 步训练继续运行。
两个部署包均经逐文件 SHA-256 校验；已有部署目录保留。

## 启动命令

Pi05 终端一：

```bash
cd /mnt/nvme/pi05
python3 deploy.py serve can-paperbag-watermelon --checkpoint 35000 --prompt-key can
```

Pi05 终端二：

```bash
cd /mnt/nvme/pi05
python3 deploy.py robot can-paperbag-watermelon --checkpoint 35000 --prompt-key can --can-interface can2 --max-steps 1000 --execute
```

LingBot：

```bash
cd /mnt/nvme/lingbot-vla-v2
bash deploy/run_orin.sh can
```

把 `can` 换成 `red-paperbag` 或 `watermelon` 即可选择其他对象。切换模型时退出上一套程序。

| 对象键 | 与训练一致的 prompt |
| --- | --- |
| `can` | Pick up the can, return to the starting position, and keep holding it. |
| `red-paperbag` | Pick up the red paper bag and hold in the reset position. |
| `watermelon` | Pick up the watermelon peel, return to the starting position, and keep holding it. |

默认 1000 个策略动作、25 Hz、execution horizon 30、prediction horizon 50、底层控制频率 100 Hz。
沿用 reset `0 0 0 0 0 0 0.07`、关节插值步长 0.01 rad、夹爪步长 0.005 m。
LingBot 可在命令后追加参数，例如 `--max-steps 500`、`--robot-interface can2` 或 `--head-camera <实际RTSP地址>`。

LingBot 对有限数值的关节目标和夹爪开度越界统一裁剪到现有限位，并记录提示后继续执行。
例如第 2 关节的 `0.19720805 rad` 会裁剪为 `0.17 rad`；整段预测在插值和发送前完成裁剪。
NaN/Inf、动作维度错误和相机检查仍按原规则处理。更新代码后需重新启动推理进程生效。

## LingBot 推理提速

可用 `--num-inference-steps` 调整 flow-matching 去噪迭代次数；默认仍为 10。
2026-10-09 在 Orin MAXN、BF16、checkpoint 52500 下，三种物品各用一条录制观测、三个相同随机种子测量，单独排除各组首次预热：

| 去噪次数 | 三种物品的推理耗时中位数范围 | 相对 10 次 |
| --- | --- | --- |
| 10 | 2.27–2.28 秒 | 1.00 倍 |
| 5 | 1.286–1.290 秒 | 约 1.77 倍 |
| 4 | 1.093–1.095 秒 | 约 2.08 倍 |

先尝试 5 次去噪：

```bash
bash /mnt/nvme/lingbot-vla-v2/deploy/run_orin.sh can --num-inference-steps 5
```

参数改变单次预测的数值积分精度，prediction horizon 仍为 50，execution horizon 仍为 30，动作发送频率仍为 25 Hz。
这些结果是单次预测提速比例，包含插值和动作执行的完整任务不会等比例加速。
5 次相对 10 次，在这些观测上关节绝对差的均值为 0.0085–0.0131 rad，最大为 0.0965 rad；夹爪绝对差最大约 14.6 mm。
输出均为有限的 `[50, 7]`，这不代表抓取成功率不变，需按物品在真机上对比。
本次没有启用编译、量化，也没有调整关节限位、夹爪裁剪或插值步长。

离线重测命令（在 Orin 没有运行其他推理程序时执行，不初始化机械臂或相机）：

```bash
cd /mnt/nvme/lingbot-vla-v2
bash deploy/orin_env.sh .venv/bin/python -m tools.benchmark_airbot_inference \
  --model deployments/multitask-52500/checkpoints/global_step_52500/hf_ckpt \
  --fixtures deployments/multitask-52500/fixtures \
  --steps 10 5 4 --repeats 3 \
  --output logs/perf-denoising-comparison.json
```

测量记录：`/mnt/nvme/lingbot-vla-v2/logs/perf-denoising-20261009.json`。

## 环境与验证

LingBot 使用独立 Python 3.10 环境；通过明确的 `.pth` 引用 Orin 已有的 ARM Torch 2.10/CUDA 12.6、Triton 3.5.1 和 AIRBOT 0.2.9.2 运行库。
独立环境固定 Transformers 4.57.3；原 GR00T 和 Pi05 环境的依赖未替换。
`deploy/orin_env.sh` 配置 CUDA、硬件库和本地 Qwen3-VL tokenizer 路径；`deploy/install_orin_runtime.sh` 记录安装过程。

加载器在 meta 设备建立模型、按 BF16 分片装入 GPU，并严格检查权重键和形状。
Jetson 上在加载前和每片加载后，对本模型文件调用 `POSIX_FADV_DONTNEED`，处理文件缓存挤占共享内存导致的 NvMap/NVML 分配错误。
GPU 单独分配 13 GiB 的诊断通过；失败与成功日志保存在 Orin 项目的 `logs/`。

Pi05 的三种真实训练观测均返回有限的 `[50, 7]` 动作，预热后约 0.69–0.73 秒/次；策略服务与机械臂客户端协议的三种 prompt 回放也通过。
LingBot 的三种观测回放均返回有限的 `[50, 7]` 动作，预热后约 2.4 秒/次，GPU 已分配内存峰值约 12.1 GiB。
这些检查验证模型加载、数据处理和输出格式，不代表真机抓取成功率；没有驱动机械臂。

验证记录：

- Pi05：`/mnt/nvme/pi05/tasks/can-paperbag-watermelon/reports/35000/`。
- Pi05 RPC：`/mnt/nvme/pi05/reports/multitask-rpc-20261009.json`。
- LingBot：`/mnt/nvme/lingbot-vla-v2/deployments/multitask-52500/orin_validation.json`。

现场检查时 `can2` 已启用，腕部 RealSense `260322275842` 连续五帧通过。
头部 `rtsp://192.168.168.168:8554/front` 的端口和取帧检查均超时，需要恢复 M1 相机网络连接后再运行真机命令。
