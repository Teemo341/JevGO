---
name: jevgo-gpu-access
description: 在 JevGO 执行依赖 GPU 的开发测试、真实模型推理或验收，以及排查 nvidia-smi、CUDA 设备不可见时使用。先区分沙箱设备隔离、设备选择、CUDA runtime 与显存不足，避免误判主机驱动故障。纯 CPU 测试和普通文档修改不要求 GPU 检查。
---

# JevGO GPU 访问与沙箱排查

## 已知项目事实

2026-10-14–15 的本机验收中，默认受限沙箱内 `nvidia-smi` 退出码为 9，无法连接 NVIDIA 驱动；经工具权限流程在设备可见的主机执行上下文复核后，命令退出码为 0，现有项目 PyTorch CUDA 与真实模型推理均可运行。

因此，默认沙箱**可能隔离 NVIDIA 设备访问**。沙箱内的驱动报错或 `torch.cuda.is_available() == False` 只说明当前进程不能使用 GPU，不能单独证明主机无 GPU、驱动损坏或 CUDA 必须重装。该案例也不证明所有沙箱或所有退出码 9 都有相同原因。

## 开发和测试时的执行规则

1. 需要 GPU 时，主动检查实际执行主机、项目路径、解释器和设备选择。当前共享资源根为 `/mnt/nas-tianwen/home/ssy/JevGO`，复用该目录的 `.venv/bin/python`；历史小写 `jevgo` 路径曾不存在，先核实实际路径。独立 worktree 中运行代码时保持 worktree 的工作目录与导入来源，共享解释器不代表切回主仓库。
2. 保存当前上下文的 `nvidia-smi` 输出、退出码，以及 `CUDA_VISIBLE_DEVICES` / `NVIDIA_VISIBLE_DEVICES` 的值，明确区分未设置与空字符串。失败时按需核对可执行文件路径、`/dev/nvidia*` 可见性和 `/proc/driver/nvidia/version`。仅能读取驱动版本文件不代表设备节点可访问；不输出整个环境变量表。
3. 若沙箱内失败，或用户终端可用而工具内不可用，在当前工具与策略允许时，主动通过工具权限/审批通道执行最小的沙箱外只读复核。保持同一主机、解释器及明确记录的设备选择，以区分执行上下文差异。已有任务授权范围内的必要排查无需先额外询问“要不要检查显卡”。
4. 当前会话已确认沙箱隔离时，后续相关 GPU 检查、已授权的推理或测试使用同样可访问设备的执行方式；子进程同样需要该访问能力。普通文件编辑和 CPU 测试仍使用默认执行方式。已有有效复核时不反复重现同一失败，但启动模型前重新查看实时显存和占用。
5. 设备可见后，用已有项目解释器分别核对 PyTorch 版本、`torch.version.cuda`、`torch.cuda.is_available()` 和可见设备数量。必要时在选定且可用的 GPU 做轻量 CUDA 算子验证；驱动可见或轻量算子通过都不替代目标模型验收。
6. 将命令、工作目录、解释器、相关设备变量、执行权限上下文、退出码及 stdout/stderr 保存到本次独立的 `.dev/artifacts/gpu-*` 目录，或当前开发任务已有的产物目录。分别报告访问权限、runtime、容量和模型执行结果，保留失败证据。

若当前工具提供 `exec_command` 且会话策略允许，设备复核的最小调用示例为：

```json
{
  "cmd": "nvidia-smi -q",
  "workdir": "/mnt/nas-tianwen/home/ssy/JevGO",
  "sandbox_permissions": "require_escalated",
  "justification": "只读复核主机 NVIDIA 驱动与 GPU，区分沙箱设备不可见和真实驱动故障。"
}
```

这表示走工具的权限流程，不是使用 `sudo` 或绕过审批。若策略不允许、审批未获准，或主机侧也失败，记录实际阻塞原因并继续不依赖 GPU 的工作；必要时说明缺少的访问条件。不要反复重试同一失败、全局关闭沙箱、修改安全策略，或把权限拒绝写成驱动损坏。

## 判读时避免混淆

| 观察 | 应有判断与后续 |
| --- | --- |
| 沙箱内失败，设备可见上下文中成功 | 属于当前执行上下文的访问差异，保留两侧证据；不据此重装驱动。 |
| `CUDA_VISIBLE_DEVICES` 为空、限定了卡，或 `NVIDIA_VISIBLE_DEVICES` 限制容器设备 | 先核对任务意图和实际映射。CPU 测试可能故意屏蔽 GPU，不擅自清空限制以访问所有卡；变量也不能恢复沙箱未暴露的设备节点。 |
| 限定一个物理 GPU 后 PyTorch 只报告一个 `cuda:0` | 这是逻辑设备重编号，不等于机器只剩一张卡。用 UUID/物理清单对照，不硬编码历史 GPU 编号。 |
| `nvidia-smi` 成功，项目 PyTorch CUDA 失败 | 继续检查项目解释器、设备选择、设备权限和 runtime/依赖，不能仅凭驱动命令宣称模型可运行。 |
| 能访问 GPU，但空闲显存不足或加载 OOM | 记录资源阻塞，结合实时空闲显存、权重精度、KV/runtime 开销及模型并行配置判断；不归为驱动故障。 |

`nvidia-smi` 的 CUDA 字段表示驱动支持的 CUDA 版本，不证明已安装相应 `nvcc`/toolkit；`torch.version.cuda` 是当前 PyTorch 构建使用的 CUDA 版本。硬件数量、版本、空闲卡和占用进程均以本次检查为准。

复用现有环境和权重。此技能不授权重装驱动/CUDA/PyTorch、下载替代权重、量化、训练、重启机器或终止其他使用者进程。显存不足时按任务既有授权选择等待、分批或已支持的多卡方案，不能静默改变实验条件。

## 证据与权限边界

历史设备访问和真实 CUDA 验证见 [GPU 验收记录](../../plans/autoresearch/ACCEPTANCE.md)；本地原始证据保存在 `.dev/artifacts/gpu-20261014-w1d2iqko/sandbox-nvidia-diagnostic.json`、`commands/hardware.json` 和 `commands/torch-cuda.stdout.txt`（后两者位于同一产物目录）。历史产物用于追溯，不是运行本技能的前置条件。

沙箱与审批是不同控制，按当前会话有效权限执行；工具异常不能成为自行放宽权限的理由。权限概念参见 [OpenAI 官方沙箱说明](https://learn.chatgpt.com/docs/sandboxing)，本机 GPU 访问差异的判断依据是上述实测记录。
