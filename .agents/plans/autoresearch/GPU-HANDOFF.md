# 独立 GPU 验收交接

本文件可交给新对话独立执行。研究目标是检验冻结 LLM、JEV 和 MCTS 是否改善自动研究/交互任务成功率，并核算额外成本；不预设提升。先读 `AGENTS.md`、autoresearch skill、`docs/autoresearch.md` 和本目录 `CONTRACTS.md`。本任务定位 AR-2/4/5/6/7、EVAL；不训练。

## 已完成与待完成

CPU 已完成公共契约、真实 MiniGrid 四方法闭环、AutoTrust 协议、SemIf 随机小模型、三个可恢复环境适配及 DiscoveryWorld 在线适配。真实 checkpoint 推理与正式研究效果均尚未验收。本次机器 `nvidia-smi` 无法连接驱动，不把权重存在或 mock 通过当作 GPU 通过。

- 集成分支：`codex/autoresearch`；代码基线取本目录 `ACCEPTANCE.md` 和开始执行时的 `git rev-parse HEAD`。
- 项目/共享资源根：`/mnt/nas-tianwen/home/ssy/JevGO`，旧小写 `jevgo` 目录不存在。
- Python：`/mnt/nas-tianwen/home/ssy/JevGO/.venv/bin/python`；不要重装或复制大环境/权重。
- 独立修改使用新 branch/worktree；没有算法变更时只产生验证产物，不改代码。测试在工作树用绝对解释器；不要用会切回主根的 env.sh。
- 本轮不允许把显存不足自动转换为量化、换 checkpoint 或训练。资源不足记录 blocked_resource，其余独立验收继续。

## 资源与准备

| backend | 本地路径 | 模式 | 资源说明 |
| --- | --- | --- | --- |
| jev_9b | models/JEV-9B | 本地 vLLM，基座生成 + decision LoRA | 权重约 18.3 GB，另需运行/KV 余量 |
| jev_27b | models/JEV-27B | 本地 typed decide 服务 | 权重约 54.7 GB，另需运行/KV 余量 |
| jev_27b_vl | models/JEV-27B-VL | typed decide + 图像 | 权重约 56.0 GB；max_num_seqs ≤ 8 |
| semif | models/Qwen3.5-4B | 同一 CUDA 模型实例生成/评分 | 权重约 9.3 GB；恰好一张可见 GPU |

这些是权重大小，不是最低显存保证。实际 revision 读取 `configs/jevgo.yaml`，校验资源清单仍以既有 manifest 为准。

```bash
cd /mnt/nas-tianwen/home/ssy/JevGO
export JEVGO_ASSETS_ROOT=/mnt/nas-tianwen/home/ssy/JevGO
export JEVGO_PY=/mnt/nas-tianwen/home/ssy/JevGO/.venv/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 TOKENIZERS_PARALLELISM=false
mkdir -p .dev/artifacts
export JEVGO_GPU_OUT="$(mktemp -d "$JEVGO_ASSETS_ROOT/.dev/artifacts/gpu-XXXXXXXX")"
git rev-parse HEAD > "$JEVGO_GPU_OUT/commit.txt"
nvidia-smi -q > "$JEVGO_GPU_OUT/hardware.txt"
```

若硬件命令失败，保存错误及驱动/GPU可见性，停止依赖 GPU 的步骤。不要把此情况写成算法失败。

## 先验收 9B

在独立终端前台启动服务，保存 stdout/stderr 和 PID；先加 `--dry-run` 检查本地命令，然后移除。不要同时占用多个大模型服务。

```bash
CUDA_VISIBLE_DEVICES=0 "$JEVGO_PY" -B -m jevgo.models.serve_autotrust \
  --model jev_9b --assets-root "$JEVGO_ASSETS_ROOT" \
  --host 127.0.0.1 --port 8009 --max-model-len 4096
```

服务就绪后，在验证终端执行：

```bash
"$JEVGO_PY" -B -m jevgo model-check \
  --model jev_9b --assets-root "$JEVGO_ASSETS_ROOT" \
  --base-url http://127.0.0.1:8009 --max-model-len 4096 \
  --output "$JEVGO_GPU_OUT/jev_9b-protocol"
"$JEVGO_PY" -B -m jevgo run \
  --config configs/autoresearch-smoke.yaml --method B0 --model jev_9b \
  --base-url http://127.0.0.1:8009 --assets-root "$JEVGO_ASSETS_ROOT" \
  --output "$JEVGO_GPU_OUT/jev_9b-B0"
"$JEVGO_PY" -B -m jevgo run \
  --config configs/autoresearch-smoke.yaml --method B3 --model jev_9b \
  --base-url http://127.0.0.1:8009 --assets-root "$JEVGO_ASSETS_ROOT" \
  --output "$JEVGO_GPU_OUT/jev_9b-B3"
```

协议通过条件：`result.ok=true`、4 个请求完整完成、真实 generation 非空、candidate 两个分数对应原选项、evidence/stop 各一个有限分数、model/adapter/revision 与配置相符、usage_complete=true、实际内部请求和 token 有记录。9B 的 logits、bias、温度路径需从响应及 metadata 核对。停止意见不能让未完成环境变成功。

闭环通过条件是所有动作来自可解析提议、真实环境反馈可追溯、搜索与 commit 分开、成本账本可重算、恢复/预算/终止语义正确。模型没有成功完成任务也必须如实保留；不能以强制达到成功率作为实现验收条件。CLI exit=0 不等于 task success。

## 其他模型逐项验收

结束前一服务后，替换启动参数：

| 模型 | 启动 --model | --port / check --base-url | 附加检查 |
| --- | --- | --- | --- |
| 27B | jev_27b | 8027 / http://127.0.0.1:8027 | `/v1/decide` 内部请求数及 fallback usage |
| 27B-VL | jev_27b_vl | 8028 / http://127.0.0.1:8028 | 启动命令并发不超 8，真实本地图像请求 |
| SemIf | 不启动 HTTP 服务 | 不传 base-url | 同一冻结实例 `.generate` 与 direct scorer，未校准标签保留 |

27B/27B-VL 的启动与 model-check 命令沿用上一节，仅替换表中模型/端口、独立输出名。客户端和服务 max_model_len 都保持 4096。

SemIf 直接执行：

```bash
CUDA_VISIBLE_DEVICES=0 "$JEVGO_PY" -B -m jevgo model-check \
  --model semif --assets-root "$JEVGO_ASSETS_ROOT" --max-model-len 4096 \
  --output "$JEVGO_GPU_OUT/semif-protocol"
```

SemIf direct scorer 的 output_tokens 为零是正常现象；输入 tokens、model_requests 和耗时必须完整。生成路径仍有输出 tokens。

为 VL 生成一张真实公开观测（CPU 环境操作，图片保留到 worker 关闭后）：

```bash
"$JEVGO_PY" -B - <<'PY'
import os
from pathlib import Path
import shutil
from jevgo.core import BudgetLedger, EpisodeSpec
from jevgo.envs.discoveryworld import DiscoveryWorldAdapter
out = Path(os.environ['JEVGO_GPU_OUT'])
env = DiscoveryWorldAdapter(assets_root=os.environ['JEVGO_ASSETS_ROOT'], output_dir=out / 'vl-observation-workers')
ledger = BudgetLedger()
try:
    step = env.reset(EpisodeSpec(environment='discoveryworld', task='Tutorial', seed=0,
        options={'difficulty': 'Normal', 'verify_replay': False, 'modality': 'vl'}), ledger=ledger)
    shutil.copyfile(step.observation.image_paths[0], out / 'public-observation.png')
    import json
    (out / 'vl-observation-ledger.json').write_text(json.dumps(ledger.summary(), indent=2))
finally:
    env.close()
PY
"$JEVGO_PY" -B -m jevgo model-check \
  --model jev_27b_vl --base-url http://127.0.0.1:8028 --max-model-len 4096 \
  --image "$JEVGO_GPU_OUT/public-observation.png" \
  --assets-root "$JEVGO_ASSETS_ROOT" --output "$JEVGO_GPU_OUT/jev_27b_vl-image-protocol"
```

只完成文本请求不能标记 VL 图像推理通过。原始模型输出、实际图像路径/哈希和协议 metadata 都需保留。

## 四方法与消融矩阵

每个模型服务就绪时单独执行以下命令，分别替换 `jev_9b` 和输出目录。SemIf 不需要服务，但验证进程须设置 `CUDA_VISIBLE_DEVICES=0`。

```bash
"$JEVGO_PY" -B -m jevgo.evaluation matrix \
  --config configs/autoresearch-matrix.yaml --model jev_9b \
  --assets-root "$JEVGO_ASSETS_ROOT" --output "$JEVGO_GPU_OUT/jev_9b-matrix" --execute
"$JEVGO_PY" -B -m jevgo.evaluation summarize \
  --manifest "$JEVGO_GPU_OUT/jev_9b-matrix/manifest.json" \
  --output "$JEVGO_GPU_OUT/jev_9b-summary.json"
```

当前矩阵为开发 smoke：两 seed、MiniGrid 单任务、四方法与四项 B3 消融。它不是完整 benchmark，也不足以证明算法改善推理精度。所有模型均用 text 观测；VL 信息增益必须另设相同任务的 text/vl 条件，不能混入算法增益。

ALFWorld 正式数据目录为空；小型 fixture 只证明适配器。ScienceWorld 已验证的固定 variation 可扩展为预先声明的任务清单。DiscoveryWorld 当前 B2/B3 为 unsupported（新 worker 私有 RNG 不一致），不能偷偷换成 B0/B1 或计为搜索成功。其完整 MCTS 验证需另起恢复能力任务，明确提出环境随机机制或快照方案再验收。

预算比较须同时检查 summary 的成本完整性、实际模型身份、任务条件和 fair_comparison_eligible。失败、缺失、跳过及未知成本都保留；不同环境分开报告。

## 回收格式与停止条件

每个模型提交一个小 JSON 索引（完整产物目录一并保留）：

```json
{
  "commit": "实际代码 SHA",
  "model_key": "jev_9b",
  "model_revision": "实际 revision",
  "hardware_file": "hardware.txt",
  "protocol_status": "passed|failed|blocked_resource",
  "episode_status": "completed|failed|blocked_resource",
  "research_effectiveness": "pending",
  "artifacts_directory": "绝对路径",
  "commands_and_exit_codes": [],
  "limitations": []
}
```

回传还应包括模型/adapter身份、服务参数、生成和各 JEV 角色的实际 usage、客观任务结果、恢复能力、错误及清理记录。真实模型层和研究效果层分别判定；不要将 `real_model_attempt` 自动改写为已验证。训练模式、SemIf 缓存优化及通用 shell 实验环境不在本任务范围。
