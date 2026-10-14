# JevGO 环境使用

所有组件共用项目根目录的 `.venv`：AutoTrust、SemIf，以及 MiniGrid、ALFWorld、ScienceWorld、DiscoveryWorld。
Python、uv、Java 随 `.tools/` 提供，无需全局安装或手动激活环境。

## 使用

在项目根目录执行：

```bash
# 首次使用或复制到新目录：校验离线材料并创建环境
bash scripts/bootstrap.sh

# 运行自己的脚本
bash scripts/env.sh JevGO python your_script.py

# 检查环境位置、离线文件、依赖锁和已安装包
bash scripts/env.sh JevGO --check
```

普通运行入口在环境缺失或项目路径改变时自动离线重建；`--check` 只检查，缺失时先运行 `bootstrap.sh`。
安装使用 `.cache/offline/` 中的本地包；模型单独下载：

```bash
bash scripts/download_weights.sh download  # 下载全部并逐文件校验 SHA256；中断后重跑可续传
bash scripts/download_weights.sh verify    # 纯离线完整校验；复制后也执行此命令
# 已有环境时可用官方加速传输（完整文件可跳过，未完成单文件可能重下）
bash scripts/download_weights.sh download --transport hf --workers 8
```

共约 138.35 GB，包含 Qwen 本体、JEV LoRA、决策头、校准参数及 tokenizer/processor；版本与 SHA256 固定在 `configs/weights-manifest.json`。

| 用途 | 项目内路径 |
| --- | --- |
| JEV 27B 文本（Qwen3.8） | `models/JEV-27B` |
| JEV 27B VL（Qwen3.8） | `models/JEV-27B-VL` |
| JEV 9B 文本（Qwen3.5） | `models/JEV-9B` |
| SemIf 官方默认基座（无额外训练权重） | `models/Qwen3.5-4B` |

只处理某一套：在命令末尾加 `--model jev_27b`、`--model jev_27b_vl`、`--model jev_9b` 或 `--model semif`。默认下载需要联网和系统 `curl`；`--transport hf` 使用已有环境中的 Hugging Face 加速库（迁移后先 `bootstrap.sh`）。校验不需要网络或 GPU。权重、续传 `.part` 和加速下载缓存均在 `models/`，不依赖外部缓存；推理时传上表本地路径。

## 复制与运行条件

目标系统为 Linux x86_64、glibc ≥2.35、GCC 11 级 libstdc++；GPU 功能需要兼容 CUDA 12.9 的 NVIDIA 驱动。
`/tmp` 需可写；ScienceWorld 需要允许本机 Java 通信。音视频解码另需系统 FFmpeg 共享库。

复制时保留权限、符号链接及隐藏目录，尤其是 `models/`、`.tools/`、`.cache/offline/`、`.third-party/autotrust/`、`configs/`、`scripts/`，以及根目录的 `pyproject.toml`、`uv.lock`、`.python-version`。
发布副本排除 `.dev/`、`.agents/`、`AGENTS.md`；迁移还可排除 `.venv/` 和 `.cache/uv/`。仅 Git clone 不包含完整离线材料。
环境离线材料约 6 GB（不含上述模型），重建后的 `.venv` 约 11 GB。复制后在新项目根目录执行 `bootstrap.sh` 和 `env.sh JevGO --check`。

模型需用上述下载命令准备并通过校验；全量任务数据仍需按实际任务另行准备。已验证最小组件运行和同机异路径离线重建，尚未做跨机实测或完整 benchmark。
