# JevGO 首版集成验收

验收日期：2026-10-14。本地集成分支：`codex/autoresearch`。项目和共享资源根：`/mnt/nas-tianwen/home/ssy/JevGO`。

P00–P12 代码、任务文档及 GPU 交接已集成；参数冻结，`learning_mode=none`。独立实现分支与 worktree 保留，未推送。实现与审查子 Agent 使用 gpt-6-astra / ultra，按主控、两名实现、一名审查调度。

## 四层结果

| 层级 | 结果与范围 |
| --- | --- |
| 算法单元 | 通过。显式状态、原始证据、分支隔离、单智能体 PUCT、真实 rollout、准确回传、预算预留、失败和中断行为均有确定性测试。 |
| 真实环境 | MiniGrid、ALFWorld 小型 TextWorld fixture、ScienceWorld JVM 实际 reset/step 和恢复通过。DiscoveryWorld 实际进程/观测/图像隔离通过，恢复验证失败时保留在线模式并拒绝 MCTS。 |
| 真实模型 | **本轮 GPU 验收已完成（2026-10-15）**。四模型真实生成/评分通过（VL 含实图；SemIf 使用独立修复）；矩阵及独立 smoke 的客观结果和失败分别保留，详见下方。 |
| 研究效果 | **正式 benchmark 待验收**。本轮仅为两 seed、单个 MiniGrid 任务的开发 smoke；不支持模型精度排名或算法提升结论。 |

## 全套 CPU 回归

代码基线 `fd07d237c5e73aac8fe267bb9a0648350ae0c252`：**277 passed，173.28 秒，退出码 0**，无跳过。命令：

```bash
cd /mnt/nas-tianwen/home/ssy/JevGO
set -o pipefail
JEVGO_ASSETS_ROOT=/mnt/nas-tianwen/home/ssy/JevGO .venv/bin/python -B -m pytest -q -p no:cacheprovider \
  --junitxml=.dev/artifacts/P12/cpu-junit.xml .dev/tests/autoresearch \
  2>&1 | tee .dev/artifacts/P12/cpu-pytest.txt
```

证据：[JUnit](../../../.dev/artifacts/P12/cpu-junit.xml)、[完整输出](../../../.dev/artifacts/P12/cpu-pytest.txt)。

| 模块 | 全套回归测试数 |
| --- | ---: |
| 共享契约/预算 | 25 |
| MiniGrid | 21 |
| AutoTrust | 55 |
| SemIf | 42 |
| 研究状态转换 | 8 |
| MCTS | 28 |
| 运行器 | 13 |
| CLI/装配 | 22 |
| ALFWorld | 6 |
| ScienceWorld | 4 |
| DiscoveryWorld | 34 |
| 评测/审计 | 9 |
| 发布归档 | 10 |

## 最终补丁回归

最终受测代码基线 `147b97305f926a95a6ad91446514120c36912ca4`：**72 passed，33.43 秒，退出码 0**，无跳过。包含研究状态 15、运行器 13、CLI 22、评测 12、发布归档 10 项，验证稳定证据/图片别名、调用前请求日志、明确计时范围及有限时间预算的比较限制。72 项与上面的 277 项有重叠，不能相加作为独立测试数；其他模块代码在本轮补丁中未改动。

```bash
cd /mnt/nas-tianwen/home/ssy/JevGO
set -o pipefail
JEVGO_ASSETS_ROOT=/mnt/nas-tianwen/home/ssy/JevGO .venv/bin/python -B -m pytest -q -p no:cacheprovider \
  --junitxml=.dev/artifacts/P12/final-followup-junit.xml \
  .dev/tests/autoresearch/test_research.py .dev/tests/autoresearch/test_runner.py \
  .dev/tests/autoresearch/test_cli.py .dev/tests/autoresearch/test_evaluation.py \
  .dev/tests/autoresearch/test_release.py \
  2>&1 | tee .dev/artifacts/P12/final-followup-pytest.txt
```

证据：[最终补丁 JUnit](../../../.dev/artifacts/P12/final-followup-junit.xml)、[完整输出](../../../.dev/artifacts/P12/final-followup-pytest.txt)、[最终代码发布验收](../../../.dev/artifacts/P12/release-147b97305f92-nn24dwxz/acceptance.json)。发布归档的 B0/B3 真实 MiniGrid 闭环再次通过；B3 结果包含明确的计时范围 metadata。

## 对比矩阵与成本审计

在 P11 独立工作树、基线 `43835c00831698e667abd1639fa4e22a65d63c51` 执行：

```bash
cd /mnt/nas-tianwen/home/ssy/JevGO/.dev/worktrees/P11
export JEVGO_ASSETS_ROOT=/mnt/nas-tianwen/home/ssy/JevGO
export JEVGO_PY=/mnt/nas-tianwen/home/ssy/JevGO/.venv/bin/python
"$JEVGO_PY" -B -m jevgo.evaluation matrix \
  --config configs/autoresearch-cpu-matrix.yaml --assets-root "$JEVGO_ASSETS_ROOT" \
  --output .dev/artifacts/P11/cpu-matrix --execute
"$JEVGO_PY" -B -m jevgo.evaluation summarize \
  --manifest .dev/artifacts/P11/cpu-matrix/manifest.json \
  --output .dev/artifacts/P11/cpu-summary.json
```

两 seed ×（B0、B1、B2、B3 full 与四个 B3 单项消融）共 **16/16** 完成；各运行的客观成功、预算内成功和已知成本审计通过，退出码 0。模型身份全部是 `deterministic_fixture`，`validation_level=cpu_fixture`。脚本后端专门用于 MiniGrid 五步路径，这个成功率不能解读成模型效果。

原始 [manifest](../../../.dev/worktrees/P11/.dev/artifacts/P11/cpu-matrix/manifest.json)、[JSON 汇总](../../../.dev/worktrees/P11/.dev/artifacts/P11/cpu-summary.json)、[CSV 汇总](../../../.dev/worktrees/P11/.dev/artifacts/P11/cpu-summary.csv) 均保留。汇总重算事件账本、预算和实际模型身份，并校验相同任务/seed/权限/预算条件；失败、跳过、成本未知不能算已证实的预算内成功。此矩阵未设置 wall_seconds 预算；该字段仅表示插桩操作加 MCTS 开销，不能解释为端到端时延或公平的速度比较。

## 发布与已有环境

全套回归中的 10 项发布测试对 `git archive HEAD` 产物执行检查。归档排除了 `.dev/`、`.agents/`、`AGENTS.md`、环境和大资源；移除 PYTHONPATH，阻止向上发现主仓库，导入解析到归档内代码，并实际完成 B0/B3 MiniGrid 五步闭环。归档命令、入口 help、shell 语法及运行退出码均为 0。

[发布验收 JSON](../../../.dev/artifacts/P12/release-fd07d237c5e7-t327ve7y/acceptance.json) 记录源 commit、归档路径、解释器、共享资源路径及准确命令。

既有 `bash scripts/env.sh jevgo --check` 退出码 0，检查 245 wheels、4356 files、1050 symlinks（size-only）及 245 个已安装包兼容性；没有重装。证据在 [共享环境检查](../../../.dev/worktrees/P12/.dev/artifacts/P12/release-6107fc6fbaaa-ajtvxdmr/shared-environment-check.json)。本次未重新制作完整离线 bundle，也未声称在空机器上完成全新离线安装。

## 后续执行条件

- GPU：原驱动连接失败已定位为执行沙箱隐藏设备；沙箱外驱动与 CUDA 正常，未修改驱动。已按 [GPU-HANDOFF.md](GPU-HANDOFF.md) 的验收条件和用户后续并行授权完成本轮真实模型验收；旧单卡资源阻塞与后续 TP=2 复验分开记录，不自动量化或换 checkpoint。
- ALFWorld：正式数据目录为空；小型 fixture 仅证明适配器行为。完整数据准备好后再建立正式任务清单。
- DiscoveryWorld：原实例与新进程的私有状态/RNG 不一致，目前只允许 B0/B1；B2/B3 明确报告不支持。恢复成功路径的协议 fixture 不代表真实场景恢复通过。
- 真实模型效果：本轮固定条件的开发矩阵、消融及成本审计已完成；正式多任务 benchmark 尚未执行，当前无算法提升结论。
- 训练、SemIf 缓存优化、通用命令行实验环境属于后续任务池。

最终代码还对已有 16 次运行重新执行纯审计汇总（未重新执行模型或环境），退出码 0，结果仍为 16/16 预算内客观成功；没有有限时间预算，比较条件成立。证据：[最终审计汇总](../../../.dev/artifacts/P12/cpu-summary-reviewed.json)。命令：

```bash
.venv/bin/python -B -m jevgo.evaluation summarize \
  --manifest .dev/worktrees/P11/.dev/artifacts/P11/cpu-matrix/manifest.json \
  --output .dev/artifacts/P12/cpu-summary-reviewed.json
```

<!-- gpu-20261014-w1d2iqko:start -->
## GPU 验收本轮结果（2026-10-14–2026-10-15，已完成）

起始 HEAD：`b198861f8dc5ea5d50d4e244af1a404e54877c2c`；记录完成时实际 HEAD：`b198861f8dc5ea5d50d4e244af1a404e54877c2c`，分支 `codex/autoresearch`。复用已有 .venv、权重和资源，`learning_mode=none`；没有安装、下载、量化、替换 checkpoint 或训练。实现修复均在独立分支/worktree，未合入集成分支。

本轮 [机器可读验收报告](../../../.dev/artifacts/gpu-20261014-w1d2iqko/acceptance-report.json)、[逐方法/消融成本表](../../../.dev/artifacts/gpu-20261014-w1d2iqko/matrix-costs.csv) 和 [命令、退出码及原始标准输出/错误](../../../.dev/artifacts/gpu-20261014-w1d2iqko/commands/) 保留完整证据。验收完成指所有要求的执行、审计及清理已结束，不表示所有测试通过或所有任务成功。报告中的 completed_with_errors 保留模型输出拒绝等错误，具体原因见终止分类。

**驱动与硬件**：原 nvidia-smi 连接失败来自执行沙箱隐藏 NVIDIA 设备（内部 rc9、外部 rc0），未修改驱动。4×NVIDIA A40，每卡 46,068 MiB，驱动 575.64.03、CUDA 12.9，现有 PyTorch 2.13.0+cu129 可运行。最初保留其他使用者的 GPU 0–2 进程，仅用 GPU 3；原使用方释放后按用户授权并行。27B/VL 原 TP=1 单卡权重容量不足记录完整保留，随后以原精度 TP=2 在 GPU 0+1 依次真实复验；SemIf 在 GPU 2/3 分批，活动行未中断重排。

**真实模型协议**：四模型生成均非空，candidate/evidence/stop 顺序、有限评分、实际身份与完整 usage 均通过；每个协议完成 4 个真实请求。VL 四请求均携带真实公开环境图片，文件路径及 SHA256 已核验。

| 模型 | 协议 | 输入 / 输出 tokens | 推理执行 commit | 本地模型 revision |
| --- | --- | ---: | --- | --- |
| jev_9b | 通过 | 396 / 16 | `b198861f8dc5` | `7ac669b8fb476ba926491e3d1c39f9330d68dbed` |
| jev_27b | 通过 | 396 / 21 | `f3a9e753f6dc` | `14b425269ed24452e0a5ad5a8f3e51d139647a88` |
| jev_27b_vl | 通过（实图） | 6,548 / 18 | `f3a9e753f6dc` | `da7a40b6450584e5843fab57bb4d80e092b377e0` |
| semif | 通过（修复后） | 588 / 20 | `eb03c1cc2674` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |

revision 由本地配置、资源及启动证据交叉确认，不是远端 HTTP revision 证明；大权重文件本轮验证清单尺寸，未重新计算全部内容哈希。SemIf 实现 revision 为 `23cf1f39fc9534fe81437200959b6dfc7106e45a`；冻结共享实例与未校准标签保留，direct scorer 的零输出 tokens 正常。VL 原始图片请求记录与图片保留，但未另捕获原始 HTTP 请求体；协议通过不代表图像理解 benchmark。

**环境闭环与客观结果**：9B 独立 B0/B3 smoke 均因 4096 上下文超限而未通过完整闭环，分别在 9/3 次提交后 HTTP 400，客观成功均 false。失败前动作来源、真实反馈、搜索/提交隔离和恢复审计通过；失败请求的 token usage 未知，这两项成本只能作为已知下界。矩阵采用原开发配置（与独立 smoke 的记忆条数不同），其结果分别保留，不能替代独立 smoke 失败。

**开发矩阵**：每模型原始两 seed ×（B0、B1、B2、B3 full 及四项 B3 消融）共 16 条，均为 MiniGrid-Empty-5x5-v0、text 观测。下表仅包含矩阵成本，协议与独立 smoke 成本另计。

| 模型 | 完成 | 客观成功 | 模型请求 | 输入 tokens | 输出 tokens | 环境步（含 replay） | 插桩秒数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| jev_9b | 16/16 | 0/16 | 9,637 | 16,309,947 | 1,214,545 | 9,293 | 13,187.555 |
| jev_27b | 16/16 | 16/16 | 4,469 | 7,336,854 | 601,670 | 3,244 | 18,154.764 |
| jev_27b_vl | 16/16 | 14/16 | 3,588 | 5,789,825 | 529,749 | 3,003 | 21,173.913 |
| semif | 16/16 | 0/16 | 12,314 | 20,196,992 | 1,303,476 | 6,096 | 78,053.910 |

| 方法 / 消融 | jev_9b | jev_27b | jev_27b_vl | semif |
| --- | ---: | ---: | ---: | ---: |
| B0 / full | 0/2 | 2/2 | 1/2 | 0/2 |
| B1 / full | 0/2 | 2/2 | 1/2 | 0/2 |
| B2 / full | 0/2 | 2/2 | 2/2 | 0/2 |
| B3 / full | 0/2 | 2/2 | 2/2 | 0/2 |
| B3 / no_candidate_judge | 0/2 | 2/2 | 2/2 | 0/2 |
| B3 / no_evidence_judge | 0/2 | 2/2 | 2/2 | 0/2 |
| B3 / no_stop_judge | 0/2 | 2/2 | 2/2 | 0/2 |
| B3 / no_working_memory | 0/2 | 2/2 | 2/2 | 0/2 |

上表为客观成功次数 / 两个预声明种子；每组详细请求、tokens 和环境成本见成本 CSV。

各模型终止分类与审计：

- **jev_9b**：12 × budget_exhausted, 4 × environment_truncated；环境证据 {'passed': 16}；成本完整=True；fair_comparison_eligible=True。
- **jev_27b**：16 × success；环境证据 {'passed': 16}；成本完整=True；fair_comparison_eligible=True。
- **jev_27b_vl**：2 × environment_truncated, 14 × success；环境证据 {'passed': 16}；成本完整=True；fair_comparison_eligible=True。
- **semif**：12 × budget_exhausted, 2 × environment_truncated, 2 × invalid_candidates；环境证据 {'passed': 16}；成本完整=True；fair_comparison_eligible=True。

SemIf 原始协议在模型加载后的 TorchVersion 元数据校验处失败（0/4 请求，命令 85.661 秒），错误及零推理 usage 保留；加载耗时未单独插桩，不能将零请求解释为零启动成本。修复后协议与矩阵使用独立执行 commit。B0 无效候选保留原始模型输出、一次修复尝试及解析错误；预算预检停止与模型输出拒绝分别报告，不归为 GPU 故障。

环境步已包含 replay，禁止再次相加。wall_seconds 仅为插桩操作加搜索开销，不是端到端耗时；并行命令存在时间重叠，不能据此比较总速度。实际请求成本来自事件/账本重算，不以预留预算代替实际调用。27B 最终 typed 核验为 1,099 个生成请求加 3,370 个评分请求，共 4,469；证据评分逐条执行导致内部请求数增加。VL 最终 typed 核验为 1,006 个生成请求加 2,582 个实际评分请求，共 3,588。两模型未观察到额外计数 fallback；异常 fallback 路径未覆盖，不能声称验证了所有 fallback 情形。

分片合并校验原始 payload/run_id、唯一性、来源哈希及委派关系；原执行 manifest 与行政占位冲突保留。最终 selected manifest、原版汇总与身份修复版汇总分开保存；纯成本/身份审计使用 `4249062a59869d84de2366081c06fb6f047ac1b3`，不能当作推理执行 commit。

**独立修复及相关测试**（不重复完整 CPU 回归，测试数存在交叉，不相加）：

1. `4f1977125414e9578d432420d9fccea8ef89c565`，`codex/gpu-http-error-details`：保留 HTTP 失败响应正文；59 项通过，rc0。
2. `4249062a59869d84de2366081c06fb6f047ac1b3`，`codex/gpu-judge-identity`：按实际 judge 身份审计，身份缺失或混合不允许公平比较；37 项通过，rc0。
3. `eb03c1cc26745e1fc2ee7992cacc7b4dbc193328`，`codex/gpu-semif-metadata`：仅规范化已知依赖的版本字符串子类，保留其他严格 JSON 校验；71 项通过，rc0。
4. `f3a9e753f6dc352e0e5a032a5741c04161a29668`，`codex/gpu-tensor-parallel`：显式正整数 TP 参数、默认 1 保持旧命令，VL max_num_seqs≤8；66 项通过，rc0，并完成真实 TP=2 协议/矩阵。

**清理与范围限制**：[最终清理记录](../../../.dev/artifacts/gpu-20261014-w1d2iqko/final-cleanup.json) 确认本轮所属 GPU 服务、客户端与监视进程已结束；未终止其他使用者进程。没有剩余 GPU 资源阻塞；初始单卡阻塞历史未删除。9B 独立 smoke 的上下文失败、SemIf 无效候选及所有客观任务失败均保留。

当前仅完成开发 smoke 和实现验收，研究效果仍 pending。两 seed、单任务不支持完整 benchmark、模型精度排名或算法提升结论。VL 矩阵使用 text 观测，实图协议不能混入算法增益；ALFWorld 正式数据缺失和 DiscoveryWorld B2/B3 恢复限制仍沿用上方记录，不在本轮强行替代验证。
<!-- gpu-20261014-w1d2iqko:end -->
