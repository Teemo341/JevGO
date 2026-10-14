# JevGO 首版集成验收

验收日期：2026-10-14。本地集成分支：`codex/autoresearch`。项目和共享资源根：`/mnt/nas-tianwen/home/ssy/JevGO`。

P00–P12 代码、任务文档及 GPU 交接已集成；参数冻结，`learning_mode=none`。独立实现分支与 worktree 保留，未推送。实现与审查子 Agent 使用 gpt-6-astra / ultra，按主控、两名实现、一名审查调度。

## 四层结果

| 层级 | 结果与范围 |
| --- | --- |
| 算法单元 | 通过。显式状态、原始证据、分支隔离、单智能体 PUCT、真实 rollout、准确回传、预算预留、失败和中断行为均有确定性测试。 |
| 真实环境 | MiniGrid、ALFWorld 小型 TextWorld fixture、ScienceWorld JVM 实际 reset/step 和恢复通过。DiscoveryWorld 实际进程/观测/图像隔离通过，恢复验证失败时保留在线模式并拒绝 MCTS。 |
| 真实模型 | **待验收**。AutoTrust 是 CPU 协议测试；SemIf 是随机小模型生成/评分及冻结参数测试。未加载指定大模型权重进行 GPU 推理。 |
| 研究效果 | **待验收**。16 个 CPU fixture 运行仅证明装配、对比和审计链路，不支持真实模型精度或 MCTS 收益结论。 |

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

研究提示稳定别名及失败请求追踪的最终补充验收将单独记录，避免把旧基线测试冒充新代码的结果。

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

- GPU：当前机器 `nvidia-smi` 无法连接驱动。按 [GPU-HANDOFF.md](GPU-HANDOFF.md) 先验收 9B 生成/评分及闭环，再逐个验收 27B、27B-VL 和 SemIf。显存不足记录资源阻塞，不自动量化或换 checkpoint。
- ALFWorld：正式数据目录为空；小型 fixture 仅证明适配器行为。完整数据准备好后再建立正式任务清单。
- DiscoveryWorld：原实例与新进程的私有状态/RNG 不一致，目前只允许 B0/B1；B2/B3 明确报告不支持。恢复成功路径的协议 fixture 不代表真实场景恢复通过。
- 真实模型效果：需在固定任务、seed、观测条件和预算上运行矩阵、消融及成本审计；当前无提升结论。
- 训练、SemIf 缓存优化、通用命令行实验环境属于后续任务池。
