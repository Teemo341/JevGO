# P13 GPU 验收修复集成

## 需求与定位

主环节 EVAL，关联 AR-2/5/7 后端与成本审计。将四个已独立验收的修复合入统一版本，保留用户现有验收记录与 GPU skill，验证组合兼容性。默认 learning_mode=none，不改变 checkpoint、精度、记忆或预算条件。非目标：重跑 64 次矩阵、扩大上下文、正式 benchmark、训练。

依赖 P03/P04/P11/P12；接口沿用 CONTRACTS.md。输入基线 `b198861`；已有用户文档和 skill 已保留为 `50963de`。

| 分支 | commit | 内容 |
| --- | --- | --- |
| codex/gpu-http-error-details | 4f1977125414 | HTTP 失败正文及未知成本保留 |
| codex/gpu-judge-identity | 4249062a5986 | 实际评分事件身份的公平比较审计 |
| codex/gpu-semif-metadata | eb03c1cc2674 | 已知依赖版本字符串正规化 |
| codex/gpu-tensor-parallel | f3a9e753f6dc | 显式 TP，VL 并发上限保持 8 |

## 实施大纲与工作环境

- 主控在 `codex/autoresearch`、`/mnt/nas-tianwen/home/ssy/JevGO` 合入；独立修复 worktree 保留在 `.dev/worktrees/gpu-*`。
- 两个 gpt-6-astra / ultra 子 Agent 分别只读审查后端修复和身份审计；主控负责合并、文档与组合验证。GPU 协议复核按独立产物子目录及明确 GPU 分配执行，不修改共享代码。
- 保留原 `.dev/artifacts/gpu-20261014-w1d2iqko` 全部证据，旧执行 commit 和新审计 commit 分开记录。
- 本轮输出根：`/mnt/nas-tianwen/home/ssy/JevGO/.dev/artifacts/gpu-integration-20261015-ng8z3gda`。
- 允许修改范围：上述四项代码/测试，docs/autoresearch.md，.agents/plans/autoresearch/ 文档；原 skill 变更仅保存不重写。
- GPU 前读取 AGENTS、autoresearch、release-layout、gpu-access skill；共享解释器为项目 `.venv/bin/python`，使用显式 assets root，不重装资源。

## 验收标准与命令

1. 四项修复均已进入集成分支，现有文档与 skill 保留；代码及报告中的 no-training、身份、预算边界不变。
2. 组合后的相关 CPU 回归通过，含发布归档；结果记录受测 commit、准确命令和退出码。
3. 集成代码对原 64 次运行只做重新审计，客观结果与成本不变，保存新汇总，原报告不修改。
4. 对集成版本复核四模型生成与 candidate/evidence/stop 协议，VL 含原真实环境图片；记录本轮实际 commit、设备、TP、usage 与自有进程清理。不以协议成功替代 9B 旧闭环失败。

```bash
cd /mnt/nas-tianwen/home/ssy/JevGO
JEVGO_ASSETS_ROOT="$PWD" .venv/bin/python -B -m pytest -q -p no:cacheprovider \
  .dev/tests/autoresearch/test_autotrust.py .dev/tests/autoresearch/test_semif.py \
  .dev/tests/autoresearch/test_evaluation.py .dev/tests/autoresearch/test_cli.py \
  .dev/tests/autoresearch/test_core.py .dev/tests/autoresearch/test_release.py
```

GPU 命令使用 GPU-HANDOFF.md 的 model-check，服务/客户端 max_model_len=4096。27B 与 VL 显式 TP=2；SemIf 一张可见 GPU。按实时占用选择设备，只结束本轮启动的进程。

## 状态与交接

状态：complete。四项修复已审查并合入，无冲突；用户记录保留于50963de。统一受测 commit `89214798653679b143006c4ce0e99b1c36ad71f4`；收尾提交仅改文档。组合CPU相关回归197项通过（133.82秒，rc0），四模型真实协议均通过（新增16请求，VL含图），旧64次重审计的结果/费用不变（新增请求0）。本轮自有进程及端口已清理。

准确命令、源文件/证据哈希、设备/TP及成本见 [机器可读报告](../../../.dev/artifacts/gpu-integration-20261015-ng8z3gda/integration-report.json) 和 [验收记录](ACCEPTANCE.md)。首次TP启动的IPC路径错误与短路径复验均保留，不改变模型条件。历史64次开发结果不改标为本次执行结果；9B旧长闭环HTTP400失败仍待预试，正式研究效果仍 pending。

下一阶段 [P14](P14-pretrial.md)（已准备，未执行）：先做上下文/记忆/输出预留与累计预算预试，复验 9B 独立闭环及诊断 SemIf 候选拒绝。每种新条件独立保存；只有稳定后才冻结正式评测任务、seed、预算与失败分类。
