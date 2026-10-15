# P03 AutoTrust后端

## 需求
支持27B文本、27B-VL、9B的生成和评分。

定位：AR-2/5/6/7。先说明本次输入输出、上下游和验证方式。前置任务：P01。
首选 Agent：gpt-6-astra，reasoning_effort=ultra。
状态：gpu_protocol_verified_integration_recheck_pending。主控在派发前记录实际 base SHA，在交付后记录 commit、准确命令、退出码和证据。

背景：[长期目标](../../skills/jevgo-autoresearch/references/project-goal.md)、[总计划](README.md)、[共享契约](CONTRACTS.md)。

## 实施大纲
27B typed decide；9B logits+bias+temperature；adapter切换、内部调用成本、本地服务入口；VL并发上限8。

只修改以下范围：jevgo/models/autotrust.py;jevgo/models/serve_autotrust.py;.dev/tests/autoresearch/test_autotrust.py。共享接口/配置由owner统一改；其他范围变更先向主控报告。使用独立worktree，不能在共享主工作区切分支。

## 验收标准
HTTP fixture验证选项、计算、路由、错误和成本；无隐式下载；真实GPU待验收。

验收命令：
```bash
cd /mnt/nas-tianwen/home/ssy/JevGO/.dev/worktrees/P03
JEVGO_ASSETS_ROOT=/mnt/nas-tianwen/home/ssy/JevGO /mnt/nas-tianwen/home/ssy/JevGO/.venv/bin/python -B -m pytest -q .dev/tests/autoresearch/test_autotrust.py
```

报告放本工作树 .dev/artifacts/P03/；缓存/临时代码放 .dev/cache/、.dev/tmp/。源码和小fixture进Git，正式代码不得依赖.dev。

## 独立执行与交接
- 项目根/共享资源根：/mnt/nas-tianwen/home/ssy/JevGO。
- branch：codex/ar-p03-autotrust；worktree：/mnt/nas-tianwen/home/ssy/JevGO/.dev/worktrees/P03。
- 基线：由主控派发时记录依赖满足的实际commit；不得自行从过期main开工。
- shell=/bin/bash，login=false；旧小写jevgo目录不存在。
- 使用共享 .venv/bin/python 与 JEVGO_ASSETS_ROOT，不复制/重装环境和权重。
- 读取worktree中的AGENTS.md以及autoresearch和release-layout技能。
- 缺GPU/数据时完成独立工作，明确阻塞。随机模型、依赖smoke、fixture不是正式效果。
- 只本地提交本任务范围；回传SHA、命令/退出码、证据、验证层级和剩余事项，由主控合入集成分支。
- 再拆分时向主控提交依赖和子任务，不自行占用更多agent槽位。新对话凭本文件及契约即可接手。

## 当前执行记录

- 基线：`38a0f5d`。
- 交付 commit：`3e6b3ae + d284cbc`。
- 验收：55 项 CPU 协议；真实 checkpoint 推理待 GPU。
- 验证命令采用本卡固定 worktree/解释器；退出码及最终集成证据见 [ACCEPTANCE.md](ACCEPTANCE.md)。
- 共享资源未复制或重装，正式代码不依赖开发目录。真实模型与正式研究效果交接见 [GPU-HANDOFF.md](GPU-HANDOFF.md)。

## 2026-10-15 更新

独立修复版本已完成真实 GPU 生成/评分开发验收，历史执行 commit 和失败见 ACCEPTANCE.md。四项修复已合入集成分支，统一版本的组合回归和协议复核按 [P13](P13-gpu-integration.md) 进行。正式研究效果仍待验证。
