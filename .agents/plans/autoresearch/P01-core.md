# P01 共享契约与账本

## 需求
统一状态、环境、模型、搜索与成本接口。

定位：AR-0/4/5/7。先说明本次输入输出、上下游和验证方式。前置任务：P00。
首选 Agent：gpt-6-astra，reasoning_effort=ultra。
状态：cpu_verified。
主控在派发前记录实际 base SHA，在交付后记录 commit、准确命令、退出码和证据。

背景：[长期目标](../../skills/jevgo-autoresearch/references/project-goal.md)、[总计划](README.md)、[共享契约](CONTRACTS.md)。

## 实施大纲
公共类型、配置校验、证据事件、预算账本、能力协议、事件存储与测试替身；冻结准确签名。

只修改以下范围：jevgo/core.py;jevgo/__init__.py;子包空__init__.py;.dev/tests/autoresearch/test_core.py;CONTRACTS.md。共享接口/配置由owner统一改；其他范围变更先向主控报告。使用独立worktree，不能在共享主工作区切分支。

## 验收标准
序列化、不可变状态和证据隔离、恢复不回滚成本、成功与耗尽分离、拒绝非none训练。

验收命令：
```bash
cd /mnt/nas-tianwen/home/ssy/JevGO/.dev/worktrees/P01
JEVGO_ASSETS_ROOT=/mnt/nas-tianwen/home/ssy/JevGO /mnt/nas-tianwen/home/ssy/JevGO/.venv/bin/python -B -m pytest -q .dev/tests/autoresearch/test_core.py
```

报告放本工作树 .dev/artifacts/P01/；缓存/临时代码放 .dev/cache/、.dev/tmp/。源码和小fixture进Git，正式代码不得依赖.dev。

## 独立执行与交接
- 项目根/共享资源根：/mnt/nas-tianwen/home/ssy/JevGO。
- branch：codex/ar-p01-core；worktree：/mnt/nas-tianwen/home/ssy/JevGO/.dev/worktrees/P01。
- 基线：由主控派发时记录依赖满足的实际commit；不得自行从过期main开工。
- shell=/bin/bash，login=false；旧小写jevgo目录不存在。
- 使用共享 .venv/bin/python 与 JEVGO_ASSETS_ROOT，不复制/重装环境和权重。
- 读取worktree中的AGENTS.md以及autoresearch和release-layout技能。
- 缺GPU/数据时完成独立工作，明确阻塞。随机模型、依赖smoke、fixture不是正式效果。
- 只本地提交本任务范围；回传SHA、命令/退出码、证据、验证层级和剩余事项，由主控合入集成分支。
- 再拆分时向主控提交依赖和子任务，不自行占用更多agent槽位。新对话凭本文件及契约即可接手。

## 当前执行记录

- 基线：`P00`。
- 交付 commit：`f8b792e + d284cbc`。
- 验收：25 项公共契约和预算测试。
- 验证命令采用本卡固定 worktree/解释器；退出码及最终集成证据见 [ACCEPTANCE.md](ACCEPTANCE.md)。
- 共享资源未复制或重装，正式代码不依赖开发目录。真实模型与正式研究效果交接见 [GPU-HANDOFF.md](GPU-HANDOFF.md)。
