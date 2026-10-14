# P07 运行器与四种方法

## 需求
完整装配B0至B3和CLI。

定位：AR-0至7。先说明本次输入输出、上下游和验证方式。前置任务：P02,P03,P04,P05,P06。
首选 Agent：gpt-6-astra，reasoning_effort=ultra。
状态：planned。主控在派发前记录实际 base SHA，在交付后记录 commit、准确命令、退出码和证据。

背景：[长期目标](../../skills/jevgo-autoresearch/references/project-goal.md)、[总计划](README.md)、[共享契约](CONTRACTS.md)。

## 实施大纲
模型环境工厂、研究状态、MCTS域、提交预算预留、真实动作提交、重新规划、配置轨迹保存与关闭。

只修改以下范围：jevgo/runner.py;jevgo/__main__.py;jevgo/factory.py;configs/autoresearch*.yaml;.dev/tests/autoresearch/test_runner.py。共享接口/配置由owner统一改；其他范围变更先向主控报告。使用独立worktree，不能在共享主工作区切分支。

## 验收标准
四方法真实MiniGrid+确定替身运行；搜索提交分开；异常关闭；CLI可复现；无.dev运行依赖。

验收命令：
```bash
cd /mnt/nas-tianwen/home/ssy/JevGO/.dev/worktrees/P07
JEVGO_ASSETS_ROOT=/mnt/nas-tianwen/home/ssy/JevGO /mnt/nas-tianwen/home/ssy/JevGO/.venv/bin/python -B -m pytest -q .dev/tests/autoresearch/test_runner.py
```

报告放本工作树 .dev/artifacts/P07/；缓存/临时代码放 .dev/cache/、.dev/tmp/。源码和小fixture进Git，正式代码不得依赖.dev。

## 独立执行与交接
- 项目根/共享资源根：/mnt/nas-tianwen/home/ssy/JevGO。
- branch：codex/ar-p07-runner；worktree：/mnt/nas-tianwen/home/ssy/JevGO/.dev/worktrees/P07。
- 基线：由主控派发时记录依赖满足的实际commit；不得自行从过期main开工。
- shell=/bin/bash，login=false；旧小写jevgo目录不存在。
- 使用共享 .venv/bin/python 与 JEVGO_ASSETS_ROOT，不复制/重装环境和权重。
- 读取worktree中的AGENTS.md以及autoresearch和release-layout技能。
- 缺GPU/数据时完成独立工作，明确阻塞。随机模型、依赖smoke、fixture不是正式效果。
- 只本地提交本任务范围；回传SHA、命令/退出码、证据、验证层级和剩余事项，由主控合入集成分支。
- 再拆分时向主控提交依赖和子任务，不自行占用更多agent槽位。新对话凭本文件及契约即可接手。
