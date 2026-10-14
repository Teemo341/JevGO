# P09 ScienceWorld适配

## 需求
JVM任务与正确成功失败语义。

定位：AR-0/4。先说明本次输入输出、上下游和验证方式。前置任务：P01。
首选 Agent：gpt-6-astra，reasoning_effort=ultra。
状态：cpu_verified。主控在派发前记录实际 base SHA，在交付后记录 commit、准确命令、退出码和证据。

背景：[长期目标](../../skills/jevgo-autoresearch/references/project-goal.md)、[总计划](README.md)、[共享契约](CONTRACTS.md)。

## 实施大纲
固定task/variation/simplifications、独立JVM、reset/replay、内部观察计数、关闭生命周期。

只修改以下范围：jevgo/envs/scienceworld.py;.dev/tests/autoresearch/test_scienceworld.py。共享接口/配置由owner统一改；其他范围变更先向主控报告。使用独立worktree，不能在共享主工作区切分支。

## 验收标准
恢复同前缀一致；done未满分不成功；失败超步数区分；无残留JVM。

验收命令：
```bash
cd /mnt/nas-tianwen/home/ssy/JevGO/.dev/worktrees/P09
JEVGO_ASSETS_ROOT=/mnt/nas-tianwen/home/ssy/JevGO /mnt/nas-tianwen/home/ssy/JevGO/.venv/bin/python -B -m pytest -q .dev/tests/autoresearch/test_scienceworld.py
```

报告放本工作树 .dev/artifacts/P09/；缓存/临时代码放 .dev/cache/、.dev/tmp/。源码和小fixture进Git，正式代码不得依赖.dev。

## 独立执行与交接
- 项目根/共享资源根：/mnt/nas-tianwen/home/ssy/JevGO。
- branch：codex/ar-p09-scienceworld；worktree：/mnt/nas-tianwen/home/ssy/JevGO/.dev/worktrees/P09。
- 基线：由主控派发时记录依赖满足的实际commit；不得自行从过期main开工。
- shell=/bin/bash，login=false；旧小写jevgo目录不存在。
- 使用共享 .venv/bin/python 与 JEVGO_ASSETS_ROOT，不复制/重装环境和权重。
- 读取worktree中的AGENTS.md以及autoresearch和release-layout技能。
- 缺GPU/数据时完成独立工作，明确阻塞。随机模型、依赖smoke、fixture不是正式效果。
- 只本地提交本任务范围；回传SHA、命令/退出码、证据、验证层级和剩余事项，由主控合入集成分支。
- 再拆分时向主控提交依赖和子任务，不自行占用更多agent槽位。新对话凭本文件及契约即可接手。

## 当前执行记录

- 基线：`7b65011 + d284cbc`。
- 交付 commit：`d69d741`。
- 验收/进度：4 real JVM tests; successful process cleanup。
- 已执行测试均退出 0；原始输出在对应工作树 `.dev/artifacts/` 或主控集成验收记录。
