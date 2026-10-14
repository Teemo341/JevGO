# P06 MCTS引擎

## 需求
同步单智能体PUCT搜索。

定位：AR-1/6。先说明本次输入输出、上下游和验证方式。前置任务：P01。
首选 Agent：gpt-6-astra，reasoning_effort=ultra。
状态：running。主控在派发前记录实际 base SHA，在交付后记录 commit、准确命令、退出码和证据。

背景：[长期目标](../../skills/jevgo-autoresearch/references/project-goal.md)、[总计划](README.md)、[共享契约](CONTRACTS.md)。

## 实施大纲
SearchDomain驱动选择扩展真实rollout回传；P/N/W/Q；预算中断；根按访问数Q稳定顺序选择。

只修改以下范围：jevgo/search/mcts.py;.dev/tests/autoresearch/test_mcts.py。共享接口/配置由owner统一改；其他范围变更先向主控报告。使用独立worktree，不能在共享主工作区切分支。

## 验收标准
手算统计一致；无漏计重计或符号反转；发现延迟奖励；预算中断一致。

验收命令：
```bash
cd /mnt/nas-tianwen/home/ssy/JevGO/.dev/worktrees/P06
JEVGO_ASSETS_ROOT=/mnt/nas-tianwen/home/ssy/JevGO /mnt/nas-tianwen/home/ssy/JevGO/.venv/bin/python -B -m pytest -q .dev/tests/autoresearch/test_mcts.py
```

报告放本工作树 .dev/artifacts/P06/；缓存/临时代码放 .dev/cache/、.dev/tmp/。源码和小fixture进Git，正式代码不得依赖.dev。

## 独立执行与交接
- 项目根/共享资源根：/mnt/nas-tianwen/home/ssy/JevGO。
- branch：codex/ar-p06-mcts；worktree：/mnt/nas-tianwen/home/ssy/JevGO/.dev/worktrees/P06。
- 基线：由主控派发时记录依赖满足的实际commit；不得自行从过期main开工。
- shell=/bin/bash，login=false；旧小写jevgo目录不存在。
- 使用共享 .venv/bin/python 与 JEVGO_ASSETS_ROOT，不复制/重装环境和权重。
- 读取worktree中的AGENTS.md以及autoresearch和release-layout技能。
- 缺GPU/数据时完成独立工作，明确阻塞。随机模型、依赖smoke、fixture不是正式效果。
- 只本地提交本任务范围；回传SHA、命令/退出码、证据、验证层级和剩余事项，由主控合入集成分支。
- 再拆分时向主控提交依赖和子任务，不自行占用更多agent槽位。新对话凭本文件及契约即可接手。

## 当前执行记录

- 基线：`38a0f5d + d284cbc`。
- 交付 commit：`pending`。
- 验收/进度：gpt-6-astra ultra, implementation agent。
- 已执行测试均退出 0；原始输出在对应工作树 `.dev/artifacts/` 或主控集成验收记录。
