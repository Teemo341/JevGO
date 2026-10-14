# P11 评测与分析

## 需求
公平比较效果及模型环境成本。

定位：EVAL。先说明本次输入输出、上下游和验证方式。前置任务：P07。
首选 Agent：gpt-6-astra，reasoning_effort=ultra。
状态：cpu_verified_fixture_only。主控在派发前记录实际 base SHA，在交付后记录 commit、准确命令、退出码和证据。

背景：[长期目标](../../skills/jevgo-autoresearch/references/project-goal.md)、[总计划](README.md)、[共享契约](CONTRACTS.md)。

## 实施大纲
任务清单矩阵、B0至B3、JEV角色消融、分模型模态汇总、分析源码和事件重算。

只修改以下范围：jevgo/evaluation.py;.dev/analysis/autoresearch/**;.dev/tests/autoresearch/test_evaluation.py。共享接口/配置由owner统一改；其他范围变更先向主控报告。使用独立worktree，不能在共享主工作区切分支。

## 验收标准
固定任务seed权限预算；失败跳过不成功；成本不遗漏；fixture结果不作模型效果。

验收命令：
```bash
cd /mnt/nas-tianwen/home/ssy/JevGO/.dev/worktrees/P11
JEVGO_ASSETS_ROOT=/mnt/nas-tianwen/home/ssy/JevGO /mnt/nas-tianwen/home/ssy/JevGO/.venv/bin/python -B -m pytest -q .dev/tests/autoresearch/test_evaluation.py
```

报告放本工作树 .dev/artifacts/P11/；缓存/临时代码放 .dev/cache/、.dev/tmp/。源码和小fixture进Git，正式代码不得依赖.dev。

## 独立执行与交接
- 项目根/共享资源根：/mnt/nas-tianwen/home/ssy/JevGO。
- branch：codex/ar-p11-evaluation；worktree：/mnt/nas-tianwen/home/ssy/JevGO/.dev/worktrees/P11。
- 基线：由主控派发时记录依赖满足的实际commit；不得自行从过期main开工。
- shell=/bin/bash，login=false；旧小写jevgo目录不存在。
- 使用共享 .venv/bin/python 与 JEVGO_ASSETS_ROOT，不复制/重装环境和权重。
- 读取worktree中的AGENTS.md以及autoresearch和release-layout技能。
- 缺GPU/数据时完成独立工作，明确阻塞。随机模型、依赖smoke、fixture不是正式效果。
- 只本地提交本任务范围；回传SHA、命令/退出码、证据、验证层级和剩余事项，由主控合入集成分支。
- 再拆分时向主控提交依赖和子任务，不自行占用更多agent槽位。新对话凭本文件及契约即可接手。

## 当前执行记录

- 基线：`7792aeb; 后集成 P07a`。
- 交付 commit：`27fc806 +0897d7a +5c40457 +ecd8abe +0dba698`。
- 验收：最终 12 项分析测试（含有限时间预算的不可比较标识）；16 个两 seed/四方法/消融 CPU 运行及成本重算通过。
- 验证命令采用本卡固定 worktree/解释器；退出码及最终集成证据见 [ACCEPTANCE.md](ACCEPTANCE.md)。
- 共享资源未复制或重装，正式代码不依赖开发目录。真实模型与正式研究效果交接见 [GPU-HANDOFF.md](GPU-HANDOFF.md)。
- 纯分析以已完成 P07 runner 为基线；矩阵执行等待 P07a 合入后再验收。正式 GPU 矩阵按模型单独启动，不同时装载四模型。
