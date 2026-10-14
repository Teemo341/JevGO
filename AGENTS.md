# JevGO 项目约定

在本项目新增或修改代码、测试、辅助脚本，或准备发布前，读取并遵循项目技能：

[jevgo-release-layout](.agents/skills/jevgo-release-layout/SKILL.md)

正式入口保留在 `scripts/`；开发测试、工具和产物统一放 `.dev/`。发布文件必须在删除 `.dev/` 后仍可独立安装和运行。

## Autoresearch 目标与算法定位

设计、实现、修改或评审 autoresearch 框架算法及其对比实验前，读取：

[jevgo-autoresearch](.agents/skills/jevgo-autoresearch/SKILL.md)

总体目标、核心循环与实验协议保存在该技能的 [长期项目记忆](.agents/skills/jevgo-autoresearch/references/project-goal.md)。在动手前先向用户简述：本次对应的 AR 环节、目标、输入输出、上下游关系、验证方式，以及对无训练模式和公平对比条件的影响；跨环节工作说明主环节和关联环节。定位不构成额外审批，不因需要定位而停下来等确认。

触发范围包括研究状态、候选生成/筛选、实验编排、MCTS、证据记忆、价值回传、停止与预算，以及评测协议。纯环境维护、权重下载和不改变框架契约的任务专用算法无需套用完整定位；单个环境适配若改变观测、动作、分支恢复、成功信号或成本计数，则按受影响环节定位。

现有环境、权重与版本以仓库配置和验证记录为准。此记忆不授权重复安装、下载或默认开启训练。
