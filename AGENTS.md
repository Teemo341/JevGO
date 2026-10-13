# JevGO 项目约定

在本项目新增或修改代码、测试、辅助脚本，或准备发布前，读取并遵循项目技能：

[jevgo-release-layout](.agents/skills/jevgo-release-layout/SKILL.md)

正式入口保留在 `scripts/`；开发测试、工具和产物统一放 `.dev/`。发布文件必须在删除 `.dev/` 后仍可独立安装和运行。
