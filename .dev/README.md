# 开发工作区（发布时排除整个目录）

| 目录 | 内容 |
| --- | --- |
| `tests/` | 组件 smoke、兼容性检查及统一验证入口 |
| `tools/` | wheel 构建、导出、元数据恢复和离线清单维护 |
| `fixtures/` | 测试专用数据，如 ALFWorld 最小任务 |
| `artifacts/` | 验证结果、构建日志及历史记录 |
| `cache/`、`tmp/` | 随机测试模型与可再生成的临时文件 |

在项目根目录执行：

```bash
bash scripts/env.sh jevgo python .dev/tests/validate_envs.py
```

默认结果写入 `.dev/artifacts/validation/`。只验证指定组件时，在命令末尾追加 `minigrid`、`alfworld` 等名称。

有意更新离线包后，运行 `.dev/tools/refresh_offline_manifest.py`，再执行 `bash scripts/bootstrap.sh` 和上述验证。构建脚本依赖 `.cache/offline/provenance/` 中固定的源码归档与补丁。

可复用分析代码（analysis/）、测试（tests/）、工具（tools/）和小型稳定素材（fixtures/）提交到 Git。日志（logs/）、结果（artifacts/）、缓存（cache/）、临时代码（tmp/）和工作树（worktrees/）不提交。整个 `.dev/` 不进入发布包。正式入口不依赖本目录；项目长期规则见 `.agents/skills/jevgo-release-layout/SKILL.md`。保留在 `.cache/offline/` 的构建来源与哈希属于交付材料，不是临时日志。
