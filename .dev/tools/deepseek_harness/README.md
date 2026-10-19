# GPT 主控与 DeepSeek 执行

本工具仅用于开发协作，不属于 JevGO 的运行依赖。删除 `.dev/` 后，正式安装、模型配置和算法仍可独立运行。GPT 负责规划、架构、分析和验收；官方 DeepSeek Harness 负责限定范围的编码、运行及 debug。

## 安装与 Codex 接入

在项目根运行：

```bash
python3 .dev/tools/deepseek_harness/install.py --register
```

安装使用项目 bundled uv，在 `.dev/cache/deepseek-harness/venv` 创建独立环境，不修改正式 `.venv`、`pyproject.toml` 或项目离线资源。直接依赖固定在 `requirements.txt`：官方 `deepseek-harness-sdk==0.2.1rc2`（预发布）、同版本官方 runtime wheel、`mcp==2.3.0`。网络访问需要 PyPI；本工具不安装模型、不加载 GPU。

注册名为 `jevgo_deepseek`，使用用户 Codex 配置中的本地 STDIO 服务；命令包含绝对路径，启动不依赖 shell 激活。重复安装可用 `--register-only` 仅注册。已有 Codex 客户端在设置中重启该 MCP 服务；若当前聊天工具目录尚未刷新，重启客户端再检查 `/mcp`。无需更改 GPT 主模型或另配 OpenAI API key。

```bash
codex mcp get jevgo_deepseek
.dev/cache/deepseek-harness/venv/bin/python .dev/tools/deepseek_harness/client.py list
.dev/cache/deepseek-harness/venv/bin/python .dev/tools/deepseek_harness/client.py deepseek_health
```

`client.py` 读取已注册的 Codex MCP 命令并通过真实 MCP STDIO 协议调用；`--local` 用于尚未注册时直接验证已安装服务。`health=ready` 只说明版本和凭据存在，联网与凭据有效性由真实任务验证。

## API key

在 [DeepSeek 平台](https://platform.deepseek.com/api_keys) 创建有余额和所需模型权限的 key，然后在本机交互终端执行：

```bash
python3 .dev/tools/deepseek_harness/config.py configure
```

隐藏输入不会回显；配置文件位于 Git 忽略的 `.dev/cache/deepseek-harness/credentials.json`，权限为 `0600`，父目录为 `0700`。不要把 key 粘贴到聊天、命令参数或仓库文件。用相同命令更换 key；删除该私有文件即可移除保存的 key。以下命令只检查状态，不打印 key：

```bash
python3 .dev/tools/deepseek_harness/config.py doctor
```

默认端点为 `https://api.deepseek.com/anthropic`，模型为 `deepseek-flash`（2026-10-19 官方 `/models` 返回的 Flash 名称，已验证固定版本 SDK 可初始化）。官方 Harness 使用 Anthropic Messages 协议，不能把 OpenAI-compatible 根地址 `https://api.deepseek.com` 直接用作这里的端点。配置时可显式指定 `--base-url` 和 `--model`。也支持 `DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL`、`DEEPSEEK_MODEL` 环境覆盖；桌面客户端的环境不一定与终端相同，推荐上述私有文件。普通模型请求仍会将任务相关上下文发送给配置的 API 服务。[官方协议说明](https://api-docs.deepseek.com/guides/anthropic_api/)

## 委派契约

| MCP 工具 | 用途 |
| --- | --- |
| `deepseek_health()` | 版本、模型、凭据状态 |
| `deepseek_submit(task, workspace, acceptance_criteria, timeout_seconds=600)` | 异步提交，立即返回 `job_id` |
| `deepseek_status(job_id)` | 查询持久状态 |
| `deepseek_result(job_id)` | 取回模型回复、改动证据和原始产物路径 |
| `deepseek_cancel(job_id)` | 取消并清理该任务的执行进程组 |

工作目录必须已存在且位于本项目内；拒绝把模型、依赖、凭据缓存或桥接状态目录作为任务工作目录。推荐在隔离工作目录或 worktree 开发。当前最多一个活跃任务，超时允许 1–3600 秒；每任务使用独立 runtime/home/session。MCP 服务重启后仍能查询、取回或取消原任务。

状态包括 `running`、`completed`、`error`、`timeout`、`cancelled`。`completed` 是 Harness 的正常结束信号；GPT 必须检查差异、工具事件中的命令/退出码、测试输出及未决问题，并自行决定验收是否通过。结果截断时完整证据仍留磁盘；快照遇到文件数/大小限制时明确标记不完整。

任务及原始证据保存于 `.dev/artifacts/deepseek-harness/jobs/<job_id>/`，包含请求、持久状态、脱敏事件、worker 日志、前后文件哈希/文本快照、统一 diff 和结果。`events.jsonl` 保存运行通知，`sdk-events.json` 保存 SDK 返回事件，供审核真实工具调用；模型总结本身不作为测试证据。

## 执行边界

完整 `sdk` profile 加载项目 AGENTS 与技能，保留官方文件和 shell 沙箱，指定 `workspace-write`，额外审批设为 `never`（拒绝扩权请求）。无法启用沙箱时失败并返回原因，不能自动切换到不受限模式。选择 `cwd` 和目录检查本身不提供安全隔离；实际限制由官方沙箱执行器提供，其规则以固定版本实现为准。

这是同账号的受信开发执行器：`workspace-write` 约束写入，并不隔离宿主可读文件或网络。以项目根作为任务目录会允许修改其中的开发工具，且同账号运行的代码可能读取私有缓存；提示词约束不提供强制隔离。小任务使用严格子目录，日常编码推荐独立 worktree；对不受信代码需要另行采用容器、独立账号或更严格的系统隔离。

worker 使用显式白名单环境，仅传入必要凭据和运行设置，不继承整个父进程的秘密环境。凭据不保存到任务请求、结果或补丁中；已知 key 从输出中脱敏。关闭独立的 DeepSeek 会话日志贡献和遥测，保留正常模型请求。生成的 SDK home/cache 包含私有运行资源，应保持在忽略目录内。

## 验证

CPU 测试使用明确标注的模拟 SDK，验证进程生命周期、取消/超时、异常、重启查询、凭据和证据边界，以及真实 MCP STDIO 协议；它不构成 DeepSeek 效果证据：

```bash
.dev/cache/deepseek-harness/venv/bin/python -m unittest discover -s .dev/tests/deepseek_harness -v
```

真实验证使用已注册的 MCP 服务和官方 DeepSeek API，产生少量 API 消耗：

```bash
.dev/cache/deepseek-harness/venv/bin/python .dev/tools/deepseek_harness/smoke.py
```

它创建有缺陷的均值函数，先确认测试失败，随后经 MCP 委派 DeepSeek 修复并运行测试，取回结果后独立重跑公开测试和额外边界用例。验证输入、输出和独立验收记录保存在 `.dev/artifacts/deepseek-harness-smoke/<run_id>/evidence.json`。执行失败保留错误记录，不能报告为已通过。

官方来源：[Harness](https://github.com/deepseek-ai/deepseek-harness)、[SDK](https://github.com/deepseek-ai/deepseek-harness/blob/master/python/sdk/README.md)、[PyPI 发布](https://pypi.org/project/deepseek-harness-sdk/0.2.1rc2/)、[Codex MCP](https://learn.chatgpt.com/docs/extend/mcp?surface=cli)。
