# JevGO Codex 邮件通知

这是开发辅助工具，只使用 Python 标准库。它连接运行 Codex 的主机上已有的 Unix WebSocket，监听 JevGO 项目的用户对话，通过已连接的 Gmail 工具发邮件。不创建对话、不运行模型、不回答确认或填写请求。

## 开关

安装后的命令：

```sh
~/.local/bin/jevgo-mail-notify on
~/.local/bin/jevgo-mail-notify off
~/.local/bin/jevgo-mail-notify status
```

`on` 开启三类通知：权限/操作确认、选择/填写（含异步提问）、每轮结束（正常完成、失败、中断）。范围由项目 ID 和项目目录共同识别，包含未归入侧栏项目、但位于该目录的聊天；自动生成的子代理不单独通知。这里的“对话结束”指一轮执行终止，不是关闭窗口或归档聊天。

同一请求和同一轮结束各通知一次。首次开启和再次开启不补发关闭期间已经结束的轮次。关闭会取消尚未发出的通知并退订本监听器的对话订阅；已提交给 Gmail 的邮件无法撤回。开关在下一次后台检查时生效，通常不到一秒。

`status` 中 `enabled` 是开关；`daemon.connected` 是 Codex 连接；`last_scan` 是最近扫描时间。`unknown` 表示邮件提交后的结果不确定，工具不会自动重发；`failed` 表示明确失败。两者都需要人工检查，不能把开关开启理解成 Gmail 始终可达。

## 运行条件

Codex 后台必须运行，Gmail 必须保持连接。默认复用设置时指定的聊天；它不可用时，会尝试同项目已加载且连接 Gmail 的聊天。程序不复制 OAuth 凭据，也不要求邮箱密码。关闭 Codex 后台或断开 Gmail 时，通知可能延迟；待发送队列会保留到通道恢复，关闭开关会清空待发队列。

实时事件通过 `thread/resume` 加入已经加载的对话；它不开始新一轮、不覆盖对话设置。每 5 秒扫描新对话和轮次记录，作为断线和短轮次的补漏。新对话首次订阅前短暂出现、并已经处理完的输入请求可能无法回溯；普通文本问题不属于独立输入事件，其所在轮次结束时仍有结束通知。

邮件只含项目、对话标题、事件类型、通用事项说明、时间和对话 ID，不转发完整命令、表单内容或对话正文。通知会产生发往配置收件人的 Gmail 邮件。

## 部署和维护

在实际运行 Codex app-server 的 Linux 主机，以相同用户部署。将配置写到本机私有文件，字段参照 `config.example.json`，其中 `mail_thread_id` 应是已连接 Gmail 的真实聊天。工具名必须来自实际 `mcpServerStatus/list` 查询。

```sh
python3 install.py --config /path/to/private-config.json
~/.local/bin/jevgo-mail-notify probe
~/.local/bin/jevgo-mail-notify test  # 发送一封明确标注的测试邮件
~/.local/bin/jevgo-mail-notify on
```

安装路径：`~/.local/share/jevgo-codex-mail/`。私有配置、SQLite 去重队列、运行状态在 `~/.local/state/jevgo-codex-mail/`。用户级服务是 `jevgo-codex-mail.service`，日志可用 `journalctl --user -u jevgo-codex-mail.service` 查看。服务自动重启；主机重启和退出登录后的运行取决于用户 systemd/linger 配置以及 Codex 后台是否可用。

完全停止程序：`systemctl --user disable --now jevgo-codex-mail.service`。这与 `off` 不同：`off` 保留后台进程，方便立即重新开启。

## 验证

从项目根运行：

```sh
PYTHONPATH=.dev/tools/codex_notifications /usr/bin/python3 -m unittest discover -s .dev/tests/codex_notifications -v
```

测试使用临时数据库和模拟 Unix WebSocket，不调用真实 Gmail。`test` 命令才会真实发送测试邮件。

实现按本机 Codex 0.153.2 生成的协议验证。升级 Codex 后若协议变化，先运行 `probe` 检查连接，不能通过重启重复发送状态不确定的邮件。

参考：[Codex App Server](https://learn.chatgpt.com/docs/app-server)。
