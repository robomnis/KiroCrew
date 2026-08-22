# WeCom Integration

Talk to your Kiro Crew agent from WeCom — through a WeCom (企业微信) AI bot. Create
the bot in your WeCom console, drop in two values, and you're chatting. Replies
stream back live.

> **WeChat vs. WeCom.** Kiro Crew connects through **WeCom (企业微信)**, the work
> edition, using its AI-bot API. It does **not** sign in to a personal WeChat
> account — people message the bot from inside WeCom.

Like Telegram, the connection is outbound-only: Kiro Crew opens a secure
WebSocket to WeCom, so there's no callback URL or open port to manage.

## The easy way: just ask Kiro Crew

You don't have to edit anything by hand. In any Kiro Crew session — the
dashboard, Slack, or the CLI — say something like *"set up the WeCom channel."*
Kiro Crew tells you where to create the WeCom AI bot, then writes your Bot ID and
Secret into `~/.kiro/crew/.env` and `config.json` and restarts the gateway for
you. You just paste the two values when it asks.

Prefer to wire it up yourself? The manual steps are below.

## Quick start

You'll need a running gateway (`kirocrew gateway`) and admin access to your
WeCom console.

1. **Create an AI bot** — in the WeCom admin console, open **应用管理 → AI 智能体**
   and create a bot. Its settings page shows a **Bot ID** and a **Secret**.
2. **Note the userids** — every WeCom member has a `userid` (账号). Collect the
   ones you want to let in.
3. **Save the credentials** to `~/.kiro/crew/.env`:
   ```
   WECOM_BOT_ID=your-bot-id
   WECOM_SECRET=your-bot-secret
   ```
4. **Turn it on** in `~/.kiro/crew/config.json`:
   ```json
   "wecom": {
     "enabled": true,
     "allowed_users": [{ "userid": "zhangsan", "name": "Zhang San" }]
   }
   ```
5. **Restart, then say hi:**
   ```bash
   kirocrew restart
   ```

Message the bot in WeCom and it answers. If it stays quiet, look for
`WeCom WS connected and subscribed` in the gateway log and confirm your userid
is allowed.

Those two values — **Bot ID** and **Secret** — are all Kiro Crew needs. There's
no corp ID, agent ID, callback URL, or AES key to wire up. Good to know: the
WeCom bot replies to messages you send it — it can't start a conversation on its
own, and it has no buttons, so when the agent offers you choices they arrive as a
numbered list you answer by typing one.

A long answer arrives as several messages rather than being cut off, and images
or files you send are not read yet — the bot says so instead of going quiet.

## Who can reach it

> **Kiro Crew runs on your machine, with your files and credentials.** So it only
> talks to the owner and the userids you name.

- Authorized senders: the **owner** (`KIROCREW_OWNER_ID`) plus anyone listed in
  `allowed_users`. With no owner and an empty list, nobody gets in.
- Whole-company access: set `"allow_all_users": true` (or flip **Allow all
  organization members** in Settings → WeCom) to skip listing each userid.
  This is an explicit opt-in — an empty list never means "everyone" — and it
  works because a WeCom AI bot is only reachable inside your own org tenant.
  Messages without a userid are still dropped.
- The WeCom AI bot carries direct messages by default — one conversation per
  userid. **A message from a group is refused unless you set
  `allow_group_chats`** (see below); the refusal is recorded in the audit log.
- Anyone else is quietly dropped and recorded in the audit log.

## Commands

- `/new` (or `新对话`, `清空`) — start a fresh conversation
- `/compact` — free up room when the context fills
- `/stop` (or `/cancel`, `停止`) — stop the reply that's currently generating
- `/help` (or `帮助`) — list the commands

In a group chat, where addressing the bot is required, send the command on its
own after the mention — `@Kiro /new`. Anything else after the mention is treated
as an ordinary message.

## Settings & reference

Everything lives in the `wecom` section of `config.json`:

| Setting | Default | What it does |
|---|---|---|
| `enabled` | `false` | Turns the channel on |
| `allowed_users` | `[]` | `{ "userid", "name" }` entries allowed to chat (empty = owner only) |
| `allow_group_chats` | `false` | Answer in group conversations, not just DMs — see below |
| `allowed_chat_ids` | `[]` | Restrict group answers to these chatids (empty = any group, once enabled) |
| `soft_threshold_pct` | `80` | Context % where the bot suggests `/compact` |
| `hard_threshold_pct` | `95` | Context % hard cutoff |
| `ws_url` | `wss://openws.work.weixin.qq.com` | WeCom AI-bot endpoint |

These two are set in `config.json` rather than in Settings → WeCom, which covers
the credentials and the user allow-list.

### Group chats

Off by default, and deliberately a separate switch from the user allow-list —
those answer different questions. The allow-list says **who may drive a turn**;
a group says **who may read the result**. Every member of a WeCom group sees the
agent's tool output and anything it quotes from your files, so turning groups on
is a disclosure decision, not a convenience one. `allow_all_users` does not open
groups.

With it on:

- Each group is **its own conversation**, keyed by its chatid. `/new` in a group
  starts the group a fresh session and leaves your private one alone, and your DM
  history never answers in the group.
- Sending still requires an allow-listed sender, so an unlisted colleague in the
  same group is ignored exactly as they are in a DM.
- Name the groups in `allowed_chat_ids` if you want a tighter boundary than
  "every group the bot is in". Refused groups are recorded in the audit log, so
  silence in a group is diagnosable.

Credentials go in `~/.kiro/crew/.env`: `WECOM_BOT_ID` and `WECOM_SECRET`.

**If something's off:** no reply usually means the sender's userid isn't allowed
or `enabled` is `false`; a missing `channel started` line means a credential is
unset; if it connects and then goes quiet, the bot may have been removed from
the WeCom console — re-add it and restart. If you send a picture or a file and
get "暂不支持接收该类型的消息", that's expected: the bot reads text only, so
describe what you need instead. If a group the bot **used to** answer has gone
quiet after an upgrade, that is `allow_group_chats` defaulting to off — grep the
audit log for `denied_group_chat` to confirm, then turn it on deliberately.

## Related docs

- [Slack Integration](slack-integration.md)
- [Telegram Integration](telegram-integration.md)
- [Getting Started](getting-started.md)
