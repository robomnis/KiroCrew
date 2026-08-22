"""WeCom command parsing.

The vocabulary is DATA — :data:`COMMAND_SPEC` — and both the matcher and the
`/help` card read it, so a command cannot be added without appearing in help.

Per-conversation generation + awaiting-compact state lives in the shared
``messaging.conversation.ConversationState`` (re-exported here so existing
callers importing it from this module keep working).
"""

from __future__ import annotations

from kiro_crew.messaging.commands import CommandSpec, build_help_text, match_command
from kiro_crew.messaging.conversation import ConversationState  # noqa: F401

#: WeCom's command vocabulary. `新对话` / `清空` are the native spellings a WeCom
#: user reaches for; `/cancel` is the alias Telegram and Discord also accept for
#: `/stop`. `/link` and `/unlink` are listed because the dispatcher answers them
#: with an explanation — a command that exists only to refuse is still a command
#: the user should be able to discover, rather than one that reads as a typo.
COMMAND_SPEC: tuple[CommandSpec, ...] = (
    CommandSpec("new", "开始新对话", aliases=("新对话", "清空")),
    CommandSpec("compact", "压缩上下文，腾出空间"),
    CommandSpec("stop", "停止当前回复", aliases=("/cancel", "停止")),
    CommandSpec("help", "显示命令列表", aliases=("帮助",)),
)

#: `/link` and `/unlink` parse and are ANSWERED with an explanation, but they are
#: not rows in the help card: spending a third of it on two commands that only
#: refuse reads as capability the channel does not have, and an `/unlink` for a
#: link that can never exist is worse than absent. The footer says it once.
_REFUSED_COMMANDS: tuple[CommandSpec, ...] = (
    CommandSpec("link", "从 dashboard 推送回复（本渠道不支持）"),
    CommandSpec("unlink", "停止从 dashboard 推送回复（本渠道不支持）"),
)

_HELP_HEADER = "🦞 Kiro Crew — 企业微信"
_HELP_FOOTER = (
    "直接发消息即可对话，回复会实时流式返回。\n"
    "较长的回复会分成多条消息；图片和文件暂不支持接收。\n"
    "本渠道回复绑定在收到的消息上，因此 /link 与 /unlink 不可用。"
)


def build_help() -> str:
    """The `/help` card, rendered from :data:`COMMAND_SPEC`."""
    return build_help_text(_HELP_HEADER, COMMAND_SPEC, _HELP_FOOTER)


def _after_leading_mention(text: str) -> str | None:
    """Return the text following ONE leading ``@name`` token, else ``None``.

    Addressing the bot is mandatory in a WeCom group, so the platform delivers
    the command as ``@Kiro /new``. Unlike Slack's ``<@BOTID>``, the mention
    arrives as plain text with no delimiter and no ``is_mention`` flag, and the
    bot's display name never reaches this module — so it is recognized purely
    structurally: a leading ``@`` run of non-whitespace, then whitespace, then
    the remainder. Exactly one token is consumed and the remainder is not
    otherwise touched, which keeps ``@a @b /new`` and ``@Kiro please /new`` out.
    """
    if not text.startswith("@"):
        return None
    parts = text.split(None, 1)
    if len(parts) != 2:
        return None
    return parts[1].strip()


def parse_command(text: str) -> str | None:
    """Return the command name *text* invokes, else ``None``."""
    stripped = (text or "").strip()
    cmd = match_command(stripped, COMMAND_SPEC + _REFUSED_COMMANDS)
    if cmd is not None:
        return cmd
    # Retry once past a group mention. Only the command CANDIDATE is normalized:
    # the message itself is never rewritten, so mentioned prose still reaches the
    # model verbatim, and the alias match stays exact so only a bare command
    # behind the mention is intercepted.
    candidate = _after_leading_mention(stripped)
    if candidate is None:
        return None
    return match_command(candidate, COMMAND_SPEC + _REFUSED_COMMANDS)
