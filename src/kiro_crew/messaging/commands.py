"""Channel command vocabulary: one spec table, matching and help from it.

Every channel's command set is the same three concerns — which words invoke a
command, what to print when the user asks, and keeping those two in agreement.
Telegram grew a `COMMAND_SPEC` plus a `build_help_text` for exactly this, Discord
hardcodes its help card as a literal, and WeCom, Weixin, Teams, Webex and
iMessage had no help at all. This module is the shared half so a sixth copy is
not written.

What stays per channel is the DATA: the vocabularies genuinely differ (WeCom
answers `/link` with a refusal, Weixin has no `/link` at all; the Chinese
channels carry native aliases such as `新对话`). What is shared is the matcher and
the renderer, which is why a command cannot be added without appearing in help:
both read the same tuple.

Stdlib-only and pure, like the rest of the Layer-2 helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["CommandSpec", "match_command", "build_help_text"]


@dataclass(frozen=True)
class CommandSpec:
    """One command: its canonical name, what it does, and how it is spelled.

    ``name`` is the value a dispatcher switches on; the displayed form adds the
    ``/``. ``aliases`` are EXTRA spellings beyond ``f"/{name}"``, which is always
    accepted: native-language words (`新对话`), and short forms (`cancel` for
    `stop`).
    """

    name: str
    help_text: str
    aliases: tuple[str, ...] = field(default=())


def match_command(text: str, specs: tuple[CommandSpec, ...]) -> str | None:
    """Return the ``name`` of the command *text* invokes, else ``None``.

    Matching is EXACT against the whole stripped message, so a message that
    merely begins with a command word stays ordinary prose the model answers.
    ``f"/{name}"`` is matched case-insensitively (a phone keyboard capitalises),
    while a non-ASCII alias is compared as written because casing is meaningless
    for it and ``str.lower()`` on some scripts is not a no-op.

    The ``/`` is hardcoded because both adopters use it. Discord prefixes with
    ``!`` (its client swallows a bare ``/``) but does not consume this module; the
    day it migrates is the day a prefix parameter earns its place.
    """
    stripped = (text or "").strip()
    if not stripped:
        return None
    lowered = stripped.lower()
    for spec in specs:
        if lowered == f"/{spec.name}":
            return spec.name
        for alias in spec.aliases:
            if stripped == alias or lowered == alias.lower():
                return spec.name
    return None


def build_help_text(header: str, specs: tuple[CommandSpec, ...], footer: str = "") -> str:
    """Render a help card from *specs*.

    Aliases are shown beside their command, because a user who was taught `新对话`
    needs to see it is the same thing as `/new` rather than discovering two
    commands. The caller puts anything else it needs to explain into *footer*.
    """
    lines = [header, ""]
    for spec in specs:
        spelled = f"/{spec.name}"
        if spec.aliases:
            spelled += " (" + " / ".join(spec.aliases) + ")"
        lines.append(f"{spelled} — {spec.help_text}")
    if footer:
        lines += ["", footer]
    return "\n".join(lines)
