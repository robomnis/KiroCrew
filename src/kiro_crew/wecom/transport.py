"""Layer 1 -- WeCom (企业微信) as a concrete ``MessagingTransport``.

Wraps the low-level :class:`WeComClient` (WebSocket long-connection + WS
streaming reply / one-shot ``response_url`` fallback) in the channel-neutral
transport contract, so the WeCom channel rides the shared ``TurnDriver``
(credential/exfil redaction + tool-approval ladder + SEL audit) instead of a
hand-rolled turn loop.

Dependency direction is ``wecom -> messaging`` (allowed); the neutral
``messaging`` package never imports ``wecom``.

WeCom differs from Slack/Telegram in three ways, all absorbed INSIDE this
transport (the neutral layers are untouched):

* Persistent outbound WebSocket (``connect``/``disconnect`` lifecycle) -- not
  push (Slack) or long-poll (Telegram).
* The turn dispatch runs as a background task (started by the client) so the
  single WS keeps sending ACK/pong during a long streaming turn -- an inline
  await would starve the ping loop and trip a false disconnect.
* No proactive send: a reply is bound to the inbound message's WS ``req_id``
  (or its one-shot ``response_url``), so ``supports_proactive_send`` is False.

Security: :meth:`authorize` is **deny-by-default** and owner-only. An empty
allow-list authorizes nobody (fail closed), never everybody. The only path to
"everybody" is the explicit ``allow_all`` opt-in (config
``wecom.allow_all_users``), which still denies frames without a userid.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Iterable
from typing import Any

from kiro_crew.messaging.dispatch import inbound_permitted
from kiro_crew.messaging.transport import (
    ConfiguredChannelTarget,
    InboundMessage,
    MessagingTransport,
    TransportCapabilities,
)
from kiro_crew.sel import sel
from kiro_crew.wecom.client import WECOM_MAX_REPLY_BYTES, WeComClient, WeComInbound

logger = logging.getLogger(__name__)

# A dispatch callback consumes an authorized WeCom inbound (carrying the WS
# routing keys ``req_id`` / ``response_url`` that the neutral InboundMessage
# cannot hold) and drives a turn. The gateway supplies the real implementation.
DispatchFn = Callable[["WeComInbound"], Awaitable[None]]

# ``max_message_chars`` is a CHARACTER budget while WeCom's cap is in BYTES, so
# declare the worst case — 4 bytes per character — rather than the byte number
# itself. Mirrors ``webex/transport.py``'s WEBEX_SAFE_MESSAGE_CHARS, and for the
# same reason: declaring the raw byte limit as a char count is what let a CJK
# answer pass the shared splitter and then be refused on the wire.
WECOM_SAFE_MESSAGE_CHARS = WECOM_MAX_REPLY_BYTES // 4

# WeCom AI-bot capabilities: WS streaming (each frame REPLACES the bubble ->
# edit=True), a byte-safe char cap, no tappable chips (max_buttons=0, so the
# renderer routes [OPTIONS:] through the shared numbered-text fallback), and NO
# proactive send (a reply is bound to the inbound message's req_id / one-shot
# response_url).
WECOM_CAPABILITIES = TransportCapabilities(
    streaming=True,
    edit=True,
    reactions=False,
    files_inbound=False,
    files_outbound=False,
    rich_blocks=False,
    threads=False,
    max_message_chars=WECOM_SAFE_MESSAGE_CHARS,
    max_buttons=0,
    supports_proactive_send=False,
)

# Reply to an inbound frame carrying media but no text. There is no download path
# (files_inbound=False), so refuse and name the kind: a silent drop leaves the
# sender believing the send worked.
_UNSUPPORTED_MEDIA_REPLY = "ℹ️ 暂不支持接收该类型的消息（{kind}），请改用文字描述。"

# The media kinds a refusal names. ``msgtype`` arrives from the wire, so only a
# value IN this map is answered and only its label is ever rendered -- an
# unrecognized kind is dropped silently rather than echoed back or guessed at.
_MEDIA_KIND_LABELS = {
    "image": "图片",
    "voice": "语音",
    "video": "视频",
    "file": "文件",
    # The vendor marks image/voice/video/file as single-chat only, so `mixed`
    # (图文混排) is the ONE non-text kind a GROUP can send. Omitting it left the
    # only group-reachable attachment on the silent-drop branch -- the defect this
    # refusal exists to remove, reintroduced for the case groups actually hit.
    "mixed": "图文",
}


class WeComTransport(MessagingTransport):
    """Concrete WeCom transport over the low-level ``WeComClient`` (WebSocket)."""

    channel_type = "wecom"

    def __init__(
        self,
        client: WeComClient,
        *,
        allowed_users: Iterable[str] = (),
        allow_all: bool = False,
        owner_id: str = "",
        allow_group_chats: bool = False,
        allowed_chat_ids: Iterable[str] = (),
        dispatch: DispatchFn | None = None,
    ) -> None:
        self._client = client
        # A group is a broader disclosure boundary than a DM: every member reads
        # the agent's tool output, so it is its OWN opt-in and is never inferred
        # from the user allow-list or from allow_all_users. Frozen at construction
        # like the allow-list, so a config edit cannot widen a decision in flight.
        self._allow_group_chats = bool(allow_group_chats)
        self._allowed_chats: frozenset[str] = frozenset(c for c in allowed_chat_ids if c)
        # Deny-by-default: freeze the allow-list so it can't mutate under an
        # in-flight decision. owner_id (when set) is always allowed.
        self._allowed: frozenset[str] = frozenset(u for u in allowed_users if u)
        # Explicit opt-in only (config wecom.allow_all_users): every org
        # member may DM the bot. This is a deliberate toggle, NEVER inferred
        # from an empty allow-list — an empty list stays fail-closed. A WeCom
        # AI bot is reachable only inside the org tenant, which is what makes
        # this a reasonable (if broad) grant.
        self._allow_all = bool(allow_all)
        self._owner_id = owner_id
        self._dispatch = dispatch
        self.capabilities = WECOM_CAPABILITIES

    @property
    def client(self) -> WeComClient:
        """The underlying WeCom WS client (held + exposed, not hidden)."""
        return self._client

    # -- Tier-1 core --------------------------------------------------------
    async def send_message(
        self, conversation_id: str, content: str, thread_id: str | None = None
    ) -> str:
        # WeCom has no stable conversation-addressed send: a reply is bound to
        # an inbound req_id (WS stream) or its one-shot response_url. Streaming
        # goes through the renderer; here conversation_id is treated as a
        # response_url fallback target for out-of-band one-shot sends.
        if conversation_id:
            await self._client.send_reply(conversation_id, content)
        return ""

    async def resolve_conversation(self, user_id: str) -> str:
        # No addressable DM channel id; the userid is the logical conversation.
        return user_id

    async def fetch_history(
        self, conversation_id: str, thread_id: str | None = None
    ) -> list[InboundMessage]:
        # WeCom AI-bot cannot page DM history; sessions persist via
        # conversation_log instead.
        return []

    def configured_targets(self) -> list[ConfiguredChannelTarget]:
        identities = set(self._allowed)
        if self._owner_id:
            identities.add(self._owner_id)
        targets = [
            ConfiguredChannelTarget(
                f"user:{user_id}",
                f"WeCom DM · {user_id}",
                available=False,
                unavailable_reason="WeCom only allows replies to an inbound message",
            )
            for user_id in sorted(identities)
        ]
        if self._allow_all and not targets:
            targets.append(
                ConfiguredChannelTarget(
                    "policy:all",
                    "WeCom · organization users",
                    available=False,
                    unavailable_reason="WeCom only allows replies to an inbound message",
                )
            )
        return targets

    # -- Lifecycle ----------------------------------------------------------
    async def connect(self) -> None:
        await self._client.start()  # launches the WS connect/serve background loop

    async def disconnect(self) -> None:
        await self._client.close()

    # -- Inbound adapter ----------------------------------------------------
    def authorize(self, msg: InboundMessage) -> bool:
        """Owner-only, deny-by-default. Empty allow-list authorizes nobody.

        ``allow_all`` (explicit config opt-in) authorizes any inbound with a
        non-empty userid — a missing/empty userid is denied even then, so an
        anonymous or malformed frame never reaches dispatch.
        """
        uid = msg.user_id
        allowed = bool(uid) and (
            self._allow_all
            or (bool(self._owner_id) and uid == self._owner_id)
            or uid in self._allowed
        )
        if not allowed:
            # Audit ALL denials (including empty/missing userid) via the helper
            # that fills event_id + timestamp, so the row survives SEL prune (a
            # raw log() with an empty timestamp sorts before the cutoff and is
            # dropped on the next prune).
            sel().log_api_access(
                caller=uid or "unknown",
                operation="wecom_transport.authorize",
                outcome="denied",
                source="wecom",
            )
        return allowed

    async def receive(self, raw_envelope: Any) -> None:
        """Normalize -> authorize -> dispatch.

        The low-level client parses inbound WS frames into ``WeComInbound`` and
        invokes this as its ``on_message`` (already wrapped in a background task
        so the WS receive loop keeps breathing during a long turn). We map onto
        the neutral ``InboundMessage`` for the deny-by-default authorize
        contract, then hand the richer ``WeComInbound`` (carrying ``req_id`` /
        ``response_url``) to the turn dispatcher.

        Authorization runs BEFORE any reply, including the unsupported-media
        refusal below, so an unallowlisted sender still learns nothing about what
        they reached.
        """
        if not isinstance(raw_envelope, WeComInbound):
            return
        inbound = raw_envelope
        msg = InboundMessage(
            channel_type="wecom",
            user_id=inbound.userid,
            # A group conversation IS the conversation, so it addresses the
            # session; a DM addresses the peer. thread_id stays None either way --
            # WeCom has no thread concept, and modelling a group as a thread would
            # promise the per-topic isolation the forum-shaped paths provide.
            conversation_id=inbound.chatid or inbound.userid,
            text=inbound.text,
            thread_id=None,
        )
        if not self.authorize(msg):
            return
        if inbound.chatid and not self._group_permitted(inbound.chatid):
            return
        if not inbound.text:
            # Emptiness alone is not a reason to drop: a WeCom image or file
            # arrives as a frame with no text, and dropping it tells the sender
            # nothing while the agent is never told anything arrived.
            #
            # Refuse only a RECOGNIZED media kind. Answering every non-"text"
            # msgtype would send an unsolicited reply for a system or event frame
            # the user never composed, and would audit it as an unsupported
            # attachment -- so an unknown kind stays a silent drop with a log line.
            if inbound.msgtype in _MEDIA_KIND_LABELS:
                await self._refuse_media(inbound)
            else:
                logger.info(
                    "WeCom: dropped a text-less frame from %s (kind not recognized)",
                    inbound.userid,
                )
            return
        if self._dispatch is not None:
            await self._dispatch(inbound)

    def _group_permitted(self, chatid: str) -> bool:
        """Whether the agent may answer in the group *chatid*. Deny by default.

        Group answers are their own opt-in (``wecom.allow_group_chats``) because
        the sender allow-list answers a different question: it says WHO may drive a
        turn, not who may READ the result. In a group every member reads the
        agent's tool output and any file contents it quotes, so inferring group
        access from a user grant would widen disclosure without the operator ever
        choosing it. ``allowed_chat_ids`` narrows further when set.

        Denials are SEL-audited: an operator who added the bot to a group and got
        silence needs the audit log to show that the group was refused rather than
        that the message never arrived.
        """
        if not self._allow_group_chats or (
            self._allowed_chats and chatid not in self._allowed_chats
        ):
            sel().log_api_access(
                caller=chatid,
                operation="wecom_transport.receive",
                outcome="denied_group_chat",
                source="wecom",
                resources=f"chatid={chatid}",
            )
            return False
        return True

    async def _refuse_media(self, inbound: WeComInbound) -> None:
        """Tell an authorized sender their attachment was not read.

        Gated on the per-message channels-governance policy FIRST. The text path
        reaches that gate inside the dispatcher, but this reply is produced here in
        the transport and never gets there — so without this check a host profile
        that denies WeCom after startup would still see the bot talk, which is the
        fail-open shape the governance ladder exists to prevent. Silent on deny,
        matching every other governance drop.

        ``msgtype`` is externally-derived, so the caller has already matched it
        against ``_MEDIA_KIND_LABELS``; only the mapped label reaches the reply or
        the audit row, never the wire value.

        Delivered over the WS bubble when a ``req_id`` is in hand, because that is
        the channel's reliable path -- ``response_url`` is absent on some frames,
        and falling straight to it would drop the refusal and reproduce the silent
        loss this exists to prevent.
        """
        if not await inbound_permitted("wecom"):
            return
        kind = _MEDIA_KIND_LABELS[inbound.msgtype]
        sel().log_api_access(
            caller=inbound.userid or "unknown",
            operation="wecom_transport.receive",
            outcome="media_unsupported",
            source="wecom",
            resources=f"kind={kind}",
        )
        body = _UNSUPPORTED_MEDIA_REPLY.format(kind=kind)
        if inbound.req_id and await self._client.send_bubble(inbound.req_id, body):
            return
        await self._client.send_reply(inbound.response_url, body)
