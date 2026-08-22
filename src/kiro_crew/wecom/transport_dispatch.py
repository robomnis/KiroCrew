"""Full new-path dispatch: WeComTransport -> TurnDriver -> WeComRenderer.

``WeComTransport.receive()`` authorizes + normalizes an inbound WS frame and
hands the ``WeComInbound`` (carrying the WS routing keys ``req_id`` /
``response_url``) to :meth:`WeComDispatcher.handle_message`, which mirrors the
Slack/Telegram transport dispatch:

    command intercept (/new, /compact)
    -> construct WeComRenderer + on_turn_start (immediate "🤔 …" placeholder)
    -> session acquire -> context build
    -> TurnDriver.run(provider, renderer)   # shared redaction + approval ladder
    -> post-turn (record_success, persist, soft/hard threshold notice)  # guarded
    -> renderer.close() + session release   # in finally

WeCom has no interactive buttons, so the dispatcher runs the driver
``decider``-less (deny-by-default for INTERACTIVE mode; ``auto``/``trust``
still work) and has no callback handler. The security ``tool_gate`` and the
``spawn_run`` auto-approve are wired inline off ``ctx_builder.hooks``
(channel-neutral) so this module never imports ``kiro_crew.slack``.

Dependency direction is ``wecom -> messaging`` (allowed).
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from kiro_crew.messaging.dispatch import (
    ChannelTurn,
    build_directive_consumer,
    drive_turn,
    inbound_permitted,
)
from kiro_crew.messaging.driver import APPROVAL_INTERACTIVE
from kiro_crew.messaging.link import (
    CHAT_TYPE_DIRECT,
    CHAT_TYPE_GROUP,
    build_dm_session_key,
    seed_generation,
)
from kiro_crew.wecom.commands import ConversationState, build_help, parse_command
from kiro_crew.wecom.renderer import WeComRenderer
from kiro_crew.wecom.transport import WECOM_CAPABILITIES

if TYPE_CHECKING:
    from kiro_crew.config.loader import KiroCrewConfig
    from kiro_crew.context import ContextBuilder
    from kiro_crew.history import ConversationLog
    from kiro_crew.session import SessionManager
    from kiro_crew.wecom.client import WeComClient, WeComInbound

logger = logging.getLogger(__name__)

# Canonical kiro-cli agent fallback so WeCom sessions load kirocrew-core
# (spawn_run etc.) instead of kiro-cli's bare built-in default when neither an
# explicit override nor agent.default_agent is configured. Mirrors the Slack /
# Telegram paths' _DEFAULT_KIROCREW_AGENT.
_DEFAULT_KIROCREW_AGENT = "kirocrew"

# ── Bot-facing strings, owned here rather than inline at each send ──
# One block so the channel's whole voice is visible at once and a wording change
# is one edit. These are CHINESE because WeCom is a Chinese-language platform and
# its users are addressed in it; backend-owned strings have no catalog path yet
# (AGENTS.md), so the owning module is the unit of ownership.
_NEW_SESSION = "✅ 已开始新对话"
_LINK_UNAVAILABLE = "ℹ️ 本渠道回复绑定在收到的消息上，不支持从 dashboard 主动推送，/link 暂不可用。"
# Three states, not two. The cancel is COOPERATIVE: the ack goes out after only
# writing session/cancel, so the turn stops at its next safe point and the bubble
# may still move for a moment -- "正在停止" is what the user will actually observe.
# And a busy session whose cancel FAILED must not be told nothing was running: that
# is the case /stop exists for, and cancel failures cluster on exactly those turns.
_STOPPING = "⏹ 正在停止当前回复…"
_STOP_FAILED = "⚠️ 停止失败，请重试。"
_NOTHING_RUNNING = "ℹ️ 当前没有正在生成的回复。"
_STEER_MERGED = "⏳ 已合并到当前回复"
_STEER_UNAVAILABLE = "⏳ 正在处理上一条，请稍后重发"
_AUTO_COMPACTED = "🗜️ 上下文接近上限，已自动压缩。"
_SOFT_THRESHOLD = "⚠️ 对话上下文已较长，回复 /compact 压缩，或 /new 开始新对话。"
_COMPACT_BUSY = "⏳ 正在处理上一条消息，请稍后再试 /compact。"
_COMPACT_NOTHING = "ℹ️ 当前没有可压缩的对话。"
_COMPACT_DONE = "🗜️ 已压缩上下文。"
_COMPACT_FAILED = "⚠️ 压缩失败，请重试。"


class WeComDispatcher:
    """Coordinates WeCom turns onto the shared ``TurnDriver``.

    One instance per gateway lifetime. Holds the per-userid conversation state
    (generation counter + soft-threshold flag). ``handle_message`` is wired as
    the transport's dispatch callback. ``client`` is set by the gateway after
    construction.
    """

    def __init__(
        self,
        *,
        sessions: "SessionManager",
        ctx_builder: "ContextBuilder",
        cfg: "KiroCrewConfig",
        owner_id: str = "",
        agent: str | None = None,
        conv_log: "ConversationLog | None" = None,
        approval_mode: str = APPROVAL_INTERACTIVE,
    ) -> None:
        self.sessions = sessions
        self.ctx_builder = ctx_builder
        self.cfg = cfg
        self.owner_id = owner_id
        self.agent = agent
        self.conv_log = conv_log
        self.approval_mode = approval_mode
        self.client: "WeComClient | None" = None
        self._conv = ConversationState(seed_fn=self._seed_gen)

    # ── Turn dispatch (transport's dispatch callback) ──────────────────────

    async def handle_message(self, inbound: "WeComInbound") -> None:
        """Drive one authorized inbound WeCom message through TurnDriver."""
        assert self.client is not None, "WeComDispatcher.client must be set"
        # Inbound channels-governance gate (off-loop) — recheck per message so a
        # host-profile deny added after connect stops dispatch without a restart
        # (the startup gate only blocks CONNECTING). Silently drop on deny.
        if not await inbound_permitted("wecom"):
            return
        userid = inbound.userid
        text = inbound.text
        route = self._route(inbound)
        logger.info("WeCom inbound from %s in %s: %d chars", userid, route, len(text or ""))

        # ── Command intercept (no LLM session needed) ──
        cmd = parse_command(text)
        if cmd == "new":
            self._conv.bump_gen(route)
            await self._ack(inbound, _NEW_SESSION)
            return
        if cmd == "compact":
            self._conv.clear_awaiting(route)
            await self._handle_compact(inbound)
            return
        if cmd in ("link", "unlink"):
            # WeCom replies are bound to an inbound request (no proactive send),
            # so a dashboard->channel mirror can never be delivered here. Reject
            # explicitly rather than silently record an inert link.
            await self._ack(inbound, _LINK_UNAVAILABLE)
            return
        if cmd == "help":
            await self._ack(inbound, build_help())
            return
        if cmd == "stop":
            # BEFORE the busy check below: /stop is the one message that must not
            # be folded into the running turn it exists to abort.
            await self._handle_stop(inbound)
            return

        # ── Mid-turn concurrency: check the CURRENT-generation key for an
        # in-flight turn BEFORE any idle/daily rotation (rotating first could
        # mint a new key and miss the running turn). WeCom replies are bound to
        # the inbound request, so a queued-then-drained reply can't be delivered
        # reliably later -- fold it into the running turn via steer.
        session_key = self._session_key(route)
        if self.sessions.is_busy(session_key):
            await self._handle_busy(inbound, session_key)
            return

        self._conv.maybe_rotate(
            route,
            time.time(),
            idle_minutes=self.cfg.messaging.idle_reset_minutes,
            daily_reset_hour=self.cfg.messaging.daily_reset_hour,
        )
        session_key = self._session_key(route)
        # The conversation the session belongs to, for attribution/UI: the group
        # when there is one, else the peer.
        conversation_id = f"wecom:{self._split_route(route)[1]}"
        # Resolve the kiro-cli agent: an explicit override wins, else the
        # configured default, else the canonical "kirocrew" agent -- so the
        # session loads kirocrew-core (spawn_run) instead of kiro-cli's bare
        # built-in default. Mirrors slack/telegram transport_dispatch.
        agent = self._resolve_agent()

        # WeCom has no interactive buttons -> no decider (deny-by-default for
        # INTERACTIVE; auto/trust still auto-approve via the driver ladder).
        renderer = WeComRenderer(
            self.client,
            inbound.req_id,
            inbound.response_url,
            WECOM_CAPABILITIES,
            session_key=session_key,
        )

        # The turn skeleton (acquire -> identity -> context -> TurnDriver ->
        # guarded post-turn -> finally close/release) lives once in
        # messaging.dispatch. Only the wecom-specific pieces are injected.
        # Immediately surface a newly-created channel session in the dashboard
        # (feature: don't wait for the ~30s reconciler). Circular import —
        # dashboard boot imports channel packages — so import lazily.
        async def _surface_new_session() -> None:
            from kiro_crew.dashboard.channel_slots import surface_dispatcher_session

            await surface_dispatcher_session(self)

        await drive_turn(
            ChannelTurn(
                channel_type="wecom",
                session_key=session_key,
                # Session-directive consumer: monitor_start / autonudge_stop /
                # ... return a marker TurnDriver decodes; apply it against THIS
                # turn's session key (dashboard-only directives stay refused
                # for channel sessions).
                directive_consumer=build_directive_consumer(
                    session_key=session_key, sessions=self.sessions, dispatcher=self
                ),
                conversation_id=conversation_id,
                agent=agent,
                user_text=text,
                renderer=renderer,
                approval_mode=self.approval_mode,
                decider=None,  # WeCom can't render approve/deny buttons
                persist=lambda user_text, reply, is_new: self._persist_turn(
                    session_key, user_text, reply, is_new, agent
                ),
                notice=lambda sk, provider: self._maybe_notice(inbound, sk, provider),
                audit_caller=f"wecom:{userid}",
                after_persist=_surface_new_session,
            ),
            sessions=self.sessions,
            ctx_builder=self.ctx_builder,
        )

    async def _handle_stop(self, inbound: "WeComInbound") -> None:
        """Hard cancel: abort the in-flight turn for this conversation.

        Cooperative before it is fatal — ``cancel(wait_ack_timeout=0)`` writes the
        ACP ``session/cancel`` notification and returns, so the acknowledgement to
        the user is immediate and the turn stops at its next safe point. Waiting
        for the ack here would hold the reply behind a turn that is, by
        definition, the thing the user wants to stop.

        The semaphore is deliberately NOT touched: ``drive_turn``'s ``finally``
        owns the release, and releasing a semaphore this method never acquired
        would free the session while its turn is still unwinding.

        There is no queue to clear on WeCom (its reply is inbound-bound, so it
        steers rather than queueing), which is what keeps this the small half of
        the mid-turn story.
        """
        assert self.client is not None
        session_key = self._session_key(self._route(inbound))
        # Three states. A busy session whose cancel could not run must NOT be told
        # nothing was running -- that is the wedged turn /stop exists for, and the
        # is_busy check one line up already proved otherwise.
        ack = _NOTHING_RUNNING
        if self.sessions.is_busy(session_key):
            ack = _STOP_FAILED
            provider = self.sessions.get_provider(session_key)
            cancel = getattr(provider, "cancel", None)
            if cancel is not None:
                try:
                    await cancel(wait_ack_timeout=0)
                    ack = _STOPPING
                except Exception:
                    logger.warning("WeCom /stop: cancel failed for %s", session_key, exc_info=True)
        await self._ack(inbound, ack)

    async def _handle_busy(self, inbound: Any, session_key: str) -> None:
        """Mid-turn message: fold into the running turn via steer.

        WeCom replies are bound to the inbound request, so a queued-then-drained
        reply can't be delivered reliably later -- WeCom steers rather than
        queueing (capability-driven, like the no-proactive-send gate).

        ``steer()`` returning True only means the session exists, not that a turn
        is active, so it can't detect the is_busy->finished race. Re-check
        ``is_busy`` (the semaphore, which tracks turn-active) instead: if the
        turn already finished, run the message as a fresh turn (safe -- is_busy
        is now False, so no re-entry loop); if a turn is still in flight but
        steer isn't possible yet (cold start, no session id), ask the user to
        resend rather than silently dropping the message.
        """
        assert self.client is not None
        if not self.sessions.is_busy(session_key):
            await self.handle_message(inbound)
            return
        provider = self.sessions.get_provider(session_key)
        steer = getattr(provider, "steer", None)
        # ``is_busy`` stays True through post-turn bookkeeping (all await
        # points), so it alone can't tell a live turn from one that just
        # finished. Gate steer on ``has_active_turn`` (parity with Telegram):
        # steering a prompt that already ended is silently swallowed and would
        # falsely acknowledge a merge. When no turn is live, fall through to the
        # resend prompt instead.
        has_active = getattr(provider, "has_active_turn", None)
        live = has_active is None or bool(has_active())
        steered = bool(
            live
            and getattr(provider, "supports_steer", False)
            and steer is not None
            and await steer(inbound.text)
        )
        if steered:
            await self._ack(inbound, _STEER_MERGED)
        else:
            await self._ack(inbound, _STEER_UNAVAILABLE)

    # ── Helpers ────────────────────────────────────────────────────────────

    def _resolve_agent(self) -> str:
        return self.agent or self.cfg.agent.default_agent or _DEFAULT_KIROCREW_AGENT

    @staticmethod
    def _route(inbound: "WeComInbound") -> str:
        """The conversation *inbound* belongs to, as ``"{chat_type}:{scope}"``.

        A GROUP is addressed by its ``chatid``: every member shares one session,
        so `/new` there rotates the group's transcript rather than the sender's
        private one, and a member's DM history never leaks into the group (nor the
        group's into their DM). A DM is addressed by the peer.

        Carried as one string because it keys ``ConversationState`` as well as the
        session, and the two must never disagree about which conversation a
        generation counter belongs to.
        """
        if inbound.chatid:
            return f"{CHAT_TYPE_GROUP}:{inbound.chatid}"
        return f"{CHAT_TYPE_DIRECT}:{inbound.userid}"

    @staticmethod
    def _split_route(route: str) -> tuple[str, str]:
        """``"{chat_type}:{scope}"`` -> ``(chat_type, scope)``.

        Split once: a chat type carries no colon, while a scope may (a future
        deeper scope path), so only the first separator is a boundary.
        """
        chat_type, _, scope = route.partition(":")
        return chat_type, scope

    def _session_key(self, route: str) -> str:
        chat_type, scope = self._split_route(route)
        return build_dm_session_key(
            "wecom",
            self._resolve_agent(),
            scope,
            gen=self._conv.current_gen(route),
            dm_scope=self.cfg.messaging.dm_scope,
            chat_type=chat_type,
        )

    def _seed_gen(self, route: str) -> int:
        chat_type, scope = self._split_route(route)
        return seed_generation(
            self.sessions,
            channel="wecom",
            agent=self._resolve_agent(),
            user_id=scope,
            dm_scope=self.cfg.messaging.dm_scope,
            chat_type=chat_type,
        )

    def _persist_turn(
        self,
        session_key: str,
        user_text: str,
        reply_text: str,
        is_new: bool,
        agent: str | None = None,
    ) -> None:
        """Record the turn to conversation_log (dashboard visibility + restart)."""
        if self.conv_log is None:
            return
        self.conv_log.append(session_key, "user", user_text, agent=agent)
        if reply_text:
            self.conv_log.append(session_key, "assistant", reply_text, agent=agent)
        if is_new:
            title = (user_text or "").strip().replace("\n", " ")[:40] or "WeCom"
            self.conv_log.set_title(session_key, title)

    async def _notice_bubble(self, req_id: str, text: str) -> None:
        """Send a threshold notice as a SEPARATE WS bubble.

        Kept out of the answer buffer -- and thus out of the persisted turn --
        so it is never replayed next turn as though the assistant said it
        (WeCom binds a reply to the inbound message, so this is the out-of-band
        equivalent of the Telegram/Slack separate-message notice).
        """
        assert self.client is not None
        if not req_id:
            return
        try:
            await self.client.send_bubble(req_id, text)
        except Exception:
            logger.debug("WeCom: notice bubble send failed", exc_info=True)

    async def _ack(self, inbound: "WeComInbound", text: str) -> None:
        """Deliver a short out-of-band ack, preferring the WS bubble.

        Every ack in this dispatcher goes through here. ``response_url`` is NOT
        present on every inbound frame — a text callback can carry only a
        ``req_id`` — and ``send_reply`` with an empty URL is a no-op, so posting
        straight to it drops the ack silently and the user sees nothing at all in
        answer to `/help` or `/stop`. The WS bubble is the reliable path when a
        ``req_id`` is in hand; ``response_url`` is the fallback for the frames that
        carry one instead. Same order as ``WeComTransport._refuse_media``, and the
        reason this is ONE method: the two shapes had already diverged once, with
        the media refusal fixed and eleven acks left on the unreliable path.
        """
        assert self.client is not None
        if inbound.req_id and await self.client.send_bubble(inbound.req_id, text):
            return
        await self.client.send_reply(inbound.response_url, text)

    async def _maybe_notice(self, inbound: "WeComInbound", session_key: str, provider: Any) -> None:
        """Context-length handling, surfaced as a separate bubble post-turn.

        Soft threshold nudges the user to /compact or /new; hard threshold forces
        a compaction so the window never overflows. The backend autocompactor is
        an additional safety net.
        """
        route = self._route(inbound)
        pct = self.sessions.check_context_usage(session_key, provider)
        if pct >= self.cfg.wecom.hard_threshold_pct:
            self._conv.clear_awaiting(route)
            try:
                await provider.compact()
                await provider.wait_for_compaction()
                await self._notice_bubble(inbound.req_id, _AUTO_COMPACTED)
            except Exception:
                logger.debug("WeCom hard-threshold compaction failed", exc_info=True)
        elif pct >= self.cfg.wecom.soft_threshold_pct and not self._conv.is_awaiting(route):
            self._conv.set_awaiting(route)
            await self._notice_bubble(
                inbound.req_id,
                _SOFT_THRESHOLD,
            )

    async def _handle_compact(self, inbound: "WeComInbound") -> None:
        """In-place ACP ``/compact`` on the user's current session."""
        assert self.client is not None
        session_key = self._session_key(self._route(inbound))
        # Serialize compaction against the turn semaphore: compacting while a
        # turn is mutating the same session races the transcript. Distinguish a
        # busy session (ask the user to retry) from an absent one (nothing to
        # compact), and always release what we acquired.
        if not await self.sessions.try_acquire(session_key):
            if self.sessions.has_session(session_key):
                await self._ack(inbound, _COMPACT_BUSY)
            else:
                await self._ack(inbound, _COMPACT_NOTHING)
            return
        try:
            provider = self.sessions.get_provider(session_key)
            if provider is None:
                await self._ack(inbound, _COMPACT_NOTHING)
                return
            await provider.compact()
            await provider.wait_for_compaction()
            await self._ack(inbound, _COMPACT_DONE)
        except Exception:
            logger.exception("WeCom /compact failed for %s", session_key)
            await self._ack(inbound, _COMPACT_FAILED)
        finally:
            self.sessions.release(session_key)
