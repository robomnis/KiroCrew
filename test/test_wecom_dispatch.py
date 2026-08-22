"""Tests for kiro_crew.wecom.transport_dispatch (WeComDispatcher) + commands."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from kiro_crew.acp.types import EVENT_COMPLETE, EVENT_TEXT_CHUNK, AcpEvent
from kiro_crew.wecom.client import WeComInbound, new_stream_id
from kiro_crew.wecom.commands import ConversationState, parse_command
from kiro_crew.wecom.transport_dispatch import WeComDispatcher

# ------------------------------------------------------------------
# Fakes
# ------------------------------------------------------------------


class FakeProvider:
    supports_steer = True

    def __init__(self, events: list) -> None:
        self._events = events
        self.approved: list = []
        self.rejected: list = []
        self.compacted = False
        self.steered: list = []
        self.cancelled: list = []
        self.active_turn = True

    def has_active_turn(self) -> bool:
        return self.active_turn

    async def steer(self, text: str) -> bool:
        self.steered.append(text)
        return True

    async def stream(self, message: str):
        for ev in self._events:
            yield ev

    async def approve_tool(self, rid) -> None:
        self.approved.append(rid)

    async def reject_tool(self, rid) -> None:
        self.rejected.append(rid)

    async def cancel(self, *, wait_ack_timeout: float = 0.0) -> None:
        self.cancelled.append(wait_ack_timeout)

    async def compact(self) -> None:
        self.compacted = True

    async def wait_for_compaction(self, timeout: float = 0.0) -> dict:
        return {"type": "completed", "summary": ""}


class FakeSessions:
    def __init__(
        self,
        provider,
        *,
        is_new=True,
        raise_on_get=None,
        ctx_pct=0.0,
        acquire=True,
        has_session=None,
    ) -> None:
        self._p = provider
        self._is_new = is_new
        self._raise = raise_on_get
        self._ctx_pct = ctx_pct
        self._acquire = acquire
        self._has_session = provider is not None if has_session is None else has_session
        self.acquired: list = []
        self.released: list = []
        self.successes: list = []
        self.failures: list = []
        self.channels: list = []
        self.last_agent = None

    async def get_or_create(self, key, *, agent, channel_id):
        self.last_agent = agent
        if self._raise is not None:
            raise self._raise
        return self._p, self._is_new, False

    async def set_channel(self, key, cid) -> None:
        self.channels.append((key, cid))

    def release(self, key) -> None:
        self.released.append(key)

    def record_success(self, key) -> None:
        self.successes.append(key)

    async def record_failure(self, key) -> None:
        self.failures.append(key)

    def check_context_usage(self, key, provider) -> float:
        return self._ctx_pct

    def get_provider(self, key):
        return self._p

    async def try_acquire(self, key) -> bool:
        self.acquired.append(key)
        return self._acquire

    def has_session(self, key) -> bool:
        return self._has_session

    def is_busy(self, key) -> bool:
        return getattr(self, "_busy", False)

    def max_generation(self, bucket: str) -> int:
        return -1


class _GateResult:
    def __init__(self, action: str = "") -> None:
        self.action = action


class FakeHooks:
    auto_approve_subagent_spawn = False

    def on_tool_call(self, title, *, session_key, agent, tool_kind, raw_params=None):
        return _GateResult("")


class FakeCtx:
    def __init__(self) -> None:
        self.hooks = FakeHooks()

    def build_message(self, text, is_new, key, *, channel_id, agent, resumed, runtime_source):
        assert runtime_source == "wecom"
        return (text, None)


class FakeClient:
    def __init__(self) -> None:
        self.frames: list[dict] = []
        self.replies: list[tuple[str, str]] = []

    async def send_stream(self, req_id, sid, content, *, finish) -> bool:
        self.frames.append({"sid": sid, "content": content, "finish": finish})
        return True

    async def send_bubble(self, req_id, content) -> bool:
        # Mirrors the real client: a NEW bubble means a fresh stream_id, finished.
        return await self.send_stream(req_id, new_stream_id(), content, finish=True)

    async def send_reply(self, url, content) -> None:
        self.replies.append((url, content))

    @property
    def acks(self) -> list[str]:
        """Out-of-band ack texts, whichever path delivered them.

        ``WeComDispatcher._ack`` prefers the WS bubble and falls back to
        ``response_url``, because ``response_url`` is absent on some frames.
        Tests assert on the TEXT that reached the user, not on which of the two
        carried it — otherwise every one of them re-pins the delivery path.
        """
        return [f["content"] for f in self.frames if f["finish"]] + [c for _, c in self.replies]


class FakeConvLog:
    def __init__(self) -> None:
        self.appended: list[tuple[str, str, str]] = []
        self.titles: dict[str, str] = {}

    def append(self, key, role, text, agent=None) -> None:
        self.appended.append((key, role, text))

    def set_title(self, key, title) -> None:
        self.titles[key] = title


def _cfg(default_agent: str = "", approval_mode: str = "interactive"):
    return SimpleNamespace(
        agent=SimpleNamespace(default_agent=default_agent, approval_mode=approval_mode),
        wecom=SimpleNamespace(hard_threshold_pct=95.0, soft_threshold_pct=80.0),
        messaging=SimpleNamespace(
            dm_scope="per-channel-peer",
            idle_reset_minutes=0,
            daily_reset_hour=-1,
            queue_mode="steer",
        ),
    )


def _dispatcher(sessions, ctx, client, *, conv_log=None, agent=None, cfg=None):
    d = WeComDispatcher(
        sessions=sessions,
        ctx_builder=ctx,
        cfg=cfg or _cfg(),
        owner_id="Wei",
        agent=agent,
        conv_log=conv_log,
        approval_mode="interactive",
    )
    d.client = client
    return d


#: The route a plain DM from the default test user keys to. The dispatcher keys
#: both the session and the per-conversation generation on "{chat_type}:{scope}",
#: so a group cannot land in a member's private session.
_DM_ROUTE = "direct:Wei"


def _inbound(text: str = "hello", userid: str = "Wei") -> WeComInbound:
    return WeComInbound(userid=userid, text=text, response_url="https://r", req_id="rq1", chatid="")


# ------------------------------------------------------------------
# Tests: full turn
# ------------------------------------------------------------------


def _deny_wecom_profile(monkeypatch, tmp_path):
    import json

    from kiro_crew.platform import governance_profiles as gp

    pdir = tmp_path / "profiles"
    pdir.mkdir(exist_ok=True)
    monkeypatch.setattr(gp, "_PROFILES_DIR", pdir)
    gp.reset_store()
    (pdir / "host.json").write_text(
        json.dumps(
            {
                "name": "host",
                "bind": {"type": "surface", "id": "host"},
                "channels": {"members": {"mode": "allow", "allow": ["slack"]}},
            }
        )
    )


class TestTurn:
    @pytest.mark.asyncio
    async def test_channels_deny_drops_inbound_message(self, tmp_path, monkeypatch) -> None:
        # HIGH (GPT round-4 #2): a channels DENY must stop handle_message from
        # driving a turn. Regression-locks the WeCom inbound chokepoint.
        from kiro_crew.platform import governance_profiles as gp

        _deny_wecom_profile(monkeypatch, tmp_path)
        provider = FakeProvider([AcpEvent(kind=EVENT_COMPLETE)])
        sessions = FakeSessions(provider)
        d = _dispatcher(sessions, FakeCtx(), FakeClient())
        try:
            await d.handle_message(_inbound("hello"))
            assert sessions.successes == []  # no turn ran
        finally:
            gp.reset_store()

    @pytest.mark.asyncio
    async def test_text_turn_bookkeeping(self) -> None:
        provider = FakeProvider(
            [AcpEvent(kind=EVENT_TEXT_CHUNK, text="hi there"), AcpEvent(kind=EVENT_COMPLETE)]
        )
        sessions = FakeSessions(provider)
        client = FakeClient()
        conv = FakeConvLog()
        d = _dispatcher(sessions, FakeCtx(), client, conv_log=conv)

        await d.handle_message(_inbound("hello"))

        key = d._session_key(_DM_ROUTE)
        # Final answer streamed with finish=True.
        assert any(f["finish"] and f["content"] == "hi there" for f in client.frames)
        # Bookkeeping: success recorded, semaphore released, turn persisted.
        assert sessions.successes == [key]
        assert sessions.released == [key]
        assert (key, "user", "hello") in conv.appended
        assert (key, "assistant", "hi there") in conv.appended

    @pytest.mark.asyncio
    async def test_agent_resolves_to_kirocrew_when_unset(self) -> None:
        provider = FakeProvider([AcpEvent(kind=EVENT_COMPLETE)])
        sessions = FakeSessions(provider)
        d = _dispatcher(sessions, FakeCtx(), FakeClient(), cfg=_cfg(default_agent=""))
        await d.handle_message(_inbound("hi"))
        assert sessions.last_agent == "kirocrew"

    @pytest.mark.asyncio
    async def test_cold_start_failure_finalizes_and_skips_release(self) -> None:
        provider = FakeProvider([AcpEvent(kind=EVENT_COMPLETE)])
        sessions = FakeSessions(provider, raise_on_get=RuntimeError("boom"))
        client = FakeClient()
        d = _dispatcher(sessions, FakeCtx(), client)

        # Must not raise — the dispatcher swallows and finalizes.
        await d.handle_message(_inbound("hi"))

        # Placeholder finalized (no perma-"🤔 …") even though get_or_create failed.
        assert any(f["finish"] for f in client.frames)
        # Never held the semaphore -> never release / record_failure it.
        assert sessions.released == []
        assert sessions.failures == []

    @pytest.mark.asyncio
    async def test_soft_threshold_notice_is_separate_and_unpersisted(self) -> None:
        provider = FakeProvider(
            [AcpEvent(kind=EVENT_TEXT_CHUNK, text="answer"), AcpEvent(kind=EVENT_COMPLETE)]
        )
        sessions = FakeSessions(provider, ctx_pct=85.0)  # >= soft (80), < hard (95)
        client = FakeClient()
        conv = FakeConvLog()
        d = _dispatcher(sessions, FakeCtx(), client, conv_log=conv)

        await d.handle_message(_inbound("hello"))

        # Notice surfaced as a SEPARATE bubble (its own stream_id, finish=True).
        assert any("对话上下文已较长" in f["content"] for f in client.frames)
        # ...but NOT persisted into the assistant turn (only the real answer is).
        assistant_texts = [t for (_, role, t) in conv.appended if role == "assistant"]
        assert assistant_texts == ["answer"]


# ------------------------------------------------------------------
# Tests: commands
# ------------------------------------------------------------------


class TestCommands:
    @pytest.mark.asyncio
    async def test_new_bumps_gen_and_acks(self) -> None:
        sessions = FakeSessions(FakeProvider([]))
        client = FakeClient()
        d = _dispatcher(sessions, FakeCtx(), client)

        await d.handle_message(_inbound("/new"))

        assert client.acks == ["✅ 已开始新对话"]
        assert d._conv.current_gen(_DM_ROUTE) == 1  # generation bumped
        assert sessions.successes == []  # no LLM turn

    @pytest.mark.asyncio
    async def test_compact_command(self) -> None:
        provider = FakeProvider([])
        sessions = FakeSessions(provider)
        client = FakeClient()
        d = _dispatcher(sessions, FakeCtx(), client)

        await d.handle_message(_inbound("/compact"))

        key = d._session_key(_DM_ROUTE)
        assert provider.compacted is True
        assert sessions.acquired == [key]
        assert sessions.released == [key]
        assert client.acks == ["🗜️ 已压缩上下文。"]

    @pytest.mark.asyncio
    async def test_compact_refused_while_turn_busy(self) -> None:
        provider = FakeProvider([])
        sessions = FakeSessions(provider, acquire=False, has_session=True)
        client = FakeClient()
        d = _dispatcher(sessions, FakeCtx(), client)

        await d.handle_message(_inbound("/compact"))

        assert provider.compacted is False
        assert sessions.released == []
        assert client.acks == ["⏳ 正在处理上一条消息，请稍后再试 /compact。"]

    @pytest.mark.asyncio
    async def test_compact_without_active_session(self) -> None:
        sessions = FakeSessions(None, acquire=False, has_session=False)
        client = FakeClient()
        d = _dispatcher(sessions, FakeCtx(), client)

        await d.handle_message(_inbound("/compact"))

        assert sessions.released == []
        assert client.acks == ["ℹ️ 当前没有可压缩的对话。"]

    @pytest.mark.asyncio
    async def test_link_rejected_on_wecom(self) -> None:
        sessions = FakeSessions(FakeProvider([]))
        client = FakeClient()
        d = _dispatcher(sessions, FakeCtx(), client)

        await d.handle_message(_inbound("/link"))

        assert len(client.acks) == 1
        assert "/link" in client.acks[0]  # explains it's unavailable here
        assert sessions.successes == []  # no LLM turn

    def test_parse_command(self) -> None:
        assert parse_command("/new") == "new"
        assert parse_command("新对话") == "new"
        assert parse_command("清空") == "new"
        assert parse_command("/compact") == "compact"
        assert parse_command("/link") == "link"
        assert parse_command("/unlink") == "unlink"
        assert parse_command("hello") is None

    def test_parse_command_after_a_group_mention(self) -> None:
        """A group chat requires @-mentioning the bot, so the command arrives prefixed."""
        assert parse_command("@Kiro /new") == "new"
        assert parse_command("@Kiro /compact") == "compact"
        assert parse_command("@Kiro 新对话") == "new"
        assert parse_command("@Kiro 清空") == "new"
        assert parse_command("@Kiro /link") == "link"
        assert parse_command("@Kiro /unlink") == "unlink"

    def test_mention_does_not_loosen_command_matching(self) -> None:
        """Only ONE leading mention token in front of an EXACT command counts.

        Everything here must keep reaching the model as ordinary text. The bot's
        display name is not known to the parser, so the mention is recognized
        structurally -- which is exactly why the rest of the match has to stay
        exact rather than becoming a prefix or substring test.
        """
        for text in (
            "@Kiro hello",  # mentioned prose
            "@Kiro",  # bare mention, no remainder
            "@Kiro please /new",  # command embedded in prose
            "@Kiro /new extra",  # trailing argument
            "@someone ordinary text",  # a mention of somebody else
            "@a @b /new",  # only one mention token is consumed
            "hello /new",  # command not at the start
            "/new @Kiro",  # trailing mention
            "/new@Kiro",  # Telegram's suffix syntax, not WeCom's
            "@Kiro /bogus",  # unknown command stays unknown
            "@Kiro/new",  # no separator -- not a mention token
        ):
            assert parse_command(text) is None, text

    @pytest.mark.asyncio
    async def test_group_mention_new_bumps_gen_and_acks(self) -> None:
        """The reported shape: '@Kiro /new' must reset, not steer into the turn."""
        sessions = FakeSessions(FakeProvider([]))
        client = FakeClient()
        d = _dispatcher(sessions, FakeCtx(), client)

        await d.handle_message(_inbound("@Kiro /new"))

        assert client.acks == ["✅ 已开始新对话"]
        assert d._conv.current_gen(_DM_ROUTE) == 1
        assert sessions.successes == []  # no LLM turn

    @pytest.mark.asyncio
    async def test_group_mention_compact_reaches_compaction(self) -> None:
        provider = FakeProvider([])
        sessions = FakeSessions(provider)
        client = FakeClient()
        d = _dispatcher(sessions, FakeCtx(), client)

        await d.handle_message(_inbound("@Kiro /compact"))

        assert provider.compacted is True
        assert client.acks == ["🗜️ 已压缩上下文。"]

    @pytest.mark.asyncio
    async def test_mentioned_prose_still_runs_a_turn_with_text_intact(self) -> None:
        """Ordinary inbound text is never rewritten -- the model sees the mention."""
        provider = FakeProvider(
            [AcpEvent(kind=EVENT_TEXT_CHUNK, text="sure"), AcpEvent(kind=EVENT_COMPLETE)]
        )
        sessions = FakeSessions(provider)
        conv = FakeConvLog()
        d = _dispatcher(sessions, FakeCtx(), FakeClient(), conv_log=conv)

        await d.handle_message(_inbound("@Kiro explain this stack trace"))

        key = d._session_key(_DM_ROUTE)
        assert sessions.successes == [key]
        assert (key, "user", "@Kiro explain this stack trace") in conv.appended
        assert d._conv.current_gen(_DM_ROUTE) == 0  # not treated as /new

    @pytest.mark.asyncio
    async def test_mentioned_unknown_command_still_runs_a_turn(self) -> None:
        provider = FakeProvider([AcpEvent(kind=EVENT_COMPLETE)])
        sessions = FakeSessions(provider)
        d = _dispatcher(sessions, FakeCtx(), FakeClient())

        await d.handle_message(_inbound("@Kiro /bogus"))

        assert sessions.successes == [d._session_key(_DM_ROUTE)]
        assert d._conv.current_gen(_DM_ROUTE) == 0

    def test_conversation_state(self) -> None:
        s = ConversationState()
        assert s.current_gen("u") == 0
        assert s.bump_gen("u") == 1
        s.set_awaiting("u")
        assert s.is_awaiting("u") is True
        s.clear_awaiting("u")
        assert s.is_awaiting("u") is False


class TestWeComMidTurn:
    @pytest.mark.asyncio
    async def test_busy_steers_and_acknowledges(self) -> None:
        provider = FakeProvider([])
        sessions = FakeSessions(provider)
        sessions._busy = True
        client = FakeClient()
        d = _dispatcher(sessions, FakeCtx(), client)

        await d.handle_message(_inbound("and also this"))

        assert provider.steered == ["and also this"]
        assert any("合并" in content for content in client.acks)
        assert sessions.successes == []  # no full turn ran while busy

    @pytest.mark.asyncio
    async def test_busy_but_turn_finished_runs_fresh(self) -> None:
        # is_busy is False by the time _handle_busy runs (turn finished in the
        # window) -> run the message as a fresh turn instead of a false ack.
        provider = FakeProvider([AcpEvent(kind=EVENT_COMPLETE)])
        sessions = FakeSessions(provider)  # _busy defaults False
        client = FakeClient()
        d = _dispatcher(sessions, FakeCtx(), client)

        await d._handle_busy(_inbound("later"), d._session_key(_DM_ROUTE))

        assert sessions.successes  # a real turn ran
        assert provider.steered == []  # not steered

    @pytest.mark.asyncio
    async def test_busy_steer_unavailable_asks_resend(self) -> None:
        # A turn is genuinely in flight but steer isn't possible (cold start):
        # ask the user to resend rather than looping or dropping the message.
        provider = FakeProvider([])
        provider.supports_steer = False
        sessions = FakeSessions(provider)
        sessions._busy = True
        client = FakeClient()
        d = _dispatcher(sessions, FakeCtx(), client)

        await d._handle_busy(_inbound("later"), d._session_key(_DM_ROUTE))

        assert any("重发" in content for content in client.acks)
        assert sessions.successes == []

    @pytest.mark.asyncio
    async def test_busy_no_active_turn_does_not_steer(self) -> None:
        # Semaphore held (post-turn bookkeeping) but no turn is live: steer must
        # not be attempted and must not falsely acknowledge a merge -- fall
        # through to the resend prompt instead.
        provider = FakeProvider([])
        provider.active_turn = False
        sessions = FakeSessions(provider)
        sessions._busy = True
        client = FakeClient()
        d = _dispatcher(sessions, FakeCtx(), client)

        await d._handle_busy(_inbound("later"), d._session_key(_DM_ROUTE))

        assert provider.steered == []
        assert not any("合并" in content for content in client.acks)
        assert any("重发" in content for content in client.acks)
        assert sessions.successes == []


class TestCommandVocabulary:
    """Every command is reachable AND discoverable — the spec table is one source.

    A command that parses but is missing from `/help` is a feature nobody finds;
    one listed but unreachable is a lie. Deriving both from COMMAND_SPEC is what
    makes the pair inseparable, so this asserts the derivation rather than a list.
    """

    def test_every_alias_in_the_spec_is_reachable(self) -> None:
        from kiro_crew.wecom.commands import _REFUSED_COMMANDS, COMMAND_SPEC

        for spec in COMMAND_SPEC + _REFUSED_COMMANDS:
            assert parse_command(f"/{spec.name}") == spec.name
            for alias in spec.aliases:
                assert parse_command(alias) == spec.name, f"{alias} unreachable"

    def test_every_visible_command_appears_in_help(self) -> None:
        from kiro_crew.wecom.commands import COMMAND_SPEC, build_help

        body = build_help()
        for spec in COMMAND_SPEC:
            assert f"/{spec.name}" in body, f"/{spec.name} missing from help"
            for alias in spec.aliases:
                assert alias in body

    def test_the_refuse_only_commands_are_reachable_but_not_help_rows(self) -> None:
        # They must still parse (so the dispatcher can explain rather than hand the
        # text to the model), but spending a third of the card on two commands that
        # only refuse reads as capability the channel does not have.
        from kiro_crew.wecom.commands import build_help

        body = build_help()
        assert parse_command("/link") == "link"
        assert parse_command("/unlink") == "unlink"
        assert "/unlink —" not in body
        assert "/link 与 /unlink 不可用" in body

    def test_a_command_behind_a_group_mention_still_parses(self) -> None:
        # Addressing the bot is mandatory in a WeCom group.
        assert parse_command("@Kiro /stop") == "stop"
        assert parse_command("@Kiro 帮助") == "help"

    def test_mentioned_prose_is_not_a_command(self) -> None:
        assert parse_command("@Kiro please /stop the deploy") is None


class TestHelp:
    @pytest.mark.asyncio
    async def test_help_replies_with_the_card_and_runs_no_turn(self) -> None:
        provider = FakeProvider([AcpEvent(kind=EVENT_COMPLETE)])
        sessions = FakeSessions(provider)
        client = FakeClient()
        d = _dispatcher(sessions, FakeCtx(), client)

        await d.handle_message(_inbound("/help"))

        assert sessions.successes == [], "help must not spend a turn"
        assert len(client.acks) == 1
        assert "/stop" in client.acks[0]


class TestStop:
    @pytest.mark.asyncio
    async def test_stop_cancels_a_live_turn_cooperatively(self) -> None:
        provider = FakeProvider([AcpEvent(kind=EVENT_COMPLETE)])
        sessions = FakeSessions(provider)
        sessions._busy = True
        client = FakeClient()
        d = _dispatcher(sessions, FakeCtx(), client)

        await d.handle_message(_inbound("/stop"))

        # Fire-and-forget: waiting for the ack would hold the reply behind the
        # very turn being stopped.
        assert provider.cancelled == [0]
        assert "正在停止" in client.acks[0]

    @pytest.mark.asyncio
    async def test_stop_with_nothing_running_says_so(self) -> None:
        provider = FakeProvider([AcpEvent(kind=EVENT_COMPLETE)])
        sessions = FakeSessions(provider)  # not busy
        client = FakeClient()
        d = _dispatcher(sessions, FakeCtx(), client)

        await d.handle_message(_inbound("/stop"))

        assert provider.cancelled == []
        assert "没有正在生成" in client.acks[0]

    @pytest.mark.asyncio
    async def test_stop_is_intercepted_BEFORE_the_steer_path(self) -> None:
        # The bug this guards: /stop arriving mid-turn would otherwise be folded
        # into the running turn as ordinary text, so the turn it exists to abort
        # would instead be told about it.
        provider = FakeProvider([AcpEvent(kind=EVENT_COMPLETE)])
        sessions = FakeSessions(provider)
        sessions._busy = True
        d = _dispatcher(sessions, FakeCtx(), FakeClient())

        await d.handle_message(_inbound("/stop"))

        assert provider.steered == []
        assert provider.cancelled == [0]

    @pytest.mark.asyncio
    async def test_stop_never_releases_a_semaphore_it_did_not_acquire(self) -> None:
        provider = FakeProvider([AcpEvent(kind=EVENT_COMPLETE)])
        sessions = FakeSessions(provider)
        sessions._busy = True
        d = _dispatcher(sessions, FakeCtx(), FakeClient())

        await d.handle_message(_inbound("/stop"))

        assert sessions.released == [], "drive_turn's finally owns the release"

    @pytest.mark.asyncio
    async def test_a_failing_cancel_is_reported_as_nothing_stopped(self) -> None:
        class Boom(FakeProvider):
            async def cancel(self, *, wait_ack_timeout: float = 0.0) -> None:
                raise RuntimeError("wire down")

        provider = Boom([AcpEvent(kind=EVENT_COMPLETE)])
        sessions = FakeSessions(provider)
        sessions._busy = True
        client = FakeClient()
        d = _dispatcher(sessions, FakeCtx(), client)

        await d.handle_message(_inbound("/stop"))

        # And NOT "nothing was running" -- the code just proved otherwise, and this
        # is exactly the wedged turn /stop exists for.
        assert "停止失败" in client.acks[0]


class TestGroupSessionScoping:
    """A group turn must not land in the sender's private session.

    `chatid` was parsed and never used, so a group message keyed to the sender's
    DM: their private history answered in the group, and the group's answered in
    their DM.
    """

    def _group(self, chatid: str = "chat-42", userid: str = "Wei") -> WeComInbound:
        return WeComInbound(
            userid=userid, text="hi", response_url="https://r", req_id="rq1", chatid=chatid
        )

    def test_a_group_and_a_dm_key_to_different_sessions(self) -> None:
        d = _dispatcher(FakeSessions(None), FakeCtx(), FakeClient())
        dm = d._session_key(d._route(_inbound("hello", "Wei")))
        group = d._session_key(d._route(self._group(userid="Wei")))
        assert dm != group
        assert "chat-42" in group and "chat-42" not in dm

    def test_two_members_of_one_group_share_its_session(self) -> None:
        # The bot in a group is one assistant everyone is talking to.
        d = _dispatcher(FakeSessions(None), FakeCtx(), FakeClient())
        a = d._session_key(d._route(self._group(userid="Wei")))
        b = d._session_key(d._route(self._group(userid="LiHaoYi")))
        assert a == b

    def test_a_group_route_is_typed_as_a_group_not_a_thread(self) -> None:
        d = _dispatcher(FakeSessions(None), FakeCtx(), FakeClient())
        assert d._route(self._group()) == "group:chat-42"
        assert d._route(_inbound("hi", "Wei")) == "direct:Wei"

    def test_a_group_does_not_collapse_into_the_unified_dm_bucket(self) -> None:
        # dm_scope=unified merges 1:1 DMs across surfaces for continuity. A group
        # joining that bucket would put private DM content in a shared room.
        cfg = _cfg()
        cfg.messaging.dm_scope = "unified"
        d = _dispatcher(FakeSessions(None), FakeCtx(), FakeClient(), cfg=cfg)
        group = d._session_key(d._route(self._group()))
        dm = d._session_key(d._route(_inbound("hi", "Wei")))
        assert group.startswith("wecom:")
        assert "chat-42" in group
        assert dm != group

    def test_new_in_a_group_rotates_the_group_not_the_senders_dm(self) -> None:
        d = _dispatcher(FakeSessions(None), FakeCtx(), FakeClient())
        dm_before = d._session_key(d._route(_inbound("hi", "Wei")))
        d._conv.bump_gen(d._route(self._group()))
        assert d._session_key(d._route(_inbound("hi", "Wei"))) == dm_before
        assert d._session_key(d._route(self._group())) != dm_before


class TestStopReportsThreeStates:
    """A busy session whose cancel failed must not be told nothing was running.

    That is the case `/stop` exists for — cancel failures cluster on exactly the
    wedged turns a user reaches for it — and reporting "nothing is running" there
    contradicts the `is_busy` check one line earlier.
    """

    @pytest.mark.asyncio
    async def test_a_provider_with_no_cancel_reports_failure_not_idle(self) -> None:
        class NoCancel(FakeProvider):
            cancel = None

        sessions = FakeSessions(NoCancel([AcpEvent(kind=EVENT_COMPLETE)]))
        sessions._busy = True
        client = FakeClient()
        d = _dispatcher(sessions, FakeCtx(), client)

        await d.handle_message(_inbound("/stop"))

        assert "停止失败" in client.acks[0]
        assert "没有正在生成" not in client.acks[0]

    @pytest.mark.asyncio
    async def test_the_success_ack_says_STOPPING_not_stopped(self) -> None:
        # The cancel is cooperative (wait_ack_timeout=0), so the turn stops at its
        # next safe point and the bubble may still move for a moment. Claiming
        # "已停止" would be a statement the user can watch being false.
        provider = FakeProvider([AcpEvent(kind=EVENT_COMPLETE)])
        sessions = FakeSessions(provider)
        sessions._busy = True
        client = FakeClient()
        d = _dispatcher(sessions, FakeCtx(), client)

        await d.handle_message(_inbound("/stop"))

        assert "正在停止" in client.acks[0]


class TestAcksSurviveAMissingResponseUrl:
    """An ack must reach the user on a frame that carries no ``response_url``.

    ``response_url`` is not present on every inbound frame — a text callback can
    arrive with only a ``req_id`` — and ``send_reply("")`` is a no-op, so an ack
    posted straight to it vanishes: the user types `/help` and sees nothing at all.
    Every ack in the dispatcher therefore prefers the WS bubble, matching
    ``WeComTransport._refuse_media``. The two shapes had already diverged once,
    with the media refusal fixed and the acks left on the unreliable path.
    """

    @staticmethod
    def _urlless(text: str) -> WeComInbound:
        return WeComInbound(userid="Wei", text=text, response_url="", req_id="rq1", chatid="")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("command", ["/help", "/new", "/stop"])
    async def test_a_command_ack_is_delivered_over_the_bubble(self, command: str) -> None:
        provider = FakeProvider([AcpEvent(kind=EVENT_COMPLETE)])
        client = FakeClient()
        d = _dispatcher(FakeSessions(provider), FakeCtx(), client)

        await d.handle_message(self._urlless(command))

        assert client.acks, f"{command} produced no ack the user could see"
        assert not client.replies, "nothing may be posted to an empty response_url"

    @pytest.mark.asyncio
    async def test_the_response_url_is_still_the_fallback_when_there_is_no_req_id(self) -> None:
        # The other half of the contract: a frame with a URL but no req_id has no
        # bubble to write to, and must not lose the ack either.
        provider = FakeProvider([AcpEvent(kind=EVENT_COMPLETE)])
        client = FakeClient()
        d = _dispatcher(FakeSessions(provider), FakeCtx(), client)

        await d.handle_message(
            WeComInbound(userid="Wei", text="/help", response_url="https://r", req_id="")
        )

        assert [c for _u, c in client.replies], "the URL fallback must carry the ack"

    @pytest.mark.asyncio
    async def test_a_refused_bubble_falls_back_rather_than_dropping_the_ack(self) -> None:
        # send_bubble returning False is a refusal, not a delivery. Treating it as
        # one would lose the ack while the fallback sat unused.
        class RefusingClient(FakeClient):
            async def send_bubble(self, req_id, content) -> bool:
                return False

        client = RefusingClient()
        d = _dispatcher(
            FakeSessions(FakeProvider([AcpEvent(kind=EVENT_COMPLETE)])), FakeCtx(), client
        )

        await d.handle_message(
            WeComInbound(userid="Wei", text="/help", response_url="https://r", req_id="rq1")
        )

        assert [c for _u, c in client.replies], "a refused bubble must fall back to the URL"
