"""Tests for kiro_crew.wecom.transport (WeComTransport, Layer 1)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from kiro_crew.messaging.transport import InboundMessage
from kiro_crew.wecom.client import WECOM_MAX_REPLY_BYTES, WeComInbound
from kiro_crew.wecom.transport import WECOM_CAPABILITIES, WeComTransport


class FakeClient:
    """Minimal WeComClient stand-in recording lifecycle + sends."""

    def __init__(self) -> None:
        self.started = False
        self.closed = False
        self.replies: list[tuple[str, str]] = []
        self.bubbles: list[tuple[str, str]] = []
        self.bubble_ok = True

    async def start(self) -> None:
        self.started = True

    async def close(self) -> None:
        self.closed = True

    async def send_bubble(self, req_id: str, content: str) -> bool:
        self.bubbles.append((req_id, content))
        return self.bubble_ok

    async def send_reply(self, url: str, content: str) -> None:
        self.replies.append((url, content))


def _inbound(userid: str = "Wei", text: str = "hi") -> WeComInbound:
    return WeComInbound(userid=userid, text=text, response_url="https://r", req_id="rq1", chatid="")


class TestCapabilities:
    def test_wecom_shape(self) -> None:
        cap = WECOM_CAPABILITIES
        assert cap.streaming is True
        assert cap.edit is True
        assert cap.max_buttons == 0  # no tappable chips
        assert cap.supports_proactive_send is False  # reply bound to inbound
        # Spelled as the derivation, not as the constant production assigns
        # from: `== WECOM_SAFE_MESSAGE_CHARS` is X == X and would survive someone
        # dropping the `// 4`, which is the whole point of the declaration.
        assert cap.max_message_chars == WECOM_MAX_REPLY_BYTES // 4


class TestConfiguredTargets:
    def test_allow_all_policy_remains_visible_but_unavailable(self) -> None:
        transport = WeComTransport(FakeClient(), allow_all=True)

        assert [target.to_dict("wecom") for target in transport.configured_targets()] == [
            {
                "channel_type": "wecom",
                "target_id": "policy:all",
                "label": "WeCom · organization users",
                "available": False,
                "unavailable_reason": "WeCom only allows replies to an inbound message",
            }
        ]


class TestAuthorize:
    def test_owner_allowed(self) -> None:
        t = WeComTransport(FakeClient(), owner_id="Wei")
        assert t.authorize(_msg("Wei")) is True

    def test_allowlist_member_allowed(self) -> None:
        t = WeComTransport(FakeClient(), allowed_users=["Wei", "LiHaoYi"])
        assert t.authorize(_msg("LiHaoYi")) is True

    def test_unknown_denied(self) -> None:
        t = WeComTransport(FakeClient(), owner_id="Wei", allowed_users=["LiHaoYi"])
        with patch("kiro_crew.wecom.transport.sel") as mock_sel:
            assert t.authorize(_msg("stranger")) is False
        mock_sel().log_api_access.assert_called_once()

    def test_empty_userid_denied(self) -> None:
        t = WeComTransport(FakeClient(), owner_id="Wei")
        with patch("kiro_crew.wecom.transport.sel"):
            assert t.authorize(_msg("")) is False

    def test_empty_allowlist_and_no_owner_denies_everyone(self) -> None:
        t = WeComTransport(FakeClient())  # fail closed
        with patch("kiro_crew.wecom.transport.sel"):
            assert t.authorize(_msg("anyone")) is False

    def test_allow_all_admits_any_userid(self) -> None:
        t = WeComTransport(FakeClient(), allow_all=True)  # explicit opt-in
        assert t.authorize(_msg("anyone")) is True
        assert t.authorize(_msg("someone-else")) is True

    def test_allow_all_still_denies_empty_userid(self) -> None:
        # Even under allow-all, an anonymous/malformed frame never dispatches.
        t = WeComTransport(FakeClient(), allow_all=True)
        with patch("kiro_crew.wecom.transport.sel"):
            assert t.authorize(_msg("")) is False

    def test_allow_all_off_is_not_inferred_from_empty_list(self) -> None:
        # The everybody grant is ONLY the explicit flag — an empty allow-list
        # plus allow_all=False stays fail-closed.
        t = WeComTransport(FakeClient(), allowed_users=[], allow_all=False)
        with patch("kiro_crew.wecom.transport.sel"):
            assert t.authorize(_msg("anyone")) is False


class TestReceive:
    @pytest.mark.asyncio
    async def test_authorized_dispatches_inbound(self) -> None:
        dispatched: list[WeComInbound] = []

        async def dispatch(inbound: WeComInbound) -> None:
            dispatched.append(inbound)

        t = WeComTransport(FakeClient(), owner_id="Wei", dispatch=dispatch)
        await t.receive(_inbound("Wei", "hello"))
        assert len(dispatched) == 1
        assert dispatched[0].text == "hello"
        assert dispatched[0].req_id == "rq1"

    @pytest.mark.asyncio
    async def test_unauthorized_does_not_dispatch(self) -> None:
        dispatched: list[WeComInbound] = []

        async def dispatch(inbound: WeComInbound) -> None:
            dispatched.append(inbound)

        t = WeComTransport(FakeClient(), owner_id="Wei", dispatch=dispatch)
        with patch("kiro_crew.wecom.transport.sel"):
            await t.receive(_inbound("stranger", "hello"))
        assert dispatched == []

    @pytest.mark.asyncio
    async def test_empty_text_dropped(self) -> None:
        dispatched: list[WeComInbound] = []

        async def dispatch(inbound: WeComInbound) -> None:
            dispatched.append(inbound)

        t = WeComTransport(FakeClient(), owner_id="Wei", dispatch=dispatch)
        await t.receive(_inbound("Wei", ""))
        assert dispatched == []


class TestMediaOnlyInbound:
    """A media-only message is a message, so it may not vanish.

    WeCom has no download path (``files_inbound=False``), but returning early on
    empty text discarded an image or file with no reply and no audit row: the
    sender saw a successful send while the agent was never told anything arrived.
    """

    def _media(self, userid: str = "Wei", msgtype: str = "image") -> WeComInbound:
        return WeComInbound(
            userid=userid, text="", response_url="https://r", req_id="rq1", msgtype=msgtype
        )

    @pytest.mark.asyncio
    async def test_an_authorized_sender_gets_one_refusal_naming_the_kind(self) -> None:
        dispatched: list[WeComInbound] = []

        async def dispatch(inbound: WeComInbound) -> None:
            dispatched.append(inbound)

        client = FakeClient()
        t = WeComTransport(client, owner_id="Wei", dispatch=dispatch)
        with patch("kiro_crew.wecom.transport.sel") as sel:
            await t.receive(self._media())

        assert dispatched == [], "no turn runs: there is nothing to send the model"
        # Delivered over the WS bubble, which is the path that always exists;
        # response_url is absent on some frames.
        assert len(client.bubbles) == 1
        req_id, body = client.bubbles[0]
        assert req_id == "rq1"
        assert "图片" in body
        assert client.replies == []
        sel().log_api_access.assert_called_once()
        kwargs = sel().log_api_access.call_args.kwargs
        # Same operation vocabulary as every other transport's receive-path row
        # (the reason lives in `outcome`), so a query grouping receive denials by
        # operation cannot miss WeCom's.
        assert kwargs["operation"] == "wecom_transport.receive"
        assert kwargs["outcome"] == "media_unsupported"
        assert kwargs["source"] == "wecom"

    @pytest.mark.asyncio
    async def test_an_unauthorized_sender_gets_nothing_at_all(self) -> None:
        # The refusal must sit BEHIND authorization: replying would tell a
        # stranger that something is listening and what it is.
        client = FakeClient()
        t = WeComTransport(client, owner_id="Wei")
        with patch("kiro_crew.wecom.transport.sel"):
            await t.receive(self._media(userid="stranger"))
        assert client.replies == []

    @pytest.mark.asyncio
    async def test_an_unrecognized_kind_is_dropped_rather_than_answered(self) -> None:
        # Only a KNOWN media kind is refused. Answering every non-text msgtype
        # would send an unsolicited reply for a system or event frame the user
        # never composed, and audit it as an unsupported attachment.
        client = FakeClient()
        t = WeComTransport(client, owner_id="Wei")
        with patch("kiro_crew.wecom.transport.sel") as sel:
            await t.receive(self._media(msgtype="event"))

        assert client.bubbles == []
        assert client.replies == []
        sel().log_api_access.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_hostile_msgtype_is_never_echoed_back_to_the_wire(self) -> None:
        # msgtype is externally-derived frame content. It is matched against the
        # known map, so an attacker-chosen value can reach neither a user-visible
        # reply nor the audit row -- it simply does not match.
        client = FakeClient()
        t = WeComTransport(client, owner_id="Wei")
        with patch("kiro_crew.wecom.transport.sel") as sel:
            await t.receive(self._media(msgtype="<script>alert(1)</script>"))

        assert client.bubbles == [] and client.replies == []
        sel().log_api_access.assert_not_called()

    @pytest.mark.asyncio
    async def test_the_refusal_falls_back_to_the_one_shot_reply(self) -> None:
        # No req_id (or a dead WS) must not lose the refusal -- that is the same
        # silent drop this path exists to remove.
        client = FakeClient()
        client.bubble_ok = False
        t = WeComTransport(client, owner_id="Wei")
        with patch("kiro_crew.wecom.transport.sel"):
            await t.receive(self._media())
        assert len(client.replies) == 1
        assert "图片" in client.replies[0][1]

    @pytest.mark.asyncio
    async def test_a_text_message_is_unaffected(self) -> None:
        dispatched: list[WeComInbound] = []

        async def dispatch(inbound: WeComInbound) -> None:
            dispatched.append(inbound)

        client = FakeClient()
        t = WeComTransport(client, owner_id="Wei", dispatch=dispatch)
        await t.receive(_inbound("Wei", "hello"))
        assert len(dispatched) == 1
        assert client.replies == []

    @pytest.mark.asyncio
    async def test_non_wecom_envelope_dropped(self) -> None:
        dispatched: list[WeComInbound] = []

        async def dispatch(inbound: WeComInbound) -> None:
            dispatched.append(inbound)

        t = WeComTransport(FakeClient(), owner_id="Wei", dispatch=dispatch)
        await t.receive({"not": "a WeComInbound"})
        assert dispatched == []


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_connect_disconnect_delegate(self) -> None:
        client = FakeClient()
        t = WeComTransport(client, owner_id="Wei")
        await t.connect()
        assert client.started is True
        await t.disconnect()
        assert client.closed is True


def _msg(userid: str) -> InboundMessage:
    return InboundMessage(channel_type="wecom", user_id=userid, conversation_id=userid, text="hi")


class TestGroupChatBoundary:
    """A group is a broader disclosure boundary than a DM, so it is its own opt-in.

    Every member of a WeCom group reads the agent's tool output and any file
    contents it quotes, while the sender allow-list only says WHO may drive a
    turn. Inferring group access from a user grant would widen disclosure without
    the operator ever choosing it.
    """

    def _group(self, chatid: str = "chat-42", userid: str = "Wei") -> WeComInbound:
        return WeComInbound(
            userid=userid, text="hi", response_url="https://r", req_id="rq1", chatid=chatid
        )

    async def _received(self, transport: WeComTransport, inbound: WeComInbound) -> list:
        seen: list = []

        async def dispatch(msg: WeComInbound) -> None:
            seen.append(msg)

        transport._dispatch = dispatch
        with patch("kiro_crew.wecom.transport.sel"):
            await transport.receive(inbound)
        return seen

    @pytest.mark.asyncio
    async def test_a_group_message_is_denied_by_default(self) -> None:
        t = WeComTransport(FakeClient(), owner_id="Wei")
        assert await self._received(t, self._group()) == []

    @pytest.mark.asyncio
    async def test_a_group_denial_is_audited(self) -> None:
        # An operator who added the bot to a group and got silence needs the audit
        # log to show the group was refused, not that the message never arrived.
        t = WeComTransport(FakeClient(), owner_id="Wei")

        async def dispatch(msg: WeComInbound) -> None:
            raise AssertionError("must not dispatch")

        t._dispatch = dispatch
        with patch("kiro_crew.wecom.transport.sel") as sel:
            await t.receive(self._group())
        kwargs = sel().log_api_access.call_args.kwargs
        assert kwargs["outcome"] == "denied_group_chat"
        assert kwargs["operation"] == "wecom_transport.receive"

    @pytest.mark.asyncio
    async def test_allow_all_users_does_not_open_groups(self) -> None:
        # The two grants answer different questions: who may DRIVE vs who may READ.
        t = WeComTransport(FakeClient(), allow_all=True)
        assert await self._received(t, self._group()) == []

    @pytest.mark.asyncio
    async def test_the_explicit_opt_in_admits_a_group(self) -> None:
        t = WeComTransport(FakeClient(), owner_id="Wei", allow_group_chats=True)
        assert len(await self._received(t, self._group())) == 1

    @pytest.mark.asyncio
    async def test_an_allow_list_narrows_which_groups(self) -> None:
        t = WeComTransport(
            FakeClient(),
            owner_id="Wei",
            allow_group_chats=True,
            allowed_chat_ids=["chat-known"],
        )
        assert await self._received(t, self._group(chatid="chat-other")) == []
        assert len(await self._received(t, self._group(chatid="chat-known"))) == 1

    @pytest.mark.asyncio
    async def test_a_group_still_requires_an_authorized_SENDER(self) -> None:
        # Enabling groups does not make the channel open: the per-sender
        # deny-by-default check still runs, and runs FIRST.
        t = WeComTransport(FakeClient(), owner_id="Wei", allow_group_chats=True)
        assert await self._received(t, self._group(userid="stranger")) == []

    @pytest.mark.asyncio
    async def test_a_dm_is_unaffected_by_the_group_gate(self) -> None:
        t = WeComTransport(FakeClient(), owner_id="Wei")  # groups off
        assert len(await self._received(t, _inbound("Wei", "hello"))) == 1

    @pytest.mark.asyncio
    async def test_a_group_conversation_id_addresses_the_GROUP(self) -> None:
        # The session must belong to the group, not to whoever spoke: otherwise a
        # member's private history answers in the group and vice versa.
        captured: list[InboundMessage] = []
        t = WeComTransport(FakeClient(), owner_id="Wei", allow_group_chats=True)
        real = t.authorize

        def spy(msg: InboundMessage) -> bool:
            captured.append(msg)
            return real(msg)

        t.authorize = spy  # type: ignore[method-assign]
        await self._received(t, self._group())
        assert captured[0].conversation_id == "chat-42"
        assert captured[0].thread_id is None, "a WeCom group is a conversation, not a thread"


class TestMediaRefusalRespectsGovernance:
    """The refusal is produced in the transport, so it needs its own policy gate.

    The text path reaches ``inbound_permitted`` inside the dispatcher. This reply
    never gets there, so without a check of its own a host profile that denies
    WeCom after startup would still see the bot talk — the fail-open shape the
    governance ladder exists to prevent.
    """

    @pytest.mark.asyncio
    async def test_a_denied_channel_sends_no_refusal(self) -> None:
        client = FakeClient()
        t = WeComTransport(client, owner_id="Wei")
        with (
            patch("kiro_crew.wecom.transport.sel"),
            patch("kiro_crew.wecom.transport.inbound_permitted", return_value=False) as gate,
        ):
            await t.receive(
                WeComInbound(
                    userid="Wei", text="", msgtype="image", response_url="https://r", req_id="rq1"
                )
            )
        gate.assert_awaited_once_with("wecom")
        assert client.bubbles == [] and client.replies == []

    @pytest.mark.asyncio
    async def test_a_permitted_channel_still_refuses_out_loud(self) -> None:
        client = FakeClient()
        t = WeComTransport(client, owner_id="Wei")
        with (
            patch("kiro_crew.wecom.transport.sel"),
            patch("kiro_crew.wecom.transport.inbound_permitted", return_value=True),
        ):
            await t.receive(
                WeComInbound(
                    userid="Wei", text="", msgtype="image", response_url="https://r", req_id="rq1"
                )
            )
        assert len(client.bubbles) == 1


class TestGroupReachableMediaKind:
    """`mixed` is the only non-text kind a WeCom group can send, so it must refuse.

    The vendor marks image / voice / video / file as single-chat only. Leaving
    `mixed` off the label map put the one group-reachable attachment on the
    silent-drop branch — the defect this refusal exists to remove.
    """

    @pytest.mark.asyncio
    async def test_a_mixed_frame_is_refused_out_loud(self) -> None:
        client = FakeClient()
        t = WeComTransport(client, owner_id="Wei", allow_group_chats=True)
        with patch("kiro_crew.wecom.transport.sel") as sel:
            await t.receive(
                WeComInbound(
                    userid="Wei",
                    text="",
                    msgtype="mixed",
                    chatid="chat-42",
                    response_url="https://r",
                    req_id="rq1",
                )
            )
        assert len(client.bubbles) == 1
        assert "图文" in client.bubbles[0][1]
        assert sel().log_api_access.call_args.kwargs["outcome"] == "media_unsupported"
