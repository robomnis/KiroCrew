"""Wire-level tests for the WeCom channel.

Runs the REAL WeComClient against fakes for each of its wire surfaces:

* the inbound callback, a WS frame (``aibot_msg_callback``) carrying the
  server-assigned ``req_id`` and the single-use ``response_url``;
* the cmd-less ``errcode`` ACK, which is both the pong and the stream-reply
  receipt and must never be mistaken for a message;
* the one-shot reply, an HTTP POST to the inbound message's ``response_url``
  (whose embedded response_code is the ONLY credential -- single-use, 1h TTL);
* the streaming reply, a WS frame (``aibot_respond_msg``) whose routing depends
  entirely on echoing back the server-assigned ``req_id``.

All of it is frame parsing and request construction, so the dispatcher tests --
which replace the client wholesale -- never exercise any of it.

The vendor-side shapes are loaded from ``test/fixtures/channels/wecom/`` instead
of restated as literals, so every claim this file makes about WeCom's API is
attributed and auditable. Two things deliberately stay literals: what we SEND
(our own construction, not a vendor claim) and the 500/non-JSON case (an
infrastructure error page is not a claim about the API either).

Those fixtures are currently ``assumed`` -- nobody has captured a real WeCom
frame. See each file's ``_provenance``, and ``test_channel_wire_conformance.py``
for the opt-in lane that would upgrade them to ``live_probe``.
"""

from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from kiro_crew.testing.channel_fixtures import load_fixture
from kiro_crew.testing.fake_channel_wire import (
    FakeWireSession,
    FakeWireWebSocket,
    WireResponse,
)
from kiro_crew.wecom.client import WECOM_MAX_REPLY_BYTES, WeComClient, WeComInbound

CHANNEL_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "channels"

# Recorded WeCom wire shapes. Sourcing them here instead of restating literals is
# what makes a vendor change a one-file edit: rewrite the fixture and every layer
# above is re-verified against the new shape.
_CALLBACK = load_fixture("wecom", "msg_callback_text", root=CHANNEL_FIXTURES).payload
_ACK_OK = load_fixture("wecom", "cmdless_ack", root=CHANNEL_FIXTURES).payload
_ACK_ERROR = load_fixture("wecom", "cmdless_ack_error", root=CHANNEL_FIXTURES).payload
_REPLY_OK = load_fixture("wecom", "reply_ok", root=CHANNEL_FIXTURES).payload

# The reply target is the URL the recorded inbound frame carries, not one this
# file invents: the response_code embedded in it is the whole credential, so
# reusing it is what shows it survived the frame -> reply round trip.
_RESPONSE_URL = _CALLBACK["body"]["response_url"]
_REPLY_PATH = urlsplit(_RESPONSE_URL).path


def _client(wire: FakeWireSession | None = None) -> WeComClient:
    client = WeComClient(bot_id="bot-1", secret="s", ws_url="wss://example.invalid/ws")
    if wire is not None:
        client._session = wire
    return client


def _wire() -> FakeWireSession:
    """A session answering the reply endpoint with the recorded success body.

    The body is copied: the fixture payload is module-level state shared by every
    test in the file, and a route target is stored by reference.
    """
    return FakeWireSession().route("POST", _REPLY_PATH, dict(_REPLY_OK))


async def _deliver(client: WeComClient, frame: dict) -> None:
    """Feed a raw inbound frame in and await whatever turn it dispatches.

    ``_dispatch_callback`` deliberately runs the turn as a background task so the
    receive loop keeps reading, so an assertion made straight after the call
    races the event loop. Awaiting the tracked tasks is the deterministic
    synchronisation point; a sleep is the flaky spelling of it.
    """
    await client._handle_message(json.dumps(frame, ensure_ascii=False))
    for task in list(client._handler_tasks):
        await task


class TestInboundCallbackFrame:
    """The recorded ``aibot_msg_callback``, replayed through the real parser.

    The dispatcher tests hand the transport a ready-made ``WeComInbound``, so the
    frame -> dataclass step is only covered here -- and it is the step where a
    renamed vendor field costs a turn its userid or its reply address without
    anything raising.
    """

    def test_the_recorded_frame_parses_into_the_fields_a_turn_needs(self) -> None:
        seen: list[WeComInbound] = []

        async def _turn() -> None:
            client = _client()

            async def _on_message(msg: WeComInbound) -> None:
                seen.append(msg)

            client.set_message_handler(_on_message)
            await _deliver(client, _CALLBACK)

        asyncio.run(_turn())

        assert len(seen) == 1
        msg = seen[0]
        body = _CALLBACK["body"]
        assert msg.req_id == _CALLBACK["headers"]["req_id"]
        assert msg.req_id, "a frame with no req_id cannot be streamed back to at all"
        assert msg.userid == body["from"]["userid"]
        assert msg.text == body["text"]["content"]
        assert msg.response_url == body["response_url"]
        assert msg.chatid == body["chatid"]
        assert msg.msgtype == body["msgtype"]

    def test_the_reply_goes_back_to_the_url_the_frame_carried(self) -> None:
        """The single-use response_url is carried, never reconstructed.

        Its embedded response_code is the only credential the reply has, so a
        client that rebuilt the URL from a base path would post to an endpoint it
        cannot authenticate against.
        """
        wire = _wire()

        async def _turn() -> None:
            client = _client(wire)

            async def _on_message(msg: WeComInbound) -> None:
                await client.send_reply(msg.response_url, "ok")

            client.set_message_handler(_on_message)
            await _deliver(client, _CALLBACK)

        asyncio.run(_turn())

        assert [r.url for r in wire.requests] == [_CALLBACK["body"]["response_url"]]


class TestCmdLessAckFrames:
    """``errcode`` present with no ``cmd``: a pong, or a stream-reply receipt.

    One shape serves both, and the ONLY discriminator is whether the echoed
    req_id belongs to a ping we sent. Either way it is a receipt, so a cmd-less
    frame that reached the callback path would spawn a turn out of our own
    delivery confirmation.
    """

    def _absorb(self, frame: dict) -> list[WeComInbound]:
        seen: list[WeComInbound] = []

        async def _turn() -> None:
            client = _client()

            async def _on_message(msg: WeComInbound) -> None:
                seen.append(msg)

            client.set_message_handler(_on_message)
            await _deliver(client, frame)

        asyncio.run(_turn())
        return seen

    def test_a_stream_receipt_is_absorbed_without_dispatching_a_turn(self) -> None:
        assert self._absorb(_ACK_OK) == []

    def test_an_error_receipt_is_absorbed_rather_than_raised_or_dispatched(self) -> None:
        # A raise here would propagate out of the receive loop and drop the
        # connection over a reply WeCom merely refused.
        assert self._absorb(_ACK_ERROR) == []

    def test_the_same_shape_clears_a_pending_pong_when_the_req_id_was_a_ping(self) -> None:
        """The liveness half of the branch: three unanswered pings close the WS.

        So a pong misread as a stream receipt leaves the counter climbing and the
        client tears down a perfectly healthy connection every 90 seconds.
        """
        ping_id = _ACK_OK["headers"]["req_id"]

        async def _turn() -> WeComClient:
            client = _client()
            client._ping_reqs.add(ping_id)
            client._pending_pongs = 1
            await _deliver(client, _ACK_OK)
            return client

        client = asyncio.run(_turn())

        assert client._pending_pongs == 0
        assert ping_id not in client._ping_reqs


class TestOneShotReply:
    def test_the_reply_posts_markdown_to_the_response_url(self) -> None:
        wire = _wire()
        client = _client(wire)

        asyncio.run(client.send_reply(_RESPONSE_URL, "**hi**"))

        req = wire.requests[0]
        assert req.method == "POST"
        assert req.url == _RESPONSE_URL, "the response_url must be used verbatim"
        # The request body is OUR construction, not a vendor claim, so it stays a
        # literal here -- a fixture would only restate what this file decides.
        assert req.json_body == {"msgtype": "markdown", "markdown": {"content": "**hi**"}}

    def test_empty_content_is_replaced_so_the_api_never_400s(self) -> None:
        wire = _wire()
        client = _client(wire)

        asyncio.run(client.send_reply(_RESPONSE_URL, ""))

        assert wire.requests[0].json_body["markdown"]["content"]

    def test_overlong_content_is_truncated_before_sending(self) -> None:
        wire = _wire()
        client = _client(wire)

        asyncio.run(client.send_reply(_RESPONSE_URL, "x" * (WECOM_MAX_REPLY_BYTES + 500)))

        sent = wire.requests[0].json_body["markdown"]["content"]
        assert (
            len(sent.encode("utf-8")) == WECOM_MAX_REPLY_BYTES
        ), "the client must cap at the documented limit, not let the API reject it"

    def test_the_cap_is_bytes_not_characters(self) -> None:
        """A CJK reply is ~3 bytes per character, so a char cap lets it through.

        The guard is what stops WeCom refusing the frame; measuring it in
        characters is the defect, because the send then looks fine locally and
        comes back as a non-zero errcode with the user seeing nothing.
        """
        wire = _wire()
        client = _client(wire)

        asyncio.run(client.send_reply(_RESPONSE_URL, "字" * WECOM_MAX_REPLY_BYTES))

        sent = wire.requests[0].json_body["markdown"]["content"]
        assert len(sent.encode("utf-8")) <= WECOM_MAX_REPLY_BYTES
        assert len(sent) < WECOM_MAX_REPLY_BYTES, "a byte cap must bite before the char count"

    def test_a_missing_response_url_sends_nothing(self) -> None:
        wire = FakeWireSession()
        client = _client(wire)

        asyncio.run(client.send_reply("", "hi"))

        assert wire.requests == []

    def test_an_http_error_does_not_raise_out_of_the_reply_path(self) -> None:
        # A raising ack would abort the turn after the model already answered.
        wire = FakeWireSession().route("POST", _REPLY_PATH, WireResponse(body="nope", status=500))
        client = _client(wire)

        asyncio.run(client.send_reply(_RESPONSE_URL, "hi"))  # must not raise


class TestStreamFrames:
    """The WS frame shape -- routing lives entirely in the echoed req_id."""

    def test_the_frame_carries_the_command_req_id_and_stream_body(self) -> None:
        ws = FakeWireWebSocket()
        client = _client()
        client._ws = ws  # type: ignore[assignment]

        ok = asyncio.run(client.send_stream("REQ-1", "STREAM-1", "partial", finish=False))

        assert ok is True
        frame = json.loads(ws.sent[0])
        assert frame["cmd"] == "aibot_respond_msg"
        assert (
            frame["headers"]["req_id"] == "REQ-1"
        ), "req_id is the ONLY routing key -- a wrong one silently delivers nowhere"
        assert frame["body"]["msgtype"] == "stream"
        assert frame["body"]["stream"] == {
            "id": "STREAM-1",
            "finish": False,
            "content": "partial",
        }

    def test_finish_locks_the_bubble(self) -> None:
        ws = FakeWireWebSocket()
        client = _client()
        client._ws = ws  # type: ignore[assignment]

        asyncio.run(client.send_stream("REQ-1", "STREAM-1", "done", finish=True))

        frame = json.loads(ws.sent[0])
        assert frame["body"]["stream"]["finish"] is True

    def test_frames_carry_the_full_accumulated_text_not_a_delta(self) -> None:
        ws = FakeWireWebSocket()
        client = _client()
        client._ws = ws  # type: ignore[assignment]

        asyncio.run(client.send_stream("REQ-1", "S", "Hel", finish=False))
        asyncio.run(client.send_stream("REQ-1", "S", "Hello", finish=True))

        contents = [json.loads(f)["body"]["stream"]["content"] for f in ws.sent]
        assert contents == [
            "Hel",
            "Hello",
        ], "each frame REPLACES the bubble, so a delta would render truncated text"

    def test_no_live_socket_reports_failure_instead_of_raising(self) -> None:
        client = _client()
        client._ws = None

        assert asyncio.run(client.send_stream("REQ-1", "S", "hi", finish=True)) is False

    def test_a_missing_req_id_reports_failure(self) -> None:
        ws = FakeWireWebSocket()
        client = _client()
        client._ws = ws  # type: ignore[assignment]

        assert asyncio.run(client.send_stream("", "S", "hi", finish=True)) is False
        assert ws.sent == [], "an unroutable frame must not be put on the wire"

    def test_a_closed_socket_reports_failure(self) -> None:
        ws = FakeWireWebSocket()
        ws.closed = True
        client = _client()
        client._ws = ws  # type: ignore[assignment]

        assert asyncio.run(client.send_stream("REQ-1", "S", "hi", finish=True)) is False


class TestWireFieldsAreCoercedToStrings:
    """A JSON list or object in any wire field must not raise out of receive.

    Both ``userid`` and ``chatid`` reach ``frozenset`` membership in the transport
    (``authorize`` and ``_group_permitted``), where an unhashable value raises
    ``TypeError`` and the message is dropped with no deny recorded. A malformed
    frame has to degrade to a DENY, which is what the empty string produces.
    """

    @staticmethod
    def _frame(**overrides: object) -> dict:
        # Built FROM the recorded fixture rather than hand-rolled, so a change to
        # the real envelope cannot leave these tests passing against a shape the
        # parser no longer accepts.
        frame = copy.deepcopy(_CALLBACK)
        frame["body"].update(overrides)
        return frame

    def _parse(self, **overrides: object) -> WeComInbound:
        seen: list[WeComInbound] = []

        async def _turn() -> None:
            client = _client()

            async def _on_message(msg: WeComInbound) -> None:
                seen.append(msg)

            client.set_message_handler(_on_message)
            await _deliver(client, self._frame(**overrides))

        asyncio.run(_turn())
        assert seen, "the frame must still be delivered, not lost to a TypeError"
        return seen[0]

    @pytest.mark.parametrize("junk", [{"a": 1}, ["x"], 7, None, True])
    def test_a_non_string_chatid_becomes_empty(self, junk: object) -> None:
        assert self._parse(chatid=junk).chatid == ""

    @pytest.mark.parametrize("junk", [{"a": 1}, ["x"], 7, None])
    def test_a_non_string_userid_becomes_empty_and_is_therefore_denied(self, junk: object) -> None:
        from kiro_crew.messaging.transport import InboundMessage
        from kiro_crew.wecom.transport import WeComTransport

        assert self._parse(**{"from": {"userid": junk}}).userid == ""
        # And an empty userid is denied even under allow_all, so the coercion lands
        # on the deny side rather than authorizing an anonymous frame.
        transport = WeComTransport(_client(), allow_all=True)
        assert not transport.authorize(
            InboundMessage(channel_type="wecom", user_id="", conversation_id="", text="hi")
        )

    @pytest.mark.parametrize("field", ["response_url", "msgtype", "text"])
    def test_the_remaining_wire_strings_are_coerced_too(self, field: str) -> None:
        if field == "text":
            # text is a nested object, so the junk goes on its content field.
            assert self._parse(text={"content": ["a", "b"]}).text == ""
            return
        assert getattr(self._parse(**{field: {"nested": "object"}}), field) == ""

    def test_a_non_string_req_id_becomes_empty_so_the_renderer_skips_streaming(self) -> None:
        # req_id is the stream correlation key. Empty is the renderer's documented
        # "no stream" signal, which falls back to the one-shot response_url -- a
        # degraded reply rather than frames addressed to a nonsense id.
        frame = self._frame()
        frame["headers"]["req_id"] = {"not": "a string"}
        seen: list[WeComInbound] = []

        async def _turn() -> None:
            client = _client()

            async def _on_message(msg: WeComInbound) -> None:
                seen.append(msg)

            client.set_message_handler(_on_message)
            await _deliver(client, frame)

        asyncio.run(_turn())
        assert seen and seen[0].req_id == ""
