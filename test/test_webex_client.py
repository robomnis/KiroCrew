"""Tests for kiro_crew.webex.client (WebexClient, low-level layer)."""

from __future__ import annotations

import asyncio
import base64
from typing import Any

import pytest

from kiro_crew.messaging.split import chunk_utf8
from kiro_crew.webex.client import WEBEX_MAX_TEXT, WebexClient, WebexInbound, hydra_id

# ``truncate_utf8`` and ``chunk_utf8`` are shared by every byte-capped channel and
# live in ``messaging.split``, where test_messaging_split.py pins their contracts.
# What stays Webex's own concern is that its sends apply them at WEBEX_MAX_TEXT,
# which the cases below pin.


class TestChunkUtf8:
    def test_empty_returns_empty(self) -> None:
        assert chunk_utf8("", WEBEX_MAX_TEXT) == []

    def test_under_cap_single_chunk(self) -> None:
        assert chunk_utf8("hello", WEBEX_MAX_TEXT) == ["hello"]

    def test_lossless_multibyte_split(self) -> None:
        # 3000 4-byte emoji = 12000 bytes: must split, never drop content.
        text = "🐾" * 3000
        chunks = chunk_utf8(text, WEBEX_MAX_TEXT)
        assert len(chunks) > 1
        assert "".join(chunks) == text  # lossless
        for c in chunks:
            assert len(c.encode("utf-8")) <= WEBEX_MAX_TEXT
            assert "\ufffd" not in c

    def test_lossless_ascii_split(self) -> None:
        text = "x" * (WEBEX_MAX_TEXT + 100)
        chunks = chunk_utf8(text, WEBEX_MAX_TEXT)
        assert chunks == ["x" * WEBEX_MAX_TEXT, "x" * 100]

    def test_mixed_content_boundary(self) -> None:
        # ASCII prefix pushes an emoji across the byte boundary — the split
        # must move the whole code point into the next chunk.
        text = "a" * (WEBEX_MAX_TEXT - 2) + "🐾🐾"
        chunks = chunk_utf8(text, WEBEX_MAX_TEXT)
        assert "".join(chunks) == text
        for c in chunks:
            assert len(c.encode("utf-8")) <= WEBEX_MAX_TEXT


class TestReadyState:
    def test_ready_starts_unset(self) -> None:
        c = _client()
        assert not c.ready.is_set()

    @pytest.mark.asyncio
    async def test_wait_ready_times_out_when_never_connected(self) -> None:
        c = _client()
        assert await c.wait_ready(timeout=0.05) is False

    @pytest.mark.asyncio
    async def test_wait_ready_returns_true_once_set(self) -> None:
        c = _client()
        c.ready.set()
        assert await c.wait_ready(timeout=0.05) is True

    def test_notify_state_calls_observer_and_swallows_errors(self) -> None:
        c = _client()
        seen: list[tuple[bool, str]] = []
        c.on_state_change = lambda ok, err: seen.append((ok, err))
        c._notify_state(True, "")
        c._notify_state(False, "boom")
        assert seen == [(True, ""), (False, "boom")]

        def _raiser(ok: bool, err: str) -> None:
            raise RuntimeError("observer bug")

        c.on_state_change = _raiser
        c._notify_state(True, "")  # must not raise


class TestHydraId:
    def test_encodes_message_id(self) -> None:
        raw = "9ba21fc0-1234-11ee-a1b2-abcdefabcdef"
        encoded = hydra_id(raw, "MESSAGE")
        decoded = base64.b64decode(encoded + "=" * (-len(encoded) % 4)).decode()
        assert decoded == f"ciscospark://us/MESSAGE/{raw}"

    def test_no_padding(self) -> None:
        assert "=" not in hydra_id("abc", "MESSAGE")

    def test_matches_webex_documented_id_format(self) -> None:
        """Webex issues UNPADDED base64 ids — this is the documented example
        message id from the Webex API reference, and our encoding of its
        decoded URI must reproduce it byte-for-byte. Note also that a
        ``ciscospark://us/MESSAGE/{uuid}`` URI is exactly 60 bytes (24-byte
        prefix + 36-byte canonical UUID), which is divisible by 3, so its
        base64 encoding NEVER carries padding — ``rstrip("=")`` is a no-op
        for every real message event and exists only as defense for
        non-canonical id shapes."""
        documented = (
            "Y2lzY29zcGFyazovL3VzL01FU1NBR0UvOTJkYjNiZTAtNDNiZC0xMWU2" "LThhZTktZGQ1YjNkZmM1NjVk"
        )
        assert hydra_id("92db3be0-43bd-11e6-8ae9-dd5b3dfc565d", "MESSAGE") == documented

    def test_uuid_uris_never_generate_padding(self) -> None:
        import base64
        import uuid

        for rtype in ("MESSAGE", "ROOM"):
            uri = f"ciscospark://us/{rtype}/{uuid.uuid4()}"
            assert len(uri) % 3 == 0  # divisible by 3 -> base64 never pads
            assert not base64.b64encode(uri.encode()).decode().endswith("=")

    def test_empty_returns_empty(self) -> None:
        assert hydra_id("", "MESSAGE") == ""


def _client(**kw: Any) -> WebexClient:
    return WebexClient(token="tok", **kw)


def _frame(
    verb: str = "post",
    actor_email: str = "user@example.com",
    raw_id: str = "raw-uuid",
    event_type: str = "conversation.activity",
) -> dict:
    return {
        "data": {
            "eventType": event_type,
            "activity": {
                "verb": verb,
                "actor": {"emailAddress": actor_email},
                "object": {"id": raw_id},
                "target": {"id": "room-uuid"},
            },
        }
    }


class TestHandleFrame:
    @pytest.mark.asyncio
    async def test_post_activity_dispatches(self, monkeypatch: pytest.MonkeyPatch) -> None:
        c = _client()
        fetched: list[str] = []

        async def fake_fetch(mid: str) -> dict:
            fetched.append(mid)
            return {
                "personEmail": "User@Example.com",
                "roomId": "ROOM",
                "text": "hello",
                "personId": "P1",
                "roomType": "direct",
            }

        received: list[WebexInbound] = []

        async def on_message(inbound: WebexInbound) -> None:
            received.append(inbound)

        monkeypatch.setattr(c, "fetch_message", fake_fetch)
        c.set_message_handler(on_message)
        c._handle_frame(_frame())
        await asyncio.gather(*c._handler_tasks)

        assert fetched == [hydra_id("raw-uuid", "MESSAGE")]
        assert len(received) == 1
        assert received[0].person_email == "user@example.com"  # lowercased
        assert received[0].room_id == "ROOM"
        assert received[0].room_type == "direct"

    @pytest.mark.asyncio
    async def test_self_message_skipped_by_actor_email(self) -> None:
        c = _client()
        c.bot_email = "bot@webex.bot"
        c._handle_frame(_frame(actor_email="Bot@Webex.Bot".lower()))
        assert c._handler_tasks == set()

    @pytest.mark.asyncio
    async def test_self_message_skipped_by_person_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        c = _client()
        c.bot_person_id = "BOT_PID"

        async def fake_fetch(mid: str) -> dict:
            return {"personId": "BOT_PID", "personEmail": "x@y.z", "roomId": "R", "text": "t"}

        received: list[WebexInbound] = []

        async def on_message(inbound: WebexInbound) -> None:
            received.append(inbound)

        monkeypatch.setattr(c, "fetch_message", fake_fetch)
        c.set_message_handler(on_message)
        c._handle_frame(_frame())
        await asyncio.gather(*c._handler_tasks)
        assert received == []

    def test_non_post_verb_ignored(self) -> None:
        c = _client()
        c._handle_frame(_frame(verb="add"))
        assert c._handler_tasks == set()

    def test_other_event_type_ignored(self) -> None:
        c = _client()
        c._handle_frame(_frame(event_type="apheleia.subscription_update"))
        assert c._handler_tasks == set()

    def test_malformed_frames_ignored(self) -> None:
        c = _client()
        c._handle_frame("not a dict")
        c._handle_frame({"data": "not a dict"})
        c._handle_frame({"data": {"eventType": "conversation.activity", "activity": None}})
        assert c._handler_tasks == set()


class TestOutbound:
    @pytest.mark.asyncio
    async def test_send_to_room_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        c = _client()
        calls: list[tuple[str, str, dict | None]] = []

        async def fake_api(method: str, path: str, payload: dict | None, timeout: int = 30):
            calls.append((method, path, payload))
            return {"id": "MSG1"}

        monkeypatch.setattr(c, "_api", fake_api)
        mid = await c.send_message("ROOMID", "hi")
        assert mid == "MSG1"
        method, path, payload = calls[0]
        assert (method, path) == ("POST", "/messages")
        assert payload == {"markdown": "hi", "roomId": "ROOMID"}

    @pytest.mark.asyncio
    async def test_send_to_email_uses_to_person_email(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        c = _client()
        calls: list[dict | None] = []

        async def fake_api(method: str, path: str, payload: dict | None, timeout: int = 30):
            calls.append(payload)
            return {"id": "MSG1"}

        monkeypatch.setattr(c, "_api", fake_api)
        await c.send_message("user@example.com", "hi")
        assert calls[0] == {"markdown": "hi", "toPersonEmail": "user@example.com"}

    @pytest.mark.asyncio
    async def test_send_truncates_to_cap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        c = _client()
        seen: list[str] = []

        async def fake_api(method: str, path: str, payload: dict | None, timeout: int = 30):
            assert payload is not None
            seen.append(payload["markdown"])
            return {"id": "M"}

        monkeypatch.setattr(c, "_api", fake_api)
        await c.send_message("R", "x" * (WEBEX_MAX_TEXT + 500))
        assert len(seen[0]) == WEBEX_MAX_TEXT

    @pytest.mark.asyncio
    async def test_edit_message_payload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        c = _client()
        calls: list[tuple[str, str, dict | None]] = []

        async def fake_api(method: str, path: str, payload: dict | None, timeout: int = 30):
            calls.append((method, path, payload))
            return {}

        monkeypatch.setattr(c, "_api", fake_api)
        ok = await c.edit_message("MSG1", "ROOM", "new text")
        assert ok is True
        method, path, payload = calls[0]
        assert (method, path) == ("PUT", "/messages/MSG1")
        assert payload == {"roomId": "ROOM", "markdown": "new text"}

    @pytest.mark.asyncio
    async def test_edit_failure_returns_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        c = _client()

        async def fake_api(method: str, path: str, payload: dict | None, timeout: int = 30):
            return None  # e.g. 400 edit-limit reached

        monkeypatch.setattr(c, "_api", fake_api)
        assert await c.edit_message("MSG1", "ROOM", "text") is False


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_close_drains_handlers_and_blocks_new_session(self) -> None:
        c = _client()
        await c.close()
        # Handler tasks drained.
        assert c._handler_tasks == set()
        # A subsequent session acquisition must fail closed.
        with pytest.raises(RuntimeError, match="WebexClient is closed"):
            await c._ensure_session()

    @pytest.mark.asyncio
    async def test_close_cancels_in_flight_handler(self) -> None:
        c = _client()
        # Inject a never-completing handler task, mirroring how _handle_frame
        # tracks live turn tasks.
        task: asyncio.Task[Any] = asyncio.ensure_future(asyncio.sleep(100))
        c._handler_tasks.add(task)
        await c.close()
        assert task.done()
        assert task.cancelled()
        assert c._handler_tasks == set()


async def _noop_sleep(d: float) -> None:
    return None


class _FakeResponse:
    """Minimal ``aiohttp`` response usable as an async context manager."""

    def __init__(self, status: int, body: Any = None, headers: dict | None = None) -> None:
        self.status = status
        self._body = body
        self.headers = headers or {}

    async def __aenter__(self) -> "_FakeResponse":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def json(self, content_type: Any = None) -> Any:
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class _FakeSession:
    """Records calls and replays a queued response per method."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, str]] = []
        self.posts: list[str] = []
        self.gets: list[str] = []
        self.request_results: list[Any] = []
        self.post_results: list[Any] = []
        self.get_results: list[Any] = []
        self.closed = False

    @staticmethod
    def _next(queue: list[Any]) -> Any:
        result = queue.pop(0) if queue else _FakeResponse(200, {})
        if isinstance(result, Exception):
            raise result
        return result

    def request(self, method: str, url: str, **kw: Any) -> Any:
        self.requests.append((method, url))
        return self._next(self.request_results)

    def post(self, url: str, **kw: Any) -> Any:
        self.posts.append(url)
        return self._next(self.post_results)

    def get(self, url: str, **kw: Any) -> Any:
        self.gets.append(url)
        return self._next(self.get_results)


def _with_session(c: WebexClient, session: _FakeSession, monkeypatch: pytest.MonkeyPatch) -> None:
    async def ensure() -> Any:
        return session

    monkeypatch.setattr(c, "_ensure_session", ensure)


class TestIdentity:
    """The bot's own identity is what makes self-message filtering possible.

    An unresolved or wrongly-cased ``bot_email`` makes the echo filter in
    ``_handle_frame`` miss, so the bot answers its own messages in a loop.
    """

    @pytest.mark.asyncio
    async def test_the_first_email_is_stored_lowercased(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        c = _client()

        async def fake_api(method: str, path: str, payload: dict | None, timeout: int = 30):
            assert (method, path) == ("GET", "/people/me")
            return {"emails": ["Bot@Example.COM", "alt@example.com"], "id": "PERSON1"}

        monkeypatch.setattr(c, "_api", fake_api)
        await c._fetch_me()
        # Lowercased because the WS actor email is compared against it directly.
        assert c.bot_email == "bot@example.com"
        assert c.bot_person_id == "PERSON1"

    @pytest.mark.asyncio
    async def test_an_empty_email_list_leaves_the_field_empty_rather_than_raising(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        c = _client()

        async def fake_api(method: str, path: str, payload: dict | None, timeout: int = 30):
            return {"emails": [], "id": "P"}

        monkeypatch.setattr(c, "_api", fake_api)
        await c._fetch_me()
        assert c.bot_email == ""
        # bot_person_id still resolves, and _hydrate_and_dispatch's belt-and-braces
        # filter is what keeps the echo guard working without the email.
        assert c.bot_person_id == "P"

    @pytest.mark.asyncio
    async def test_a_failed_call_leaves_identity_untouched(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        c = _client()
        c.bot_email = "keep@example.com"

        async def fake_api(method: str, path: str, payload: dict | None, timeout: int = 30):
            return None  # transport error / non-2xx

        monkeypatch.setattr(c, "_api", fake_api)
        await c._fetch_me()
        assert c.bot_email == "keep@example.com"


class TestDeviceRegistration:
    """WDM device registration is the only source of a WebSocket URL.

    The device cap is the interesting case: an account that has registered its
    maximum gets a non-2xx on POST, and reusing an EXISTING device is what keeps
    the channel working instead of failing every reconnect forever.
    """

    @pytest.mark.asyncio
    async def test_a_fresh_registration_returns_the_socket_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        c = _client()
        s = _FakeSession()
        s.post_results = [_FakeResponse(201, {"webSocketUrl": "wss://x/1"})]
        _with_session(c, s, monkeypatch)
        assert await c._get_websocket_url() == "wss://x/1"
        assert not s.gets, "a successful POST must not also list devices"

    @pytest.mark.asyncio
    async def test_the_device_cap_falls_back_to_reusing_the_first_device(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        c = _client()
        s = _FakeSession()
        s.post_results = [_FakeResponse(409, {})]
        s.get_results = [
            _FakeResponse(
                200, {"devices": [{"webSocketUrl": "wss://x/reused"}, {"webSocketUrl": "b"}]}
            )
        ]
        _with_session(c, s, monkeypatch)
        assert await c._get_websocket_url() == "wss://x/reused"

    @pytest.mark.asyncio
    async def test_no_devices_to_reuse_returns_empty_rather_than_raising(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        c = _client()
        s = _FakeSession()
        s.post_results = [_FakeResponse(409, {})]
        s.get_results = [_FakeResponse(200, {"devices": []})]
        _with_session(c, s, monkeypatch)
        # Empty, not an exception: _connect_and_serve turns it into a ClientError
        # so _run_loop's backoff owns the retry.
        assert await c._get_websocket_url() == ""

    @pytest.mark.asyncio
    async def test_a_transport_error_logs_only_the_exception_type(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        import aiohttp

        c = WebexClient(token="SUPERSECRET")
        s = _FakeSession()
        s.post_results = [aiohttp.ClientError("connect to https://wdm/devices?token=SUPERSECRET")]
        _with_session(c, s, monkeypatch)
        with caplog.at_level("WARNING"):
            assert await c._get_websocket_url() == ""
        # The URL and response body are token-adjacent: only the type is logged.
        assert "ClientError" in caplog.text
        assert "SUPERSECRET" not in caplog.text


class TestReconnectBackoff:
    """The reconnect loop must not hot-loop on a connection that never works.

    A bad token produces a connection the server accepts and closes immediately,
    which reads as a CLEAN close. Resetting the backoff on every clean close would
    then spin the loop at full rate against the API forever, so the reset is gated
    on the connection having actually LIVED (``_MIN_HEALTHY_CONN_SECS``).
    """

    @staticmethod
    def _stub_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
        delays: list[float] = []

        async def fake_sleep(d: float) -> None:
            delays.append(d)

        monkeypatch.setattr("kiro_crew.webex.client.asyncio.sleep", fake_sleep)
        return delays

    @pytest.mark.asyncio
    async def test_the_delay_doubles_and_caps_at_sixty_seconds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import aiohttp

        c = _client()
        delays = self._stub_sleep(monkeypatch)
        attempts = {"n": 0}

        async def failing() -> None:
            attempts["n"] += 1
            if attempts["n"] >= 9:
                c._closed = True
            raise aiohttp.ClientError("nope")

        monkeypatch.setattr(c, "_connect_and_serve", failing)
        await c._run_loop()
        assert delays[:5] == [1.0, 2.0, 4.0, 8.0, 16.0]
        assert max(delays) == 60.0, "the backoff must saturate rather than grow without bound"

    @pytest.mark.asyncio
    async def test_an_immediate_clean_close_still_backs_off(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        c = _client()
        delays = self._stub_sleep(monkeypatch)
        attempts = {"n": 0}

        async def instant_clean_close() -> None:
            attempts["n"] += 1
            if attempts["n"] >= 3:
                c._closed = True

        monkeypatch.setattr(c, "_connect_and_serve", instant_clean_close)
        await c._run_loop()
        # Growing delays prove the attempt counter was NOT reset -- the hot-loop
        # guard. A bad token lands here, and this is what keeps it off the API.
        assert delays == [1.0, 2.0]
        assert "closed connection immediately" in c.last_error

    @pytest.mark.asyncio
    async def test_a_healthy_connection_resets_the_backoff(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from kiro_crew.webex import client as client_mod

        c = _client()
        delays = self._stub_sleep(monkeypatch)
        clock = {"t": 0.0}
        monkeypatch.setattr(client_mod.time, "monotonic", lambda: clock["t"])
        attempts = {"n": 0}

        async def serve() -> None:
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise OSError("first failure arms the backoff")
            if attempts["n"] == 2:
                clock["t"] += client_mod._MIN_HEALTHY_CONN_SECS + 1  # a connection that LIVED
                return
            if attempts["n"] == 3:
                raise OSError("second failure")
            c._closed = True  # stop the loop on a clean pass, so attempt 3 slept

        monkeypatch.setattr(c, "_connect_and_serve", serve)
        await c._run_loop()
        # Back to 1.0 rather than 2.0: the healthy connection cleared the counter.
        assert delays == [1.0, 1.0]

    @pytest.mark.asyncio
    async def test_cancellation_exits_without_sleeping(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        c = _client()
        delays = self._stub_sleep(monkeypatch)

        async def cancelled() -> None:
            raise asyncio.CancelledError

        monkeypatch.setattr(c, "_connect_and_serve", cancelled)
        await c._run_loop()
        assert delays == [], "a cancelled loop must not schedule a reconnect"

    @pytest.mark.asyncio
    async def test_an_unexpected_error_is_reported_not_swallowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        c = _client()
        self._stub_sleep(monkeypatch)
        seen: list[tuple[bool, str]] = []
        c.on_state_change = lambda ok, err: seen.append((ok, err))
        attempts = {"n": 0}

        async def boom() -> None:
            attempts["n"] += 1
            c._closed = attempts["n"] >= 2
            raise ZeroDivisionError("not an aiohttp error")

        monkeypatch.setattr(c, "_connect_and_serve", boom)
        await c._run_loop()
        # The dashboard badge must learn about it -- a swallowed unexpected error
        # leaves the channel showing "connected" while nothing is served.
        assert seen and seen[0][0] is False
        assert "unexpected error" in c.last_error


class TestApiRetry:
    @pytest.mark.asyncio
    async def test_a_429_is_retried_once_after_the_retry_after_delay(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        c = _client()
        s = _FakeSession()
        s.request_results = [
            _FakeResponse(429, {}, {"Retry-After": "2"}),
            _FakeResponse(200, {"id": "M"}),
        ]
        _with_session(c, s, monkeypatch)
        slept: list[float] = []

        async def fake_sleep(d: float) -> None:
            slept.append(d)

        monkeypatch.setattr("kiro_crew.webex.client.asyncio.sleep", fake_sleep)
        assert await c._api("POST", "/messages", {"markdown": "x"}) == {"id": "M"}
        assert slept == [2.0]
        assert len(s.requests) == 2

    @pytest.mark.asyncio
    async def test_a_second_429_gives_up_rather_than_looping(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        c = _client()
        s = _FakeSession()
        s.request_results = [
            _FakeResponse(429, {}, {"Retry-After": "1"}),
            _FakeResponse(429, {}, {"Retry-After": "1"}),
        ]
        _with_session(c, s, monkeypatch)
        monkeypatch.setattr("kiro_crew.webex.client.asyncio.sleep", _noop_sleep)
        assert await c._api("GET", "/x", None) is None
        assert len(s.requests) == 2, "exactly one retry -- the back-off is not a loop"

    @pytest.mark.asyncio
    async def test_an_unparseable_retry_after_falls_back_to_a_bounded_delay(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        c = _client()
        s = _FakeSession()
        s.request_results = [
            _FakeResponse(429, {}, {"Retry-After": "tomorrow"}),
            _FakeResponse(200, {}),
        ]
        _with_session(c, s, monkeypatch)
        slept: list[float] = []

        async def fake_sleep(d: float) -> None:
            slept.append(d)

        monkeypatch.setattr("kiro_crew.webex.client.asyncio.sleep", fake_sleep)
        await c._api("GET", "/x", None)
        assert slept == [1.0]

    @pytest.mark.asyncio
    async def test_an_absurd_retry_after_is_clamped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        c = _client()
        s = _FakeSession()
        s.request_results = [
            _FakeResponse(429, {}, {"Retry-After": "86400"}),
            _FakeResponse(200, {}),
        ]
        _with_session(c, s, monkeypatch)
        slept: list[float] = []

        async def fake_sleep(d: float) -> None:
            slept.append(d)

        monkeypatch.setattr("kiro_crew.webex.client.asyncio.sleep", fake_sleep)
        await c._api("GET", "/x", None)
        # Clamped: a turn must not hang for a day because a header said so.
        assert slept == [10.0]

    @pytest.mark.asyncio
    async def test_a_204_becomes_an_empty_dict_not_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        c = _client()
        s = _FakeSession()
        s.request_results = [_FakeResponse(204, None)]
        _with_session(c, s, monkeypatch)
        # {} and None mean different things to every caller: {} is success with no
        # body, None is failure. A 204 delete must not read as a failed call.
        assert await c._api("DELETE", "/messages/M", None) == {}

    @pytest.mark.asyncio
    async def test_a_2xx_with_a_junk_body_is_still_a_success(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        c = _client()
        s = _FakeSession()
        s.request_results = [_FakeResponse(200, ValueError("not json"))]
        _with_session(c, s, monkeypatch)
        assert await c._api("GET", "/x", None) == {}

    @pytest.mark.asyncio
    async def test_a_non_2xx_logs_the_status_and_never_the_body(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        c = _client()
        s = _FakeSession()
        s.request_results = [_FakeResponse(403, {"message": "token AKIAIOSFODNN7EXAMPLE denied"})]
        _with_session(c, s, monkeypatch)
        with caplog.at_level("WARNING"):
            assert await c._api("GET", "/people/me", None) is None
        assert "http=403" in caplog.text
        # Response bodies are externally-derived and may quote credentials back.
        assert "AKIAIOSFODNN7EXAMPLE" not in caplog.text

    @pytest.mark.asyncio
    async def test_a_transport_error_returns_none_and_logs_only_the_type(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        c = _client()
        s = _FakeSession()
        s.request_results = [asyncio.TimeoutError()]
        _with_session(c, s, monkeypatch)
        with caplog.at_level("WARNING"):
            assert await c._api("GET", "/x", None) is None
        assert "TimeoutError" in caplog.text


class TestSessionReuse:
    @pytest.mark.asyncio
    async def test_concurrent_callers_share_one_session(self) -> None:
        c = _client()
        try:
            sessions = await asyncio.gather(*(c._ensure_session() for _ in range(8)))
            # One session, or the client leaks unclosed connectors under load.
            assert len({id(s) for s in sessions}) == 1
        finally:
            await c.close()


class TestProxyResolution:
    def test_the_first_set_proxy_variable_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from kiro_crew.webex.client import _resolve_proxy

        for var in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY"):
            monkeypatch.delenv(var, raising=False)
            monkeypatch.delenv(var.lower(), raising=False)
        monkeypatch.setenv("HTTP_PROXY", "http://p:1")
        monkeypatch.setenv("HTTPS_PROXY", "http://p:2")
        # HTTPS_PROXY is checked first: an https API call must not be sent through
        # the plain-http proxy when both are set.
        assert _resolve_proxy() == "http://p:2"

    def test_no_proxy_variables_resolve_to_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from kiro_crew.webex.client import _resolve_proxy

        for var in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY"):
            monkeypatch.delenv(var, raising=False)
            monkeypatch.delenv(var.lower(), raising=False)
        assert _resolve_proxy() is None
