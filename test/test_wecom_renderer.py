"""Tests for kiro_crew.wecom.renderer (WeComRenderer, Layer 2b)."""

from __future__ import annotations

import pytest

from kiro_crew.messaging.split import split_markdown_safe
from kiro_crew.wecom.client import WECOM_MAX_REPLY_BYTES, new_stream_id
from kiro_crew.wecom.renderer import (
    _MAX_SCAFFOLD_RATIO,
    WeComDeliveryError,
    WeComRenderer,
)
from kiro_crew.wecom.transport import WECOM_CAPABILITIES

# The trailer grammar, its ReDoS hardening and the numbered-text fallback are
# shared by every zero-widget channel and pinned once in
# test_options_cap_contract.py (TestRenderOptionsAsText +
# TestZeroWidgetChannelEnforcement::test_wecom).


class FakeClient:
    """Records WS stream frames and response_url fallback POSTs."""

    def __init__(self, stream_ok: bool = True) -> None:
        self.frames: list[dict] = []
        self.replies: list[tuple[str, str]] = []
        self._stream_ok = stream_ok

    async def send_stream(self, req_id: str, stream_id: str, content: str, *, finish: bool) -> bool:
        self.frames.append(
            {
                "req_id": req_id,
                "stream_id": stream_id,
                "content": content,
                "finish": finish,
            }
        )
        return self._stream_ok

    async def send_bubble(self, req_id: str, content: str) -> bool:
        # Mirrors the real client: a NEW bubble means a fresh stream_id, finished.
        return await self.send_stream(req_id, new_stream_id(), content, finish=True)

    async def send_reply(self, url: str, content: str) -> None:
        self.replies.append((url, content))


def _renderer(client: FakeClient, req_id: str = "rq1") -> WeComRenderer:
    return WeComRenderer(client, req_id, "https://resp.url", WECOM_CAPABILITIES)


class TestStreaming:
    @pytest.mark.asyncio
    async def test_turn_start_sends_placeholder(self) -> None:
        c = FakeClient()
        r = _renderer(c)
        await r.on_turn_start()
        assert c.frames[0]["content"] == "🤔 …"
        assert c.frames[0]["finish"] is False

    @pytest.mark.asyncio
    async def test_turn_start_idempotent(self) -> None:
        c = FakeClient()
        r = _renderer(c)
        await r.on_turn_start()
        await r.on_turn_start()  # second call no-ops
        assert len(c.frames) == 1

    @pytest.mark.asyncio
    async def test_final_answer_is_accumulated_text(self) -> None:
        c = FakeClient()
        r = _renderer(c)
        await r.on_turn_start()
        await r.on_text_chunk("Hello ")
        await r.on_text_chunk("world")
        await r.on_done()
        final = c.frames[-1]
        assert final["content"] == "Hello world"
        assert final["finish"] is True

    @pytest.mark.asyncio
    async def test_options_trailer_becomes_numbered_text(self) -> None:
        # WeCom renders no chips, but the choices are the answers to the question
        # the body just asked -- dropping them left the user with no way to see
        # what was offered.
        c = FakeClient()
        r = _renderer(c)
        await r.on_turn_start()
        await r.on_text_chunk("Pick one\n\n[OPTIONS: A | B | C]")
        await r.on_done()
        assert c.frames[-1]["content"] == "Pick one\n\n1. A\n2. B\n3. C"

    @pytest.mark.asyncio
    async def test_tool_footer_pushed(self) -> None:
        c = FakeClient()
        r = _renderer(c)
        await r.on_turn_start()
        await r.on_tool_call("t1", "fs_read", tool_kind="read")
        # force-pushed frame carries the transient footer
        assert any("🔧 正在运行：fs_read" in f["content"] for f in c.frames)

    @pytest.mark.asyncio
    async def test_error_done_shows_error_text(self) -> None:
        c = FakeClient()
        r = _renderer(c)
        await r.on_turn_start()
        await r.on_done(stop_reason="error")
        assert c.frames[-1]["content"] == "⚠️ 出错了，请重试"
        assert c.frames[-1]["finish"] is True


class TestFallback:
    @pytest.mark.asyncio
    async def test_no_req_id_uses_response_url(self) -> None:
        c = FakeClient()
        r = WeComRenderer(c, "", "https://resp.url", WECOM_CAPABILITIES)
        await r.on_turn_start()  # no stream (no req_id)
        await r.on_text_chunk("reply text")
        await r.on_done()
        assert c.frames == []  # never streamed
        assert c.replies == [("https://resp.url", "reply text")]

    @pytest.mark.asyncio
    async def test_stream_died_falls_back_to_response_url(self) -> None:
        c = FakeClient(stream_ok=False)  # every send_stream reports failure
        r = _renderer(c)
        await r.on_turn_start()  # placeholder send reports False -> stream_ok flips off
        await r.on_text_chunk("answer")
        await r.on_done()
        assert c.replies == [("https://resp.url", "answer")]


class TestClose:
    @pytest.mark.asyncio
    async def test_close_after_done_is_noop(self) -> None:
        c = FakeClient()
        r = _renderer(c)
        await r.on_turn_start()
        await r.on_text_chunk("done text")
        await r.on_done()
        finish_frames_before = sum(1 for f in c.frames if f["finish"])
        await r.close()
        finish_frames_after = sum(1 for f in c.frames if f["finish"])
        assert finish_frames_before == finish_frames_after == 1

    @pytest.mark.asyncio
    async def test_close_without_done_finalizes(self) -> None:
        c = FakeClient()
        r = _renderer(c)
        await r.on_turn_start()
        await r.on_text_chunk("partial")
        await r.close()  # turn never reached on_done (e.g. cold-start failure)
        assert any(f["finish"] for f in c.frames)


class TestPromptChoice:
    @pytest.mark.asyncio
    async def test_prompt_choice_is_noop(self) -> None:
        c = FakeClient()
        r = _renderer(c)
        await r.on_turn_start()
        before = len(c.frames)
        await r.on_prompt_choice([{"label": "yes"}], "rq")  # WeCom has no buttons
        assert len(c.frames) == before  # nothing rendered, no raise


class TestOverCapAnswer:
    """A long answer is SPLIT across bubbles, never truncated.

    WeCom was the only channel that sliced an over-cap reply and dropped the
    tail: a stream frame replaces one bubble, so there was nowhere for the
    remainder to go and nothing said so. Overflow now rides additional bubbles
    on the same ``req_id``, each with its own ``stream_id``.
    """

    def _long_answer(self) -> str:
        """An answer over the wire's BYTE cap, in plain prose the splitter can cut.

        Sized against WECOM_MAX_REPLY_BYTES, not the declared character cap: the
        renderer only splits once the real encoded size exceeds the wire budget.
        """
        line = "the quick brown fox jumps over the lazy dog. " * 20
        lines = [line] * (2 * WECOM_MAX_REPLY_BYTES // len(line))
        return "\n".join(lines)

    @pytest.mark.asyncio
    async def test_nothing_is_lost_and_each_part_is_its_own_bubble(self) -> None:
        c = FakeClient()
        r = _renderer(c)
        answer = self._long_answer()
        await r.on_turn_start()
        await r.on_text_chunk(answer)
        await r.on_done()

        finished = [f for f in c.frames if f["finish"]]
        assert len(finished) > 1, "an over-cap answer must span more than one bubble"
        # Every bubble is finished, addressed to the same inbound req_id, and has
        # its own stream id -- a reused id would overwrite the previous bubble.
        assert {f["req_id"] for f in finished} == {"rq1"}
        assert len({f["stream_id"] for f in finished}) == len(finished)
        assert all(len(f["content"]) <= WECOM_CAPABILITIES.max_message_chars for f in finished)
        # Every authored word survives, in order -- which the old slice did not.
        # Compared per part because the splitter trims the whitespace it cut on,
        # so the seams lose a newline by design and only the words are stable.
        delivered: list[str] = []
        for frame in finished:
            delivered.extend(frame["content"].split())
        assert delivered == answer.split()

    @pytest.mark.asyncio
    async def test_an_under_cap_answer_stays_one_bubble(self) -> None:
        c = FakeClient()
        r = _renderer(c)
        await r.on_turn_start()
        await r.on_text_chunk("short answer")
        await r.on_done()
        finished = [f for f in c.frames if f["finish"]]
        assert len(finished) == 1
        assert finished[0]["content"] == "short answer"
        assert (
            finished[0]["stream_id"] == c.frames[0]["stream_id"]
        ), "the sealed answer must land in the bubble the placeholder opened"

    @pytest.mark.asyncio
    async def test_a_fenced_code_block_is_not_bisected(self) -> None:
        # The shared splitter seals an open fence and reopens it; a naive cut
        # would leave one bubble rendering as code and the next as prose.
        body = "```python\n" + "x = 1\n" * (WECOM_MAX_REPLY_BYTES // 3) + "```"
        c = FakeClient()
        r = _renderer(c)
        await r.on_turn_start()
        await r.on_text_chunk(body)
        await r.on_done()
        finished = [f for f in c.frames if f["finish"]]
        assert len(finished) > 1
        for frame in finished:
            assert (
                frame["content"].count("```") % 2 == 0
            ), "every bubble must open and close its own fence"

    @pytest.mark.asyncio
    async def test_a_failed_overflow_send_fails_the_turn(self) -> None:
        # The exception IS the contract. Swallowing it would leave the user
        # holding part of an answer while drive_turn ran record_success and
        # persisted the WHOLE text as delivered -- a transcript that disagrees
        # with what the user can read, and nothing downstream could tell.
        # Stops at the first failure rather than pushing the rest into a stream
        # the server just refused.
        class FailAfterFirst(FakeClient):
            async def send_stream(self, req_id, stream_id, content, *, finish):
                await super().send_stream(req_id, stream_id, content, finish=finish)
                return len([f for f in self.frames if f["finish"]]) <= 1

        c = FailAfterFirst()
        r = _renderer(c)
        await r.on_turn_start()
        await r.on_text_chunk(self._long_answer())
        with pytest.raises(WeComDeliveryError):
            await r.on_done()
        finished = [f for f in c.frames if f["finish"]]
        assert len(finished) == 2, "one sealed answer plus exactly one failed attempt"

    @pytest.mark.asyncio
    async def test_teardown_suppresses_a_delivery_failure(self) -> None:
        # close() runs from drive_turn's finally, where the turn is already
        # unwinding: raising there would mask whatever actually unwound it.
        class AlwaysFail(FakeClient):
            async def send_stream(self, req_id, stream_id, content, *, finish):
                await super().send_stream(req_id, stream_id, content, finish=finish)
                return len([f for f in self.frames if f["finish"]]) <= 1

        c = AlwaysFail()
        r = _renderer(c)
        await r.on_turn_start()
        await r.on_text_chunk(self._long_answer())
        await r.close()  # must not raise


class TestSteerReceipt:
    """A folded steer leaves a mark IN the answer, not just an out-of-band ack.

    The dispatcher acks a steer in its own bubble, but the answer itself showed no
    sign of where the fold happened, so a reader could not tell which half
    answered what. WeCom is one bubble per turn, so the boundary is inline.
    """

    @pytest.mark.asyncio
    async def test_a_chip_heads_the_post_steer_text(self) -> None:
        c = FakeClient()
        r = _renderer(c)
        await r.on_turn_start()
        await r.on_text_chunk("first half")
        await r.on_steer_consumed("also check the weather")
        await r.on_text_chunk("second half")
        await r.on_done()
        body = c.frames[-1]["content"]
        assert "> ↪️ also check the weather" in body
        assert body.index("first half") < body.index("↪️") < body.index("second half")

    @pytest.mark.asyncio
    async def test_a_trailing_steer_emits_no_chip(self) -> None:
        # Lazy materialization: a steer folded at the very end was already covered
        # by the answer, and a chip with nothing under it is noise.
        c = FakeClient()
        r = _renderer(c)
        await r.on_turn_start()
        await r.on_text_chunk("the whole answer")
        await r.on_steer_consumed("already covered")
        await r.on_done()
        assert "↪️" not in c.frames[-1]["content"]

    @pytest.mark.asyncio
    async def test_an_empty_summary_emits_no_chip(self) -> None:
        c = FakeClient()
        r = _renderer(c)
        await r.on_turn_start()
        await r.on_text_chunk("before")
        await r.on_steer_consumed("")
        await r.on_text_chunk("after")
        await r.on_done()
        assert "↪️" not in c.frames[-1]["content"]

    @pytest.mark.asyncio
    async def test_a_chip_at_the_start_of_a_turn_needs_no_leading_blank(self) -> None:
        c = FakeClient()
        r = _renderer(c)
        await r.on_turn_start()
        await r.on_steer_consumed("early fold")
        await r.on_text_chunk("body")
        await r.on_done()
        assert c.frames[-1]["content"].startswith("> ↪️ early fold")


class TestDeadStreamFallback:
    """The one-shot fallback is the degraded path, not a silent one.

    ``response_url`` permits ONE reply, so an oversize answer loses its tail to the
    client's byte guard. Returning normally would let ``drive_turn`` persist the
    whole text and record success while the user holds a cut-off answer — the same
    inconsistency the overflow path raises for.
    """

    @pytest.mark.asyncio
    async def test_a_short_answer_on_a_dead_stream_is_delivered_and_succeeds(self) -> None:
        c = FakeClient(stream_ok=False)
        r = _renderer(c)
        await r.on_turn_start()
        await r.on_text_chunk("short answer")
        await r.on_done()  # must not raise: nothing was lost
        assert c.replies == [("https://resp.url", "short answer")]

    @pytest.mark.asyncio
    async def test_an_oversize_answer_on_a_dead_stream_delivers_then_fails(self) -> None:
        over = "字" * WECOM_MAX_REPLY_BYTES  # ~3x the byte cap
        c = FakeClient(stream_ok=False)
        r = _renderer(c)
        await r.on_turn_start()
        await r.on_text_chunk(over)
        with pytest.raises(WeComDeliveryError):
            await r.on_done()
        # Delivered FIRST: a truncated answer beats none.
        assert len(c.replies) == 1
        assert c.replies[0][1] == over

    @pytest.mark.asyncio
    async def test_teardown_suppresses_the_fallback_failure_too(self) -> None:
        c = FakeClient(stream_ok=False)
        r = _renderer(c)
        await r.on_turn_start()
        await r.on_text_chunk("字" * WECOM_MAX_REPLY_BYTES)
        await r.close()  # must not raise
        assert len(c.replies) == 1


class TestOverBudgetPartFromTheSplitter:
    """The splitter's documented over-limit exception must cost no bytes.

    ``split_markdown_safe`` may return one chunk that exceeds ``limit`` by the
    fence scaffolding it synthesizes, and a fence delimiter is arbitrarily long —
    so a character budget alone is not a byte guarantee, and no constant subtracted
    from it would be either. Nothing is truncated on either route: an over-budget
    part is re-chunked in BYTES, and markdown rendering badly across those cuts is
    the whole price. An earlier shape truncated the part and failed the turn, which
    delivered a mutilated answer AND reported failure when neither was necessary.

    This fixture's opener is long enough that the pre-check in
    ``_split_for_delivery`` now byte-chunks the whole answer before the splitter
    runs, so the assertions below exercise that route. The direct-splitter test is
    what keeps recording WHY the per-part byte bound exists — it is still reachable
    for a part that goes over in BYTES without a long marker, which CJK does by
    expanding 5 000 characters into 15 000 bytes.
    """

    def _scaffold_heavy(self) -> str:
        """An answer the splitter provably returns an over-budget part for.

        Two ingredients. The opener is thousands of 4-byte emoji, so every chunk
        that REOPENS the fence carries that whole line again. The body is a run of
        fence characters, which admits no cut clean on both sides — so the splitter
        places the line WHOLE (its documented residue policy) and adds the reopener
        plus a synthetic closer on top of the character limit. Measured: a
        21 007-byte part against a 20 000-byte wire.
        """
        opener = "~~~" + "\U0001f600" * 4000
        return f"{opener}\n" + "`" * 4999 + "\n" + "x\n" * 5 + "~~~\n"

    def test_the_fixture_really_does_produce_an_over_budget_part(self) -> None:
        # Guards the guard: if the splitter ever stops returning an over-budget
        # part for this input, the two tests below would pass for the wrong reason.
        from kiro_crew.messaging.split import split_markdown_safe

        parts = split_markdown_safe(self._scaffold_heavy(), WECOM_CAPABILITIES.max_message_chars)
        assert max(len(p.encode("utf-8")) for p in parts) > WECOM_MAX_REPLY_BYTES

    @pytest.mark.asyncio
    async def test_no_bubble_exceeds_the_byte_cap(self) -> None:
        c = FakeClient()
        r = _renderer(c)
        await r.on_turn_start()
        await r.on_text_chunk(self._scaffold_heavy())
        await r.on_done()
        for frame in c.frames:
            assert len(frame["content"].encode("utf-8")) <= WECOM_MAX_REPLY_BYTES

    @pytest.mark.asyncio
    async def test_nothing_is_dropped_and_the_turn_succeeds(self) -> None:
        # Truncating the over-budget part cost 1 007 bytes of the fence run AND
        # failed the turn. Byte-chunking it costs neither: nothing is dropped, so
        # there is nothing for drive_turn to mis-persist and no failure to report.
        # Asserted on the fence run rather than by reassembling the whole answer,
        # because the splitter legitimately ADDS scaffolding (a reopener per chunk
        # plus a synthetic closer) and trims the whitespace it cut on -- so exact
        # equality would pin the splitter's formatting, not this method's losslessness.
        source = self._scaffold_heavy()
        c = FakeClient()
        r = _renderer(c)
        await r.on_turn_start()
        await r.on_text_chunk(source)
        await r.on_done()  # no WeComDeliveryError

        delivered = "".join(f["content"] for f in c.frames if f["finish"])
        assert delivered.count("`") >= source.count("`")
        assert len(delivered.encode("utf-8")) >= len(source.strip().encode("utf-8"))
        assert not c.replies, "the stream was alive; the one-shot fallback must stay unused"


class TestDegenerateSplitFanOut:
    """A long delimiter run must not turn one answer into thousands of bubbles.

    Every chunk of a split fence carries the opener again plus a synthetic closer,
    so a bare run of delimiters makes that scaffolding overtake the text. The
    fallback is blind byte chunking: markdown may render badly across those cuts,
    which is the honest price of delivering every character.
    """

    #: 20 002 chars: a 5001-backtick line (an opener nothing can cut cleanly) plus
    #: ordinary text. Measured on the shared splitter: 15 002 parts, 150 MB.
    PATHOLOGICAL = "`" * 5001 + "\n" + "x" * 15000

    def test_the_raw_splitter_really_does_fan_out_for_this_input(self) -> None:
        # Guard on the guard: if the shared splitter is ever fixed, the test below
        # would pass for the wrong reason.
        from kiro_crew.messaging.split import split_markdown_safe

        parts = split_markdown_safe(self.PATHOLOGICAL, WECOM_CAPABILITIES.max_message_chars)
        assert len(parts) > 10_000

    @pytest.mark.asyncio
    async def test_delivery_is_bounded_and_loses_nothing(self) -> None:
        c = FakeClient()
        r = _renderer(c)
        await r.on_turn_start()
        await r.on_text_chunk(self.PATHOLOGICAL)
        await r.on_done()  # must not raise: byte chunking is lossless

        finished = [f for f in c.frames if f["finish"]]
        ceiling = -(-len(self.PATHOLOGICAL.encode("utf-8")) // WECOM_MAX_REPLY_BYTES)
        assert len(finished) <= ceiling, f"{len(finished)} bubbles for a {ceiling}-bubble answer"
        assert "".join(f["content"] for f in finished) == self.PATHOLOGICAL
        for frame in finished:
            assert len(frame["content"].encode("utf-8")) <= WECOM_MAX_REPLY_BYTES


class TestTheDegenerateSplitIsRefusedBeforeItAllocates:
    """The runaway split must be declined, not detected after the fact.

    ``_MAX_SCAFFOLD_RATIO`` catches a blown-up split by measuring the result, which
    means the 150 MB was already allocated. On a gateway serving many sessions that
    allocation IS the failure, so the provable case is refused up front: if a fence
    delimiter line does not fit the per-chunk budget, every chunk must carry it twice
    and cannot converge.
    """

    #: A 5001-backtick line (nothing can cut it cleanly) plus ordinary text.
    _RUNAWAY = "`" * 5001 + "\n" + "x" * 15000

    @pytest.mark.asyncio
    async def test_the_splitter_is_never_called(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def exploding_split(text: str, limit: int) -> list[str]:
            raise AssertionError("split_markdown_safe must not run on a provable runaway")

        monkeypatch.setattr("kiro_crew.wecom.renderer.split_markdown_safe", exploding_split)
        c = FakeClient()
        r = _renderer(c)
        await r.on_turn_start()
        await r.on_text_chunk(self._RUNAWAY)
        await r.on_done()
        delivered = [f for f in c.frames if f["finish"]]
        assert delivered, "the answer must still be delivered, by byte chunking"
        assert all(len(f["content"].encode("utf-8")) <= WECOM_MAX_REPLY_BYTES for f in delivered)

    @pytest.mark.asyncio
    async def test_every_character_still_arrives(self) -> None:
        c = FakeClient()
        r = _renderer(c)
        await r.on_turn_start()
        await r.on_text_chunk(self._RUNAWAY)
        await r.on_done()
        rebuilt = "".join(f["content"] for f in c.frames if f["finish"])
        # Byte chunking is a pure partition, so this is exact equality -- unlike the
        # fence-safe path, which legitimately adds scaffolding and trims seams.
        assert rebuilt == self._RUNAWAY.strip()

    @pytest.mark.asyncio
    async def test_a_normal_fenced_answer_still_takes_the_fence_safe_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The guard must not fire on ordinary code blocks, or every long fenced
        # answer would lose its fence boundaries for nothing.
        called: list[int] = []
        real = split_markdown_safe

        def counting_split(text: str, limit: int) -> list[str]:
            called.append(limit)
            return real(text, limit)

        monkeypatch.setattr("kiro_crew.wecom.renderer.split_markdown_safe", counting_split)
        c = FakeClient()
        r = _renderer(c)
        await r.on_turn_start()
        await r.on_text_chunk("```python\n" + "print('x')\n" * 3000 + "```\n")
        await r.on_done()
        assert called, "an ordinary fenced answer must still be split fence-safely"

    def test_the_marker_scan_measures_the_delimiter_line_not_the_body(self) -> None:
        from kiro_crew.wecom.renderer import _longest_fence_marker_line

        assert _longest_fence_marker_line("no fences here\njust prose") == 0
        # The info string rides the opener line, so it counts: it is re-emitted whole.
        assert _longest_fence_marker_line("~~~" + "a" * 100 + "\nbody\n~~~\n") == 104
        # An indented fence still opens one.
        assert _longest_fence_marker_line("  ```ts\nx\n  ```\n") == 8


class TestTheFenceGuardThresholdIsDerivedNotTuned:
    """The pre-check threshold must follow ``_MAX_SCAFFOLD_RATIO``, not a constant.

    A chunk cut inside a fence carries the delimiter line twice, so content budget
    per chunk is ``limit - 2*marker`` and amplification is
    ``limit / (limit - 2*marker)``. Bounding that by the ratio the backstop measures
    is what makes the two guards one bound expressed twice. A "does the marker fit"
    test (``marker >= limit``) looks sufficient and is not: at ``marker = limit/2 - 1``
    the line fits while leaving two characters of content per chunk.
    """

    LIMIT = WECOM_CAPABILITIES.max_message_chars

    def _trips(self, marker: int) -> bool:
        return 2 * _MAX_SCAFFOLD_RATIO * marker >= (_MAX_SCAFFOLD_RATIO - 1) * self.LIMIT

    def test_the_threshold_matches_the_amplification_bound(self) -> None:
        # Algebra check: exactly the markers whose predicted amplification exceeds
        # the ratio must trip, so a future change to _MAX_SCAFFOLD_RATIO moves both
        # guards together instead of leaving the pre-check pinned to an old number.
        for marker in range(1, self.LIMIT // 2):
            budget = self.LIMIT - 2 * marker
            predicted = self.LIMIT / budget if budget > 0 else float("inf")
            assert self._trips(marker) == (predicted >= _MAX_SCAFFOLD_RATIO), marker

    def test_an_ordinary_code_fence_does_not_trip_it(self) -> None:
        assert not self._trips(len("```python") + 1)

    def test_a_marker_that_fits_but_starves_the_content_budget_trips_it(self) -> None:
        # The case a bare "does it fit" guard would let through.
        assert not (self.LIMIT // 2 - 1) >= self.LIMIT
        assert self._trips(self.LIMIT // 2 - 1)

    @pytest.mark.asyncio
    async def test_the_starving_marker_never_reaches_the_splitter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A 2 499-backtick opener plus a 20 000-character body: the delimiter line
        # fits the budget, so it survives a fit-only guard and then allocates roughly
        # 87 M characters.
        def exploding_split(text: str, limit: int) -> list[str]:
            raise AssertionError("split_markdown_safe must not run on a starving budget")

        monkeypatch.setattr("kiro_crew.wecom.renderer.split_markdown_safe", exploding_split)
        answer = "`" * (self.LIMIT // 2 - 2) + "\n" + "x" * 20000
        c = FakeClient()
        r = _renderer(c)
        await r.on_turn_start()
        await r.on_text_chunk(answer)
        await r.on_done()
        delivered = [f for f in c.frames if f["finish"]]
        assert "".join(f["content"] for f in delivered) == answer.strip()
        assert all(len(f["content"].encode("utf-8")) <= WECOM_MAX_REPLY_BYTES for f in delivered)


class TestADeadStreamWithNoResponseUrlFailsTheTurn:
    """A total delivery failure must not report success.

    ``send_reply("")`` logs and returns — it does not raise. So with a dead stream,
    an empty ``response_url`` and an answer that fits the cap, the byte-loss check
    sees ``lost == 0`` and nothing else would have failed the turn: ``drive_turn``
    would run ``record_success`` and persist the full text for a reply that had no
    channel to travel on. Reachable whenever a ``req_id`` frame's stream write
    fails, which is exactly when the fallback is needed most.
    """

    @staticmethod
    def _renderer_without_url(client: FakeClient) -> WeComRenderer:
        return WeComRenderer(client, "rq1", "", WECOM_CAPABILITIES)

    @pytest.mark.asyncio
    async def test_it_raises_even_though_nothing_was_truncated(self) -> None:
        class DeadStream(FakeClient):
            async def send_stream(self, req_id, sid, content, *, finish) -> bool:
                await super().send_stream(req_id, sid, content, finish=finish)
                return False

        c = DeadStream()
        r = self._renderer_without_url(c)
        await r.on_turn_start()
        await r.on_text_chunk("a short answer, well under the byte cap")
        with pytest.raises(WeComDeliveryError) as caught:
            await r.on_done()
        assert "no response_url" in str(caught.value)
        assert not c.replies, "there was no channel to post to"

    @pytest.mark.asyncio
    async def test_a_present_url_still_delivers_and_succeeds(self) -> None:
        # The guard must not fire on the case the fallback exists to serve.
        class DeadStream(FakeClient):
            async def send_stream(self, req_id, sid, content, *, finish) -> bool:
                await super().send_stream(req_id, sid, content, finish=finish)
                return False

        c = DeadStream()
        r = WeComRenderer(c, "rq1", "https://resp.url", WECOM_CAPABILITIES)
        await r.on_turn_start()
        await r.on_text_chunk("a short answer")
        await r.on_done()
        assert [content for _u, content in c.replies] == ["a short answer"]

    @pytest.mark.asyncio
    async def test_teardown_still_suppresses_it(self) -> None:
        # close() runs from drive_turn's finally, where raising would mask whatever
        # unwound the turn. The delivery failure is logged there, not propagated.
        c = FakeClient(stream_ok=False)
        r = self._renderer_without_url(c)
        await r.on_text_chunk("unsent")
        await r.close()  # must not raise
        assert r._finalized
