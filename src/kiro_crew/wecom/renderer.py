"""Layer 2b -- WeCom ``Renderer``.

Maps the channel-neutral ``OutputEvent`` stream (routed by the base
:class:`Renderer`'s ``dispatch``) onto WeCom AI-bot ``aibot_respond_msg`` WS
frames:

* ``on_turn_start`` -- a "🤔 …" placeholder stream frame.
* ``on_text_chunk`` -- throttled full-text stream frames (each frame REPLACES
  the bubble content); a trailing ``[OPTIONS:]`` trailer becomes a numbered text
  list (WeCom renders no tappable chips).
* ``on_tool_call`` -- a transient ``🔧 正在运行：{tool}…`` footer.
* ``on_prompt_choice`` -- no-op: WeCom has no interactive buttons, and the
  driver only dispatches this for INTERACTIVE + a decider; WeCom runs
  decider-less (deny-by-default), so this never fires.
* ``on_compaction`` -- logged only (a mid-turn out-of-band frame would pollute
  the single answer bubble; the dispatcher surfaces threshold notices).
* ``on_done`` -- the final ``finish=true`` frame (locks the bubble), plus one
  extra bubble per part when the answer exceeds the wire's byte cap, falling
  back to a one-shot ``response_url`` POST if the WS stream is
  unavailable/expired.

Dependency direction is ``wecom -> messaging`` (allowed).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

from kiro_crew.messaging.renderer import Renderer, render_options_as_text
from kiro_crew.messaging.split import chunk_utf8, split_markdown_safe
from kiro_crew.messaging.transport import TransportCapabilities
from kiro_crew.wecom.client import WECOM_MAX_REPLY_BYTES, new_stream_id

if TYPE_CHECKING:
    from kiro_crew.wecom.client import WeComClient

logger = logging.getLogger(__name__)

# Min seconds between intermediate stream frames: paces the typewriter effect
# and avoids hammering the stream API. The final finish frame ignores it.
_STREAM_THROTTLE_S = 0.7

# Placeholder shown immediately while the agent is still generating.
_THINKING = "🤔 …"


def _longest_fence_marker_line(content: str) -> int:
    """Length of the longest fence DELIMITER line, newline included.

    That line is what ``split_markdown_safe`` re-emits on every chunk it cuts
    inside a fence (as the reopener) and again as the synthetic closer, so it is
    the per-chunk scaffolding cost. Measuring it is one O(n) pass; paying for the
    split and measuring the result is not (see ``_split_for_delivery``).
    """
    longest = 0
    for line in content.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            longest = max(longest, len(line) + 1)
    return longest


# How much total text the fence-safe split may add before it stops being worth it.
# Every chunk of a split fence carries the opener again plus a synthetic closer, so
# a single long delimiter run makes that scaffolding overtake the answer: a
# 20 002-character reply built from a 5001-backtick line measured 15 002 parts and
# 150 MB of text, i.e. 15 002 bubbles. 4x is generous for legitimate scaffolding
# (the emoji-opener case that exercises the byte guard sits near 2x) and four
# orders of magnitude below the degenerate case.
_MAX_SCAFFOLD_RATIO = 4


class WeComDeliveryError(RuntimeError):
    """Part of an answer could not be delivered.

    A named type rather than a bare ``RuntimeError`` so a caller can tell an
    under-delivered answer apart from any other failure; the message carries what
    was lost.
    """


class WeComRenderer(Renderer):
    """Streams a turn to WeCom via WS ``aibot_respond_msg`` frames + fallback.

    Sends a "thinking" placeholder on ``on_turn_start``, throttled full-text
    updates (each frame carries the full accumulated text, replacing the
    bubble) on ``on_text_chunk``, and a final ``finish=true`` frame on
    ``on_done``. If the WS stream path is unavailable/expired it falls back to a
    single one-shot POST to the inbound message's ``response_url``.
    """

    channel_type = "wecom"

    def __init__(
        self,
        client: "WeComClient",
        req_id: str,
        response_url: str,
        capabilities: TransportCapabilities,
        *,
        session_key: str = "",
    ) -> None:
        super().__init__(capabilities)
        self._client = client
        self._req_id = req_id
        self._response_url = response_url
        self._session_key = session_key
        self._stream_id = new_stream_id()
        self._buf: list[str] = []
        self._last_send = 0.0
        # Stream only if we have a req_id to correlate; else go straight to the
        # one-shot response_url fallback at on_done.
        self._stream_ok = bool(req_id)
        self._tool = ""
        self._started = False
        self._finalized = False
        # Steer chip awaiting the text it heads (see on_steer_consumed).
        self._pending_chip = ""

    # -- lifecycle ----------------------------------------------------------
    async def on_turn_start(self) -> None:
        if self._started:  # idempotent (dispatch + driver both call it)
            return
        self._started = True
        if self._stream_ok:
            self._stream_ok = await self._client.send_stream(
                self._req_id, self._stream_id, _THINKING, finish=False
            )
        self._last_send = time.monotonic()

    async def on_text_chunk(self, text: str) -> None:
        self._materialize_chip()
        self._buf.append(text)
        self._tool = ""  # text resumed -> drop the transient tool footer
        await self._push(force=False)

    async def on_steer_consumed(self, summary: str = "") -> None:
        """Record that kiro-cli folded a mid-turn steer, for an in-answer receipt.

        The dispatcher already acked the steer out of band ("已合并到当前回复"), but
        that ack is a separate bubble: the answer itself showed no sign of where
        the fold happened, so a reader could not tell which half answered what.
        WeCom cannot rotate to a new message the way Discord and Telegram do — one
        turn is one bubble — so the boundary is marked inline with a quote chip.

        Materialized LAZILY, on the next text chunk. A steer folded at the very
        end of a stream (the answer already covered it) would otherwise leave a
        chip with nothing under it, and the out-of-band ack is receipt enough.
        """
        self._pending_chip = (summary or "").strip()

    def _materialize_chip(self) -> None:
        """Emit the pending steer chip, once, ahead of the text that follows it."""
        if not self._pending_chip:
            return
        prefix = "\n\n" if self._buf else ""
        self._buf.append(f"{prefix}> ↪️ {self._pending_chip}\n\n")
        self._pending_chip = ""

    async def on_thinking(self, text: str) -> None:
        # WeCom does not surface reasoning inline.
        return None

    async def on_tool_call(
        self, tool_call_id: str, title: str, tool_kind: str = "", tool_purpose: str = ""
    ) -> None:
        self._tool = title or tool_kind or "工具"
        await self._push(force=True)

    async def on_prompt_choice(self, options: list[dict[str, Any]], request_id: str | int) -> None:
        # WeCom has no interactive buttons. The driver only dispatches
        # prompt_choice for INTERACTIVE + a decider, and WeCom runs decider-less
        # (deny-by-default), so this is never reached -- kept as a safe no-op to
        # satisfy the Renderer contract.
        logger.debug("WeCom: prompt_choice ignored (no interactive buttons)")

    async def on_compaction(self, context_usage_pct: float) -> None:
        # A mid-turn out-of-band frame would pollute the single answer bubble;
        # the dispatcher surfaces soft/hard threshold notices post-turn instead.
        logger.debug("WeCom: compaction status %.0f%%", context_usage_pct)

    async def on_done(self, stop_reason: str = "") -> None:
        if self._finalized:
            return
        self._finalized = True
        self._tool = ""  # never lock a tool footer into the final answer
        ok = stop_reason != "error"
        content = self.text() or ("…" if ok else "⚠️ 出错了，请重试")
        if self._stream_ok:
            parts = await self._split_for_delivery(content)
            sent = await self._client.send_stream(
                self._req_id, self._stream_id, parts[0], finish=True
            )
            if sent:
                await self._send_overflow(parts[1:])
                return
        # Fallback: the stream never started or died mid-turn. ``response_url``
        # permits ONE reply, so an oversize answer loses its tail here -- the only
        # lossy delivery left on this channel, and unavoidable, since there is no
        # second frame to put the remainder in. Deliver what fits FIRST (a truncated
        # answer beats none), then fail the turn: returning normally would let
        # ``drive_turn`` run ``record_success`` and persist the FULL text, so the
        # transcript would claim a delivery the user cannot read and nothing
        # downstream could tell. A server-side refusal is a different failure and is
        # NOT covered: a non-zero ``errcode`` ACK is logged by the client and
        # dropped, so a frame the server rejected still records a successful turn.
        #
        # An ABSENT response_url is a total loss, not a partial one, and it has to be
        # checked separately: ``send_reply("")`` logs and returns, so with an answer
        # that fits the cap ``lost`` is 0 and nothing here would have raised — the
        # turn would report success for a reply that had no channel to travel on.
        # That is the same defect as an over-cap tail, in its worst form.
        if not self._response_url:
            logger.warning("WeCom: dead stream and no response_url — failing the turn")
            raise WeComDeliveryError("dead stream and no response_url: nothing was delivered")
        lost = max(0, len(content.encode("utf-8")) - WECOM_MAX_REPLY_BYTES)
        await self._client.send_reply(self._response_url, content)
        if lost > 0:
            logger.warning(
                "WeCom: dead-stream one-shot reply lost %d byte(s) — failing the turn", lost
            )
            raise WeComDeliveryError(f"dead-stream one-shot reply lost {lost} byte(s)")

    async def _split_for_delivery(self, content: str) -> list[str]:
        """The answer as the bubbles it will ship in. Lossless.

        Split only when the answer genuinely exceeds the wire's BYTE cap.
        ``max_message_chars`` is the worst case (4 bytes per character) so shared
        callers can trust it without inspecting text, but here the real encoded
        size is known — so an ASCII answer well under the byte cap still arrives
        as ONE bubble instead of being cut into four.

        Each part is then bounded in BYTES, because a character budget is not a
        sufficient guarantee: ``split_markdown_safe`` documents one chunk that may
        exceed ``limit`` by the fence scaffolding it synthesizes (a reopener line
        plus a synthetic closer), and a fence delimiter is arbitrarily long — a
        4000-emoji opener measured 21 007 bytes against the 20 000-byte wire. A
        conservative character budget cannot close that gap: the scaffolding's size
        is a property of the text, not a constant to subtract. So an over-budget
        part is re-chunked in BYTES rather than truncated, which costs markdown
        fidelity across those cuts and loses nothing — and only for the offending
        part, so the fence-safe boundaries the splitter found for every other part
        survive.

        The splitter is regex-heavy (milliseconds on fence-dense text), so the
        split runs off-loop; the common under-cap answer skips the thread hop
        entirely rather than paying it to do nothing.
        """
        if len(content.encode("utf-8")) <= WECOM_MAX_REPLY_BYTES:
            return [content]
        limit = self.capabilities.max_message_chars
        # Refuse the split BEFORE paying for it when the amplification is already
        # predictable. A chunk cut inside a fence carries the delimiter line TWICE
        # (reopener + synthetic closer), so the content budget per chunk is
        # `limit - 2*marker` and the output grows by `limit / (limit - 2*marker)`.
        # Bounding that by the same _MAX_SCAFFOLD_RATIO the backstop measures gives
        # the threshold below -- so the two guards are one bound expressed twice
        # (predicted, then measured) and cannot drift apart if the ratio changes.
        #
        # Testing only whether the marker FITS (`marker >= limit`) is not enough: at
        # `marker = limit/2 - 1` the line fits and leaves two characters of content
        # per chunk, which is ~10 000 chunks for a 20 000-character body. The
        # measured case was 15 002 parts and 150 MB, and on a gateway serving many
        # sessions that allocation is the memory-exhaustion window itself, not a
        # wasted millisecond.
        marker = _longest_fence_marker_line(content)
        if limit > 0 and 2 * _MAX_SCAFFOLD_RATIO * marker >= (_MAX_SCAFFOLD_RATIO - 1) * limit:
            logger.warning(
                "WeCom: a %d-char fence delimiter line leaves too little of the "
                "%d-char chunk budget for content — byte-chunking without attempting "
                "a fence-safe split",
                marker,
                limit,
            )
            return chunk_utf8(content, WECOM_MAX_REPLY_BYTES)
        split = await asyncio.to_thread(split_markdown_safe, content, limit)
        split = split or [content]
        # BACKSTOP for a blowup the pre-check above does not model. Kept rather than
        # replaced: the pre-check is causal but it is a predicate on ONE property of
        # the text, and this one measures what actually came back. Neither caps the
        # bubble count -- a fixed cap would newly truncate a legitimately long answer,
        # which is the defect this method exists to remove. Falling back to blind byte
        # chunking is lossless and bounded at ceil(bytes / cap) bubbles; markdown may
        # render badly across those cuts, which is the honest price of every character.
        if sum(len(p) for p in split) > _MAX_SCAFFOLD_RATIO * len(content):
            logger.warning(
                "WeCom: fence-safe split fanned out to %d part(s) for %d chars "
                "— falling back to byte chunking",
                len(split),
                len(content),
            )
            return chunk_utf8(content, WECOM_MAX_REPLY_BYTES)
        parts: list[str] = []
        for part in split:
            if len(part.encode("utf-8")) <= WECOM_MAX_REPLY_BYTES:
                parts.append(part)
                continue
            logger.warning(
                "WeCom: a split part exceeded the wire cap by %d byte(s) "
                "— byte-chunking that part",
                len(part.encode("utf-8")) - WECOM_MAX_REPLY_BYTES,
            )
            parts.extend(chunk_utf8(part, WECOM_MAX_REPLY_BYTES))
        return parts

    async def _send_overflow(self, parts: list[str]) -> None:
        """Deliver split overflow as additional bubbles.

        Each part needs its OWN ``stream_id`` -- reusing one replaces the previous
        bubble's content instead of adding to it -- and must be finished, or the
        bubble sits open until WeCom expires it.

        RAISES on a failed send, and the exception is the point: swallowing it
        would leave the user holding part of an answer while ``drive_turn`` ran
        ``record_success`` and persisted the WHOLE text as delivered. The
        transcript would then disagree with what the user can actually read, and
        nothing downstream could tell. Reaching ``drive_turn``'s except branch
        records the failure instead. Same contract as the Weixin renderer's
        ``_send``; ``close()`` is the teardown path and suppresses it, because by
        then the turn is already unwinding and this would mask the real cause.

        Stops at the FIRST failure rather than pushing the remaining parts into a
        stream the server has already refused.
        """
        for index, part in enumerate(parts):
            if not await self._client.send_bubble(self._req_id, part):
                undelivered = len(parts) - index
                logger.warning(
                    "WeCom: overflow bubble send failed; %d part(s) undelivered "
                    "— failing the turn",
                    undelivered,
                )
                raise WeComDeliveryError(f"{undelivered} overflow bubble(s) undelivered")

    async def close(self) -> None:
        """Idempotent teardown: finalize the turn if it never reached on_done.

        Runs from ``drive_turn``'s ``finally``, so a delivery failure here is
        logged and suppressed — raising would mask whatever unwound the turn.
        """
        if not self._finalized:
            try:
                await self.on_done(stop_reason="error")
            except Exception:
                logger.warning("WeCom: final send failed during teardown", exc_info=True)

    # -- helpers ------------------------------------------------------------
    def text(self) -> str:
        """The turn's answer so far, with ``[OPTIONS:]`` as numbered text.

        No tool footer: used both for the final frame and to persist the reply.
        """
        return render_options_as_text("".join(self._buf).strip(), self.capabilities)

    def _compose(self) -> str:
        """Streaming content = answer text + optional transient tool footer."""
        body = self.text()
        if self._tool:
            footer = f"🔧 正在运行：{self._tool}…"
            return f"{body}\n\n{footer}" if body else footer
        return body

    async def _push(self, *, force: bool) -> None:
        if not self._stream_ok:
            return
        now = time.monotonic()
        if not force and now - self._last_send < _STREAM_THROTTLE_S:
            return
        content = self._compose()
        if not content:
            return
        # A live frame REPLACES one bubble, so there is nowhere to put overflow
        # mid-stream: keep the leading window and let ``on_done`` split the sealed
        # answer across bubbles. Nothing is lost -- this frame is transient and the
        # full text is still in the buffer.
        cap = self.capabilities.max_message_chars
        if cap > 0:
            content = content[:cap]
        self._stream_ok = await self._client.send_stream(
            self._req_id, self._stream_id, content, finish=False
        )
        self._last_send = now
