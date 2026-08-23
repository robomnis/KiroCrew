"""The document channel that artifact and widget frames load from.

These frames render HTML the model wrote. The channel exists because a ``blob:``
URL is refused outright by some WebKit-based in-app browsers, so the bytes are
served as a real document instead — which means the dashboard's own origin now
has a URL that returns model-authored HTML. What keeps that safe is asserted
here, not assumed:

* the path token is the credential, and it is bound to the requesting client
* every invalid condition fails closed as 404, so the route is not an oracle
* ``Content-Security-Policy: sandbox`` travels on the response, so the document
  has an opaque origin even when opened top-level rather than in a frame
* the stash is bounded, so a long session cannot grow it without limit
"""

from __future__ import annotations

import time

import pytest

from kiro_crew.dashboard.handlers import sandbox_doc as sd


@pytest.fixture(autouse=True)
def _clean_stash():
    with sd._lock:
        sd._stash.clear()
    yield
    with sd._lock:
        sd._stash.clear()


def test_a_token_is_bound_to_the_client_that_minted_it() -> None:
    """The token rides in the frame's own URL, where model script can read it."""
    token = sd._signer.mint("doc1", "10.0.0.1")
    assert sd._signer.verify(token, "doc1", "10.0.0.1")
    assert not sd._signer.verify(token, "doc1", "10.0.0.2"), (
        "a token lifted out of the frame's location worked from another client — "
        "binding it to the minting connection is what makes exfiltration useless"
    )


def test_a_token_does_not_carry_to_another_document() -> None:
    token = sd._signer.mint("doc1", "c")
    assert not sd._signer.verify(
        "doc2", token, "c"
    ), "a token minted for one document authorized another"


def test_an_expired_token_is_refused() -> None:
    exp = int(time.time()) - 1
    token = f"{exp}.{sd._signer._mac(exp, ('doc1', 'c'))}"
    assert not sd._signer.verify(token, "doc1", "c")


@pytest.mark.parametrize(
    "token",
    [
        "",
        "nonsense",
        "9999999999.",
        ".abc",
        "abc.def",
        "9" * 400 + ".x",  # digit-limit blowup guard
    ],
)
def test_malformed_tokens_are_refused_without_raising(token: str) -> None:
    assert sd._signer.verify(token, "doc1", "c") is False


def test_a_tampered_mac_is_refused() -> None:
    token = sd._signer.mint("doc1", "c")
    exp, _, mac = token.partition(".")
    flipped = ("0" if mac[0] != "0" else "1") + mac[1:]
    assert not sd._signer.verify(f"{exp}.{flipped}", "doc1", "c")


def test_the_stash_is_bounded_by_entry_count() -> None:
    """Entries past the in-flight grace window ARE evicted, oldest first.

    The cap is asserted on documents that already had their chance to be
    fetched. An entry still in flight is deliberately exempt — see
    `test_an_in_flight_document_is_never_evicted_to_make_room` — so this ages the
    entries past the grace window, which is exactly when eviction is legitimate.
    """
    now = time.time()
    stale_exp = now + sd._TOKEN_TTL_SECS - sd._IN_FLIGHT_GRACE_SECS - 1
    with sd._lock:
        for i in range(sd._MAX_ENTRIES + 20):
            sd._stash[f"d{i}"] = (stale_exp, "x")
        assert sd._prune(now) is True
        assert len(sd._stash) <= sd._MAX_ENTRIES, (
            "the stash grew past its entry cap; a long dashboard session would "
            "hold every document it ever rendered"
        )
        # Eviction is oldest-first, so the most recent entry must survive.
        assert f"d{sd._MAX_ENTRIES + 19}" in sd._stash


def test_the_stash_is_bounded_by_total_bytes() -> None:
    now = time.time()
    stale_exp = now + sd._TOKEN_TTL_SECS - sd._IN_FLIGHT_GRACE_SECS - 1
    big = "x" * (sd._MAX_BYTES // 4)
    with sd._lock:
        for i in range(8):
            sd._stash[f"d{i}"] = (stale_exp, big)
        assert sd._prune(now) is True
        total = sum(len(html) for _, html in sd._stash.values())
        assert total <= sd._MAX_BYTES, "the stash grew past its byte cap"


def test_a_stash_full_of_in_flight_documents_reports_no_room() -> None:
    """The other half of the contract: `_prune` says so instead of evicting.

    The mint turns this False into a 503 and drops its own entry, so the caps
    still hold at the handler even though `_prune` alone can sit above them for
    the length of the grace window.
    """
    now = time.time()
    with sd._lock:
        for i in range(sd._MAX_ENTRIES + 5):
            sd._stash[f"d{i}"] = (now + sd._TOKEN_TTL_SECS, "x")
        assert sd._prune(now) is False, (
            "_prune claimed it made room while every entry was in flight — it "
            "must have evicted a document nobody had fetched yet"
        )


def test_expired_entries_are_pruned() -> None:
    now = time.time()
    with sd._lock:
        sd._stash["stale"] = (now - 1, "x")
        sd._stash["fresh"] = (now + 900, "y")
        sd._prune(now)
        assert "stale" not in sd._stash and "fresh" in sd._stash


def test_the_response_grants_only_what_the_embedding_frame_grants() -> None:
    """The header is what makes a same-origin URL safe for model-authored HTML.

    Without the ``sandbox`` directive, opening the URL top-level would run that
    HTML on the dashboard's own origin with its cookies and storage. The flags
    must also be no WIDER than the embedding iframes': a flag granted here but not
    on the frame lets a document opened top-level do something the same document
    cannot do inside the frame.
    """
    src = sd.__file__ or ""
    with open(src, encoding="utf-8") as fh:
        body = fh.read()
    csp_start = body.index('resp.headers["Content-Security-Policy"]')
    csp = body[csp_start : csp_start + 400]
    assert "sandbox allow-scripts" in csp, (
        "the response lost its sandbox directive — a top-level open would then "
        "run model HTML on the dashboard origin"
    )
    for flag in ("allow-popups", "allow-popups-to-escape-sandbox"):
        assert flag in csp, (
            f"the CSP sandbox is missing {flag}, which the embedding iframe grants; "
            "a narrower CSP silently removes a capability widgets already rely on"
        )
    assert "allow-forms" not in csp, (
        "the CSP grants allow-forms, which NEITHER embedding frame grants — a "
        "document opened top-level could then submit forms it cannot submit in "
        "the frame, which is wider than the surface being replaced"
    )
    assert "nosniff" in body and "no-store" in body


def test_an_in_flight_document_is_never_evicted_to_make_room() -> None:
    """The failure mode this replaces was silent, and reachable without an attacker.

    An entry is removed when it is SERVED, so every live entry is one nobody has
    fetched yet. Dropping the oldest to fit a newer one invalidates a URL some
    frame is about to load — and because the MINT succeeded, the frontend has no
    failure to show and that frame just renders blank. A gallery of image-bearing
    artifacts can push several MB of pending documents through at once, so this
    is normal use, not an edge case.
    """
    src = sd.__file__ or ""
    with open(src, encoding="utf-8") as fh:
        body = fh.read()
    prune = body[body.index("def _prune(") : body.index("def _audit(")]
    assert "_IN_FLIGHT_GRACE_SECS" in prune, (
        "eviction no longer protects an in-flight entry, so a busy gallery can "
        "silently blank an already-minted frame"
    )
    assert "popitem(last=False)" not in prune, (
        "unconditional oldest-first eviction is back — that is the shape that "
        "drops a document nobody has fetched yet"
    )
    mint = body[
        body.index("async def api_stash_sandbox_doc") : body.index("async def serve_sandbox_doc")
    ]
    assert "stash_full" in mint and "503" in mint, (
        "the mint no longer refuses when it cannot be given room; it must fail "
        "visibly rather than evict a live document"
    )


def test_the_stash_caps_still_bound_it() -> None:
    """Protecting in-flight entries must not turn the stash unbounded."""
    src = sd.__file__ or ""
    with open(src, encoding="utf-8") as fh:
        body = fh.read()
    assert "_MAX_ENTRIES" in body and "_MAX_BYTES" in body
    prune = body[body.index("def _prune(") : body.index("def _audit(")]
    assert (
        "_MAX_ENTRIES" in prune and "_MAX_BYTES" in prune
    ), "_prune stopped consulting one of its caps"

    """The control that actually holds when the client binding cannot.

    ``request.remote`` is the PROXY's address whenever the dashboard is reached
    through one, so every client shares it and the binding is worth nothing in
    exactly the deployments where a leaked URL is most reachable. Popping the
    entry before the body is written means a URL that leaks is already spent.
    """
    src = sd.__file__ or ""
    with open(src, encoding="utf-8") as fh:
        body = fh.read()
    serve = body[body.index("async def serve_sandbox_doc") :]
    assert "_stash.pop(doc_id, None)" in serve, (
        "the serving handler no longer consumes the entry, so an exfiltrated URL "
        "can be replayed by anyone sharing the requesting address"
    )
    assert "_stash.get(doc_id)" not in serve, (
        "the handler reads the entry without removing it — that is the replayable "
        "shape this test exists to prevent"
    )


def test_the_pop_happens_under_the_lock() -> None:
    """Two concurrent GETs must not both win: the pop is the atomic step."""
    src = sd.__file__ or ""
    with open(src, encoding="utf-8") as fh:
        body = fh.read()
    serve = body[body.index("async def serve_sandbox_doc") :]
    lock_at = serve.index("with _lock:")
    pop_at = serve.index("_stash.pop(doc_id, None)")
    assert lock_at < pop_at, "the consuming pop is outside the lock"


def test_the_serving_route_is_on_the_auth_bypass_list() -> None:
    """The token is the credential, so the route must bypass session auth — and
    the prefix in the middleware must be the one the handler actually serves."""
    from kiro_crew.dashboard import token_auth

    assert sd.SANDBOX_DOC_PREFIX in token_auth._BYPASS_PREFIXES, (
        "the document route is not on the bypass list, so a sandboxed frame "
        "without a cookie would get a 403 page instead of the document"
    )
