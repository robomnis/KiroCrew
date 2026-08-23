/**
 * Pull the human-readable deny reason out of a blocked tool row's content.
 *
 * When a security-policy rule or a PreToolUse hook blocks a tool call, the
 * gateway appends a second tool message sharing the pill's `tool_call_id`:
 *
 *     🚫 Running: <command> — Blocked by security policy: <pattern>
 *     <why the pattern fired, when the match was structural>
 *
 * The Output panel used to discard that content and show a fixed
 * "blocked by security policy" line, because the row could arrive carrying only
 * a bare title: a later `tool_call_update` title refinement rewrote the row as
 * `"<icon> <title>"`, deleting the reason. With that rewrite fixed backend-side
 * the content is dependable, so the reason can be shown instead of a placeholder
 * that tells the user nothing about WHICH rule fired or why.
 *
 * Keyed on the backend's own `DENY_REASON_PREFIX` (`security.py`), which is the
 * one part of the string with a contract — `RecoveryCard` matches the same
 * marker. Everything from it to the end is returned, so the second line (the
 * argv-structural explanation) survives; a row without the marker yields "" and
 * the caller keeps its placeholder.
 *
 * Held as a REGEX, not a string constant, for the same reason `RecoveryCard`'s
 * `POLICY_RE` is: this is a WIRE VALUE matched byte-for-byte against a Python
 * constant, never copy. As a string literal inside an ALL-CAPS module constant
 * the i18n gate reads it as untranslated UI text and asks for a catalog key —
 * and translating it would silently stop every deny reason from being found.
 */
const DENY_REASON_MARKER = /Blocked by security policy:/

export function extractDenyReason(rowContent: string): string {
  if (!rowContent) return ''
  const found = rowContent.match(DENY_REASON_MARKER)
  if (!found || found.index === undefined) return ''
  const reason = rowContent.slice(found.index).trim()
  // A marker with nothing after it is a placeholder, not a reason — let the
  // caller's own localized placeholder win rather than rendering a bare colon.
  return reason === found[0] ? '' : reason
}
