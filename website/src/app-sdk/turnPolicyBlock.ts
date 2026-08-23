import type { ChatMessage } from '../types'

/**
 * The gateway's display-only marker for a policy block whose reason was steered
 * into the running turn. WIRE VALUE, matched byte-for-byte against
 * `REFUSAL_INBAND_RECOVERY_PREFIX` in `src/kiro_crew/dashboard/state.py` — never
 * rendered, never translated.
 */
const TOOL_BLOCKED_PREFIX = '[Tool blocked — reason sent to the agent]'

/**
 * True when the turn containing `index` had a tool call blocked by policy.
 *
 * Used to drop the generic "Steered" chip from that turn's reply. The chip is
 * correct for a steer the PERSON sent and wrong here: the same mechanism carries
 * a system policy notice, so the chip reads as though the user had steered the
 * turn — the exact misattribution the notice exists to correct. The blocked-tool
 * card already states what happened, so the chip is not just ambiguous but
 * redundant.
 *
 * Scanning backwards to the turn head means the answer is identical while the
 * reply is still streaming and after a history reload: the blocked row and the
 * notice row are both appended at deny time, which is before any of the reply's
 * text arrives. A `meta` flag on the assistant row could not do this — the
 * streaming row has no persisted meta yet, so the chip would appear live and
 * vanish on refresh.
 *
 * A turn that ALSO carries a steer the person sent keeps its chip: their steer
 * deserves its acknowledgement, and suppressing on the presence of a policy
 * notice alone would silently swallow it.
 */
export function turnHadPolicyBlock(messages: ChatMessage[], index: number): boolean {
  let blocked = false
  for (let i = Math.min(index, messages.length - 1); i >= 0; i--) {
    const m = messages[i]
    if (!m) continue
    // Turn head: a real user message. A steer is also role `user`, so it must
    // NOT end the scan — it lives INSIDE the turn it was injected into.
    if (m.role === 'user') {
      if ((m.meta as Record<string, unknown> | undefined)?.steer) return false
      break
    }
    if (m.role === 'inject' && m.content?.startsWith(TOOL_BLOCKED_PREFIX)) blocked = true
  }
  return blocked
}
