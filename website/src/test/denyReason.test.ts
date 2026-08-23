import { describe, it, expect } from 'vitest'
import { extractDenyReason } from '../utils/denyReason'

// The Output panel of a blocked tool call used to show a fixed
// "blocked by security policy" line and discard the row's real content, so the
// user could never see WHICH rule fired. These pin the extraction that replaces
// that placeholder, and the cases where the placeholder must still win.
describe('extractDenyReason', () => {
  const ROW =
    '🚫 Running: python3 -c "import x" — Blocked by security policy: ' +
    'kiro[-.]?crew\\b[^|;&#>/*]*\\btoken\\b\n' +
    "Matched structurally on the command's argv, not by the pattern text above."

  it('returns the reason starting at the contract marker', () => {
    expect(extractDenyReason(ROW)).toMatch(/^Blocked by security policy:/)
  })

  it('drops the row title so the panel shows the reason, not the command', () => {
    const out = extractDenyReason(ROW)
    expect(out).not.toContain('🚫')
    expect(out).not.toContain('Running:')
  })

  it('keeps the pattern that fired', () => {
    expect(extractDenyReason(ROW)).toContain('\\btoken\\b')
  })

  it('keeps the second explanation line', () => {
    // The structural note is the part that makes a floor hit intelligible; a
    // single-line extraction would silently drop exactly that.
    expect(extractDenyReason(ROW)).toContain('Matched structurally')
  })

  it('yields empty for a row with no reason, so the placeholder wins', () => {
    expect(extractDenyReason('🚫 shell (hook blocked)')).toBe('')
    expect(extractDenyReason('🚫 shell')).toBe('')
  })

  it('yields empty for a bare marker rather than rendering a lone colon', () => {
    expect(extractDenyReason('🚫 shell — Blocked by security policy:')).toBe('')
  })

  it('handles an absent row', () => {
    expect(extractDenyReason('')).toBe('')
  })
})
