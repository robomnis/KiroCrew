/**
 * Video harness for the WHATSAPP CHANNEL pairing flow.
 *
 * The panel's states are photographed by `capture-whatsapp-panel.mjs`; this
 * records the TRANSITIONS between them, which a still cannot show: an operator
 * arriving at an unpaired channel, asking to pair, the QR card appearing with
 * its waiting spinner, and the panel settling into the connected view with the
 * access policy revealed.
 *
 * The QR is a code for fixed placeholder text, never a live pairing code: a real
 * one is a credential, and whoever scans it links their phone to the operator's
 * account.
 *
 * Emits a .webm from Playwright and, when ffmpeg is present, an .mp4 and a .gif
 * beside it, because a GitHub PR body embeds a raw-URL GIF but not a raw-URL
 * webm.
 *
 * Usage: node scripts/capture-whatsapp-panel-video.mjs [outDir]
 */
import { chromium } from 'playwright'
import { mkdirSync, renameSync, existsSync } from 'node:fs'
import { join } from 'node:path'
import { execFileSync } from 'node:child_process'
import { serveDist } from './lib/serve-dist.mjs'
import { logPageProblems, stubDashboardApi, json } from './lib/stub-dashboard-api.mjs'
import { QR_PNG } from './lib/whatsapp-fixtures.mjs'

const OUT = process.argv[2] || '../temp-screenshots/whatsapp-panel'
const LABEL = 'whatsapp-pairing-flow'
const VIEW = { width: 1280, height: 900 }

/** Mutable so the routes can advance the channel through its lifecycle. */
const state = {
  config: {
    configured: false,
    connected: false,
    connect_error: '',
    read_only: false,
    enabled: true,
    dm_policy: 'self',
    allowed_wa_ids: [],
    groups: [],
    session_folder: '',
    state: 'unpaired',
  },
  qr: { state: 'unpaired', qr_data_url: null, detail: '' },
}

function routes() {
  return async (path, route) => {
    if (path === '/api/whatsapp/config') return json(route, state.config), true
    if (path === '/api/channels/whatsapp/qr/status') return json(route, state.qr), true
    if (path === '/api/whatsapp/groups') return json(route, { groups: [] }), true
    if (path === '/api/governance/channels') return json(route, { whatsapp: true }), true
    if (path === '/api/channels/whatsapp/qr/start') {
      state.qr = { state: 'pairing', qr_data_url: QR_PNG, detail: '' }
      state.config = { ...state.config, state: 'pairing' }
      return json(route, { ok: true }), true
    }
    return false
  }
}

const wait = ms => new Promise(r => setTimeout(r, ms))

async function main() {
  mkdirSync(OUT, { recursive: true })
  const { srv, base } = await serveDist()
  const browser = await chromium.launch()
  const context = await browser.newContext({
    viewport: VIEW,
    deviceScaleFactor: 1, // a 2x video is needlessly heavy for a PR body
    recordVideo: { dir: OUT, size: VIEW },
  })
  const page = await context.newPage()
  logPageProblems(page)
  try {
    await stubDashboardApi(page, { theme: 'dark', extra: routes() })
    await page.goto(base + '/settings?tab=channels&channel=whatsapp', {
      waitUntil: 'domcontentloaded',
    })
    await page.getByText(/WhatsApp/).first().waitFor({ state: 'visible', timeout: 20000 })
    await wait(1600) // let the reader take in the unpaired state

    const pair = page.getByRole('button', { name: /Pair/ }).first()
    await pair.waitFor({ state: 'visible', timeout: 15000 })
    await pair.click()
    await wait(2600) // qr/start -> phase 'waiting' -> the card with the code

    // The phone scans: the channel comes up and the policy controls appear.
    state.config = { ...state.config, configured: true, connected: true, state: 'connected' }
    state.qr = { state: 'connected', qr_data_url: null, detail: '' }
    await wait(2600)

    // And the access policy an operator would set next.
    const policy = page.getByRole('combobox', { name: /Who can command the agent/ }).first()
    if (await policy.isVisible().catch(() => false)) {
      await policy.click()
      await wait(1200)
      const option = page.getByRole('option', { name: /allowed numbers/ }).first()
      if (await option.isVisible().catch(() => false)) {
        state.config = {
          ...state.config,
          dm_policy: 'allowlist',
          allowed_wa_ids: ['447700900111'],
        }
        await option.click()
      } else {
        await page.keyboard.press('Escape')
      }
      await wait(2200)
    }
  } finally {
    const video = page.video()
    await context.close() // finalizes the recording
    await browser.close()
    srv.close()
    if (video) {
      const src = await video.path()
      const webm = join(OUT, `${LABEL}.webm`)
      renameSync(src, webm)
      console.log('wrote', webm)
      try {
        const mp4 = join(OUT, `${LABEL}.mp4`)
        execFileSync('ffmpeg', ['-y', '-i', webm, '-movflags', 'faststart',
          '-pix_fmt', 'yuv420p', '-vf', 'scale=1280:-2', mp4], { stdio: 'ignore' })
        console.log('wrote', mp4)
        const gif = join(OUT, `${LABEL}.gif`)
        execFileSync('ffmpeg', ['-y', '-i', webm,
          '-vf', 'fps=8,scale=980:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse',
          gif], { stdio: 'ignore' })
        console.log('wrote', gif)
      } catch {
        console.log('ffmpeg unavailable or failed; the .webm alone is the artifact')
      }
      if (!existsSync(webm)) console.error('video missing after finalize')
    }
  }
}

main().catch(err => {
  console.error(err)
  process.exit(1)
})
