# Voice UI — UX Design (v2 — Full Duplex)

## Feature: Real-Time Full-Duplex Voice Interview Mode

### User Story
As a candidate, I want to speak and hear the AI interviewer simultaneously — like a real phone call —
so that the practice session feels natural and uninterrupted, with no walkie-talkie turn-taking.

---

### What Changed in v2 (Full Duplex)
| Was (v1 push-to-talk) | Now (v2 full duplex) |
|---|---|
| Mic button = click to start/stop recording | Mic button = mute/unmute (mic is ON by default when call starts) |
| "SPEAK" / "RECORDING" states on button | "Muted" / "Live" states on button |
| Turn-based: AI speaks → user speaks | Both directions flow continuously and simultaneously |
| Single visualizer (recording XOR ai-speaking) | Dual visualizer: user mic level + AI level shown simultaneously |
| No call timer | Elapsed call timer always visible ("In Call • 0:42") |
| UI implies turn-taking ("wait for AI", "now speak") | UI implies a live call — no turn cues |
| `#recording-indicator` shows/hides on user action | `#recording-indicator` repurposed as mic-live indicator (always shown during call) |

---

### User Flow

```
SETUP SCREEN
  ↓
  [Mode selector: "Text Interview" (default) | "Voice Interview"]
  ↓ (user picks role, selects Voice, clicks Start)
  [Start Interview] →
                       VOICE mode: chat screen + voice call bar
                                   mic activates immediately
                                   call timer starts

VOICE CALL — IN PROGRESS
  ↓
  Both audio directions active simultaneously:
    User speaks any time  → mic-visualizer pulses (blue)
    AI speaks any time    → ai-visualizer pulses  (green)
    (Can overlap — this is the whole point)

VOICE CALL CONTROLS
  [🎤 Live]   → click → [🚫 Muted] (mic silenced; AI can still speak)
  [🚫 Muted]  → click → [🎤 Live]  (mic re-enabled)
  [🔇 Mute AI] → click → [🔊 Unmute AI] (AI audio silenced)
  [End Call]  → confirms → session ends, return to setup

ERROR STATES
  Mic permission denied → error banner, mic shows "Denied" state
  WS disconnect         → same reconnect prompt as text mode
  AI audio error        → toast; call continues (user can still speak)
```

---

### Screens & Wireframes

#### Screen 1: Setup — Interview Mode Selector (unchanged from v1)

```
┌─────────────────────────────────────────────────┐
│  🤖  AI Interview Platform                      │
├─────────────────────────────────────────────────┤
│                                                 │
│   ╔═════════════════════════════════════════╗   │
│   ║  Ace Your Next Interview               ║   │
│   ║  Practice with an AI interviewer…      ║   │
│   ╠═════════════════════════════════════════╣   │
│   ║                                        ║   │
│   ║  Your Name (optional)                  ║   │
│   ║  [________________________]            ║   │
│   ║                                        ║   │
│   ║  Interview Role *                      ║   │
│   ║  [  Select a role…             ▾ ]     ║   │
│   ║                                        ║   │
│   ║  Interview Mode                        ║   │
│   ║  ┌─────────────┐  ┌─────────────────┐ ║   │
│   ║  │ 💬 Text     │  │  🎙 Voice       │ ║   │
│   ║  │  Interview  │  │   Interview     │ ║   │
│   ║  └─────────────┘  └─────────────────┘ ║   │
│   ║    ● selected         ○ not selected  ║   │
│   ║                                        ║   │
│   ║  [ 🎙  Start Interview              ]  ║   │
│   ╚═════════════════════════════════════════╝   │
└─────────────────────────────────────────────────┘
```

**Mode Selector behaviour (unchanged):**
- Two-button radio group (`role="radiogroup"`)
- Text Interview selected by default
- When Voice is selected: shows mic-tip "Your microphone will be activated on start"
- Start button label/icon reflects selected mode

---

#### Screen 2: Chat Screen — Full-Duplex Voice Call Active

```
┌──────────────────────────────────────────────────────┐
│  🤖 AI Interview Platform   [Software Engineer]      │
│                              ● Connected  🎙 Voice   │
├──────────────────────────────────────────────────────┤
│                                                      │
│  🤖  ┌────────────────────────────────────────┐     │
│       │ Tell me about a challenging project   │     │
│       │ you led recently.            10:32 AM │     │
│       └────────────────────────────────────────┘     │
│                                                      │
│       ┌─────────────────────────────────┐       🧑  │
│       │ I led the migration of our…     │           │
│       │                       10:33 AM  │           │
│       └─────────────────────────────────┘           │
│                                                      │
├──────────────────────────────────────────────────────┤
│  VOICE CALL BAR (full-duplex — always-on during call)│
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │  🔴 In Call  •  1:42          [End Call ✕]   │   │  ← call-status-row
│  ├──────────────────────────────────────────────┤   │
│  │                                              │   │
│  │    ┌─────────────────┐  ┌─────────────────┐  │   │
│  │    │   YOUR MIC      │  │    AI VOICE     │  │   │
│  │    │  ▮▮▮▮▯▯▯▯▯▯▯▯  │  │  ░░░▒▒▓▓▒▒░░░  │  │   │  ← dual visualizer
│  │    │  (blue waveform)│  │  (green waveform)│  │   │
│  │    └─────────────────┘  └─────────────────┘  │   │
│  │                                              │   │
│  │    [🎤 Live]              [🔊 AI Audio]       │   │  ← control row
│  │     (click to mute)       (click to mute AI) │   │
│  │                                              │   │
│  └──────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────┘

MUTED MIC STATE:
  [🚫 Muted]   ← red background, strikethrough mic icon
                  AI can still speak; user's audio is suppressed

AI MUTED STATE:
  [🔇 AI Muted] ← same muted visual as speaker-btn v1

BOTH ACTIVE SIMULTANEOUSLY:
  ┌─────────────────┐  ┌─────────────────┐
  │   YOUR MIC      │  │    AI VOICE     │
  │  ▮▮▮▮▮▮▯▯▯▯    │  │  ░▒▓▓▓▒░░░░░░  │  ← BOTH pulse at the same time
  └─────────────────┘  └─────────────────┘
```

**Full-duplex call bar layout:**
- **Top row** — red "In Call" dot + elapsed timer (counts up from 0:00) + "End Call" button right-aligned
- **Middle section** — two visualizer panels side by side, equal width
  - Left: user's mic activity (blue bars, CSS animation; driven by real AnalyserNode data if available)
  - Right: AI's voice activity (green bars, CSS animation; driven by `audio_start`/`audio_end` events)
  - Both panels are always visible during the call; they pulse independently
- **Bottom row** — two control buttons: mic mute toggle (left) + AI audio mute toggle (right)

---

#### Screen 2b: Chat Screen — Text Mode (unchanged)

```
┌──────────────────────────────────────────────────────┐
│  🤖 AI Interview Platform   [Software Engineer]      │
│                              ● Connected  💬 Text    │
├──────────────────────────────────────────────────────┤
│  [chat messages — unchanged]                         │
├──────────────────────────────────────────────────────┤
│  [ Type your answer…                      ] [Send ▶] │
│  [✕ End Interview]                         0/2000    │
└──────────────────────────────────────────────────────┘
```

---

### Component Specs

| Component | Element / ID | States | Behaviour |
|-----------|-------------|--------|-----------|
| **Mode selector** | `#interview-mode` div with `role="radiogroup"` | `text-selected` (default), `voice-selected` | Clicking a button adds `.mode-btn--active`, updates `data-mode` on container. No submit. |
| **Call status row** | `#voice-call-status` | `in-call` (visible during voice session), `hidden` (before/after call) | Shows red dot + "In Call" label + elapsed timer `#call-timer` + "End Call" button. Timer starts when WS opens in voice mode. JS updates timer text every second. |
| **Call timer** | `#call-timer` | ticking (voice session active), `0:00` (reset) | `role="timer"`, `aria-live="off"` (not announced every second — that would be noisy). Format: `M:SS` up to `59:59`, then `H:MM:SS`. |
| **Mic mute button** | `#mic-btn` | `live` (default — mic is ON), `muted` (user pressed to silence mic), `denied` (permission error) | **live:** green border, mic icon, label "Live". **muted:** red bg, slashed mic icon, label "Muted". **denied:** grey, disabled. Does NOT start/stop recording — mic pipeline is always active; this button only sets `isRecording = false` (stops sending chunks). |
| **User mic visualizer** | `#audio-visualizer` (div with 5 viz-bars) | `active` (always during call), `idle` (flat, very low animation) | Blue bars. CSS animation always runs during call; bar heights driven by AnalyserNode data if available, otherwise pure CSS animation. No `hidden` toggle — always visible while voice bar is shown. |
| **AI voice visualizer** | `#ai-speaking-indicator` (repurposed) | `speaking` (green bars animate fast), `silent` (bars flatline / very low animation) | Green bars. Animates when `isAISpeaking === true`. JS adds/removes `.speaking` class (not `.active` show/hide). Both visualizers coexist always. |
| **AI mute button** | `#speaker-btn` | `on` (default), `muted` | Icon 🔊 when on, 🔇 when muted. `.muted` class toggle. Circular, 44px touch target. |
| **End call button** | `#end-btn-voice` | default, hover, disabled | Danger-outline style. Positioned in call-status top row (right-aligned). Triggers `endInterview()`. |
| **Voice error banner** | `#voice-error-banner` | `hidden` (default), `visible` | Error text updated by JS. Shown above control row when mic denied or stream lost. |
| **Voice mode badge** | `#voice-mode-toggle` | `voice`, `text` | Pill in chat header. Reflects current mode. Not a primary control in full duplex (no in-session mode switching during a live call — confusing). |

---

### DOM IDs Reference (for JavaScript binding)

```html
<!-- REQUIRED IDs — do not rename; JS binds to these exactly -->

<!-- Setup screen -->
#interview-mode         — radiogroup container; data-mode="text"|"voice"

<!-- Chat header -->
#voice-mode-toggle      — mode badge pill

<!-- Voice call bar (replaces text bar in voice mode) -->
#voice-call-status      — top row: red dot + timer + End Call btn
#call-timer             — elapsed seconds span inside #voice-call-status
#audio-visualizer       — user mic visualizer (div, 5×.viz-bar spans)
                          NOTE: app.js _startVisualizer() calls .getContext('2d')
                          on this element — expects a <canvas>. If a <canvas> is
                          used, CSS bar animation is replaced by JS canvas drawing.
                          DECISION: Use <canvas id="audio-visualizer"> and let
                          app.js drive the waveform. For the AI side, keep the
                          CSS-bar visualizer in a separate element.
#ai-speaking-indicator  — AI voice visualizer (div, 5×.ai-bar spans, CSS-only)
#mic-btn                — mic mute/unmute toggle (NOT start/stop recording)
#speaker-btn            — AI audio mute/unmute toggle
#voice-error-banner     — mic permission / stream error message

<!-- Existing IDs (unchanged) -->
#setup-screen, #chat-screen, #setup-form, #start-btn
#user-name, #role-select
#chat-messages, #message-input, #send-btn, #end-btn
#typing-indicator, #connection-status, #current-role, #char-count
#end-btn-voice          — End Interview in voice mode (in call-status row)
#recording-indicator    — DEPRECATED in v2; keep element for JS compat but
                          repurpose label text to "● Mic Live" vs "🚫 Muted"
```

**⚠️ CRITICAL: `#audio-visualizer` must be `<canvas>` element**
app.js `_startVisualizer()` calls `audioVisualizer.getContext('2d')` — it treats
`#audio-visualizer` as a `<canvas>`. If it is a `<div>`, getContext returns null
and the visualizer silently fails. The HTML MUST use `<canvas id="audio-visualizer">`.
The AI visualizer (`#ai-speaking-indicator`) is a separate `<div>` with CSS bars.

---

### Interaction Notes

#### Mic button — full-duplex semantics
- **On voice session start:** mic pipeline opens immediately. `isRecording = true` automatically. Mic button shows "Live" state (green border, 🎤 icon).
- **User clicks mic button:** toggles `isRecording`. When muted: button turns red, icon changes to 🚫, label "Muted", audio chunks stop being sent. Mic stream stays open.
- **User clicks mic button again:** `isRecording = true` resumes, button returns to "Live" state.
- **Mic denied:** button shows "Denied" state (grey, disabled), error banner shown.
- **There is no "waiting" or "AI is speaking, please wait" state.** The mic is always open unless the user explicitly mutes it.

#### Dual visualizers — simultaneous activity
- **User mic visualizer** (`#audio-visualizer` — canvas): real waveform from AnalyserNode. Blue colour. Runs continuously while voice session is active.
- **AI voice visualizer** (`#ai-speaking-indicator` — CSS bars): green, animates only when `isAISpeaking === true`. When AI is silent, bars flatline (very small height, slow animation). JS toggles `.speaking` class on this element.
- **Both can be animated at the same time.** This is the primary visual difference from v1.

#### Call timer
- Starts counting when WS `open` fires in voice mode.
- `setInterval` every 1 second, updates `#call-timer` text content.
- Cleared in `_cleanupVoice()` and `resetToSetup()`.
- Format: `0:00` → `1:23` → `10:05`.

#### AI speaking indicator — v2 behaviour
- Does NOT show/hide (no `hidden` toggle). Always visible in voice call bar.
- JS toggles `.speaking` class on `#ai-speaking-indicator` div.
- `.speaking` class: green bars animate at full height, fast.
- Without `.speaking`: bars flatline at 15% height, slow idle animation (shows visual continuity).

#### End call
- "End Call" button is in the call-status row (top of voice bar), right-aligned.
- Same `endInterview()` function as text mode.
- Confirm dialog: "Are you sure you want to end the call?"

#### Loading state (while waiting for first AI word)
- Existing `#typing-indicator` is still used — consistent with text mode.
- Once AI audio starts (`audio_start` message), typing indicator hides and AI visualizer activates.

#### Error states
- **Mic permission denied:** `#voice-error-banner` shown. Mic button shows "Denied" state (opacity 0.4, `aria-disabled`). Call can continue — AI can still ask questions; user must answer in text (graceful degradation).
- **AI audio error:** toast-style banner in chat (3s auto-dismiss). Call continues.
- **WS disconnect:** same reconnect prompt as text mode.

#### No "turn-taking" UI
- Remove any UI that says "Now speaking" / "Listening…" / "Please wait".
- The call-status row shows "🔴 In Call" for the entire duration — no state changes.
- The AI visualizer pulsing green is the only signal that AI is speaking. User does not need to wait for it to stop.

#### Mobile (thumb-reach)
- Mic mute and AI mute buttons are side-by-side in a centred row — within thumb reach on mobile.
- End Call button is in the top row — requires deliberate reach (intentional, prevents accidental end).
- Min touch target: 44×44px all controls.

---

### CSS Class Contract (JS ↔ CSS)

| JS action | CSS change | Visual result |
|---|---|---|
| Voice session starts | Remove `hidden` from `#voice-input-bar`; add `hidden` to `#text-input-bar` | Voice call bar appears |
| `isRecording = true` (mic live) | `#mic-btn`: no special class (default = live) | Green border, 🎤 icon |
| `isRecording = false` (user muted) | `#mic-btn.muted` | Red bg, 🚫 icon |
| Mic denied | `#mic-btn.denied` + `aria-disabled` | Grey, 0.4 opacity |
| `isAISpeaking = true` | `#ai-speaking-indicator.speaking` | Green bars animate at full height |
| `isAISpeaking = false` | `#ai-speaking-indicator` (no class) | Green bars at idle flatline |
| Speaker muted | `#speaker-btn.muted` | Red bg 🔇 icon |

---

### Color & Typography

Inherits all existing tokens. Voice-specific additions:

| Token | Value | Usage |
|-------|-------|-------|
| `--color-mic-live` | `#16A34A` (green-600) | Mic button border/bg when active and live |
| `--color-mic-live-border` | `rgba(22,163,74,0.40)` | Mic live state border glow |
| `--color-mic-muted` | `#DC2626` (danger) | Mic button when user muted |
| `--color-viz-user` | `#2563EB` (primary) | User mic waveform / canvas stroke |
| `--color-viz-ai` | `#22C55E` (connected) | AI voice bars |
| `--color-call-dot` | `#EF4444` | "In Call" live indicator dot |
| `--color-call-dot-ring` | `rgba(239,68,68,0.25)` | Pulsing ring around call dot |
| `--mic-btn-size` | `52px` | Mic mute button (smaller than v1 — not the hero control) |
| `--speaker-btn-size` | `52px` | Speaker mute button (same size as mic for visual balance) |

Typography: same Inter stack, all sizes from existing tokens.

---

### Accessibility Checklist

- [ ] `#interview-mode` has `role="radiogroup"` + `aria-labelledby="interview-mode-label"`
- [ ] Each mode button has `role="radio"` + `aria-checked="true|false"`
- [ ] `#mic-btn` has `aria-label="Mute microphone"` / `"Unmute microphone"` (toggled by JS)
- [ ] `#mic-btn` has `aria-pressed="false"` (false = live/unmuted) / `"true"` (muted)
- [ ] `#audio-visualizer` (`<canvas>`) has `aria-hidden="true"` (decorative)
- [ ] `#ai-speaking-indicator` has `role="status"` + `aria-live="polite"` + `aria-label` toggled JS
- [ ] `#speaker-btn` has `aria-label="Mute AI audio"` / `"Unmute AI audio"` + `aria-pressed`
- [ ] `#call-timer` has `role="timer"` + `aria-live="off"` (not announced each second)
- [ ] `#voice-call-status` has `aria-label="Voice call in progress"`
- [ ] All voice controls meet 44×44px minimum touch target
- [ ] Muted state: not conveyed by colour alone — icon AND label both change

---

### Open Questions

- [ ] **Canvas vs CSS bars for user mic visualizer:** app.js `_startVisualizer()` draws a waveform to a `<canvas>`. The HTML must use `<canvas id="audio-visualizer">` — CSS bar animation won't work here. AI visualizer uses CSS bars (`#ai-speaking-indicator`). This split is intentional.
- [ ] **Mic auto-start:** Should `isRecording` be set to `true` automatically when the voice WS opens, or should the user click "Go Live" first? Current design: auto-start (matches phone-call metaphor). If the team wants a "ready" pre-call screen, raise with PO.
- [ ] **In-session mode switching:** v1 had a badge to switch between text↔voice mid-session. Removed from v2 primary flow — switching modes during a live call is confusing. `#voice-mode-toggle` badge still present in header but its click handler should not be active during a call.
