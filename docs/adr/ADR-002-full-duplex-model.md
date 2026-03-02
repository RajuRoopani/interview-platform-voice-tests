# ADR-002: Full-Duplex Voice Model — Concurrent Tasks vs. Turn-Based Protocol

**Status:** Accepted  
**Date:** 2025-01  
**Supersedes:** v1.0 push-to-talk design (see voice-architecture.md §Changelog)  
**Deciders:** Software Architect, Team Claw

---

## Context

The initial voice architecture (v1.0) used a **push-to-talk, turn-based model**:

- The user pressed a button to activate the mic (`voice_start_speaking`)
- The server relayed audio until the user released (`voice_stop_speaking`)
- The server sent `ai_speaking_start` / `ai_speaking_end` signals
- The browser **muted the mic** while the AI was speaking to prevent echo
- The protocol was effectively half-duplex: only one party spoke at a time

The user requirement was subsequently clarified explicitly:

> *"It should be full duplex."*

Full duplex means both parties can speak simultaneously, like a phone call — the user can interrupt the AI mid-sentence and the AI will stop and respond to the interruption.

This is a fundamentally different concurrency model, not a minor tweak. It required revisiting the server async model, the WebSocket protocol, and the browser audio strategy.

---

## Decision

**Adopt a fully concurrent dual-task model on the server and an always-on mic in the browser.**

### Server: `asyncio.gather()` over two independent coroutines

```
asyncio.gather(
    _ws_receive_loop(websocket, svc),    # Task A: browser → Gemini
    _gemini_receive_loop(websocket, svc) # Task B: Gemini → browser
)
```

- Task A reads binary audio frames from the browser WebSocket and forwards them to `svc.send_audio()` in a tight loop — no waiting for Task B.
- Task B reads from `svc.receive()` (a streaming async generator over the Gemini Live connection) and forwards audio bytes and transcript frames to the browser — no waiting for Task A.
- Neither task blocks or gates the other. Both yield to the asyncio event loop on every `await`.
- `gather()` provides coordinated teardown: if either task exits (browser disconnect, session end, exception), the other is immediately cancelled and `svc.close()` is called in the `finally` block.

### Browser: always-on mic from session start to session end

- `getUserMedia()` is called once at session start with `echoCancellation: true`.
- The `AudioWorklet` streams PCM16 chunks continuously via `ws.send()`.
- There is no "mic active" toggle, no mic mute event, no `isSpeaking` state flag.

### VAD and barge-in: delegated entirely to Gemini Live

- Gemini Live's built-in Voice Activity Detection determines when the user has finished speaking (end-of-turn) without any explicit signal from the browser.
- Gemini Live's built-in barge-in detection interrupts its own response when user audio is detected, emitting `server_content.interrupted = true`.
- The server receives the `interrupted` flag and forwards `{ type: "ai_interrupted" }` to the browser.
- The browser resets `nextPlayTime = audioContext.currentTime`, flushing the pending playback queue.

---

## Consequences

### What this enables

1. **Natural conversation flow.** The user can interrupt the AI at any time. The AI stops within 1–2 RTTs of the interruption being detected.
2. **No push-to-talk UX friction.** There is no button to press/hold. The interface behaves like a phone call.
3. **No echo problem from mic muting.** Browser-native AEC (`echoCancellation: true`) handles echo cancellation hardware-side, which is more robust than software mic gating.
4. **Clean server model.** Two simple loops, clearly separated by direction. Easy to reason about, test, and debug.

### Trade-offs accepted

1. **`asyncio.gather()` error handling is strict.** If either task raises an unexpected exception, both tasks are cancelled. This is intentional — a half-broken session (one direction down) is worse than a clean teardown. All expected exceptions (`WebSocketDisconnect`, session end) must be caught inside the loops, not propagated.

2. **Browser AEC dependency.** The always-on mic design relies on `echoCancellation: true` working well in the browser. On devices without hardware AEC (some older mobile devices), the AI may hear its own audio through the mic, causing feedback loops. Mitigation: test on target devices; if needed, add amplitude gating in the AudioWorklet as a fallback.

3. **Higher connection overhead.** Both the browser-to-server WS and the server-to-Gemini Live WS are continuously active for the session duration. This is inherent to full duplex and is acceptable for the interview platform's expected concurrency.

4. **Removed `voice_start_speaking` / `voice_stop_speaking` protocol messages.** Code that previously depended on these events (notably v1.0 UX code) must be updated. These signals are silently ignored if received in v2.0, preventing hard breakage.

### Alternatives rejected

**Alternative: Two separate WebSocket connections (one per direction)**  
Rejected: doubled connection overhead, more complex browser code, and session lifecycle synchronization between two sockets is error-prone.

**Alternative: Server-side VAD with Silero or WebRTC VAD**  
Rejected: adds a native library dependency, increases server CPU load, and duplicates functionality already built into Gemini Live. Gemini Live's VAD is tuned to its own audio model — a separate VAD would introduce desynchronization artifacts.

**Alternative: Retain mic muting, add barge-in via amplitude threshold**  
Rejected: mic muting is fundamentally incompatible with full duplex. If the mic is muted while AI speaks, the user cannot interrupt. The entire point of full duplex is that both streams are open simultaneously.

---

## Implementation Checklist

- [x] **`response_modalities`** — use `["AUDIO", "TEXT"]`. TEXT modality enables transcript output from Gemini alongside audio. The `_gemini_receive_loop` `if response.text:` branch is already wired to forward transcript frames to the browser. *(Decision confirmed 2025-01)*
- [ ] `GeminiVoiceService.send_audio()` is safe to call concurrently with `receive()` (separate send/receive paths in the Gemini Live SDK — verify this assumption)
- [ ] `asyncio.gather()` teardown tested: simulate browser disconnect mid-session and verify Gemini Live session is closed within 5 s
- [ ] `ai_interrupted` → browser queue flush tested: verify `nextPlayTime` reset stops audio within one buffer period (~256 ms)
- [ ] Browser AEC tested on Chrome, Firefox, Safari with headphones and speakers
- [ ] `voice_start_speaking` / `voice_stop_speaking` received in voice mode → silently ignored (not an error)

---

*See also: `docs/voice-architecture.md` for full design including audio formats, protocol, and file change summary.*
