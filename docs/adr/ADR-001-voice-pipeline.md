# ADR-001: Voice Pipeline — Gemini Live AUDIO Modality vs. Client-Side STT/TTS

**Status:** Accepted  
**Date:** 2025-01  
**Deciders:** Software Architect, Team Claw

---

## Context

The AI Interview Platform was originally built as a text-only chat interface. The product requirement has evolved to support real-time voice-based interviews. Two fundamentally different approaches were available:

**Option A — Gemini Live AUDIO modality:**  
The browser captures raw microphone audio (PCM16 16 kHz), streams it over WebSocket to the server, which relays it to `gemini-2.0-flash-live-001` via `client.aio.live.connect()` with `response_modalities=["AUDIO"]`. Gemini returns audio bytes (PCM16 24 kHz) which are relayed back to the browser for playback.

**Option B — Client-side Web Speech API (STT → text → TTS):**  
The browser uses `SpeechRecognition` to transcribe the user's speech to text, sends that text through the existing WebSocket JSON protocol unchanged, receives the text response, and reads it aloud with `SpeechSynthesis`.

---

## Decision

**We chose Option A — Gemini Live AUDIO modality.**

---

## Consequences

### Why Option A wins

1. **Single-hop latency.** Option B stacks three sequential latency sources: STT processing time, Gemini text inference time, and TTS synthesis time. Option A processes speech end-to-end within Gemini Live in a single hop — typically 500–800 ms TTFA (time to first audio), versus 1.5–3 s for a stacked pipeline.

2. **Natural conversational prosody.** Gemini Live understands speech rhythm, pauses, and intonation — not just the words. It can detect an incomplete thought vs. a finished answer from prosody alone. A text-based pipeline loses all of this.

3. **Interruption / barge-in.** Gemini Live supports real-time interruption: if the user starts talking while the AI is responding, the AI stops mid-sentence. This is the defining UX feature of a real-time voice conversation. It is not achievable with Option B.

4. **No third-party STT/TTS dependency.** Option B would require either the browser's Web Speech API (inconsistent cross-browser, no offline support, privacy concerns with cloud STT) or a paid STT/TTS API (added cost + latency).

5. **The architecture doc already specifies `gemini-2.0-flash-live-001`.** This was the intended model from the start. The text implementation was a stepping stone.

### Trade-offs of Option A

- **More complex server code.** The WebSocket handler must multiplex binary audio frames with JSON control frames. Mitigated by keeping text and voice paths as separate service classes.
- **Binary WebSocket frames.** The existing all-JSON protocol must be extended. Mitigated by making `mode` opt-in with `"text"` as default — existing clients and tests are unaffected.
- **HTTPS required for mic access in production.** `getUserMedia()` is blocked by browsers on non-`localhost` HTTP origins. This is a deployment concern, not an architecture concern — all production deployments should use HTTPS.
- **Persistent upstream connection per session.** Gemini Live holds a WebSocket open for the session duration. More connection overhead than stateless HTTP calls. Acceptable for the expected concurrency of an interview platform.

### Why Option B was rejected

Option B does not deliver the stated requirement of *real-time voice*. Push-to-talk with STT→text→TTS produces an experience indistinguishable from text chat with an extra step. It adds browser API unreliability (Web Speech API varies significantly across browsers) while delivering none of the conversational naturalness that makes voice interviews valuable.

---

## Implementation Notes

- Text mode (`mode: "text"` in the `start` message) uses `GeminiInterviewService` and `generate_content_stream` — unchanged.
- Voice mode (`mode: "voice"`) uses the new `GeminiVoiceService` class and `aio.live.connect()`.
- The `mode` field defaults to `"text"` — full backward compatibility.
- See `docs/voice-architecture.md` for the complete technical design.
