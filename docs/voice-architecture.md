# Voice-Based Real-Time Interview — Architecture Update

> **Status:** Approved for implementation  
> **Version:** 2.0 — Full Duplex revision  
> **Supersedes:** v1.0 (push-to-talk design)  
> **Extends:** `docs/architecture.md`  
> **Date:** 2025-01  
> **Author:** Software Architect, Team Claw

---

## Changelog

| Version | Change |
|---------|--------|
| v1.0 | Initial voice design — push-to-talk, mic muted during AI speech |
| **v2.0** | **Full-duplex redesign — continuous mic, concurrent async tasks, no turn-taking** |
| **v2.1** | **Resolved Open Question #1: `response_modalities=["AUDIO","TEXT"]` — confirmed** |

---

## 1. Design Decision: Which Voice Pipeline?

Two options were evaluated:

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **A — Gemini Live AUDIO modality** | Browser captures mic → PCM16 frames sent via WS to server → server relays to `gemini-2.0-flash-live-001` with `AUDIO` input + `AUDIO` output → server relays audio bytes back to browser | Native understanding of speech cadence, prosody, interruption; single RTT hop; full-duplex built-in | Requires server-side audio relay; binary WS frames; more complex browser playback pipeline |
| **B — Client-side STT → text → TTS** | Browser `SpeechRecognition` API → text → existing WS → text response → browser `SpeechSynthesis` | Reuses existing text pipeline unchanged; simpler server | Double latency (STT + TTS), no barge-in, turn-based only — **cannot satisfy full-duplex requirement** |

**Decision: Option A — Gemini Live AUDIO modality.**

Critically, Option B is architecturally incapable of full duplex. `SpeechRecognition` is turn-based; it cannot capture audio while audio is playing. Gemini Live's `AUDIO` modality was purpose-built for simultaneous bidirectional audio.

See `docs/adr/ADR-001-voice-pipeline.md` and `docs/adr/ADR-002-full-duplex-model.md`.

---

## 2. Full-Duplex Model — Core Principle

> **Both audio directions are ALWAYS open simultaneously. There is no turn-taking, no mic mute, no "wait for AI to finish" gate.**

This is the critical architectural difference from v1.0:

| Concern | v1.0 (push-to-talk) | **v2.0 (full duplex)** |
|---------|---------------------|------------------------|
| Mic state | Toggled on/off by user button | **Always on** after session start |
| AI speech gate | Mic muted while AI speaks | **Mic never muted** |
| User interruption | Not supported | **Built-in: user speech stops AI** |
| Turn signaling | `voice_start_speaking` / `voice_stop_speaking` events | **Removed — Gemini Live VAD handles turn detection** |
| Async model | Sequential receive → process → respond | **Two concurrent tasks: send loop + receive loop** |

---

## 3. High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  BROWSER                                                                      │
│                                                                               │
│  ┌──────────────────────────────┐    ┌──────────────────────────────────┐    │
│  │  MIC CAPTURE (always active) │    │  AUDIO PLAYBACK                  │    │
│  │                              │    │                                  │    │
│  │  MediaStream (getUserMedia)  │    │  AudioContext                    │    │
│  │  └─ AudioWorkletNode         │    │  └─ BufferSourceNode queue       │    │
│  │     └─ Float32→Int16 PCM16   │    │     └─ nextPlayTime accumulator  │    │
│  └────────────┬─────────────────┘    └──────────────┬───────────────────┘    │
│               │ binary frames (continuous)           ▲ binary frames          │
│               │                                      │                        │
│               └────────────────┬─────────────────────┘                        │
│                                │  WebSocket /ws/interview                     │
│                                │  (binary = audio, string = JSON control)     │
└────────────────────────────────┼─────────────────────────────────────────────┘
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│  FASTAPI SERVER  (backend/main.py)                                              │
│                                                                                 │
│  interview_ws()  ────  on "start" / mode="voice"                                │
│       │                                                                         │
│       ├── Task A: _ws_receive_loop()    ← reads binary + JSON from browser     │
│       │       │  binary frame → svc.send_audio(pcm_bytes)                      │
│       │       │  JSON "end"   → svc.close()                                    │
│       │                                                                         │
│       └── Task B: _gemini_receive_loop()  ← reads from Gemini Live             │
│               │  audio frame  → ws.send_bytes(pcm_bytes)   → browser           │
│               │  text frame   → ws.send_json(transcript)   → browser           │
│               │  turn_complete→ ws.send_json(ai_turn_end)  → browser           │
│                                                                                 │
│  Both tasks run concurrently via asyncio.gather() — neither blocks the other   │
│                                                                                 │
└───────────────────────────────────────┬────────────────────────────────────────┘
                                    │ Gemini Live WebSocket (persistent)
                                    ▼
                       ┌────────────────────────────┐
                       │  gemini-2.0-flash-live-001  │
                       │  input:  AUDIO (PCM16 16k)  │
                       │  output: AUDIO (PCM16 24k)  │
                       │  VAD:    built-in            │
                       │  barge-in: built-in          │
                       └────────────────────────────┘
```

---

## 4. Concurrent Task Model (Server)

This is the heart of the full-duplex design. The server runs **two independent asyncio tasks per voice session**:

```
interview_ws() coroutine
        │
        ├─ awaits session start message
        ├─ creates GeminiVoiceService, calls svc.connect()
        ├─ sends session_ready to browser
        │
        └─ asyncio.gather(
               _ws_receive_loop(websocket, svc),    # Task A
               _gemini_receive_loop(websocket, svc) # Task B
           )
           # gather() exits when EITHER task raises or returns
           # finally: svc.close()
```

### Task A — `_ws_receive_loop` (Browser → Gemini)

```
loop:
  message = await websocket.receive()   # blocks until next WS frame
  if bytes:  await svc.send_audio(message["bytes"])
  if text:
    type == "end"  → break (signals graceful session end)
    type == "ping" → send pong (keepalive)
    other          → log and ignore (no turn-signaling needed)
```

### Task B — `_gemini_receive_loop` (Gemini → Browser)

```
loop:
  async for response in svc.receive():  # streams from Gemini Live
    if response.audio:
        await websocket.send_bytes(response.audio)
    if response.text:
        await websocket.send_json({type: "transcript", ...})
    if response.server_content.turn_complete:
        await websocket.send_json({type: "ai_turn_complete"})
```

**Why `asyncio.gather()` instead of `create_task()`?**  
`gather()` ensures both tasks are cancelled and cleaned up together when either fails. If the browser disconnects (Task A raises `WebSocketDisconnect`), Task B is immediately cancelled and the Gemini session is closed. With bare `create_task()`, Task B would leak until timeout.

---

## 5. Barge-In / Interruption Handling

Gemini Live has **built-in barge-in detection**. When the user speaks while Gemini is generating audio:

1. Gemini Live detects the incoming audio and **automatically truncates** its response.
2. Gemini emits an `interrupted` signal on the receive stream.
3. The server forwards an `ai_interrupted` JSON control frame to the browser.
4. The browser **flushes its playback queue** and stops scheduling new audio buffers.

```
Server side — in _gemini_receive_loop:
  if response.server_content and response.server_content.interrupted:
      # Cancel any pending audio already queued on server side
      await websocket.send_json({"type": "ai_interrupted", "data": {}})

Browser side — on receiving "ai_interrupted":
  nextPlayTime = audioContext.currentTime  # reset playback clock
  # (in-flight AudioBufferSourceNodes finish their current buffer,
  #  but no new buffers are scheduled — effective immediate cutoff)
```

**No server-side VAD is needed.** Gemini Live's built-in VAD handles:
- End-of-utterance detection (when to start responding)
- Barge-in detection (when to stop responding)

This eliminates the `voice_start_speaking` / `voice_stop_speaking` protocol messages entirely.


---

## 6. Audio Format Specification

### 6.1 Microphone Input (Browser → Server → Gemini)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Format | **PCM 16-bit signed, little-endian** | Gemini Live required input format |
| Sample rate | **16 000 Hz** | Gemini Live requirement |
| Channels | **1 (mono)** | Gemini Live requirement |
| Chunk size | **4 096 samples (~256 ms)** | Balances latency vs. frame overhead |
| Container | **Raw PCM bytes** (no WAV header) | Gemini Live accepts raw LINEAR16 |
| WS frame type | **Binary** | `ArrayBuffer` sent via `ws.send(buffer)` |
| Capture timing | **Continuous from session start** | No push-to-talk; mic is always streaming |

### 6.2 AI Audio Output (Gemini → Server → Browser)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Format | **PCM 16-bit signed, little-endian** | Gemini Live default output |
| Sample rate | **24 000 Hz** | Gemini Live default output sample rate |
| Channels | **1 (mono)** | Gemini Live output |
| WS frame type | **Binary** | Raw PCM bytes forwarded directly |
| Browser playback | **AudioContext + AudioBufferSourceNode** | Queued with `nextPlayTime` accumulator |
| Echo handling | **Browser AEC via `getUserMedia` constraints** | `echoCancellation: true` — no mic mute needed |

---

## 7. Revised WebSocket Protocol

All existing text-mode JSON message types are unchanged. Voice mode adds binary audio frames and a reduced set of JSON control messages.

### 7.1 Frame Discrimination

| Frame type | Interpretation |
|-----------|----------------|
| **String frame** | JSON control message |
| **Binary frame** | Raw PCM16 audio data |

### 7.2 Client → Server Messages

#### Existing (unchanged)
```json
{ "type": "start",   "data": { "role": "Software Engineer", "candidate_name": "Alex", "mode": "text" } }
{ "type": "message", "data": { "content": "I would use a hash map..." } }
{ "type": "end",     "data": {} }
```

#### Updated `start` — add `mode: "voice"`
```json
{
  "type": "start",
  "data": {
    "role": "Software Engineer",
    "candidate_name": "Alex",
    "mode": "voice"
  }
}
```
`mode` defaults to `"text"` if omitted — **full backward compatibility**.

#### ~~Removed in v2.0~~ (push-to-talk signals — no longer needed)
```
voice_start_speaking  ← REMOVED (mic is always on)
voice_stop_speaking   ← REMOVED (Gemini Live VAD handles turn detection)
```

#### Binary frames (Client → Server)
Raw PCM16 LE mono 16 kHz. Sent **continuously** from session start to session end.  
No JSON wrapper — the binary frame is the audio payload.

### 7.3 Server → Client Messages

#### Existing (unchanged)
```json
{ "type": "chunk",    "data": { "content": "partial text", "done": false } }
{ "type": "response", "data": { "content": "full text",    "done": true  } }
{ "type": "error",    "data": { "content": "...", "code": "RATE_LIMIT"  } }
```

#### Session ready
```json
{ "type": "session_ready", "data": { "mode": "voice", "output_sample_rate": 24000 } }
```

#### Transcript (voice mode only)
```json
{ "type": "transcript", "data": { "speaker": "ai",   "content": "Tell me about...", "final": true } }
{ "type": "transcript", "data": { "speaker": "user",  "content": "I would use...",   "final": true } }
```

#### AI turn complete
```json
{ "type": "ai_turn_complete", "data": {} }
```
AI has finished its response. Browser may use this to update UI state (e.g., hide speaking indicator).

#### AI interrupted (barge-in occurred)
```json
{ "type": "ai_interrupted", "data": {} }
```
Browser must **flush its audio playback queue** immediately — do not play buffered audio that was not yet scheduled.

#### ~~Removed in v2.0~~ (mic gating signals — no longer needed)
```
ai_speaking_start  ← REMOVED (mic is never muted)
ai_speaking_end    ← REMOVED
```

#### Binary frames (Server → Client)
Raw PCM16 LE mono 24 kHz AI speech bytes. Sent as they arrive from Gemini Live.

### 7.4 Error Codes

All existing codes remain. Voice-specific codes:

| Code | Meaning |
|------|---------|
| `AUDIO_FORMAT_ERROR` | Binary frame received in text mode, or malformed audio |
| `LIVE_SESSION_ERROR` | Gemini Live connection failed to establish |
| `VOICE_NOT_SUPPORTED` | Server config missing API key or Live capability |

---

## 8. Backend Implementation

### 8.1 `backend/gemini_service.py` — `GeminiVoiceService`

```python
class GeminiVoiceService:
    """
    Manages a Gemini Live session with AUDIO input/output modality.
    Instantiated per-connection — never shared across sessions.

    The session is fully duplex: send_audio() and receive() can be called
    concurrently from separate asyncio tasks without coordination.
    """

    async def connect(self) -> None:
        """
        Open client.aio.live.connect() with AUDIO modalities.
        Configures built-in VAD and barge-in (no server-side VAD needed).
        """

    async def send_audio(self, pcm_bytes: bytes) -> None:
        """
        Forward raw PCM16 LE 16kHz mono bytes to the Live session.
        Called continuously from _ws_receive_loop. Thread-safe via asyncio.
        """

    async def receive(self) -> AsyncGenerator[LiveResponse, None]:
        """
        Yield responses from Gemini Live as they arrive.
        Each response may contain: .data (audio bytes), .text, 
        .server_content.turn_complete, .server_content.interrupted.
        This generator runs for the lifetime of the session.
        """

    async def close(self) -> None:
        """Gracefully close the Live session."""
```

**Gemini Live config:**
```python
config = genai_types.LiveConnectConfig(
    # ✅ CONFIRMED: use ["AUDIO", "TEXT"] — TEXT modality makes Gemini return
    # its generated speech as a text transcript alongside audio.
    # The _gemini_receive_loop already handles response.text → transcript frame.
    # Latency delta is imperceptible (<50ms) for interview sessions.
    response_modalities=["AUDIO", "TEXT"],
    system_instruction=build_system_prompt(role, candidate_name),
    speech_config=genai_types.SpeechConfig(
        voice_config=genai_types.VoiceConfig(
            prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                voice_name="Charon"
            )
        )
    ),
    # Gemini Live enables built-in VAD and barge-in by default.
    # No explicit VAD config needed — do NOT set end_of_turn manually.
)
```

### 8.2 `backend/main.py` — Full-Duplex Handler

```python
@app.websocket("/ws/interview")
async def interview_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    
    if not GEMINI_API_KEY:
        await _send_error(websocket, "API key not configured.", "MISSING_API_KEY")
        await websocket.close()
        return

    svc = None
    session_mode = "text"

    try:
        # Phase 1: wait for the "start" control message
        init_msg = await websocket.receive_text()
        msg = json.loads(init_msg)
        assert msg["type"] == "start"
        session_mode = msg["data"].get("mode", "text")

        if session_mode == "voice":
            svc = GeminiVoiceService(
                role=msg["data"]["role"],
                candidate_name=msg["data"]["candidate_name"],
            )
            await svc.connect()
            await _send_json(websocket, {
                "type": "session_ready",
                "data": {"mode": "voice", "output_sample_rate": 24000}
            })
            # Phase 2: run both directions concurrently
            await asyncio.gather(
                _ws_receive_loop(websocket, svc),
                _gemini_receive_loop(websocket, svc),
            )

        else:
            # Existing text-mode path (unchanged)
            svc = GeminiInterviewService(...)
            await _stream_text_session(websocket, svc, msg)

    except WebSocketDisconnect:
        pass  # Normal — client closed connection
    except Exception as e:
        await _send_error(websocket, str(e), "INTERNAL_ERROR")
    finally:
        if svc:
            await svc.close()


async def _ws_receive_loop(websocket: WebSocket, svc: GeminiVoiceService) -> None:
    """Task A: browser → Gemini. Runs until client sends 'end' or disconnects."""
    while True:
        message = await websocket.receive()
        if "bytes" in message:
            await svc.send_audio(message["bytes"])
        elif "text" in message:
            ctrl = json.loads(message["text"])
            if ctrl.get("type") == "end":
                break  # graceful session end; exits gather()
            # All other JSON messages in voice mode are silently ignored.
            # No voice_start_speaking / voice_stop_speaking in v2.0.


async def _gemini_receive_loop(websocket: WebSocket, svc: GeminiVoiceService) -> None:
    """Task B: Gemini → browser. Runs until Gemini closes the session."""
    async for response in svc.receive():
        if response.data:
            # Raw PCM16 audio bytes → browser playback
            await websocket.send_bytes(response.data)
        if response.text:
            await _send_json(websocket, {
                "type": "transcript",
                "data": {"speaker": "ai", "content": response.text, "final": True}
            })
        if response.server_content:
            if response.server_content.interrupted:
                await _send_json(websocket, {"type": "ai_interrupted", "data": {}})
            if response.server_content.turn_complete:
                await _send_json(websocket, {"type": "ai_turn_complete", "data": {}})
```


---

## 9. Frontend Implementation

### 9.1 Full-Duplex Browser Model

The browser maintains **two independent, concurrent operations** after session start:

| Operation | Mechanism | Runs when |
|-----------|-----------|-----------| 
| **Mic capture → send** | AudioWorklet → `ws.send(pcmBuffer)` | Continuously, from `session_ready` until session end |
| **Receive → playback** | `ws.onmessage` binary → `enqueueAudioPlayback()` | Continuously, as frames arrive |

These two operations never gate each other. The browser does **not** check "is AI speaking?" before sending mic data.

### 9.2 Key State Variables

```javascript
let sessionMode      = 'text';    // 'text' | 'voice'
let audioContext     = null;      // AudioContext (created on session_ready)
let micStream        = null;      // MediaStream from getUserMedia
let audioWorkletNode = null;      // AudioWorkletNode for capture
let nextPlayTime     = 0;         // Scheduled playback clock (AudioContext time)
// NOTE: No isSpeaking, no micEnabled flag — mic is always on in voice mode
```

### 9.3 Session Lifecycle

```
user clicks "Start Interview" (voice mode)
  → send { type: "start", data: { mode: "voice", ... } }
  → await session_ready from server
  → initAudioContext(output_sample_rate)
  → startMicCapture()           ← mic opens, audio flows to server immediately
  → show voice UI (hide text input bar)

[interview in progress — fully duplex]
  → mic audio streams continuously to server
  → AI audio plays back as received
  → transcripts update chat log

user clicks "End Interview"
  → stopMicCapture()
  → send { type: "end", data: {} }
  → ws.close()
```

### 9.4 Barge-In Handling (Browser)

```javascript
// On receiving "ai_interrupted" from server:
ws.onmessage = (event) => {
  if (event.data instanceof ArrayBuffer) {
    enqueueAudioPlayback(event.data);
    return;
  }
  const msg = JSON.parse(event.data);
  if (msg.type === 'ai_interrupted') {
    // Reset playback queue — stop scheduling new audio
    nextPlayTime = audioContext.currentTime;
    // In-flight AudioBufferSourceNodes will finish their current small buffer
    // but no new ones will be scheduled. Perceived cutoff is near-instant.
  }
  // ... handle other message types
};
```

### 9.5 `static/mic-worklet.js` (new file)

```javascript
class MicCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buffer = [];
    this._chunkSamples = 4096; // ~256ms at 16kHz
  }

  process(inputs) {
    const channel = inputs[0][0];
    if (!channel) return true;
    for (let i = 0; i < channel.length; i++) {
      this._buffer.push(Math.max(-1, Math.min(1, channel[i])));
    }
    while (this._buffer.length >= this._chunkSamples) {
      const chunk = this._buffer.splice(0, this._chunkSamples);
      const int16 = new Int16Array(this._chunkSamples);
      for (let i = 0; i < this._chunkSamples; i++) {
        int16[i] = chunk[i] * 32767;
      }
      // Transfer buffer ownership — zero-copy postMessage
      this.port.postMessage(int16, [int16.buffer]);
    }
    return true; // keep processor alive
  }
}
registerProcessor('mic-capture-processor', MicCaptureProcessor);
```

### 9.6 UI — Voice Controls (Full Duplex)

**No push-to-talk button.** The voice UI has:

1. **Live/End toggle** — single button that starts the session (activates mic) and ends it
2. **Speaking indicators** — animated indicators showing who is currently speaking (derived from audio level / transcript events), purely decorative
3. **Transcript panel** — chat log updated with `transcript` messages
4. **End Interview button** — sends `{ type: "end" }` and closes WS

```html
<!-- Voice controls (visible only in voice mode) -->
<div id="voice-controls" class="voice-controls" style="display:none">
  <div class="speaking-indicators">
    <div id="user-speaking-indicator" class="indicator" aria-label="Your mic is active">
      🎙️ <span>You</span>
    </div>
    <div id="ai-speaking-indicator"   class="indicator" aria-label="AI is speaking">
      🤖 <span>Interviewer</span>
    </div>
  </div>
  <!-- Text input is hidden in voice mode; transcripts appear in chat log -->
</div>
```

**No mic toggle button.** Mic is controlled entirely by session start/end.

---

## 10. File Change Summary

| File | Change Type | Owner | Notes |
|------|------------|-------|-------|
| `backend/gemini_service.py` | **Add** `GeminiVoiceService` class | `senior_dev_1` | Per-connection instance; `send_audio()` + async `receive()` generator |
| `backend/main.py` | **Add** voice branch with `asyncio.gather(_ws_receive_loop, _gemini_receive_loop)` | `senior_dev_1` | Both loops run concurrently; `gather()` for coordinated teardown |
| `static/app.js` | **Extend** — voice state, AudioContext, continuous mic capture, playback queue, barge-in flush | `ux_engineer` + `senior_dev_2` | **Remove** push-to-talk handlers; mic is always-on |
| `static/mic-worklet.js` | **New file** | `ux_engineer` | AudioWorkletProcessor; Float32→Int16 conversion |
| `static/index.html` | **Modify** — mode toggle + voice indicators (no push-to-talk button) | `ux_engineer` | Additive; no existing elements removed |
| `static/style.css` | **Extend** — `.voice-controls`, `.indicator`, `.speaking-active` animations | `ux_engineer` | |
| `tests/test_voice.py` | **Add** concurrent send+receive tests, barge-in tests, always-on mic tests | `junior_dev_1` | Test `asyncio.gather` teardown on disconnect |

---

## 11. Backward Compatibility

- `mode` field defaults to `"text"` when absent — existing clients unaffected.
- All existing JSON message types unchanged.
- Text-mode sessions use `GeminiInterviewService` exclusively.
- The `voice_start_speaking` / `voice_stop_speaking` messages, if sent by an old client, are **silently ignored** in v2.0 voice mode (not an error).
- `ai_speaking_start` / `ai_speaking_end` are no longer emitted; clients that depend on them will simply not receive them (safe — they were optional UI hints).

---

## 12. Non-Functional Considerations

### Security
- Mic permission requires HTTPS in production (`localhost` exempt).
- Audio data: encrypted by WSS. Never written to disk on the server.
- API key: server-side env var only, never forwarded to browser.
- Binary frame rate limit: enforce ~100 KB/s inbound per connection to prevent flood abuse.

### Performance
- `asyncio.gather()` — both tasks yield to the event loop on every `await`. No blocking.
- AudioWorklet mic capture runs on a dedicated audio thread — zero main-thread jank.
- PCM playback uses `nextPlayTime` accumulator — gapless scheduling, no audible glitches.
- Barge-in latency: user speech → Gemini interrupt → `ai_interrupted` → browser queue flush ≈ 1–2 RTTs (~100–200 ms on typical broadband).

### Scalability
- Each voice session holds one persistent upstream Gemini Live WebSocket.
- Voice sessions are heavier than text sessions (~32 KB/s inbound audio + overhead).
- For >100 concurrent voice sessions, dedicate a separate process/pod for voice WS handling.

### Browser Compatibility
- `AudioWorklet`: Chrome 66+, Firefox 76+, Safari 14.1+.
- `getUserMedia` with `echoCancellation`: all modern browsers.
- Fallback: if `AudioWorklet` unavailable, fall back to `ScriptProcessorNode` (deprecated but supported). Flag in implementation task.
- **AEC (Acoustic Echo Cancellation):** `echoCancellation: true` in `getUserMedia` activates browser-native AEC. This handles echo on most devices without server-side intervention — a key advantage of always-on mic vs. mic muting.

---

## 13. Open Questions for Implementation

1. **`response_modalities` — include TEXT?** ✅ **RESOLVED — use `["AUDIO", "TEXT"]`.**  
   `["AUDIO"]` only gives audio; adding `"TEXT"` causes Gemini to return its generated speech as a text transcript alongside audio, populating the chat log panel. The `_gemini_receive_loop` already has the `if response.text:` branch wired and ready. Latency delta is imperceptible for interview sessions (<50 ms). **sr1 must implement with `response_modalities=["AUDIO", "TEXT"]`** — see updated config in Section 8.1.

2. **Silence detection / idle timeout:** If the user's mic goes silent for >60 s (e.g., they walked away), the session should auto-close. Implement as a watchdog timer reset on each received audio byte.

3. **Audio visualizer:** Animated waveform/bars on the user's speaking indicator can be driven by the AudioWorklet's amplitude data — a `port.postMessage` with an RMS level on each chunk. This is a UX enhancement for a follow-up.

4. **User transcript:** Gemini Live can return user speech transcripts (via `input_transcription`). Enabling this allows the chat log to show both sides of the conversation in real time.

---

*See also: `docs/adr/ADR-001-voice-pipeline.md`, `docs/adr/ADR-002-full-duplex-model.md`*
