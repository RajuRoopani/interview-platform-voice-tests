## 13. Open Questions for Implementation

1. **`response_modalities` — include TEXT?** ✅ **RESOLVED — use `["AUDIO", "TEXT"]`.**
   `["AUDIO"]` only gives audio; adding `"TEXT"` causes Gemini to return its generated speech as a text transcript alongside audio, which populates the chat log panel. The `_gemini_receive_loop` handler already has the `if response.text:` branch ready. Latency delta is imperceptible for interview sessions (<50 ms). **sr1 must implement with `response_modalities=["AUDIO", "TEXT"]`.**

2. **Silence detection / idle timeout:** If the user's mic goes silent for >60 s (e.g., they walked away), the session should auto-close. Implement as a watchdog timer reset on each received audio byte.

3. **Audio visualizer:** Animated waveform/bars on the user's speaking indicator can be driven by the AudioWorklet's amplitude data — a `port.postMessage` with an RMS level on each chunk. This is a UX enhancement for a follow-up.

4. **User transcript:** Gemini Live can return user speech transcripts (via `input_transcription`). Enabling this allows the chat log to show both sides of the conversation in real time.
