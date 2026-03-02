/**
 * mic-worklet.js — Audio Worklet Processor for microphone capture.
 *
 * Runs on the AudioWorklet thread (off the main thread) to capture
 * mic audio and convert it from Float32 to Int16 PCM16 format in
 * 4096-sample chunks, matching the Gemini Live API input requirement:
 *   - Format: PCM 16-bit signed, little-endian
 *   - Sample rate: 16 000 Hz (set on the AudioContext / getUserMedia)
 *   - Channels: 1 (mono)
 *   - Chunk size: 4096 samples (~256 ms at 16 kHz)
 *
 * Usage (from app.js):
 *   await audioContext.audioWorklet.addModule('/static/mic-worklet.js');
 *   const workletNode = new AudioWorkletNode(audioContext, 'mic-capture-processor');
 *   workletNode.port.onmessage = (e) => ws.send(e.data.buffer);
 *   micSourceNode.connect(workletNode);
 *   workletNode.connect(audioContext.destination); // keeps node alive
 */

'use strict';

class MicCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    // Internal sample accumulator — collects samples until a full chunk is ready.
    this._buffer = [];
    // Target chunk size: 4096 samples @ 16 kHz ≈ 256 ms of audio.
    this._chunkSamples = 4096;
  }

  /**
   * Called by the audio engine for each block of audio (128 frames by default).
   * Accumulates samples and emits a PCM16 Int16Array once a full chunk is ready.
   *
   * @param {Float32Array[][]} inputs  - inputs[0][0] is the mono mic channel
   * @returns {boolean} true to keep the processor alive
   */
  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel) return true; // No input yet — keep alive

    // Accumulate clamped samples into the internal buffer.
    for (let i = 0; i < channel.length; i++) {
      // Clamp to [-1, 1] to prevent Int16 overflow on hot microphones.
      this._buffer.push(Math.max(-1, Math.min(1, channel[i])));
    }

    // Emit complete 4096-sample chunks as Int16 PCM16.
    while (this._buffer.length >= this._chunkSamples) {
      const chunk = this._buffer.splice(0, this._chunkSamples);
      const int16 = new Int16Array(this._chunkSamples);

      for (let i = 0; i < this._chunkSamples; i++) {
        // Float32 [-1, 1] → Int16 [-32768, 32767]
        int16[i] = Math.round(chunk[i] * 32767);
      }

      // Transfer buffer ownership (zero-copy) to the main thread.
      // app.js receives this via workletNode.port.onmessage and sends
      // int16.buffer as a binary WebSocket frame.
      this.port.postMessage(int16, [int16.buffer]);
    }

    return true; // Keep processor alive for the session duration.
  }
}

registerProcessor('mic-capture-processor', MicCaptureProcessor);
