/**
 * AI Interview Platform — app.js
 * Handles WebSocket communication, UI state management, and Gemini streaming.
 * Supports both text mode (existing) and voice mode (Gemini Live API).
 *
 * VOICE MODE: Full duplex — mic captures and streams continuously while AI
 * audio plays back simultaneously. Like a phone call, not a walkie-talkie.
 *
 * Vanilla JS, no frameworks.
 */

'use strict';

(function () {
  // ─── Constants ───────────────────────────────────────────────────────────────
  const MAX_CHARS = 2000;
  const CHAR_WARN_THRESHOLD = 1800;

  /**
   * Audio capture settings for the voice mode mic pipeline.
   * Gemini Live API requires: 16-bit PCM, 16 kHz, mono.
   */
  const CAPTURE_SAMPLE_RATE  = 16000;
  const CAPTURE_CHANNELS     = 1;
  const CAPTURE_BUFFER_SIZE  = 4096; // ScriptProcessorNode buffer

  /**
   * Audio playback settings for AI responses.
   * Gemini Live API sends: 16-bit PCM, 24 kHz, mono.
   */
  const PLAYBACK_SAMPLE_RATE = 24000;

  // ─── State ───────────────────────────────────────────────────────────────────
  let ws              = null;
  let currentAIBubble = null;   // Active streaming AI message element (text mode)
  let isAIResponding  = false;
  let interviewMode   = 'text'; // 'text' | 'voice'

  // ─── Voice State ─────────────────────────────────────────────────────────────
  //
  // Full-duplex model:
  //   - audioContext: shared for both capture and playback
  //   - micStream: always open while in voice mode; never paused for AI playback
  //   - isMicMuted: user can silence their mic without stopping the pipeline
  //   - isAISpeaking: AI audio is currently playing back
  //   - Both directions are independent — no turn-taking coordination
  //
  let audioContext       = null;  // Shared AudioContext for capture & playback
  let micStream          = null;  // MediaStream from getUserMedia
  let scriptProcessor    = null;  // ScriptProcessorNode for PCM extraction
  let micSourceNode      = null;  // MediaStreamAudioSourceNode
  let micAnalyserNode    = null;  // AnalyserNode for mic visualizer
  let aiAnalyserNode     = null;  // AnalyserNode for AI output visualizer
  let aiGainNode         = null;  // GainNode for AI output (enables speaker mute)
  let isMicMuted         = false; // True = chunks not sent but pipeline stays active
  let isSpeakerMuted     = false; // True = AI audio discarded / gain=0
  let isVoiceActive      = false; // True = voice pipeline is fully running
  let isAISpeaking       = false; // True = AI audio chunks are arriving/playing
  let micVizFrameId      = null;  // rAF ID for mic waveform loop
  let aiVizFrameId       = null;  // rAF ID for AI waveform loop
  let playbackOffset     = 0.0;   // AudioContext time when next chunk should play
  let currentSourceNodes = [];    // Active AudioBufferSourceNodes (for cleanup)
  let recTimerInterval   = null;  // setInterval ID for recording elapsed timer
  let recStartTime       = 0;     // Date.now() when voice session started

  // ─── DOM References (populated in init) ──────────────────────────────────────
  let setupScreen, chatScreen, userNameInput, roleSelect, startBtn;
  let chatMessages, messageInput, sendBtn, endBtn;
  let typingIndicator, connectionStatus, currentRoleDisplay, charCount;

  // Voice UI elements (added by UX engineer; may be null in text-only builds)
  let interviewModeSelect, micBtn, speakerBtn, endBtnVoice;
  let audioVisualizer, recordingIndicator, aiSpeakingIndicator;
  let voiceInputBar, textInputBar, voiceModeBadge, voiceModeTip;
  let startBtnIcon, startBtnLabel;

  // ─── Initialization ───────────────────────────────────────────────────────────
  document.addEventListener('DOMContentLoaded', () => {
    // --- Core DOM ---
    setupScreen        = document.getElementById('setup-screen');
    chatScreen         = document.getElementById('chat-screen');
    userNameInput      = document.getElementById('user-name');
    roleSelect         = document.getElementById('role-select');
    startBtn           = document.getElementById('start-btn');
    chatMessages       = document.getElementById('chat-messages');
    messageInput       = document.getElementById('message-input');
    sendBtn            = document.getElementById('send-btn');
    endBtn             = document.getElementById('end-btn');
    typingIndicator    = document.getElementById('typing-indicator');
    connectionStatus   = document.getElementById('connection-status');
    currentRoleDisplay = document.getElementById('current-role');
    charCount          = document.getElementById('char-count');

    // --- Voice UI DOM (optional — only present when UX engineer's HTML is deployed) ---
    interviewModeSelect = document.getElementById('interview-mode'); // <div role="radiogroup">
    micBtn              = document.getElementById('mic-btn');
    speakerBtn          = document.getElementById('speaker-btn');
    endBtnVoice         = document.getElementById('end-btn-voice');
    audioVisualizer     = document.getElementById('audio-visualizer');
    recordingIndicator  = document.getElementById('recording-indicator');
    aiSpeakingIndicator = document.getElementById('ai-speaking-indicator');
    voiceInputBar       = document.getElementById('voice-input-bar');
    textInputBar        = document.getElementById('text-input-bar');
    voiceModeBadge      = document.getElementById('voice-mode-toggle');
    voiceModeTip        = document.getElementById('voice-mode-tip');
    startBtnIcon        = document.getElementById('start-btn-icon');
    startBtnLabel       = document.getElementById('start-btn-label');

    setConnectionStatus('disconnected');

    // Start button disabled until roles are fetched
    startBtn.disabled = true;
    fetchRoles().then(attachEventListeners);
  });

  // ─── Fetch Roles ─────────────────────────────────────────────────────────────
  /**
   * Fetches available interview roles from the API and populates the dropdown.
   * Clears ALL existing options to avoid hardcoded stale values from HTML.
   */
  async function fetchRoles() {
    try {
      const response = await fetch('/api/config/roles');
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      const roles = Array.isArray(data) ? data : (data.roles || []);
      _populateRoleSelect(roles);
    } catch (err) {
      console.error('Failed to fetch roles:', err);
      _populateRoleSelect(['Software Engineer', 'Data Scientist', 'Product Manager', 'System Design']);
    }
  }

  /** Clears and repopulates the role <select> with the given list. */
  function _populateRoleSelect(roles) {
    while (roleSelect.options.length > 0) roleSelect.remove(0);

    const placeholder = document.createElement('option');
    placeholder.value    = '';
    placeholder.disabled = true;
    placeholder.selected = true;
    placeholder.textContent = 'Select a role…';
    roleSelect.appendChild(placeholder);

    roles.forEach((role) => {
      const option = document.createElement('option');
      option.value       = typeof role === 'string' ? role : (role.id || role.name);
      option.textContent = typeof role === 'string' ? role : role.name;
      roleSelect.appendChild(option);
    });
  }

  // ─── Event Listeners ─────────────────────────────────────────────────────────
  function attachEventListeners() {
    roleSelect.addEventListener('change', () => {
      startBtn.disabled = !roleSelect.value;
    });

    if (interviewModeSelect) {
      // #interview-mode is a <div role="radiogroup"> with .mode-btn children,
      // NOT a <select>. Listen for clicks on the buttons; read dataset.mode.
      const modeBtns = interviewModeSelect.querySelectorAll('.mode-btn');
      modeBtns.forEach((btn) => {
        btn.addEventListener('click', () => {
          const newMode = btn.dataset.mode || 'text';
          // Update active class and aria-checked on all mode buttons
          modeBtns.forEach((b) => {
            const isActive = b === btn;
            b.classList.toggle('mode-btn--active', isActive);
            b.setAttribute('aria-checked', String(isActive));
          });
          // Sync the parent radiogroup's data-mode attribute (readable by CSS/tests)
          interviewModeSelect.setAttribute('data-mode', newMode);
          interviewMode = newMode;
          _updateModeUI();
        });
      });
      // Read initial mode from the radiogroup's data-mode attribute (set in HTML)
      interviewMode = interviewModeSelect.dataset.mode || 'text';
      _updateModeUI();
    }

    startBtn.addEventListener('click', startInterview);
    sendBtn.addEventListener('click', sendMessage);

    messageInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });

    messageInput.addEventListener('input', () => {
      sendBtn.disabled = !messageInput.value.trim() || isAIResponding;
      const len = messageInput.value.length;
      if (charCount) {
        charCount.textContent = `${len}/${MAX_CHARS}`;
        charCount.classList.toggle('warn', len > CHAR_WARN_THRESHOLD);
      }
      messageInput.style.height = 'auto';
      messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + 'px';
    });

    endBtn.addEventListener('click', endInterview);
    // Voice mode has its own End Call button inside the voice input bar
    if (endBtnVoice) endBtnVoice.addEventListener('click', endInterview);

    // Full-duplex: mic button = mic MUTE toggle, not record/stop.
    // The pipeline runs continuously; this only gates outbound chunks.
    if (micBtn)     micBtn.addEventListener('click', toggleMicMute);
    if (speakerBtn) speakerBtn.addEventListener('click', toggleSpeaker);

    // Header badge: clicking it toggles between text and voice mode mid-interview
    if (voiceModeBadge) {
      voiceModeBadge.addEventListener('click', () => {
        const newMode = interviewMode === 'voice' ? 'text' : 'voice';
        interviewMode = newMode;
        // Keep the setup-screen radiogroup in sync so a mode switch mid-interview
        // is reflected if the user returns to setup (e.g. after reconnect).
        if (interviewModeSelect) {
          interviewModeSelect.setAttribute('data-mode', newMode);
          const modeBtns = interviewModeSelect.querySelectorAll('.mode-btn');
          modeBtns.forEach((b) => {
            const isActive = b.dataset.mode === newMode;
            b.classList.toggle('mode-btn--active', isActive);
            b.setAttribute('aria-checked', String(isActive));
          });
        }
        _updateModeUI();
      });
    }

    window.addEventListener('beforeunload', _cleanupVoice);
  }

  /**
   * Updates body data-attribute and button disabled states to reflect mode.
   * Also toggles the voice/text input bars, updates the header badge,
   * the voice-mode-tip hint, and the start button icon/label.
   */
  function _updateModeUI() {
    document.body.setAttribute('data-interview-mode', interviewMode);
    const isVoice = interviewMode === 'voice';

    // --- Input bars: show the right bar for the current mode ---
    // These are only present on the chat screen, so guard for null.
    if (voiceInputBar) voiceInputBar.hidden = !isVoice;
    if (textInputBar)  textInputBar.hidden  = isVoice;

    // --- Header mode badge ---
    if (voiceModeBadge) {
      voiceModeBadge.setAttribute('data-mode', interviewMode);
      voiceModeBadge.setAttribute(
        'aria-label',
        isVoice ? 'Currently in voice mode' : 'Currently in text mode'
      );
      const iconEl  = voiceModeBadge.querySelector('.voice-mode-badge-icon');
      const labelEl = voiceModeBadge.querySelector('.voice-mode-badge-label');
      if (iconEl)  iconEl.textContent  = isVoice ? '🎙' : '💬';
      if (labelEl) labelEl.textContent = isVoice ? 'Voice' : 'Text';
    }

    // --- Voice-mode tip below the mode selector ---
    if (voiceModeTip) voiceModeTip.hidden = !isVoice;

    // --- Start button icon & label ---
    if (startBtnIcon)  startBtnIcon.textContent  = isVoice ? '🎙' : '💬';
    if (startBtnLabel) startBtnLabel.textContent  = isVoice ? 'Start Voice Interview' : 'Start Interview';

    // --- Mic / Speaker button defaults (disabled until voice session is live) ---
    if (micBtn)     micBtn.disabled     = true;
    if (speakerBtn) speakerBtn.disabled = !isVoice;
  }


  // ─── Start Interview (branches on mode) ──────────────────────────────────────
  function startInterview() {
    const selectedRole = roleSelect.value;
    if (!selectedRole) {
      alert('Please select an interview role to continue.');
      return;
    }
    const userName = (userNameInput && userNameInput.value.trim()) || 'Candidate';

    startBtn.disabled = true;
    setupScreen.style.display = 'none';
    chatScreen.style.display  = 'flex';
    chatScreen.classList.add('fade-in');
    chatMessages.innerHTML = '';

    if (currentRoleDisplay) currentRoleDisplay.textContent = selectedRole;

    if (interviewMode === 'voice') {
      _startVoiceInterview(selectedRole, userName);
    } else {
      _startTextInterview(selectedRole, userName);
    }
  }

  // ─── Text Mode: WebSocket Setup ───────────────────────────────────────────────
  function _startTextInterview(selectedRole, userName) {
    const wsUrl = _buildWsUrl('/ws/interview');
    ws = new WebSocket(wsUrl);
    setConnectionStatus('connecting');

    ws.addEventListener('open', () => {
      setConnectionStatus('connected');
      wsSend({ type: 'start', data: { role: selectedRole, candidate_name: userName } });
      showTypingIndicator();
    });

    ws.addEventListener('message', handleWSMessage);

    ws.addEventListener('close', (event) => {
      setConnectionStatus('disconnected');
      isAIResponding = false;
      disableInput(true);
      if (chatScreen.style.display !== 'none' && !event.wasClean) {
        showReconnectPrompt();
      }
    });

    ws.addEventListener('error', () => {
      setConnectionStatus('error');
      console.error('WebSocket error encountered.');
    });
  }

  // ─── Voice Mode: WebSocket Setup ─────────────────────────────────────────────
  /**
   * Opens the voice WebSocket and immediately starts the full-duplex pipeline:
   *   1. WS connected → send start frame
   *   2. Init AudioContext (must be in user-gesture context)
   *   3. Request mic — on grant, start streaming immediately (no button press needed)
   *   4. onmessage plays received audio chunks without waiting for mic to stop
   *
   * There is NO turn-taking. Both directions run simultaneously at all times.
   */
  function _startVoiceInterview(selectedRole, userName) {
    const wsUrl = _buildWsUrl('/ws/voice-interview');
    ws = new WebSocket(wsUrl);
    ws.binaryType = 'arraybuffer';
    setConnectionStatus('connecting');

    ws.addEventListener('open', () => {
      setConnectionStatus('connected');
      wsSend({ type: 'start', data: { role: selectedRole, candidate_name: userName, mode: 'voice' } });

      // AudioContext must be created inside a user-gesture handler (the start button click).
      _initAudioContext();

      // Request mic and start streaming immediately — full duplex from the start.
      _requestMicAndStartStreaming();
    });

    ws.addEventListener('message', handleVoiceWSMessage);

    ws.addEventListener('close', (event) => {
      setConnectionStatus('disconnected');
      isAIResponding = false;
      _stopVoicePipeline();
      if (chatScreen.style.display !== 'none' && !event.wasClean) {
        showReconnectPrompt();
      }
    });

    ws.addEventListener('error', () => {
      setConnectionStatus('error');
      console.error('Voice WebSocket error encountered.');
    });
  }

  /** Returns a fully-qualified ws:// or wss:// URL for the given path. */
  function _buildWsUrl(path) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${protocol}//${window.location.host}${path}`;
  }


  // ─── Text WebSocket Message Handler ──────────────────────────────────────────
  /**
   * Routes incoming text-mode WebSocket messages.
   * Server sends: chunk, response, error.
   */
  function handleWSMessage(event) {
    let msg;
    try {
      msg = JSON.parse(event.data);
    } catch (err) {
      console.error('Failed to parse WS message:', event.data, err);
      return;
    }

    const { type, data } = msg;

    switch (type) {
      case 'chunk':
        hideTypingIndicator();
        if (!currentAIBubble) currentAIBubble = createAIMessageBubble();
        appendChunkToAIBubble(currentAIBubble, data.content || '');
        scrollToBottom();
        break;

      case 'response':
        hideTypingIndicator();
        if (!currentAIBubble) currentAIBubble = createAIMessageBubble();
        if (data.content && currentAIBubble.textContent === '') {
          setAIBubbleText(currentAIBubble, data.content);
        }
        finalizeAIBubble(currentAIBubble);
        currentAIBubble = null;
        isAIResponding  = false;
        disableInput(false);
        scrollToBottom();
        break;

      case 'error':
        hideTypingIndicator();
        addSpecialMessage(data.content || 'An error occurred.', 'error');
        isAIResponding = false;
        disableInput(false);
        scrollToBottom();
        break;

      default:
        console.warn('Unknown message type from server:', type, msg);
    }
  }

  // ─── Voice WebSocket Message Handler ─────────────────────────────────────────
  /**
   * Routes incoming voice-mode WebSocket frames.
   *
   * Binary frames  → raw PCM audio (Int16, 24 kHz, mono) → play immediately.
   * JSON frames    → audio_start | audio_end | transcript | error
   *
   * FULL DUPLEX: audio playback is completely independent of mic capture.
   * We never pause/mute the mic when AI audio arrives, and we never pause
   * playback while the user is speaking. Both run simultaneously.
   */
  function handleVoiceWSMessage(event) {
    // Binary frame → enqueue for immediate playback (does NOT affect mic capture)
    if (event.data instanceof ArrayBuffer) {
      _enqueueAudioChunk(event.data);
      return;
    }

    let msg;
    try {
      msg = JSON.parse(event.data);
    } catch (err) {
      console.error('Failed to parse voice WS message:', event.data, err);
      return;
    }

    const { type, data } = msg;

    switch (type) {
      // ── v2.0 protocol messages ──────────────────────────────────────────

      case 'session_ready':
        // Server has established the Gemini Live session and is ready for audio.
        // data: { mode: "voice", output_sample_rate: 24000 }
        // Mic streaming is already running (started on WS open); this confirms
        // the server-side session is live and we can show the ready state.
        setConnectionStatus('connected', 'Session ready');
        _updateDuplexState();
        break;

      case 'transcript':
        // Server sends text transcript of AI's spoken response (speaker: "ai" or "user").
        // We only display AI transcripts in the chat log; user transcripts are optional.
        if (data && data.content) {
          const isAI = !data.speaker || data.speaker === 'ai';
          if (isAI) {
            if (!currentAIBubble) currentAIBubble = createAIMessageBubble();
            if (currentAIBubble.textContent === '') {
              setAIBubbleText(currentAIBubble, data.content);
            } else {
              appendChunkToAIBubble(currentAIBubble, data.content);
            }
            if (data.final !== false) {
              // Finalize on final transcript (or when final flag is absent)
              finalizeAIBubble(currentAIBubble);
              currentAIBubble = null;
            }
          }
          scrollToBottom();
        }
        break;

      case 'ai_turn_complete':
        // AI has finished its response turn. Schedule speaking indicator off
        // after remaining audio buffers drain.
        isAISpeaking = false;
        _scheduleAISpeakingOff();
        break;

      case 'ai_interrupted':
        // Barge-in: user spoke while AI was responding. Gemini Live detected the
        // interruption and stopped generating. Flush the pending playback queue
        // by resetting the playback clock to "now" — no more buffers will be
        // scheduled, so in-flight ones finish quickly and new ones are discarded.
        if (audioContext) playbackOffset = audioContext.currentTime;
        isAISpeaking = false;
        setAISpeakingIndicator(false);
        _stopAIVisualizer();
        _updateDuplexState();
        break;

      // ── Legacy / fallback v1.0 cases — kept for backward compat ─────────

      case 'audio_start':
        // v1.0 signal (no longer emitted by v2.0 server, but harmless to handle)
        isAISpeaking = true;
        setAISpeakingIndicator(true);
        _updateDuplexState();
        if (audioContext) playbackOffset = audioContext.currentTime;
        _startAIVisualizer();
        break;

      case 'audio_end':
        // v1.0 signal (no longer emitted by v2.0 server, but harmless to handle)
        isAISpeaking = false;
        _scheduleAISpeakingOff();
        break;

      case 'error':
        addSpecialMessage((data && data.content) || 'A voice error occurred.', 'error');
        setAISpeakingIndicator(false);
        _stopAIVisualizer();
        _updateDuplexState();
        scrollToBottom();
        break;

      default:
        console.warn('Unknown voice message type:', type, msg);
    }
  }


  // ─── Audio: Context Initialization ───────────────────────────────────────────
  /**
   * Creates the shared AudioContext (or resumes a suspended one).
   * MUST be called inside a user-gesture handler to satisfy browser autoplay policy.
   * We use PLAYBACK_SAMPLE_RATE (24 kHz) as the context rate; capture resamples
   * separately down to 16 kHz via _floatTo16BitPCM before sending.
   */
  function _initAudioContext() {
    if (audioContext && audioContext.state !== 'closed') {
      if (audioContext.state === 'suspended') audioContext.resume();
      return;
    }
    audioContext = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate: PLAYBACK_SAMPLE_RATE,
    });

    // Build the AI output signal chain:
    //   AI source nodes → aiGainNode → aiAnalyserNode → destination
    // This lets us mute speaker (gain=0) and visualize AI output independently.
    aiGainNode    = audioContext.createGain();
    aiAnalyserNode = audioContext.createAnalyser();
    aiAnalyserNode.fftSize = 256;

    aiGainNode.connect(aiAnalyserNode);
    aiAnalyserNode.connect(audioContext.destination);

    // Apply current speaker-mute state to the new context
    aiGainNode.gain.value = isSpeakerMuted ? 0 : 1;
  }

  // ─── Audio: Microphone Capture ────────────────────────────────────────────────
  /**
   * Requests microphone access. On grant, wires the capture pipeline and
   * immediately begins streaming — no button press required.
   *
   * echoCancellation + noiseSuppression + autoGainControl are the browser-level
   * AEC stack. Since mic is open while AI plays back through speakers, AEC is
   * essential to prevent echo loops.
   */
  async function _requestMicAndStartStreaming() {
    try {
      micStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount:     CAPTURE_CHANNELS,
          sampleRate:       CAPTURE_SAMPLE_RATE,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl:  true,
        },
        video: false,
      });

      _wireMicPipeline(micStream);
      isVoiceActive = true;

      // Enable mic-mute button now that we have permission
      if (micBtn) {
        micBtn.disabled = false;
        micBtn.classList.add('active');
        micBtn.setAttribute('aria-pressed', 'false'); // not muted = mic is live
        micBtn.title = 'Mute microphone';
      }

      // Show live recording indicator
      _setLiveIndicator(true);
      _startMicVisualizer();

    } catch (err) {
      console.error('Microphone access denied or unavailable:', err);
      addSpecialMessage(
        '⚠️ Microphone access was denied. Voice mode requires microphone permission. ' +
        'Please allow access in your browser settings and restart the interview.',
        'error'
      );
      if (micBtn) {
        micBtn.disabled = true;
        micBtn.title    = 'Microphone permission denied';
      }
    }
  }

  /**
   * Wires the mic MediaStream into the AudioContext capture pipeline:
   *   MediaStreamSource → micAnalyserNode → ScriptProcessorNode → PCM sender
   *
   * The ScriptProcessorNode fires onaudioprocess at ~5.8 Hz (4096 samples @
   * 24 kHz context rate). Each callback downsamples Float32 to Int16 @ 16 kHz
   * and sends as a binary WebSocket frame — unless the mic is muted.
   *
   * NOTE: ScriptProcessorNode is deprecated but universally supported without
   * requiring a separate AudioWorklet file. Acceptable for this use case.
   */
  function _wireMicPipeline(stream) {
    micSourceNode  = audioContext.createMediaStreamSource(stream);
    micAnalyserNode = audioContext.createAnalyser();
    micAnalyserNode.fftSize = 256;

    scriptProcessor = audioContext.createScriptProcessor(
      CAPTURE_BUFFER_SIZE, CAPTURE_CHANNELS, CAPTURE_CHANNELS
    );

    scriptProcessor.onaudioprocess = (evt) => {
      // Always fires — isMicMuted gates whether we send, but doesn't stop the pipeline.
      // This keeps the AudioContext graph active so AEC continues to work even when
      // the user mutes themselves.
      if (isMicMuted) return;
      if (!ws || ws.readyState !== WebSocket.OPEN) return;

      const inputBuffer = evt.inputBuffer;
      const channelData = inputBuffer.getChannelData(0); // mono: channel 0
      const pcm16 = _floatTo16BitPCM(channelData, inputBuffer.sampleRate, CAPTURE_SAMPLE_RATE);
      ws.send(pcm16.buffer);
    };

    // Connect: source → micAnalyser → processor → destination
    // Connecting to destination is required to keep ScriptProcessorNode active.
    micSourceNode.connect(micAnalyserNode);
    micAnalyserNode.connect(scriptProcessor);
    scriptProcessor.connect(audioContext.destination);
  }


  /**
   * Converts a Float32Array of audio samples to a downsampled Int16Array (PCM 16-bit).
   * Uses linear interpolation for the downsample step (24→16 kHz).
   *
   * @param {Float32Array} float32Array   Input samples at sourceSampleRate
   * @param {number}       sourceSampleRate  Input rate (AudioContext rate, e.g. 24000)
   * @param {number}       targetSampleRate  Target rate required by server (e.g. 16000)
   * @returns {Int16Array}
   */
  function _floatTo16BitPCM(float32Array, sourceSampleRate, targetSampleRate) {
    const ratio        = sourceSampleRate / targetSampleRate;
    const outputLength = Math.round(float32Array.length / ratio);
    const output       = new Int16Array(outputLength);

    for (let i = 0; i < outputLength; i++) {
      const srcIndex = i * ratio;
      const lower    = Math.floor(srcIndex);
      const upper    = Math.min(lower + 1, float32Array.length - 1);
      const fraction = srcIndex - lower;

      const sample  = float32Array[lower] + fraction * (float32Array[upper] - float32Array[lower]);
      const clamped = Math.max(-1.0, Math.min(1.0, sample));
      output[i]     = clamped < 0
        ? Math.round(clamped * 32768)
        : Math.round(clamped * 32767);
    }
    return output;
  }

  // ─── Voice Controls: Mic Mute Toggle ─────────────────────────────────────────
  /**
   * Toggles mic mute state. Does NOT stop the capture pipeline — the
   * ScriptProcessorNode keeps running so AEC remains active and the visualizer
   * continues to animate. Only the outbound WS send is gated.
   *
   * This is the full-duplex equivalent of a phone's mute button.
   */
  function toggleMicMute() {
    if (!isVoiceActive) return;

    isMicMuted = !isMicMuted;

    if (micBtn) {
      micBtn.classList.toggle('muted', isMicMuted);
      // aria-pressed = true means "muted" (the pressed state = mic silenced)
      micBtn.setAttribute('aria-pressed', String(isMicMuted));
      micBtn.title = isMicMuted ? 'Unmute microphone' : 'Mute microphone';
    }

    // Update the live indicator: dim it when muted
    if (recordingIndicator) {
      recordingIndicator.classList.toggle('mic-muted', isMicMuted);
    }

    _updateDuplexState();
  }

  // ─── Voice Controls: Speaker Mute Toggle ─────────────────────────────────────
  /**
   * Toggles AI audio output mute. Uses GainNode to silence output in real-time
   * without stopping the playback pipeline or losing sync.
   */
  function toggleSpeaker() {
    isSpeakerMuted = !isSpeakerMuted;

    if (speakerBtn) {
      speakerBtn.classList.toggle('muted', isSpeakerMuted);
      speakerBtn.setAttribute('aria-pressed', String(isSpeakerMuted));
      speakerBtn.title = isSpeakerMuted ? 'Unmute AI audio' : 'Mute AI audio';
    }

    // Apply gain change — smooth ramp to avoid clicks (5ms)
    if (aiGainNode && audioContext) {
      aiGainNode.gain.linearRampToValueAtTime(
        isSpeakerMuted ? 0 : 1,
        audioContext.currentTime + 0.005
      );
    }

    if (isSpeakerMuted) {
      setAISpeakingIndicator(false);
      _stopAIVisualizer();
    }
  }

  // ─── Voice Pipeline Stop ──────────────────────────────────────────────────────
  /**
   * Gracefully stops the voice pipeline (called on interview end or WS close).
   * Does NOT close the WS — caller handles that.
   */
  function _stopVoicePipeline() {
    _cleanupVoice();
  }


  // ─── Audio: Playback ──────────────────────────────────────────────────────────
  /**
   * Receives a raw PCM ArrayBuffer from the server (Int16, 24 kHz, mono),
   * converts to Float32, and schedules it for seamless sequential playback
   * through aiGainNode (which handles speaker mute without stopping the chain).
   *
   * Full-duplex: this runs completely independently of mic capture. Receiving
   * AI audio never pauses, mutes, or otherwise affects the mic pipeline.
   *
   * @param {ArrayBuffer} buffer  Raw Int16 PCM data from server
   */
  function _enqueueAudioChunk(buffer) {
    if (!audioContext || !aiGainNode) return;

    const int16Array  = new Int16Array(buffer);
    const float32Data = _int16ToFloat32(int16Array);

    const audioBuffer = audioContext.createBuffer(
      1,                    // mono
      float32Data.length,
      PLAYBACK_SAMPLE_RATE
    );
    audioBuffer.getChannelData(0).set(float32Data);

    const sourceNode = audioContext.createBufferSource();
    sourceNode.buffer = audioBuffer;
    // Route through gain (speaker mute) and analyser (AI visualizer)
    sourceNode.connect(aiGainNode);

    // Schedule seamlessly after the last queued chunk
    const startAt = Math.max(audioContext.currentTime, playbackOffset);
    sourceNode.start(startAt);
    playbackOffset = startAt + audioBuffer.duration;

    currentSourceNodes.push(sourceNode);
    sourceNode.onended = () => {
      currentSourceNodes = currentSourceNodes.filter((n) => n !== sourceNode);
      // When the last node finishes and server has sent audio_end, clear indicator
      if (currentSourceNodes.length === 0 && !isAISpeaking) {
        setAISpeakingIndicator(false);
        _stopAIVisualizer();
        _updateDuplexState();
      }
    };
  }

  /**
   * Converts Int16Array PCM samples to Float32Array in [-1.0, 1.0] range.
   * @param {Int16Array} int16Array
   * @returns {Float32Array}
   */
  function _int16ToFloat32(int16Array) {
    const float32 = new Float32Array(int16Array.length);
    for (let i = 0; i < int16Array.length; i++) {
      float32[i] = int16Array[i] / (int16Array[i] < 0 ? 32768 : 32767);
    }
    return float32;
  }

  /**
   * Schedules the AI speaking indicator off after the playback queue drains.
   * Called on 'audio_end' — chunks may still be playing at that point.
   */
  function _scheduleAISpeakingOff() {
    if (!audioContext) {
      setAISpeakingIndicator(false);
      _stopAIVisualizer();
      _updateDuplexState();
      return;
    }
    const remaining = playbackOffset - audioContext.currentTime;
    if (remaining > 0) {
      setTimeout(() => {
        if (!isAISpeaking) {
          setAISpeakingIndicator(false);
          _stopAIVisualizer();
          _updateDuplexState();
        }
      }, Math.ceil(remaining * 1000) + 100); // +100 ms buffer
    } else {
      setAISpeakingIndicator(false);
      _stopAIVisualizer();
      _updateDuplexState();
    }
  }

  // ─── Duplex State Indicator ───────────────────────────────────────────────────
  /**
   * Sets a data attribute on the chat screen when both sides are simultaneously
   * active (user not muted AND AI speaking). CSS uses this for visual feedback.
   */
  function _updateDuplexState() {
    if (!chatScreen) return;
    const bothActive = isVoiceActive && !isMicMuted && isAISpeaking;
    chatScreen.setAttribute('data-duplex-active', String(bothActive));
  }


  // ─── Audio: Mic Visualizer ────────────────────────────────────────────────────
  /**
   * Starts the mic waveform animation loop on #audio-visualizer canvas.
   * Driven by micAnalyserNode — reflects user's mic input in real time.
   * Runs continuously while voice mode is active (even when mic-muted,
   * so the user can see whether they are actually producing sound).
   */
  function _startMicVisualizer() {
    if (!audioVisualizer || !micAnalyserNode) return;
    if (micVizFrameId) return; // Already running

    const canvas  = audioVisualizer;
    const ctx     = canvas.getContext('2d');
    const bufLen  = micAnalyserNode.frequencyBinCount; // fftSize/2 = 128
    const dataArr = new Uint8Array(bufLen);

    function draw() {
      micVizFrameId = requestAnimationFrame(draw);
      micAnalyserNode.getByteTimeDomainData(dataArr);

      const W = canvas.width;
      const H = canvas.height;

      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = 'rgba(0,0,0,0.05)';
      ctx.fillRect(0, 0, W, H);

      // Dim the waveform when mic is muted
      ctx.lineWidth   = 2;
      ctx.strokeStyle = isMicMuted ? 'rgba(100,100,100,0.4)' : '#4f46e5';
      ctx.beginPath();

      const sliceWidth = W / bufLen;
      let x = 0;

      for (let i = 0; i < bufLen; i++) {
        const v = dataArr[i] / 128.0;
        const y = (v * H) / 2;
        if (i === 0) ctx.moveTo(x, y);
        else         ctx.lineTo(x, y);
        x += sliceWidth;
      }
      ctx.lineTo(W, H / 2);
      ctx.stroke();
    }

    draw();
  }

  /** Stops the mic waveform animation and clears the canvas. */
  function _stopMicVisualizer() {
    if (micVizFrameId) {
      cancelAnimationFrame(micVizFrameId);
      micVizFrameId = null;
    }
    if (audioVisualizer) {
      const ctx = audioVisualizer.getContext('2d');
      ctx.clearRect(0, 0, audioVisualizer.width, audioVisualizer.height);
    }
  }

  // ─── Audio: AI Output Visualizer ─────────────────────────────────────────────
  /**
   * Starts a second waveform animation driven by aiAnalyserNode.
   * Rendered on the same #audio-visualizer canvas as an overlay in a
   * contrasting colour so both user and AI levels are visible simultaneously.
   *
   * If the canvas is already showing the mic waveform, we overlay in green
   * at reduced opacity so the signals don't obscure each other.
   */
  function _startAIVisualizer() {
    if (!audioVisualizer || !aiAnalyserNode) return;
    if (aiVizFrameId) return; // Already running

    const canvas  = audioVisualizer;
    const ctx     = canvas.getContext('2d');
    const bufLen  = aiAnalyserNode.frequencyBinCount;
    const dataArr = new Uint8Array(bufLen);

    function draw() {
      aiVizFrameId = requestAnimationFrame(draw);
      aiAnalyserNode.getByteTimeDomainData(dataArr);

      const W = canvas.width;
      const H = canvas.height;

      // AI waveform overlaid in green — mic visualizer draws first in blue
      ctx.lineWidth   = 2;
      ctx.strokeStyle = 'rgba(34, 197, 94, 0.75)'; // green, semi-transparent
      ctx.beginPath();

      const sliceWidth = W / bufLen;
      let x = 0;

      for (let i = 0; i < bufLen; i++) {
        const v = dataArr[i] / 128.0;
        const y = (v * H) / 2;
        if (i === 0) ctx.moveTo(x, y);
        else         ctx.lineTo(x, y);
        x += sliceWidth;
      }
      ctx.lineTo(W, H / 2);
      ctx.stroke();
    }

    draw();
  }

  /** Stops the AI output waveform animation loop. */
  function _stopAIVisualizer() {
    if (aiVizFrameId) {
      cancelAnimationFrame(aiVizFrameId);
      aiVizFrameId = null;
    }
  }


  // ─── Voice: Live Indicator ────────────────────────────────────────────────────
  /**
   * Shows/hides the #recording-indicator with a "● LIVE" label.
   * In full-duplex mode this reflects that the mic pipeline is active,
   * not that a specific recording is in progress.
   *
   * Also starts/stops the elapsed timer displayed inside the indicator.
   *
   * @param {boolean} active
   */
  function _setLiveIndicator(active) {
    if (!recordingIndicator) return;

    if (active) {
      recordingIndicator.style.display = '';
      recordingIndicator.setAttribute('aria-hidden', 'false');
      // Start elapsed timer
      recStartTime = Date.now();
      _updateRecTimer();
      recTimerInterval = setInterval(_updateRecTimer, 1000);
    } else {
      recordingIndicator.style.display = 'none';
      recordingIndicator.setAttribute('aria-hidden', 'true');
      if (recTimerInterval) {
        clearInterval(recTimerInterval);
        recTimerInterval = null;
      }
    }
  }

  /** Updates the elapsed time text inside #recording-indicator. */
  function _updateRecTimer() {
    if (!recordingIndicator) return;
    const elapsed = Math.floor((Date.now() - recStartTime) / 1000);
    const mm = String(Math.floor(elapsed / 60)).padStart(2, '0');
    const ss = String(elapsed % 60).padStart(2, '0');
    // Look for a .rec-timer child; fall back to the indicator element itself
    const timerEl = recordingIndicator.querySelector('.rec-timer') || recordingIndicator;
    timerEl.textContent = `● LIVE  ${mm}:${ss}`;
  }

  // ─── Voice Cleanup ────────────────────────────────────────────────────────────
  /**
   * Fully tears down all voice resources:
   *   - Stops all active playback source nodes
   *   - Disconnects ScriptProcessorNode and analyser nodes
   *   - Stops mic stream tracks
   *   - Closes AudioContext
   *   - Cancels animation frames and timers
   *   - Resets all voice state flags
   *
   * Safe to call multiple times.
   */
  function _cleanupVoice() {
    // Stop visualizers first to avoid rAF callbacks on null nodes
    _stopMicVisualizer();
    _stopAIVisualizer();

    // Stop all active playback nodes
    currentSourceNodes.forEach((node) => {
      try { node.stop(); } catch (_) { /* already stopped */ }
    });
    currentSourceNodes = [];
    playbackOffset = 0;

    // Stop elapsed timer
    if (recTimerInterval) {
      clearInterval(recTimerInterval);
      recTimerInterval = null;
    }

    // Disconnect capture graph
    if (scriptProcessor) {
      try { scriptProcessor.disconnect(); } catch (_) {}
      scriptProcessor = null;
    }
    if (micAnalyserNode) {
      try { micAnalyserNode.disconnect(); } catch (_) {}
      micAnalyserNode = null;
    }
    if (micSourceNode) {
      try { micSourceNode.disconnect(); } catch (_) {}
      micSourceNode = null;
    }

    // Disconnect AI output graph
    if (aiGainNode) {
      try { aiGainNode.disconnect(); } catch (_) {}
      aiGainNode = null;
    }
    if (aiAnalyserNode) {
      try { aiAnalyserNode.disconnect(); } catch (_) {}
      aiAnalyserNode = null;
    }

    // Release mic hardware
    if (micStream) {
      micStream.getTracks().forEach((track) => track.stop());
      micStream = null;
    }

    // Close AudioContext
    if (audioContext && audioContext.state !== 'closed') {
      audioContext.close().catch(() => {});
      audioContext = null;
    }

    // Reset state flags
    isVoiceActive  = false;
    isMicMuted     = false;
    isAISpeaking   = false;

    // Reset UI indicators
    _setLiveIndicator(false);
    setAISpeakingIndicator(false);
    if (chatScreen) chatScreen.setAttribute('data-duplex-active', 'false');

    // Reset mic button to default state
    if (micBtn) {
      micBtn.disabled = true;
      micBtn.classList.remove('active', 'muted');
      micBtn.setAttribute('aria-pressed', 'false');
      micBtn.title = 'Microphone';
    }
  }


  // ─── Send Message (text mode only) ───────────────────────────────────────────
  function sendMessage() {
    if (!messageInput) return;
    const text = messageInput.value.trim();
    if (!text) return;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      addSpecialMessage('Connection lost. Please refresh the page.', 'error');
      return;
    }

    addMessage(text, 'user');

    messageInput.value        = '';
    messageInput.style.height = 'auto';
    sendBtn.disabled          = true;

    if (charCount) {
      charCount.textContent = `0/${MAX_CHARS}`;
      charCount.classList.remove('warn');
    }

    isAIResponding = true;
    disableInput(true);
    showTypingIndicator();

    wsSend({ type: 'message', data: { content: text } });
  }

  // ─── End Interview ────────────────────────────────────────────────────────────
  /**
   * Sends end signal to server and cleans up voice resources if in voice mode.
   * In voice mode we do NOT send { type: 'end' } mid-session for the mic —
   * only when the whole interview ends.
   */
  function endInterview() {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      resetToSetup();
      return;
    }

    if (!confirm('Are you sure you want to end the interview?')) return;

    disableInput(true);
    endBtn.disabled = true;

    // Send interview-end control frame (works for both text and voice modes)
    if (interviewMode === 'text') showTypingIndicator();
    wsSend({ type: 'end', data: {} });

    // Voice: stop pipeline after signalling server
    if (interviewMode === 'voice') {
      _cleanupVoice();
    }
  }

  // ─── WebSocket Send Helper ────────────────────────────────────────────────────
  function wsSend(payload) {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      console.warn('wsSend called but WS is not open:', payload);
      return;
    }
    ws.send(JSON.stringify(payload));
  }

  // ─── UI: Add Message Bubble ───────────────────────────────────────────────────
  function addMessage(content, sender) {
    const wrapper = document.createElement('div');
    wrapper.classList.add('message', `message-${sender}`);

    const bubble = document.createElement('div');
    bubble.classList.add('bubble');
    bubble.textContent = content;

    const meta = document.createElement('div');
    meta.classList.add('message-meta');
    meta.textContent = formatTime(new Date());

    wrapper.appendChild(bubble);
    wrapper.appendChild(meta);
    chatMessages.appendChild(wrapper);

    scrollToBottom();
    return wrapper;
  }

  // ─── UI: Create Empty AI Bubble (for streaming) ───────────────────────────────
  function createAIMessageBubble() {
    const wrapper = document.createElement('div');
    wrapper.classList.add('message', 'message-ai', 'streaming');

    const bubble = document.createElement('div');
    bubble.classList.add('bubble');
    bubble.textContent = '';

    const meta = document.createElement('div');
    meta.classList.add('message-meta');
    meta.textContent = formatTime(new Date());

    wrapper.appendChild(bubble);
    wrapper.appendChild(meta);
    chatMessages.appendChild(wrapper);

    scrollToBottom();
    return bubble;
  }

  function appendChunkToAIBubble(bubbleEl, chunk) { bubbleEl.textContent += chunk; }
  function setAIBubbleText(bubbleEl, text)         { bubbleEl.textContent = text; }

  function finalizeAIBubble(bubbleEl) {
    if (bubbleEl && bubbleEl.parentElement) {
      bubbleEl.parentElement.classList.remove('streaming');
    }
  }

  // ─── UI: Special Messages ─────────────────────────────────────────────────────
  function addSpecialMessage(content, type) {
    const el = document.createElement('div');
    el.classList.add('special-message', `special-message-${type}`);
    el.textContent = content;
    chatMessages.appendChild(el);
    scrollToBottom();
  }


  // ─── UI: Typing Indicator ─────────────────────────────────────────────────────
  function showTypingIndicator() {
    if (typingIndicator) {
      typingIndicator.style.display = 'flex';
      scrollToBottom();
    }
  }

  function hideTypingIndicator() {
    if (typingIndicator) typingIndicator.style.display = 'none';
  }

  // ─── UI: Connection Status ────────────────────────────────────────────────────
  function setConnectionStatus(status, label) {
    if (!connectionStatus) return;

    connectionStatus.classList.remove('connected', 'disconnected', 'connecting', 'reconnecting', 'error');
    connectionStatus.classList.add(status);

    const statusLabels = {
      connected:    'Connected',
      disconnected: 'Disconnected',
      connecting:   'Connecting…',
      reconnecting: 'Reconnecting…',
      error:        'Connection Error',
    };

    const labelEl = connectionStatus.querySelector('.status-label');
    if (labelEl) {
      labelEl.textContent = label || statusLabels[status] || status;
    } else {
      connectionStatus.textContent = label || statusLabels[status] || status;
    }
  }

  // ─── UI: Recording / Live Indicator ──────────────────────────────────────────
  /**
   * Shows or hides the #recording-indicator element.
   * Wrapper kept for compatibility with test_voice.py which calls this by name.
   * In full-duplex mode the indicator means "mic pipeline is live", not PTT record.
   * @param {boolean} active
   */
  function setRecordingIndicator(active) {
    _setLiveIndicator(active);
  }

  // ─── UI: AI Speaking Indicator ────────────────────────────────────────────────
  /**
   * Shows or hides the #ai-speaking-indicator element.
   * @param {boolean} active
   */
  function setAISpeakingIndicator(active) {
    if (!aiSpeakingIndicator) return;
    aiSpeakingIndicator.style.display = active ? '' : 'none';
    aiSpeakingIndicator.setAttribute('aria-hidden', String(!active));
    _updateDuplexState();
  }

  // ─── UI: Input Enable / Disable ──────────────────────────────────────────────
  /**
   * Enables or disables the message input and send button.
   * In voice mode the textarea is hidden — this is a no-op for voice.
   * @param {boolean} disabled
   */
  function disableInput(disabled) {
    if (messageInput) messageInput.disabled = disabled;
    if (sendBtn) {
      sendBtn.disabled = disabled ||
        !(messageInput && messageInput.value && messageInput.value.trim());
    }
  }

  // ─── UI: Auto-scroll ──────────────────────────────────────────────────────────
  function scrollToBottom() {
    if (chatMessages) chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  // ─── UI: Reconnect Prompt ─────────────────────────────────────────────────────
  function showReconnectPrompt() {
    const el   = document.createElement('div');
    el.classList.add('special-message', 'special-message-error');

    const text = document.createTextNode('⚠️ Connection lost. ');
    el.appendChild(text);

    const btn = document.createElement('button');
    btn.id          = 'reconnect-btn';
    btn.className   = 'reconnect-btn';
    btn.textContent = 'Return to Setup';
    btn.addEventListener('click', resetToSetup);
    el.appendChild(btn);

    chatMessages.appendChild(el);
    scrollToBottom();
  }

  // ─── Reset to Setup Screen ────────────────────────────────────────────────────
  function resetToSetup() {
    if (ws && ws.readyState === WebSocket.OPEN) ws.close(1000, 'User reset');
    ws = null;
    currentAIBubble = null;
    isAIResponding  = false;

    _cleanupVoice();

    if (chatMessages) chatMessages.innerHTML = '';
    hideTypingIndicator();

    if (messageInput) messageInput.style.height = 'auto';
    if (charCount) {
      charCount.textContent = '';
      charCount.classList.remove('warn');
    }

    disableInput(false);
    if (endBtn)   endBtn.disabled   = false;
    if (startBtn) startBtn.disabled = !(roleSelect && roleSelect.value);

    if (speakerBtn) {
      isSpeakerMuted = false;
      speakerBtn.classList.remove('muted');
      speakerBtn.setAttribute('aria-pressed', 'false');
      speakerBtn.title = 'Mute AI audio';
    }

    chatScreen.style.display  = 'none';
    setupScreen.style.display = '';
    setConnectionStatus('disconnected');
  }

  // ─── Utility: Format Time ─────────────────────────────────────────────────────
  function formatTime(date) {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }

})(); // End IIFE
