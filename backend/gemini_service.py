"""
Gemini Interview Service
Handles all interaction with the Google Gemini API for conducting
AI-powered mock interviews with real-time streaming responses.

Provides two service classes:
- GeminiInterviewService  — text-based streaming (gemini-2.0-flash)
- GeminiVoiceService      — real-time full-duplex audio (gemini-2.0-flash-live-001)

v2.0 changes (GeminiVoiceService):
- Added ``receive()`` async generator as the primary consumer interface.
  Yields ``LiveResponse``-like objects with ``.data``, ``.text``, and
  ``.server_content`` attributes (``interrupted``, ``turn_complete``).
- ``iter_audio_chunks()`` retained as a **deprecated** compatibility shim
  for any code that references the v1.0 ``(event_type, data)`` tuple API.
- Background ``_receive_loop`` / asyncio.Queue internal model replaced by
  direct ``async for response in self._session.receive()`` inside
  ``receive()``; the queue is still used to bridge the background task to
  the generator so ``send_audio()`` and ``receive()`` stay fully concurrent.
"""

from __future__ import annotations

import asyncio
import logging
import os
import warnings
from typing import AsyncGenerator, AsyncIterator, Callable, Awaitable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------

_TEXT_MODEL = "gemini-2.0-flash"
_VOICE_MODEL = "gemini-2.0-flash-live-001"

# Gemini Live API audio format:
#   input  — PCM 16-bit signed, 16 kHz, mono  (client microphone)
#   output — PCM 16-bit signed, 24 kHz, mono  (AI voice response)
VOICE_INPUT_SAMPLE_RATE: int = 16_000
VOICE_OUTPUT_SAMPLE_RATE: int = 24_000

# Sentinel placed on the internal queue to signal end-of-stream.
_QUEUE_SENTINEL = object()

# ---------------------------------------------------------------------------
# System prompt template (shared by both service classes)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """You are a professional {role} interviewer. Your name is Alex.
You are interviewing {candidate_name} for a {role} position.

Guidelines:
- Start with a warm greeting and introduce yourself
- Ask one question at a time, wait for the response
- Ask relevant technical and behavioral questions for the {role} role
- Provide follow-up questions based on answers
- Be encouraging but also challenging
- When the interview ends, provide constructive feedback including:
  - Strengths observed
  - Areas for improvement
  - Overall assessment
  - Tips for future interviews
"""

# ---------------------------------------------------------------------------
# Conversation history helpers (text mode)
# ---------------------------------------------------------------------------


def _make_user_content(text: str) -> dict:
    """Build a user Content dict for the Gemini API."""
    return {"role": "user", "parts": [{"text": text}]}


def _make_model_content(text: str) -> dict:
    """Build a model Content dict for the Gemini API."""
    return {"role": "model", "parts": [{"text": text}]}


# ---------------------------------------------------------------------------
# Text-based service (existing, unchanged)
# ---------------------------------------------------------------------------


class GeminiInterviewService:
    """
    Encapsulates a single interview session backed by the Gemini API.

    The service maintains a conversation history so that every call to
    `send_message` has the full context of the preceding exchanges, giving
    the AI an authentic sense of how the conversation has progressed.

    Usage::

        svc = GeminiInterviewService(role="Software Engineer")
        async for chunk in svc.start_session():
            print(chunk, end="", flush=True)

        async for chunk in svc.send_message("I have 5 years of Python experience."):
            print(chunk, end="", flush=True)

        async for chunk in svc.end_session():
            print(chunk, end="", flush=True)

        await svc.close()
    """

    def __init__(self, role: str, candidate_name: str = "Candidate") -> None:
        self.role = role
        self.candidate_name = candidate_name
        self._system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            role=role,
            candidate_name=candidate_name,
        )
        self._conversation_history: list[dict] = []
        self._full_response_buffer: str = ""

        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            logger.warning(
                "GEMINI_API_KEY is not set. "
                "The service will raise an error when a session is started."
            )
            self._client = None
            self._api_key_missing = True
        else:
            try:
                from google import genai  # type: ignore[import]

                self._client = genai.Client(api_key=api_key)
                self._genai = genai
                self._api_key_missing = False
            except ImportError as exc:
                logger.error("google-genai package is not installed: %s", exc)
                self._client = None
                self._api_key_missing = True

    # ------------------------------------------------------------------
    # Internal streaming helper
    # ------------------------------------------------------------------

    async def _stream_response(
        self, prompt_addition: str, role_tag: str = "user"
    ) -> AsyncGenerator[str, None]:
        """
        Append *prompt_addition* to the conversation history, call Gemini
        with the full history, stream the response chunks, and then record
        the completed response back into the history.

        Raises:
            RuntimeError: If the API key is missing.
            Exception: Propagated from the Gemini SDK (auth errors, rate
                limits, etc.) after logging.
        """
        if self._api_key_missing or self._client is None:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured. "
                "Please set it in your .env file."
            )

        # Add the new user turn to history
        if role_tag == "user":
            self._conversation_history.append(_make_user_content(prompt_addition))

        full_text_parts: list[str] = []

        try:
            from google.genai import types as genai_types  # type: ignore[import]

            response_stream = await self._client.aio.models.generate_content_stream(
                model=_TEXT_MODEL,
                contents=self._conversation_history,
                config=genai_types.GenerateContentConfig(
                    system_instruction=self._system_prompt,
                ),
            )

            async for chunk in response_stream:
                text = getattr(chunk, "text", None)
                if text:
                    full_text_parts.append(text)
                    yield text

        except Exception as exc:
            logger.exception("Error streaming response from Gemini: %s", exc)
            raise

        # Record the completed assistant turn
        full_response = "".join(full_text_parts)
        if full_response:
            self._conversation_history.append(_make_model_content(full_response))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start_session(self) -> AsyncGenerator[str, None]:
        """
        Kick off the interview session.

        Sends an instruction to Alex to greet the candidate and ask the
        first interview question. Yields response chunks as they arrive.
        """
        opening_prompt = (
            f"Please greet {self.candidate_name} warmly, introduce yourself as Alex, "
            f"briefly explain the interview format for the {self.role} position, "
            "and then ask your first interview question."
        )
        async for chunk in self._stream_response(opening_prompt):
            yield chunk

    async def send_message(self, user_message: str) -> AsyncGenerator[str, None]:
        """
        Forward a candidate message to Gemini and yield response chunks.

        The full conversation history (including this new message) is sent
        with every request so the model has complete context.
        """
        async for chunk in self._stream_response(user_message):
            yield chunk

    async def end_session(self) -> AsyncGenerator[str, None]:
        """
        Signal end-of-interview and request comprehensive feedback.

        Yields feedback chunks as they arrive from Gemini.
        """
        feedback_prompt = (
            "The interview is now over. Please provide comprehensive feedback for "
            f"{self.candidate_name} including:\n"
            "1. Key strengths demonstrated during the interview\n"
            "2. Areas for improvement with specific advice\n"
            "3. Overall assessment and suitability for the role\n"
            "4. Actionable tips for future interviews\n\n"
            "Be specific, constructive, and encouraging."
        )
        async for chunk in self._stream_response(feedback_prompt):
            yield chunk

    async def close(self) -> None:
        """
        Release any resources held by the service.

        The google-genai async client does not require explicit closing, but
        this method is provided for symmetry and future-proofing.
        """
        logger.debug(
            "GeminiInterviewService closed for role=%s candidate=%s",
            self.role,
            self.candidate_name,
        )
        self._client = None


# ---------------------------------------------------------------------------
# Live response wrapper — v2.0 receive() interface
# ---------------------------------------------------------------------------


class _LiveResponseWrapper:
    """
    Lightweight wrapper that normalises a raw Gemini Live session response
    into the shape expected by the v2.0 ``receive()`` interface:

    Attributes
    ----------
    data : bytes | None
        Raw PCM audio bytes from the AI, if present in this response chunk.
    text : str | None
        Transcript text, if present in this response chunk.
    server_content : _ServerContent | None
        Control signals: ``interrupted`` and ``turn_complete`` flags.

    This wrapper insulates ``_gemini_receive_loop`` in main.py from the raw
    Gemini SDK object shape, and makes the contract mockable in tests.
    """

    class _ServerContent:
        __slots__ = ("interrupted", "turn_complete")

        def __init__(self, interrupted: bool, turn_complete: bool) -> None:
            self.interrupted = interrupted
            self.turn_complete = turn_complete

    def __init__(
        self,
        data: bytes | None,
        text: str | None,
        interrupted: bool,
        turn_complete: bool,
    ) -> None:
        self.data = data
        self.text = text
        # Only attach server_content when there's something meaningful to report,
        # matching the SDK's own behaviour of omitting the field when absent.
        if interrupted or turn_complete:
            self.server_content: _LiveResponseWrapper._ServerContent | None = (
                self._ServerContent(interrupted=interrupted, turn_complete=turn_complete)
            )
        else:
            self.server_content = None


def _parse_live_response(response: object) -> _LiveResponseWrapper:
    """
    Extract audio bytes, text, and server_content signals from a raw
    Gemini Live session response object and return a ``_LiveResponseWrapper``.

    Works with any object shape the Gemini SDK may return — uses ``getattr``
    with safe defaults throughout so future SDK changes don't raise.
    """
    audio_bytes: bytes | None = None
    transcript_text: str | None = None
    interrupted = False
    turn_complete = False

    server_content = getattr(response, "server_content", None)
    if server_content is not None:
        # Extract interrupted / turn_complete signals
        interrupted = bool(getattr(server_content, "interrupted", False))
        turn_complete = bool(getattr(server_content, "turn_complete", False))

        model_turn = getattr(server_content, "model_turn", None)
        if model_turn is not None:
            parts = getattr(model_turn, "parts", None) or []
            for part in parts:
                inline_data = getattr(part, "inline_data", None)
                if inline_data is not None:
                    pcm = getattr(inline_data, "data", None)
                    if pcm:
                        # Accumulate audio — real SDK sends one part per response
                        # but be defensive in case there are multiple parts.
                        audio_bytes = pcm if audio_bytes is None else audio_bytes + pcm

                text = getattr(part, "text", None)
                if text:
                    transcript_text = (
                        text if transcript_text is None else transcript_text + text
                    )

    # Some SDK versions surface .data / .text at the top level (newer API)
    top_data = getattr(response, "data", None)
    if top_data and isinstance(top_data, bytes):
        audio_bytes = top_data if audio_bytes is None else audio_bytes + top_data

    top_text = getattr(response, "text", None)
    if top_text and isinstance(top_text, str):
        transcript_text = (
            top_text if transcript_text is None else transcript_text + top_text
        )

    return _LiveResponseWrapper(
        data=audio_bytes,
        text=transcript_text,
        interrupted=interrupted,
        turn_complete=turn_complete,
    )


# ---------------------------------------------------------------------------
# Voice-based service (Gemini Live API) — Full Duplex v2.0
# ---------------------------------------------------------------------------

# Type alias for the audio sender callback (kept for backward compat).
AudioSenderCallback = Callable[[bytes], Awaitable[None]]


class GeminiVoiceService:
    """
    Real-time full-duplex voice interview session backed by the Gemini Live API.

    The service opens a persistent bidirectional audio session with
    ``gemini-2.0-flash-live-001``.  Both directions flow simultaneously:

    - The caller feeds raw PCM audio chunks in via ``send_audio()`` (from
      the browser microphone) at any time, even while the AI is responding.
    - The caller consumes AI audio responses via ``receive()``, an async
      generator that yields ``_LiveResponseWrapper`` objects containing
      audio bytes, transcript text, and control signals.

    This full-duplex design means the user can interrupt the AI at any time
    (barge-in), and the Gemini Live API handles graceful turn management.

    Audio format contract
    ---------------------
    - **Input  (client → Gemini):** PCM 16-bit signed, 16 kHz, mono
    - **Output (Gemini → client):** PCM 16-bit signed, 24 kHz, mono

    v2.0 primary interface
    -----------------------
    ``receive()`` — async generator yielding ``_LiveResponseWrapper`` objects:

    +---------------------------------+----------------------------------------+
    | Attribute                       | Meaning                                |
    +=================================+========================================+
    | ``.data``                       | ``bytes`` — raw PCM audio chunk        |
    +---------------------------------+----------------------------------------+
    | ``.text``                       | ``str`` — AI transcript text           |
    +---------------------------------+----------------------------------------+
    | ``.server_content.interrupted`` | ``bool`` — barge-in detected           |
    +---------------------------------+----------------------------------------+
    | ``.server_content.turn_complete``| ``bool`` — AI turn finished           |
    +---------------------------------+----------------------------------------+

    Deprecated v1.0 interface
    -------------------------
    ``iter_audio_chunks()`` is kept for backward compatibility with tests that
    reference the ``(event_type, data)`` tuple API.  Do not use in new code.

    Usage (inside an async context)::

        svc = GeminiVoiceService(role="Software Engineer", candidate_name="Alex")
        await svc.connect()          # open Gemini Live session + start interview

        # Both tasks run concurrently via asyncio.gather():
        await asyncio.gather(
            _ws_receive_loop(websocket, svc),
            _gemini_receive_loop(websocket, svc),
        )
        # finally: await svc.close()
    """

    def __init__(self, role: str, candidate_name: str = "Candidate") -> None:
        self.role = role
        self.candidate_name = candidate_name
        self._system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            role=role,
            candidate_name=candidate_name,
        )

        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            logger.warning(
                "GEMINI_API_KEY is not set — GeminiVoiceService will raise on connect."
            )
            self._client = None
            self._api_key_missing = True
        else:
            try:
                from google import genai  # type: ignore[import]

                self._client = genai.Client(api_key=api_key)
                self._genai = genai
                self._api_key_missing = False
            except ImportError as exc:
                logger.error("google-genai package is not installed: %s", exc)
                self._client = None
                self._api_key_missing = True

        # Set after connect()
        self._session = None        # Gemini Live session context manager value
        self._session_ctx = None    # context manager object (has __aexit__)
        self._connected: bool = False

        # Internal queue: bridges the background _receive_loop to receive().
        # Items are _LiveResponseWrapper instances; _QUEUE_SENTINEL signals EOF.
        self._event_queue: asyncio.Queue = asyncio.Queue()

        # Background task running the Gemini receive loop
        self._receive_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """
        Open the Gemini Live session and send the opening greeting prompt.

        Must be called before ``send_audio()`` or ``receive()``.

        Raises:
            RuntimeError: If GEMINI_API_KEY is not configured.
            Exception: Propagated from the Gemini SDK on auth / network failures.
        """
        if self._api_key_missing or self._client is None:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured. "
                "Please set it in your .env file."
            )

        try:
            from google.genai import types as genai_types  # type: ignore[import]

            live_config = genai_types.LiveConnectConfig(
                response_modalities=["AUDIO"],
                system_instruction=self._system_prompt,
                speech_config=genai_types.SpeechConfig(
                    voice_config=genai_types.VoiceConfig(
                        prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                            voice_name="Charon",
                        )
                    )
                ),
                # Gemini Live enables built-in VAD and barge-in by default.
                # Do NOT set end_of_turn manually — let Gemini Live VAD decide.
            )

            self._session_ctx = self._client.aio.live.connect(
                model=_VOICE_MODEL,
                config=live_config,
            )
            self._session = await self._session_ctx.__aenter__()
            self._connected = True
            logger.info(
                "GeminiVoiceService connected: role=%s candidate=%s",
                self.role,
                self.candidate_name,
            )
        except Exception as exc:
            logger.exception("Failed to connect to Gemini Live API: %s", exc)
            raise

        # Send the opening greeting prompt so Gemini starts the interview.
        opening_prompt = (
            f"Please greet {self.candidate_name} warmly, introduce yourself as Alex, "
            f"briefly explain the interview format for the {self.role} position, "
            "and then ask your first interview question."
        )
        await self._session.send(input=opening_prompt, end_of_turn=True)

        # Start the background task that consumes Gemini Live responses and
        # puts wrapped response objects onto the internal queue.
        self._receive_task = asyncio.create_task(
            self._receive_loop(),
            name="gemini-voice-receive",
        )
        logger.info(
            "Voice interview started for candidate=%s", self.candidate_name
        )

    # ------------------------------------------------------------------
    # Audio I/O (v2.0)
    # ------------------------------------------------------------------

    async def send_audio(self, pcm_bytes: bytes) -> None:
        """
        Send a raw PCM audio chunk (16-bit, 16 kHz, mono) to Gemini.

        Non-blocking fire-and-forget.  The user can send audio at any time —
        even while the AI is producing a response — enabling true barge-in.

        Safe to call concurrently with ``receive()`` from a separate asyncio
        task: the Gemini SDK uses separate send/receive paths internally.

        Raises:
            RuntimeError: If the session is not connected.
        """
        if not self._connected or self._session is None:
            raise RuntimeError(
                "GeminiVoiceService is not connected. Call connect() first."
            )

        try:
            from google.genai import types as genai_types  # type: ignore[import]

            # Wrap raw bytes in a Blob with the correct MIME type for PCM audio.
            await self._session.send(
                input=genai_types.LiveClientRealtimeInput(
                    media_chunks=[
                        genai_types.Blob(
                            data=pcm_bytes,
                            mime_type="audio/pcm;rate=16000",
                        )
                    ]
                )
            )
        except Exception as exc:
            logger.exception("Error sending audio to Gemini Live: %s", exc)
            raise

    async def receive(self) -> AsyncGenerator[_LiveResponseWrapper, None]:
        """
        Async generator that yields Gemini Live responses as they arrive.

        **v2.0 primary interface.**  Replaces ``iter_audio_chunks()`` for the
        ``_gemini_receive_loop`` task in main.py.

        Each yielded ``_LiveResponseWrapper`` may contain:

        - ``.data``   — ``bytes | None``  raw PCM audio chunk (24 kHz mono)
        - ``.text``   — ``str | None``    AI transcript text
        - ``.server_content.interrupted``  — ``bool`` barge-in detected
        - ``.server_content.turn_complete`` — ``bool`` AI turn finished

        The generator terminates when the Gemini session closes (sentinel
        received on the internal queue).

        Safe to run concurrently with ``send_audio()`` via ``asyncio.gather()``.
        """
        while True:
            item = await self._event_queue.get()
            if item is _QUEUE_SENTINEL:
                # Receive loop has exited — no more responses.
                break
            yield item  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Deprecated v1.0 compatibility interface
    # ------------------------------------------------------------------

    async def iter_audio_chunks(self) -> AsyncIterator[tuple[str, object]]:
        """
        **Deprecated — use ``receive()`` instead.**

        Compatibility shim for v1.0 consumers that expect
        ``(event_type, data)`` tuples from the audio event stream.

        Maps ``_LiveResponseWrapper`` objects to the old tuple format:

        +-----------------+------------------------------------------+
        | event_type      | data type                                |
        +=================+==========================================+
        | ``"audio"``     | ``bytes`` — raw PCM (24 kHz mono)       |
        +-----------------+------------------------------------------+
        | ``"audio_start"`` | ``None``                              |
        +-----------------+------------------------------------------+
        | ``"audio_end"`` | ``None``                                 |
        +-----------------+------------------------------------------+
        | ``"transcript"``| ``tuple[str, str]`` — (text, speaker)   |
        +-----------------+------------------------------------------+
        """
        warnings.warn(
            "iter_audio_chunks() is deprecated in v2.0; use receive() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        audio_turn_active = False
        async for wrapper in self.receive():
            if wrapper.data:
                if not audio_turn_active:
                    audio_turn_active = True
                    yield ("audio_start", None)
                yield ("audio", wrapper.data)
            if wrapper.text:
                yield ("transcript", (wrapper.text, "ai"))
            if wrapper.server_content:
                if wrapper.server_content.turn_complete and audio_turn_active:
                    audio_turn_active = False
                    yield ("audio_end", None)

    # ------------------------------------------------------------------
    # Background receive loop (internal)
    # ------------------------------------------------------------------

    async def _receive_loop(self) -> None:
        """
        Background task: consume raw Gemini Live responses, parse them into
        ``_LiveResponseWrapper`` objects, and put them on the internal queue.

        Always puts the sentinel on the queue when it exits so that
        ``receive()`` (and the deprecated ``iter_audio_chunks()``) terminate.
        """
        if self._session is None:
            await self._event_queue.put(_QUEUE_SENTINEL)
            return

        try:
            async for raw_response in self._session.receive():
                wrapper = _parse_live_response(raw_response)
                await self._event_queue.put(wrapper)

        except asyncio.CancelledError:
            logger.debug("GeminiVoiceService receive loop cancelled.")
        except Exception as exc:
            logger.exception("Error in Gemini Live receive loop: %s", exc)
            raise
        finally:
            # Always signal end-of-stream so receive() terminates cleanly.
            await self._event_queue.put(_QUEUE_SENTINEL)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """
        Cleanly shut down the Gemini Live session and cancel the receive task.

        Safe to call multiple times and in a ``finally`` block.
        """
        # Cancel background receive task first
        if self._receive_task is not None and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await self._receive_task
            except (asyncio.CancelledError, Exception):
                pass
            self._receive_task = None

        # Drain the queue and put the sentinel so any waiting
        # receive() / iter_audio_chunks() call unblocks immediately.
        try:
            while not self._event_queue.empty():
                self._event_queue.get_nowait()
        except Exception:
            pass
        try:
            await self._event_queue.put(_QUEUE_SENTINEL)
        except Exception:
            pass

        # Exit the Gemini Live session context manager
        if self._session_ctx is not None and self._connected:
            try:
                await self._session_ctx.__aexit__(None, None, None)
            except Exception as exc:
                logger.warning("Error closing Gemini Live session: %s", exc)

        self._session = None
        self._session_ctx = None
        self._connected = False
        logger.info(
            "GeminiVoiceService closed for role=%s candidate=%s",
            self.role,
            self.candidate_name,
        )
