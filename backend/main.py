"""
AI Interview Platform — FastAPI Application v2.0
=================================================
Routes
------
  GET  /                       Serve the frontend SPA (static/index.html)
  GET  /api/health             Health check endpoint
  GET  /api/config/roles       Return the list of available interview roles
  WS   /ws/interview           Real-time text interview session via WebSocket
  WS   /ws/voice-interview     Real-time full-duplex voice interview via WebSocket

WebSocket message protocol — text mode (/ws/interview)
-------------------------------------------------------
Client → Server  (JSON)
  {"type": "start",   "data": {"role": "<role>", "candidate_name": "<name>"}}
  {"type": "message", "data": {"content": "<user utterance>"}}
  {"type": "end",     "data": {}}

Server → Client  (JSON)
  {"type": "chunk",    "data": {"content": "<partial text>", "done": false}}
  {"type": "response", "data": {"content": "<full text>",    "done": true}}
  {"type": "error",    "data": {"content": "<error message>"}}

WebSocket message protocol — voice mode (/ws/voice-interview)  [v2.0]
-----------------------------------------------------------------------
Full-duplex: both audio directions flow simultaneously without turn-taking.
Gemini Live's built-in VAD handles end-of-turn and barge-in detection;
no explicit push-to-talk signalling is needed from the client.

Client → Server  (JSON text frames — control plane)
  {"type": "start", "data": {"role": "...", "candidate_name": "...", "mode": "voice"}}
  {"type": "end",   "data": {}}
  All other JSON types are silently ignored (backward-compat with v1.0 clients
  that may send voice_start_speaking / voice_stop_speaking).

Client → Server  (binary frames — audio data plane)
  Raw PCM audio: 16-bit signed, 16 kHz, mono.
  Maximum frame size: 256 KB.

Server → Client  (JSON text frames — control plane)
  {"type": "session_ready", "data": {"mode": "voice", "output_sample_rate": 24000}}
      — sent once after the Gemini Live session is open; client should start mic.
  {"type": "transcript",   "data": {"speaker": "ai", "content": "...", "final": true}}
      — AI speech transcript (and optionally user transcript when enabled).
  {"type": "ai_turn_complete", "data": {}}
      — AI has finished its response turn.
  {"type": "ai_interrupted",   "data": {}}
      — Barge-in detected; client should flush its audio playback queue.
  {"type": "error",            "data": {"content": "..."}}

Server → Client  (binary frames — audio data plane)
  Raw PCM audio: 16-bit signed, 24 kHz, mono.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.gemini_service import GeminiInterviewService, GeminiVoiceService

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

load_dotenv()  # Load .env if present — harmless when vars already in environment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

if not os.getenv("GEMINI_API_KEY", "").strip():
    logger.warning(
        "⚠️  GEMINI_API_KEY is not set. "
        "The app will start, but interview sessions will not work until "
        "the key is supplied via the GEMINI_API_KEY environment variable."
    )

# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent  # /workspace/interview_platform/
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="AI Interview Platform",
    description="Practice job interviews with a real-time AI interviewer powered by Gemini.",
    version="2.0.0",
)

# Mount static assets (CSS, JS, images).  The directory must exist for the
# mount to succeed — create it lazily so the app starts even before the
# UX engineer delivers assets.
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ---------------------------------------------------------------------------
# Available interview roles
# ---------------------------------------------------------------------------

INTERVIEW_ROLES: list[str] = [
    "Software Engineer",
    "Data Scientist",
    "Product Manager",
    "System Design",
]

# ---------------------------------------------------------------------------
# Constants — exported so tests can reference them directly
# ---------------------------------------------------------------------------

# Maximum allowed length for a single candidate message.
# Aligned with the frontend HTML maxlength attribute.
MAX_MESSAGE_LENGTH: int = 2000

# Maximum binary audio frame size (256 KB).
# Prevents memory exhaustion from oversized frames.
MAX_AUDIO_CHUNK_BYTES: int = 256 * 1024  # 256 KB

# Minimal fallback HTML returned when index.html has not been deployed yet.
_FALLBACK_HTML: str = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AI Interview Platform</title>
</head>
<body>
  <h1>AI Interview Platform</h1>
  <p>The frontend assets are not yet available.  Backend is running.</p>
  <p>Visit <a href="/docs">/docs</a> for the API documentation.</p>
</body>
</html>"""

# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False, response_model=None)
async def serve_index():  # type: ignore[return]
    """Serve the single-page application entry point.

    Falls back to a minimal inline HTML page when index.html has not been
    deployed, so the backend can be tested independently without causing a
    500 error.
    """
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        # Return a safe inline fallback — never attempt FileResponse on a
        # path that does not exist, as that raises a FileNotFoundError.
        return HTMLResponse(content=_FALLBACK_HTML, status_code=200)
    return FileResponse(path=str(index_path), media_type="text/html")


@app.get("/api/health", include_in_schema=True)
async def health_check() -> JSONResponse:
    """Simple health check endpoint.

    Returns ``{"status": "ok"}`` with HTTP 200.  Used by the frontend,
    load-balancers, and tests to verify the server is reachable.
    """
    return JSONResponse(content={"status": "ok"})


@app.get("/api/config/roles", response_model=list[str])
async def get_roles() -> list[str]:
    """Return the list of available interview roles."""
    return INTERVIEW_ROLES


# ---------------------------------------------------------------------------
# WebSocket helper utilities
# ---------------------------------------------------------------------------


async def _send_json(ws: WebSocket, payload: dict[str, Any]) -> None:
    """Serialize *payload* to JSON and send it over *ws*."""
    await ws.send_text(json.dumps(payload))


async def _send_error(ws: WebSocket, message: str) -> None:
    """Send a structured error message to the client."""
    await _send_json(ws, {"type": "error", "data": {"content": message}})


async def _stream_to_ws(
    ws: WebSocket,
    generator,  # AsyncGenerator[str, None]
) -> str:
    """
    Drive an async generator that yields text chunks, forwarding each chunk
    to the WebSocket client as it arrives.

    Returns the concatenated full response text so the caller can send the
    final `response` frame.
    """
    accumulated: list[str] = []

    async for chunk in generator:
        if chunk:
            accumulated.append(chunk)
            await _send_json(
                ws,
                {
                    "type": "chunk",
                    "data": {"content": chunk, "done": False},
                },
            )

    full_response = "".join(accumulated)
    await _send_json(
        ws,
        {
            "type": "response",
            "data": {"content": full_response, "done": True},
        },
    )
    return full_response


# ---------------------------------------------------------------------------
# Text WebSocket endpoint (unchanged)
# ---------------------------------------------------------------------------


@app.websocket("/ws/interview")
async def interview_ws(websocket: WebSocket) -> None:
    """
    Real-time text interview session endpoint.

    Lifecycle
    ---------
    1. Accept connection (reject if API key is missing).
    2. Wait for a `start` message — instantiate ``GeminiInterviewService``
       and stream the opening greeting.
    3. Process `message` messages — stream AI responses.
    4. On `end` message — stream feedback, then close cleanly.
    5. Handle disconnections and errors gracefully throughout.
    """
    await websocket.accept()
    logger.info("WebSocket connection accepted from %s", websocket.client)

    # Guard: reject immediately if the API key is absent
    if not os.getenv("GEMINI_API_KEY", "").strip():
        await _send_error(
            websocket,
            "GEMINI_API_KEY is not configured on the server. "
            "Please contact the administrator.",
        )
        await websocket.close(code=1011)
        return

    interview_service: GeminiInterviewService | None = None

    try:
        while True:
            raw = await websocket.receive_text()

            # ----------------------------------------------------------------
            # Parse incoming message
            # ----------------------------------------------------------------
            try:
                msg = json.loads(raw)
                msg_type: str = msg.get("type", "")
                msg_data: dict[str, Any] = msg.get("data", {})
            except (json.JSONDecodeError, AttributeError):
                await _send_error(websocket, "Invalid JSON message format.")
                continue

            # ----------------------------------------------------------------
            # Handle: start
            # ----------------------------------------------------------------
            if msg_type == "start":
                role: str = str(msg_data.get("role", "Software Engineer"))
                # Sanitize candidate_name: strip non-printable/control
                # characters and cap at 100 chars to prevent prompt-injection
                # payloads from escaping the Gemini system prompt context.
                raw_name: str = str(msg_data.get("candidate_name", "Candidate"))
                candidate_name: str = (
                    "".join(ch for ch in raw_name if ch.isprintable()).strip()[:100]
                    or "Candidate"
                )

                if role not in INTERVIEW_ROLES:
                    await _send_error(
                        websocket,
                        f"Unknown role '{role}'. "
                        f"Valid roles are: {', '.join(INTERVIEW_ROLES)}.",
                    )
                    continue

                # Clean up any existing session
                if interview_service is not None:
                    await interview_service.close()

                interview_service = GeminiInterviewService(
                    role=role,
                    candidate_name=candidate_name,
                )
                logger.info(
                    "Interview started: role=%s candidate=%s",
                    role,
                    candidate_name,
                )

                try:
                    await _stream_to_ws(websocket, interview_service.start_session())
                except RuntimeError as exc:
                    await _send_error(websocket, str(exc))
                except Exception as exc:
                    logger.exception("Error during session start: %s", exc)
                    await _send_error(
                        websocket,
                        _classify_gemini_error(exc),
                    )

            # ----------------------------------------------------------------
            # Handle: message
            # ----------------------------------------------------------------
            elif msg_type == "message":
                if interview_service is None:
                    await _send_error(
                        websocket,
                        "No active interview session. Please send a 'start' message first.",
                    )
                    continue

                user_content: str = msg_data.get("content", "").strip()
                if not user_content:
                    await _send_error(websocket, "Message content cannot be empty.")
                    continue

                # Enforce server-side length cap (HTML maxlength is client-only
                # and trivially bypassed via a raw WebSocket client).
                if len(user_content) > MAX_MESSAGE_LENGTH:
                    await _send_error(
                        websocket,
                        f"Message too long. Maximum length is {MAX_MESSAGE_LENGTH} characters.",
                    )
                    continue

                try:
                    await _stream_to_ws(
                        websocket, interview_service.send_message(user_content)
                    )
                except RuntimeError as exc:
                    await _send_error(websocket, str(exc))
                except Exception as exc:
                    logger.exception("Error during message handling: %s", exc)
                    await _send_error(websocket, _classify_gemini_error(exc))

            # ----------------------------------------------------------------
            # Handle: end
            # ----------------------------------------------------------------
            elif msg_type == "end":
                if interview_service is None:
                    await _send_error(
                        websocket,
                        "No active interview session to end.",
                    )
                    continue

                try:
                    await _stream_to_ws(
                        websocket, interview_service.end_session()
                    )
                except RuntimeError as exc:
                    await _send_error(websocket, str(exc))
                except Exception as exc:
                    logger.exception("Error during session end: %s", exc)
                    await _send_error(websocket, _classify_gemini_error(exc))
                finally:
                    await interview_service.close()
                    interview_service = None

                # Signal graceful close to the client
                await websocket.close(code=1000)
                return

            # ----------------------------------------------------------------
            # Unknown message type
            # ----------------------------------------------------------------
            else:
                await _send_error(
                    websocket,
                    f"Unknown message type '{msg_type}'. "
                    "Expected one of: start, message, end.",
                )

    except WebSocketDisconnect:
        logger.info(
            "WebSocket client disconnected from %s", websocket.client
        )
    except Exception as exc:
        logger.exception("Unexpected WebSocket error: %s", exc)
        try:
            await _send_error(websocket, "An unexpected server error occurred.")
        except Exception:
            pass  # Connection may already be dead
    finally:
        if interview_service is not None:
            await interview_service.close()
            logger.debug("Interview service cleaned up on disconnect.")


# ---------------------------------------------------------------------------
# Voice WebSocket endpoint — Full Duplex v2.0
# ---------------------------------------------------------------------------


async def _ws_receive_loop(websocket: WebSocket, svc: GeminiVoiceService) -> None:
    """
    Task A — Browser → Gemini (continuously).

    Reads frames from the WebSocket and forwards them to the Gemini Live
    session until the client signals end-of-session or disconnects.

    Frame handling:
    - Binary frame  → ``svc.send_audio(pcm_bytes)``
    - JSON "end"    → ``break``  (exits loop → exits gather())
    - All other JSON → silently ignored (backward-compat: v1.0 clients may
      send ``voice_start_speaking`` / ``voice_stop_speaking`` — not an error)

    Raises ``WebSocketDisconnect`` on unexpected client disconnect, which
    propagates out of ``asyncio.gather()`` and triggers ``svc.close()`` in the
    ``finally`` block of ``voice_interview_ws``.
    """
    while True:
        message = await websocket.receive()

        if "disconnect" in message.get("type", ""):
            # WebSocket closed by client — raise so gather() tears down Task B.
            raise WebSocketDisconnect(code=1000)

        # ---- Binary frame: raw PCM audio ---------------------------------
        if message.get("bytes") is not None:
            pcm_data: bytes = message["bytes"]

            if len(pcm_data) > MAX_AUDIO_CHUNK_BYTES:
                await _send_error(
                    websocket,
                    f"Audio chunk too large. Maximum size is "
                    f"{MAX_AUDIO_CHUNK_BYTES // 1024} KB.",
                )
                continue

            await svc.send_audio(pcm_data)

        # ---- Text frame: control message ---------------------------------
        elif message.get("text") is not None:
            try:
                ctrl = json.loads(message["text"])
                ctrl_type: str = ctrl.get("type", "")
            except (json.JSONDecodeError, AttributeError):
                await _send_error(websocket, "Invalid JSON message format.")
                continue

            if ctrl_type == "end":
                # Graceful session end — break exits the loop, which causes
                # _ws_receive_loop to return normally.  asyncio.gather() will
                # then cancel _gemini_receive_loop automatically.
                break

            # All other message types (including legacy voice_start_speaking /
            # voice_stop_speaking) are silently ignored in v2.0 voice mode.
            # Logging at DEBUG level only to avoid noise.
            logger.debug(
                "Voice receive loop: ignoring control message type=%r", ctrl_type
            )


async def _gemini_receive_loop(
    websocket: WebSocket, svc: GeminiVoiceService
) -> None:
    """
    Task B — Gemini → Browser (continuously).

    Consumes responses from ``svc.receive()`` and forwards them to the client:
    - ``.data``                          → binary frame (raw PCM 24 kHz)
    - ``.text``                          → JSON transcript frame
    - ``.server_content.interrupted``   → JSON ``ai_interrupted`` frame
    - ``.server_content.turn_complete`` → JSON ``ai_turn_complete`` frame

    Exits when the Gemini session closes (``svc.receive()`` generator ends),
    which causes ``asyncio.gather()`` to cancel ``_ws_receive_loop``.
    """
    async for response in svc.receive():
        if response.data:
            # Raw PCM16 audio bytes → browser playback queue
            await websocket.send_bytes(response.data)

        if response.text:
            await _send_json(
                websocket,
                {
                    "type": "transcript",
                    "data": {
                        "speaker": "ai",
                        "content": response.text,
                        "final": True,
                    },
                },
            )

        if response.server_content:
            if response.server_content.interrupted:
                await _send_json(websocket, {"type": "ai_interrupted", "data": {}})
            if response.server_content.turn_complete:
                await _send_json(websocket, {"type": "ai_turn_complete", "data": {}})


@app.websocket("/ws/voice-interview")
async def voice_interview_ws(websocket: WebSocket) -> None:
    """
    Real-time full-duplex voice interview session endpoint (v2.0).

    Phase 1 — Handshake
    -------------------
    Wait for a ``{"type": "start"}`` JSON control message, extract role and
    candidate_name, open the Gemini Live session, and send ``session_ready``.

    Phase 2 — Full-duplex concurrent I/O via asyncio.gather()
    ----------------------------------------------------------
    Run two independent module-level coroutines concurrently:

    - ``_ws_receive_loop(websocket, svc)``   — Task A: Browser → Gemini
    - ``_gemini_receive_loop(websocket, svc)`` — Task B: Gemini → Browser

    ``asyncio.gather()`` provides atomic teardown: if either task exits or
    raises (e.g. ``WebSocketDisconnect`` in Task A), the other is immediately
    cancelled.  ``svc.close()`` is called in the ``finally`` block regardless
    of how the session ends.
    """
    await websocket.accept()
    logger.info("Voice WebSocket connection accepted from %s", websocket.client)

    # Guard: reject immediately if the API key is absent
    if not os.getenv("GEMINI_API_KEY", "").strip():
        await _send_error(
            websocket,
            "GEMINI_API_KEY is not configured on the server. "
            "Please contact the administrator.",
        )
        await websocket.close(code=1011)
        return

    voice_service: GeminiVoiceService | None = None

    # -------------------------------------------------------------------------
    # Phase 1: Handshake — wait for "start" before entering full-duplex mode.
    # We need role/candidate_name to open the Gemini session, so handle this
    # synchronously before spawning the concurrent send/receive tasks.
    # -------------------------------------------------------------------------

    try:
        while voice_service is None:
            frame = await websocket.receive()

            if "disconnect" in frame.get("type", ""):
                return

            if frame.get("bytes") is not None:
                # Binary audio before start — client is early; send a clear error
                await _send_error(
                    websocket,
                    "No active voice session. Send a 'start' message before audio.",
                )
                continue

            if frame.get("text") is None:
                continue

            try:
                msg = json.loads(frame["text"])
                msg_type: str = msg.get("type", "")
                msg_data: dict[str, Any] = msg.get("data", {})
            except (json.JSONDecodeError, AttributeError):
                await _send_error(websocket, "Invalid JSON message format.")
                continue

            if msg_type == "end":
                await _send_error(websocket, "No active voice interview session to end.")
                continue

            if msg_type != "start":
                await _send_error(
                    websocket,
                    f"Unknown message type '{msg_type}'. "
                    "Expected one of: start, end.",
                )
                continue

            # ---- Parse "start" message -----------------------------------
            role: str = str(msg_data.get("role", "Software Engineer"))
            raw_name: str = str(msg_data.get("candidate_name", "Candidate"))
            candidate_name: str = (
                "".join(ch for ch in raw_name if ch.isprintable()).strip()[:100]
                or "Candidate"
            )

            if role not in INTERVIEW_ROLES:
                await _send_error(
                    websocket,
                    f"Unknown role '{role}'. "
                    f"Valid roles are: {', '.join(INTERVIEW_ROLES)}.",
                )
                continue

            # Open the Gemini Live session
            voice_service = GeminiVoiceService(role=role, candidate_name=candidate_name)
            logger.info(
                "Voice interview starting: role=%s candidate=%s", role, candidate_name
            )

            try:
                await voice_service.connect()
            except RuntimeError as exc:
                await _send_error(websocket, str(exc))
                await voice_service.close()
                voice_service = None
                continue
            except Exception as exc:
                logger.exception("Error starting voice session: %s", exc)
                await _send_error(websocket, _classify_gemini_error(exc))
                await voice_service.close()
                voice_service = None
                continue

            # Notify the client: session open, audio can now flow.
            # v2.0: "session_ready" replaces the old "ready" message.
            await _send_json(
                websocket,
                {
                    "type": "session_ready",
                    "data": {"mode": "voice", "output_sample_rate": 24000},
                },
            )

    except WebSocketDisconnect:
        logger.info(
            "Voice WebSocket disconnected during handshake from %s", websocket.client
        )
        if voice_service is not None:
            await voice_service.close()
        return
    except Exception as exc:
        logger.exception("Error during voice WebSocket handshake: %s", exc)
        try:
            await _send_error(websocket, "An unexpected server error occurred.")
        except Exception:
            pass
        if voice_service is not None:
            await voice_service.close()
        return

    # -------------------------------------------------------------------------
    # Phase 2: Full-duplex — both directions run simultaneously.
    #
    # asyncio.gather() is the key architectural choice here:
    #   - Both tasks yield to the event loop on every await — no blocking.
    #   - If either task returns or raises, gather() cancels the other and
    #     the finally block runs svc.close().  No task leaks.
    # -------------------------------------------------------------------------

    try:
        await asyncio.gather(
            _ws_receive_loop(websocket, voice_service),
            _gemini_receive_loop(websocket, voice_service),
        )
    except WebSocketDisconnect:
        # Normal — client closed the connection during full-duplex phase.
        logger.info(
            "Voice WebSocket client disconnected during session from %s",
            websocket.client,
        )
    except Exception as exc:
        logger.exception("Unexpected error in full-duplex voice session: %s", exc)
        try:
            await _send_error(websocket, "An unexpected server error occurred.")
        except Exception:
            pass
    finally:
        if voice_service is not None:
            await voice_service.close()
            logger.debug("Voice interview service cleaned up.")

        try:
            await websocket.close(code=1000)
        except Exception:
            pass  # May already be closed


# ---------------------------------------------------------------------------
# Gemini error classification
# ---------------------------------------------------------------------------


def _classify_gemini_error(exc: Exception) -> str:
    """
    Map common Gemini / networking exceptions to user-friendly messages.

    Handles standard HTTP status codes and common Google API error keywords
    including authentication, permission denied, quota, rate-limiting,
    timeout, and model-not-found patterns.
    """
    error_str = str(exc).lower()

    if (
        "api_key" in error_str
        or "api key" in error_str
        or "401" in error_str
        or "permission" in error_str
        or "denied" in error_str
    ):
        return (
            "Authentication failed. Please check that a valid GEMINI_API_KEY "
            "is configured on the server."
        )
    if "quota" in error_str or "rate" in error_str or "429" in error_str:
        return (
            "The AI service is temporarily rate-limited. "
            "Please wait a few seconds and try again."
        )
    if "timeout" in error_str or "deadline" in error_str:
        return "The request to the AI service timed out. Please try again."
    if "not found" in error_str or "404" in error_str:
        return "The requested AI model was not found. Please contact the administrator."

    return (
        "An error occurred while communicating with the AI service. "
        "Please try again."
    )


# ---------------------------------------------------------------------------
# Entry point (development)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
