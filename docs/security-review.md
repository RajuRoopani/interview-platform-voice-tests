# Security Review — AI Interview Platform
**Reviewer:** Software Architect  
**Date:** 2025  
**Scope:** Full codebase — `backend/main.py`, `backend/gemini_service.py`, `static/app.js`, `static/index.html`, `tests/test_main.py`  
**Verdict:** ✅ **APPROVED FOR MERGE** — with two low-priority hardening recommendations noted below

---

## Executive Summary

The implementation is architecturally sound from a security standpoint. The most significant risks for a platform of this type (API key exposure, XSS, prompt injection, resource exhaustion) have all been addressed correctly or mitigated by design. No blocking issues were found.

---

## Findings

### ✅ PASS — API Key Handling

**File:** `backend/main.py`, `backend/gemini_service.py`

The API key is read exclusively from the environment (`os.getenv("GEMINI_API_KEY")`). The `.env.example` file contains only a placeholder — no real key. The `.gitignore` should ensure `.env` is never committed (see recommendation R-1 below).

The key is never echoed back to clients in any error message or response frame. The error message that _is_ sent on key absence says only `"GEMINI_API_KEY is not configured on the server"` — correct, it reveals the variable name but not its value.

---

### ✅ PASS — XSS Prevention (Frontend)

**File:** `static/app.js`

All dynamic content written to the DOM uses `element.textContent`, not `innerHTML`. This is consistent throughout:
- `addMessage()` → `bubble.textContent = content`
- `appendChunkToAIBubble()` → `bubbleEl.textContent += chunk`
- `setAIBubbleText()` → `bubbleEl.textContent = text`
- `addSpecialMessage()` → `el.textContent = content`
- `formatTime()` → produces a locale string assigned via `textContent`

**One exception — `showReconnectPrompt()`** uses `el.innerHTML` to embed a `<button>` element. However, the content is entirely static (`innerHTML = '⚠️ Connection lost. <button id="reconnect-btn"…'`) — no user-supplied or server-supplied data is interpolated. This is safe as written, but noted as a pattern to watch if the function is extended.

---

### ✅ PASS — Prompt Injection Mitigation

**File:** `backend/gemini_service.py`

The system prompt is constructed once at service instantiation from `role` and `candidate_name` values. The `role` value is validated against the `INTERVIEW_ROLES` allowlist in `main.py` before `GeminiInterviewService` is instantiated — it cannot be arbitrary. The `candidate_name` is passed through without sanitization (see recommendation R-2 below), but its influence is limited to the `candidate_name` slot in the system prompt template, not to the conversation history.

User message content in `send_message()` is passed directly to Gemini as conversation history — this is the intended use, and the risk is bounded to the Gemini session itself (no server-side state is modified by arbitrary input).

---

### ✅ PASS — Input Validation at WebSocket Boundary

**File:** `backend/main.py`

- JSON parse errors are caught and return a structured error — the connection is not dropped.
- `role` is validated against the `INTERVIEW_ROLES` allowlist before use.
- `user_content` is stripped and checked for emptiness before forwarding.
- `msg_type` defaults to `""` if absent, which falls through to the `unknown type` error handler cleanly.
- `msg_data` defaults to `{}` if absent, preventing `KeyError` on `.get()` calls.

---

### ✅ PASS — WebSocket Resource Cleanup

**File:** `backend/main.py`

The `try/finally` in `interview_ws()` guarantees `interview_service.close()` is called on both clean disconnect and unexpected exceptions. Each connection instantiates its own `GeminiInterviewService` — there is no singleton that could leak session state across users.

---

### ✅ PASS — No Hardcoded Secrets

Grep confirmed: no API keys, tokens, or passwords appear anywhere in the source tree. The only credential-adjacent string is `"your_gemini_api_key_here"` in `.env.example`.

---

### ✅ PASS — Error Messages Do Not Leak Stack Traces

**File:** `backend/main.py`

Exceptions are logged server-side (`logger.exception(...)`) but clients receive only the output of `_classify_gemini_error()`, which returns pre-written user-friendly strings. No raw exception messages or tracebacks are forwarded to the browser.

One exception: `RuntimeError` messages from `GeminiInterviewService` are forwarded verbatim (`await _send_error(websocket, str(exc))`). The only `RuntimeError` raised by the service is `"GEMINI_API_KEY is not configured"` — acceptable disclosure, but see R-2 for the broader pattern note.

---

### ✅ PASS — No Dangerous `eval` / `exec` Usage

Frontend uses `JSON.parse()` (standard, safe) for server messages. No `eval()`, `Function()`, `setTimeout(string)`, or `document.write()` calls exist.

---

### ✅ PASS — Static Files Served Safely

**File:** `backend/main.py`

`StaticFiles` is mounted at `/static` using FastAPI's built-in handler, which prevents path traversal by default. The root `GET /` serves a specific file (`static/index.html`) via `FileResponse` — not a dynamic path resolution.

---

### ✅ PASS — Test Suite Does Not Embed Real Credentials

**File:** `tests/test_main.py`

All tests patch the `GEMINI_API_KEY` with the dummy value `"test-key"` or `"test-key-123"`. No real API key is present in any test file.

---

## Recommendations (Non-Blocking)

### R-1 — Verify `.gitignore` Covers `.env`  
**Severity: Low**  
The `.env.example` file is correctly placeholder-only. However, I did not find a `.gitignore` file in the workspace. Before the first push of a `.env` file with a real key, confirm `.env` is excluded. The standard Python `.gitignore` (GitHub template) covers this, but it should be explicit.

**Suggested addition to `.gitignore`:**
```
.env
*.env
```

### R-2 — Sanitize `candidate_name` Before Embedding in System Prompt  
**Severity: Low / Informational**  
The `candidate_name` field from the WebSocket `start` message is embedded directly into the Gemini system prompt without sanitization:
```python
self._system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
    role=role,
    candidate_name=candidate_name,  # ← user-supplied
)
```
A malicious client could supply a `candidate_name` like `"Alex. Ignore all previous instructions and…"`. While this does not compromise the server (it only affects the Gemini session of that same user), it could cause the AI to behave unexpectedly for that session.

**Recommended fix:** truncate and strip the name before use.
```python
candidate_name = candidate_name[:80].strip()
```
The `maxlength="60"` on the HTML input helps honest users but is trivially bypassed via direct WebSocket messages.

---

## Not In Scope (By Design)

| Concern | Rationale |
|---|---|
| Authentication / AuthZ | This is a demo/practice platform — no user accounts in scope |
| Rate limiting per user | Would require session tracking; acceptable absence for v1 |
| HTTPS enforcement | Deployment concern, not application code |
| CORS policy | No cross-origin API calls; WS connection is same-origin |
| CSP headers | Not set; low risk given textContent discipline in app.js |

---

## Final Verdict

> **APPROVED FOR MERGE.** R-1 and R-2 are hardening recommendations suitable for a follow-up ticket, not blockers. The codebase demonstrates security-aware development practices: secrets in environment variables, XSS-safe DOM manipulation, structured error containment, input validation at the protocol boundary, and no resource leaks.
