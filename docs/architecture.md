# Interview Platform — Architecture Design

> **Source of truth:** Team Wiki → "Interview Platform Architecture"  
> This file is a workspace copy for developer reference.

---

## Stack
- **Backend:** Python 3.11+, FastAPI, `google-genai` SDK, `python-dotenv`
- **Frontend:** Vanilla HTML5 / CSS3 / ES2020 JS (no build step)
- **Transport:** WebSocket (`/ws/interview`) for real-time bidirectional communication
- **AI:** Google Gemini Live — `gemini-2.0-flash-live-001`

---

## Project Structure

```
interview_platform/
├── backend/
│   ├── main.py            # FastAPI app, routes, WebSocket handler
│   └── gemini_service.py  # Gemini SDK integration layer
├── static/
│   ├── index.html         # SPA shell
│   ├── style.css          # Styling
│   └── app.js             # WebSocket client + UI state machine
├── tests/
│   ├── __init__.py
│   └── test_main.py       # Pytest suite
├── requirements.txt
├── .env.example
└── README.md
```

---

## API Contracts

### HTTP

| Method | Path | Response |
|--------|------|----------|
| `GET` | `/` | `200 text/html` — serves `static/index.html` |
| `GET` | `/api/config/roles` | `200 {"roles": ["Software Engineer", "Data Scientist", "Product Manager", "System Design"]}` |
| `GET` | `/api/health` | `200 {"status": "ok", "gemini_configured": true\|false}` |

### WebSocket: `WS /ws/interview`

**Client → Server:**
```json
{"type": "start",   "data": {"role": "Software Engineer", "name": "Alex"}}
{"type": "message", "data": {"content": "I would use a hash map..."}}
{"type": "end",     "data": {}}
```

**Server → Client:**
```json
{"type": "greeting",  "data": {"content": "Hi Alex! ...", "done": true}}
{"type": "chunk",     "data": {"content": "partial text...", "done": false}}
{"type": "chunk",     "data": {"content": "", "done": true}}
{"type": "feedback",  "data": {"content": "## Feedback\n...", "done": true}}
{"type": "error",     "data": {"content": "human-readable msg", "code": "MISSING_API_KEY"}}
{"type": "status",    "data": {"content": "Thinking…"}}
```

**Error codes:** `MISSING_API_KEY` | `INVALID_API_KEY` | `RATE_LIMIT` | `SESSION_ERROR` | `INVALID_MESSAGE` | `INTERNAL_ERROR`

---

## Gemini Integration

### GeminiService Interface
```python
class GeminiService:
    def __init__(self, api_key: str | None)
    async def start_session(self, role: str, candidate_name: str) -> AsyncGenerator[str, None]
    async def send_message(self, content: str) -> AsyncGenerator[str, None]
    async def end_session(self) -> str          # returns full feedback markdown
    async def cleanup(self) -> None
    
    @property
    def is_configured(self) -> bool
    @property
    def is_active(self) -> bool
```

### Live Session Pattern
```python
async with client.aio.live.connect(
    model="gemini-2.0-flash-live-001",
    config=types.LiveConnectConfig(
        response_modalities=["TEXT"],
        system_instruction=build_system_prompt(role, candidate_name),
    )
) as session:
    await session.send(input=user_message, end_of_turn=True)
    async for response in session.receive():
        if response.text:
            yield response.text
```

### System Prompt Template
```
You are an expert technical interviewer conducting a {role} interview.
The candidate's name is {candidate_name}.

Your behaviour:
- Ask one focused question at a time; wait for the candidate's full answer.
- Start with a warm greeting, then ask your first interview question.
- Adapt difficulty based on the quality of previous answers.
- Provide brief encouraging acknowledgement before the next question.
- Do NOT reveal answers or give excessive hints.
- Keep responses under 120 words unless technical depth is required.
- On receiving "END_SESSION": produce structured Markdown feedback covering
  Overall Assessment, Technical Depth (/10), Communication (/10),
  Strengths, and Areas to Improve.

Role context for {role}:
- Software Engineer: algorithms, data structures, system design basics, OOP
- Data Scientist: ML concepts, statistics, Python/SQL, model evaluation
- Product Manager: product sense, metrics, prioritisation, user empathy
- System Design: scalability, distributed systems, trade-offs, CAP theorem
```

### Session Lifecycle
```
start_session(role, name)
  ├─ validate role ∈ SUPPORTED_ROLES
  ├─ build system_prompt
  ├─ open aio.live.connect() context
  ├─ send "BEGIN_INTERVIEW" trigger
  └─ stream greeting chunks → caller

send_message(content)
  ├─ send content to active session
  └─ yield response chunks → caller (done=True on final)

end_session()
  ├─ send "END_SESSION" sentinel
  ├─ collect full feedback text
  └─ close aio.live context

cleanup()          ← always called in WebSocket finally block
  └─ force-close any open session
```

---

## Error Handling

| Scenario | Code | Behaviour |
|---|---|---|
| API key not set | `MISSING_API_KEY` | Accept WS, send error on `start`, server stays up |
| Invalid/expired key | `INVALID_API_KEY` | Send error frame; do not retry |
| Rate limit (429) | `RATE_LIMIT` | Send error frame with retry suggestion |
| Gemini 5xx | `SESSION_ERROR` | Send error frame |
| Bad JSON from client | `INVALID_MESSAGE` | Send error frame; keep connection open |
| Unknown message type | `INVALID_MESSAGE` | Send error frame; keep connection open |
| WS disconnect | — | `finally: await gemini_service.cleanup()` |
| Unexpected exception | `INTERNAL_ERROR` | Send error frame + log stack trace |

---

## Frontend Notes

- **State machine:** `IDLE → CONNECTING → ACTIVE → ENDED`
- **⚠️ app.js expected > 200 lines** — use chunked writes with `append=True` after first chunk
- Verify structural completeness: `DOMContentLoaded` init block must be present at EOF

---

## Key Decisions

| Decision | Choice |
|---|---|
| Transport | WebSocket (bidirectional required) |
| Gemini SDK | `google-genai` + `client.aio.live.connect()` |
| Model | `gemini-2.0-flash-live-001` |
| Session scope | `GeminiService` instantiated per-connection (isolation) |
| Static serving | FastAPI `StaticFiles` mount; root → `index.html` |
| Frontend | Vanilla JS (no build step) |

---

*Full ADRs and rationale: see Wiki → "Interview Platform Architecture"*
