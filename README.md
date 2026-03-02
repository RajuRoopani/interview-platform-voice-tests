# AI Interview Platform

Practice real-world technical interviews with an AI interviewer powered by Google Gemini. Choose between **text-based** or **real-time voice** interviews to match your learning style. Get role-specific questions, instant feedback, and build genuine confidence before your big interview day.

---

## Table of Contents

1. [Tech Stack](#tech-stack)
2. [Features](#features)
3. [Prerequisites](#prerequisites)
4. [Setup Instructions](#setup-instructions)
5. [How to Use](#how-to-use)
6. [API Endpoints](#api-endpoints)
7. [WebSocket Message Format](#websocket-message-format)
8. [Running Tests](#running-tests)
9. [Project Structure](#project-structure)
10. [Architecture Overview](#architecture-overview)
11. [Screenshots & Demo](#screenshots--demo)

---

## Tech Stack

- **Backend:** Python 3.9+, FastAPI, Uvicorn
- **Frontend:** Vanilla HTML5, CSS3, ES2020 JavaScript (no build step required)
- **Real-Time Communication:** WebSocket (`/ws/interview` for text, `/ws/voice-interview` for voice)
- **AI Engine:** 
  - Text interviews: Google Gemini API (`google-genai` SDK, `gemini-2.0-flash` model)
  - Voice interviews: Google Gemini Live API (`gemini-2.0-flash-live-001` model for real-time audio streaming)
- **Audio (Voice mode):** Web Audio API for microphone capture and playback
- **Testing:** Pytest with `pytest-asyncio` for async tests
- **Environment Management:** `python-dotenv`

---

## Features

✨ **Real-Time AI-Powered Interviews** — Have a live conversation with an intelligent AI interviewer that adapts to your responses.

🎯 **Multiple Interview Roles** — Practice for different positions:
- Software Engineer (algorithms, data structures, system design)
- Data Scientist (ML concepts, statistics, model evaluation)
- Product Manager (product sense, metrics, prioritization)
- System Design (scalability, distributed systems, CAP theorem)

📡 **Streaming Responses** — Watch the AI's answer appear in real-time, just like a live interviewer.

💬 **Professional Chat Interface** — Clean, accessible chat UI with typing indicators and connection status.

🎓 **Instant Feedback** — After your interview ends, receive comprehensive feedback covering:
  - Key strengths demonstrated
  - Areas for improvement with specific advice
  - Overall assessment and suitability for the role
  - Actionable tips for future interviews

🔄 **Multiple Practice Sessions** — Conduct unlimited interviews to build confidence.

---

## Prerequisites

- **Python 3.9 or higher** (`python --version` to check)
- **Google Gemini API Key** (obtain from [Google AI Studio](https://aistudio.google.com/apikey))
- **pip** (usually included with Python; `pip --version` to check)

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/interview_platform.git
cd interview_platform
```

### 2. Create a Virtual Environment (Recommended)

```bash
# On macOS / Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

```bash
# Copy the example .env file
cp .env.example .env

# Open .env in your editor and add your Gemini API key
# GEMINI_API_KEY=your_actual_gemini_api_key_here
```

**Getting your API Key:**
1. Visit [Google AI Studio](https://aistudio.google.com/apikey)
2. Click **Create API Key**
3. Copy the generated key
4. Paste it into your `.env` file (keep it secret!)

### 5. Run the Application

```bash
uvicorn backend.main:app --reload
```

Output:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete
```

### 6. Open in Your Browser

Visit [http://localhost:8000](http://localhost:8000) in your web browser.

### 7. Browser Setup for Voice Mode (Optional)

If you plan to use voice interview mode:

**Chrome/Edge (Recommended)**
- Voice mode is fully supported
- The browser will request microphone permission when you start a voice interview
- Click **Allow** to grant permission

**Firefox**
- Voice mode requires `about:config` adjustment
- Set `dom.webkitBlobBuilder.enabled` to `true`
- Set `media.navigator.permission.disabled` to `true` (or allow permission on first use)

**Safari**
- Voice mode is supported on Safari 14.1+
- Requires HTTPS (will not work on `localhost` with `http://`)
- For local testing, use a self-signed certificate or disable HTTPS requirement in developer settings

**Other Browsers**
- Voice mode may not work if the browser doesn't support Web Audio API
- Fallback to text mode (always available)

---

## How to Use

### Text Interview

**Step-by-Step Interview Flow**

1. **Enter Your Name** (Optional)
   - Type your name in the "Your Name" field on the setup screen
   - If left blank, you'll be called "Candidate" by the AI

2. **Select an Interview Role** (Required)
   - Choose from:
     - **Software Engineer** — algorithms, design patterns, real-time coding
     - **Data Scientist** — ML, statistics, model evaluation, SQL
     - **Product Manager** — product strategy, metrics, prioritization
     - **System Design** — scalability, distributed systems, architecture trade-offs

3. **Click "Start Interview"**
   - The app connects to the Gemini API via WebSocket (`/ws/interview`)
   - Your AI interviewer will greet you warmly and introduce the session

4. **Chat with the AI Interviewer**
   - Type your answers naturally in the input box
   - Press **Enter** or click **Send** to submit
   - The AI will stream its response in real-time
   - Answer one question at a time; the AI will ask follow-ups based on your responses

5. **Conduct Your Interview**
   - The interview typically includes 3–5 questions per session (5–15 minutes)
   - Answer as you would in a real interview
   - The AI will challenge you progressively based on your performance
   - Ask clarifying questions if you need them

6. **End the Interview**
   - Click **End Interview** when finished
   - The AI will generate comprehensive feedback:
     - Strengths observed
     - Areas for improvement with advice
     - Overall assessment
     - Tips for your next interview

7. **Review Feedback & Practice Again**
   - Read the AI's feedback carefully
   - Use the insights to improve your answers
   - Click "Start Interview" to begin a new practice session

### Voice Interview (Full-Duplex Real-Time Conversations)

Voice mode allows you to practice with **full-duplex real-time audio conversations**, simulating an authentic phone interview experience where both you and the interviewer can speak simultaneously—just like a real phone call.

#### What is Full Duplex?

**Full duplex** means the microphone stays continuously open, and both you and the AI interviewer can speak at the same time. There's no push-to-talk button, no waiting for the AI to finish before you can speak, and no artificial turn-taking. The conversation flows naturally and dynamically, exactly like a real interview over the phone.

**How It Works**
- Once voice mode starts, your microphone is continuously open and monitoring for speech
- The AI interviewer's audio plays through your speakers in real-time
- You can speak naturally, interrupt the interviewer, or speak while they're speaking
- The conversation flows like a natural phone interview — dynamic and responsive
- Real-time audio streaming via Gemini Live API (`gemini-2.0-flash-live-001`)
- Low-latency bidirectional audio with ~100–500ms latency depending on network conditions

**To Use Voice Mode**
1. Ensure your browser supports Web Audio API and WebRTC (modern browser required)
2. Grant microphone and speaker permissions when prompted
3. On the setup screen, select **Voice Mode**
4. Choose your interview role and click "Start Interview"
5. Speak naturally as you would in a real phone interview — the mic is always listening
6. Interrupt the interviewer or ask clarifying questions anytime
7. Click "End Interview" when finished to receive feedback

#### Key Features of Full-Duplex Voice Mode

**🎤 Continuously Open Microphone**
- No push-to-talk button or button-holding required
- Microphone remains active throughout the interview
- The system automatically detects when you start and stop speaking
- Enables natural conversation flow without awkward pauses or artificial turn-taking

**🎯 Interruption Support**
- Interrupt the AI mid-response by simply speaking
- The interviewer will pause, listen to your interruption, and respond naturally
- Mirrors the dynamic nature of real phone interviews
- Encourages natural dialogue and follow-up questions without formal protocols

**🔊 Echo Cancellation & Audio Quality**
- Built-in browser echo cancellation prevents microphone feedback
- Automatic gain control ensures consistent volume levels
- Noise suppression filters out background noise for clarity
- Professional-grade audio preprocessing for high-quality interviews

**Browser Requirements for Voice Mode**
- **Modern browser with Web Audio API & WebRTC support:**
  - ✅ Chrome 74+ (recommended for best full-duplex support)
  - ✅ Firefox 66+
  - ✅ Edge 79+
  - ✅ Safari 14.1+
- **Working microphone and speakers/headphones** (built-in or external)
- **Stable internet connection** (full-duplex requires continuous WebSocket connection; ideally 5+ Mbps)
- **Microphone permission granted** (granted via browser prompt)

**Tips for Voice Mode**
- **Test beforehand:** Verify your microphone and speakers work using your browser's built-in voice features or a voice chat service (e.g., Google Meet)
- **Choose a quiet space:** Find a room with minimal background noise — close windows, turn off fans, ask others to be quiet
- **Speak clearly:** Use a normal, conversational pace — slightly slower than everyday speech but natural and professional
- **Listen carefully:** Pay close attention to the interviewer's questions and respond thoughtfully
- **Don't wait for pauses:** You can interrupt naturally without formal turn-taking — just speak when you have something to say
- **Use professional language:** Communicate as you would in a real interview, even though the conversation is natural and flowing
- **Stay engaged:** Maintain the energy and professionalism of a real phone interview
- **Ask for clarity:** If you didn't understand a question, ask the interviewer to repeat or rephrase it

**Example Voice Interview Flow**

```
1. You:  "Hi, I'm ready to start!"
2. AI:   "Great! Welcome to the interview. Let me start with a technical question. 
          Can you walk me through how you would design a load balancer?"
3. [AI is still explaining the question...]
4. You:  [You interrupt naturally] "Sorry, just to clarify—does this involve distributed systems?"
5. AI:   [Pauses and responds to your interruption] "Great question! Yes, it does involve distributed 
          systems. Here's why that's important..."
6. [You both continue naturally, back-and-forth, no button-pressing required]
```

---

## API Endpoints

### HTTP Endpoints

| Method | Path | Response | Purpose |
|--------|------|----------|---------|
| `GET` | `/` | `200 text/html` | Serves the SPA frontend (`index.html`) |
| `GET` | `/api/config/roles` | `200 application/json` | Returns list of available interview roles |
| `GET` | `/api/health` | `200 application/json` | Health check; indicates if Gemini API is configured |

#### Example Requests & Responses

**GET `/api/config/roles`**
```bash
curl http://localhost:8000/api/config/roles
```
Response:
```json
{
  "roles": [
    "Software Engineer",
    "Data Scientist",
    "Product Manager",
    "System Design"
  ]
}
```

**GET `/api/health`**
```bash
curl http://localhost:8000/api/health
```
Response:
```json
{
  "status": "ok",
  "gemini_configured": true
}
```

### WebSocket Endpoints

#### Text Mode (Current)
**URL:** `ws://localhost:8000/ws/interview` (or `wss://` if using HTTPS)

**Protocol:** JSON messages with `type` and `data` fields (see [WebSocket Message Format](#websocket-message-format) below)

#### Voice Mode (Full-Duplex)
**URL:** `ws://localhost:8000/ws/voice-interview` (or `wss://` if using HTTPS)

**Protocol:** Binary WebSocket frames containing:
- **Client → Server:** Audio chunks (PCM-encoded or WebM) from user's microphone
- **Server → Client:** Audio chunks (PCM-encoded or WebM) from AI interviewer's response

---

## WebSocket Message Format

### Text Mode Protocol (`/ws/interview`)

The text interview happens over a persistent WebSocket connection. Messages are exchanged as JSON objects with `type` and `data` fields.

### Client → Server Messages

#### `start` — Begin Interview
```json
{
  "type": "start",
  "data": {
    "role": "Software Engineer",
    "name": "Alice Chen"
  }
}
```

#### `message` — Send User Response
```json
{
  "type": "message",
  "data": {
    "content": "I would use a hash table to solve this problem in O(1) time..."
  }
}
```

#### `end` — End Interview & Request Feedback
```json
{
  "type": "end",
  "data": {}
}
```

---

### Server → Client Messages

#### `greeting` — Opening Question
Sent after the interview starts. Contains Alex's warm greeting and first question.
```json
{
  "type": "greeting",
  "data": {
    "content": "Hi Alice! Welcome! Today I'll be assessing your technical skills. Let's start with an easy one...",
    "done": true
  }
}
```

#### `chunk` — Streaming Response (Mid-Response)
Sent repeatedly as the AI generates its answer. Multiple chunks combine to form the complete response.
```json
{
  "type": "chunk",
  "data": {
    "content": "That's a great approach! You correctly ",
    "done": false
  }
}
```
```json
{
  "type": "chunk",
  "data": {
    "content": "identified the time complexity.",
    "done": false
  }
}
```
```json
{
  "type": "chunk",
  "data": {
    "content": " Now let me ask the follow-up...",
    "done": true
  }
}
```

#### `response` — Full Response (Complete)
Sent after all chunks are delivered. Contains the full AI message (optional; frontend can reconstruct from chunks).
```json
{
  "type": "response",
  "data": {
    "content": "That's a great approach! You correctly identified the time complexity. Now let me ask the follow-up...",
    "done": true
  }
}
```

#### `feedback` — Interview Complete with Feedback
Sent when the interview ends. Contains structured feedback markdown.
```json
{
  "type": "feedback",
  "data": {
    "content": "## Interview Feedback for Alice\n\n### Overall Assessment\nSolid technical fundamentals with clear communication...\n\n### Strengths\n- Clear explanation of algorithm approach\n- Good understanding of time/space trade-offs\n...\n\n### Areas for Improvement\n- Could have discussed edge cases\n...",
    "done": true
  }
}
```

#### `error` — Error Response
Sent if something goes wrong (missing API key, rate limit, etc.).
```json
{
  "type": "error",
  "data": {
    "message": "GEMINI_API_KEY is not configured. Please set it in your .env file.",
    "code": "MISSING_API_KEY"
  }
}
```

#### `status` — Status Update
Sent to update the client on server state (e.g., "Thinking...").
```json
{
  "type": "status",
  "data": {
    "message": "Thinking...",
    "status": "connected"
  }
}
```

---

### Voice Mode Protocol (`/ws/voice-interview`) — Full-Duplex Audio Streaming

The voice interview stream occurs over a WebSocket connection carrying **simultaneous bidirectional audio frames**, enabling full-duplex conversations where both user and AI can speak at the same time.

**Full-Duplex Audio Architecture**
- **Simultaneous audio streams:** User microphone → Server and Server → User speakers run concurrently
- **No turn-taking required:** Both parties can transmit and receive audio at the same time
- **Real-time bidirectional flow:** Audio frames flow in both directions without waiting for the other party to finish
- **Continuous monitoring:** Microphone remains open and actively listening throughout the session
- **Automatic speech detection:** The system detects user speech and AI speech without explicit signaling

**Audio Format**
- **Encoding:** PCM 16-bit mono or WebM format
- **Sample Rate:** 16 kHz (standard for speech recognition and synthesis)
- **Frame Duration:** 100–500ms chunks (variable depending on network latency)
- **Bidirectional:** Client and server send/receive audio simultaneously with minimal latency (~100-500ms total)

**Client ⇄ Server (Simultaneous Audio Exchange)**
```
User speaks → Microphone captures PCM frames → WebSocket sends to server
             (while simultaneously receiving...)
Server sends AI audio → Audio frames received → Speaker/headphones play back
```

Both streams are active at the same time, enabling natural, overlapping conversation just like a phone call.

**Binary WebSocket Frames**

**User Audio (Client → Server)**
```
Binary WebSocket frame containing PCM audio data from user's microphone
Frame structure: [audio_chunk_bytes]
Sent continuously while microphone is capturing speech
```

**AI Audio (Server → Client)**
```
Binary WebSocket frame containing synthesized audio from Gemini Live API
Frame structure: [audio_chunk_bytes]
Sent in real-time as the AI speaks, without waiting for user to finish
```

**Session Control (JSON over WebSocket)**

**Start Voice Interview**
```json
{
  "type": "start",
  "data": {
    "role": "Software Engineer",
    "candidate_name": "Alice Chen",
    "mode": "voice"
  }
}
```
Response: Server opens full-duplex audio channel and sends `session_ready` message.

**Session Ready (Server → Client)**
```json
{
  "type": "session_ready",
  "data": {
    "mode": "voice",
    "output_sample_rate": 24000
  }
}
```
Sent immediately after voice session is established. Browser should start mic capture after this message.

**End Voice Interview**
```json
{
  "type": "end",
  "data": {}
}
```
Response: Server closes audio streams and sends interview feedback (same format as text mode).

**Server → Client Control Messages**

**AI Turn Complete**
```json
{
  "type": "ai_turn_complete",
  "data": {}
}
```
Sent when the AI finishes speaking and is waiting for user input. Browser can use this to update UI state.

**AI Interrupted (Barge-In)**
```json
{
  "type": "ai_interrupted",
  "data": {}
}
```
Sent when the user interrupts the AI mid-response. Browser must immediately flush its audio playback queue.

**Transcripts**
```json
{
  "type": "transcript",
  "data": {
    "content": "That's a great answer...",
    "speaker": "ai",
    "final": true
  }
}
```
Optional transcript messages showing what the AI heard or said. `speaker` is either `"ai"` or `"user"`.

---

**Key Full-Duplex Behaviors**
1. **Continuous listening:** User microphone streams continuously from session start to end
2. **Simultaneous bidirectional audio:** Both send and receive happen at the same time via `asyncio.gather()` on the server
3. **Immediate playback:** Server streams AI audio as it arrives; no wait-for-turn gate
4. **Barge-in support:** User speech interrupts AI response via Gemini Live's built-in barge-in detection; server sends `ai_interrupted` signal
5. **Echo cancellation:** Browser-native echo cancellation (via Web Audio API `echoCancellation: true`) prevents mic feedback—no need for server-side mic muting
6. **No turn-signaling:** Gemini Live's voice activity detection (VAD) handles when to respond; no explicit `voice_start_speaking` / `voice_stop_speaking` messages needed

---

## Running Tests

The project includes a comprehensive test suite using Pytest.

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Test File

```bash
pytest tests/test_main.py -v
```

### Run Tests with Coverage

```bash
pytest tests/ --cov=backend --cov-report=term-missing
```

### Expected Output Example

```
tests/test_main.py::test_root_returns_html PASSED                      [25%]
tests/test_main.py::test_api_config_roles PASSED                       [25%]
tests/test_main.py::test_api_health_ok PASSED                          [25%]
tests/test_main.py::test_api_health_missing_key PASSED                 [25%]

======================== 4 passed in 0.32s ========================
```

---

## Project Structure

```
interview_platform/
├── backend/
│   ├── __init__.py                # Package marker
│   ├── main.py                    # FastAPI app, routes, WebSocket handler
│   └── gemini_service.py          # Gemini API integration layer
├── static/
│   ├── index.html                 # HTML shell (single-page app)
│   ├── style.css                  # Styling with design tokens
│   └── app.js                     # Frontend logic, WebSocket client, UI state
├── tests/
│   ├── __init__.py
│   └── test_main.py               # Pytest suite for backend
├── docs/
│   └── architecture.md            # Detailed architecture & API contracts
├── requirements.txt               # Python dependencies
├── .env.example                   # Template for environment variables
├── .env                           # Your actual API keys (gitignored)
└── README.md                      # This file
```

---

## Architecture Overview

### High-Level Flow

```
Browser (Vanilla JS)
    ↓ WebSocket
FastAPI Backend
    ↓ HTTP
Google Gemini API
    ↓ Response Stream
FastAPI Backend
    ↓ WebSocket Chunks
Browser (Vanilla JS)
```

### Key Components

**`backend/main.py`**
- FastAPI application entry point
- Serves static files (SPA) from `/static` directory
- HTTP Routes:
  - `GET /` — Serves `index.html`
  - `GET /api/config/roles` — Returns available interview roles
  - `GET /api/health` — Health check endpoint
- WebSocket Endpoints:
  - `WS /ws/interview` — Text-based interview handler
  - `WS /ws/voice-interview` — Voice-based interview handler with full-duplex concurrent task model
    - Runs two independent `asyncio` tasks simultaneously using `asyncio.gather()`:
      - **Task A:** Receive binary audio from browser microphone → relay to Gemini Live
      - **Task B:** Receive audio/transcripts from Gemini Live → stream back to browser
    - Both tasks run concurrently, enabling true full-duplex bidirectional audio

**`backend/gemini_service.py`**
- `GeminiInterviewService` class — encapsulates a single text interview session
- `GeminiVoiceService` class — encapsulates a single full-duplex voice interview session via Gemini Live API
- Text Mode:
  - Manages conversation history for context awareness
  - Handles streaming responses from Gemini API
  - Methods: `start_session()`, `send_message()`, `end_session()`, `close()`
- Voice Mode:
  - Real-time bidirectional audio streaming via Gemini Live API (`gemini-2.0-flash-live-001`)
  - Built-in voice activity detection (VAD) and barge-in support
  - Methods: `connect()`, `send_audio()`, `receive()` (async generator), `close()`

**`static/index.html`**
- Single-page app shell with three screens:
  - **Setup Screen** — name input, role selector, interview mode selector (text or voice), start button
  - **Text Chat Screen** — message list, input bar, end button (for text interviews)
  - **Voice Controls Screen** — microphone status, recording indicator, playback control, end button (for voice interviews)

**`static/app.js`**
- WebSocket client for text interviews
- Voice audio capture and playback handlers
- Event handlers for user interactions
- UI state management (connecting, connected, disconnected, recording, playing)
- Message rendering (text: bubbles; voice: waveform indicators)

**`static/style.css`**
- Professional design tokens (colors, spacing, typography)
- Responsive layout with flexbox
- Dark header, light content area
- Smooth animations for transitions

---

## Configuration

### Environment Variables

**`.env.example`** template:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

**Required:**
- `GEMINI_API_KEY` — Your API key from [Google AI Studio](https://aistudio.google.com/apikey)

**Optional:**
- `PYTHONENV` — Set to `production` for production deployments (defaults to development)

### Supported Interview Roles

```python
ROLES = [
    "Software Engineer",
    "Data Scientist",
    "Product Manager",
    "System Design",
]
```

Each role triggers a different system prompt tuned to that position's typical interview questions and skills.

---

## Troubleshooting

### Text Interview Issues

#### Issue: `GEMINI_API_KEY is not configured`

**Cause:** The `.env` file doesn't exist or the key wasn't added.

**Fix:**
```bash
cp .env.example .env
# Edit .env and add your actual API key
```

#### Issue: WebSocket connection fails or drops

**Cause:** Network issue, server crash, or API rate limit.

**Fix:**
1. Check browser console (F12) for error messages
2. Verify your API key is valid at [Google AI Studio](https://aistudio.google.com/apikey)
3. Restart the server: `uvicorn backend.main:app --reload`
4. Try a fresh interview session

#### Issue: Responses are very slow or don't appear

**Cause:** Network latency, Gemini API rate limiting, or server overload.

**Fix:**
1. Check your internet connection
2. Try again in a few minutes (if rate limited)
3. Check the browser console for network errors
4. Verify `GEMINI_API_KEY` is valid

#### Issue: Tests fail with import errors

**Cause:** Dependencies not installed.

**Fix:**
```bash
pip install -r requirements.txt
pytest tests/ -v
```

### Voice Interview Issues

#### Issue: "Microphone permission denied"

**Cause:** The browser requires explicit permission to access your microphone, and you clicked "Deny" or your device doesn't have a mic.

**Fix:**
1. Check that your device has a working microphone
2. In your browser settings, allow the site permission to access the microphone:
   - **Chrome/Edge:** Settings → Privacy → Site Permissions → Microphone → Allow [your-site.com]
   - **Firefox:** Preferences → Privacy & Security → Permissions → Microphone → Allow this site
   - **Safari:** System Preferences → Security & Privacy → Microphone → Allow your-browser
3. Reload the page and try again

#### Issue: Microphone works but no audio output

**Cause:** Audio output is muted or speakers/headphones aren't connected.

**Fix:**
1. Check that your speakers or headphones are connected
2. Verify system volume is not muted
3. Check browser volume (some browsers have independent volume controls)
4. Test audio output on another website (e.g., YouTube)

#### Issue: Voice interview cuts out or has poor quality

**Cause:** Unstable network connection or high latency.

**Fix:**
1. Check your internet connection speed (use speedtest.net)
2. Move closer to your router or use a wired connection
3. Close other bandwidth-heavy applications (downloads, streaming)
4. Try again during a less congested time of day

#### Issue: The AI doesn't respond to my voice

**Cause:** Microphone isn't capturing audio, or background noise is too loud.

**Fix:**
1. Test microphone on another site (e.g., voice call service)
2. Reduce background noise (close windows, turn off fans, find a quiet space)
3. Speak clearly and at a normal volume
4. Check that the browser's audio recording indicator shows recording is active

#### Issue: Voice mode not available (greyed out button)

**Cause:** Your browser doesn't support Web Audio API, or the microphone isn't working properly.

**Fix:**
1. Use a modern browser (Chrome, Edge, Firefox 55+, Safari 14.1+)
2. Check that you're on the latest version
3. Ensure your browser is up to date and try refreshing the page

---

## Running in Production

### Using Gunicorn

```bash
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker backend.main:app --bind 0.0.0.0:8000
```

### Using Docker

Create a `Dockerfile`:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t interview-platform .
docker run -e GEMINI_API_KEY=your_key -p 8000:8000 interview-platform
```

---

## Screenshots & Demo

### Setup Screen
The initial landing page where users enter their name and select an interview role:
- Professional header with logo
- Hero section with value proposition
- Name input (optional)
- Role dropdown selector
- Large "Start Interview" button
- Helpful tips strip

### Chat Screen
The interview conversation interface:
- Header showing current role and connection status
- Scrollable chat history with user and AI messages
- Typing indicator while AI thinks
- Input textarea with auto-expand
- Character counter
- Send button (disabled while AI responds)
- End Interview button for early termination
- Timestamp on each message

### Connection Status Indicator
- **Connected** (green) — Ready to chat
- **Disconnected** (red) — Connection lost
- **Reconnecting** (orange) — Attempting to reconnect

### Message Bubbles
- **User messages** — Blue bubbles on the right, white text
- **AI messages** — White bubbles on the left, dark text, avatar emoji
- **Typing indicator** — Three animated dots while AI formulates response
- **Feedback** — Special styled message at end of interview

[Screenshot placeholders below — add actual UI screenshots here]

```
┌─────────────────────────────────────────────┐
│  🤖  AI Interview Platform         Status  │
├─────────────────────────────────────────────┤
│                                             │
│  [AI Avatar] Hi! Welcome to the interview! │
│              Let's start...                 │
│                                             │
│                          [You] Great! I'm  │
│                                ready.      │
│                                             │
│  [AI Avatar] 🔄 🔄 🔄 (thinking...)        │
│                                             │
├─────────────────────────────────────────────┤
│  [Input: Type your answer...]  [Send]      │
│                            [End Interview] │
└─────────────────────────────────────────────┘
```

---

## Performance Notes

- **First message latency:** 1–3 seconds (API connection + model load)
- **Streaming chunks:** ~100–300ms per chunk (depends on network & Gemini response time)
- **Session duration:** 5–15 minutes typical; can be shorter or longer
- **Concurrent users:** Designed for single-user interviews; each connection gets its own service instance

---

## Security

⚠️ **Never commit `.env` with your actual API key!**

- `.env` is listed in `.gitignore`
- Always use `.env.example` as a template
- Rotate your API key if it's accidentally committed

### Best Practices

1. Keep your `GEMINI_API_KEY` secret
2. Use environment variables for all secrets (never hardcode)
3. In production, use a secrets manager (AWS Secrets Manager, Google Secret Manager, etc.)
4. Enable API key restrictions in Google AI Studio (restrict to specific IPs or services if possible)

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Add or update tests as needed
5. Run tests: `pytest tests/ -v`
6. Commit: `git commit -m "feat: describe your change"`
7. Push: `git push origin feature/my-feature`
8. Open a Pull Request

---

## License

This project is released under the MIT License. See LICENSE file for details.

---

## Support & Feedback

- **Report bugs** — Open an issue on GitHub
- **Feature requests** — Discuss in Discussions or Issues
- **Questions?** — Check the [Architecture](docs/architecture.md) document for technical details

---

## Acknowledgments

- **Google Gemini API** — Powering the intelligent interviewer
- **FastAPI** — Modern, fast web framework for Python
- **Vanilla JavaScript** — Pure, zero-dependency frontend

---

**Happy interviewing! 🚀**

Good luck with your real interviews!
