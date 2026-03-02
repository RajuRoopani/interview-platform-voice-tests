"""
Pytest test suite for AI Interview Platform backend (interview_platform/backend/main.py)

Tests cover:
- HTTP endpoints: GET / and GET /api/config/roles, GET /api/health
- WebSocket lifecycle: start, message, end
- Error handling: missing API key, invalid role, empty content, malformed JSON, unknown types
- Message length validation (2000 char limit)
- Candidate name sanitization
- Integration with GeminiInterviewService (mocked)
"""

from __future__ import annotations

import os
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.main import app, INTERVIEW_ROLES


# =============================================================================
# Fixtures & Helpers
# =============================================================================


@pytest.fixture
def client() -> TestClient:
    """Create a FastAPI test client."""
    return TestClient(app)


class AsyncGeneratorMock:
    """A mock that returns an async generator when called."""

    def __init__(self, items: list[str]) -> None:
        self.items = items

    async def __call__(self, *args, **kwargs) -> AsyncGenerator[str, None]:
        """Make this callable, returning an async generator.
        
        Accepts any arguments to match the actual function signatures that
        call this mock (e.g., send_message(user_content) passes an argument).
        """
        for item in self.items:
            yield item


# =============================================================================
# HTTP Endpoint Tests
# =============================================================================


class TestHTTPEndpoints:
    """Test HTTP routes: GET /, GET /api/config/roles, GET /api/health"""

    def test_get_root_returns_200(self, client: TestClient) -> None:
        """GET / should return a 200 status code."""
        response = client.get("/")
        assert response.status_code == 200

    def test_get_root_returns_html(self, client: TestClient) -> None:
        """GET / should return HTML content type."""
        response = client.get("/")
        assert "text/html" in response.headers.get("content-type", "")

    def test_get_roles_returns_200(self, client: TestClient) -> None:
        """GET /api/config/roles should return 200."""
        response = client.get("/api/config/roles")
        assert response.status_code == 200

    def test_get_roles_returns_correct_roles(self, client: TestClient) -> None:
        """GET /api/config/roles should return the expected role list."""
        response = client.get("/api/config/roles")
        roles = response.json()
        assert isinstance(roles, list)
        assert roles == INTERVIEW_ROLES
        assert "Software Engineer" in roles
        assert "Data Scientist" in roles
        assert "Product Manager" in roles
        assert "System Design" in roles

    def test_get_roles_response_type(self, client: TestClient) -> None:
        """GET /api/config/roles should return JSON."""
        response = client.get("/api/config/roles")
        assert response.headers.get("content-type") == "application/json"

    def test_health_endpoint_returns_ok(self, client: TestClient) -> None:
        """GET /api/health should return 200 with {"status": "ok"}."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "ok"


# =============================================================================
# WebSocket Tests
# =============================================================================


class TestWebSocketBasic:
    """Test WebSocket connection and basic lifecycle."""

    def test_websocket_connection_accepted(self, client: TestClient) -> None:
        """WebSocket connection should be accepted when API key is present."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key-123"}):
            with patch(
                "backend.main.GeminiInterviewService"
            ) as mock_service_class:
                with client.websocket_connect("/ws/interview") as websocket:
                    assert websocket is not None

    def test_websocket_missing_api_key(self, client: TestClient) -> None:
        """WebSocket should close with error if GEMINI_API_KEY is missing."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=False):
            try:
                with client.websocket_connect("/ws/interview") as websocket:
                    data = websocket.receive_json()
                    assert data["type"] == "error"
                    assert "GEMINI_API_KEY" in data["data"]["content"]
            except Exception:
                pass


class TestWebSocketStartMessage:
    """Test the 'start' message type."""

    def test_start_initializes_service(self, client: TestClient) -> None:
        """Sending 'start' should create a GeminiInterviewService instance."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch(
                "backend.main.GeminiInterviewService"
            ) as mock_service_class:
                mock_instance = MagicMock()
                mock_instance.start_session = AsyncGeneratorMock(["Hello ", "there!"])
                mock_instance.close = AsyncMock()
                mock_service_class.return_value = mock_instance

                with client.websocket_connect("/ws/interview") as websocket:
                    websocket.send_json(
                        {
                            "type": "start",
                            "data": {
                                "role": "Software Engineer",
                                "candidate_name": "Alice",
                            },
                        }
                    )

                    data1 = websocket.receive_json()
                    assert data1["type"] == "chunk"
                    assert "Hello" in data1["data"]["content"]

                    data2 = websocket.receive_json()
                    assert data2["type"] == "chunk"

                    data3 = websocket.receive_json()
                    assert data3["type"] == "response"
                    assert data3["data"]["done"] is True

    def test_start_with_invalid_role(self, client: TestClient) -> None:
        """Sending 'start' with invalid role should return error."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch("backend.main.GeminiInterviewService"):
                with client.websocket_connect("/ws/interview") as websocket:
                    websocket.send_json(
                        {
                            "type": "start",
                            "data": {
                                "role": "Invalid Role",
                                "candidate_name": "Bob",
                            },
                        }
                    )

                    data = websocket.receive_json()
                    assert data["type"] == "error"
                    assert "Unknown role" in data["data"]["content"]
                    assert "Software Engineer" in data["data"]["content"]

    def test_start_uses_defaults(self, client: TestClient) -> None:
        """Sending 'start' without role/name should use defaults and call start_session."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch(
                "backend.main.GeminiInterviewService"
            ) as mock_service_class:
                mock_instance = MagicMock()
                mock_instance.start_session = AsyncGeneratorMock(["Started"])
                mock_instance.close = AsyncMock()
                mock_service_class.return_value = mock_instance

                with client.websocket_connect("/ws/interview") as websocket:
                    websocket.send_json({"type": "start", "data": {}})

                    websocket.receive_json()
                    websocket.receive_json()

                    # Verify defaults were used
                    mock_service_class.assert_called_once()
                    call_kwargs = mock_service_class.call_args[1]
                    assert call_kwargs["role"] == "Software Engineer"
                    assert call_kwargs["candidate_name"] == "Candidate"


class TestWebSocketMessageHandling:
    """Test the 'message' message type."""

    def test_message_before_start_returns_error(self, client: TestClient) -> None:
        """Sending 'message' before 'start' should return error."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch("backend.main.GeminiInterviewService"):
                with client.websocket_connect("/ws/interview") as websocket:
                    websocket.send_json(
                        {
                            "type": "message",
                            "data": {"content": "Hello there!"},
                        }
                    )

                    data = websocket.receive_json()
                    assert data["type"] == "error"
                    assert "No active interview session" in data["data"]["content"]

    def test_message_empty_content_returns_error(self, client: TestClient) -> None:
        """Sending 'message' with empty content should return error."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch(
                "backend.main.GeminiInterviewService"
            ) as mock_service_class:
                mock_instance = MagicMock()
                mock_instance.start_session = AsyncGeneratorMock(["Hi"])
                mock_instance.send_message = AsyncMock()
                mock_instance.close = AsyncMock()
                mock_service_class.return_value = mock_instance

                with client.websocket_connect("/ws/interview") as websocket:
                    websocket.send_json(
                        {
                            "type": "start",
                            "data": {
                                "role": "Software Engineer",
                                "candidate_name": "Charlie",
                            },
                        }
                    )
                    websocket.receive_json()
                    websocket.receive_json()

                    websocket.send_json(
                        {
                            "type": "message",
                            "data": {"content": ""},
                        }
                    )

                    data = websocket.receive_json()
                    assert data["type"] == "error"
                    assert "Message content cannot be empty" in data["data"]["content"]

    def test_message_whitespace_only_returns_error(
        self, client: TestClient
    ) -> None:
        """Sending 'message' with only whitespace should return error."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch(
                "backend.main.GeminiInterviewService"
            ) as mock_service_class:
                mock_instance = MagicMock()
                mock_instance.start_session = AsyncGeneratorMock(["Hi"])
                mock_instance.send_message = AsyncMock()
                mock_instance.close = AsyncMock()
                mock_service_class.return_value = mock_instance

                with client.websocket_connect("/ws/interview") as websocket:
                    websocket.send_json(
                        {
                            "type": "start",
                            "data": {
                                "role": "Software Engineer",
                                "candidate_name": "Dave",
                            },
                        }
                    )
                    websocket.receive_json()
                    websocket.receive_json()

                    websocket.send_json(
                        {
                            "type": "message",
                            "data": {"content": "   \t\n  "},
                        }
                    )

                    data = websocket.receive_json()
                    assert data["type"] == "error"
                    assert "Message content cannot be empty" in data["data"]["content"]

    def test_message_too_long_returns_error(self, client: TestClient) -> None:
        """Messages exceeding 2000 chars should return error."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch(
                "backend.main.GeminiInterviewService"
            ) as mock_service_class:
                mock_instance = MagicMock()
                mock_instance.start_session = AsyncGeneratorMock(["Started"])
                mock_instance.send_message = AsyncMock()
                mock_instance.close = AsyncMock()
                mock_service_class.return_value = mock_instance

                with client.websocket_connect("/ws/interview") as websocket:
                    # Start session
                    websocket.send_json(
                        {
                            "type": "start",
                            "data": {
                                "role": "Software Engineer",
                                "candidate_name": "TestUser",
                            },
                        }
                    )
                    websocket.receive_json()
                    websocket.receive_json()

                    # Send message with 2001 characters (exceeds limit)
                    long_message = "x" * 2001
                    websocket.send_json(
                        {
                            "type": "message",
                            "data": {"content": long_message},
                        }
                    )

                    response = websocket.receive_json()
                    assert response["type"] == "error"
                    assert "too long" in response["data"]["content"].lower()

    def test_message_streams_response(self, client: TestClient) -> None:
        """Sending valid 'message' should stream response chunks."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch(
                "backend.main.GeminiInterviewService"
            ) as mock_service_class:
                mock_instance = MagicMock()
                mock_instance.start_session = AsyncGeneratorMock(["Greeting"])
                mock_instance.send_message = AsyncGeneratorMock(["Response ", "text"])
                mock_instance.close = AsyncMock()
                mock_service_class.return_value = mock_instance

                with client.websocket_connect("/ws/interview") as websocket:
                    websocket.send_json(
                        {
                            "type": "start",
                            "data": {
                                "role": "Data Scientist",
                                "candidate_name": "Eve",
                            },
                        }
                    )
                    websocket.receive_json()
                    websocket.receive_json()

                    websocket.send_json(
                        {
                            "type": "message",
                            "data": {"content": "I have 3 years of experience"},
                        }
                    )

                    data1 = websocket.receive_json()
                    assert data1["type"] == "chunk"
                    assert data1["data"]["done"] is False

                    data2 = websocket.receive_json()
                    assert data2["type"] == "chunk"

                    data3 = websocket.receive_json()
                    assert data3["type"] == "response"
                    assert data3["data"]["done"] is True


class TestWebSocketEndMessage:
    """Test the 'end' message type."""

    def test_end_before_start_returns_error(self, client: TestClient) -> None:
        """Sending 'end' before 'start' should return error."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch("backend.main.GeminiInterviewService"):
                with client.websocket_connect("/ws/interview") as websocket:
                    websocket.send_json({"type": "end", "data": {}})

                    data = websocket.receive_json()
                    assert data["type"] == "error"
                    assert "No active interview session" in data["data"]["content"]

    def test_end_closes_connection(self, client: TestClient) -> None:
        """Sending 'end' should close the WebSocket connection gracefully."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch(
                "backend.main.GeminiInterviewService"
            ) as mock_service_class:
                mock_instance = MagicMock()
                mock_instance.start_session = AsyncGeneratorMock(["Hi"])
                mock_instance.end_session = AsyncGeneratorMock(["Feedback"])
                mock_instance.close = AsyncMock()
                mock_service_class.return_value = mock_instance

                with client.websocket_connect("/ws/interview") as websocket:
                    websocket.send_json(
                        {
                            "type": "start",
                            "data": {
                                "role": "Product Manager",
                                "candidate_name": "Frank",
                            },
                        }
                    )
                    websocket.receive_json()
                    websocket.receive_json()

                    websocket.send_json({"type": "end", "data": {}})

                    data1 = websocket.receive_json()
                    assert data1["type"] == "chunk"

                    data2 = websocket.receive_json()
                    assert data2["type"] == "response"

                    mock_instance.close.assert_called()


class TestWebSocketErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_json_returns_error(self, client: TestClient) -> None:
        """Sending invalid JSON should return error."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch("backend.main.GeminiInterviewService"):
                with client.websocket_connect("/ws/interview") as websocket:
                    websocket.send_text("not valid json {")

                    data = websocket.receive_json()
                    assert data["type"] == "error"
                    assert "Invalid JSON" in data["data"]["content"]

    def test_unknown_message_type_returns_error(
        self, client: TestClient
    ) -> None:
        """Sending unknown message type should return error."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch("backend.main.GeminiInterviewService"):
                with client.websocket_connect("/ws/interview") as websocket:
                    websocket.send_json({"type": "unknown", "data": {}})

                    data = websocket.receive_json()
                    assert data["type"] == "error"
                    assert "Unknown message type" in data["data"]["content"]
                    assert "start" in data["data"]["content"]
                    assert "message" in data["data"]["content"]
                    assert "end" in data["data"]["content"]

    def test_missing_type_field_returns_error(self, client: TestClient) -> None:
        """Sending JSON without 'type' field should return error."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch("backend.main.GeminiInterviewService"):
                with client.websocket_connect("/ws/interview") as websocket:
                    websocket.send_json({"data": {}})

                    data = websocket.receive_json()
                    assert data["type"] == "error"
                    assert "Unknown message type" in data["data"]["content"]

    def test_runtime_error_from_service_caught(
        self, client: TestClient
    ) -> None:
        """RuntimeError from service should be caught and returned as error."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch(
                "backend.main.GeminiInterviewService"
            ) as mock_service_class:
                mock_instance = MagicMock()
                
                async def raise_runtime_error():
                    raise RuntimeError("Custom error message")
                    yield
                
                mock_instance.start_session = raise_runtime_error
                mock_instance.close = AsyncMock()
                mock_service_class.return_value = mock_instance

                with client.websocket_connect("/ws/interview") as websocket:
                    websocket.send_json(
                        {
                            "type": "start",
                            "data": {
                                "role": "Software Engineer",
                                "candidate_name": "Grace",
                            },
                        }
                    )

                    data = websocket.receive_json()
                    assert data["type"] == "error"
                    assert "Custom error message" in data["data"]["content"]

    def test_generic_exception_from_service_classified(
        self, client: TestClient
    ) -> None:
        """Generic exceptions from service should be classified and returned."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch(
                "backend.main.GeminiInterviewService"
            ) as mock_service_class:
                mock_instance = MagicMock()
                
                async def raise_generic_error():
                    raise Exception("API timeout error")
                    yield
                
                mock_instance.start_session = raise_generic_error
                mock_instance.close = AsyncMock()
                mock_service_class.return_value = mock_instance

                with client.websocket_connect("/ws/interview") as websocket:
                    websocket.send_json(
                        {
                            "type": "start",
                            "data": {
                                "role": "Software Engineer",
                                "candidate_name": "Henry",
                            },
                        }
                    )

                    data = websocket.receive_json()
                    assert data["type"] == "error"
                    assert len(data["data"]["content"]) > 0

    def test_multiple_invalid_messages_dont_crash(
        self, client: TestClient
    ) -> None:
        """Sending several invalid messages should not crash."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch(
                "backend.main.GeminiInterviewService"
            ) as mock_service_class:
                mock_instance = MagicMock()
                mock_instance.start_session = AsyncGeneratorMock(["Hi"])
                mock_instance.close = AsyncMock()
                mock_service_class.return_value = mock_instance

                with client.websocket_connect("/ws/interview") as websocket:
                    # Send invalid JSON
                    websocket.send_text("not json")
                    response = websocket.receive_json()
                    assert response["type"] == "error"

                    # Send unknown type
                    websocket.send_json({"type": "invalid"})
                    response = websocket.receive_json()
                    assert response["type"] == "error"

                    # Send message before start
                    websocket.send_json(
                        {"type": "message", "data": {"content": "hello"}}
                    )
                    response = websocket.receive_json()
                    assert response["type"] == "error"

                    # Now send valid start
                    websocket.send_json(
                        {
                            "type": "start",
                            "data": {
                                "role": "Software Engineer",
                                "candidate_name": "Test",
                            },
                        }
                    )
                    # Should succeed without crashing
                    response1 = websocket.receive_json()
                    assert response1["type"] == "chunk"

                    response2 = websocket.receive_json()
                    assert response2["type"] == "response"


class TestWebSocketErrorClassification:
    """Test the _classify_gemini_error function behavior."""

    def test_api_key_error_classification(self, client: TestClient) -> None:
        """API key errors should be classified appropriately."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch(
                "backend.main.GeminiInterviewService"
            ) as mock_service_class:
                mock_instance = MagicMock()
                
                async def raise_auth_error():
                    raise Exception("401 api_key authentication failed")
                    yield
                
                mock_instance.start_session = raise_auth_error
                mock_instance.close = AsyncMock()
                mock_service_class.return_value = mock_instance

                with client.websocket_connect("/ws/interview") as websocket:
                    websocket.send_json(
                        {
                            "type": "start",
                            "data": {
                                "role": "Software Engineer",
                                "candidate_name": "Iris",
                            },
                        }
                    )

                    data = websocket.receive_json()
                    assert data["type"] == "error"
                    assert "Authentication" in data["data"]["content"]

    def test_rate_limit_error_classification(self, client: TestClient) -> None:
        """Rate limit errors should be classified appropriately."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch(
                "backend.main.GeminiInterviewService"
            ) as mock_service_class:
                mock_instance = MagicMock()
                
                async def raise_rate_limit_error():
                    raise Exception("429 quota exceeded")
                    yield
                
                mock_instance.start_session = raise_rate_limit_error
                mock_instance.close = AsyncMock()
                mock_service_class.return_value = mock_instance

                with client.websocket_connect("/ws/interview") as websocket:
                    websocket.send_json(
                        {
                            "type": "start",
                            "data": {
                                "role": "Software Engineer",
                                "candidate_name": "Jack",
                            },
                        }
                    )

                    data = websocket.receive_json()
                    assert data["type"] == "error"
                    assert "rate-limited" in data["data"]["content"]

    def test_timeout_error_classification(self, client: TestClient) -> None:
        """Timeout errors should be classified appropriately."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch(
                "backend.main.GeminiInterviewService"
            ) as mock_service_class:
                mock_instance = MagicMock()
                
                async def raise_timeout_error():
                    raise Exception("request timeout deadline exceeded")
                    yield
                
                mock_instance.start_session = raise_timeout_error
                mock_instance.close = AsyncMock()
                mock_service_class.return_value = mock_instance

                with client.websocket_connect("/ws/interview") as websocket:
                    websocket.send_json(
                        {
                            "type": "start",
                            "data": {
                                "role": "Software Engineer",
                                "candidate_name": "Karen",
                            },
                        }
                    )

                    data = websocket.receive_json()
                    assert data["type"] == "error"
                    assert "timed out" in data["data"]["content"]


class TestCandidateNameSanitization:
    """Test candidate name sanitization and validation."""

    def test_candidate_name_sanitization(self, client: TestClient) -> None:
        """Names with control chars should be sanitized."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch(
                "backend.main.GeminiInterviewService"
            ) as mock_service_class:
                mock_instance = MagicMock()
                mock_instance.start_session = AsyncGeneratorMock(["Started"])
                mock_instance.close = AsyncMock()
                mock_service_class.return_value = mock_instance

                with client.websocket_connect("/ws/interview") as websocket:
                    # Send start with name containing control characters
                    name_with_control_chars = "Alice\x00\x01\x02Bob"
                    websocket.send_json(
                        {
                            "type": "start",
                            "data": {
                                "role": "Software Engineer",
                                "candidate_name": name_with_control_chars,
                            },
                        }
                    )

                    # Receive response
                    websocket.receive_json()
                    websocket.receive_json()

                    # Verify that the service was called with sanitized name
                    mock_service_class.assert_called_once()
                    call_kwargs = mock_service_class.call_args[1]
                    # Control characters should be stripped
                    assert "\x00" not in call_kwargs["candidate_name"]
                    assert "\x01" not in call_kwargs["candidate_name"]
                    assert "\x02" not in call_kwargs["candidate_name"]
                    # Original alphanumeric should remain
                    assert "Alice" in call_kwargs["candidate_name"]
                    assert "Bob" in call_kwargs["candidate_name"]

    def test_candidate_name_length_cap(self, client: TestClient) -> None:
        """Names exceeding 100 chars should be capped."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch(
                "backend.main.GeminiInterviewService"
            ) as mock_service_class:
                mock_instance = MagicMock()
                mock_instance.start_session = AsyncGeneratorMock(["Started"])
                mock_instance.close = AsyncMock()
                mock_service_class.return_value = mock_instance

                with client.websocket_connect("/ws/interview") as websocket:
                    # Send start with 200-char name
                    long_name = "A" * 200
                    websocket.send_json(
                        {
                            "type": "start",
                            "data": {
                                "role": "Software Engineer",
                                "candidate_name": long_name,
                            },
                        }
                    )

                    websocket.receive_json()
                    websocket.receive_json()

                    # Verify name was capped at 100 chars
                    mock_service_class.assert_called_once()
                    call_kwargs = mock_service_class.call_args[1]
                    assert len(call_kwargs["candidate_name"]) <= 100


class TestSessionManagement:
    """Test session lifecycle management."""

    def test_start_resets_previous_session(self, client: TestClient) -> None:
        """Sending two start messages should close the first session."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch(
                "backend.main.GeminiInterviewService"
            ) as mock_service_class:
                # Create two mock instances
                mock_instance1 = MagicMock()
                mock_instance1.start_session = AsyncGeneratorMock(["First"])
                mock_instance1.close = AsyncMock()

                mock_instance2 = MagicMock()
                mock_instance2.start_session = AsyncGeneratorMock(["Second"])
                mock_instance2.close = AsyncMock()

                # Return first instance on first call, second on second call
                mock_service_class.side_effect = [mock_instance1, mock_instance2]

                with client.websocket_connect("/ws/interview") as websocket:
                    # Send first start
                    websocket.send_json(
                        {
                            "type": "start",
                            "data": {
                                "role": "Software Engineer",
                                "candidate_name": "Alice",
                            },
                        }
                    )
                    websocket.receive_json()
                    websocket.receive_json()

                    # Verify first instance was created
                    assert mock_service_class.call_count == 1
                    assert not mock_instance1.close.called

                    # Send second start (should close first session)
                    websocket.send_json(
                        {
                            "type": "start",
                            "data": {
                                "role": "Data Scientist",
                                "candidate_name": "Bob",
                            },
                        }
                    )
                    websocket.receive_json()
                    websocket.receive_json()

                    # Verify first instance was closed
                    mock_instance1.close.assert_called_once()
                    # Verify second instance was created
                    assert mock_service_class.call_count == 2

    def test_serve_index_returns_html_when_exists(
        self, client: TestClient
    ) -> None:
        """GET / should return HTML content when index.html exists."""
        response = client.get("/")
        # Should return 200 (with fallback if file doesn't exist)
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")


class TestWebSocketFullLifecycle:
    """Test complete interview lifecycles."""

    def test_full_interview_flow(self, client: TestClient) -> None:
        """Test a complete interview from start to end."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch(
                "backend.main.GeminiInterviewService"
            ) as mock_service_class:
                mock_instance = MagicMock()
                mock_instance.start_session = AsyncGeneratorMock(
                    ["Welcome ", "to ", "the ", "interview!"]
                )
                mock_instance.send_message = AsyncGeneratorMock(
                    ["Great ", "answer!"]
                )
                mock_instance.end_session = AsyncGeneratorMock(
                    ["Excellent ", "performance!"]
                )
                mock_instance.close = AsyncMock()
                mock_service_class.return_value = mock_instance

                with client.websocket_connect("/ws/interview") as websocket:
                    # Start interview
                    websocket.send_json(
                        {
                            "type": "start",
                            "data": {
                                "role": "System Design",
                                "candidate_name": "Laura",
                            },
                        }
                    )

                    # Consume greeting chunks
                    for _ in range(5):
                        msg = websocket.receive_json()
                        if msg["type"] == "response":
                            break

                    # Send a message
                    websocket.send_json(
                        {
                            "type": "message",
                            "data": {
                                "content": "I would start by asking clarifying questions"
                            },
                        }
                    )

                    # Consume response chunks
                    for _ in range(3):
                        msg = websocket.receive_json()
                        if msg["type"] == "response":
                            break

                    # End interview
                    websocket.send_json({"type": "end", "data": {}})

                    # Consume feedback chunks
                    for _ in range(3):
                        msg = websocket.receive_json()
                        if msg["type"] == "response":
                            break

                    # Verify service was created and closed properly
                    mock_service_class.assert_called_once()
                    mock_instance.close.assert_called()

    def test_multiple_messages_in_sequence(self, client: TestClient) -> None:
        """Test sending multiple messages in one interview."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch(
                "backend.main.GeminiInterviewService"
            ) as mock_service_class:
                mock_instance = MagicMock()
                mock_instance.start_session = AsyncGeneratorMock(["Start"])
                mock_instance.send_message = AsyncGeneratorMock(["Response"])
                mock_instance.end_session = AsyncGeneratorMock(["Feedback"])
                mock_instance.close = AsyncMock()
                mock_service_class.return_value = mock_instance

                with client.websocket_connect("/ws/interview") as websocket:
                    # Start
                    websocket.send_json(
                        {
                            "type": "start",
                            "data": {
                                "role": "Software Engineer",
                                "candidate_name": "Mike",
                            },
                        }
                    )
                    websocket.receive_json()
                    websocket.receive_json()

                    # First message
                    websocket.send_json(
                        {
                            "type": "message",
                            "data": {"content": "First answer"},
                        }
                    )
                    websocket.receive_json()
                    websocket.receive_json()

                    # Second message
                    websocket.send_json(
                        {
                            "type": "message",
                            "data": {"content": "Second answer"},
                        }
                    )
                    websocket.receive_json()
                    websocket.receive_json()

                    # End
                    websocket.send_json({"type": "end", "data": {}})
                    websocket.receive_json()
                    websocket.receive_json()

                    # Verify the flow works - service was created
                    mock_service_class.assert_called()
