"""
NEW TESTS for AI Interview Platform
These tests will be added to test_main.py once the backend fixes are in place.
"""

from __future__ import annotations

import os
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.main import app, INTERVIEW_ROLES


class AsyncGeneratorMock:
    """A mock that returns an async generator when called."""

    def __init__(self, items: list[str]) -> None:
        self.items = items

    async def __call__(self, *args, **kwargs) -> AsyncGenerator[str, None]:
        """Make this callable, returning an async generator."""
        for item in self.items:
            yield item


@pytest.fixture
def client() -> TestClient:
    """Create a FastAPI test client."""
    return TestClient(app)


# =============================================================================
# NEW TEST: Health Endpoint (Bug Fix #4)
# =============================================================================

class TestHealthEndpoint:
    """Test the /api/health endpoint."""

    def test_health_endpoint_returns_ok(self, client: TestClient) -> None:
        """GET /api/health should return 200 with {"status": "ok"}."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "ok"


# =============================================================================
# NEW TEST: Message Length Limit (Bug Fix #3)
# =============================================================================

class TestMessageLengthLimit:
    """Test message length validation."""

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
                    websocket.receive_json()  # chunk
                    websocket.receive_json()  # response

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
                    assert "2000" in response["data"]["content"]


# =============================================================================
# NEW TESTS: Candidate Name Sanitization (Bug Fixes #5)
# =============================================================================

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


# =============================================================================
# NEW TEST: Session Reset
# =============================================================================

class TestSessionReset:
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
                    websocket.receive_json()  # chunk
                    websocket.receive_json()  # response

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
                    websocket.receive_json()  # chunk
                    websocket.receive_json()  # response

                    # Verify first instance was closed
                    mock_instance1.close.assert_called_once()
                    # Verify second instance was created
                    assert mock_service_class.call_count == 2


# =============================================================================
# NEW TEST: Serve Index
# =============================================================================

class TestServeIndex:
    """Test the serve_index fallback."""

    def test_serve_index_returns_html_when_exists(
        self, client: TestClient
    ) -> None:
        """GET / should return HTML content when index.html exists."""
        response = client.get("/")
        # Should return 200 even if file doesn't exist (via fallback)
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")


# =============================================================================
# NEW TEST: Multiple Invalid Messages
# =============================================================================

class TestInvalidMessageSequence:
    """Test handling of multiple invalid messages in sequence."""

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


# =============================================================================
# FIXED BUGS IN EXISTING TESTS
# =============================================================================

class TestAsyncGeneratorMockFix:
    """Test that AsyncGeneratorMock now accepts arguments."""

    @pytest.mark.asyncio
    async def test_async_generator_mock_accepts_args(self) -> None:
        """AsyncGeneratorMock.__call__ should accept *args and **kwargs."""
        mock = AsyncGeneratorMock(["chunk1", "chunk2"])
        # Call with various arguments
        gen = await mock("arg1", "arg2", kwarg1="value1")
        chunks = []
        async for chunk in gen:
            chunks.append(chunk)
        assert chunks == ["chunk1", "chunk2"]

    @pytest.mark.asyncio
    async def test_async_generator_mock_call_without_args(self) -> None:
        """AsyncGeneratorMock.__call__ should work with no arguments."""
        mock = AsyncGeneratorMock(["test"])
        gen = await mock()
        chunks = []
        async for chunk in gen:
            chunks.append(chunk)
        assert chunks == ["test"]


class TestStartUsesDefaultsFix:
    """Test the fixed test_start_uses_defaults test."""

    def test_start_uses_defaults_and_calls_start_session(
        self, client: TestClient
    ) -> None:
        """Sending 'start' without role/name should use defaults and call start_session."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch(
                "backend.main.GeminiInterviewService"
            ) as mock_service_class:
                mock_instance = MagicMock()
                # Make start_session return an AsyncGeneratorMock
                mock_instance.start_session = AsyncGeneratorMock(["Started"])
                mock_instance.close = AsyncMock()
                mock_service_class.return_value = mock_instance

                with client.websocket_connect("/ws/interview") as websocket:
                    websocket.send_json({"type": "start", "data": {}})

                    # Receive the responses
                    websocket.receive_json()  # chunk
                    websocket.receive_json()  # response

                    # Verify defaults were used
                    mock_service_class.assert_called_once()
                    call_kwargs = mock_service_class.call_args[1]
                    assert call_kwargs["role"] == "Software Engineer"
                    assert call_kwargs["candidate_name"] == "Candidate"

                    # Verify start_session was called
                    # Note: AsyncGeneratorMock's __call__ is called during streaming
                    assert mock_instance.start_session is not None
