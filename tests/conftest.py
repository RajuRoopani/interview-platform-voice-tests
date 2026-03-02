"""
conftest.py — Test discovery and path configuration for interview_platform tests.

Ensures that `from backend.main import app` works regardless of where pytest
is invoked from, by adding the project root
(/workspace/interview_platform/) to sys.path.
"""

from __future__ import annotations

import sys
from pathlib import Path

# /workspace/interview_platform/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
