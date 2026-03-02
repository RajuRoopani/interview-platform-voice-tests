"""
Root-level conftest.py for interview_platform.

Adds the project root to sys.path so pytest can discover tests in
interview_platform/tests/ and resolve `from backend.main import app`
correctly when pytest is run from /workspace or from within the
interview_platform/ directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

# /workspace/interview_platform/
_PROJECT_ROOT = Path(__file__).resolve().parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
