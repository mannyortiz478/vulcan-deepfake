"""Test helper to ensure project root is on sys.path for pytest runs.

This makes `import src...` work when running tests from the project root or CI.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
