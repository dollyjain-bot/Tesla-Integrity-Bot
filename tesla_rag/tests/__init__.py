"""Test package for the Tesla Integrity Bot. Run from the project root: python -m unittest discover -s tests -t ."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
