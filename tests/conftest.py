import sys
from pathlib import Path

import pytest

# Project root on path so imports work
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))


@pytest.fixture
def project_root():
    return root
