import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from cdqc.config import load_config


@pytest.fixture
def cfg(tmp_path):
    """내장 기본값 + 임시 root의 Config."""
    c = load_config(None, root=str(tmp_path))
    return c
