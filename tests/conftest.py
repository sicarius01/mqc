import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from cdqc import Params


@pytest.fixture
def params():
    return Params()
