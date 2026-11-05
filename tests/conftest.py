import sys
from pathlib import Path

import mpmath as mp
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Module-level test parameters (e.g. mp.pi/6 in parametrize lists) are created at
# import time, so set the precision here as well as in the fixture below.
mp.mp.dps = 30


@pytest.fixture(autouse=True)
def _precision():
    """Build angles such as mp.pi/3 at 30 digits inside every test."""
    with mp.workdps(30):
        yield
