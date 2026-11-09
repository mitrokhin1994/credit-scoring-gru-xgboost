import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scoring.synth import SynthConfig, generate  # noqa: E402


@pytest.fixture(scope="session")
def small_data():
    return generate(SynthConfig(n_clients=1500, seed=7))
