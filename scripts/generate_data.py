"""Generate the synthetic portfolio and the 300+ tabular features into data/.

    python scripts/generate_data.py [--clients 45000] [--seed 42]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scoring.config import DATA_DIR  # noqa: E402
from scoring.features import build_features  # noqa: E402
from scoring.synth import SynthConfig, generate, save  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=int, default=SynthConfig.n_clients)
    ap.add_argument("--seed", type=int, default=SynthConfig.seed)
    a = ap.parse_args()
    t0 = time.time()
    data = generate(SynthConfig(n_clients=a.clients, seed=a.seed))
    save(data, DATA_DIR)
    feats = build_features(data.applications, data.loans, data.events, data.external)
    feats.to_parquet(DATA_DIR / "features.parquet", index=False)
    print(f"{len(data.applications):,} applications, {len(data.events):,} events, "
          f"{feats.shape[1] - 1} features -> {DATA_DIR} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
