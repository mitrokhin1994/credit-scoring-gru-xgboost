"""Project-wide constants: paths, dates, targets."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
ARTEFACTS_DIR = ROOT / "artefacts"
FIGURES_DIR = ROOT / "reports" / "figures"

SEED = 42

# Timeline of the synthetic portfolio
HISTORY_START = pd.Timestamp("2024-01-01")  # first loans of the oldest clients
SAMPLE_START = pd.Timestamp("2025-01-01")   # repeat applications that enter the modelling sample
OOT_START = pd.Timestamp("2026-01-01")      # out-of-time holdout starts here
SAMPLE_END = pd.Timestamp("2026-04-30")     # last application date

TARGET = "npl30d90"                         # DPD >= 30 within 90 days after issue
AUX_TARGETS = ["fpd30", "fpd15", "fpd10"]   # first-payment defaults, extra GRU heads
CLIENT_COL = "client_id"
APP_COL = "application_id"
DATE_COL = "application_date"
SEGMENTS = ["L2", "L3", "L4+"]              # 2nd, 3rd, 4th+ loan of the client
