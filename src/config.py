import os
from pathlib import Path

# Standard local resolution: src/config.py is one level down from the project root
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
CACHE_DIR = ROOT / 'cache'
OUTPUT_DIR = ROOT / 'output'

# Initialize output directories
for d in [CACHE_DIR, OUTPUT_DIR, OUTPUT_DIR / 'figures']:
    d.mkdir(parents=True, exist_ok=True)

# Global convention: reproducibility
SEED = 42

# Global convention: universe thresholds (Defaults tunable under Step 10)
UNIVERSE_PRC_MIN = 5.0
UNIVERSE_ME_PCT = 0.20

# Short-eligible thresholds
SHORT_ME_PCT = 0.40
SHORT_DOLVOL_PCT = 0.30
SHORT_ZEROTRADES_PCT = 0.50  # Median

# Preprocessing thresholds
MISSING_FLAG_THRESHOLD = 0.20