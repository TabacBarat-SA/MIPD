from pathlib import Path
import os
ROOT = Path(os.environ.get('CLIENTE360_ROOT', Path(__file__).resolve().parents[1]))
RAW = ROOT / 'data/raw'
SYN = ROOT / 'data/synthetic'
OUT = ROOT / 'data/processed'
ART = ROOT / 'artifacts'
REPORT = ROOT / 'reports'
SEED = 42
CUTS = {'train': '2017-10-01', 'val': '2018-01-01', 'test': '2018-04-01'}
for p in [SYN, OUT, ART, REPORT]: p.mkdir(parents=True, exist_ok=True)
