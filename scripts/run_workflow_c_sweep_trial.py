from __future__ import annotations
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
from src.workflow_c_sweep import run_trial

def main():
    parser = argparse.ArgumentParser(description="Execute exactly one Slurm array task.")
    parser.add_argument('--sweep-dir', type=Path, required=True)
    args = parser.parse_args()
    print(run_trial(args.sweep_dir, int(os.environ['SLURM_ARRAY_TASK_ID'])))

if __name__ == '__main__':
    main()
