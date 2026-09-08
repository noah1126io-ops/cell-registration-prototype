from __future__ import annotations
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.workflow_c_sweep import summarize_sweep

def main():
    parser = argparse.ArgumentParser(description="Summarize a prepared Workflow C sweep.")
    parser.add_argument('--sweep-dir', type=Path, required=True)
    args = parser.parse_args()
    result = summarize_sweep(args.sweep_dir)
    print(result if isinstance(result, str) else f"Wrote summary for {len(result)} trials to {args.sweep_dir / 'summary'}")

if __name__ == '__main__':
    main()
