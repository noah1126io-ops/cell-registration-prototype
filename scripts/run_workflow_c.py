from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.workflow_c_worker import run_worker


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Workflow C CUDA job inside Slurm.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(run_worker(args.config))


if __name__ == "__main__":
    main()
