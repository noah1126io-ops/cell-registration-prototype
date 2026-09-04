from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.workflow_c_slurm import submit_workflow_c


def main() -> None:
    parser = argparse.ArgumentParser(description="Create and submit one Workflow C GPU run.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    run_dir, job_id = submit_workflow_c(args.config, args.runs_dir, run_id=args.run_id)
    print(f"run_dir={run_dir}")
    print(f"job_id={job_id}")


if __name__ == "__main__":
    main()
