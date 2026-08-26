from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.flow_ablation_benchmark import (
    STANDARD_ABLATIONS,
    ablation_benchmark_artifacts,
    run_ablation_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Workflow C synthetic flow ablations.")
    parser.add_argument("--output-dir", type=Path, default=Path("ablation_benchmark_output"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--smoke", action="store_true", help="Run a small Full/-Density check.")
    arguments = parser.parse_args()
    ablations = ("Full", "-Density") if arguments.smoke else STANDARD_ABLATIONS
    dropouts = (0.0, 0.25) if arguments.smoke else (0.0, 0.10, 0.25, 0.50, 0.75)
    results, summary, metadata = run_ablation_benchmark(
        ablations=ablations,
        dropout_fractions=dropouts,
        seed=arguments.seed,
        size=arguments.size,
        iterations_per_level=arguments.iterations,
    )
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in ablation_benchmark_artifacts(results, summary, metadata).items():
        (arguments.output_dir / name).write_bytes(payload)
    print(f"Wrote {len(results)} runs to {arguments.output_dir.resolve()}")


if __name__ == "__main__":
    main()
