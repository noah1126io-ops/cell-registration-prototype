from __future__ import annotations
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.workflow_c_sweep import create_sweep, read_json

def main():
    parser = argparse.ArgumentParser(description="Create a deterministic Joint Flow grid sweep; does not submit jobs.")
    parser.add_argument('--base-config', type=Path)
    parser.add_argument('--sweep-config', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'sweeps')
    parser.add_argument('--sweep-id')
    parser.add_argument('--max-trials', type=int, help='Explicit override of default 100-trial limit, including baseline')
    parser.add_argument('--concurrency', type=int)
    args = parser.parse_args()
    base = args.base_config
    if base is None:
        value = read_json(args.sweep_config).get('base_config')
        if not value:
            parser.error('--base-config or sweep specification base_config is required')
        base = args.sweep_config.resolve().parent / value
    directory = create_sweep(base, args.sweep_config, args.output_root, sweep_id=args.sweep_id,
                             max_trials=args.max_trials, concurrency=args.concurrency)
    manifest = read_json(directory / 'manifest.json')
    print(f"sweep_dir={directory}")
    print(f"grid_trials={manifest['grid_trials']}; total_trials={manifest['total_trials']}; concurrency={manifest['concurrency']}")

if __name__ == '__main__':
    main()
