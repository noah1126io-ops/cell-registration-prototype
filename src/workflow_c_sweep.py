"""Deterministic Workflow C experiment orchestration; no registration changes."""
from __future__ import annotations

import copy
import csv
import hashlib
import inspect
import itertools
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np

from src.workflow_c_slurm import INPUT_KEYS, update_status, utc_now, write_json_atomic

ROOT = Path(__file__).resolve().parents[1]
SUPPORTED = (
    'stage_a_learning_rate', 'stage_b_learning_rate',
    'stage_a_update_smoothing', 'stage_b_update_smoothing',
    'density_weight', 'support_weight', 'structure_weight', 'soft_jacobian_weight',
)
ARTIFACT = 'workflow_c_registration_result.zip'


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':'))


def effective_config(base):
    from src.density_flow import tissue_aware_density_flow_registration, joint_density_tissue_structure_registration
    from src.joint_flow import two_stage_joint_flow_registration
    # These density arguments are supplied explicitly by Joint Flow, not forwarded kwargs.
    excluded = {'bounds', 'success_metric_fixed_points', 'success_metric_moving_points',
                'moving_tissue_mask', 'moving_tissue_metadata', 'moving_structure_image',
                'fixed_structure_feature_grid', 'moving_structure_feature_grid',
                'fixed_support_feature_grid', 'moving_support_feature_grid',
                'density_channel_weight', 'tissue_support_channel_weight', 'structure_channel_weight'}
    defaults = {}
    for function in (tissue_aware_density_flow_registration, two_stage_joint_flow_registration,
                     joint_density_tissue_structure_registration):
        for name, parameter in inspect.signature(function).parameters.items():
            if parameter.default is not inspect.Parameter.empty and name not in excluded:
                defaults[name] = parameter.default
    config = copy.deepcopy(base)
    if config.get('fine_method', 'joint density + tissue-structure flow') != 'joint density + tissue-structure flow':
        raise ValueError('Sweeps support Joint Flow only')
    supplied = config.get('parameters', {})
    unknown = set(supplied) - set(defaults) - {'bounds'}
    if unknown:
        raise ValueError(f'Unsupported base parameters: {sorted(unknown)}')
    if supplied.get('device', 'cuda') != 'cuda' or supplied.get('dtype', 'float64') != 'float64':
        raise ValueError('Sweep execution requires cuda / float64')
    defaults.update(supplied)
    defaults.update(device='cuda', dtype='float64')
    config['parameters'] = json.loads(canonical(defaults))
    config.setdefault('fine_method', 'joint density + tissue-structure flow')
    config.setdefault('inverse_iterations', 12)
    config.setdefault('inverse_tolerance_pixels', 0.05)
    return config


def trial_configs(base, spec, max_trials=100):
    if spec.get('method', 'grid') != 'grid':
        raise ValueError('Only grid search is supported')
    if type(max_trials) is not int or max_trials < 1:
        raise ValueError('max_trials must be a positive integer')
    space = spec.get('parameters', {})
    if not isinstance(space, dict) or set(space) - set(SUPPORTED):
        raise ValueError('Only the documented eight sweep parameters are allowed; safety thresholds cannot be swept')
    keys = sorted(space)
    values = []
    for key in keys:
        candidates = space[key]
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f'{key} requires a nonempty list')
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 or
               ('learning_rate' in key and v == 0) for v in candidates):
            raise ValueError(f'Invalid numeric values for {key}')
        values.append(sorted(set(float(v) for v in candidates)))
    product_count = math.prod(map(len, values))
    if product_count > max_trials:
        raise ValueError(f'Grid has {product_count} trials, exceeding max_trials={max_trials}')
    # Baseline first, then sorted Cartesian product, with numeric duplicate elimination.
    configs = [copy.deepcopy(base)]
    baseline_key = tuple(float(base['parameters'][key]) for key in keys)
    seen = {baseline_key}
    for combination in itertools.product(*values):
        if combination in seen:
            continue
        seen.add(combination)
        config = copy.deepcopy(base)
        config['parameters'].update(zip(keys, combination))
        configs.append(config)
    if len(configs) > max_trials:
        raise ValueError(f'Grid plus baseline has {len(configs)} trials, exceeding max_trials={max_trials}')
    return configs, product_count


def create_sweep(base_path, spec_path, output_root, *, sweep_id=None, max_trials=None, concurrency=None):
    base_path, spec_path = Path(base_path).resolve(), Path(spec_path).resolve()
    original, spec = read_json(base_path), read_json(spec_path)
    base = effective_config(original)
    limit = max_trials if max_trials is not None else spec.get('max_trials', 100)
    concurrency = concurrency if concurrency is not None else spec.get('concurrency', 2)
    if type(concurrency) is not int or concurrency < 1:
        raise ValueError('concurrency must be a positive integer')
    configs, product_count = trial_configs(base, spec, limit)
    sweep_id = sweep_id or f"{utc_now()[:10]}-{uuid.uuid4().hex[:8]}"
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,100}', sweep_id):
        raise ValueError('Unsafe sweep_id')
    sources = {}
    for key, value in base.get('inputs', {}).items():
        if key not in INPUT_KEYS:
            raise ValueError(f'Unsupported input: {key}')
        source = Path(value).expanduser()
        source = (base_path.parent / source).resolve() if not source.is_absolute() else source.resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        sources[key] = source
    if 'synthetic' not in base and not {'fixed_points', 'moving_points', 'he_image', 'tissue_mask', 'metadata'} <= sources.keys():
        raise ValueError('Real Joint Flow case is missing required inputs')
    directory = Path(output_root).resolve() / sweep_id
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'input').mkdir()
    identities, staged, by_source = {}, {}, {}
    for key, source in sources.items():
        if source not in by_source:
            destination = directory / 'input' / f'{key}{source.suffix}'
            shutil.copy2(source, destination)
            checksum = digest(source)
            if digest(destination) != checksum:
                raise IOError(f'Input staging verification failed: {source}')
            destination.chmod(0o444)
            by_source[source] = (destination, checksum)
        destination, checksum = by_source[source]
        staged[key] = f'input/{destination.name}'
        identities[key] = {'source': str(source), 'path': staged[key], 'sha256': checksum,
                           'size_bytes': destination.stat().st_size}
    trials = []
    for index, config in enumerate(configs):
        trial_id = f'{index:04d}'
        trial = directory / 'trials' / trial_id
        trial.mkdir(parents=True)
        (trial / 'input').symlink_to('../../input', target_is_directory=True)
        config['inputs'] = staged
        write_json_atomic(trial / 'config.json', config)
        update_status(trial, 'queued', sweep_id=sweep_id, trial_id=trial_id, array_index=index)
        trials.append({'trial_id': trial_id, 'index': index, 'is_baseline': index == 0,
                       'run_directory': f'trials/{trial_id}', 'config_sha256': digest(trial / 'config.json'),
                       'effective_parameters': config['parameters']})
    sha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    write_json_atomic(directory / 'sweep_config.json', spec)
    write_json_atomic(directory / 'manifest.json', {
        'sweep_id': sweep_id, 'created_at': utc_now(), 'git_commit': sha, 'git_dirty': bool(dirty),
        'base_config_source': str(base_path), 'original_base_config': original,
        'base_config': read_json(directory / 'trials/0000/config.json'),
        'search_space': spec.get('parameters', {}), 'method': 'grid', 'grid_trials': product_count,
        'total_trials': len(trials), 'max_trials': limit, 'concurrency': concurrency,
        'inputs': identities, 'trials': trials, 'array_job_id': None,
        'source_sha256': {str(p.relative_to(ROOT)): digest(p) for folder in ('src', 'scripts', 'slurm') for p in sorted((ROOT / folder).glob('*')) if p.suffix in ('.py', '.sbatch')},
        'environment': {'python': sys.version, 'platform': platform.platform(), 'hostname': platform.node()},
    })
    sweep_status(directory)
    return directory


def trial_path(directory, index):
    directory = Path(directory).resolve()
    manifest = read_json(directory / 'manifest.json')
    if type(index) is not int or not 0 <= index < len(manifest['trials']):
        raise ValueError('Array index outside manifest')
    entry = manifest['trials'][index]
    expected = f'trials/{index:04d}'
    if entry['index'] != index or entry['trial_id'] != f'{index:04d}' or entry['run_directory'] != expected:
        raise ValueError('Unsafe or inconsistent trial mapping')
    path = directory / expected
    if path.resolve() != path or (path / 'config.json').is_symlink():
        raise ValueError('Trial path must not escape sweep directory')
    if digest(path / 'config.json') != entry['config_sha256']:
        raise ValueError('Trial config checksum mismatch')
    return path


def sweep_status(directory):
    directory = Path(directory)
    manifest = read_json(directory / 'manifest.json')
    counts = dict.fromkeys(('queued', 'running', 'completed', 'failed'), 0)
    for entry in manifest['trials']:
        status = read_json(trial_path(directory, entry['index']) / 'status.json')
        counts[status['state']] += 1
    result = {'total': len(manifest['trials']), **counts, 'updated_at': utc_now()}
    write_json_atomic(directory / 'status.json', result)
    return result


def verify_sources(manifest):
    for name, checksum in manifest.get('source_sha256', {}).items():
        path = ROOT / name
        if not path.resolve().is_relative_to(ROOT) or digest(path) != checksum:
            raise ValueError(f'Source changed since sweep creation: {name}; create a new sweep')


def submit_sweep(directory):
    directory = Path(directory).resolve()
    manifest = read_json(directory / 'manifest.json')
    if manifest['array_job_id'] is not None:
        raise ValueError('Sweep already submitted; automatic resubmission is not supported')
    verify_sources(manifest)
    if type(manifest['concurrency']) is not int or manifest['concurrency'] < 1:
        raise ValueError('Invalid concurrency')
    count = manifest['total_trials']
    if count != len(manifest['trials']) or not 1 <= count <= manifest['max_trials']:
        raise ValueError('Invalid trial count')
    for index in range(count):
        trial_path(directory, index)
    print(f"total_trials={count}; array=0-{count - 1}%{manifest['concurrency']}", flush=True)
    # Exclusive marker prevents accidental duplicate submission, including concurrent CLI calls.
    with (directory / 'submission_started.json').open('x') as stream:
        json.dump({'at': utc_now()}, stream)
    command = ['sbatch', '--parsable', f"--array=0-{count-1}%{manifest['concurrency']}",
               f'--output={directory}/slurm-%A_%a.stdout.log', f'--error={directory}/slurm-%A_%a.stderr.log',
               str(ROOT / 'slurm/workflow_c_sweep_gpu.sbatch'), str(ROOT), str(directory)]
    try:
        response = subprocess.run(command, check=True, capture_output=True, text=True)
        job_id = response.stdout.strip().split(';')[0]
        if not job_id.isdigit():
            raise RuntimeError(f'Unexpected sbatch output: {response.stdout!r}')
        manifest.update(array_job_id=job_id, submission_command=command, submitted_at=utc_now())
        write_json_atomic(directory / 'manifest.json', manifest)
        # Do not write queued here: a fast worker may already be running/completed.
        for entry in manifest['trials']:
            write_json_atomic(trial_path(directory, entry['index']) / 'slurm_job.json', {
                'array_job_id': job_id, 'array_task_id': entry['index'],
                'partition': 'mib-dbia', 'gres': 'gpu:1', 'nodes': 1, 'ntasks': 1,
                'cpus_per_task': 4, 'memory': '16G', 'time': '00:30:00'})
        return job_id
    except BaseException as exc:
        if isinstance(exc, (subprocess.CalledProcessError, FileNotFoundError)):
            for entry in manifest['trials']:
                trial = trial_path(directory, entry['index'])
                if read_json(trial / 'status.json')['state'] == 'queued':
                    update_status(trial, 'failed', error=f'sbatch submission failed: {exc}')
            sweep_status(directory)
        write_json_atomic(directory / 'submission_error.json', {'error': str(exc), 'stderr': getattr(exc, 'stderr', None), 'stdout': getattr(exc, 'stdout', None), 'at': utc_now()})
        raise


def run_trial(directory, index):
    from src.workflow_c_worker import run_worker
    directory = Path(directory).resolve()
    trial = trial_path(directory, index)
    if read_json(trial / 'status.json')['state'] != 'queued':
        raise ValueError('Trial is not queued; refusing to overwrite previous execution')
    with (trial / 'execution_started.json').open('x') as stream:
        json.dump({'at': utc_now(), 'job_id': os.environ.get('SLURM_JOB_ID'),
                   'array_job_id': os.environ.get('SLURM_ARRAY_JOB_ID'), 'array_task_id': index}, stream)
    try:
        manifest = read_json(directory / 'manifest.json')
        verify_sources(manifest)
        for identity in manifest['inputs'].values():
            path = directory / identity['path']
            if not path.resolve().is_relative_to(directory / 'input') or digest(path) != identity['sha256']:
                raise ValueError('Shared input identity mismatch')
        if (trial / 'input').resolve() != directory / 'input':
            raise ValueError('Invalid shared input link')
        os.environ.setdefault('SLURM_TMPDIR', f"/tmp/cellreg_{os.environ.get('SLURM_JOB_ID', 'unknown')}_{index}")
        return run_worker(trial / 'config.json')
    except BaseException as exc:
        if read_json(trial / 'status.json')['state'] != 'completed':
            update_status(trial, 'failed', error=f'{type(exc).__name__}: {exc}')
        raise


def artifact_row(path, expected_parameters=None):
    from src.workflow_c_result import load_workflow_c_result_artifact
    artifact = load_workflow_c_result_artifact(Path(path).read_bytes())
    if expected_parameters is not None and canonical(artifact.parameters) != canonical(expected_parameters):
        raise ValueError('Artifact parameters do not match trial config')
    metrics, provenance = artifact.metrics, artifact.provenance
    safety, final = metrics['safety'], metrics['joint_flow']['final']
    applied = metrics['applied']
    finite = all(np.isfinite(a).all() for a in artifact.arrays.values() if np.issubdtype(a.dtype, np.number))
    row = {key: applied[key] for key in ('symmetric_median_distance', 'within_3', 'within_5', 'within_10', 'mutual_nearest_fraction')}
    row.update({key: final[key] for key in ('displacement_p95', 'displacement_max', 'jacobian_min', 'jacobian_p05', 'jacobian_median', 'jacobian_p95', 'jacobian_max')})
    row.update(displacement_median=final['displacement_p50'], foldover=safety['fraction_jacobian_foldover_le_0'],
               runtime=metrics['runtime_breakdown']['total_runtime_seconds'], hostname=provenance.get('hostname'),
               gpu_model=provenance.get('gpu_model'), slurm_job_id=provenance.get('SLURM_JOB_ID'),
               applied=artifact.applied, rejected=not artifact.applied,
               fine_status='applied' if artifact.applied else 'rejected',
               rejection_reasons='' if artifact.applied else 'no_joint_checkpoint_passed_final_application_safety')
    # Existing artifacts do not serialize FineWarpResult.message/rejection_reason separately.
    row['fine_status_source'] = 'artifact.manifest.fine_applied'
    numeric = [v for v in row.values() if type(v) in (float, int)]
    row['valid'] = bool(artifact.applied and safety.get('finite_output') is True and finite
                        and all(math.isfinite(v) for v in numeric) and row['foldover'] == 0
                        and row['jacobian_min'] > 0)
    return row


def summarize_sweep(directory):
    directory = Path(directory).resolve()
    manifest = read_json(directory / 'manifest.json')
    rows = []
    for entry in manifest['trials']:
        trial = trial_path(directory, entry['index'])
        status = read_json(trial / 'status.json')
        artifact = trial / 'result' / ARTIFACT
        row = {'trial_id': entry['trial_id'], 'array_index': entry['index'], 'is_baseline': entry['is_baseline'],
               'state': status['state'], 'valid': False, 'artifact': str(artifact),
               'config': str(trial / 'config.json'), 'array_job_id': manifest['array_job_id'],
               'error': status.get('error', ''), **entry['effective_parameters']}
        if status['state'] == 'completed':
            try:
                row.update(artifact_row(artifact, entry['effective_parameters']))
            except Exception as exc:
                row['error'] = f'Artifact validation failed: {exc}'
        row['validity'] = 'VALID' if row['valid'] else ('PENDING' if status['state'] in ('queued', 'running') else 'INVALID')
        rows.append(row)
    baseline = rows[0]
    for row in rows:
        for label, key in [('median_distance','symmetric_median_distance'), ('within5','within_5'),
                           ('mutual_nearest','mutual_nearest_fraction'), ('displacement_p95','displacement_p95')]:
            a, b = row.get(key), baseline.get(key)
            row[f'delta_{label}_vs_baseline'] = a-b if isinstance(a,(float,int)) and isinstance(b,(float,int)) else None
    valid = [row for row in rows if row['valid']]
    ranked = sorted(valid, key=lambda r: (r['symmetric_median_distance'], -r['within_5'],
                                         -r['mutual_nearest_fraction'], r['displacement_p95'], r['array_index']))
    ranked = [dict(row, rank=i+1) for i,row in enumerate(ranked)]
    summary = directory / 'summary'
    summary.mkdir(exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    for name, data in [('trials',rows), ('valid_trials',valid), ('ranking',ranked)]:
        destination = summary / f'{name}.csv'
        temporary = destination.with_suffix('.csv.tmp')
        with temporary.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=(['rank'] if name=='ranking' else []) + columns)
            writer.writeheader()
            writer.writerows({k: canonical(v) if isinstance(v,(dict,list)) else v for k,v in row.items()} for row in data)
        os.replace(temporary, destination)
    counts = sweep_status(directory)
    write_json_atomic(summary / 'summary.json', {**counts, 'valid': len(valid), 'invalid': sum(r['validity']=='INVALID' for r in rows)})
    return rows
