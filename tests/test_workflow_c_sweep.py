import copy
import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src import workflow_c_sweep as sweep
from src.workflow_c_slurm import update_status
from test_workflow_c_result import _artifact_bytes


@pytest.fixture(autouse=True)
def no_real_sbatch(monkeypatch):
    real = sweep.subprocess.run
    def guarded(command, *args, **kwargs):
        assert command[0] != 'sbatch', 'Unit tests must never submit real jobs'
        return real(command, *args, **kwargs)
    monkeypatch.setattr(sweep.subprocess, 'run', guarded)


@pytest.fixture
def prepared(tmp_path):
    base = tmp_path / 'base.json'
    spec = tmp_path / 'space.json'
    base.write_text(json.dumps({'synthetic': {'size': 32, 'n_points': 30}, 'parameters': {}}))
    spec.write_text(json.dumps({'method': 'grid', 'parameters': {
        'stage_a_learning_rate': [0.15, 0.1, 0.1], 'stage_b_learning_rate': [0.075, 0.05]}}))
    return sweep.create_sweep(base, spec, tmp_path / 'sweeps', sweep_id='test')


def result_bytes(*, applied=True, median=1.0, foldover=0.0, finite=True):
    source = _artifact_bytes(applied=applied)
    metrics = {
        'applied': {'symmetric_median_distance': median, 'within_3': 0.4, 'within_5': 0.6,
                    'within_10': 0.8, 'mutual_nearest_fraction': 0.5},
        'safety': {'finite_output': finite, 'fraction_jacobian_foldover_le_0': foldover},
        'joint_flow': {'final': {'displacement_p50': 0.5, 'displacement_p95': 0.5, 'displacement_max': 0.5,
                               'jacobian_min': 1., 'jacobian_p05': 1., 'jacobian_median': 1., 'jacobian_p95': 1., 'jacobian_max': 1.}},
        'runtime_breakdown': {'total_runtime_seconds': 1.}}
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(source)) as archive, zipfile.ZipFile(out, 'w') as dest:
        for name in archive.namelist():
            dest.writestr(name, json.dumps(metrics) if name == 'metrics.json' else archive.read(name))
    return out.getvalue()


def complete(directory, index, **kwargs):
    trial = sweep.trial_path(directory, index)
    (trial / 'result').mkdir()
    payload = result_bytes(**kwargs)
    output = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(payload)) as source, zipfile.ZipFile(output, 'w') as destination:
        for name in source.namelist():
            destination.writestr(name, json.dumps(sweep.read_json(trial / 'config.json')['parameters']) if name == 'parameters.json' else source.read(name))
    (trial / 'result' / sweep.ARTIFACT).write_bytes(output.getvalue())
    update_status(trial, 'completed')


def test_cartesian_baseline_duplicates_and_effective_parameters(prepared):
    manifest = sweep.read_json(prepared / 'manifest.json')
    assert manifest['grid_trials'] == manifest['total_trials'] == 4
    assert [t['trial_id'] for t in manifest['trials']] == ['0000', '0001', '0002', '0003']
    assert [t['is_baseline'] for t in manifest['trials']] == [True, False, False, False]
    assert manifest['trials'][0]['effective_parameters']['stage_a_learning_rate'] == 0.1
    assert manifest['trials'][0]['effective_parameters']['support_weight'] == 0.25  # public wrapper default
    assert len(manifest['trials'][0]['effective_parameters']) > 50
    base = sweep.read_json(prepared / 'trials/0000/config.json')
    configs, count = sweep.trial_configs(base, sweep.read_json(prepared / 'sweep_config.json'))
    again, _ = sweep.trial_configs(base, {'parameters': {'stage_b_learning_rate': [0.05, 0.075], 'stage_a_learning_rate': [0.1, 0.15]}})
    assert configs == again
    for config in configs:
        for key in base:
            if key != 'parameters':
                assert config[key] == base[key]
        for key, value in base['parameters'].items():
            if key not in ('stage_a_learning_rate', 'stage_b_learning_rate'):
                assert config['parameters'][key] == value


def test_baseline_outside_grid_and_size_guard():
    base = sweep.effective_config({'synthetic': {}})
    spec = {'parameters': {'stage_a_learning_rate': [0.2, 0.3]}}
    configs, n = sweep.trial_configs(base, spec)
    assert n == 2 and len(configs) == 3
    with pytest.raises(ValueError, match='baseline'):
        sweep.trial_configs(base, spec, 2)
    with pytest.raises(ValueError, match='exceeding'):
        sweep.trial_configs(base, spec, 1)
    assert len(sweep.trial_configs(base, spec, 3)[0]) == 3


@pytest.mark.parametrize('parameters', [
    {'max_displacement': [100.]}, {'minimum_jacobian_p05': [0.]},
    {'stage_a_learning_rate': []}, {'structure_weight': [float('nan')]},
    {'support_weight': [-1]}, {'stage_b_learning_rate': [False]},
    {'stage_a_scales_um': [[8., 4.]]},
])
def test_reject_unsafe_search_spaces(parameters):
    with pytest.raises(ValueError):
        sweep.trial_configs(sweep.effective_config({}), {'parameters': parameters})


def test_mapping_and_path_safety(prepared):
    for index in range(4):
        assert sweep.trial_path(prepared, index).name == f'{index:04d}'
    for index in [-1, 4, '0']:
        with pytest.raises(ValueError):
            sweep.trial_path(prepared, index)
    config = prepared / 'trials/0001/config.json'
    config.write_text('{}')
    with pytest.raises(ValueError, match='checksum'):
        sweep.trial_path(prepared, 1)
    manifest = sweep.read_json(prepared / 'manifest.json')
    manifest['trials'][0]['run_directory'] = '../../escape'
    sweep.write_json_atomic(prepared / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='mapping'):
        sweep.trial_path(prepared, 0)


def test_shared_staging_relative_paths_and_input_identity(tmp_path, monkeypatch):
    np.save(tmp_path / 'points.npy', np.zeros((4,2)))
    (tmp_path / 'metadata.json').write_text('{}')
    base = tmp_path / 'base.json'
    base.write_text(json.dumps({'inputs': {key: 'points.npy' for key in ['fixed_points','moving_points','he_image','tissue_mask']} | {'metadata':'metadata.json'}}))
    spec = tmp_path / 'spec.json'
    spec.write_text('{"parameters": {"structure_weight": [0.15, 0.2]}}')
    with pytest.raises(ValueError, match='Unsafe'):
        sweep.create_sweep(base, spec, tmp_path / 'sweeps', sweep_id='../escape')
    directory = sweep.create_sweep(base, spec, tmp_path / 'sweeps')
    assert len(list((directory / 'input').iterdir())) == 2
    assert (directory / 'trials/0000/input').is_symlink()
    assert (directory / 'trials/0001/input').resolve() == directory / 'input'
    source = directory / 'input/fixed_points.npy'
    source.chmod(0o644)
    source.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='identity'):
        sweep.run_trial(directory, 0)
    assert sweep.read_json(directory / 'trials/0000/status.json')['state'] == 'failed'
    assert sweep.read_json(directory / 'trials/0001/status.json')['state'] == 'queued'


def test_submission_array_resources_and_no_state_regression(prepared, monkeypatch):
    calls = []
    def submit(command, **kwargs):
        calls.append(command)
        update_status(prepared / 'trials/0000', 'running')
        return SimpleNamespace(stdout='12345;cluster\n')
    monkeypatch.setattr(sweep.subprocess, 'run', submit)
    assert sweep.submit_sweep(prepared) == '12345'
    assert '--array=0-3%2' in calls[0]
    assert sweep.read_json(prepared / 'trials/0000/status.json')['state'] == 'running'
    assert sweep.read_json(prepared / 'trials/0003/slurm_job.json')['array_task_id'] == 3
    with pytest.raises(ValueError, match='already'):
        sweep.submit_sweep(prepared)
    assert len(calls) == 1
    launcher = (sweep.ROOT / 'slurm/workflow_c_sweep_gpu.sbatch').read_text()
    for request in ['--gres=gpu:1', '--nodes=1', '--ntasks=1', '--cpus-per-task=4', '--mem=16G', '--time=00:30:00']:
        assert request in launcher


def test_worker_mapping_states_scratch_and_failure_isolation(prepared, monkeypatch):
    from src import workflow_c_worker as worker
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    monkeypatch.delenv('SLURM_TMPDIR', raising=False)
    calls = []
    def fake_run(path):
        calls.append(sweep.read_json(path)['parameters'])
        update_status(path.parent, 'running')
        if path.parent.name == '0001':
            raise RuntimeError('isolated failure')
        update_status(path.parent, 'completed')
        return path
    monkeypatch.setattr(worker, 'run_worker', fake_run)
    sweep.run_trial(prepared, 0)
    with pytest.raises(RuntimeError, match='isolated'):
        sweep.run_trial(prepared, 1)
    sweep.run_trial(prepared, 2)
    status = sweep.sweep_status(prepared)
    assert (status['completed'], status['failed'], status['queued']) == (2,1,1)
    assert calls[1]['stage_b_learning_rate'] == 0.075
    history = sweep.read_json(prepared / 'trials/0000/status.json')['history']
    assert [x['state'] for x in history] == ['queued','running','completed']


def test_aggregation_validity_ranking_deltas_incomplete(prepared):
    complete(prepared, 0, median=2.)
    complete(prepared, 1, median=1.)
    complete(prepared, 2, applied=False, median=0.01)
    rows = sweep.summarize_sweep(prepared)
    assert [r['validity'] for r in rows] == ['VALID','VALID','INVALID','PENDING']
    assert rows[1]['delta_median_distance_vs_baseline'] == -1.
    assert rows[3]['delta_median_distance_vs_baseline'] is None
    import csv
    with (prepared / 'summary/ranking.csv').open() as stream:
        ranked = list(csv.DictReader(stream))
    assert [r['trial_id'] for r in ranked] == ['0001','0000']
    assert rows[2]['rejection_reasons']
    assert rows[0]['jacobian_p05'] == 1.


@pytest.mark.parametrize('kwargs', [{'foldover': 0.01}, {'finite': False}, {'median': float('nan')}])
def test_unsafe_artifact_excluded(prepared, kwargs):
    complete(prepared, 0, **kwargs)
    rows = sweep.summarize_sweep(prepared)
    assert rows[0]['validity'] == 'INVALID'
    assert sweep.read_json(prepared / 'summary/summary.json')['valid'] == 0


def test_corrupt_artifact_and_failed_trial_preserved(prepared):
    trial = sweep.trial_path(prepared, 0)
    (trial / 'result').mkdir()
    (trial / 'result' / sweep.ARTIFACT).write_bytes(b'broken')
    update_status(trial, 'completed')
    update_status(sweep.trial_path(prepared, 1), 'failed', error='oom')
    rows = sweep.summarize_sweep(prepared)
    assert 'Artifact validation failed' in rows[0]['error']
    assert rows[1]['error'] == 'oom'
    assert rows[2]['validity'] == 'PENDING'


def test_tied_ranking_is_deterministic(prepared):
    for index in range(4):
        complete(prepared, index)
    sweep.summarize_sweep(prepared)
    first = (prepared / 'summary/ranking.csv').read_bytes()
    sweep.summarize_sweep(prepared)
    assert (prepared / 'summary/ranking.csv').read_bytes() == first


def test_default_expansion_preserves_cpu_registration():
    from src.density_flow import joint_density_tissue_structure_registration
    from src.synthetic_registration_benchmark import generate_synthetic_registration_sample
    base = sweep.read_json(sweep.ROOT / 'examples/workflow_c_synthetic_gpu.json')
    sample = generate_synthetic_registration_sample(**base['synthetic'])
    def run(parameters):
        parameters = dict(parameters, device='cpu', dtype='float64')
        return joint_density_tissue_structure_registration(sample.fixed_points, sample.moving_points,
            affine_he_image=sample.moving_image, affine_he_tissue_mask=sample.moving_tissue_mask,
            affine_he_metadata=sample.metadata, bounds=sample.bounds, **parameters)
    original = run(base['parameters'])
    expanded = run(sweep.effective_config(base)['parameters'])
    assert original.applied == expanded.applied
    assert original.rejection_reason == expanded.rejection_reason
    for key in ['transformed_points','attempted_transformed_points','displacement_x','displacement_y']:
        np.testing.assert_array_equal(getattr(original,key), getattr(expanded,key))
    for key in ['safety','applied','attempted','optimization_history']:
        assert original.metrics[key] == expanded.metrics[key]


def test_submission_failure_keeps_stderr_and_marks_trials_failed(prepared, monkeypatch):
    def fail(command, **kwargs):
        raise sweep.subprocess.CalledProcessError(1, command, output='', stderr='controller unavailable')
    monkeypatch.setattr(sweep.subprocess, 'run', fail)
    with pytest.raises(sweep.subprocess.CalledProcessError):
        sweep.submit_sweep(prepared)
    assert sweep.read_json(prepared / 'submission_error.json')['stderr'] == 'controller unavailable'
    assert sweep.sweep_status(prepared)['failed'] == 4
    with pytest.raises(FileExistsError):
        sweep.submit_sweep(prepared)


def test_changed_source_rejected_before_submission(prepared):
    manifest = sweep.read_json(prepared / 'manifest.json')
    manifest['source_sha256']['src/workflow_c_sweep.py'] = 'changed'
    sweep.write_json_atomic(prepared / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='Source changed'):
        sweep.submit_sweep(prepared)
    assert not (prepared / 'submission_started.json').exists()


def test_artifact_parameter_mismatch_cannot_rank(prepared):
    trial = sweep.trial_path(prepared, 0)
    (trial / 'result').mkdir()
    (trial / 'result' / sweep.ARTIFACT).write_bytes(result_bytes())
    update_status(trial, 'completed')
    rows = sweep.summarize_sweep(prepared)
    assert rows[0]['validity'] == 'INVALID'
    assert 'parameters do not match' in rows[0]['error']


def test_real_worker_scratch_copy_cleanup_and_no_duplicate_execution(prepared, monkeypatch, tmp_path):
    from src import workflow_c_worker as worker
    monkeypatch.setenv('SLURM_TMPDIR', str(tmp_path / 'scratch'))
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    monkeypatch.setattr(worker, '_require_slurm_cuda', lambda: ('123', {'gpu_model': 'test GPU'}))
    seen = []
    def execute(config, scratch, provenance):
        assert (scratch / 'config.json').exists()
        assert (scratch / 'input').is_dir() and not (scratch / 'input').is_symlink()
        seen.append(config)
        return result_bytes()
    monkeypatch.setattr(worker, 'execute_config', execute)
    artifact = sweep.run_trial(prepared, 2)
    assert artifact.exists()
    assert not (tmp_path / 'scratch/0002').exists()
    assert seen[0]['parameters']['stage_a_learning_rate'] == 0.15
    assert sweep.read_json(prepared / 'trials/0002/status.json')['state'] == 'completed'
    with pytest.raises(ValueError, match='not queued'):
        sweep.run_trial(prepared, 2)
    assert len(seen) == 1


def test_nonfinite_displacement_array_excluded(prepared):
    complete(prepared, 0)
    path = prepared / 'trials/0000/result' / sweep.ARTIFACT
    output = io.BytesIO()
    with zipfile.ZipFile(path) as source, zipfile.ZipFile(output, 'w') as dest:
        for name in source.namelist():
            data = source.read(name)
            if name == 'registration_result.npz':
                with np.load(io.BytesIO(data), allow_pickle=False) as archive:
                    arrays = dict(archive)
                arrays['attempted_displacement_x'][0,0] = np.nan
                values = io.BytesIO()
                np.savez_compressed(values, **arrays)
                data = values.getvalue()
            dest.writestr(name, data)
    path.write_bytes(output.getvalue())
    assert sweep.summarize_sweep(prepared)[0]['validity'] == 'INVALID'
