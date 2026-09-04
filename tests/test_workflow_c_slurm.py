import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.workflow_c_slurm import read_run_status, submit_workflow_c, update_status
from src.workflow_c_worker import _require_slurm_cuda


def test_status_records_queued_running_completed_history(tmp_path):
    update_status(tmp_path, "queued")
    update_status(tmp_path, "running", slurm_job_id="123")
    status = update_status(tmp_path, "completed", artifact="result/result.zip")

    assert status["state"] == "completed"
    assert [row["state"] for row in status["history"]] == ["queued", "running", "completed"]
    assert read_run_status(tmp_path) == status


def test_submit_stages_inputs_and_records_slurm_job(monkeypatch, tmp_path):
    fixed = tmp_path / "fixed.npy"
    np.save(fixed, np.array([[1.0, 2.0]]))
    config = tmp_path / "source.json"
    config.write_text(json.dumps({"inputs": {"fixed_points": str(fixed), "moving_points": str(fixed)}}))
    monkeypatch.setattr(
        "src.workflow_c_slurm.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout="98765;cluster\n"),
    )

    run_dir, job_id = submit_workflow_c(config, tmp_path / "runs", run_id="case-1")

    assert job_id == "98765"
    assert (run_dir / "input" / "fixed_points.npy").is_file()
    assert (run_dir / "input" / "moving_points.npy").is_file()
    assert (run_dir / "result").is_dir()
    staged = json.loads((run_dir / "config.json").read_text())
    assert staged["inputs"]["fixed_points"] == "input/fixed_points.npy"
    assert read_run_status(run_dir)["state"] == "queued"
    slurm = json.loads((run_dir / "slurm_job.json").read_text())
    assert slurm["partition"] == "mib-dbia"
    assert slurm["gres"] == "gpu:1"
    assert slurm["nodes"] == slurm["ntasks"] == 1
    assert slurm["cpus_per_task"] == 4
    assert slurm["memory"] == "16G"
    assert slurm["time"] == "00:30:00"


def test_cuda_worker_fails_clearly_without_slurm_or_gpu(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(RuntimeError, match="must run inside a Slurm allocation"):
        _require_slurm_cuda()

    monkeypatch.setenv("SLURM_JOB_ID", "123")
    with pytest.raises(RuntimeError, match="has no allocated GPU"):
        _require_slurm_cuda()


def test_batch_script_requests_one_gpu_and_does_not_start_streamlit():
    source = (Path(__file__).parents[1] / "slurm" / "workflow_c_gpu.sbatch").read_text()
    assert "#SBATCH --partition=mib-dbia" in source
    assert "#SBATCH --gres=gpu:1" in source
    assert "#SBATCH --nodes=1" in source
    assert "#SBATCH --ntasks=1" in source
    assert "scripts/run_workflow_c.py" in source
    assert 'RUN_DIR/../..' in source
    assert "trap mark_launch_failed ERR" in source
    assert "streamlit" not in source.lower()
