from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VALID_STATES = {"queued", "running", "completed", "failed"}
INPUT_KEYS = {"fixed_points", "moving_points", "he_image", "tissue_mask", "metadata"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def update_status(run_dir: Path, state: str, **details: Any) -> dict[str, Any]:
    if state not in VALID_STATES:
        raise ValueError(f"Invalid Workflow C job state: {state}")
    path = run_dir / "status.json"
    previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    updated_at = utc_now()
    history = list(previous.get("history", []))
    if not history or history[-1].get("state") != state:
        history.append({"state": state, "at": updated_at})
    payload = {**previous, **details, "state": state, "updated_at": updated_at, "history": history}
    payload.setdefault("run_id", run_dir.name)
    payload.setdefault("created_at", payload["updated_at"])
    write_json_atomic(path, payload)
    return payload


def read_run_status(run_dir: Path) -> dict[str, Any]:
    return json.loads((Path(run_dir) / "status.json").read_text(encoding="utf-8"))


def stage_config(source: Path, run_dir: Path) -> dict[str, Any]:
    config = json.loads(source.read_text(encoding="utf-8"))
    inputs = dict(config.get("inputs", {}))
    input_dir = run_dir / "input"
    for key, value in inputs.items():
        if key not in INPUT_KEYS:
            raise ValueError(f"Unsupported input key: {key}")
        source_path = Path(value).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        input_dir.mkdir(parents=True, exist_ok=True)
        destination = input_dir / f"{key}{source_path.suffix}"
        shutil.copy2(source_path, destination)
        inputs[key] = str(destination.relative_to(run_dir))
    config["inputs"] = inputs
    write_json_atomic(run_dir / "config.json", config)
    return config


def submit_workflow_c(config_path: Path, runs_dir: Path, *, run_id: str | None = None) -> tuple[Path, str]:
    run_id = run_id or f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}"
    run_dir = Path(runs_dir).resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "result").mkdir()
    stage_config(Path(config_path).resolve(), run_dir)
    update_status(run_dir, "queued")
    batch_script = Path(__file__).resolve().parents[1] / "slurm" / "workflow_c_gpu.sbatch"
    command = [
        "sbatch", "--parsable",
        f"--output={run_dir / 'stdout.log'}",
        f"--error={run_dir / 'stderr.log'}",
        str(batch_script), str(run_dir / "config.json"),
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        job_id = completed.stdout.strip().split(";", 1)[0]
        if not job_id.isdigit():
            raise RuntimeError(f"Unexpected sbatch job id: {completed.stdout!r}")
        write_json_atomic(run_dir / "slurm_job.json", {
            "job_id": job_id, "submitted_at": utc_now(), "command": command,
            "partition": "mib-dbia", "gres": "gpu:1", "nodes": 1,
            "ntasks": 1, "cpus_per_task": 4, "memory": "16G", "time": "00:30:00",
        })
        update_status(run_dir, "queued", slurm_job_id=job_id)
        return run_dir, job_id
    except BaseException as exc:
        update_status(run_dir, "failed", error=f"sbatch submission failed: {exc}")
        raise
