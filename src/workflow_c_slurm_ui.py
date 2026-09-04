from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.workflow_c_result import load_workflow_c_result_artifact
from src.workflow_c_slurm import read_run_status


ACTIVE_RUN_KEY = "workflow-c-slurm-active-run"


def safe_run_dir(runs_dir: Path, run_id: str) -> Path:
    root = Path(runs_dir).resolve()
    candidate = (root / run_id).resolve()
    if candidate.parent != root or not run_id or Path(run_id).name != run_id:
        raise ValueError("Run ID must name a direct child of the Workflow C runs directory.")
    return candidate


def recent_runs(runs_dir: Path, limit: int = 15) -> list[dict[str, Any]]:
    root = Path(runs_dir).resolve()
    if not root.is_dir():
        return []
    records = []
    for child in root.iterdir():
        if not child.is_dir() or not (child / "status.json").is_file():
            continue
        try:
            status = read_run_status(child)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        records.append({"run_id": child.name, "run_dir": str(child), **status})
    return sorted(records, key=lambda row: row.get("updated_at", ""), reverse=True)[:limit]


def load_run_view(runs_dir: Path, run_id: str) -> dict[str, Any]:
    run_dir = safe_run_dir(runs_dir, run_id)
    status = read_run_status(run_dir)
    view = {"run_id": run_id, "run_dir": str(run_dir), "status": status, "state": status["state"]}
    if status["state"] == "completed":
        artifact_path = run_dir / "result" / "workflow_c_registration_result.zip"
        if not artifact_path.is_file():
            view["artifact_error"] = "Completed job result artifact is missing."
        else:
            try:
                payload = artifact_path.read_bytes()
                artifact = load_workflow_c_result_artifact(payload)
                view.update(artifact_bytes=payload, artifact=artifact)
            except (OSError, ValueError) as exc:
                view["artifact_error"] = f"Completed result artifact is invalid: {exc}"
    if status["state"] == "failed":
        for name in ("stdout.log", "stderr.log"):
            path = run_dir / name
            view[name] = path.read_text(encoding="utf-8", errors="replace")[-4000:] if path.is_file() else ""
    return view


def activate_run(session_state, run_dir: Path, job_id: str) -> None:
    session_state[ACTIVE_RUN_KEY] = {
        "run_id": run_dir.name, "run_dir": str(run_dir.resolve()), "job_id": str(job_id),
    }
