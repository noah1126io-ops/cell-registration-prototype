from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import io
import json
import math
import platform
import re
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


EXPECTED_ARTIFACTS = (
    "points/affine_he_nuclei.csv",
    "points/attempted_he_nuclei.csv",
    "points/final_applied_he_nuclei.csv",
    "fields/attempted_displacement_field.npz",
    "fields/applied_displacement_field.npz",
    "fields/grid_x.npy",
    "fields/grid_y.npy",
    "images/affine_he.png",
    "images/attempted_he.png",
    "images/final_applied_he.png",
    "images/affine_overlay.png",
    "images/attempted_overlay.png",
    "images/final_overlay.png",
    "images/displacement_magnitude.png",
    "images/local_residual_displacement.png",
    "images/jacobian.png",
    "images/attempted_warp_grid.png",
    "images/applied_warp_grid.png",
    "images/distance_histogram.png",
)


def json_safe(value: Any) -> Any:
    """Convert scientific Python values to strict JSON-compatible values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if isinstance(value, Path):
        return str(value)
    return str(value)


def sanitize_filename_component(value: str, *, fallback: str = "run") -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())
    text = re.sub(r"-{2,}", "-", text)
    text = re.sub(r"_{2,}", "_", text)
    text = re.sub(r"\.{2,}", ".", text).strip("-._")
    return text[:80] or fallback


def generate_run_id(
    label: str,
    *,
    now: datetime | None = None,
    existing_run_ids: Iterable[str] = (),
) -> str:
    current = now or datetime.now().astimezone()
    base = f"workflow_c_{current.strftime('%Y%m%d_%H%M%S')}_{sanitize_filename_component(label)}"
    existing = set(existing_run_ids)
    if base not in existing:
        return base
    suffix = 2
    while f"{base}_{suffix}" in existing:
        suffix += 1
    return f"{base}_{suffix}"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def input_file_record(name: str, data: bytes) -> dict[str, Any]:
    payload = bytes(data)
    return {
        "filename": Path(name).name,
        "size_bytes": len(payload),
        "sha256": sha256_bytes(payload),
        "content": payload,
    }


def flatten_mapping(values: Mapping[str, Any], *, prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in values.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(flatten_mapping(value, prefix=path))
        else:
            flattened[path] = json_safe(value)
    return flattened


def parameter_changes(
    current: Mapping[str, Any],
    previous: Mapping[str, Any] | None,
    *,
    changed_only: bool = True,
) -> list[dict[str, Any]]:
    current_flat = flatten_mapping(current)
    previous_flat = flatten_mapping(previous or {})
    rows = []
    for parameter in sorted(set(current_flat) | set(previous_flat)):
        old = previous_flat.get(parameter)
        new = current_flat.get(parameter)
        changed = old != new
        if changed_only and not changed:
            continue
        if parameter not in previous_flat:
            change_type = "added"
        elif parameter not in current_flat:
            change_type = "removed"
        elif changed:
            change_type = "changed"
        else:
            change_type = "unchanged"
        numeric_delta = None
        numeric_ratio = None
        if (
            isinstance(old, (int, float))
            and not isinstance(old, bool)
            and isinstance(new, (int, float))
            and not isinstance(new, bool)
        ):
            numeric_delta = float(new) - float(old)
            if float(old) != 0.0:
                numeric_ratio = float(new) / float(old)
        rows.append(
            {
                "parameter": parameter,
                "previous_value": old,
                "current_value": new,
                "change_type": change_type,
                "numeric_delta": numeric_delta,
                "numeric_ratio": numeric_ratio,
            }
        )
    return rows


def _csv_bytes(rows: list[Mapping[str, Any]], fieldnames: list[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json_safe(row.get(key)) for key in fieldnames})
    return buffer.getvalue().encode("utf-8")


def parameters_flat_csv(parameters: Mapping[str, Any]) -> bytes:
    rows = [{"parameter": key, "value": value} for key, value in flatten_mapping(parameters).items()]
    return _csv_bytes(rows, ["parameter", "value"])


def parameter_changes_csv(rows: list[Mapping[str, Any]]) -> bytes:
    return _csv_bytes(
        rows,
        ["parameter", "previous_value", "current_value", "change_type", "numeric_delta", "numeric_ratio"],
    )


def metrics_summary_csv(metrics: Mapping[str, Any]) -> bytes:
    rows = [{"metric": key, "value": value} for key, value in flatten_mapping(metrics).items()]
    return _csv_bytes(rows, ["metric", "value"])


def experiment_history_csv(history: list[Mapping[str, Any]]) -> bytes:
    fields = [
        "run_id", "label", "fine_method", "status", "rejection_reason",
        "affine_median", "attempted_median", "delta_median", "final_median",
        "mutual_before", "mutual_attempted", "jacobian_min", "jacobian_max",
        "local_residual_p95", "max_displacement", "timestamp",
    ]
    return _csv_bytes(history, fields)


def git_provenance(repository_path: str | Path) -> dict[str, Any]:
    cwd = str(repository_path)
    result = {"git_commit_sha": None, "git_dirty": None}
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True, timeout=5
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, check=True, timeout=5
        ).stdout
        result = {"git_commit_sha": sha or None, "git_dirty": bool(status.strip())}
    except (OSError, subprocess.SubprocessError):
        pass
    return result


def package_versions(package_names: Iterable[str]) -> dict[str, str | None]:
    versions = {}
    for name in package_names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def default_provenance(repository_path: str | Path = ".") -> dict[str, Any]:
    return {
        **git_provenance(repository_path),
        "python_version": sys.version,
        "platform": platform.platform(),
        "os": platform.system(),
        "package_versions": package_versions(
            ["streamlit", "numpy", "pandas", "opencv-python", "scikit-image", "scipy", "matplotlib", "Pillow", "tifffile"]
        ),
    }


def _run_summary_markdown(
    run_id: str,
    label: str,
    notes: str,
    metrics: Mapping[str, Any],
    changes: list[Mapping[str, Any]],
) -> bytes:
    lines = [
        f"# Workflow C experiment: {label or run_id}",
        "",
        f"- Run ID: `{run_id}`",
        f"- Fine method: `{metrics.get('fine_method')}`",
        f"- Status: `{metrics.get('fine_status')}`",
        f"- Rejection reason: `{metrics.get('rejection_reason') or 'none'}`",
        "",
        "## Notes",
        "",
        notes.strip() or "No notes provided.",
        "",
        "## Key metrics",
        "",
    ]
    for key, value in flatten_mapping(metrics).items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Changes from previous run", ""])
    if changes:
        for row in changes:
            lines.append(
                f"- `{row['parameter']}`: `{row['previous_value']}` -> `{row['current_value']}`"
            )
    else:
        lines.append("No changed parameters or no comparison run was available.")
    lines.extend(["", "Research prototype only. Not for diagnostic use.", ""])
    return "\n".join(lines).encode("utf-8")


def build_workflow_c_run_bundle(
    *,
    run_id: str,
    label: str,
    notes: str,
    parameters: Mapping[str, Any],
    metrics: Mapping[str, Any],
    artifacts: Mapping[str, bytes | None],
    optimization_history: list[Mapping[str, Any]] | None = None,
    previous_parameters: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    input_files: list[Mapping[str, Any]] | None = None,
    include_original_inputs: bool = False,
    now_utc: datetime | None = None,
    now_local: datetime | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """Build one self-contained, non-pickle Workflow C experiment bundle."""
    utc_timestamp = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
    local_timestamp = now_local or datetime.now().astimezone()
    safe_parameters = json_safe(parameters)
    safe_metrics = json_safe(metrics)
    changes = parameter_changes(safe_parameters, previous_parameters)
    files: dict[str, bytes] = {
        "run_summary.md": _run_summary_markdown(run_id, label, notes, safe_metrics, changes),
        "parameters.json": json.dumps(safe_parameters, indent=2, allow_nan=False).encode("utf-8"),
        "parameters_flat.csv": parameters_flat_csv(safe_parameters),
        "parameter_changes.csv": parameter_changes_csv(changes),
        "metrics_summary.json": json.dumps(safe_metrics, indent=2, allow_nan=False).encode("utf-8"),
        "metrics_summary.csv": metrics_summary_csv(safe_metrics),
    }
    history_rows = json_safe(optimization_history or [])
    if history_rows:
        history_fields = sorted({key for row in history_rows for key in row})
        files["optimization_history.csv"] = _csv_bytes(history_rows, history_fields)
    for path, payload in artifacts.items():
        if payload is not None:
            files[path.replace("\\", "/").lstrip("/")] = bytes(payload)

    sanitized_inputs = []
    for item in input_files or []:
        record = {
            "filename": Path(str(item.get("filename", "input"))).name,
            "size_bytes": int(item.get("size_bytes", 0)),
            "sha256": item.get("sha256"),
        }
        sanitized_inputs.append(record)
        if include_original_inputs and item.get("content") is not None:
            safe_name = sanitize_filename_component(record["filename"], fallback="input")
            candidate = f"inputs/{safe_name}"
            suffix = 2
            while candidate in files:
                stem = Path(safe_name).stem
                extension = Path(safe_name).suffix
                candidate = f"inputs/{stem}_{suffix}{extension}"
                suffix += 1
            files[candidate] = bytes(item["content"])

    expected = {
        "run_summary.md", "parameters.json", "parameters_flat.csv", "parameter_changes.csv",
        "metrics_summary.json", "metrics_summary.csv", "optimization_history.csv", *EXPECTED_ARTIFACTS,
    }
    provenance_values = {
        "git_commit_sha": None,
        "git_dirty": None,
        "python_version": None,
        "platform": None,
        "os": None,
        "package_versions": {},
        "array_shapes": {},
        "point_counts": {},
        "coordinate_conventions": {},
        **json_safe(provenance or {}),
    }
    manifest = {
        "run_id": run_id,
        "label": label,
        "utc_timestamp": utc_timestamp.isoformat(),
        "local_timestamp": local_timestamp.isoformat(),
        **provenance_values,
        "input_files": sanitized_inputs,
        "include_original_inputs": bool(include_original_inputs),
        "included_files": sorted([*files, "manifest.json"]),
        "skipped_unavailable_files": sorted(expected - set(files)),
    }
    files["manifest.json"] = json.dumps(json_safe(manifest), indent=2, allow_nan=False).encode("utf-8")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.writestr(path, files[path])
    return buffer.getvalue(), manifest
