"""Registration Explorer research jobs on Slurm (reuses the Workflow C run-directory/status conventions).

Guards: identical configs reuse a finished result (cache), a limit on concurrently active explorer jobs,
and cancel only acts on a job ID recorded in our own run directory after checking owner and job name.
"""
from __future__ import annotations

import getpass
import hashlib
import json
import os
import subprocess
import uuid
from datetime import datetime
from pathlib import Path

from src.workflow_c_slurm import read_run_status, update_status, utc_now, write_json_atomic

REPO = Path(__file__).resolve().parents[2]
RUNS_DIR = Path(os.environ.get("CELLREG_EXPLORER_RUNS_DIR", str(REPO / "runs/registration-explorer")))
SBATCH = REPO / "slurm/registration_explorer_cpu.sbatch"
JOB_NAME = "cellreg-explorer"
MAX_ACTIVE = int(os.environ.get("CELLREG_EXPLORER_MAX_ACTIVE", "2"))
# research Workflow C (shared engine src/registration_core): what the UI offers for new runs
WORKFLOWS = {"auto": "自動：画像が使えれば画像で対応領域を探索し、核探索と並べて検証（推奨）",
             "points": "自動：核点群のみで探索（画像を使わない）",
             "a2": "自動：核密度による対応領域探索（従来法 A2）",
             "assisted": "注釈を使った探索（ROI・除外・初期位置・対応点。自動採択しない）"}
# legacy ablation variants (2026-10-20 runner, kept so old runs stay reproducible)
RUNNABLE = {"X3": "E0x (X3: union canvas + cropped ICP)", "X0": "X0 (no canvas extension, full ICP)", "X1": "X1 (union canvas, full ICP)",
            "X2": "X2 (bbox canvas, cropped ICP)", "A2": "Legacy A2 (existing method_a2 + A2 detector)"}
# image representation compared by the image search (registration_core.features); first = default for new runs
IMAGE_FEATURES = {"occ+cleft": "組織の占有率と間隙の網目（核を除く、2026-10-27〜）",
                  "tissue+nuclear": "従来：tissue＋nuclear（2026-10-22。IFのtissueは核シグナルが支配的）"}
ENGINE = "registration_core"
ENGINE_SLURM = {"time": "01:30:00", "mem": "12G"}


def config_key(cfg: dict) -> str:
    if cfg.get("engine") == ENGINE:
        core = {k: cfg.get(k) for k in ("engine", "sample", "fixed", "localization", "use_images", "annotation_ids", "constraints")}
        if cfg.get("image_feature"):      # absent in runs before 2026-10-27 -> their keys are unchanged
            core["image_feature"] = cfg["image_feature"]
    else:
        core = {k: cfg[k] for k in ("sample", "method", "variant", "fixed")}
    return hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()[:16]


def list_runs(runs_dir: Path = RUNS_DIR) -> list[dict]:
    out = []
    if not runs_dir.is_dir():
        return out
    for d in runs_dir.iterdir():
        if not (d / "status.json").is_file():
            continue
        try:
            st = read_run_status(d)
            cfg = json.loads((d / "config.json").read_text())
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        out.append({"run_id": d.name, "run_dir": str(d), "config": cfg, **st})
    return sorted(out, key=lambda r: r.get("updated_at", ""), reverse=True)


def cached_run(cfg: dict, runs_dir: Path = RUNS_DIR):
    key = config_key(cfg)
    for r in list_runs(runs_dir):
        if r["config"].get("key") == key and r.get("state") == "completed" and (Path(r["run_dir"]) / "result.json").is_file():
            return r
    return None


def active_same(cfg: dict, runs_dir: Path = RUNS_DIR):
    key = config_key(cfg)
    return next((r for r in active_runs(runs_dir) if r["config"].get("key") == key), None)


def active_runs(runs_dir: Path = RUNS_DIR):
    return [r for r in list_runs(runs_dir) if r.get("state") in ("queued", "running")]


def validate(cfg: dict, sample_info) -> list[str]:
    errs = []
    if cfg.get("engine") == ENGINE:
        if cfg.get("localization") not in ("auto", "points", "a2", "image"):
            errs.append(f"unknown localization {cfg.get('localization')}")
        if cfg["fixed"] not in sample_info.modalities:
            errs.append("fixed modality not in sample")
        for k, m in sample_info.modalities.items():
            if not (m.nuclei and m.nuclei.get("path") and Path(m.nuclei["path"]).is_file()):
                errs.append(f"{k}: nuclei file missing")
            if cfg.get("localization") != "a2" and m.extent() is None:
                errs.append(f"{k}: imaged extent UNKNOWN -> only the A2 search is applicable")
        if cfg.get("mode") == "assisted" and not cfg.get("annotation_ids"):
            errs.append("assisted run needs at least one selected annotation")
        return errs
    if cfg["variant"] not in RUNNABLE:
        errs.append(f"unknown method {cfg['variant']}")
    if cfg["fixed"] not in sample_info.modalities:
        errs.append("fixed modality not in sample")
    for k, m in sample_info.modalities.items():
        if not (m.nuclei and m.nuclei.get("path") and Path(m.nuclei["path"]).is_file()):
            errs.append(f"{k}: nuclei file missing")
    if cfg["variant"] != "A2":
        for k, m in sample_info.modalities.items():
            if m.pixel_um is None or m.extent() is None:
                errs.append(f"{k}: pixel spacing / imaged extent UNKNOWN -> E0x scale grid (physical units) not applicable")
    return errs


def submit(cfg: dict, manifest: Path, runs_dir: Path = RUNS_DIR, *, dry_run: bool = False):
    """Submit one job; returns (run_dir, job_id).  Raises if the active-job limit is reached.
    Engine runs (cfg["engine"] == "registration_core") write artifact.json (cellreg-registration-artifact/1)."""
    if len(active_runs(runs_dir)) >= MAX_ACTIVE:
        raise RuntimeError(f"active explorer jobs >= {MAX_ACTIVE}; wait or cancel one")
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    run_dir = runs_dir.resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    full = {**cfg, "key": config_key(cfg), "manifest": str(Path(manifest).resolve()), "created_by": getpass.getuser(), "research_mode": True}
    write_json_atomic(run_dir / "config.json", full)
    update_status(run_dir, "queued")
    extra = [f"--time={ENGINE_SLURM['time']}", f"--mem={ENGINE_SLURM['mem']}"] if cfg.get("engine") == ENGINE else []
    cmd = ["sbatch", "--parsable", f"--job-name={JOB_NAME}", f"--chdir={REPO}", f"--output={run_dir / 'stdout.log'}",
           f"--error={run_dir / 'stderr.log'}", *extra, str(SBATCH), str(run_dir)]
    if dry_run:
        write_json_atomic(run_dir / "slurm_job.json", {"job_id": None, "dry_run": True, "command": cmd})
        return run_dir, None
    try:
        done = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=60)
        job_id = done.stdout.strip().split(";", 1)[0]
        if not job_id.isdigit():
            raise RuntimeError(f"unexpected sbatch output {done.stdout!r}")
        write_json_atomic(run_dir / "slurm_job.json", {"job_id": job_id, "submitted_at": utc_now(), "command": cmd, "partition": "mib-dbia",
                                                       "cpus_per_task": 4, "memory": ENGINE_SLURM["mem"] if extra else "6G",
                                                       "time": ENGINE_SLURM["time"] if extra else "00:45:00"})
        update_status(run_dir, "queued", slurm_job_id=job_id)
        return run_dir, job_id
    except BaseException as exc:
        update_status(run_dir, "failed", error=f"sbatch submission failed: {exc}")
        raise


def slurm_state(job_id: str) -> str | None:
    try:
        out = subprocess.run(["squeue", "-h", "-j", str(job_id), "-o", "%T|%u|%j"], capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out or None


def refresh(run: dict) -> dict:
    """Reconcile status.json with Slurm for queued/running jobs that vanished from the queue without a result."""
    d = Path(run["run_dir"])
    st = read_run_status(d)
    jid = st.get("slurm_job_id")
    done = (d / "result.json").is_file() or (d / "artifact.json").is_file()
    if st["state"] in ("queued", "running") and jid and slurm_state(jid) is None and not done:
        # give the worker's own status write precedence; mark only if it never reported
        err = (d / "stderr.log").read_text(errors="replace")[-2000:] if (d / "stderr.log").is_file() else ""
        st = update_status(d, "failed", error="job left the Slurm queue without a result", stderr_tail=err)
    return {**run, **st}


def cancel(run_dir: Path, runs_dir: Path = RUNS_DIR) -> str:
    run_dir = Path(run_dir).resolve()
    if run_dir.parent != runs_dir.resolve():
        raise ValueError("not an explorer run directory")
    st = read_run_status(run_dir)
    jid = st.get("slurm_job_id")
    if not jid or st["state"] not in ("queued", "running"):
        return "nothing to cancel"
    q = slurm_state(jid)
    if q is None:
        return "job is no longer in the queue"
    _, user, name = q.split("|")
    if user != getpass.getuser() or name != JOB_NAME:
        raise PermissionError(f"refusing to cancel job {jid} (owner {user}, name {name})")
    subprocess.run(["scancel", str(jid)], check=True, timeout=30)
    update_status(run_dir, "failed", error="cancelled by user from the Registration Explorer")
    return f"cancelled job {jid}"


def read_log(run_dir: Path, name: str, tail=4000) -> str:
    p = Path(run_dir) / name
    return p.read_text(errors="replace")[-tail:] if p.is_file() else ""
