"""Phase A/B preparation (run on Slurm, read-only on originals and on the research store).

  python -B prep.py features      engine feature extraction on the original images -> cache/feat_*.npz (+ parity vs 10-27)
  python -B prep.py manifest      baseline_manifest.json (engine freeze: code hashes, configs, rules, input fingerprints)
  python -B prep.py cases         reference poses + derived case geometry -> cases.json (written BEFORE any engine run)
"""
from __future__ import annotations

import inspect
import json
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from common import (ACCEPTED_6_2, CACHE, CORRECT_UM, HERE, REPO, RES_UM, DerivedSide, feat_path, landmarks_8_6, load_feats,
                    pose_params, save_feats, sha256_file, store_transform)
from src.registration_core import workflow as W
from src.registration_core.features import Modality, extract
from src.registration_core.pipeline import ACCEPT, TRUST
from src.registration_core.reference import fit_reference
from src.registration_core.search import SearchConfig
from src.registration_explorer import catalog

SAMPLES = ("031-6_2", "031-8_6")
OLD = REPO / "analysis/real-image-localization-failure-audit-20261027/cache"


def features():
    man = catalog.load_manifest()
    par = {}
    for s in SAMPLES:
        for k in ("HE", "IF"):
            si = catalog.side_input(man[s], k)
            t0 = time.time()
            fm = extract(Modality(si.frame(), np.asarray(si.points, float)), res_um=RES_UM)
            CACHE.mkdir(exist_ok=True)
            save_feats(fm, feat_path(s, k))
            old = np.load(OLD / f"feat_{s}_{k}_2.npz", allow_pickle=True)
            par[f"{s}:{k}"] = {m: float(np.abs(fm.maps[m].astype(float) - old[f"map_{m}"].astype(float)).max()) for m in fm.maps}
            par[f"{s}:{k}"]["runtime_s"] = time.time() - t0
            print(s, k, par[f"{s}:{k}"], flush=True)
    (CACHE / "feature_parity_vs_20261027.json").write_text(json.dumps(par, indent=1))


def _git(*a):
    return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout


def manifest():
    files = sorted([*(REPO / "src/registration_core").glob("*.py"), *(REPO / "src/registration_explorer").rglob("*.py"),
                    REPO / "src/workflow_c_research.py", REPO / "src/research_store.py", REPO / "app.py"])
    man = catalog.load_manifest()
    inputs = {}
    for s in SAMPLES:
        for k in ("HE", "IF"):
            m = man[s].modality(k)
            inputs[f"{s}:{k}"] = {"image_path": m.image_path, "image_sha256": sha256_file(m.image_path), "image_bytes": int(Path(m.image_path).stat().st_size),
                                  "nuclei_path": m.nuclei["path"], "nuclei_sha256": sha256_file(m.nuclei["path"]), "pixel_um": m.pixel_um,
                                  "shape_px": list(m.shape_px), "feature_npz_sha256": sha256_file(feat_path(s, k))}
    out = {
        "created_at": datetime.now(timezone.utc).isoformat(), "purpose": "engine freeze before external validation (Phase A)",
        "git": {"branch": _git("branch", "--show-current").strip(), "head": _git("rev-parse", "HEAD").strip(),
                "status_porcelain": _git("status", "--porcelain").splitlines(),
                "tracked_diff_sha256": sha256_file(CACHE / "baseline_tracked.diff"), "tracked_diff_file": "cache/baseline_tracked.diff"},
        "code_sha256": {str(p.relative_to(REPO)): sha256_file(p) for p in files},
        "core_version": W.CORE_VERSION, "artifact_schema": W.SCHEMA,
        "workflow_config_defaults": json.loads(json.dumps(asdict(W.WorkflowConfig()), default=list)),
        "search_config_defaults": json.loads(json.dumps(asdict(SearchConfig()), default=list)),
        "feature_extraction": {"features": ["tissue", "nuclear", "edt", "occ", "cleft"], "image_feature_default": W.WorkflowConfig().image_feature,
                               "extract_defaults": {k: v.default for k, v in inspect.signature(extract).parameters.items() if v.default is not inspect._empty},
                               "res_um": RES_UM, "pyramid_level_rule": "largest level with ds*pixel_um <= 0.75*res_um"},
        "icp": {"method": "cropped_icp (overlap-restricted, similarity -> affine)", "trust_region_image": TRUST,
                "relocation_flag_um": SearchConfig().distinct_um, "adjacent": "diagnostic only (search pose kept)"},
        "c3_rules": {"same_or_unknown": {"ACCEPT": ACCEPT, "search_verdict": "UNIQUE", "frames": "fixed and IF frame"},
                     "adjacent": {"QUADRANT_RULE": json.loads(json.dumps(W.QUADRANT_RULE, default=list)), "status": "TISSUE_LEVEL, accepted=False"}},
        "candidates": {"n_candidates_refined": W.WorkflowConfig().n_candidates, "n_fine": SearchConfig().n_fine},
        "section_relation_inference": inspect.getsource(W.infer_section_relation),
        "inputs": inputs,
        "correct_region_threshold_um": CORRECT_UM,
        "rule_of_this_study": "no default parameter or threshold is changed after seeing results; any variant is a separate run",
    }
    (HERE / "baseline_manifest.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str))
    print("manifest written")


def _footprint_samples(ext, n=41):
    xs, ys = np.linspace(0, ext[0], n), np.linspace(0, ext[1], n)
    return np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)


def cases():
    ref = {}
    L = landmarks_8_6()
    f = fit_reference([l["fixed"] for l in L], [l["moving"] for l in L])
    ref["031-8_6"] = {"M": np.asarray(f["M"]).tolist(), "t": np.asarray(f["t"]).tolist(), "source": "landmark fit (8 pairs, user 2026-10-27; LOO median 23.7 um)",
                      "landmarks": L}
    M, t = store_transform(ACCEPTED_6_2)
    ref["031-6_2"] = {"M": M.tolist(), "t": t.tolist(), "source": f"accepted automatic transform {ACCEPTED_6_2} (not independent)"}
    for s in ref:
        ref[s]["params"] = pose_params(ref[s]["M"])
    out = []

    def add(cid, he, if_, he_ops=(), if_ops=(), expected="POSITIVE", ref_sample=None, kind="", note=""):
        out.append({"case_id": cid, "he_sample": he, "if_sample": if_, "he_ops": [list(o) for o in he_ops], "if_ops": [list(o) for o in if_ops],
                    "expected": expected, "reference_sample": ref_sample, "kind": kind, "note": note})

    for s in ("031-8_6", "031-6_2"):
        tag = s[4:]
        he = DerivedSide(s, "HE"); mv = DerivedSide(s, "IF")
        Mr, tr = np.asarray(ref[s]["M"]), np.asarray(ref[s]["t"])
        W_if, H_if = mv.extent
        add(f"P{tag}_orig", s, s, ref_sample=s, kind="original pair (development data)")
        add(f"P{tag}_IFhalfL", s, s, if_ops=[("crop", 0, 0, 512, H_if)], ref_sample=s, kind="IF field = left half (FOV ratio x2)")
        add(f"P{tag}_IF500", s, s, if_ops=[("crop", 262, 262, 500, 500)], ref_sample=s, kind="IF field = central 500x500 um (FOV ratio ~18)")
        # HE crop so that ~60 % of the IF footprint stays inside the HE field
        fp = _footprint_samples(mv.extent) @ Mr.T + tr
        xs = np.sort(fp[:, 0])
        x_cut = float(np.floor(xs[int(0.6 * len(xs))] / RES_UM) * RES_UM)
        x0 = float(max(0.0, np.floor((fp[:, 0].min() - 1400) / RES_UM) * RES_UM))
        frac = float((fp[:, 0] < x_cut).mean())
        add(f"P{tag}_HEpartial60", s, s, he_ops=[("crop", x0, 0, x_cut - x0, he.extent[1])], ref_sample=s,
            kind=f"HE cropped: {frac:.2f} of the IF footprint inside the HE field (partial overlap)")
        add(f"N{tag}_IFmirror", s, s, if_ops=[("mirror_x",)], expected="NEGATIVE", ref_sample=None,
            kind="IF mirrored (search is proper-only: no valid pose exists)")
        # HE strip that excludes the reference footprint (+60 um margin): same tissue, non-corresponding region
        bx0, by0 = fp.min(0) - 60; bx1, by1 = fp.max(0) + 60
        We, He = he.extent
        strips = {"left": (0, 0, bx0, He), "right": (bx1, 0, We - bx1, He), "top": (0, 0, We, by0), "bottom": (0, by1, We, He - by1)}
        strips = {k: tuple(float(np.floor(v / RES_UM) * RES_UM) for v in s4) for k, s4 in strips.items() if s4[2] > 0 and s4[3] > 0}
        best = max(strips, key=lambda k: min(strips[k][2], strips[k][3]))
        x, y, w, h = strips[best]
        add(f"N{tag}_HEexclude", s, s, he_ops=[("crop", x, y, w, h)], expected="NEGATIVE",
            kind=f"HE strip '{best}' {w:.0f}x{h:.0f} um that excludes the reference footprint (same tissue, wrong region)")
        add(f"N{tag}_IF500xHEexclude", s, s, he_ops=[("crop", x, y, w, h)], if_ops=[("crop", 262, 262, 500, 500)], expected="NEGATIVE",
            kind=f"IF central 500x500 um vs HE strip '{best}' without the reference region (full overlap possible, wrong region)")
    add("Ncross_HE6_2xIF8_6", "031-6_2", "031-8_6", expected="PRESUMED_NEGATIVE",
        kind="HE 031-6 XY02 x IF 031-8_6 (different stage positions; correspondence not verified)")
    add("Ncross_HE8_6xIF6_2", "031-8_6", "031-6_2", expected="PRESUMED_NEGATIVE",
        kind="HE 031-7 XY06 x IF 031-6_2 (different stage positions; correspondence not verified)")
    # carry the reference into each derived frame
    for c in out:
        c["reference"] = None
        if c["reference_sample"]:
            he = DerivedSide(c["he_sample"], "HE"); mv = DerivedSide(c["if_sample"], "IF")
            for o in c["he_ops"]:
                he.apply(o)
            for o in c["if_ops"]:
                mv.apply(o)
            from common import carry_reference
            M2, t2 = carry_reference(np.asarray(ref[c["reference_sample"]]["M"]), np.asarray(ref[c["reference_sample"]]["t"]), he, mv)
            c["reference"] = {"M": M2.tolist(), "t": t2.tolist(), "source": ref[c["reference_sample"]]["source"]}
            if c["reference_sample"] == "031-8_6":   # landmarks in the derived frames (only those inside both fields)
                lm = []
                for l in ref["031-8_6"]["landmarks"]:
                    fx = he.A @ np.asarray(l["fixed"], float) + he.b
                    mx = mv.A @ np.asarray(l["moving"], float) + mv.b
                    if (0 <= fx[0] <= he.extent[0] and 0 <= fx[1] <= he.extent[1] and 0 <= mx[0] <= mv.extent[0] and 0 <= mx[1] <= mv.extent[1]):
                        lm.append({"fixed": fx.tolist(), "moving": mx.tolist()})
                c["reference"]["landmarks"] = lm
        print(c["case_id"], c["kind"], flush=True)
    (HERE / "cases.json").write_text(json.dumps({"created_at": datetime.now(timezone.utc).isoformat(),
                                                 "note": "written before any engine run of this study; derived from development data (NOT independent)",
                                                 "correct_region_threshold_um": CORRECT_UM, "references": {k: {kk: vv for kk, vv in v.items() if kk != "landmarks"} for k, v in ref.items()},
                                                 "cases": out}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    {"features": features, "manifest": manifest, "cases": cases}[sys.argv[1]]()
