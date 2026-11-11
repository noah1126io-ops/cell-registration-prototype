"""Data Profiler prototype (RESEARCH MODE ONLY; not imported by production code).

Measures, BEFORE any global search, what a pair offers to each registration strategy.  Everything here uses only
one side at a time (or metadata) - nothing depends on a candidate pose, so the profile cannot be circular with the
search or the validation.  Post-search quantities (margin, overlap, quadrant agreement, nuclear proximity) belong to
the validation layer, see data_profiler_spec.md.

  python -B data_profiler.py            -> profiles.csv / profiles.json for every case of cases.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom

PROFILE_VERSION = "data_profiler 0.1 (2026-11-11, research prototype)"


# ----------------------------------------------------------------------------- side-level metrics
def _coarse(a, res_um, to_um=8.0):
    return zoom(np.asarray(a, np.float32), res_um / to_um, order=1, prefilter=False)


def repetitiveness(cleft, fov, res_um, exclude_um=80.0, to_um=8.0):
    """Self-similarity of the cleft map: highest normalised autocorrelation at shifts > exclude_um (1 = perfectly periodic,
    ~0 = unique structure).  A high value warns that a quarter of the field / a wrong region can look alike."""
    x = _coarse(cleft, res_um, to_um)
    m = _coarse(fov, res_um, to_um) > 0.5
    x = np.where(m, x - x[m].mean(), 0.0)
    H, W = x.shape
    F = np.fft.rfft2(x, s=(2 * H, 2 * W))
    ac = np.fft.irfft2(F * np.conj(F), s=(2 * H, 2 * W))
    ac = np.fft.fftshift(ac)
    c = np.array(ac.shape) // 2
    ac = ac / max(ac[c[0], c[1]], 1e-12)
    yy, xx = np.indices(ac.shape)
    r = np.hypot(yy - c[0], xx - c[1]) * to_um
    valid = (r > exclude_um) & (r < 0.5 * min(H, W) * to_um)
    return float(ac[valid].max()) if valid.any() else float("nan")


def side_profile(fm, points, extent, pixel_um):
    q = fm.quality or {}
    occ, cleft, nuc, tb, env = (fm.maps[k] for k in ("occ", "cleft", "nuclear", "tissue_bin", "envelope"))
    area_mm2 = float(np.prod(extent) / 1e6)
    in_t = tb > 0
    cl = cleft - 1.0                                                     # band-passed cleft signal (offset removed)
    occ_nuc_corr = float(np.corrcoef(_coarse(occ, fm.res_um).ravel(), _coarse(nuc, fm.res_um).ravel())[0, 1])
    return {
        "pixel_um": pixel_um, "extent_um": [float(e) for e in extent], "area_mm2": area_mm2,
        "image_usable": bool(q.get("usable")), "tissue_fraction": float(in_t.mean()), "envelope_fraction": float(env.mean()),
        "boundary_info": float(1.0 - env.mean()) if env.mean() < 0.98 else 0.0,     # field fully inside tissue -> no outline
        "occ_mean": float(occ.mean()), "cleft_strength": float(np.std(cl[env > 0])) if (env > 0).any() else 0.0,
        "occ_nuclear_corr": occ_nuc_corr,                                  # high -> "tissue" channel is really nuclear
        "repetitiveness": repetitiveness(cleft, fm.fov, fm.res_um),
        "n_nuclei": int(len(points)), "nuclei_per_mm2": float(len(points) / max(area_mm2, 1e-9)),
        "nuclear_map_auc": q.get("nuclear_map_auc_own_nuclei_vs_tissue"), "nuclei_in_tissue_fraction": q.get("nuclei_in_tissue_fraction"),
        "points_inside_extent_fraction": float(np.mean((points[:, 0] >= 0) & (points[:, 0] <= extent[0]) & (points[:, 1] >= 0) & (points[:, 1] <= extent[1]))) if len(points) else 0.0,
    }


# ----------------------------------------------------------------------------- pair-level
def relation_evidence(metadata_relation: str | None, he_name: str, if_name: str, sample_id: str | None):
    """Section relation from metadata (authoritative when present) vs file names (a hint only).  Never infers SAME
    from nuclear agreement (that is a post-search observation, kept separate)."""
    meta = (metadata_relation or "UNKNOWN").split()[0].upper()
    hint = "UNKNOWN"
    h = re.search(r"_HE_(?:reference_)?(\d+-\d+)_", he_name or "")
    m = re.match(r"(\d+-\d+)_", (sample_id or "") + "_")
    if h and m and (if_name or "").startswith(sample_id or "\0"):
        hint = "SAME" if h.group(1) == m.group(1) else "DIFFERENT"
    conflict = meta != "UNKNOWN" and hint != "UNKNOWN" and meta != hint
    value = meta if meta != "UNKNOWN" else "UNKNOWN"
    conf = "CONFIRMED_METADATA" if meta != "UNKNOWN" and not conflict else ("CONFLICT" if conflict else ("FILENAME_HINT_ONLY" if hint != "UNKNOWN" else "NONE"))
    return {"metadata": meta, "filename_hint": hint, "value": value, "confidence": conf, "conflict": conflict}


def pair_profile(fixed_fm, moving_fm, fixed_in, moving_in, metadata_relation=None, sample_id=None):
    f = side_profile(fixed_fm, np.asarray(fixed_in.points), fixed_in.extent, fixed_in.pixel_um)
    m = side_profile(moving_fm, np.asarray(moving_in.points), moving_in.extent, moving_in.pixel_um)
    rel = relation_evidence(metadata_relation, Path(fixed_in.image_path or "").name, Path(moving_in.image_path or "").name, sample_id)
    coord_ok = bool(fixed_in.pixel_um and moving_in.pixel_um and f["points_inside_extent_fraction"] > 0.95 and m["points_inside_extent_fraction"] > 0.95)
    return {"profile_version": PROFILE_VERSION, "fixed": f, "moving": m, "relation": rel,
            "images_both": bool(fixed_in.image_path and moving_in.image_path), "nuclei_both": bool(len(fixed_in.points) and len(moving_in.points)),
            "fov_area_ratio": f["area_mm2"] / max(m["area_mm2"], 1e-9), "coordinates_consistent": coord_ok,
            "nuclei_density_ratio": m["nuclei_per_mm2"] / max(f["nuclei_per_mm2"], 1e-9)}


def flat(p, case_id):
    row = {"case_id": case_id, "relation_value": p["relation"]["value"], "relation_confidence": p["relation"]["confidence"],
           "relation_filename_hint": p["relation"]["filename_hint"], "fov_area_ratio": p["fov_area_ratio"],
           "coordinates_consistent": p["coordinates_consistent"], "nuclei_density_ratio": p["nuclei_density_ratio"]}
    for side in ("fixed", "moving"):
        for k, v in p[side].items():
            if k != "extent_um":
                row[f"{side}_{k}"] = v
    return row


if __name__ == "__main__":
    import csv
    from benchmark_runner import METADATA_RELATION, build
    from common import HERE, load_cases
    rows, full = [], {}
    for c in load_cases():
        fi, mi, feats = build(c)
        meta = "DIFFERENT" if c["he_sample"] != c["if_sample"] else METADATA_RELATION.get(c["he_sample"])
        p = pair_profile(feats[fi.frame_name], feats[mi.frame_name], fi, mi, meta, c["he_sample"] if c["he_sample"] == c["if_sample"] else None)
        full[c["case_id"]] = p
        rows.append(flat(p, c["case_id"]))
        print(c["case_id"], {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items() if k in
                             ("relation_confidence", "fov_area_ratio", "moving_cleft_strength", "moving_repetitiveness", "moving_occ_nuclear_corr", "moving_boundary_info")}, flush=True)
    (HERE / "profiles.json").write_text(json.dumps(full, indent=1, default=float))
    with open(HERE / "profiles.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
