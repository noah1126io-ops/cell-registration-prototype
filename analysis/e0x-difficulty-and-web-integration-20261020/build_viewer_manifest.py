"""Write viewer_sample_manifest.json (the Registration Explorer's sample registry) from files that exist.

Every path is checked; metadata that the files do not carry is written as null (shown as UNKNOWN).
  python -B build_viewer_manifest.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import tifffile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PROJECTS = REPO.parent
OIR = REPO / "analysis/optional-image-assisted-registration-20261019"
HE_NUC = REPO / "analysis/he-stardist-seqif-registration-20261019/out"
OLD_RUN = REPO / "runs/20260904-161247-698fe6c9"
OLD_R2 = REPO / "analysis/real-registration-incremental-value-20261015/cache/real_stages.json"


def ome(path):
    with tifffile.TiffFile(path) as t:
        s = t.series[0]
        om = t.ome_metadata or ""
        px = re.search(r'PhysicalSizeX="([^"]+)"', om)
        return {"shape_px": list(s.levels[0].shape[:2]), "pixel_um": float(px.group(1)) if px else None,
                "has_stage_origin": bool(re.search(r"<Plane[^>]*PositionX", om))}


def dual_sample(name, provenance):
    d = PROJECTS / name
    he_tif = next(d.glob(f"{name}_HE*.ome.tif"))
    if_tif = d / f"{name}_seqif.ome.tif"
    hi, ii = ome(he_tif), ome(if_tif)
    origin = "UNKNOWN (no stage origin in OME; frame-local (0,0) used)"
    return {
        "sample_id": name, "label": name,
        "coordinate_system": "frame-local world um per modality; x right, y down; world = level-0 pixel index x pixel_um; stage/crop origin UNKNOWN",
        "provenance": provenance,
        "modalities": {
            "HE": {"label": "H&E", "kind": "brightfield_rgb", "pixel_um": hi["pixel_um"], "shape_px": hi["shape_px"], "image_path": str(he_tif),
                   "channels": "RGB brightfield H&E (uint8)", "origin": origin if not hi["has_stage_origin"] else "see OME",
                   "nuclei": {"format": "csv_px", "path": str(HE_NUC / name / "he_nuclei.csv"), "x": "centroid_x_px", "y": "centroid_y_px", "units": "px",
                              "source": "StarDist 2D_versatile_he (analysis he-stardist-seqif-registration-20261019)"},
                   "tissue_npz": str(OIR / f"cache/feats_{name}_HE_2um.npz")},
            "IF": {"label": "蛍光 (seqIF)", "kind": "fluorescence_display_rgb", "pixel_um": ii["pixel_um"], "shape_px": ii["shape_px"], "image_path": str(if_tif),
                   "channels": "uint8 RGB display rendering: R=Desmoplakin, G=Dystrophin1, B=Hoechst+Desmoplakin (no pure nuclear channel)",
                   "origin": origin if not ii["has_stage_origin"] else "see OME",
                   "nuclei": {"format": "geojson_px_props", "path": str(d / f"{name}_seg_seqif_level0.geojson"), "x": "centroid_x", "y": "centroid_y", "units": "px",
                              "source": "provider GeoJSON *_seg_seqif_level0.geojson"},
                   "tissue_npz": str(OIR / f"cache/feats_{name}_IF_2um.npz")},
        },
        "result_sources": [
            {"type": "oir-20261019", "artifacts_dir": str(OIR / "artifacts"), "cache_dir": str(OIR / "cache"), "metrics_csv": str(OIR / "mode_comparison.csv"),
             "previews_dir": str(OIR / "previews")},
            {"type": "e0x-ablation-20261020", "cache_dir": str(HERE / "cache")},
            {"type": "explorer-runs", "runs_dir": str(REPO / "runs/registration-explorer")},
        ],
    }


def old_pair():
    cfg = json.loads((OLD_RUN / "config.json").read_text())["affine"]
    r2 = json.loads(OLD_R2.read_text())["stages"]["R2"]
    return {
        "sample_id": "serial_section_pair_20260904", "label": "serial_section_pair_20260904 (point-only)",
        "coordinate_system": "HE: production crop pixels (spacing UNKNOWN in this record); fluorescence: world um. The transform maps HE px -> fluorescence um.",
        "provenance": {"summary": "Earlier Point-only real pair (Workflow C run 20260904-161247-698fe6c9). Both nuclear point sets; no images registered for this viewer.",
                       "he": "original_moving_points.npy (HE nuclei, px)", "fluorescence": "original_fixed_points.npy (fluorescence nuclei, um)",
                       "same_section": None},
        "modalities": {
            "HE": {"label": "H&E", "kind": "points_only", "pixel_um": None, "shape_px": None, "image_path": None,
                   "extent_override": [float(cfg["image_width"]), float(cfg["image_height"])],
                   "nuclei": {"format": "npy", "path": str(OLD_RUN / "input/original_moving_points.npy"), "units": "native", "world_units": "px",
                              "source": "Workflow C input (HE crop px)"}},
            "IF": {"label": "蛍光", "kind": "points_only", "pixel_um": None, "shape_px": None, "image_path": None,
                   "nuclei": {"format": "npy", "path": str(OLD_RUN / "input/original_fixed_points.npy"), "units": "native", "world_units": "um",
                              "source": "Workflow C input (um)"}},
        },
        "result_sources": [
            {"type": "stored-transform", "label": "Legacy A2 R2 (analysis real-registration-incremental-value-20261015)", "method": "Legacy A2",
             "fixed": "IF", "moving": "HE", "M": r2["M"], "t": r2["t"], "source_file": str(OLD_R2), "verdict": "A2",
             "accepted": None, "note": "no research-artifact verdict recorded; shown as preview with UI-recomputed diagnostics"},
            {"type": "explorer-runs", "runs_dir": str(REPO / "runs/registration-explorer")},
        ],
    }


def main():
    prov62 = {"summary": "031-6_2: H&E 031-6 XY02 and seqIF 031-6_2 background.",
              "user_correction_2026-10-20": "Treated as DIFFERENT sections (user correction; overrides the provider note and the 2026-10-19 REPORT wording 'same section').",
              "provider_note": "HE shown in the 031-6_2 view: 031-6 XY02, level 0, whole HE set (registered to this field: scale 0.985, NCC 0.706).",
              "same_section": False, "reference_transform": "provider image-NCC placement (not ground truth, not independent)"}
    prov86 = {"summary": "031-8_6: H&E from 031-7 XY06 (031-8 has no H&E) and seqIF 031-8_6 background.",
              "provider_note": "REFERENCE ONLY - 031-8 has no HE; not registered to this field.",
              "same_section": False, "section_spacing": "UNKNOWN", "reference_transform": "none (correspondence region not annotated)"}
    samples = [dual_sample("031-6_2", prov62), dual_sample("031-8_6", prov86), old_pair()]
    for s in samples:
        for m in s["modalities"].values():
            for p in (m.get("image_path"), m["nuclei"].get("path"), m.get("tissue_npz")):
                if p and not Path(p).exists():
                    raise FileNotFoundError(p)
    out = {"schema": "registration-explorer-manifest/0.1", "generated": "2026-10-20", "samples": samples}
    (HERE / "viewer_sample_manifest.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("wrote", HERE / "viewer_sample_manifest.json", [s["sample_id"] for s in samples])


if __name__ == "__main__":
    main()
