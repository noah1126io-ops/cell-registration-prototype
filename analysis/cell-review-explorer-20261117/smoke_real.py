"""Real-data end-to-end run of the cell-review pipeline (no UI): auto -> scripted review -> fields -> HE warp.

The scripted edits are mechanical stand-ins for a human (exclude the 5 longest auto pairs, add 3 pairs from ambiguous rows'
rank-1 candidate, reassign 1 pair to its rank-2 candidate).  They exercise the data path; they are NOT a biological review.
Outputs: runs/cell_review/<sample>/... (git-ignored) and a small summary json here.
usage: python smoke_real.py --sample 031-8_6 [--roi x0 y0 x1 y1]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from src.registration_explorer import catalog, imaging  # noqa: E402
from src.registration_research import candidate_adapter as ca  # noqa: E402
from src.registration_research import cell_review as CR  # noqa: E402

BENCH = REPO / "analysis/adaptive-registration-validation-20261111/runs"
GATE = {"031-8_6": ("P8_6_orig__D", "adjacent"), "031-6_2": ("P6_2_orig__D", "same")}


def he_reader(mod):
    info = imaging.pyramid_info(mod.image_path)
    return {"read": lambda lvl, x0, y0, x1, y1: imaging.read_region(mod.image_path, lvl, x0, y0, x1, y1),
            "level_shapes": info.level_shapes, "downsamples": info.downsamples, "pixel_um": mod.pixel_um}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--roi", type=float, nargs=4)
    ap.add_argument("--img-roi", type=float, nargs=4)
    a = ap.parse_args()
    S = catalog.load_manifest()[a.sample]
    he, iff = S.modality("HE"), S.modality("IF")
    he_pts, he_ids = catalog.load_nuclei(he)
    if_pts, if_ids = catalog.load_nuclei(iff)
    case, gate = GATE[a.sample]
    cand = ca.from_benchmark_result(BENCH / case / "result.json", gate)[0]
    M_ifhe, t_ifhe = cand.T("final")                      # benchmark: fixed = HE, moving = IF
    M, t = CR.inverse_affine(M_ifhe, t_ifhe)              # -> HE -> IF
    tracemalloc.start()
    t0 = time.perf_counter()
    auto = CR.compute_auto(he_pts, he_ids, if_pts, if_ids, M, t, roi=a.roi, if_extent=iff.extent(),
                           meta={"sample_id": a.sample, "global_source": f"{case} gate={gate} candidate {cand.id} (inverted to HE->IF)"})
    t_auto = time.perf_counter() - t0
    d_auto = CR.save_auto(auto, a.sample)
    rv = CR.Review.start(auto)
    tab = auto.table
    m = tab[tab.status == "matched"].sort_values("top1_distance_um", ascending=False)
    for h in m.he_index.head(5):
        rv.apply("exclude", int(h), user="smoke-script", reason="scripted: longest auto pair")
    amb = tab[tab.status == "ambiguous"].head(3)
    for h, f in zip(amb.he_index, amb.top1_if_index):
        rv.apply("add", int(h), int(f), user="smoke-script", reason="scripted: rank-1 of an ambiguous row")
    c2 = auto.candidates[(auto.candidates["rank"] == 2) & auto.candidates.he_index.isin(m.he_index.iloc[5:6])]
    if len(c2):
        rv.apply("reassign", int(c2.he_index.iloc[0]), int(c2.if_index.iloc[0]), user="smoke-script", reason="scripted: rank-2")
    rv.undo(user="smoke-script"); rv.redo(user="smoke-script")
    p_rev = rv.save(a.sample)
    t1 = time.perf_counter()
    fields = CR.compute_fields(auto, rv, he_pts, if_pts)
    t_fields = time.perf_counter() - t1
    roi = a.img_roi or auto.meta["roi_if_frame"]
    imgs, recs = {}, {}
    t2 = time.perf_counter()
    if he.image_on_disk():
        rd = he_reader(he)
        for name, fr in (("he_affine", None), ("he_reviewed", fields["REVIEWED"]), ("he_D", fields["D"])):
            img, valid, rec = CR.warp_he_roi(rd, M, t, roi, 2.0, fr)
            imgs[name], recs[name] = img, rec
    t_img = time.perf_counter() - t2
    ex = CR.export_bundle(a.sample, auto, rv, fields, he_pts, he_ids, if_pts, if_ids, images=imgs, image_records=recs,
                          settings={"script": "smoke_real.py", "img_roi": roi})
    peak = tracemalloc.get_traced_memory()[1] / 1e6
    out = {"sample": a.sample, "global_source": auto.meta["global_source"], "roi": auto.meta["roi_if_frame"],
           "n_he_roi": auto.meta["n_he_roi"], "n_if_candidates": auto.meta["n_if_candidates"], "status_counts": auto.meta["status_counts"],
           "conflicting_auto_pairs": auto.meta["conflicting_auto_pairs"], "auto_dir": str(d_auto.relative_to(REPO)),
           "review_file": str(p_rev.relative_to(REPO)), "review_ops": len(rv.ops), "reviewed_pairs": len(rv.pairs()),
           "review_conflicts": len(rv.conflicts()), "fields": {k: v.summary() for k, v in fields.items()},
           "review_effect": CR.displacement_change(fields["AUTO"], fields["REVIEWED"], CR.map_points(M, t, he_pts)[auto.he_idx]),
           "images": recs, "export_dir": str(ex.relative_to(REPO)),
           "timing_s": {"auto": t_auto, "fields": t_fields, "images": t_img}, "peak_tracemalloc_mb": peak}
    (HERE / f"smoke_{a.sample}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=CR._json_default))
    print(json.dumps({k: out[k] for k in ("status_counts", "reviewed_pairs", "timing_s", "peak_tracemalloc_mb")}, default=str))
    for k, v in out["fields"].items():
        print(k, v["applied"], v["reason"], v.get("anchors_n"), v.get("anchor_coverage_cells"), v.get("jacobian_min"), v.get("displacement_max_um"))
    for k, v in recs.items():
        print(k, v["status"], v["inverse_residual_max_um"], v["valid_fraction"])


if __name__ == "__main__":
    main()
