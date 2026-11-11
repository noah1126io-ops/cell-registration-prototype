"""Aggregate real + synthetic results, apply the prespecified status / fallback rules, write CSV/JSON/figures/artifacts."""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from oir.frames import Affine2D, TransformArtifact, json_safe  # noqa: E402
from oir.pipeline import ACCEPT, TRUST, footprint  # noqa: E402
from oir.sample import load_sample  # noqa: E402
from oir.search import pose_matrix  # noqa: E402

CACHE = OUT / "cache"
IMAGE_METHODS = ("E1", "E2", "E3")
MAIN = {"E0": "E0_HEmov", "E0x": "E0x_HEfix", "E1": "E1he_HEfix", "E2": "E2_HEfix", "E3": "E3zsum_HEfix"}


def load_real():
    recs = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(CACHE / "real_031-*_*.json")))]
    return {(r["sample"], r["config"]): r for r in recs}


def second_peak_distance(r):
    s = r["search"]
    top, sec = s.get("top"), s.get("second")
    if not top or not sec or "chirality" not in sec:
        return np.nan
    c = np.asarray(r["moving_extent"]) / 2
    pt = pose_matrix(top["chirality"], top["theta"], top["scale"]) @ c + np.asarray(top["t"])
    ps = pose_matrix(sec["chirality"], sec["theta"], sec["scale"]) @ c + np.asarray(sec["t"])
    return float(np.linalg.norm(pt - ps))


def status(r, real):
    """Prespecified decision rules (architecture.md section 5)."""
    st, reasons = [], []
    vis = r["visible_images"]
    if vis:
        st.append("IMAGE_AVAILABLE")
        usable = all(r["feature_quality"][k]["usable"] for k in vis)
        st.append("IMAGE_USABLE" if usable else "IMAGE_INCOMPATIBLE")
        if not usable:
            reasons.append("image quality: " + "; ".join(sum((r["feature_quality"][k]["reasons"] for k in vis), [])))
    verdict = r["search"].get("verdict")
    ev = r["eval"]
    geo = bool(ev.get("geometric_pass"))
    search_ok = verdict in ("UNIQUE", "A2")
    if r["method"] in IMAGE_METHODS:
        if "IMAGE_INCOMPATIBLE" in st:
            accepted = False
        elif not search_ok or not geo:
            st.append("IMAGE_REGISTRATION_UNCERTAIN")
            if not search_ok:
                m = r['search'].get('margin')
                reasons.append(f"search verdict {verdict} (margin {'inf' if m is None else f'{m:.3f}'})")
            if not geo:
                reasons.append(f"nuclear geometry inconsistent (<=3um {ev['final']['le_3um']:.3f}, x{ev['inlier_over_null']:.1f} null)")
            accepted = False
        else:
            accepted = True
    else:
        accepted = geo and search_ok
        if not geo:
            reasons.append(f"nuclear geometry inconsistent (<=3um {ev['final']['le_3um']:.3f}, x{ev['inlier_over_null']:.1f} null)")
        if not search_ok:
            reasons.append(f"search verdict {verdict}")
    final_source = r["config"]
    if r["method"] in IMAGE_METHODS and not accepted:
        fb = real.get((r["sample"], f"E0x_{r['fixed']}fix"))
        if fb is not None and status(fb, real)[1]:
            st.append("POINT_ONLY_FALLBACK"); final_source = fb["config"]
        else:
            st.append("POINT_ONLY_FALLBACK"); final_source = None
            reasons.append("point-only fallback also not accepted -> no transform accepted")
    return st, accepted, reasons, final_source


def flat(r, real):
    ev = r["eval"]; fin = ev.get("final", {}); img = ev.get("image") or {}
    st, acc, reasons, fsrc = status(r, real)
    ho = [h.get("le_3um") for h in r["heldout"] if h.get("le_3um") is not None]
    hon = [h.get("null_le_3um") for h in r["heldout"] if h.get("null_le_3um") is not None]
    top = r["search"].get("top") or {}
    p = ev.get("params_he_to_if", {})
    return {
        "sample": r["sample"], "config": r["config"], "method": r["method"], "mode": r["mode"], "fixed": r["fixed"], "moving": r["moving"],
        "inputs": "points" + ("" if not r["visible_images"] else " + " + "+".join(r["visible_images"]) + " image"),
        "image_features_used_in_search": ";".join(r["image_features_used"]), "points_used_in_search": r["point_used_in_search"],
        "search_verdict": r["search"].get("verdict"), "search_margin": r["search"].get("margin"),
        "top_layer_z": json.dumps({k: (None if v is None else round(v, 2)) for k, v in (top.get("layer_z") or {}).items()}),
        "top_layer_ncc": json.dumps({k: (None if v is None else round(v, 3)) for k, v in (top.get("layer_ncc") or {}).items()}),
        "second_peak_distance_um": second_peak_distance(r),
        "search_pose_le_3um": (ev.get("search_pose") or {}).get("le_3um"),
        "ref_median_um": (ev.get("vs_reference_um") or {}).get("median"), "ref_p95_um": (ev.get("vs_reference_um") or {}).get("p95"),
        "nn_median_um": fin.get("nn_median_um"), "nn_p95_um": fin.get("nn_p95_um"),
        "le_1um": fin.get("le_1um"), "le_2um": fin.get("le_2um"), "le_3um": fin.get("le_3um"), "le_5um": fin.get("le_5um"),
        "null_le_3um": (ev.get("null") or {}).get("le_3um"), "inlier_over_null": ev.get("inlier_over_null"), "n_if_eval": ev.get("n_if_eval"),
        "heldout_le_3um_mean": float(np.mean(ho)) if ho else np.nan, "heldout_null_le_3um_mean": float(np.mean(hon)) if hon else np.nan,
        "geometric_pass": ev.get("geometric_pass"),
        "img_nuclear_corr": img.get("nuclear_corr"), "img_nuclear_block_corr_median": img.get("nuclear_block_corr_median"),
        "img_tissue_corr": img.get("tissue_corr"), "img_tissue_dice": img.get("tissue_dice"), "img_boundary_chamfer_um": img.get("boundary_chamfer_um"),
        "img_nuclear_independent_of_search": (img.get("independent_of_search") or {}).get("nuclear_corr"),
        "img_tissue_independent_of_search": (img.get("independent_of_search") or {}).get("tissue_corr"),
        "scale_he_to_if": p.get("scale"), "rotation_deg_he_to_if": p.get("rotation_deg"), "reflect": p.get("reflect"), "anisotropy": p.get("anisotropy"),
        "icp_scale_change": (r["refine"].get("pose_change") or {}).get("scale_change"), "icp_rot_change_deg": (r["refine"].get("pose_change") or {}).get("rot_change_deg"),
        "icp_median_shift_um": (r["refine"].get("pose_change") or {}).get("median_shift_um"), "icp_rejected": r["refine"].get("icp_rejected"),
        "runtime_s": r["runtime_s"], "peak_mem_mb": r["peak_mem_mb"], "n_poses": r["search"].get("n_poses") or r["search"].get("n_hypotheses"),
        "res_coarse_um": r["cfg"]["res_coarse"] if r["method"] != "E0" else 4.0, "res_fine_um": r["cfg"]["res_fine"] if r["method"] != "E0" else 4.0,
        "feature_res_um": r["extra"].get("feat_res", 2.0),
        "status": ";".join(st), "accepted": acc, "failure_reasons": " | ".join(reasons), "final_transform_source": fsrc,
    }


def T_he_to_if(r):
    T = Affine2D.from_dict(r["T_final"])
    return T if r["fixed"] == "IF" else T.inverse()


def main():
    real = load_real()
    rows = [flat(r, real) for r in real.values()]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "mode_comparison.csv", index=False)
    df[["sample", "config", "method", "mode", "fixed", "inputs", "image_features_used_in_search", "search_verdict", "search_margin", "top_layer_z",
        "top_layer_ncc", "second_peak_distance_um", "search_pose_le_3um", "ref_median_um", "ref_p95_um"]].to_csv(OUT / "region_search_metrics.csv", index=False)
    df[["sample", "config", "method", "fixed", "nn_median_um", "nn_p95_um", "le_1um", "le_2um", "le_3um", "le_5um", "null_le_3um", "inlier_over_null",
        "heldout_le_3um_mean", "heldout_null_le_3um_mean", "geometric_pass", "scale_he_to_if", "rotation_deg_he_to_if", "reflect", "anisotropy",
        "icp_scale_change", "icp_rot_change_deg", "icp_median_shift_um", "icp_rejected", "img_nuclear_corr", "img_nuclear_block_corr_median",
        "img_nuclear_independent_of_search", "img_tissue_corr", "img_tissue_dice", "img_boundary_chamfer_um", "img_tissue_independent_of_search",
        "status", "accepted", "final_transform_source"]].to_csv(OUT / "registration_metrics.csv", index=False)

    # swap consistency: HE-fixed vs IF-fixed (same method), in the IF frame
    swap = []
    for s in ("031-6_2", "031-8_6"):
        S = load_sample(s)
        g = np.stack(np.meshgrid(np.linspace(0, S.iff.frame.extent_um()[0], 21), np.linspace(0, S.iff.frame.extent_um()[1], 21)), -1).reshape(-1, 2)
        for a, b in (("E0x_HEfix", "E0x_IFfix"), ("E2_HEfix", "E2_IFfix"), ("E3zsum_HEfix", "E3zsum_IFfix"), ("E0_HEmov", "E0_IFmov")):
            if (s, a) in real and (s, b) in real:
                Ta, Tb = T_he_to_if(real[(s, a)]), T_he_to_if(real[(s, b)])
                d = np.linalg.norm(Ta.apply(Tb.inverse().apply(g)) - g, axis=1)
                # A (HE->IF) after inverse of B (IF->HE) is the identity if both represent the same physical map
                swap.append({"sample": s, "pair": f"{a} vs {b}", "median_um": float(np.median(d)), "max_um": float(d.max())})
    swap = pd.DataFrame(swap)

    # runtime
    rt = df[["sample", "config", "method", "mode", "runtime_s", "peak_mem_mb", "n_poses", "res_coarse_um", "res_fine_um", "feature_res_um"]].copy()
    fq = []
    for s in ("031-6_2", "031-8_6"):
        q = json.loads((CACHE / f"feats_{s}_quality.json").read_text())
        for k, v in q.items():
            fq.append({"sample": s, "config": f"feature_extraction_{k}", "method": "features", "mode": "", "runtime_s": v["runtime_s"],
                       "peak_mem_mb": np.nan, "n_poses": np.nan, "res_coarse_um": np.nan, "res_fine_um": np.nan,
                       "feature_res_um": float(k.split("_")[1].replace("um", "")), "source_level": v["source_level"], "grid_shape": str(v["shape"])})
    rt = pd.concat([rt, pd.DataFrame(fq)], ignore_index=True)

    # synthetic
    syn = []
    for f in sorted(glob.glob(str(CACHE / "syn_*.json"))):
        d = json.loads(Path(f).read_text())
        for m, r in d["methods"].items():
            ev = r.get("eval") or {}
            verdict_ok = r["verdict"] in ("UNIQUE", "A2")
            accepted = bool(verdict_ok and (ev.get("geometric_pass") or False))
            if m in IMAGE_METHODS:
                accepted = accepted
            syn.append({"case": d["case"], "source": d["source"], "condition": d["condition"], "absent": d["absent"], "q_shared": d["q_shared"],
                        "method": m, "verdict": r["verdict"], "margin": r.get("margin"),
                        "region_found": r.get("region_found"), "registration_success": r.get("registration_success"),
                        "median_err_um": (r.get("truth_final") or {}).get("median_um"), "rms_err_um": (r.get("truth_final") or {}).get("rms_um"),
                        "search_median_err_um": (r.get("truth_search") or {}).get("median_um"),
                        "le_3um": (ev.get("final") or {}).get("le_3um"), "null_le_3um": (ev.get("null") or {}).get("le_3um"),
                        "geometric_pass": ev.get("geometric_pass"), "accepted": accepted,
                        "true_id_rate_shared": r.get("true_id_rate_shared"), "precision_le3um_pairs": r.get("precision_of_le3um_pairs"),
                        "runtime_s": r["runtime_s"], "peak_mem_mb": r["peak_mem_mb"]})
    syn = pd.DataFrame(syn)
    if len(syn):
        syn.to_csv(OUT / "synthetic_results.csv", index=False)
        syn["false_accept"] = syn["absent"] & syn["accepted"]
        syn["correct_accept"] = (~syn["absent"]) & syn["accepted"] & syn["registration_success"].fillna(False)
        syn["wrong_accept"] = (~syn["absent"]) & syn["accepted"] & ~syn["registration_success"].fillna(False)
        summ = syn.groupby(["condition", "method"]).agg(n=("case", "size"), region_found=("region_found", "mean"), reg_success=("registration_success", "mean"),
                                                         accepted=("accepted", "mean"), correct_accept=("correct_accept", "mean"), wrong_accept=("wrong_accept", "mean"),
                                                         false_accept=("false_accept", "mean"), median_err_um=("median_err_um", "median"),
                                                         true_id=("true_id_rate_shared", "median"), precision_le3=("precision_le3um_pairs", "median"),
                                                         runtime_s=("runtime_s", "median")).reset_index()
        summ.to_csv(OUT / "synthetic_summary.csv", index=False)
        for _, r in syn.groupby("method").agg(runtime_s=("runtime_s", "median"), peak_mem_mb=("peak_mem_mb", "median")).reset_index().iterrows():
            rt = pd.concat([rt, pd.DataFrame([{"sample": "synthetic (median of 32)", "config": r["method"], "method": r["method"], "runtime_s": r["runtime_s"], "peak_mem_mb": r["peak_mem_mb"]}])], ignore_index=True)
    else:
        summ = pd.DataFrame()
    rt.to_csv(OUT / "runtime_metrics.csv", index=False)

    # artifacts for every real config
    (OUT / "artifacts").mkdir(exist_ok=True)
    for (s, c), r in real.items():
        S = load_sample(s)
        frames = {"HE": S.he.frame, "IF": S.iff.frame}
        st, acc, reasons, fsrc = status(r, real)
        T = Affine2D.from_dict(r["T_final"]) if r["T_final"] else None
        art = TransformArtifact(
            sample=s, method=c, mode=r["mode"], source_frame=frames[r["moving"]].to_dict(), target_frame=frames[r["fixed"]].to_dict(),
            forward_transform=T.to_dict() if T else {}, inverse_transform=T.inverse().to_dict() if T else {},
            transform_direction=f"computed {r['moving']}->{r['fixed']} (moving->fixed); inverse stored for image resampling",
            output_frame=frames[r["fixed"]].name,
            model="similarity search pose (ICP rejected)" if r["refine"].get("icp_rejected") else ("A2 + production ICP" if r["method"] == "E0" else "similarity ICP -> affine ICP (cropped)"),
            roi={"evaluation": "IF nuclei inside mapped HE footprint (10 um inset)", "origin": "frame-local (UNKNOWN stage origin)"},
            validity_region_target_um=T.apply(footprint(r["moving_extent"])).tolist() if T else [],
            inputs={"points": True, "visible_images": r["visible_images"], "images_used_in_search": r["image_features_used"],
                    "point_used_in_search": r["point_used_in_search"]},
            image_status=st, confidence={"search_verdict": r["search"].get("verdict"), "search_margin": r["search"].get("margin"),
                                         "geometric_pass": r["eval"].get("geometric_pass"), "inlier_over_null": r["eval"].get("inlier_over_null")},
            failure_reasons=reasons + [f"final transform taken from {fsrc}"] if fsrc != c else reasons, accepted=acc)
        art.write(OUT / "artifacts" / f"{s}_{c}.json")

    p0 = json.loads((CACHE / "p0_compat.json").read_text()) if (CACHE / "p0_compat.json").exists() else None
    diag = {"date": "2026-10-19", "acceptance_prespecified": ACCEPT, "icp_trust_region_prespecified": TRUST,
            "search_verdict_rule": "AMBIGUOUS if relative margin to the 2nd distinct candidate < 0.10; LOW_PEAK if combined robust z < 6 per layer; else UNIQUE",
            "status_rules": "architecture.md section 5", "p0_compat": p0, "swap_consistency": swap.to_dict("records"),
            "real": df.to_dict("records"), "synthetic_summary": summ.to_dict("records") if len(summ) else None,
            "main_configs": MAIN}
    (OUT / "diagnostic_summary.json").write_text(json.dumps(json_safe(diag), indent=2))
    swap.to_csv(OUT / "cache" / "swap_consistency.csv", index=False)
    print(df[["sample", "config", "search_verdict", "le_3um", "nn_median_um", "ref_median_um", "img_nuclear_corr", "status", "accepted", "runtime_s"]].to_string())
    print(swap.to_string())
    if len(summ):
        print(summ.to_string())


if __name__ == "__main__":
    main()
