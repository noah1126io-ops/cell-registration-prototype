"""Parity (shared engine vs. the analysis copies / saved results) and end-to-end runs of the research Workflow C.

  python -B run_core_checks.py --task <name>      (one task per srun; results -> cache/<name>.json)
  python -B run_core_checks.py --list
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
CACHE = HERE / "cache"
OIR_DIR = REPO / "analysis/optional-image-assisted-registration-20261019"
E0X_DIR = REPO / "analysis/e0x-difficulty-and-web-integration-20261020"

from src.registration_core import Constraints, WorkflowConfig, run_workflow, write_artifact  # noqa: E402
from src.registration_core.frames import json_safe  # noqa: E402
from src.registration_explorer.catalog import load_manifest, side_input  # noqa: E402


def maxdiff(a, b):
    return float(np.abs(np.asarray(a, float) - np.asarray(b, float)).max())


def sides(sample, fixed, use_image=True):
    S = load_manifest()[sample]
    return side_input(S, fixed, use_image=use_image), side_input(S, S.other(fixed), use_image=use_image)


def legacy_paths():
    for p in (str(OIR_DIR), str(REPO / "analysis/affine-global-initialization-20261014")):
        if p not in sys.path:
            sys.path.insert(0, p)


# ----------------------------------------------------------------------------- parity tasks
def parity_a2():
    """Vendored A2 == analysis init_lib.method_a2 (031-6_2, HE moving -> IF fixed)."""
    legacy_paths()
    import init_lib
    from src.registration_core import a2
    f, m = sides("031-6_2", "IF")
    r_old = init_lib.method_a2(m.points, f.points)
    r_new = a2.method_a2(m.points, f.points)
    d_old = init_lib.detector(r_old.M, r_old.t, m.points, f.points, basin_margin=r_old.diagnostics["basin_margin"])
    d_new = a2.detector(r_new.M, r_new.t, m.points, f.points, basin_margin=r_new.diagnostics["basin_margin"])
    return {"max_abs_dM": maxdiff(r_old.M, r_new.M), "max_abs_dt": maxdiff(r_old.t, r_new.t),
            "detector_old": d_old["detector_status"], "detector_new": d_new["detector_status"]}


def parity_e0x():
    """Vendored pipeline.run_method('E0x') == analysis oir run_method on the same inputs; and == saved 10-19 result."""
    legacy_paths()
    from oir import pipeline as old
    from src.registration_core import pipeline as new
    f, m = sides("031-6_2", "HE")
    t0 = time.perf_counter()
    Ro = old.run_method("031-6_2", "E0x", old.Side("HE", f.points, f.extent, None), old.Side("IF", m.points, m.extent, None), "P0")
    Rn = new.run_method("031-6_2", "E0x", new.Side("HE", f.points, f.extent, None), new.Side("IF", m.points, m.extent, None), "P0")
    saved = json.loads((OIR_DIR / "cache/real_031-6_2_E0x_HEfix.json").read_text())
    return {"old_vs_new_max_abs_dM": maxdiff(Ro.T.M, Rn.T.M), "old_vs_new_max_abs_dt": maxdiff(Ro.T.t, Rn.T.t),
            "new_vs_saved_20261019_dM": maxdiff(Rn.T.M, saved["T_final"]["M"]), "new_vs_saved_20261019_dt": maxdiff(Rn.T.t, saved["T_final"]["t"]),
            "verdict_old": Ro.search["verdict"], "verdict_new": Rn.search["verdict"], "runtime_s": time.perf_counter() - t0}


# ----------------------------------------------------------------------------- workflow runs
def wf(sample, fixed, localization="auto", use_image=True, name=None, cons=None, image_path_override=None):
    f, m = sides(sample, fixed, use_image)
    if image_path_override is not None:          # simulated image read failure: file missing
        f.image_path = image_path_override
    art = run_workflow(sample, f, m, WorkflowConfig(localization=localization), cons, log=lambda s: print(s, flush=True))
    write_artifact(art, CACHE / f"artifact_{name}.json")
    sel = next((c for c in art["candidates"] if c["id"] == art["decision"]["selected_candidate"]), None)
    summ = {"decision": {k: art["decision"][k] for k in ("status", "accepted", "selected_candidate", "reason", "failure_stage")},
            "per_source": art["decision"]["per_source"], "image_status": art["image_status"].get("flags"),
            "image_reasons": art["image_status"].get("reasons"), "runtime_s": art["runtime_s"]}
    if sel:
        summ["selected"] = {"id": sel["id"], "flags": sel["flags"], "fixed_frame": sel["validation"].get("fixed_frame"),
                            "if_frame_final": (sel["validation"].get("if_frame") or {}).get("final"),
                            "if_frame_null": (sel["validation"].get("if_frame") or {}).get("null"),
                            "if_frame_image": (sel["validation"].get("if_frame") or {}).get("image"), "T_final": sel["T_final"]}
    return summ, art


def wf_points_62():
    """Point-only (images hidden) on 031-6_2 HE fixed: must reproduce the saved E0x (1.91 um) and the 10-20 X3 ablation."""
    summ, art = wf("031-6_2", "HE", "auto", use_image=False, name="points_031-6_2_HEfix")
    saved = json.loads((OIR_DIR / "cache/real_031-6_2_E0x_HEfix.json").read_text())
    abl = json.loads((E0X_DIR / "cache/abl_031-6_2_HEfix_X3.json").read_text())
    T = summ["selected"]["T_final"]
    summ["vs_saved_20261019_E0x"] = {"dM": maxdiff(T["M"], saved["T_final"]["M"]), "dt": maxdiff(T["t"], saved["T_final"]["t"]),
                                      "saved_nn_median_um": saved["eval"]["final"]["nn_median_um"], "saved_le_3um": saved["eval"]["final"]["le_3um"]}
    summ["vs_runner_20261020_X3"] = {"dM": maxdiff(T["M"], abl["candidates"][0]["T_final"]["M"]), "dt": maxdiff(T["t"], abl["candidates"][0]["T_final"]["t"]),
                                      "runner_accepted": abl["accepted"]}
    return summ


def wf_auto(sample, fixed):
    def f():
        summ, art = wf(sample, fixed, "auto", name=f"auto_{sample}_{fixed}fix")
        if sample == "031-6_2" and fixed == "HE":
            saved = json.loads((OIR_DIR / "cache/real_031-6_2_E2tn_HEfix.json").read_text())
            img = next((c for c in art["candidates"] if c["id"] == "image#0"), None)
            if img:
                summ["image_vs_saved_20261019_E2tn"] = {"search_verdict_saved": saved["search"]["verdict"], "search_verdict_new": art["localization"]["image"]["verdict"],
                                                        "dM_search": maxdiff(img["T_search"]["M"], saved["T_search"]["M"]), "dt_search": maxdiff(img["T_search"]["t"], saved["T_search"]["t"])}
        return summ
    return f


def wf_a2_serial():
    """Point-only legacy pair (no images, HE in px): A2 path; compare with the stored A2 (R2) transform."""
    summ, art = wf("serial_section_pair_20260904", "IF", "a2", use_image=False, name="a2_serial_IFfix")
    stored = json.loads((REPO / "analysis/real-registration-incremental-value-20261015/cache/real_stages.json").read_text())["stages"]["R2"]
    T = next(c for c in art["candidates"] if c["id"] == "a2#0")["T_final"]
    summ["vs_stored_R2"] = {"dM": maxdiff(T["M"], stored["M"]), "dt": maxdiff(T["t"], stored["t"])}
    return summ


def wf_imgfail():
    """Image read failure on 031-6_2 (fixed image path points to a missing file) -> point search only, recorded."""
    summ, _ = wf("031-6_2", "HE", "auto", name="imgfail_031-6_2_HEfix", image_path_override="/nonexistent/missing.ome.tif")
    return summ


def wf_assisted_86():
    """031-8_6 with a user ROI around the top point candidate's region + exclusion of nothing: ASSISTED, never accepted."""
    auto = json.loads((CACHE / "artifact_auto_031-8_6_HEfix.json").read_text())
    poly = next(c for c in auto["candidates"] if c["id"] == auto["decision"]["selected_candidate"])["validation"]["overlap"]["overlap_polygon_fixed"]
    P = np.asarray(poly); c = P.mean(0)
    roi = (c + (P - c) * 1.6).tolist()
    summ, _ = wf("031-8_6", "HE", "auto", name="assisted_roi_031-8_6_HEfix", cons=Constraints(search_roi=roi, annotation_ids=["check-roi"]))
    return summ


TASKS = {"parity_a2": parity_a2, "parity_e0x": parity_e0x, "wf_points_62": wf_points_62,
         "wf_auto_62_HE": wf_auto("031-6_2", "HE"), "wf_auto_62_IF": wf_auto("031-6_2", "IF"),
         "wf_auto_86_HE": wf_auto("031-8_6", "HE"), "wf_auto_86_IF": wf_auto("031-8_6", "IF"),
         "wf_a2_serial": wf_a2_serial, "wf_imgfail": wf_imgfail, "wf_assisted_86": wf_assisted_86}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--task"); ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        print(" ".join(TASKS))
    else:
        CACHE.mkdir(exist_ok=True)
        t0 = time.perf_counter()
        out = TASKS[a.task]()
        out["task_wall_s"] = time.perf_counter() - t0
        (CACHE / f"{a.task}.json").write_text(json.dumps(json_safe(out), indent=1, ensure_ascii=False))
        print(json.dumps(json_safe(out), ensure_ascii=False)[:3000], flush=True)
