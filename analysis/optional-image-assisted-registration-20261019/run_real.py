"""Run one (sample, config) on the real dual-image samples; writes cache/real_<sample>_<config>.json.

  python -B run_real.py --sample 031-6_2 --config E2_HEfix
  python -B run_real.py --features 031-6_2          # build + cache feature maps (2 um and 4 um)
  python -B run_real.py --p0compat                   # Phase B: E0 wrapper == direct A2 (old Real + new samples)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from oir import metrics as mt  # noqa: E402
from oir.features import FeatureMaps, extract  # noqa: E402
from oir.frames import Affine2D, json_safe  # noqa: E402
from oir.pipeline import Side, evaluate, heldout, run_method  # noqa: E402
from oir.sample import SAMPLES, load_sample  # noqa: E402
from oir.search import SearchConfig  # noqa: E402

CACHE = OUT / "cache"

# name -> (method, fixed, visible images, cfg overrides, image feature, extra)
CONFIGS = {
    "E0_HEmov":      ("E0", "IF", (), {}, None, {}),
    "E0_IFmov":      ("E0", "HE", (), {}, None, {}),
    "E0x_HEfix":     ("E0x", "HE", (), {}, None, {}),
    "E0x_IFfix":     ("E0x", "IF", (), {}, None, {}),
    "E1he_HEfix":    ("E1", "HE", ("HE",), {"combine": "zsum"}, None, {}),
    "E1if_HEfix":    ("E1", "HE", ("IF",), {"combine": "zsum"}, None, {}),
    "E2_HEfix":      ("E2", "HE", ("HE", "IF"), {}, "tissue", {}),
    "E2_IFfix":      ("E2", "IF", ("HE", "IF"), {}, "tissue", {}),
    "E2nuc_HEfix":   ("E2", "HE", ("HE", "IF"), {}, "nuclear", {}),
    "E2tn_HEfix":    ("E2", "HE", ("HE", "IF"), {"combine": "zsum"}, "tissue+nuclear", {}),
    "E3zsum_HEfix":  ("E3", "HE", ("HE", "IF"), {"combine": "zsum"}, "tissue", {}),
    "E3zmin_HEfix":  ("E3", "HE", ("HE", "IF"), {"combine": "zmin"}, "tissue", {}),
    "E3rerank_HEfix": ("E3", "HE", ("HE", "IF"), {"combine": "rerank"}, "tissue", {}),
    "E3zsum_IFfix":  ("E3", "IF", ("HE", "IF"), {"combine": "zsum"}, "tissue", {}),
    "E2reflect_HEfix": ("E2", "HE", ("HE", "IF"), {"chiralities": ("proper", "reflect")}, "tissue", {}),
    "E0xreflect_HEfix": ("E0x", "HE", (), {"chiralities": ("proper", "reflect")}, None, {}),
    "E2_c16f8_HEfix": ("E2", "HE", ("HE", "IF"), {"res_coarse": 16.0, "res_fine": 8.0}, "tissue", {}),
    "E2_feat4_HEfix": ("E2", "HE", ("HE", "IF"), {}, "tissue", {"feat_res": 4.0}),
    "NCshuf_E0x":    ("E0x", "HE", (), {}, None, {"shuffle_moving_um": 128.0}),
    "NCshuf_E2":     ("E2", "HE", ("HE", "IF"), {}, "tissue", {"shuffle_moving_um": 128.0}),
    "NCshuf_E3":     ("E3", "HE", ("HE", "IF"), {"combine": "zsum"}, "tissue", {"shuffle_moving_um": 128.0}),
}


def save_feats(fm: FeatureMaps, path):
    np.savez_compressed(path, **{f"map_{k}": v for k, v in fm.maps.items()}, fov=fm.fov,
                        meta=json.dumps(json_safe({"frame": fm.frame, "modality": fm.modality, "res_um": fm.res_um, "quality": fm.quality,
                                                   "runtime_s": fm.runtime_s, "source_level": fm.source_level})))


def load_feats(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    maps = {k[4:]: z[k] for k in z.files if k.startswith("map_")}
    return FeatureMaps(meta["frame"], meta["modality"], meta["res_um"], maps, z["fov"], meta["quality"], meta["runtime_s"], meta["source_level"])


def build_features(sample):
    S = load_sample(sample)
    rows = {}
    for res in (2.0, 4.0):
        for key, mod in (("HE", S.he), ("IF", S.iff)):
            fm = extract(mod, res_um=res)
            save_feats(fm, CACHE / f"feats_{sample}_{key}_{int(res)}um.npz")
            rows[f"{key}_{int(res)}um"] = {**fm.quality, "runtime_s": fm.runtime_s, "shape": list(fm.shape)}
            print(sample, key, res, json.dumps(json_safe(fm.quality)), round(fm.runtime_s, 1), flush=True)
    (CACHE / f"feats_{sample}_quality.json").write_text(json.dumps(json_safe(rows), indent=2))


def shuffle_side(side: Side, tile_um, seed=0):
    """Tile-permute the moving side's maps and points (keeps local statistics, destroys arrangement)."""
    rng = np.random.default_rng(seed)
    fm = side.feats
    nt = (np.array(side.extent) // tile_um).astype(int)
    perm = rng.permutation(nt[0] * nt[1])
    pts = side.points
    ti = np.floor(pts / tile_um).astype(int)
    ok = (ti[:, 0] < nt[0]) & (ti[:, 1] < nt[1])
    pts, ti = pts[ok], ti[ok]
    lin = ti[:, 1] * nt[0] + ti[:, 0]
    new = perm[lin]
    new_xy = np.c_[new % nt[0], new // nt[0]]
    pts2 = pts - ti * tile_um + new_xy * tile_um
    maps = None
    if fm is not None:
        tp = int(round(tile_um / fm.res_um))
        maps = {}
        for k, v in fm.maps.items():
            out = np.zeros_like(v)
            for a in range(nt[0] * nt[1]):
                sx, sy = a % nt[0], a // nt[0]; dx, dy = perm[a] % nt[0], perm[a] // nt[0]
                out[dy * tp:(dy + 1) * tp, dx * tp:(dx + 1) * tp] = v[sy * tp:(sy + 1) * tp, sx * tp:(sx + 1) * tp]
            maps[k] = out
        fov = np.zeros_like(fm.fov); fov[: nt[1] * tp, : nt[0] * tp] = 1
        fm = FeatureMaps(fm.frame + ":shuffled", fm.modality, fm.res_um, maps, fov, fm.quality)
    ext = tuple(nt * tile_um)
    return Side(side.key, pts2, ext, fm)


def reference_he_to_if(sample):
    rec = json.loads((CACHE / f"audit_{sample}.json").read_text())
    P = rec["provider_seg_he_affine_ifpx_to_hepx"]
    S = load_sample(sample)
    he_um, if_um = S.he.frame.pixel_um, S.iff.frame.pixel_um
    if_to_he = Affine2D(np.asarray(P["M"]) * he_um / if_um, np.asarray(P["t"]) * he_um, "IF", "HE")
    return if_to_he.inverse()


def run_config(sample, name):
    method, fixed_key, visible, over, feat, extra = CONFIGS[name]
    S = load_sample(sample)
    fres = extra.get("feat_res", 2.0)
    he_f = load_feats(CACHE / f"feats_{sample}_HE_{int(fres)}um.npz")
    if_f = load_feats(CACHE / f"feats_{sample}_IF_{int(fres)}um.npz")
    he_eval = load_feats(CACHE / f"feats_{sample}_HE_2um.npz"); if_eval = load_feats(CACHE / f"feats_{sample}_IF_2um.npz")
    he = Side("HE", S.he.nuclei_um, S.he.frame.extent_um(), he_f if "HE" in visible else None)
    iff = Side("IF", S.iff.nuclei_um, S.iff.frame.extent_um(), if_f if "IF" in visible else None)
    fixed, moving = (he, iff) if fixed_key == "HE" else (iff, he)
    if "shuffle_moving_um" in extra:
        moving = shuffle_side(Side(moving.key, moving.points, moving.extent, (if_f if moving.key == "IF" else he_f) if visible else None), extra["shuffle_moving_um"])
    mode = {0: "P0", 1: "P1", 2: "P2"}[len(visible)]
    cfg = SearchConfig(**over)
    R = run_method(sample, method, fixed, moving, mode, cfg, image_feature=feat or "tissue")
    he_e = Side("HE", he.points, he.extent, None); if_e = Side("IF", iff.points, iff.extent, None)
    if "shuffle_moving_um" in extra:   # evaluate against the shuffled moving set (what the method saw)
        if_e = Side("IF", moving.points, moving.extent, None)
    ref = reference_he_to_if(sample)
    ev = evaluate(R, he_e, if_e, he_eval, if_eval if "shuffle_moving_um" not in extra else None, ref_he_to_if=ref)
    ho = [] if "shuffle_moving_um" in extra else heldout(R, he_e, if_e, fixed, moving)
    rec = {"sample": sample, "config": name, "method": method, "mode": mode, "fixed": fixed.key, "moving": moving.key,
           "visible_images": list(visible), "cfg": cfg.__dict__, "image_feature": feat, "extra": extra,
           "T_final": R.T.to_dict() if R.T else None, "T_search": R.T_search.to_dict() if R.T_search else None,
           "search": R.search, "refine": {k: v for k, v in R.refine.items()}, "failure_reasons": R.failure_reasons,
           "image_features_used": R.image_features_used, "point_used_in_search": R.point_used_in_search,
           "runtime_s": R.runtime_s, "peak_mem_mb": R.peak_mem_mb, "eval": ev, "heldout": ho,
           "feature_quality": {"HE": he_f.quality, "IF": if_f.quality}, "moving_extent": list(moving.extent), "fixed_extent": list(fixed.extent)}
    (CACHE / f"real_{sample}_{name}.json").write_text(json.dumps(json_safe(rec), indent=2))
    fin = ev.get("final", {})
    print(name, sample, R.search.get("verdict"), "le3=%.3f" % fin.get("le_3um", np.nan), "nn=%.2f" % fin.get("nn_median_um", np.nan),
          "null3=%.3f" % ev.get("null", {}).get("le_3um", np.nan), "ref=%s" % ev.get("vs_reference_um"), "rt=%.0fs" % R.runtime_s, flush=True)


def p0compat():
    """Phase B: the E0 wrapper must reproduce the existing A2 exactly."""
    import init_lib
    out = {}
    real = init_lib.load_real()
    stored = json.loads((OUT.parent / "real-registration-incremental-value-20261015/cache/real_stages.json").read_text())["stages"]["R2"]
    he = Side("HE", real["moving_px"], (real["width"], real["height"]), None)
    fx = Side("IF", real["fixed_all"], (1.0, 1.0), None)
    t0 = time.perf_counter()
    R = run_method("serial_section_pair_20260904", "E0", fx, he, "P0")
    out["old_real"] = {"max_abs_dM_vs_stored_R2": float(np.abs(R.T.M - np.asarray(stored["M"])).max()),
                       "max_abs_dt_vs_stored_R2": float(np.abs(R.T.t - np.asarray(stored["t"])).max()), "runtime_s": time.perf_counter() - t0}
    print("old real", out["old_real"], flush=True)
    for s in SAMPLES:
        S = load_sample(s)
        direct = init_lib.method_a2(S.he.nuclei_um, S.iff.nuclei_um)
        R = run_method(s, "E0", Side("IF", S.iff.nuclei_um, S.iff.frame.extent_um(), None), Side("HE", S.he.nuclei_um, S.he.frame.extent_um(), None), "P0")
        # images visible must not change P0: run the wrapper again with features attached (they must be ignored)
        he_f = load_feats(CACHE / f"feats_{s}_HE_2um.npz"); if_f = load_feats(CACHE / f"feats_{s}_IF_2um.npz")
        R2 = run_method(s, "E0", Side("IF", S.iff.nuclei_um, S.iff.frame.extent_um(), if_f), Side("HE", S.he.nuclei_um, S.he.frame.extent_um(), he_f), "P2")
        out[s] = {"max_abs_dM_wrapper_vs_direct": float(np.abs(R.T.M - direct.M).max()), "max_abs_dt_wrapper_vs_direct": float(np.abs(R.T.t - direct.t).max()),
                  "max_abs_dM_with_images_attached": float(np.abs(R2.T.M - direct.M).max()), "max_abs_dt_with_images_attached": float(np.abs(R2.T.t - direct.t).max())}
        print(s, out[s], flush=True)
    (CACHE / "p0_compat.json").write_text(json.dumps(json_safe(out), indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample"); ap.add_argument("--config"); ap.add_argument("--features"); ap.add_argument("--p0compat", action="store_true")
    a = ap.parse_args()
    if a.features:
        build_features(a.features)
    elif a.p0compat:
        p0compat()
    else:
        run_config(a.sample, a.config)
