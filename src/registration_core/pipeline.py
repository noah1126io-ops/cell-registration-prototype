"""Method runners E0 / E0x / E1 / E2 / E3 with a common result + artifact, and common evaluation.

E0   existing A2 (init_lib.method_a2 + its production ICP refine), called unmodified.        P0
E0x  point-only partial-overlap search (union canvas, proper chirality) + cropped production ICP. P0
E1   E0x candidates jointly scored with a cross layer (image side tissue vs other side nuclei).   P1
E2   image-only structural search (tissue occupancy) -> cropped production ICP in a trust region. P2
E3   joint image + point search (explicit z-combination rule) -> cropped production ICP.         P2
Evaluation is always reported in the IF frame, whichever side was fixed.
"""
from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass, field

import numpy as np

from ..pointset_registration import SimilarityTransform, affine_icp, icp_similarity  # noqa: E402
from . import a2 as init_lib  # noqa: E402  (A2 vendored verbatim from analysis/affine-global-initialization-20261014/init_lib.py)

from . import metrics as mt  # noqa: E402
from .features import FeatureMaps, coarsen, point_density  # noqa: E402
from .frames import Affine2D, TransformArtifact  # noqa: E402
from .search import CHIR, Layer, PoseSearch, SearchConfig, pose_matrix, rot  # noqa: E402

# prespecified acceptance (reuses the A2 detector thresholds; not tuned on these samples)
ACCEPT = {"inlier_radius_um": 3.0, "min_inlier": 0.25, "min_inlier_over_null": 3.0}
TRUST = {"max_scale_change": 0.03, "max_rot_change_deg": 2.0, "max_shift_um": 15.0}


@dataclass
class Side:
    key: str              # HE | IF
    points: np.ndarray    # world um
    extent: tuple         # imaged extent (um), from frame metadata (not from points)
    feats: FeatureMaps | None   # None if the image is hidden / absent


@dataclass
class RunResult:
    sample: str
    method: str
    mode: str
    fixed: str
    moving: str
    T: Affine2D | None                 # moving world -> fixed world (final)
    T_search: Affine2D | None
    search: dict = field(default_factory=dict)
    refine: dict = field(default_factory=dict)
    status: list = field(default_factory=list)
    failure_reasons: list = field(default_factory=list)
    runtime_s: float = 0.0
    peak_mem_mb: float = 0.0
    image_features_used: list = field(default_factory=list)
    point_used_in_search: bool = True


def footprint(extent):
    w, h = extent
    return np.array([[0, 0], [w, 0], [w, h], [0, h]], float)


# ----------------------------------------------------------------------------- refinement
def cropped_icp(A, t, chir, moving, fixed, mov_extent, fix_extent, margin=20.0):
    """Production icp_similarity -> affine_icp, using only points inside the other side's mapped footprint."""
    C = CHIR[chir]

    def crop(M, tt):
        Tm = Affine2D(M, tt, "m", "f")
        mov_in = mt.inside_poly(Tm.apply(moving), footprint(fix_extent), -margin)
        fix_in = mt.inside_poly(Tm.inverse().apply(fixed), footprint(mov_extent), -margin)
        return mov_in, fix_in

    mi, fi = crop(A, t)
    if mi.sum() < 20 or fi.sum() < 20:
        return None
    sR = A @ C                      # = s R
    s0 = float(np.sqrt(abs(np.linalg.det(sR))))
    th0 = float(np.arctan2(sR[1, 0], sR[0, 0]))
    sim, _, _ = icp_similarity(moving[mi] @ C.T, fixed[fi], initial=SimilarityTransform(s0, th0, np.asarray(t, float)),
                               max_iter=80, trim_quantile=0.8)
    Ms, ts = sim.to_affine()
    M_sim = Ms @ C
    mi, fi = crop(M_sim, ts)
    Mi, ti = Ms, ts
    err, n = np.nan, 0
    for _ in range(2):
        Mi, ti, err, n = affine_icp(moving[mi] @ C.T, fixed[fi], Mi, ti, max_iter=80, trim_quantile=0.7)
        mi, fi = crop(Mi @ C, ti)
    return {"M_sim": M_sim, "t_sim": ts, "M": Mi @ C, "t": ti, "objective": float(err), "n_pairs": int(n),
            "n_moving_used": int(mi.sum()), "n_fixed_used": int(fi.sum())}


def pose_change(T0: Affine2D, T1: Affine2D, probe):
    p0, p1 = T0.params(), T1.params()
    d = np.linalg.norm(T0.apply(probe) - T1.apply(probe), axis=1)
    dr = (p1["rotation_deg"] - p0["rotation_deg"] + 180) % 360 - 180
    return {"scale_change": p1["scale"] / p0["scale"] - 1, "rot_change_deg": dr, "median_shift_um": float(np.median(d)),
            "max_shift_um": float(d.max())}


# ----------------------------------------------------------------------------- layers
def _fov_mask(fm):
    return fm.fov.astype(np.float32)


def build_layers(method, fixed: Side, moving: Side, res, image_feature="tissue"):
    pts = Layer("points", "points", fixed_points=fixed.points, moving_points=moving.points, res_um=res)
    if method == "E0x":
        return [pts], []
    if method == "E1":
        img_side = fixed if fixed.feats is not None else moving
        fm = coarsen(img_side.feats, res)
        if img_side is fixed:   # fixed image tissue vs moving nuclei density
            cross = Layer("cross", "cross", fixed_map=fm.maps["tissue"], fixed_mask=_fov_mask(fm), moving_points=moving.points, res_um=res)
        else:                   # fixed nuclei density vs moving image tissue
            cross = Layer("cross", "cross", fixed_points=fixed.points, moving_map=fm.maps["tissue"], moving_mask=_fov_mask(fm), res_um=res)
        return [pts, cross], [f"{img_side.key}:tissue (cross vs nuclei)"]
    ff, mf = coarsen(fixed.feats, res), coarsen(moving.feats, res)
    feats = image_feature.split("+")
    img_layers = [Layer(f"img_{k}", "image", fixed_map=ff.maps[k], fixed_mask=_fov_mask(ff), moving_map=mf.maps[k],
                        moving_mask=_fov_mask(mf), res_um=res) for k in feats]
    used = [f"HE:{k}" for k in feats] + [f"IF:{k}" for k in feats]
    if method == "E2":
        return img_layers, used
    if method == "E3":
        return img_layers + [pts], used
    raise ValueError(method)


def run_search(method, fixed: Side, moving: Side, cfg: SearchConfig, image_feature="tissue"):
    # build layers at the fine resolution; the search coarsens maps on the fly by rendering at canvas res
    layers, used = build_layers(method, fixed, moving, cfg.res_fine, image_feature)
    fc = np.asarray(fixed.extent) / 2.0
    mc = np.asarray(moving.extent) / 2.0
    ps = PoseSearch(layers, fixed.extent, moving.extent, fc, mc, cfg)
    return ps.run(), used


# ----------------------------------------------------------------------------- method entry
def run_method(sample, method, fixed: Side, moving: Side, mode, cfg: SearchConfig | None = None, image_feature="tissue",
               trust_region=None):
    cfg = cfg or SearchConfig()
    tracemalloc.start(); t_start = time.perf_counter()
    R = RunResult(sample, method, mode, fixed.key, moving.key, None, None)
    fname, mname = f"{sample}:{fixed.key}", f"{sample}:{moving.key}"
    if method == "E0":
        res = init_lib.method_a2(moving.points, fixed.points)        # existing A2 + production refine, unmodified
        R.T = Affine2D(res.M, res.t, mname, fname)
        hyp = res.refined[0][3]
        R.T_search = Affine2D(pose_matrix(hyp["chirality"], hyp["theta"], hyp["scale"]), hyp["translation"], mname, fname)
        R.search = {"verdict": "A2", "peak_ncc": res.diagnostics.get("peak_ncc"), "peak_z": res.diagnostics.get("peak_robust_z"),
                    "basin_margin": res.diagnostics.get("basin_margin"), "n_hypotheses": res.n_hypotheses_scored, "top": hyp}
        R.refine = {"objective": res.objective}
        R.image_features_used = []
    else:
        out, used = run_search(method, fixed, moving, cfg, image_feature)
        R.image_features_used = used
        R.point_used_in_search = method in ("E0x", "E1", "E3")
        top = out["top"]
        R.search = {"verdict": out["verdict"], "margin": out["margin"], "n_poses": out["n_poses"], "search_runtime_s": out["runtime_s"],
                    "top": top.as_dict() if top else None, "second": out["second"].as_dict() if out["second"] else None,
                    "fine": [c.as_dict() for c in out["fine"][:6]], "coarse_top": [c.as_dict() for c in out["coarse_top"][:6]]}
        if top is None:
            R.failure_reasons.append("no search candidate")
        else:
            A = top.A()
            R.T_search = Affine2D(A, top.t, mname, fname)
            ref = cropped_icp(A, top.t, top.chirality, moving.points, fixed.points, moving.extent, fixed.extent)
            if ref is None:
                R.failure_reasons.append("too few points in overlap for ICP")
                R.T = R.T_search
            else:
                T_icp = Affine2D(ref["M"], ref["t"], mname, fname)
                probe = footprint(moving.extent)
                ch = pose_change(R.T_search, T_icp, probe)
                ref["pose_change"] = ch
                R.refine = ref
                tr = trust_region if trust_region is not None else (TRUST if method in ("E2", "E3", "E1") else None)
                if tr and (abs(ch["scale_change"]) > tr["max_scale_change"] or abs(ch["rot_change_deg"]) > tr["max_rot_change_deg"]
                           or ch["median_shift_um"] > tr["max_shift_um"]):
                    R.refine["icp_rejected"] = "ICP left the search trust region"
                    R.T = R.T_search
                else:
                    R.T = T_icp
    R.runtime_s = time.perf_counter() - t_start
    R.peak_mem_mb = tracemalloc.get_traced_memory()[1] / 2**20
    tracemalloc.stop()
    return R


# ----------------------------------------------------------------------------- evaluation (IF frame)
def to_if(T: Affine2D, fixed_key):
    """Return HE-world -> IF-world transform from a moving->fixed result."""
    return T if fixed_key == "IF" else T.inverse()


def evaluate(R: RunResult, he: Side, iff: Side, he_full: FeatureMaps | None, if_full: FeatureMaps | None, eval_res=2.0,
             ref_he_to_if: Affine2D | None = None, block_um=100.0):
    if R.T is None:
        return {"ok": False}
    T = to_if(R.T, R.fixed)                         # HE world -> IF world
    he_w = T.apply(he.points)
    if_fp = footprint(iff.extent)
    he_fp_in_if = T.apply(footprint(he.extent))
    if_mask = mt.inside_poly(iff.points, he_fp_in_if, 10.0)        # IF nuclei imaged by HE
    he_mask = mt.inside_poly(he_w, if_fp, -10.0)
    ev = {"ok": True, "params_he_to_if": T.params(), "n_if_eval": int(if_mask.sum()), "n_he_in_if_field": int(he_mask.sum())}
    ev["final"] = mt.proximity(he_w[he_mask], iff.points, if_mask)
    ev["null"] = mt.mirror_null(he_w[he_mask], iff.points, if_mask)
    ev["inlier_over_null"] = ev["final"]["le_3um"] / max(ev["null"]["le_3um"], 1e-6)
    if R.T_search is not None:
        Ts = to_if(R.T_search, R.fixed)
        ev["search_pose"] = mt.proximity(Ts.apply(he.points), iff.points, if_mask)
    # geometric acceptance (prespecified)
    ev["geometric_pass"] = bool(ev["final"]["le_3um"] >= ACCEPT["min_inlier"] and ev["inlier_over_null"] >= ACCEPT["min_inlier_over_null"])
    # reference comparison (provider transform for the same-section pair)
    if ref_he_to_if is not None:
        g = np.stack(np.meshgrid(np.linspace(0, iff.extent[0], 21), np.linspace(0, iff.extent[1], 21)), -1).reshape(-1, 2)
        # IF grid -> HE via both; distance measured in IF frame after mapping back with ours
        d = np.linalg.norm(T.apply(ref_he_to_if.inverse().apply(g)) - g, axis=1)
        ev["vs_reference_um"] = {"median": float(np.median(d)), "p95": float(np.percentile(d, 95)), "max": float(d.max())}
        ev["region_agrees_reference"] = bool(np.median(d) <= 20.0)
    # image consistency in the IF frame (HE maps warped by the inverse map)
    if he_full is not None and if_full is not None:
        shp = if_full.shape
        Tm = Affine2D(T.M, T.t, "HE", "IF")
        valid = np.isfinite(mt.warp_map_into(np.ones(he_full.shape, np.float32), he_full.res_um, Tm, shp, eval_res, order=0))
        img = {}
        for k in ("nuclear", "tissue"):
            w = mt.warp_map_into(he_full.maps[k], he_full.res_um, Tm, shp, eval_res)
            img[f"{k}_corr"] = mt.masked_corr(w, if_full.maps[k], valid)
            bc = mt.block_corrs(w, if_full.maps[k], valid, int(round(block_um / eval_res)))
            img[f"{k}_block_corr_median"] = float(np.median(bc)) if len(bc) else np.nan
            img[f"{k}_block_corr_frac_pos"] = float(np.mean(bc > 0.1)) if len(bc) else np.nan
        wt = mt.warp_map_into(he_full.maps["tissue_bin"].astype(np.float32), he_full.res_um, Tm, shp, eval_res)
        img["tissue_dice"] = mt.dice(wt, if_full.maps["tissue_bin"].astype(np.float32), valid)
        we = mt.warp_map_into(he_full.maps["envelope"].astype(np.float32), he_full.res_um, Tm, shp, eval_res, order=0) > 0.5
        img["boundary_chamfer_um"], img["n_boundary_px"] = mt.boundary_chamfer(if_full.maps["envelope"], we, valid, eval_res)
        img["valid_fraction_of_if"] = float(valid.mean())
        img["independent_of_search"] = {k: (f"HE:{k.split('_')[0]}" not in R.image_features_used) for k in ("nuclear_corr", "tissue_corr")}
        ev["image"] = img
    return ev


def heldout(R: RunResult, he: Side, iff: Side, fixed: Side, moving: Side, block_um=100.0):
    """Refinement-only held-out: ICP from the method's search pose with HE nuclei of one checker colour,
    evaluated on IF nuclei in the other colour's region against HE nuclei of that colour (the search is not re-run)."""
    if R.T_search is None or R.method == "E0":
        if R.method != "E0":
            return []
    out = []
    he_col = mt.checker_colour(he.points, block_um)
    for k in (0, 1):
        sel = he_col == k
        if R.method == "E0":
            r = init_lib.refine(init_lib.Hypothesis(R.search["top"]["chirality"], R.search["top"]["theta"], R.search["top"]["scale"],
                                                     np.asarray(R.search["top"]["translation"]), 0.0, "E0 heldout"),
                                he.points[sel] if moving.key == "HE" else moving.points, fixed.points if moving.key == "HE" else he.points[sel])
            M, t = r[0], r[1]
        else:
            top = R.search["top"]
            A = pose_matrix(top["chirality"], top["theta"], top["scale"])
            mp = he.points[sel] if moving.key == "HE" else moving.points
            fp = fixed.points if moving.key == "HE" else he.points[sel]
            ref = cropped_icp(A, np.asarray(top["t"]), top["chirality"], mp, fp, moving.extent, fixed.extent)
            if ref is None:
                out.append({"fold": k, "le_3um": np.nan}); continue
            M, t = ref["M"], ref["t"]
            if R.refine.get("icp_rejected"):
                M, t = A, np.asarray(top["t"])
        Tk = to_if(Affine2D(M, t, "m", "f"), R.fixed)
        hw = Tk.apply(he.points)
        # IF nuclei whose location falls in the held-out colour (colour defined in HE world -> map IF to HE)
        if_in_he = Tk.inverse().apply(iff.points)
        if_col = mt.checker_colour(if_in_he, block_um)
        m_eval = (if_col != k) & mt.inside_poly(iff.points, Tk.apply(footprint(he.extent)), 10.0)
        pr = mt.proximity(hw[he_col != k], iff.points, m_eval)
        nl = mt.mirror_null(hw[he_col != k], iff.points, m_eval)
        out.append({"fold": k, **pr, "null_le_3um": nl["le_3um"]})
    return out


def make_artifact(R: RunResult, ev, frames: dict, image_status, accepted, extra_reasons=(), moving_extent=None):
    T = R.T
    return TransformArtifact(
        sample=R.sample, method=R.method, mode=R.mode,
        source_frame=frames[R.moving].to_dict(), target_frame=frames[R.fixed].to_dict(),
        forward_transform=T.to_dict() if T else {}, inverse_transform=T.inverse().to_dict() if T else {},
        transform_direction=f"computed {R.moving}->{R.fixed} (moving->fixed); inverse stored for resampling",
        output_frame=frames[R.fixed].name, model="affine (similarity ICP -> affine ICP)" if R.refine.get("M") is not None and not R.refine.get("icp_rejected") else "similarity (search pose)",
        roi={"evaluation": "IF nuclei inside the mapped HE footprint (10 um inset)", "frames": "full imaged extents"},
        validity_region_target_um=(T.apply(footprint(moving_extent)).tolist() if (T is not None and moving_extent is not None) else []),
        inputs={"points": True, "images_used_in_search": R.image_features_used, "point_used_in_search": R.point_used_in_search},
        image_status=image_status, confidence={"search": {k: R.search.get(k) for k in ("verdict", "margin", "peak_ncc", "peak_z")},
                                                 "geometric_pass": ev.get("geometric_pass"), "inlier_over_null": ev.get("inlier_over_null")},
        failure_reasons=list(R.failure_reasons) + list(extra_reasons), accepted=bool(accepted))
