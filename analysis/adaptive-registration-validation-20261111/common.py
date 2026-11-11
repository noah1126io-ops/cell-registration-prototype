"""Shared code of the 2026-11-11 external-validation study (adaptive-registration-validation-20261111).

Nothing here changes the engine: registration_core 1.2 is imported as-is.  The only hook is in the benchmark runner,
which replaces ``workflow.extract`` by a lookup of pre-extracted feature maps so that *derived* conditions (IF crop,
HE crop, IF mirror, cross-sample pairing) can be fed to the unchanged engine.  The maps themselves are produced by
the unchanged ``registration_core.features.extract`` on the original images (prep.py, parity-checked).

Convention (registration_core): fixed(HE)_world = M @ moving(IF)_world + t, um, frame-local.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from src.registration_core.features import FeatureMaps, Modality, extract  # noqa: E402
from src.registration_core.metrics import auc  # noqa: E402
from src.registration_explorer import catalog  # noqa: E402

CACHE = HERE / "cache"
RUNS = HERE / "runs"
FIG = HERE / "figures"
STORE = REPO / "runs/research-store/research.sqlite"
RES_UM = 2.0
LANDMARK_ANN = "ann-20261027082721-b4b29aac"       # 031-8_6, 8 pairs (user, 2026-10-27)
ACCEPTED_6_2 = "tf-20261022082737-404825d8"        # 031-6_2 ACCEPTED automatic transform
# prespecified BEFORE any run of this study (2026-11-11): a candidate is "at the reference region" when the probe
# median distance to the reference pose is <= 100 um (8_6 landmark LOO ~24 um; wrong candidates were >= 900 um).
CORRECT_UM = 100.0


# ----------------------------------------------------------------------------- read-only research store
def _db():
    return sqlite3.connect(f"file:{STORE}?mode=ro", uri=True)


def store_transform(tf_id):
    M, t = _db().execute("select M_json, t_json from transforms where id = ?", (tf_id,)).fetchone()
    return np.array(json.loads(M), float), np.array(json.loads(t), float)


def landmarks_8_6():
    return json.loads(_db().execute("select payload_json from annotations where id = ?", (LANDMARK_ANN,)).fetchone()[0])["landmarks"]


# ----------------------------------------------------------------------------- feature maps
def feat_path(sample, key):
    return CACHE / f"feat_{sample}_{key}_{RES_UM:g}.npz"


def save_feats(fm: FeatureMaps, p: Path):
    np.savez_compressed(p, frame=fm.frame, modality=fm.modality, res_um=fm.res_um, fov=fm.fov, runtime_s=fm.runtime_s,
                        source_level=fm.source_level, quality=json.dumps(fm.quality, default=float),
                        **{f"map_{k}": v for k, v in fm.maps.items()})


def load_feats(sample, key) -> FeatureMaps:
    z = np.load(feat_path(sample, key), allow_pickle=True)
    maps = {k[4:]: z[k] for k in z.files if k.startswith("map_")}
    return FeatureMaps(str(z["frame"]), str(z["modality"]), float(z["res_um"]), maps, z["fov"], json.loads(str(z["quality"])),
                       float(z["runtime_s"]), int(z["source_level"]))


def quality(fm_maps, fov, points_um, res_um):
    """Same quality definition as registration_core.features.extract, recomputed on derived (cropped / mirrored) maps."""
    tissue_bin, env, nucn = fm_maps["tissue_bin"], fm_maps["envelope"], fm_maps["nuclear"]
    H, W = tissue_bin.shape
    ij = np.rint(np.asarray(points_um) / res_um).astype(int)
    okp = (ij[:, 0] >= 0) & (ij[:, 0] < W) & (ij[:, 1] >= 0) & (ij[:, 1] < H)
    at_n = nucn[ij[okp, 1], ij[okp, 0]]
    rng = np.random.default_rng(0)
    tis_idx = np.argwhere(tissue_bin)
    rnd = tis_idx[rng.choice(len(tis_idx), min(20000, len(tis_idx)), replace=False)] if len(tis_idx) else np.zeros((0, 2), int)
    at_r = nucn[rnd[:, 0], rnd[:, 1]]
    q = {"tissue_fraction_fov": float(tissue_bin.mean()), "envelope_fraction_fov": float(env.mean()),
         "nuclei_in_tissue_fraction": float(tissue_bin[ij[okp, 1], ij[okp, 0]].mean()) if okp.any() else float("nan"),
         "nuclear_map_auc_own_nuclei_vs_tissue": auc(at_n, at_r) if len(at_n) and len(at_r) else float("nan"), "derived": True}
    q["usable"] = bool(0.02 < q["tissue_fraction_fov"] < 0.995 and (q["nuclear_map_auc_own_nuclei_vs_tissue"] or 0) > 0.7)
    q["reasons"] = [] if q["usable"] else ["derived map quality check failed"]
    return q


# ----------------------------------------------------------------------------- derived sides
class DerivedSide:
    """One modality after a list of ops.  ops: ("crop", x0, y0, w, h) um | ("mirror_x",).  Tracks the moving->new
    affine (new = A @ old + b) so reference poses can be carried along."""

    def __init__(self, sample, key):
        man = catalog.load_manifest()
        self.sample, self.key = sample, key
        self.si = catalog.side_input(man[sample], key)
        self.points = np.asarray(self.si.points, float)
        self.fm = load_feats(sample, key)
        self.extent = np.array(self.si.extent, float)
        self.A, self.b = np.eye(2), np.zeros(2)
        self.ops = []

    def apply(self, op):
        r = self.fm.res_um
        maps = dict(self.fm.maps)
        fov = self.fm.fov
        if op[0] == "crop":
            x0, y0, w, h = op[1:]
            i0, j0, ni, nj = (int(round(v / r)) for v in (y0, x0, h, w))
            maps = {k: v[i0:i0 + ni, j0:j0 + nj] for k, v in maps.items()}
            fov = fov[i0:i0 + ni, j0:j0 + nj]
            ox, oy = j0 * r, i0 * r
            keep = (self.points[:, 0] >= ox) & (self.points[:, 0] < ox + nj * r) & (self.points[:, 1] >= oy) & (self.points[:, 1] < oy + ni * r)
            self.points = self.points[keep] - [ox, oy]
            self.b = self.b - [ox, oy]
            self.extent = np.array([nj * r, ni * r])
        elif op[0] == "mirror_x":
            maps = {k: v[:, ::-1].copy() for k, v in maps.items()}
            fov = fov[:, ::-1].copy()
            W = self.extent[0]
            self.points = np.column_stack([W - self.points[:, 0], self.points[:, 1]])
            F = np.diag([-1.0, 1.0])
            self.A, self.b = F @ self.A, F @ self.b + [W, 0]
        else:
            raise ValueError(op)
        self.fm = FeatureMaps(self.fm.frame, self.fm.modality, r, maps, fov, self.fm.quality, 0.0, self.fm.source_level)
        self.ops.append(list(op))
        return self

    def finish(self, frame_name):
        self.fm = FeatureMaps(frame_name, self.fm.modality, self.fm.res_um, self.fm.maps, self.fm.fov,
                              quality(self.fm.maps, self.fm.fov, self.points, self.fm.res_um) if self.ops else self.fm.quality,
                              0.0, self.fm.source_level)
        from src.registration_core import SideInput
        s = self.si
        return SideInput(key=self.key, points=self.points, extent=tuple(float(v) for v in self.extent), frame_name=frame_name,
                         pixel_um=s.pixel_um, image_path=s.image_path, image_kind=s.image_kind, shape_px=s.shape_px,
                         pyramid_downsamples=s.pyramid_downsamples, nuclei_source=s.nuclei_source)


def carry_reference(M, t, fixed: DerivedSide, moving: DerivedSide):
    """Reference fixed = M moving + t in original frames -> derived frames (fixed' = Af f + bf, moving' = Am m + bm)."""
    Am_inv = np.linalg.inv(moving.A)
    M2 = fixed.A @ M @ Am_inv
    t2 = fixed.A @ (t - M @ Am_inv @ moving.b) + fixed.b
    return M2, t2


def probe(moving_extent, r=400.0):
    c = np.asarray(moving_extent) / 2.0
    rr = min(r, 0.4 * float(min(moving_extent)))
    return c + np.array([[0, 0], [rr, 0], [0, rr], [-rr, 0], [0, -rr]], float)


def probe_dist(Ma, ta, Mb, tb, pr):
    return float(np.median(np.linalg.norm(pr @ np.asarray(Ma).T + ta - (pr @ np.asarray(Mb).T + tb), axis=1)))


def pose_params(M):
    M = np.asarray(M, float)
    det = np.linalg.det(M)
    C = np.diag([1.0, -1.0]) if det < 0 else np.eye(2)
    sR = M @ C
    return {"scale": float(np.sqrt(abs(det))), "theta_deg": float(np.degrees(np.arctan2(sR[1, 0], sR[0, 0]))), "reflect": bool(det < 0)}


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def load_cases():
    return json.loads((HERE / "cases.json").read_text())["cases"]
