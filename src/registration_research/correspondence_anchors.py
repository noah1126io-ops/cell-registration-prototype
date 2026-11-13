"""Anchor sets for the local-deformation comparison (moving -> fixed displacements, both already in one frame).

  mnn         mutual nearest neighbours <= 10 um (src.pointset_registration.mutual_nearest_pairs, production helper)
  b2          B2 matched pairs (frozen adoption)
  b2_weighted B2 matched pairs, displacement x calibrated confidence (same shrinkage idea as UOT conservative)
UOT conservative builds its own barycentric anchors inside validation_lib.run_uot and is not duplicated here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.pointset_registration import mutual_nearest_pairs

MIN_ANCHORS = 20          # prespecified (experiment_protocol.md §4.1): fewer -> Affine only
MNN_RADIUS_UM = 10.0      # the production fine-warp default match radius


@dataclass
class Anchors:
    name: str
    moving_index: np.ndarray
    fixed_index: np.ndarray
    positions: np.ndarray        # moving positions
    displacements: np.ndarray    # fixed - moving (possibly shrunk)
    weights: np.ndarray

    def __len__(self):
        return int(len(self.moving_index))


def mnn(moving, fixed, radius=MNN_RADIUS_UM) -> Anchors:
    mi, fi, _ = mutual_nearest_pairs(moving, fixed, max_distance=radius)
    return Anchors("mnn", mi, fi, moving[mi], fixed[fi] - moving[mi], np.ones(len(mi)))


def b2(moving, fixed, frame, weighted=False) -> Anchors:
    m = frame["status"].to_numpy(str) == "matched"
    mi = frame["moving_index"].to_numpy(int)[m]
    fi = frame["top1_fixed_index"].to_numpy(int)[m]
    w = frame["calibrated_confidence"].to_numpy(float)[m] if weighted else np.ones(len(mi))
    d = (fixed[fi] - moving[mi]) * np.minimum(w, 1.0)[:, None]
    return Anchors("b2_weighted" if weighted else "b2", mi, fi, moving[mi], d, w)


def quality(a: Anchors, true_fixed_index=None, extent_cells=None, cell_um=100.0):
    """Anchor count, precision against truth (synthetic), spatial coverage over 100 um cells of the moving cloud."""
    out = {"anchors_n": len(a)}
    if true_fixed_index is not None and len(a):
        tfi = np.asarray(true_fixed_index, int)
        out["anchor_precision"] = float(np.mean(tfi[a.moving_index] == a.fixed_index))
    if extent_cells is not None:
        cells = {tuple(c) for c in np.floor(a.positions / cell_um).astype(int)} if len(a) else set()
        out["anchor_coverage_cells"] = len(cells & extent_cells) / max(len(extent_cells), 1)
    return out


def occupied_cells(points, cell_um=100.0, min_n=3):
    c, n = np.unique(np.floor(np.asarray(points) / cell_um).astype(int), axis=0, return_counts=True)
    return {tuple(x) for x, k in zip(c, n) if k >= min_n}
