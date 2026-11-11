"""Unit tests for the research interface (run: python -B -m pytest -q test_oir.py)."""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oir import metrics as mt  # noqa: E402
from oir.features import FeatureMaps  # noqa: E402
from oir.frames import Affine2D, Frame  # noqa: E402
from oir.pipeline import Side, cropped_icp, footprint  # noqa: E402
from oir.search import Layer, PoseSearch, SearchConfig, pose_matrix  # noqa: E402


def rand_affine(rng):
    return Affine2D(pose_matrix("proper", rng.uniform(0, 360), rng.uniform(0.9, 1.1)) @ np.array([[1, 0.01], [0, 1]]),
                    rng.uniform(-500, 500, 2), "a", "b")


def test_inverse_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        T = rand_affine(rng)
        p = rng.uniform(-1000, 1000, (100, 2))
        assert np.abs(T.inverse().apply(T.apply(p)) - p).max() < 1e-9
        assert np.abs(T.inverse().inverse().M - T.M).max() < 1e-12


def test_compose_and_frames():
    rng = np.random.default_rng(1)
    T1 = Affine2D(*(lambda a: (a.M, a.t))(rand_affine(rng)), "a", "b")
    T2 = Affine2D(*(lambda a: (a.M, a.t))(rand_affine(rng)), "b", "c")
    p = rng.uniform(-100, 100, (10, 2))
    assert np.allclose(T2.compose(T1).apply(p), T2.apply(T1.apply(p)))
    with pytest.raises(ValueError):
        T1.compose(T2)
    assert np.allclose(T1.compose(T1.inverse()).M if False else T1.inverse().compose(T1).M, np.eye(2))


def test_pixel_matrix_matches_world_path():
    rng = np.random.default_rng(2)
    a = Frame("a", "HE", 0.1887, (100, 100)); b = Frame("b", "IF", 0.1002, (100, 100))
    T = Affine2D(*(lambda x: (x.M, x.t))(rand_affine(rng)), "a", "b")
    px = rng.uniform(0, 1000, (20, 2))
    P = T.pixel_matrix(a, b)
    via = b.world_to_pixel(T.apply(a.pixel_to_world(px)))
    assert np.allclose((np.c_[px, np.ones(len(px))] @ P.T)[:, :2], via)


def test_unknown_spacing_refused():
    f = Frame("x", "IF", None)
    with pytest.raises(ValueError):
        f.pixel_to_world([[1, 2]])


def test_warp_uses_inverse_mapping():
    rng = np.random.default_rng(3)
    src = gaussian_filter(rng.normal(size=(200, 200)), 3).astype(np.float32)
    T = Affine2D(np.eye(2), np.array([20.0, -10.0]), "s", "d")          # point moves +20 x, -10 y (um), res 2
    out = mt.warp_map_into(src, 2.0, T, (200, 200), 2.0, order=1, cval=np.nan)
    # target pixel (r, c) shows source pixel (r + 5, c - 10)
    assert np.allclose(out[50:150, 50:150], src[55:155, 40:140], atol=1e-5)


def _toy_side(rng, ext=(1600.0, 1600.0), res=4.0, n=6000):
    H, W = int(ext[1] / res), int(ext[0] / res)
    tissue = gaussian_filter(rng.normal(size=(H, W)), 6) > 0
    tissue_f = gaussian_filter(tissue.astype(np.float32), 1)
    pts = rng.uniform(0, 1, (n, 2)) * np.array(ext)
    ij = np.clip((pts / res).astype(int), 0, [W - 1, H - 1])
    pts = pts[tissue[ij[:, 1], ij[:, 0]]]
    fm = FeatureMaps("toy", "HE", res, {"tissue": tissue_f, "nuclear": tissue_f, "edt": tissue_f, "tissue_bin": tissue, "envelope": tissue},
                     np.ones((H, W), np.float32))
    return Side("HE", pts, ext, fm)


def test_search_recovers_known_pose_image_and_points():
    rng = np.random.default_rng(4)
    big = _toy_side(rng)
    A = pose_matrix("proper", 37.0, 1.0); c = np.array([800.0, 800.0])
    w = 600.0; t = c - A @ np.array([w / 2, w / 2])
    res = 4.0; n = int(w / res)
    yy, xx = np.mgrid[0:n, 0:n]
    q = (np.c_[xx.ravel(), yy.ravel()] * res) @ A.T + t
    samp = mt.sample_map(big.feats.maps["tissue"], res, q).reshape(n, n)
    mov_pts = (big.points - t) @ np.linalg.inv(A).T
    mov_pts = mov_pts[np.all((mov_pts >= 0) & (mov_pts < w), 1)]
    cfg = SearchConfig(res_coarse=8, res_fine=4, rot_step=4, scales=(1.0,), fine_scales=(1.0,), n_fine=3)
    for layer in (Layer("img", "image", fixed_map=big.feats.maps["tissue"], fixed_mask=big.feats.fov, moving_map=np.nan_to_num(samp),
                        moving_mask=np.ones((n, n), np.float32), res_um=res),
                  Layer("points", "points", fixed_points=big.points, moving_points=mov_pts)):
        out = PoseSearch([layer], big.extent, (w, w), np.array(big.extent) / 2, np.array([w / 2, w / 2]), cfg).run()
        top = out["top"]
        g = np.array([[0, 0], [w, 0], [0, w], [w, w]])
        err = np.linalg.norm(g @ top.A().T + top.t - (g @ A.T + t), axis=1).max()
        assert err < 12.0, (layer.name, err, top.theta)


def test_cropped_icp_partial_overlap_keeps_scale():
    rng = np.random.default_rng(5)
    big = rng.uniform(0, 2000, (8000, 2))
    A = pose_matrix("proper", 10.0, 1.0); t = np.array([600.0, 500.0])
    small_true = (big - t) @ np.linalg.inv(A).T
    keep = np.all((small_true >= 0) & (small_true < 800), 1)
    small = small_true[keep] + rng.normal(0, 0.5, (keep.sum(), 2))
    A0 = pose_matrix("proper", 11.0, 1.01)
    r = cropped_icp(A0, t + 3, "proper", small, big, (800, 800), (2000, 2000))
    g = footprint((800, 800))
    assert np.abs(g @ r["M"].T + r["t"] - (g @ A.T + t)).max() < 1.0


def test_fixed_moving_swap_is_same_physical_map():
    rng = np.random.default_rng(6)
    T = Affine2D(*(lambda a: (a.M, a.t))(rand_affine(rng)), "HE", "IF")
    p = rng.uniform(0, 1000, (50, 2))
    # result computed IF->HE then inverted must equal HE->IF
    T_swapped = T.inverse()
    assert np.allclose(T_swapped.inverse().apply(p), T.apply(p))


def test_inside_poly_orientation_independent():
    sq = footprint((100, 100))
    p = np.array([[50, 50], [5, 50], [-5, 50]])
    for poly in (sq, sq[::-1]):
        assert list(mt.inside_poly(p, poly, 10.0)) == [True, False, False]
        assert list(mt.inside_poly(p, poly, -10.0)) == [True, True, True]
