import json
from pathlib import Path

import numpy as np
import pytest

from src.registration_explorer import imaging, research_mode_enabled, results, scene
from src.registration_explorer.catalog import availability, load_manifest, load_nuclei, manifest_path

HAS_DATA = manifest_path().is_file()
needs_data = pytest.mark.skipif(not HAS_DATA, reason="research manifest / sample data not present")


def _tiff(tmp_path, shape=(1100, 1300, 3), tile=256):
    tifffile = pytest.importorskip("tifffile")
    rng = np.random.default_rng(0)
    a = rng.integers(0, 255, shape, dtype=np.uint8)
    p = tmp_path / "t.ome.tif"
    with tifffile.TiffWriter(p) as w:
        w.write(a, tile=(tile, tile), subifds=1, photometric="rgb")
        w.write(a[::2, ::2], tile=(tile, tile), subfiletype=1, photometric="rgb")
    return str(p), a


def test_read_region_matches_whole_level(tmp_path):
    p, a = _tiff(tmp_path)
    got = imaging.read_region(p, 0, 37, 250, 801, 1033)
    assert np.array_equal(got, a[250:1033, 37:801])


def test_whole_read_of_large_level_is_refused(tmp_path, monkeypatch):
    p, _ = _tiff(tmp_path)
    monkeypatch.setattr(imaging, "MAX_FULL_LEVEL_PIXELS", 1000)
    imaging.read_level_cached.cache_clear()
    with pytest.raises(imaging.ImageUnavailable):
        imaging.read_level_cached(p, 0)


def test_missing_image_is_unavailable_not_fatal():
    with pytest.raises(imaging.ImageUnavailable):
        imaging.pyramid_info("/nonexistent/x.ome.tif")


def test_warp_preview_inverse_map_agrees_with_forward_points():
    # source image with one bright pixel at level-0 index (40, 25); forward-map its world position and
    # check the inverse-warped target image peaks there.
    src = np.zeros((60, 80, 3), np.uint8); src[25, 40] = 255
    um = 0.5
    th = np.deg2rad(17.0); s = 1.3
    M = s * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]); t = np.array([12.0, -3.0])
    corner = imaging.level_corner_um(um)
    out, valid, rec = imaging.warp_preview(src, um, corner, M, t, -0.25, (80.0, 70.0), 0.5, order=1)
    p_world = M @ (np.array([40, 25]) * um) + t
    j, i = np.unravel_index(np.argmax(out[..., 0]), out.shape[:2])[::-1]
    centre = -0.25 + (np.array([j, i]) + 0.5) * 0.5
    assert np.linalg.norm(centre - p_world) <= 0.5 * 1.0 + 1e-6
    assert rec["mapping"].startswith("inverse") and valid.dtype == bool
    assert not valid[0, 0] or out[0, 0].sum() == 0


def test_oriented_inverse_round_trip():
    M = np.array([[0.98, -0.01], [0.01, 0.99]]); t = np.array([540.0, 490.0])
    Mi, ti, derived = results.oriented(M, t, "HE", "IF")
    assert derived
    p = np.array([[10.0, 20.0], [500.0, 800.0]])
    back = (p @ M.T + t) @ Mi.T + ti
    assert np.allclose(back, p)
    M2, t2, d2 = results.oriented(M, t, "HE", "HE")
    assert not d2 and np.allclose(M2, M)
    with pytest.raises(TypeError):
        results.oriented(M, t, object(), "HE")


def test_safety_labels_never_promote_preview():
    base = dict(rid="x", sample="s", method="E2", config="c", fixed="HE", moving="IF", M=np.eye(2), t=np.zeros(2), margin=0.1,
                failure_reasons=[], fallback_source=None, metrics={}, candidates=[], source="t", files=[])
    assert results.ResultRecord(verdict="UNIQUE", accepted=True, status_flags=[], **base).safety_label() == "UNIQUE / Accepted"
    assert results.ResultRecord(verdict="AMBIGUOUS", accepted=False, status_flags=[], **base).safety_label() == "AMBIGUOUS"
    assert results.ResultRecord(verdict="LOW_PEAK", accepted=False, status_flags=["POINT_ONLY_FALLBACK"], **base).safety_label() == "POINT_ONLY_FALLBACK"
    assert results.ResultRecord(verdict="UNIQUE", accepted=False, status_flags=[], **base).safety_label() == "FAIL"
    assert results.ResultRecord(verdict="UNIQUE", accepted=True, status_flags=["IMAGE_INCOMPATIBLE"], **base).safety_label() == "IMAGE_UNUSABLE"


def test_research_mode_flag_default_off(monkeypatch):
    monkeypatch.delenv("CELLREG_RESEARCH_MODE", raising=False)
    assert not research_mode_enabled()
    monkeypatch.setenv("CELLREG_RESEARCH_MODE", "1")
    assert research_mode_enabled()


@needs_data
def test_saved_results_load_without_recomputation():
    S = load_manifest()
    recs62 = results.load_records(S["031-6_2"])
    e0x = results.find(recs62, "oir:031-6_2:E0x_HEfix")
    assert e0x is not None and e0x.accepted and e0x.safety_label() == "UNIQUE / Accepted"
    assert abs(e0x.metrics["nn_median_um"] - 1.91) < 0.01 and abs(e0x.metrics["le_3um"] - 0.599) < 0.001
    e2 = results.find(recs62, "oir:031-6_2:E2_HEfix")
    assert e2.safety_label() == "POINT_ONLY_FALLBACK" and e2.fallback_source == "E0x_HEfix"
    recs86 = results.load_records(S["031-8_6"])
    e86 = results.find(recs86, "oir:031-8_6:E0x_HEfix")
    assert e86.safety_label() == "AMBIGUOUS" and not e86.accepted and len(e86.candidates) >= 2


@needs_data
def test_point_only_sample_and_image_hiding():
    S = load_manifest()
    po = S["serial_section_pair_20260904"]
    av = availability(po)
    assert not av["HE"]["image"] and not av["IF"]["image"] and av["HE"]["nuclei"]
    av2 = availability(S["031-6_2"], simulate_image_failure=True)
    assert not av2["HE"]["image"] and "simulated" in av2["HE"]["image_reason"]
    rec = results.load_records(po)[0]
    sc, diag, _ = scene.build_scene(po, av, "IF", rec.M, rec.t, state_label="t", banners=[], opts=scene.ImageOptions())
    assert sc["images"] == [] and len(sc["points"]) == 2 and diag["n_eval"] > 100


@needs_data
def test_fixed_moving_swap_same_physical_map_031_6_2():
    S = load_manifest()["031-6_2"]
    recs = results.load_records(S)
    a = results.find(recs, "oir:031-6_2:E0x_HEfix"); b = results.find(recs, "oir:031-6_2:E0x_IFfix")
    Ma, ta, _ = results.oriented(a.M, a.t, a.fixed, "IF")
    g = np.stack(np.meshgrid(np.linspace(0, 1026, 11), np.linspace(0, 1026, 11)), -1).reshape(-1, 2)
    he = (g - b.t) @ np.linalg.inv(b.M).T
    assert np.median(np.linalg.norm(he @ Ma.T + ta - g, axis=1)) < 1.0


def test_manual_adjustment_rotates_and_scales_about_pivot():
    from src.registration_explorer import manual
    M = np.array([[0.9, -0.1], [0.1, 0.9]]); t = np.array([100.0, 50.0])
    pivot = (300.0, 200.0)
    adj = manual.Adjustment(rotation_deg=90.0, scale=2.0, dx=5.0, dy=-3.0, pivot=pivot)
    M2, t2 = manual.compose(M, t, adj)
    p = np.array([[10.0, 20.0], [400.0, -30.0]])
    base = p @ M.T + t
    rel = base - np.asarray(pivot)
    expected = 2.0 * np.c_[-rel[:, 1], rel[:, 0]] + np.asarray(pivot) + [5.0, -3.0]   # +90° in frame coords (x, y) -> (-y, x)
    assert np.allclose(p @ M2.T + t2, expected)
    # the pivot itself only moves by the translation
    q = np.linalg.solve(M, np.asarray(pivot) - t)
    assert np.allclose(M2 @ q + t2, np.asarray(pivot) + [5.0, -3.0])
    assert manual.Adjustment().is_identity() and not adj.is_identity()
    Mi, ti = manual.compose(M, t, manual.Adjustment(pivot=pivot))
    assert np.allclose(Mi, M) and np.allclose(ti, t)
    with pytest.raises(ValueError):
        manual.compose(M, t, manual.Adjustment(scale=0.0))


def test_manual_transform_saved_as_unverified_record(tmp_path, monkeypatch):
    from src.registration_explorer import manual
    monkeypatch.setattr(manual, "MANUAL_DIR", tmp_path)
    adj = manual.Adjustment(rotation_deg=1.5, scale=1.01, dx=2.0, dy=0.0, pivot=(10.0, 10.0))
    M, t = manual.compose(np.eye(2), np.zeros(2), adj)
    rec = manual.build_record(sample="s1", fixed="HE", moving="IF", base={"label": "Before"}, adj=adj, M=M, t=t, name="試し 1")
    p1, p2 = manual.save(rec), manual.save(rec)
    assert p1 != p2 and p1.is_file() and p2.is_file()            # never overwrites
    (tmp_path / "s1" / "broken.json").write_text('{"schema": "registration-explorer-manual/1", "sample": "s1"}')
    recs = results._manual_records("s1")
    assert len(recs) == 2                                         # malformed file skipped
    r = recs[0]
    assert r.method == "MAN" and r.accepted is False and r.safety_label() == "MANUAL (UNVERIFIED)"
    assert r.fixed == "HE" and r.moving == "IF" and np.allclose(r.M, M) and np.allclose(r.t, t)
    assert results._manual_records("other") == []


def test_config_key_unchanged_for_old_runs_and_split_by_image_feature():
    from src.registration_explorer import jobs
    old = {"engine": jobs.ENGINE, "sample": "031-8_6", "fixed": "HE", "localization": "auto", "use_images": True, "annotation_ids": [], "constraints": None}
    k_old = jobs.config_key(old)
    import hashlib, json as _json
    assert k_old == hashlib.sha256(_json.dumps(old, sort_keys=True).encode()).hexdigest()[:16]      # same formula as before 10-27
    assert jobs.config_key({**old, "image_feature": "occ+cleft"}) != k_old
    assert jobs.config_key({**old, "image_feature": "occ+cleft"}) != jobs.config_key({**old, "image_feature": "tissue+nuclear"})
    assert list(jobs.IMAGE_FEATURES)[0] == "occ+cleft"
