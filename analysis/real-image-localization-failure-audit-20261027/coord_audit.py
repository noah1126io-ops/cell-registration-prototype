"""Test 4: coordinate-system checks (pixel <-> world, forward / inverse, search canvas image vs point rendering,
Explorer display orientation).  Output: cache/coord_audit.json"""
import json

import numpy as np

from common import CACHE, features, side, side_input, store_transform
from src.registration_core.search import Canvas, rot
from src.registration_explorer import results, scene

out = {}
# 1. pixel <-> world round trip (world = px * pixel_um, frame-local origin) for both images of both samples
for sid in ("031-8_6", "031-6_2"):
    for k in ("HE", "IF"):
        si = side_input(sid, k)
        px = np.random.default_rng(0).uniform(0, min(si.shape_px), (1000, 2))
        w = px * si.pixel_um
        out[f"pixel_world_roundtrip_max_px:{sid}:{k}"] = float(np.abs(w / si.pixel_um - px).max())
        out[f"extent_vs_shape:{sid}:{k}"] = {"extent_um": list(si.extent), "shape_px_hw": list(si.shape_px), "pixel_um": si.pixel_um,
                                             "nuclei_bbox_um": [np.asarray(si.points).min(0).round(1).tolist(), np.asarray(si.points).max(0).round(1).tolist()]}
        q = features(sid, k).quality
        out[f"own_nuclei_on_own_image_auc:{sid}:{k}"] = q["nuclear_map_auc_own_nuclei_vs_tissue"]
# 2. forward / inverse of the manual reference and Explorer orientation
M, t, _ = store_transform()
p = np.random.default_rng(1).uniform(0, 1026, (500, 2))
Mi, ti, derived = results.oriented(M, t, "HE", "IF")
out["forward_inverse_roundtrip_max_um"] = float(np.abs(scene.apply(Mi, ti, scene.apply(M, t, p)) - p).max())
out["oriented_flags_inverse"] = bool(derived)
# 3. search canvas: an image impulse and a point at the same moving location must land on the same canvas pixel
F, Mv = side("031-8_6", "HE", with_image=False), side("031-8_6", "IF", with_image=False)
for res in (8.0, 4.0):
    cv = Canvas(F.extent, Mv.extent, np.asarray(F.extent) / 2, res, 1.2)
    A = 1.07 * rot(23.0); t0 = cv.center - A @ (np.asarray(Mv.extent) / 2)
    src_res = 4.0
    errs = []
    for xy in ([300.0, 200.0], [700.0, 850.0], [512.0, 512.0]):
        img = np.zeros((int(1026 / src_res) + 1,) * 2, np.float32)
        r, c = int(round(xy[1] / src_res)), int(round(xy[0] / src_res)); img[r - 1:r + 2, c - 1:c + 2] = 1.0
        im = cv.render_map(img, src_res, A, t0)
        pt = cv.raster_points((np.array([[c * src_res, r * src_res]]) @ A.T + t0), 4.0)
        a = np.array(np.unravel_index(np.argmax(im), im.shape), float); b = np.array(np.unravel_index(np.argmax(pt), pt.shape), float)
        errs.append(float(np.linalg.norm(a - b) * res))
    out[f"canvas_image_vs_point_um@{res:g}"] = errs
(CACHE / "coord_audit.json").write_text(json.dumps(out, indent=1, default=float))
print(json.dumps(out, indent=1, default=float))
