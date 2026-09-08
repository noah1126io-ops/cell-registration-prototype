# data20260619 — HE ↔ GeoJSON alignment (self-contained, S2500586-1 example)

Reproduces `ALIGN_HE_TO_GEOJSON.md`: warp one HE image so its nuclei centers
overlay one fluorescence nuclei GeoJSON. No SpatialArc / bundling.

## Contents

| file | role |
|------|------|
| `sample4_cropped.jpg` | HE image (S2500586-1, 901×901 px, ~1.13 µm/px) |
| `S2500586-1_dl20_um.geojson` | fluorescence nuclei segmentation (world µm, Y-flipped) |
| `he_nuclei_stardist_S2500586-1.npy` | precomputed StarDist HE nucleus centers (HE px frame) |
| `align_he_to_geojson.py` | **main** — affine + fine center-snap warp + overlay |
| `stardist_detect.py` | regenerate the `.npy` (needs internet/StarDist) |
| `register_he_to_geojson.py` | dependency: nucleus detectors + similarity ICP |
| `ALIGN_HE_TO_GEOJSON.md` | method writeup |

## Run

```
# (optional) regenerate centers — needs StarDist + internet, run locally:
pip install stardist csbdeep tensorflow scikit-image pillow
python stardist_detect.py

# main step — needs only numpy/scipy/scikit-image/pillow/matplotlib:
pip install numpy scipy scikit-image pillow matplotlib
python align_he_to_geojson.py
```

Outputs in this folder:
- `warpedHE_S2500586-1_dl20_um_yflip.png` — warped HE on the GeoJSON world-µm grid
- `warp_S2500586-1_dl20_um.json` — `image_bounds [x0,y0,x1,y1]`, flip, Jacobian min
- `overlay_S2500586-1_dl20_um.png` — warped HE + GeoJSON nuclei (cyan), zoomed

Expected (S2500586-1): mutual-pair median 5.78 µm → ~2.6 µm, within-3 µm ~56%,
Jacobian min > 0. If `--centroids` is absent, add `--allow-classical`.
Tunables: `--sig-k --lam --r-match --sig-c --sig-r`.
