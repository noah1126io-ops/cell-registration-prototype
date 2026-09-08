# Aligning one HE image to one nuclei GeoJSON (example: S2500586-1)

> **Scope: research.** Lives in `tmp/seqfish_20260520/reg_check/`.
> This note covers ONLY the core problem: take **one HE image** and **one
> fluorescence-derived nuclei GeoJSON** and produce a **warped HE that overlays
> the GeoJSON**. Bundling / SpatialArc / multi-sample batching are out of scope.

## Inputs (S2500586-1)

| input | file | frame |
|-------|------|-------|
| HE image | `sample4_cropped.jpg` | 901×901 px, ~1.13 µm/px |
| nuclei GeoJSON | `S2500586-1_dl20_um.geojson` | world µm, Y-flipped (`image_height_um=1017.856`) |

The GeoJSON is the segmentation of the **fluorescence** image (nuclei polygons +
`centroid_x/centroid_y` in µm). HE and fluorescence are **adjacent serial
sections**, so they differ by a global transform plus a small, spatially-varying
distortion. We register HE → GeoJSON frame.

## Goal

Output a warped HE raster on the GeoJSON's world-µm grid such that **HE nucleus
centers land on the GeoJSON centroids** where a real correspondence exists, and
regions without a confident match are left essentially unchanged.

## Method (3 stages)

### 1. HE nuclei centers
Detect nuclei in the HE with **StarDist** (`2D_versatile_he`), 4× upscaled
(nuclei are only ~8–9 px at this resolution; the model needs them bigger).
→ ~1842 centers in the 901-px HE frame.

Notes:
- StarDist needs `tensorflow`+`csbdeep` and downloads a model, so it is run on a
  machine with internet (e.g., local Jupyter), saved to
  `he_nuclei_stardist_S2500586-1.npy`.
- A no-ML fallback exists (hematoxylin color-deconvolution + peak detection),
  lower quality but no dependencies.

### 2. Global affine (HE px → world µm)
Multi-start similarity ICP (Umeyama + trimmed RANSAC) matching HE centers to
GeoJSON centroids, trying **both Y orientations** (serial-section flip
ambiguity), then refined to a full 6-DOF affine. For S2500586-1: flip=Yes,
scale ≈ 1.10, rotation ≈ small; this gets the two point sets into the same
frame (residual a few µm, but per-cell offsets remain).

### 3. Fine center-snap (non-rigid, confidence-gated)
This is what pulls individual HE centers onto their GeoJSON centroids:

1. **Mutual nearest-neighbour** pairs between affine-aligned HE centers and
   GeoJSON centroids within `r_match=10 µm`.
2. **Confidence** per pair = closeness(residual) × local-coherence(agreement
   with neighbouring pair displacements) × support. Spurious / isolated /
   incoherent matches → ~0 weight.
3. **Displacement field** by confidence-weighted Nadaraya-Watson regression
   (Gaussian bandwidth `sig_k=12 µm`) with a ridge term so **low-support /
   unmatched regions decay to zero displacement (identity = unchanged)**.
4. Field must stay **diffeomorphic** (Jacobian determinant > 0 — no folding).

Why not simpler: per-nucleus matching without the coherence gate just chases
noise (serial-section cells don't correspond 1:1); a single global spline
over-warps and folds. The confidence gate + small bandwidth + ridge is what
gives a clean, local, non-folding correction.

### 4. Apply to the HE image
Compose `affine ∘ displacement`, invert (fixed-point), and resample the HE
pixels onto the GeoJSON world-µm grid → warped HE aligned to the GeoJSON.

## Result (S2500586-1)

Median distance between mutually-matched HE↔GeoJSON nucleus pairs:

| stage | mutual-pair median | within 3 µm |
|-------|--------------------|-------------|
| affine only | 5.78 µm | 17% |
| + fine center-snap | **2.61 µm** | **56%** |

Jacobian min 0.38 (no folding), max displacement 7.8 µm.

> The all-nearest-neighbour median stays ~9 µm because ~60% of HE nuclei have no
> confident counterpart (serial sections) and are correctly left unmoved. Judge
> quality by the **mutual-pair** median, not all-NN.

## How to run (warp only)

```
python tmp/seqfish_20260520/reg_check/warp_and_bundle.py \
    --sid S2500586-1 --no-bundle
```

Produces:
- `warpedHE_S2500586-1_yflip.png` — warped HE on the GeoJSON world-µm grid
- `warp_S2500586-1.json` — `image_bounds [x0,y0,x1,y1]`, flip, Jacobian min

(`--allow-classical` uses the no-ML detector if the StarDist `.npy` is absent.
Tunables: `--sig-k`, `--lam`, `--r-match`, `--sig-c`, `--sig-r`.)

## Visual check

`cpd_center_align.py` renders an overlay of the warped HE with the rasterized
GeoJSON nuclei (cyan outlines on dark HE nuclei), zoomed, for affine-only vs
+fine — see `42_cpd_zoom_mid.png`.

## Limitations

- HE resolution (~1.13 µm/px) caps detection and center precision; a
  higher-resolution HE would improve both.
- Serial sections: exact per-cell overlap is not physically achievable; the
  ~2.6 µm mutual median is near the practical floor for this sample.
