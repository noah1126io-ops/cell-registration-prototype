# Cell Registration Prototype

Precomputed point, mask, GeoJSON, and tabular segmentation outputs can be loaded into this local Streamlit app to prototype registration, matching, and QC visualization.

This application is a research prototype only. It is not intended for diagnosis, clinical decision-making, treatment planning, or any other medical use.

This app is not intended to process raw pre-segmentation images. It expects numerical data created by external tools such as Cellpose, StarDist, QuPath, or related analysis pipelines. Raw images are optional QC backgrounds, not required primary inputs.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Supported Workflows

### Workflow A: Data-only point registration

This is the primary workflow. It uses fixed and moving point files without requiring raw images.

Supported point inputs:

- `.npy` arrays with shape `(n, 2)`
- `.csv` files with `x,y`, `centroid_x,centroid_y`, or `point_id,centroid_x,centroid_y`

The app previews normalized point tables, creates density maps, estimates affine registration, transforms moving points, performs matching, displays before/after scatter plots, and exports `matched_points.csv` plus `transform_summary.json`.

Raw fixed/moving images can optionally be uploaded as QC backgrounds for overlays and warped-image export.

### Workflow B: mask-to-mask registration

This is a secondary workflow for existing integer label-mask inputs. It uses:

- fixed image
- moving image
- fixed integer label mask
- moving integer label mask

The app extracts cell features from both masks, creates centroid-based density maps, estimates an affine transform from the density maps, transforms the moving image/mask/cell centroids, estimates cell correspondence candidates, and exports CSV/PNG/JSON quality-control artifacts.

### Workflow C: HE-to-GeoJSON nuclei alignment preparation

This is a planned future workflow based on the existing HE-to-fluorescence nuclei GeoJSON research pipeline. It will use:

- HE image
- HE-side nucleus center `.npy` file, detected beforehand by StarDist or another nuclei detector
- fluorescence-side nuclei segmentation GeoJSON

The goal is to warp the HE image into the fluorescence nuclei GeoJSON world-µm coordinate system, then export aligned image and QC artifacts. This mode is not implemented in the Streamlit app yet.

## Existing HE-to-GeoJSON Research Pipeline

The reference pipeline aligns one HE image to one fluorescence nuclei GeoJSON without SpatialArc bundling.

- The HE image is warped into the fluorescence nuclei GeoJSON world-µm coordinate system.
- HE-side nucleus centers are assumed to be precomputed as `.npy`, for example from StarDist.
- The fluorescence nuclei GeoJSON stores nuclei segmentation in world-µm coordinates and may require Y-flip handling.
- The alignment performs an affine step followed by a fine center-snap warp.
- Outputs include:
  - overlay image of warped HE plus GeoJSON nuclei
  - warp JSON containing image bounds, flip metadata, and quality metrics
  - warped HE image on the GeoJSON world-µm grid
- Deformation validity is checked with metrics such as minimum Jacobian value; positive Jacobian minimum is expected to avoid broken folds.
- Future Streamlit integration should keep this as a separate mode from mask-to-mask cell registration.

## Implemented Features

- Workflow selector:
  - data-only point registration
  - mask-to-mask registration
  - HE-to-GeoJSON alignment preparation
- Data-only point registration from `.npy` or `.csv`
- Point table preview and normalization
- Density-map generation from point centroids without raw images
- Affine registration and point transformation
- Position-only matching when area/eccentricity are unavailable
- Scatter and match-line QC plots on a blank coordinate canvas
- Optional raw image upload as QC background only
- Upload 4 files:
  - fixed image
  - moving image
  - fixed mask
  - moving mask
- Preview uploaded images and integer label masks
- Extract per-cell features from fixed and moving masks
  - centroid
  - area
  - perimeter
  - eccentricity
  - major/minor axis length
  - bounding box
  - mean intensity when an image is available
- Export feature tables:
  - `fixed_cell_features.csv`
  - `moving_cell_features.csv`
- Create fixed and moving cell density maps from cell centroids
- Export density maps:
  - `fixed_density_map.png`
  - `moving_density_map.png`
- Estimate affine registration from `fixed_density_map` and `moving_density_map`
- Apply the affine transform to:
  - moving image
  - moving mask
  - moving cell centroids
- Display registration overlays before and after affine alignment
- Export `transformation_summary.json`
- Estimate cell correspondence candidates after affine registration
- Export `cell_correspondence.csv`
- Visualize matched cell pairs on the fixed image
- Export `matched_cells_overlay.png`

## Planned Segmentation Sources

- Integer label masks remain the supported input for the current mask-to-mask workflow.
- Cellpose may be added later as an optional segmentation adapter.
- StarDist-derived `.npy` nucleus center coordinates are planned as a point-set input source.
- StarDist-derived `.npy` files may store centers as `xy` or `yx`; confirm the coordinate order before using them.
- GeoJSON nuclei segmentations are planned for HE-to-GeoJSON alignment.
- The internal feature/matching code should operate on normalized centroid tables rather than assuming one segmentation engine.

## Not Implemented Yet

- Optional Cellpose segmentation adapter
- StarDist `.npy` nuclei-center upload in the UI
- Non-rigid registration
- Batch processing
- Moving-side unmatched cell reporting
- HE-to-GeoJSON nuclei alignment mode
- GeoJSON nuclei input
- HE nuclei `.npy` input
- World-um coordinate transforms and Y-flip handling
- Jacobian-based warp quality checks
- Production-grade quality control reports

## Notes

- Masks are expected to be integer label images where `0` is background and positive values are cell IDs.
- Affine registration uses OpenCV when `opencv-python` is installed. If registration fails, the app falls back to the identity transform and shows a warning.
- The current cell matching step is intended as an MVP for exploratory research and should be reviewed carefully before downstream analysis.
