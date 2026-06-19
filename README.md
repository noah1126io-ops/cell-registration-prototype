# Cell Registration Prototype

Serial tissue section images and their cell segmentation masks can be loaded into this local Streamlit app to prototype density-map registration and cell correspondence estimation.

This application is a research prototype only. It is not intended for diagnosis, clinical decision-making, treatment planning, or any other medical use.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Implemented Features

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

## Not Implemented Yet

- Built-in Cellpose execution
- Non-rigid registration
- Batch processing
- Moving-side unmatched cell reporting
- Production-grade quality control reports

## Notes

- Masks are expected to be integer label images where `0` is background and positive values are cell IDs.
- Affine registration uses OpenCV when `opencv-python` is installed. If registration fails, the app falls back to the identity transform and shows a warning.
- The current cell matching step is intended as an MVP for exploratory research and should be reviewed carefully before downstream analysis.
