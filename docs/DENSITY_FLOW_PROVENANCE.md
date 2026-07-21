# Density-flow implementation provenance

## Record

- Date: 2026-07-21
- Working name: Tissue-aware density flow [Experimental]
- Scope: Workflow C only
- Author/agent: OpenAI Codex, working with the repository owner
- Implementation module: `src/density_flow.py`
- Integration module: `app.py`
- Tests: `tests/test_density_flow.py` and existing Workflow C regression tests

## Independent implementation statement

This implementation was written independently for this repository. No STalign source code was copied, vendored, installed, imported, translated line by line, or used as a runtime dependency. No STalign function names, internal APIs, comments, or source-code organization were reproduced.

The repository license was not changed as part of this work.

## Conceptual scientific references

1. Clifton K, Anant M, Aihara G, et al. "STalign: Alignment of spatial transcriptomics data using diffeomorphic metric mapping." Nature Communications 14, 8123 (2023). DOI: https://doi.org/10.1038/s41467-023-43915-7
2. Beg MF, Miller MI, Trouve A, Younes L. "Computing Large Deformation Metric Mappings via Geodesic Flows of Diffeomorphisms." International Journal of Computer Vision 61, 139-157 (2005).

Only publicly described mathematical concepts were used: point-density rasterization, Gaussian multiscale representations, regularized deformation fields, small-update composition, and Jacobian-based deformation QC.

## Important implementation differences

- The method operates on affine-transformed HE nuclei and valid GeoJSON nuclei in a shared 2D world-coordinate grid.
- Densities use bilinear point deposition and independent mass normalization.
- Tissue support is estimated from fixed valid nuclei and a boundary-distance confidence map.
- Initialization chooses among zero shift, robust center shift, and density cross-correlation.
- Residual updates are derived from density differences and density gradients, then explicitly smoothed and magnitude-limited.
- Updates are composed with backtracking Jacobian checks rather than solved as a geodesic shooting problem.
- Safety is decided using this application's existing Workflow C point-set and deformation QC contract.
- HE raster warping uses a repository-specific iterative inverse map over the affine world image; rejected fields retain attempted QC output while final output falls back to affine-only.

## Claims intentionally not made

- This method is not described as STalign or as an STalign reimplementation.
- No mathematical or numerical equivalence to STalign or LDDMM software is claimed.
- No improvement in biological registration accuracy is claimed without experimental validation.
