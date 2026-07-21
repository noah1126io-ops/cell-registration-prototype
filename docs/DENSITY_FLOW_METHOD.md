# Tissue-aware density-flow registration

## Status and scope

This is an experimental, independently written Workflow C method for residual alignment after affine registration. It transforms affine-registered HE nuclei points toward valid fixed GeoJSON nuclei. Fixed GeoJSON points never move.

The method transforms HE points and can generate an experimental HE raster warp. Raster output is safety-gated independently: attempted output always remains available for QC, while final output uses the density-flow field only when the point/field safety checks pass.

The method is not STalign, does not import STalign, and does not claim equivalent behavior or biological accuracy.

## Representation

Fixed points `F` and moving points `M` are rasterized independently on one world-coordinate xy grid with pixel size `h`. Bilinear point deposition reduces quantization artifacts. For Gaussian scale `sigma_k`, each density is normalized by its own total mass:

```text
rho_F,k = G_sigma_k * raster(F) / sum(G_sigma_k * raster(F))
rho_M,k = G_sigma_k * raster(phi(M)) / sum(G_sigma_k * raster(phi(M)))
```

This normalization prevents unequal detected-nucleus counts from dominating the density term. Default scales are 8, 4, and 2 grid pixels, optimized from coarse to fine.

## Tissue weighting

A support density is generated from fixed valid GeoJSON nuclei at a broad scale. Thresholding defines a tissue-support mask. A distance transform gives lower confidence near its boundary. The density force is multiplied by this support and boundary-confidence map, suppressing motion toward unsupported background.

This point-derived support is a conservative fallback. Workflow C still reports image-derived tissue classification separately when an HE image is available.

## Objective

For displacement field `d`, Jacobian determinant `J`, tissue weight `w`, and current scale `k`, the diagnostic objective is:

```text
E(d) = mean(w * (rho_F,k - rho_M,k)^2)
     + lambda_s * mean(||grad d||^2)
     + lambda_m * mean(||d||^2)
     + lambda_J * mean(max(J_min - J, 0)^2)
     + lambda_b * mean((1 - w) * ||d||^2)
     + lambda_i * E_inverse(d)
```

`E_inverse` is optional and disabled by default. The implemented approximation penalizes disagreement between the field and the field sampled through its own forward map. It is an engineering regularizer, not a full inverse deformation solve.

The optimization update is independently designed from the normalized density residual and the summed fixed/moving density gradients. Raw point count is not an optimization target, and exact one-to-one cell matches are not required.

## Small-update composition

The algorithm first compares three global residual translations: zero, robust median-center difference, and density cross-correlation. It retains the candidate with the best symmetric point-set median as a stable initialization.

At every scale, a smooth residual update is estimated, clipped to a small fraction of a density pixel, and composed with the current field:

```text
phi_new(x) = phi(x) + u(phi(x))
```

Each proposed composition is reduced by backtracking when its Jacobian becomes non-finite, approaches fold-over, expands excessively, or exceeds the provisional displacement envelope. This is diffeomorphic-style numerical integration; it is not a proof that every accepted map is mathematically diffeomorphic.

## Experimental HE raster warp

The point field is a forward map `phi(x) = x + d(x)`. It is not used directly as an output-to-input raster sampling map. For every output world-grid pixel `y`, the implementation approximately solves the inverse relation by fixed-point iteration:

```text
x_(n+1) = y - d(x_n)
```

The converged source world coordinate `x` is converted explicitly to affine-image `(row, column)` coordinates using the selected output origin. `scipy.ndimage.map_coordinates` then samples the affine HE image with linear interpolation. This inverse mapping visits every output pixel and avoids holes caused by forward splatting.

Three image states are retained: affine-only, attempted density-flow, and final applied. If safety rejects the field, attempted remains visible but final is an affine-only copy.

## Shared safety decision

The attempted field and points are always retained for QC. Application is rejected and final points fall back to affine-only when any of these checks fail:

- non-finite points, field, or Jacobian;
- Jacobian below the configured minimum or fold-over fraction above zero;
- Jacobian above the configured maximum;
- maximum or p95 displacement above configured limits;
- symmetric valid-region median distance worsens;
- the fraction of HE points outside fixed tissue support increases materially;
- a strong x/y reversal signal is detected before optimization.

Reported QC includes bidirectional median and mean distance, fractions within 3/5/10 um in both directions, mutual-nearest-neighbor fraction, Jacobian min/max/median, fold-over fraction, maximum and p95 displacement, outside-tissue fractions, finite-value status, and optimization history.

## Limitations

- The tissue support used inside this method is inferred from valid fixed points rather than a histology segmentation.
- Density similarity can have ambiguous local optima in repeated or sparse structures.
- The x/y reversal check is heuristic and can be disabled after coordinate order is independently validated.
- Real-data biological accuracy has not been established.
- HE raster warping is experimental and requires visual QC; full-resolution tiled export is not implemented.

## Conceptual references

- Clifton K, Anant M, Aihara G, et al. "STalign: Alignment of spatial transcriptomics data using diffeomorphic metric mapping." Nature Communications 14, 8123 (2023). https://doi.org/10.1038/s41467-023-43915-7
- Beg MF, Miller MI, Trouve A, Younes L. "Computing Large Deformation Metric Mappings via Geodesic Flows of Diffeomorphisms." International Journal of Computer Vision 61, 139-157 (2005).

These publications are conceptual scientific background only and were not used as source-code templates.
