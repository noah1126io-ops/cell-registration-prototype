# Density Flow array residency

Phase 4 keeps one Density Flow implementation and selects its array operations
through `ArrayBackend`.  Both backends use float64.

- GPU resident: point impulses used by the optimizer, density/support/structure
  grids, displacement and update fields, Gaussian filters, gradients,
  Laplacians, field composition, objective grids, Jacobians, finite checks, and
  every backtracking candidate.
- CPU resident: input validation, affine/global initialization, cKDTree point
  metrics, checkpoint ranking/policy, scalar history, pandas tables, plotting,
  validation, and artifact export.
- Explicit synchronization: objective and hard-safety scalars needed by Python
  acceptance logic; sampled N x 2 transformed points at accepted checkpoints;
  selected/final displacement fields once after optimization.  Full grids are
  not copied to CPU for ordinary checkpoint point metrics.

There are no GPU-model branches. `device="cpu"` remains the default and uses
NumPy/SciPy; `device="cuda"` uses CuPy/cupyx; `device="auto"` resolves once at
the start of Density Flow.

## Joint Density + Tissue-Structure Flow

Joint Flow resolves `device` once and passes the same backend and float64 dtype
to both Density Flow optimizer stages. Stage A and Stage B therefore reuse the
GPU-resident optimizer above; there is no separate CUDA optimizer.

The Stage A/B boundary is deliberately CPU-resident. The selected Stage A
field is transferred once, then HE and mask warping, scikit-image `rgb2hed`,
distance transforms, nuclear/structure feature construction, and world-grid
resampling run on CPU. The completed Stage B support and structure grids are
transferred once to the selected backend. The boundary field is rounded to
nine decimal places before discrete raster sampling so sub-tolerance backend
noise cannot flip integer image values. Stage A's reported selected field is
not replaced by this boundary-only representation.

Stage boundaries synchronize CUDA before runtime measurements are recorded.
Joint results store overall and per-stage backend provenance, plus the explicit
intermediate transfer policy. Full intermediate arrays are retained only when
`retain_research_diagnostic_fields=True`.
