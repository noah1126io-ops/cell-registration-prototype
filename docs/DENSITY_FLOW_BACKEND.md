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
