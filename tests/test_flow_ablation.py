import numpy as np
import pytest

from src.density_flow import _field_objective, tissue_aware_density_flow_registration
from src.flow_ablation import (
    FlowAblationConfig,
    ablation_preset,
    normalize_ablation_config,
)
from src.flow_ablation_benchmark import (
    ablation_benchmark_artifacts,
    moving_dropout_indices,
    run_ablation_benchmark,
)


def _points():
    x, y = np.meshgrid(np.arange(10.0, 71.0, 10.0), np.arange(10.0, 71.0, 10.0))
    return np.column_stack([x.ravel(), y.ravel()])


def _density_run(config=None, *, retain=False):
    fixed = _points()
    moving = fixed + np.array([2.0, -1.0])
    return tissue_aware_density_flow_registration(
        fixed,
        moving,
        density_pixel_size=2.0,
        density_blur_scales=(2.0,),
        optimization_levels=1,
        iterations_per_level=2,
        learning_rate=0.08,
        max_displacement=10.0,
        displacement_p95_limit=10.0,
        detect_axis_reversal=False,
        minimum_absolute_median_improvement=0.001,
        minimum_relative_median_improvement=0.0001,
        ablation_config=config,
        retain_research_diagnostic_fields=retain,
    )


def test_ablation_disabled_reproduces_density_flow_default():
    legacy = _density_run(None)
    explicit = _density_run(FlowAblationConfig())
    np.testing.assert_allclose(legacy.attempted_displacement_x, explicit.attempted_displacement_x)
    np.testing.assert_allclose(legacy.attempted_displacement_y, explicit.attempted_displacement_y)
    assert legacy.metrics["density_flow"]["best_objective"] == explicit.metrics["density_flow"]["best_objective"]


def test_disabled_density_has_zero_objective_and_no_density_update_field():
    result = _density_run(
        FlowAblationConfig(use_density_term=False),
        retain=True,
    )
    density_rows = [row for row in result.metrics["objective_terms"] if row["term"] == "density"]
    assert density_rows
    assert all(row["enabled"] is False and row["weighted_value"] == 0.0 for row in density_rows)
    assert "density_update_x" not in result.metrics["selected_update_fields"]


@pytest.mark.parametrize(
    "term,weight_argument,config_argument",
    [
        ("support", "tissue_support_channel_weight", "use_support_term"),
        ("structure", "structure_channel_weight", "use_structure_term"),
    ],
)
def test_disabled_image_channel_removes_objective_and_direct_update(
    term, weight_argument, config_argument
):
    fixed = _points()
    moving = fixed + np.array([1.0, 1.0])
    yy, xx = np.indices((80, 80), dtype=float)
    tissue_mask = (xx - 40.0) ** 2 + (yy - 40.0) ** 2 < 30.0**2
    metadata = {
        "bounds_um": [0.0, 0.0, 80.0, 80.0],
        "output_pixel_size_um": 1.0,
        "output_origin": "upper-left",
        "width": 80,
        "height": 80,
        "row0_world_y": 0.5,
        "col0_world_x": 0.5,
    }
    kwargs = {
        weight_argument: 1.0,
        "moving_tissue_mask": tissue_mask,
        "moving_tissue_metadata": metadata,
    }
    if term == "structure":
        kwargs.update(
            moving_structure_image=(xx + 2.0 * yy),
            fixed_structure_feature_grid=np.tile(np.linspace(0, 1, 41), (41, 1)),
        )
    config_values = {config_argument: False, "use_density_term": False}
    result = tissue_aware_density_flow_registration(
        fixed,
        moving,
        bounds=(0.0, 0.0, 80.0, 80.0),
        density_pixel_size=2.0,
        density_blur_scales=(2.0,),
        optimization_levels=1,
        iterations_per_level=1,
        learning_rate=0.08,
        max_displacement=10.0,
        displacement_p95_limit=10.0,
        detect_axis_reversal=False,
        ablation_config=FlowAblationConfig(**config_values),
        retain_research_diagnostic_fields=True,
        **kwargs,
    )
    rows = [row for row in result.metrics["objective_terms"] if row["term"] == term]
    assert rows and all(row["weighted_value"] == 0.0 and not row["enabled"] for row in rows)
    assert f"{term}_update_x" not in result.metrics["selected_update_fields"]


def test_objective_raw_weighted_arithmetic_and_groups():
    shape = (9, 9)
    fixed_density = np.zeros(shape)
    fixed_density[4, 4] = 1.0
    moving_density = np.zeros(shape)
    moving_density[4, 5] = 1.0
    field_x = np.full(shape, 0.2)
    field_y = np.zeros(shape)
    tissue = np.ones(shape)
    jacobian = np.ones(shape)
    result = _field_objective(
        fixed_density,
        moving_density,
        field_x,
        field_y,
        tissue,
        jacobian,
        smoothness_weight=0.1,
        magnitude_weight=0.2,
        jacobian_weight=1.0,
        boundary_weight=0.3,
        inverse_consistency_weight=0.0,
        jacobian_min_threshold=0.1,
        jacobian_max_threshold=3.0,
        pixel_size=1.0,
        density_weight=2.0,
    )
    decomposition = result["objective_decomposition"]
    for term in decomposition["terms"].values():
        assert term["weighted_value"] == pytest.approx(
            term["raw_value"] * term["weight"] if term["enabled"] else 0.0
        )
    assert decomposition["weighted_total"] == pytest.approx(
        decomposition["weighted_data_total"] + decomposition["weighted_regularization_total"]
    )
    assert result["total"] == decomposition["weighted_total"]


def test_stage_configs_can_differ_and_hard_validity_is_not_ablatable():
    stage_a = ablation_preset("Stage A support only", base=FlowAblationConfig())
    stage_b = ablation_preset("Stage B structure only", base=FlowAblationConfig())
    assert stage_a.use_support_term and not stage_a.use_density_term
    assert stage_b.use_structure_term and not stage_b.use_density_term
    assert "finite" not in FlowAblationConfig.__dataclass_fields__
    assert "foldover" not in FlowAblationConfig.__dataclass_fields__
    with pytest.raises(ValueError, match="Unknown flow ablation flags"):
        normalize_ablation_config({"disable_foldover_check": True})


def test_dropout_selection_supports_random_and_spatial_bias():
    points = np.column_stack([np.arange(20.0), np.zeros(20)])
    random_keep = moving_dropout_indices(points, 0.5, seed=7)
    biased_keep = moving_dropout_indices(points, 0.5, seed=7, spatially_biased=True)
    assert len(random_keep) == len(biased_keep) == 10
    np.testing.assert_array_equal(biased_keep, np.arange(10))
    assert not np.array_equal(random_keep, biased_keep)


def test_full_minus_one_benchmark_and_npz_style_artifacts_smoke():
    results, summary, metadata = run_ablation_benchmark(
        ablations=("Full", "-Density"),
        dropout_fractions=(0.0, 0.25),
        include_spatially_biased_dropout=True,
        seed=3,
        size=64,
        iterations_per_level=1,
    )
    assert {"Full", "-Density"} == set(results["ablation"])
    assert {"random", "spatially_biased"} == set(results["dropout_mode"])
    assert np.isfinite(results["displacement_epe_p95_um"]).all()
    assert not summary.empty
    artifacts = ablation_benchmark_artifacts(results, summary, metadata)
    assert set(artifacts) == {
        "ablation_results.csv", "ablation_summary.csv", "ablation_metadata.json"
    }
    assert b"dropout_fraction" in artifacts["ablation_results.csv"]
