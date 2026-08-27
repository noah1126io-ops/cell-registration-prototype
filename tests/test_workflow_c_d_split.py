import inspect

from app import (
    main,
    show_he_geojson_preparation,
    show_mask_to_mask_workflow,
    show_point_registration_workflow,
    show_raster_deformation_workflow,
)


def test_main_exposes_separate_workflow_c_and_d_entries():
    source = inspect.getsource(main)

    assert '"Workflow C: Point Registration"' in source
    assert '"Workflow D: Raster Deformation"' in source
    assert "show_he_geojson_preparation()" in source
    assert "show_raster_deformation_workflow()" in source


def test_workflow_c_exports_registration_artifact_without_running_final_raster_block():
    source = inspect.getsource(show_he_geojson_preparation)

    assert "workflow_c_raster_outputs_enabled = False" in source
    assert "if he_image is not None and workflow_c_raster_outputs_enabled" in source
    assert "build_workflow_c_result_artifact" in source
    assert 'artifacts["workflow_c_result.zip"]' in source


def test_workflow_d_consumes_artifact_and_does_not_run_registration():
    source = inspect.getsource(show_raster_deformation_workflow)

    assert "load_workflow_c_result_artifact" in source
    assert "run_workflow_d_raster_deformation" in source
    assert "estimate_affine_with_y_flip" not in source
    assert "tissue_aware_density_flow_registration" not in source
    assert "joint_density_tissue_structure_registration" not in source


def test_workflows_a_and_b_do_not_reference_workflow_d():
    for workflow in (show_point_registration_workflow, show_mask_to_mask_workflow):
        source = inspect.getsource(workflow)
        assert "Workflow D" not in source
        assert "run_workflow_d_raster_deformation" not in source

