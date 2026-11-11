"""Hand an ACCEPTED research registration to Workflow D in its existing input format.

Workflow D consumes the production Workflow C result artifact (workflow-c-registration-result-v1): HE nuclei in HE
level-0 pixels (moving) -> fluorescence world um (fixed), ``p_world = A @ flip(p_px) + t``, plus a fine displacement
field.  A research artifact stores ``fixed_world = M @ moving_world + t`` in frame-local um.  For HE
(world = px * pixel_um, frame-local origin) and no flip:  A = M_he->if * pixel_um,  t = t_he->if.

Only accepted, fully automatic results are converted.  Unaccepted candidates, assisted runs and manual transforms are
refused (they stay PREVIEW ONLY).  The fine stage is 'off' (zero field, fine_applied=False), so Workflow D renders the
affine warp; Workflow D itself is unchanged.
"""
from __future__ import annotations

import numpy as np

from ..workflow_c_result import build_workflow_c_result_artifact
from .frames import Affine2D
from .workflow import SideInput, fingerprint

ACCEPTED_STATUSES = ("ACCEPTED", "POINT_ONLY_FALLBACK")


class NotAcceptedError(ValueError):
    pass


def check_handoff(art: dict) -> list[str]:
    """Reasons that forbid the hand-off (empty list = allowed)."""
    d = art.get("decision") or {}
    why = []
    if not d.get("accepted"):
        why.append(f"not accepted (status {d.get('status')})")
    if d.get("status") not in ACCEPTED_STATUSES:
        why.append(f"status {d.get('status')} is not an accepted status")
    if art.get("mode") != "automatic":
        why.append("assisted run (user annotations were used)")
    if not art.get("transform"):
        why.append("no transform")
    keys = {art.get("input", {}).get("fixed", {}).get("key"), art.get("input", {}).get("moving", {}).get("key")}
    if keys != {"HE", "IF"}:
        why.append("Workflow D needs an H&E / fluorescence pair")
    return why


def he_to_if(art: dict) -> Affine2D:
    T = Affine2D.from_dict(art["transform"]["forward"])
    return T if art["input"]["fixed"]["key"] == "IF" else T.inverse()


def to_workflow_c_artifact(art: dict, he: SideInput, iff: SideInput, *, field_spacing_um=2.0, output_pixel_size_um=1.0) -> bytes:
    why = check_handoff(art)
    if why:
        raise NotAcceptedError("; ".join(why))
    for side in (he, iff):
        exp = art["input"]["fixed" if art["input"]["fixed"]["key"] == side.key else "moving"]["points_sha256"]
        if fingerprint(side)["points_sha256"] != exp:
            raise ValueError(f"{side.key}: nuclei differ from the ones the registration used (fingerprint mismatch)")
    if he.pixel_um is None or he.shape_px is None:
        raise ValueError("H&E pixel spacing / image size UNKNOWN")
    T = he_to_if(art)
    A = np.asarray(T.M, float) * he.pixel_um
    t = np.asarray(T.t, float)
    he_px = np.asarray(he.points, float) / he.pixel_um
    affine_pts = he_px @ A.T + t
    fixed = np.asarray(iff.points, float)
    allp = np.vstack([fixed, affine_pts])
    lo, hi = np.floor(allp.min(0)) - 10.0, np.ceil(allp.max(0)) + 10.0
    ny = int(np.ceil((hi[1] - lo[1]) / field_spacing_um)) + 1
    nx = int(np.ceil((hi[0] - lo[0]) / field_spacing_um)) + 1
    gx, gy = np.meshgrid(lo[0] + np.arange(nx) * field_spacing_um, lo[1] + np.arange(ny) * field_spacing_um)
    zero = np.zeros_like(gx)
    d = art["decision"]
    sel = next((c for c in art["candidates"] if c["id"] == d["selected_candidate"]), {})
    v = sel.get("validation", {})
    metrics = {"source": "research Workflow C (registration_core)", "decision_status": d["status"], "accepted": True,
               "selected_candidate": d["selected_candidate"], "fixed_frame": v.get("fixed_frame"),
               "if_frame_final": (v.get("if_frame") or {}).get("final"), "rejection_reason": "fine stage not run (research hand-off: affine only)"}
    params = {"research_run_id": art["run_id"], "localization": art["config"]["localization"], "acceptance_rule": d["rule"]}
    prov = {"research_artifact_schema": art["schema"], "research_run_id": art["run_id"], "software": art["software"], "input": art["input"],
            "conversion": "A = M_he->if * he_pixel_um; t = t_he->if; HE px = HE world / pixel_um; no flip; fine field zero"}
    h, w = he.shape_px
    return build_workflow_c_result_artifact(
        fixed_geojson_points=fixed, original_moving_he_points=he_px, affine_he_points=affine_pts,
        attempted_registered_he_points=affine_pts, applied_registered_he_points=affine_pts, affine_matrix=A, affine_translation=t,
        affine_flip_x=False, affine_flip_y=False, affine_image_width=float(w), affine_image_height=float(h),
        attempted_displacement_x=zero, attempted_displacement_y=zero, applied_displacement_x=zero, applied_displacement_y=zero,
        grid_x=gx, grid_y=gy, field_bounds=(float(lo[0]), float(lo[1]), float(lo[0] + (nx - 1) * field_spacing_um), float(lo[1] + (ny - 1) * field_spacing_um)),
        field_spacing=field_spacing_um, fine_method="off", fine_applied=False, output_pixel_size_um=output_pixel_size_um,
        output_origin="upper-left", inverse_iterations=8, inverse_tolerance_pixels=0.01, metrics=metrics, parameters=params, provenance=prov)
