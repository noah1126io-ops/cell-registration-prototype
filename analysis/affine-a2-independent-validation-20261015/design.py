"""Pre-registered experiment plan (fixed 2026-10-15 before any A2/A0/A3 run on the independent generator).

Every case (main and control) has its own latent tissue, keyed by (family, condition-or-control name, replicate), so no
two cases share a base pattern. Cases of the same family/condition share the generator, not the realisation.
"""
from __future__ import annotations

FAMILIES = ("G1", "G2", "G3", "G4")
REPS = (0, 1)

# Base: offset 60 um, rotation 2 deg, reflected HE frame, 1.10 um/px, q_shared 0.65, detection 0.93/0.90,
# weak non-affine deformation (2 um RMS), Real-like density 0.0020 /um^2, fixed FOV 1000 um, moving 900 px (990 um).
CONDITIONS = {
    "C01_base": {},
    "C02_off0_proper_nodeform": dict(offset_um=0.0, rot_deg=0.0, reflect=False, deform_rms_um=0.0),
    "C03_off100_rot3": dict(offset_um=100.0, rot_deg=3.0),
    "C04_off150_rot6": dict(offset_um=150.0, rot_deg=6.0),
    "C05_oor_off320": dict(offset_um=320.0),
    "C06_composite_scale": dict(offset_um=120.0, rot_deg=5.0, pixel_um=1.30, mov_px=(760, 520), density=0.0030),
    "C07_shared_low": dict(q_shared=0.30),
    "C08_detect_loss": dict(q_shared=0.85, det_fixed=0.50, det_moving=0.50),
    "C09_overlap_mid_contained": dict(offset_um=40.0, mov_px=(1230, 1230)),
    "C10_overlap_low_smallfixed": dict(offset_um=60.0, fixed_fov_um=(520.0, 520.0)),
    "C11_asym_crop_partial": dict(offset_um=160.0, mov_px=(900, 600)),
    "C12_deform_moderate_loss": dict(deform_rms_um=6.0, loss_patches=3),
    "C13_deform_strong": dict(deform_rms_um=12.0),
    "C14_density_high_hetero": dict(density=0.0042, hetero=1.8),
}

# factor tags used for stratified reporting (one primary tag per condition)
FACTOR = {
    "C01_base": "base", "C02_off0_proper_nodeform": "transform", "C03_off100_rot3": "transform",
    "C04_off150_rot6": "transform", "C05_oor_off320": "out_of_range", "C06_composite_scale": "transform",
    "C07_shared_low": "correspondence", "C08_detect_loss": "correspondence", "C09_overlap_mid_contained": "overlap",
    "C10_overlap_low_smallfixed": "overlap", "C11_asym_crop_partial": "overlap", "C12_deform_moderate_loss": "deformation",
    "C13_deform_strong": "deformation", "C14_density_high_hetero": "density",
}


def main_cases():
    return [(f, c, r) for f in FAMILIES for c in CONDITIONS for r in REPS]


# Negative / adversarial controls (independent generator). expect: what a PASS means.
#   no_shared  -> any PASS is a false accept (no cell-level correspondence exists)
#   truth      -> PASS is a false accept only if the affine is wrong (truth exists)
#   wrong_chir -> search restricted to the wrong chirality; any PASS is a false accept
CONTROLS = (
    [("NC1_unrelated", f, 0, {}, "unrelated", None, "no_shared") for f in FAMILIES]
    + [("NC2_density_matched", f, 0, {}, "density_matched", None, "no_shared") for f in FAMILIES]
    + [("NC3_repeated_near_exact", "G4", r, dict(copy_jitter_um=1.2, copy_keep=0.92), "normal", None, "truth") for r in range(4)]
    + [("NC4_tiny_overlap_off900", f, 0, dict(offset_um=900.0), "normal", None, "truth") for f in FAMILIES]
    + [("NC5_wrong_chirality", f, 0, {}, "normal", "proper", "wrong_chir") for f in FAMILIES]
    + [("NC6_oor_off400", f, 0, dict(offset_um=400.0), "normal", None, "truth") for f in FAMILIES]
)

PRIMARY = "A2"
BASELINE = "A0"
AUX = "A3"


def methods_for_main(rep):
    """A2 + A0 everywhere; A3 (auxiliary) on replicate 0 of every condition."""
    return ("A0", "A2", "A3") if rep == 0 else ("A0", "A2")


def methods_for_control(name):
    return ("A0", "A2", "A3") if name.startswith("NC3") else ("A0", "A2")


# Downstream (frozen Graph/UOT/B2): every main case with A0, A2 and the TRUE global affine (oracle reference, diagnostic
# only), plus A2 on the no-shared-nucleus controls.
def downstream_jobs():
    jobs = [("main", i, k) for i in range(len(main_cases())) for k in ("A0", "A2", "TRUE")]
    jobs += [("control", j, "A2") for j, c in enumerate(CONTROLS) if c[0] in ("NC1_unrelated", "NC2_density_matched")]
    return jobs


# Success definitions (fixed in advance)
AFFINE_SUCCESS_RMS_UM = 5.0          # RMS over the common physical region (moving-image grid whose true image is in fixed FOV)
GRAPH_SUCCESS = dict(top1_min=0.80, b2_precision_min=0.90)   # on the production-equivalent subset, common cells
