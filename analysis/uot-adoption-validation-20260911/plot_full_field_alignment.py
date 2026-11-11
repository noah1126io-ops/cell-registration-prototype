"""Render full-field point-alignment diagnostics from the frozen real-data outputs."""
from __future__ import annotations

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from scipy.spatial import cKDTree

from src.density_flow import warp_affine_image_with_density_flow
from src.workflow_c_result import load_workflow_c_result_artifact


OUT = Path(__file__).resolve().parent
ROOT = PROJECT_ROOT
INPUT = ROOT / "sweeps/real-grid27-20260908-v2/input"
JOINT_ARTIFACT = Path(
    "/mnt/mib-dbia08_1/users/noa.ganeko/analysis/"
    "density-vs-joint-20260909/joint_workflow_c_registration_result.zip"
)

METHOD_COLORS = {
    "Affine": "#777777",
    "UOT conservative": "#009E73",
    "Joint": "#D55E00",
}
VECTOR_MAGNIFICATION = 8.0


def nearest_fixed_distances(fixed: np.ndarray, moving: np.ndarray) -> np.ndarray:
    return cKDTree(fixed).query(moving, k=1, workers=-1)[0]


def image_extent(metadata: dict) -> tuple[float, float, float, float]:
    min_x, min_y, max_x, max_y = map(float, metadata["bounds_um"])
    return min_x, max_x, min_y, max_y


def plot_alignment_distance(
    fixed: np.ndarray,
    methods: list[tuple[str, np.ndarray]],
    metadata: dict,
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    vmax = 20.0
    norm = Normalize(0.0, vmax)
    fig, axes = plt.subplots(1, len(methods), figsize=(18, 6.4), sharex=True, sharey=True)
    for ax, (name, points) in zip(axes, methods, strict=True):
        distances = nearest_fixed_distances(fixed, points)
        metrics[name] = {
            "moving_to_fixed_nn_median_um": float(np.median(distances)),
            "moving_to_fixed_nn_p95_um": float(np.percentile(distances, 95)),
            "moving_within_5_um_fraction": float(np.mean(distances <= 5.0)),
        }
        ax.scatter(
            fixed[:, 0], fixed[:, 1], s=7, facecolors="none", edgecolors="black",
            linewidths=0.32, alpha=0.55, rasterized=True, label=f"Fixed (n={len(fixed):,})",
        )
        scatter = ax.scatter(
            points[:, 0], points[:, 1], s=8, c=np.clip(distances, 0.0, vmax),
            cmap="turbo", norm=norm, alpha=0.82, linewidths=0, rasterized=True,
            label=f"Moving (n={len(points):,})",
        )
        row = metrics[name]
        ax.set_title(
            f"{name}\nmedian NN {row['moving_to_fixed_nn_median_um']:.2f} µm; "
            f"within 5 µm {100 * row['moving_within_5_um_fraction']:.1f}%"
        )
        ax.set_aspect("equal")
        ax.grid(alpha=0.12, linewidth=0.5)
        ax.legend(loc="lower right", fontsize=8, framealpha=0.92)
        ax.set_xlabel("world x (µm)")
    axes[0].set_ylabel("world y (µm)")
    axes[0].set_xlim(metadata["bounds_um"][0], metadata["bounds_um"][2])
    axes[0].set_ylim(metadata["bounds_um"][1], metadata["bounds_um"][3])
    colorbar = fig.colorbar(scatter, ax=axes, fraction=0.018, pad=0.02)
    colorbar.set_label("Moving point → nearest fixed point distance (µm; clipped at 20)")
    fig.suptitle(
        "Full-field point alignment on the latest real-data trial\n"
        "Black rings are registration fixed points; color reports local residual proximity",
        fontsize=15,
    )
    fig.subplots_adjust(left=0.055, right=0.92, bottom=0.09, top=0.84, wspace=0.08)
    fig.savefig(OUT / "full_field_point_alignment.png", dpi=200)
    plt.close(fig)
    return metrics


def plot_movements(
    fixed: np.ndarray,
    affine: np.ndarray,
    methods: list[tuple[str, np.ndarray]],
    tissue_mask: np.ndarray,
    metadata: dict,
) -> dict[str, dict[str, float]]:
    magnitudes = [np.linalg.norm(points - affine, axis=1) for _, points in methods]
    vmax = max(float(np.percentile(values, 99)) for values in magnitudes)
    norm = Normalize(0.0, vmax)
    extent = image_extent(metadata)
    fig, axes = plt.subplots(1, len(methods), figsize=(13, 6.6), sharex=True, sharey=True)
    movement_metrics: dict[str, dict[str, float]] = {}
    for ax, (name, points), magnitude in zip(axes, methods, magnitudes, strict=True):
        displacement = points - affine
        movement_metrics[name] = {
            "displacement_median_um": float(np.median(magnitude)),
            "displacement_p95_um": float(np.percentile(magnitude, 95)),
            "displacement_max_um": float(np.max(magnitude)),
        }
        ax.imshow(
            tissue_mask, cmap="Greys", alpha=0.10, origin="upper", extent=extent,
            interpolation="nearest",
        )
        ax.scatter(
            fixed[:, 0], fixed[:, 1], s=4, color="black", alpha=0.28,
            linewidths=0, rasterized=True, label="Fixed",
        )
        quiver = ax.quiver(
            affine[:, 0], affine[:, 1], displacement[:, 0], displacement[:, 1],
            magnitude, cmap="viridis", norm=norm, angles="xy", scale_units="xy",
            scale=1.0 / VECTOR_MAGNIFICATION, width=0.00135, headwidth=3.2,
            headlength=4.1, headaxislength=3.8, alpha=0.68, rasterized=True,
        )
        ax.scatter(
            points[:, 0], points[:, 1], s=3.5, color=METHOD_COLORS[name],
            alpha=0.72, linewidths=0, rasterized=True, label="Registered endpoint (actual)",
        )
        row = movement_metrics[name]
        ax.set_title(
            f"{name}\nactual displacement median/p95: "
            f"{row['displacement_median_um']:.2f}/{row['displacement_p95_um']:.2f} µm"
        )
        ax.set_aspect("equal")
        ax.grid(alpha=0.12, linewidth=0.5)
        ax.legend(loc="lower right", fontsize=8, framealpha=0.92)
        ax.set_xlabel("world x (µm)")
    axes[0].set_ylabel("world y (µm)")
    axes[0].set_xlim(metadata["bounds_um"][0], metadata["bounds_um"][2])
    axes[0].set_ylim(metadata["bounds_um"][1], metadata["bounds_um"][3])
    colorbar = fig.colorbar(quiver, ax=axes, fraction=0.024, pad=0.025)
    colorbar.set_label("Actual displacement magnitude (µm)")
    fig.suptitle(
        f"Full-field movement from the common Affine starting positions\n"
        f"Arrow directions are exact; arrow lengths are {VECTOR_MAGNIFICATION:.0f}× magnified for visibility; "
        "colored dots are actual endpoints",
        fontsize=15,
    )
    fig.subplots_adjust(left=0.06, right=0.90, bottom=0.09, top=0.83, wspace=0.09)
    fig.savefig(OUT / "full_field_point_movements.png", dpi=200)
    plt.close(fig)
    return movement_metrics


def plot_he_overlays(
    fixed: np.ndarray,
    affine: np.ndarray,
    uot_points: np.ndarray,
    joint_points: np.ndarray,
    affine_image: np.ndarray,
    uot_image: np.ndarray,
    joint_image: np.ndarray,
    metadata: dict,
) -> None:
    extent = image_extent(metadata)
    panels = [
        ("Affine only", affine_image, affine),
        ("UOT conservative", uot_image, uot_points),
        ("Joint", joint_image, joint_points),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.4), sharex=True, sharey=True)
    for ax, (name, image, points) in zip(axes, panels, strict=True):
        ax.imshow(image, origin="upper", extent=extent, interpolation="nearest")
        ax.scatter(
            fixed[:, 0], fixed[:, 1], s=9, facecolors="none", edgecolors="#00FFFF",
            linewidths=0.45, alpha=0.75, rasterized=True, label="Fixed",
        )
        ax.scatter(
            points[:, 0], points[:, 1], s=5, color="#FF00AA", alpha=0.70,
            linewidths=0, rasterized=True, label="Moving",
        )
        ax.set_title(name)
        ax.set_aspect("equal")
        ax.legend(loc="lower right", fontsize=8, framealpha=0.92)
        ax.set_xlabel("world x (µm)")
    axes[0].set_ylabel("world y (µm)")
    axes[0].set_xlim(metadata["bounds_um"][0], metadata["bounds_um"][2])
    axes[0].set_ylim(metadata["bounds_um"][1], metadata["bounds_um"][3])
    fig.suptitle(
        "Whole-image HE and point overlays from the latest real-data trial\n"
        "Cyan rings: fixed fluorescence nuclei; magenta dots: HE nuclei positions",
        fontsize=15,
    )
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.09, top=0.84, wspace=0.07)
    fig.savefig(OUT / "full_field_he_point_overlay.png", dpi=200)
    plt.close(fig)


def main() -> None:
    metadata = json.loads((INPUT / "metadata.json").read_text())
    fixed = np.load(INPUT / "fixed_points.npy", allow_pickle=False)
    affine_input = np.load(INPUT / "moving_points.npy", allow_pickle=False)
    affine_image = np.load(INPUT / "he_image.npy", allow_pickle=False)
    tissue_mask = np.load(INPUT / "tissue_mask.npy", allow_pickle=False)

    joint_artifact = load_workflow_c_result_artifact(JOINT_ARTIFACT.read_bytes())
    joint_arrays = joint_artifact.arrays
    affine = np.asarray(joint_arrays["affine_he_points"], dtype=float)
    joint_points = np.asarray(joint_arrays["applied_registered_he_points"], dtype=float)
    joint_x = np.asarray(joint_arrays["applied_displacement_x"], dtype=float)
    joint_y = np.asarray(joint_arrays["applied_displacement_y"], dtype=float)
    bounds = tuple(float(value) for value in joint_arrays["field_bounds"])
    spacing = float(joint_arrays["field_spacing_um"])

    uot_data = np.load(OUT / "real_uot_conservative.npz", allow_pickle=False)
    uot_points = np.asarray(uot_data["points"], dtype=float)
    uot_x = np.asarray(uot_data["field_x"], dtype=float)
    uot_y = np.asarray(uot_data["field_y"], dtype=float)

    if not np.array_equal(affine, affine_input):
        raise RuntimeError("Saved Joint Affine positions differ from the frozen sweep input.")
    if not (len(affine) == len(uot_points) == len(joint_points)):
        raise RuntimeError("Moving-point arrays do not preserve a common point order.")

    alignment_metrics = plot_alignment_distance(
        fixed,
        [("Affine", affine), ("UOT conservative", uot_points), ("Joint", joint_points)],
        metadata,
    )
    movement_metrics = plot_movements(
        fixed,
        affine,
        [("UOT conservative", uot_points), ("Joint", joint_points)],
        tissue_mask,
        metadata,
    )

    uot_image, uot_inverse = warp_affine_image_with_density_flow(
        affine_image,
        metadata,
        uot_x,
        uot_y,
        field_bounds=bounds,
        field_spacing=spacing,
        inverse_iterations=30,
        inverse_convergence_tolerance_pixels=0.05,
        return_diagnostics=True,
    )
    joint_image, joint_inverse = warp_affine_image_with_density_flow(
        affine_image,
        metadata,
        joint_x,
        joint_y,
        field_bounds=bounds,
        field_spacing=spacing,
        inverse_iterations=30,
        inverse_convergence_tolerance_pixels=0.05,
        return_diagnostics=True,
    )
    plot_he_overlays(
        fixed, affine, uot_points, joint_points,
        affine_image, uot_image, joint_image, metadata,
    )

    summary = {
        "source": "frozen outputs from UOT adoption validation 2026-09-11",
        "registration_rerun": False,
        "fixed_count": int(len(fixed)),
        "moving_count": int(len(affine)),
        "affine_input_exactly_matches_joint_artifact": True,
        "vector_display_magnification": VECTOR_MAGNIFICATION,
        "alignment_metrics": alignment_metrics,
        "movement_metrics": movement_metrics,
        "raster_inverse_diagnostics": {
            "UOT conservative": uot_inverse,
            "Joint": joint_inverse,
        },
        "outputs": [
            "full_field_point_alignment.png",
            "full_field_point_movements.png",
            "full_field_he_point_overlay.png",
        ],
    }
    (OUT / "full_field_visualization_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
