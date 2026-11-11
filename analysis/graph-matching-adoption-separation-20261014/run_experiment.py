"""Run the analysis-only Graph matching/adoption separation experiment."""
from __future__ import annotations

import subprocess
import sys
import time
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import experiment_lib as lib

_viz_spec = importlib.util.spec_from_file_location(
    "graph_adoption_visualization", HERE / "visualization.py"
)
viz = importlib.util.module_from_spec(_viz_spec)
sys.modules[_viz_spec.name] = viz
_viz_spec.loader.exec_module(viz)


def _case_metrics(frames: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (_, method), table in frames.groupby(["scenario_id", "method"], sort=False):
        rows.append(lib.method_metrics(table, record_type="case"))
    return pd.DataFrame(rows)


def _aggregate_test(frames: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = frames.loc[frames["split"] == "Test"].copy()
    rows, curves = [], []
    for method, table in test.groupby("method", sort=False):
        metric = lib.method_metrics(table, record_type="held_out_test")
        curve = lib.precision_coverage_curve(table)
        ece, brier = lib.expected_calibration_error(table)
        metric.update(
            {
                "coverage_at_90_precision_empirical": lib.empirical_coverage_at_precision(curve, 0.90),
                "coverage_at_95_precision_empirical": lib.empirical_coverage_at_precision(curve, 0.95),
                "coverage_at_98_precision_empirical": lib.empirical_coverage_at_precision(curve, 0.98),
                "precision_at_10_coverage": lib.precision_at_coverage(curve, 0.10),
                "precision_at_25_coverage": lib.precision_at_coverage(curve, 0.25),
                "precision_at_50_coverage": lib.precision_at_coverage(curve, 0.50),
                "calibration_ece": ece,
                "calibration_brier": brier,
            }
        )
        rows.append(metric)
        curves.append(curve)
    return pd.DataFrame(rows), pd.concat(curves, ignore_index=True)


def _calibration_table(frames: pd.DataFrame) -> pd.DataFrame:
    validation = frames.loc[frames["split"] == "Validation"]
    test = frames.loc[frames["split"] == "Test"]
    rows = []
    for method in frames["method"].drop_duplicates():
        rows.extend(lib.calibration_summary(validation, test, method))
    return pd.DataFrame(rows)


def _baseline_reproduction() -> pd.DataFrame:
    source = pd.read_csv(lib.COST_SOURCE / "synthetic_correspondence_metrics.csv")
    source = source.loc[(source["record_type"] == "aggregate") & (source["difficulty"] == "All")]
    mapping = {
        "A3 Position+Graph": lib.METHOD_B0,
        "A6 Full": lib.METHOD_B3,
    }
    table = source.loc[source["condition"].isin(mapping)].copy()
    table["method"] = table["condition"].map(mapping)
    expected = {
        lib.METHOD_B0: (0.8523434259643301, 0.8071157305616630),
        lib.METHOD_B3: (0.8361675653255910, 0.7990053874844593),
    }
    for row in table.itertuples():
        top1, f1 = expected[row.method]
        if abs(row.uot_top1_accuracy - top1) > 1e-8 or abs(row.matched_f1 - f1) > 1e-8:
            raise AssertionError(f"Existing candidate-fixed baseline changed for {row.method}")
    return table[
        [
            "method",
            "n_rows",
            "common_n",
            "true_match_candidate_recall",
            "uot_top1_accuracy",
            "matched_precision",
            "matched_recall",
            "matched_f1",
            "unmatched_f1",
        ]
    ]


def _domain_summary(
    frames: pd.DataFrame,
    geometry: pd.DataFrame,
    real_frames: pd.DataFrame,
    real_fixed: np.ndarray,
    real_moving: np.ndarray,
) -> pd.DataFrame:
    test_b0 = frames.loc[(frames["split"] == "Test") & (frames["method"] == lib.METHOD_B0)].copy()
    test_b2 = frames.loc[(frames["split"] == "Test") & (frames["method"] == lib.METHOD_B2), [
        "scenario_id", "moving_index", "calibrated_confidence", "status"
    ]].rename(columns={"calibrated_confidence": "b2_confidence", "status": "b2_status"})
    test_b0 = test_b0.merge(test_b2, on=["scenario_id", "moving_index"], validate="one_to_one")
    test_b0 = test_b0.merge(geometry, on=["scenario_id", "moving_index"], validate="one_to_one")
    real_b0 = real_frames.loc[real_frames["method"] == lib.METHOD_B0].copy()
    real_b2 = real_frames.loc[real_frames["method"] == lib.METHOD_B2, [
        "moving_index", "calibrated_confidence", "status"
    ]].rename(columns={"calibrated_confidence": "b2_confidence", "status": "b2_status"})
    real_b0 = real_b0.merge(real_b2, on="moving_index", validate="one_to_one")
    real_density, real_knn = lib.point_density_and_knn(real_moving)
    real_b0["nuclear_density_per_um2"] = real_density
    real_b0["knn_distance_um"] = real_knn

    def long_rows(group: str, table: pd.DataFrame) -> list[dict]:
        variables = {
            "nuclear_density_per_um2": table["nuclear_density_per_um2"],
            "knn_distance_um": table["knn_distance_um"],
            "candidate_distance_um": table["spatial_distance_um"],
            "graph_support": table["top1_graph_support"],
            "minimum_contextual_cost": table["absolute_min_cost"],
            "row_mass_ratio": table["row_mass_ratio"],
            "candidate_margin": table["absolute_cost_margin"],
            "confidence": table["b2_confidence"],
            "predicted_unmatched_fraction": np.asarray([np.mean(table["b2_status"] == "unmatched")]),
        }
        rows = []
        for variable, raw in variables.items():
            values = np.asarray(raw, dtype=float)
            values = values[np.isfinite(values)]
            rows.append(
                {
                    "dataset_group": group,
                    "variable": variable,
                    "count": len(values),
                    "p05": float(np.percentile(values, 5)),
                    "median": float(np.median(values)),
                    "p95": float(np.percentile(values, 95)),
                    "mean": float(np.mean(values)),
                }
            )
        return rows

    return pd.DataFrame(long_rows("Real-like Test", test_b0) + long_rows("Real", real_b0))


def _real_metrics(frames: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, table in frames.groupby("method", sort=False):
        status = table["status"].to_numpy(str)
        matched = status == "matched"
        destinations = table.loc[matched, "top1_fixed_index"].to_numpy(int)
        duplicate = int(len(destinations) - len(np.unique(destinations))) if len(destinations) else 0
        rows.append(
            {
                "method": method,
                "moving_n": len(table),
                "matched_n": int(np.count_nonzero(matched)),
                "ambiguous_n": int(np.count_nonzero(status == "ambiguous")),
                "unmatched_n": int(np.count_nonzero(status == "unmatched")),
                "coverage": float(np.mean(matched)),
                "candidate_concentration_median": float(np.median(table["normalized_top1_mass"])),
                "row_mass_ratio_median": float(np.median(table["row_mass_ratio"])),
                "relative_matching_score_median": float(np.median(table["relative_matching_score"])),
                "graph_support_median": float(np.median(table["top1_graph_support"])),
                "synthetic_derived_confidence_median": float(np.median(table["calibrated_confidence"])),
                "reciprocal_fraction": float(np.mean(table["reciprocal"])),
                "duplicate_destination_count": duplicate,
                "duplicate_destination_rate": float(duplicate / max(len(destinations), 1)),
                "accuracy_available": False,
            }
        )
    return pd.DataFrame(rows)


def _negative_metrics(frames: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (_, method), table in frames.groupby(["scenario_id", "method"], sort=False):
        metric = lib.method_metrics(table, record_type="negative_control")
        metric["control_kind"] = table["strength"].iloc[0]
        metric["false_matched_count"] = metric["matched_n"]
        metric["rejection_fraction"] = 1.0 - metric["coverage"]
        rows.append(metric)
    return pd.DataFrame(rows)


def _decision(
    comparison: pd.DataFrame,
    graph_summary: pd.DataFrame,
) -> tuple[str, dict]:
    values = comparison.set_index("method")
    b0, b2 = values.loc[lib.METHOD_B0], values.loc[lib.METHOD_B2]
    position = values.loc[lib.METHOD_POSITION]
    raw = values.loc[lib.METHOD_GRAPH_RAW]
    graph_auc = float(graph_summary["graph_candidate_auc"].mean())
    graph_gain = float(b0["top1_accuracy"] - position["top1_accuracy"])
    separated_gain = float(
        b2["coverage_at_95_precision_empirical"] - b0["coverage_at_95_precision_empirical"]
    )
    facts = {
        "graph_auc": graph_auc,
        "graph_top1_gain_vs_position": graph_gain,
        "separated_coverage_gain_at_95_precision": separated_gain,
        "uot_top1_gain_vs_raw_graph": float(b0["top1_accuracy"] - raw["top1_accuracy"]),
        "relative_uot_top1_gain_vs_raw_graph": float(
            values.loc[lib.METHOD_B1, "top1_accuracy"] - raw["top1_accuracy"]
        ),
        "uot_duplicate_rate_change_vs_raw_graph": float(
            values.loc[lib.METHOD_B1, "top1_duplicate_destination_rate_all"]
            - raw["top1_duplicate_destination_rate_all"]
        ),
    }
    if graph_auc < 0.65 or graph_gain < 0.02:
        return "D. GRAPH DOMAIN GAP DOMINATES", facts
    if separated_gain > 0.02 and b2["matched_precision"] >= b0["matched_precision"] - 0.01:
        return "A. SEPARATED MATCHING SUPPORTED", facts
    if (
        raw["top1_accuracy"] >= b0["top1_accuracy"] - 0.005
        and raw["matched_f1"] >= b2["matched_f1"] - 0.01
    ):
        return "E. UOT NOT NECESSARY FOR THIS STAGE", facts
    if b0["matched_f1"] >= b2["matched_f1"] - 0.01:
        return "B. GRAPH + UOT BASELINE SUFFICIENT", facts
    if b2["false_match_rate"] > b0["false_match_rate"] + 0.05:
        return "C. MASS-BASED REJECTION STILL NECESSARY", facts
    return "F. INCONCLUSIVE", facts


def _fmt(value: float) -> str:
    return "N/A" if not np.isfinite(value) else f"{value:.4f}"


def _report(
    comparison: pd.DataFrame,
    calibration: pd.DataFrame,
    graph_summary: pd.DataFrame,
    candidate_sources: pd.DataFrame,
    negative: pd.DataFrame,
    real: pd.DataFrame,
    domain: pd.DataFrame,
    baseline: pd.DataFrame,
    decision: str,
    facts: dict,
    thresholds: dict,
) -> str:
    values = comparison.set_index("method")
    rows = []
    for method in lib.MAIN_METHODS:
        value = values.loc[method]
        rows.append(
            f"| {method} | {value.top1_accuracy:.4f} | {value.matched_precision:.4f} | "
            f"{value.matched_recall:.4f} | {value.matched_f1:.4f} | {value.unmatched_f1:.4f} | "
            f"{_fmt(value.coverage_at_95_precision_empirical)} |"
        )
    b0, b1, b2, b3 = [values.loc[m] for m in lib.MAIN_METHODS]
    raw = values.loc[lib.METHOD_GRAPH_RAW]
    position = values.loc[lib.METHOD_POSITION]
    test95 = calibration.loc[
        (calibration["method"] == lib.METHOD_B2) & (calibration["target_precision"] == 0.95)
    ].iloc[0]
    graph_by_strength = graph_summary.groupby("strength").agg(
        auc=("graph_candidate_auc", "mean"),
        true_support=("true_support_median", "mean"),
        wrong_support=("wrong_support_median", "mean"),
    )
    real_values = real.set_index("method")
    domain_values = domain.pivot(index="variable", columns="dataset_group", values="median")
    neg = negative.groupby("method").agg(
        false_matches=("false_matched_count", "sum"),
        coverage=("coverage", "mean"),
        rejection=("rejection_fraction", "mean"),
    )
    source_agg = candidate_sources.loc[candidate_sources["split"] == "Test"].groupby("candidate_source").agg(
        recall=("candidate_recall_30", "mean"),
        top1=("position_graph_raw_top1_accuracy", "mean"),
        auc=("graph_candidate_auc", "mean"),
    )
    return f"""| Method | Top-1 | Matched Precision | Matched Recall | Matched F1 | Unmatched F1 | Coverage at 95% Precision |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

The table is the pooled held-out Real-like Test set. The last column is the descriptive held-out precision--coverage envelope, not a threshold retuned for deployment. Validation-fixed threshold performance is reported separately below.

# Main finding

The pre-specified research decision is **{decision}**.

Relative row-centering changed Graph-UOT Top-1 from {b0.top1_accuracy:.4f} to {b1.top1_accuracy:.4f} and native matched F1 from {b0.matched_f1:.4f} to {b1.matched_f1:.4f}. The separated method used the relative Graph-UOT Top-1, but replaced the absolute-mass gate with a synthetic-only logistic adoption model. Its matched precision/recall/F1 were {b2.matched_precision:.4f}/{b2.matched_recall:.4f}/{b2.matched_f1:.4f}.

The B2 match model was fit only on five Train base patterns. Its threshold was selected only on five Validation base patterns at target precision 0.95, then frozen. On held-out Test, that threshold produced precision {_fmt(test95.test_precision_at_validation_threshold)} and coverage {_fmt(test95.test_coverage_at_validation_threshold)}; target-achieved={bool(test95.test_target_achieved)}. Test labels were used to evaluate, never to refit or select the operational threshold.

# Is Graph still useful?

Position-only Top-1 was {position.top1_accuracy:.4f}; Position+Graph-UOT was {b0.top1_accuracy:.4f}, a change of {facts['graph_top1_gain_vs_position']:+.4f}. Candidate-level Graph AUC averaged {facts['graph_auc']:.4f}.

| Strength | Graph AUC | True support median | Other support median |
|---|---:|---:|---:|
{chr(10).join(f'| {idx} | {row.auc:.4f} | {row.true_support:.4f} | {row.wrong_support:.4f} |' for idx, row in graph_by_strength.iterrows())}

Candidate-source sensitivity on held-out cases:

| Candidate source | Recall@30 | Raw Position+Graph Top-1 | Graph AUC |
|---|---:|---:|---:|
{chr(10).join(f'| {idx} | {row.recall:.4f} | {row.top1:.4f} | {row.auc:.4f} |' for idx, row in source_agg.iterrows())}

Graph remains useful only to the extent supported by these Real-like persistent IDs. It is not evidence of Real correspondence accuracy.

# Is UOT necessary?

Graph ranking without UOT achieved Top-1 {raw.top1_accuracy:.4f}; relative Graph-UOT achieved {b1.top1_accuracy:.4f}. Their matched F1 values after their respective adoption logic were {raw.matched_f1:.4f} and {b2.matched_f1:.4f}. Across all rows, within-case duplicate-destination rates were {raw.top1_duplicate_destination_rate_all:.4f} for raw Graph ranking and {b1.top1_duplicate_destination_rate_all:.4f} for relative Graph-UOT. Relative UOT changed Top-1 by {facts['relative_uot_top1_gain_vs_raw_graph']:+.4f}; absolute-cost B0 changed it by {facts['uot_top1_gain_vs_raw_graph']:+.4f}. Thus UOT adds little to ranking in the row-centered path, but B2 is the only Graph-centered operational rule that transferred the Validation-selected 95% precision target to Test; raw Graph adoption reached {raw.matched_precision:.4f}, below 0.95.

Duplicate destinations are algorithmic many-to-one warnings, not proof of biological error; real 2-D detections may include split/merge exceptions.

# Does adoption separation help?

At the descriptive held-out 95% precision envelope, B0/B1/B2 coverages were {_fmt(b0.coverage_at_95_precision_empirical)}, {_fmt(b1.coverage_at_95_precision_empirical)}, and {_fmt(b2.coverage_at_95_precision_empirical)}. B2 minus B0 was {facts['separated_coverage_gain_at_95_precision']:+.4f}.

Negative controls generated {int(neg.loc[lib.METHOD_B0, 'false_matches'])} B0, {int(neg.loc[lib.METHOD_B1, 'false_matches'])} B1, and {int(neg.loc[lib.METHOD_B2, 'false_matches'])} B2 false accepted matches across the three controls. Identity permutation preserves geometry while declaring all IDs unrelated, so high confidence in that control exposes the fundamental inability of geometry-only matching to detect identity replacement; it is deliberately confounded rather than a realistic serial-section sample.

# Synthetic--Real gap

| Variable | Real-like Test median | Real median |
|---|---:|---:|
{chr(10).join(f'| {idx} | {_fmt(row.get("Real-like Test", np.nan))} | {_fmt(row.get("Real", np.nan))} |' for idx, row in domain_values.iterrows())}

The Real-like generator matched density by construction while retaining spatial nonuniformity, partial overlap, independent local missing clusters, section-specific nuclei, unequal cardinality, nonlinear deformation, and centroid noise. Any residual differences above limit external validity; Real values were not transformed to resemble synthetic.

# Real application

Real has no cell-level truth. Counts therefore measure adoption behavior only.

| Method | Matched | Ambiguous | Unmatched | Coverage | Confidence median | Reciprocal | Duplicate destinations |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(f'| {method} | {int(real_values.loc[method,"matched_n"])} | {int(real_values.loc[method,"ambiguous_n"])} | {int(real_values.loc[method,"unmatched_n"])} | {real_values.loc[method,"coverage"]:.4f} | {real_values.loc[method,"synthetic_derived_confidence_median"]:.4f} | {real_values.loc[method,"reciprocal_fraction"]:.4f} | {int(real_values.loc[method,"duplicate_destination_count"])} |' for method in lib.MAIN_METHODS)}

B2 confidence is derived exclusively from synthetic Train data and is not claimed to be a calibrated Real correctness probability. More Real matches are not interpreted as higher accuracy.

# Existing candidate-fixed baseline reproduction

The prior 15-case candidate-fixed results were read from the frozen Experiment A outputs and checked against their known values: Position+Graph Top-1/F1 {baseline.set_index('method').loc[lib.METHOD_B0,'uot_top1_accuracy']:.4f}/{baseline.set_index('method').loc[lib.METHOD_B0,'matched_f1']:.4f}; Full {baseline.set_index('method').loc[lib.METHOD_B3,'uot_top1_accuracy']:.4f}/{baseline.set_index('method').loc[lib.METHOD_B3,'matched_f1']:.4f}. These are not mixed with the new held-out Real-like table.

# Research decision

**{decision}**

The decision rule was fixed in code before interpreting Real results. Real adoption count is not part of the decision.

# Recommended next step

1. Repeat the frozen B2 calibrator on at least one independently designed low-density generator family; the current split prevents seed/template leakage but remains one generator family.
2. Obtain a small blinded Real correspondence set and evaluate candidate recall plus the frozen precision--coverage curve before changing any production threshold.
3. If UOT adds little beyond raw Graph ranking, test a simpler one-to-one conflict resolver as a separate, pre-registered experiment rather than weakening the mass gate ad hoc.

# Scope

No deformation field was generated. Workflow C, Streamlit UI, Workflow D, production safety gates, and production algorithms were not changed. Row centering is a controlled diagnostic/matching prototype, not an adopted production algorithm.
"""


def main() -> None:
    started = time.time()
    if (lib.OUT / "REPORT.md").exists():
        raise FileExistsError("Refusing to overwrite an existing completed experiment")
    lib.OUT.mkdir(parents=True, exist_ok=True)
    scenarios = lib.real_like_scenarios()
    split_manifest = pd.DataFrame(scenarios)
    if split_manifest["template_group"].duplicated().any():
        raise AssertionError("A base point template appears in more than one split")
    split_manifest.to_csv(lib.OUT / "split_manifest.csv", index=False)

    base_frames, graph_pairs, candidate_sources, geometry_rows, case_summaries = [], [], [], [], []
    candidate_hashes = {}
    for scenario in scenarios:
        print(f"Real-like {scenario['scenario_id']}", flush=True)
        sample = lib.generate_real_like_sample(seed=scenario["seed"], strength=scenario["strength"])
        density, knn = lib.point_density_and_knn(sample.moving_points)
        case_summaries.append(
            {
                **scenario,
                "fixed_n": len(sample.fixed_points),
                "moving_n": len(sample.moving_points),
                "common_n": int(sample.common_moving_mask.sum()),
                "true_unmatched_n": int((~sample.common_moving_mask).sum()),
                "moving_density_per_um2": density,
                "moving_knn_median_um": float(np.median(knn)),
            }
        )
        geometry_rows.append(
            pd.DataFrame(
                {
                    "scenario_id": scenario["scenario_id"],
                    "moving_index": np.arange(len(sample.moving_points)),
                    "nuclear_density_per_um2": density,
                    "knn_distance_um": knn,
                }
            )
        )
        context = lib.build_fixed_context(sample.fixed_points, sample.moving_points)
        candidate_hashes[scenario["scenario_id"]] = lib.candidate_hash(context.candidates.indices)
        runs = lib.method_runs(context)
        for method, run in runs.items():
            frame = lib.result_frame(sample, scenario, context, run, method=method)
            frame["dataset_group"] = "Real-like"
            base_frames.append(frame)
        raw = lib.graph_ranking_frame(sample, scenario, context)
        raw["dataset_group"] = "Real-like"
        base_frames.append(raw)
        graph_pairs.append(lib.graph_discrimination_rows(sample, scenario, context))
        if scenario["split"] == "Test":
            candidate_sources.append(lib.candidate_source_diagnostics(sample, scenario, context))

    base = pd.concat(base_frames, ignore_index=True)
    graph_pairs_table = pd.concat(graph_pairs, ignore_index=True)
    candidate_source_table = pd.concat(candidate_sources, ignore_index=True)
    geometry = pd.concat(geometry_rows, ignore_index=True)

    print("Fit separated adoption models", flush=True)
    match_model, unmatched_model, b2_thresholds, b2 = lib.fit_separated_models(
        base,
        source_method=lib.METHOD_B1,
        feature_names=lib.UOT_FEATURES,
        output_method=lib.METHOD_B2,
    )
    raw_match_model, raw_unmatched_model, raw_thresholds, raw_adopted = lib.fit_separated_models(
        base,
        source_method=lib.METHOD_GRAPH_RAW,
        feature_names=lib.RAW_FEATURES,
        output_method=lib.METHOD_GRAPH_RAW,
    )
    keep = base.loc[~base["method"].isin([lib.METHOD_GRAPH_RAW])]
    frames = pd.concat([keep, b2, raw_adopted], ignore_index=True, sort=False)
    if set(frames["method"].unique()) != {
        lib.METHOD_POSITION,
        lib.METHOD_B0,
        lib.METHOD_B1,
        lib.METHOD_B2,
        lib.METHOD_B3,
        lib.METHOD_GRAPH_RAW,
    }:
        raise AssertionError("Unexpected method set")

    per_case = _case_metrics(frames)
    comparison, curves = _aggregate_test(frames)
    calibration = _calibration_table(frames)
    cal95 = calibration.loc[calibration["target_precision"] == 0.95, [
        "method", "test_coverage_if_target_achieved"
    ]].rename(columns={"test_coverage_if_target_achieved": "coverage_at_95_precision_validation_fixed"})
    comparison = comparison.merge(cal95, on="method", validate="one_to_one")

    graph_summary_rows = []
    test_graph = graph_pairs_table.loc[graph_pairs_table["split"] == "Test"]
    for (scenario_id, strength), table in test_graph.groupby(["scenario_id", "strength"], sort=False):
        true = table["is_true_candidate"].to_numpy(bool)
        graph_summary_rows.append(
            {
                "scenario_id": scenario_id,
                "strength": strength,
                "graph_candidate_auc": lib.rank_auc(true, table["graph_support"]),
                "true_support_median": float(table.loc[true, "graph_support"].median()),
                "wrong_support_median": float(table.loc[~true, "graph_support"].median()),
            }
        )
    graph_summary = pd.DataFrame(graph_summary_rows)

    print("Negative controls", flush=True)
    negative_frames = []
    controls = (
        (5101, "identity_permutation"),
        (5102, "unrelated_pattern"),
        (5103, "translated_disjoint"),
    )
    for seed, kind in controls:
        print(f"Control {kind}", flush=True)
        sample = lib.make_negative_control(seed, kind)
        scenario = {
            "scenario_id": f"negative_{kind}_{seed}",
            "split": "Control",
            "strength": kind,
            "seed": seed,
            "template_group": f"negative_{seed}",
        }
        context = lib.build_fixed_context(sample.fixed_points, sample.moving_points)
        runs = lib.method_runs(context)
        local = []
        for method in (lib.METHOD_B0, lib.METHOD_B1, lib.METHOD_B3):
            frame = lib.result_frame(sample, scenario, context, runs[method], method=method)
            frame["dataset_group"] = "Negative control"
            local.append(frame)
        b1_source = local[1]
        b2_control = lib.apply_separated_adoption(
            b1_source,
            match_model,
            unmatched_model,
            b2_thresholds["match"]["threshold"],
            b2_thresholds["unmatched"]["threshold"],
            method=lib.METHOD_B2,
        )
        local.append(b2_control)
        negative_frames.extend(local)
    negative_table = pd.concat(negative_frames, ignore_index=True)
    negative_metrics = _negative_metrics(negative_table)

    print("Real exploratory application", flush=True)
    real_fixed = np.load(lib.REAL_INPUT / "success_metric_fixed_points.npy", allow_pickle=False)
    real_moving = np.load(lib.REAL_INPUT / "success_metric_moving_points.npy", allow_pickle=False)
    real_sample = SimpleNamespace(
        fixed_points=real_fixed,
        moving_points=real_moving,
        fixed_ids=np.asarray([f"real_fixed_{i}" for i in range(len(real_fixed))], dtype=object),
        moving_ids=np.asarray([f"real_moving_{i}" for i in range(len(real_moving))], dtype=object),
        true_fixed_index=np.full(len(real_moving), -1, dtype=int),
        common_moving_mask=np.zeros(len(real_moving), dtype=bool),
        seed=20260904,
        difficulty="Real",
    )
    real_scenario = {
        "scenario_id": "serial_section_pair_20260904",
        "split": "Real",
        "strength": "Real",
        "seed": 20260904,
        "template_group": "real_unlabeled",
    }
    real_context = lib.build_fixed_context(real_fixed, real_moving)
    real_runs = lib.method_runs(real_context)
    real_base = []
    for method in (lib.METHOD_B0, lib.METHOD_B1, lib.METHOD_B3):
        frame = lib.result_frame(real_sample, real_scenario, real_context, real_runs[method], method=method)
        frame["dataset_group"] = "Real"
        real_base.append(frame)
    real_b2 = lib.apply_separated_adoption(
        real_base[1],
        match_model,
        unmatched_model,
        b2_thresholds["match"]["threshold"],
        b2_thresholds["unmatched"]["threshold"],
        method=lib.METHOD_B2,
    )
    real_frames = pd.concat([*real_base, real_b2], ignore_index=True)
    real_metrics = _real_metrics(real_frames)

    domain = _domain_summary(frames, geometry, real_frames, real_fixed, real_moving)
    baseline = _baseline_reproduction()
    decision, decision_facts = _decision(comparison, graph_summary)

    # Save analysis artifacts only after all integrity checks and fits complete.
    comparison.to_csv(lib.OUT / "method_comparison.csv", index=False)
    per_case.to_csv(lib.OUT / "per_case_metrics.csv", index=False)
    curves.to_csv(lib.OUT / "precision_coverage.csv", index=False)
    calibration.to_csv(lib.OUT / "calibration_metrics.csv", index=False)
    pd.DataFrame(case_summaries).to_csv(lib.OUT / "real_like_synthetic_summary.csv", index=False)
    negative_metrics.to_csv(lib.OUT / "negative_control_metrics.csv", index=False)
    real_metrics.to_csv(lib.OUT / "real_exploratory_metrics.csv", index=False)
    frames.to_csv(lib.OUT / "per_nucleus_results.csv", index=False)
    negative_table.to_csv(lib.OUT / "negative_control_per_nucleus.csv", index=False)
    real_frames.to_csv(lib.OUT / "real_exploratory_per_nucleus.csv", index=False)
    graph_summary.to_csv(lib.OUT / "graph_discrimination_summary.csv", index=False)
    candidate_source_table.to_csv(lib.OUT / "candidate_source_sensitivity.csv", index=False)
    domain.to_csv(lib.OUT / "domain_comparison.csv", index=False)
    baseline.to_csv(lib.OUT / "baseline_reproduction.csv", index=False)

    test_curves = curves.loc[
        (curves["split"] == "Test") & curves["method"].isin(lib.MAIN_METHODS)
    ]
    main_comparison = comparison.loc[comparison["method"].isin(lib.MAIN_METHODS)]
    viz.precision_coverage(test_curves, lib.OUT / "precision_coverage.png")
    viz.matched_f1(main_comparison, lib.OUT / "matched_f1_comparison.png")
    viz.graph_discrimination(test_graph, lib.OUT / "graph_discrimination.png")
    viz.unmatched_detection(main_comparison, lib.OUT / "unmatched_detection.png")
    viz.mass_vs_accuracy(
        frames.loc[(frames["split"] == "Test") & (frames["method"] == lib.METHOD_B0)],
        lib.OUT / "mass_vs_accuracy.png",
    )
    viz.real_like_domain_gap(domain, lib.OUT / "real_like_domain_gap.png")
    viz.real_candidates(
        real_fixed,
        real_moving,
        real_frames.loc[real_frames["method"] == lib.METHOD_B2],
        lib.OUT / "real_candidate_examples.png",
    )

    source_paths = [
        lib.SOURCE / "config.py",
        lib.SOURCE / "candidate_generation.py",
        lib.SOURCE / "graph_consistency.py",
        lib.SOURCE / "contextual_uot.py",
        lib.COST_SOURCE / "run_cost_ablation.py",
    ]
    summary = {
        "purpose": "graph_matching_adoption_separation_analysis_only",
        "branch": subprocess.run(
            ["git", "branch", "--show-current"], cwd=lib.ROOT, text=True, capture_output=True, check=True
        ).stdout.strip(),
        "elapsed_seconds": time.time() - started,
        "decision": decision,
        "decision_facts": decision_facts,
        "split_policy": {
            "group_key": "base point pattern seed",
            "train_seeds": list(lib.SPLIT_SEEDS["Train"]),
            "validation_seeds": list(lib.SPLIT_SEEDS["Validation"]),
            "test_seeds": list(lib.SPLIT_SEEDS["Test"]),
            "test_used_for_training_or_threshold_selection": False,
            "real_used_for_training_or_threshold_selection": False,
        },
        "b2_match_model": match_model.to_dict(),
        "b2_unmatched_model": unmatched_model.to_dict(),
        "b2_thresholds": b2_thresholds,
        "raw_graph_match_model": raw_match_model.to_dict(),
        "raw_graph_unmatched_model": raw_unmatched_model.to_dict(),
        "raw_graph_thresholds": raw_thresholds,
        "candidate_hash_by_case": candidate_hashes,
        "existing_baseline_reproduced": True,
        "negative_controls": [kind for _, kind in controls],
        "real_has_ground_truth": False,
        "warping_performed": False,
        "production_modified": False,
        "source_sha256": {str(path.relative_to(lib.ROOT)): lib.sha256_file(path) for path in source_paths},
        "deterministic_test_trace_sha256": __import__("hashlib").sha256(
            frames.loc[frames["split"] == "Test"].to_csv(index=False).encode()
        ).hexdigest(),
    }
    lib.save_json(lib.OUT / "diagnostic_summary.json", summary)
    report = _report(
        comparison,
        calibration,
        graph_summary,
        candidate_source_table,
        negative_metrics,
        real_metrics,
        domain,
        baseline,
        decision,
        decision_facts,
        b2_thresholds,
    )
    (lib.OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(f"Complete: {decision} ({time.time() - started:.1f}s)", flush=True)


if __name__ == "__main__":
    main()

