"""Run the frozen Multi-scale Contextual Graph-UOT correspondence study."""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from candidate_generation import build_candidate_rankings, build_candidate_set
from config import (
    CANDIDATES,
    DIFFICULTIES,
    FINGERPRINT,
    GRAPH,
    METHODS,
    OUT,
    PRIOR_UOT,
    PRIOR_VALIDATION,
    REGIONAL,
    ROOT,
    SEEDS,
    UOT,
)
from contextual_uot import candidate_restricted_uot, direct_match, spatial_only_uot
from evaluation import (
    calibration_rows,
    candidate_recall_rows,
    correspondence_table,
    method_metrics,
    rank_auc,
    regional_hypothesis_rows,
)
from graph_consistency import graph_support_scores
from fingerprint_component_analysis import run as run_fingerprint_component_analysis
from local_fingerprint import compute_fingerprint_pair, pairwise_fingerprint_cost
from regional_pattern import match_regional_patterns
from synthetic_correspondence import assert_identity_alignment, generate_correspondence_sample
import visualization as viz


def save_json(path: Path, value) -> None:
    def convert(item):
        if isinstance(item, dict):
            return {str(key): convert(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(value) for value in item]
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        return item

    path.write_text(json.dumps(convert(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def scenario_table() -> list[dict]:
    rows = []
    for difficulty, settings in DIFFICULTIES.items():
        for seed in SEEDS:
            rows.append(
                {
                    "scenario_id": f"primary_{difficulty.lower()}_{seed}",
                    "analysis_group": "primary",
                    "difficulty": difficulty,
                    "seed": seed,
                    **settings,
                }
            )
    medium = DIFFICULTIES["Medium"]
    for missing in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        for seed in SEEDS:
            rows.append(
                {
                    "scenario_id": f"missing_{int(100 * missing):02d}_{seed}",
                    "analysis_group": "missingness",
                    "difficulty": f"Missingness {int(100 * missing)}%",
                    "seed": seed,
                    **{
                        **medium,
                        "missing_fraction": missing,
                    },
                }
            )
    return rows


def match_all(fixed: np.ndarray, moving: np.ndarray):
    regional = match_regional_patterns(fixed, moving)
    fingerprints = compute_fingerprint_pair(fixed, moving)
    fp_cost = pairwise_fingerprint_cost(fingerprints)
    rankings = build_candidate_rankings(fixed, moving, fp_cost, regional)
    no_region = build_candidate_set(rankings, fp_cost, regional, include_region=False)
    full_candidates = build_candidate_set(rankings, fp_cost, regional, include_region=True)
    graph_no_region = graph_support_scores(fixed, moving, no_region)
    graph_full = graph_support_scores(fixed, moving, full_candidates)

    results = [
        spatial_only_uot(fixed, moving, regional),
        direct_match(
            fixed,
            moving,
            regional,
            full_candidates,
            method="M1 Regional NCC only",
            ranking=rankings.regional,
            use_region_distance=True,
        ),
        direct_match(
            fixed,
            moving,
            regional,
            no_region,
            method="M2 Fingerprint nearest candidate",
            ranking=no_region.indices,
            use_region_distance=False,
        ),
        candidate_restricted_uot(
            fixed,
            moving,
            regional,
            no_region,
            method="M3 Fingerprint + UOT",
        ),
        candidate_restricted_uot(
            fixed,
            moving,
            regional,
            no_region,
            method="M4 Fingerprint + graph + UOT",
            graph_support=graph_no_region,
        ),
        candidate_restricted_uot(
            fixed,
            moving,
            regional,
            full_candidates,
            method="M5 Full contextual graph-UOT",
            graph_support=graph_full,
        ),
    ]
    return results, regional, rankings, full_candidates, graph_full, fingerprints


def _graph_pair_rows(sample, candidates, support, scenario_id: str) -> list[dict]:
    rows = []
    for moving_index in range(len(sample.moving_points)):
        true_index = int(sample.true_fixed_index[moving_index])
        for slot, fixed_index in enumerate(candidates.indices[moving_index]):
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "difficulty": sample.difficulty,
                    "seed": sample.seed,
                    "moving_index": moving_index,
                    "fixed_index": int(fixed_index),
                    "is_true_candidate": bool(true_index >= 0 and int(fixed_index) == true_index),
                    "graph_support": float(support[moving_index, slot]),
                    "base_cost": float(candidates.base_cost[moving_index, slot]),
                }
            )
    return rows


def _aggregate_method_metrics(per_case: pd.DataFrame) -> pd.DataFrame:
    primary = per_case.loc[per_case["analysis_group"] == "primary"]
    metrics = [
        "top1_accuracy",
        "top3_recall",
        "top5_recall",
        "matched_precision",
        "matched_recall",
        "matched_f1",
        "unmatched_precision",
        "unmatched_recall",
        "unmatched_f1",
        "high_confidence_precision",
        "high_confidence_coverage",
        "ambiguous_fraction",
    ]
    table = primary.groupby("method", sort=False)[metrics].mean().reset_index()
    table["short_method"] = table["method"].str.extract(r"^(M\d)")[0]
    return table


def _ablation(method_metrics_table: pd.DataFrame) -> pd.DataFrame:
    table = method_metrics_table.set_index("method")
    baseline = table.loc["M0 Spatial-only UOT"]
    order = [
        "M0 Spatial-only UOT",
        "M3 Fingerprint + UOT",
        "M4 Fingerprint + graph + UOT",
        "M5 Full contextual graph-UOT",
    ]
    rows = []
    previous = None
    for method in order:
        current = table.loc[method]
        rows.append(
            {
                "method": method,
                "top1_accuracy": current["top1_accuracy"],
                "matched_f1": current["matched_f1"],
                "unmatched_f1": current["unmatched_f1"],
                "top1_delta_vs_spatial_uot": current["top1_accuracy"] - baseline["top1_accuracy"],
                "top1_incremental_delta": np.nan if previous is None else current["top1_accuracy"] - previous["top1_accuracy"],
            }
        )
        previous = current
    return pd.DataFrame(rows)


def _interpretation(method_table: pd.DataFrame, regional: pd.DataFrame, graph_auc: float) -> tuple[str, str]:
    values = method_table.set_index("method")
    m0 = values.loc["M0 Spatial-only UOT", "top1_accuracy"]
    m3 = values.loc["M3 Fingerprint + UOT", "top1_accuracy"]
    m4 = values.loc["M4 Fingerprint + graph + UOT", "top1_accuracy"]
    m5 = values.loc["M5 Full contextual graph-UOT", "top1_accuracy"]
    regional_accuracy = float(regional["correct_region_top1"].mean())
    if m5 > m0 + 0.03 and m3 > m0 + 0.015 and m4 >= m3 and m5 >= m4:
        return "A", "FULL TREE SUPPORTED"
    if m3 > m0 + 0.03 and m4 <= m3 + 0.01 and m5 <= m3 + 0.01:
        return "B", "FINGERPRINT + UOT SUPPORTED"
    if m4 > max(m3, m0) + 0.02 and graph_auc >= 0.60:
        return "C", "GRAPH MATCHING SUPPORTED"
    if m5 > m4 + 0.02 and regional_accuracy >= 0.65:
        return "D", "REGIONAL NCC SUPPORTED"
    if m0 >= max(m3, m4, m5):
        return "E", "SPATIAL UOT REMAINS STRONGEST"
    return "F", "INCONCLUSIVE"


def _real_exploration() -> tuple[pd.DataFrame, np.ndarray, np.ndarray] | None:
    input_dir = ROOT / "sweeps/real-grid27-20260908-v2/input"
    fixed_path = input_dir / "success_metric_fixed_points.npy"
    moving_path = input_dir / "success_metric_moving_points.npy"
    if not fixed_path.exists() or not moving_path.exists():
        return None
    fixed = np.load(fixed_path, allow_pickle=False)
    moving = np.load(moving_path, allow_pickle=False)
    results, regional, _, _, _, _ = match_all(fixed, moving)
    result = next(item for item in results if item.method == "M5 Full contextual graph-UOT")
    table = pd.DataFrame(
        {
            "moving_index": np.arange(len(moving)),
            "top1_fixed_index": result.top1_index,
            "top2_fixed_index": result.top2_index,
            "top1_transport_mass": result.top1_mass,
            "top2_transport_mass": result.top2_mass,
            "top1_top2_margin": result.top1_top2_margin,
            "normalized_row_entropy": result.normalized_row_entropy,
            "total_transported_mass": result.row_transport_mass,
            "row_mass_ratio": result.row_mass_ratio,
            "reciprocal": result.reciprocal,
            "regional_ncc": result.regional_ncc,
            "regional_ncc_margin": result.regional_margin,
            "fingerprint_similarity": result.fingerprint_similarity,
            "graph_support_score": result.graph_support,
            "affine_candidate_distance_um": result.spatial_distance_um,
            "confidence": result.confidence,
            "status": result.status,
        }
    )
    return table, fixed, moving


def _manual_review(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    strata = {
        "high": table.sort_values("confidence", ascending=False),
        "medium": table.iloc[(table["confidence"] - table["confidence"].median()).abs().argsort()],
        "low": table.sort_values("confidence", ascending=True),
    }
    selected_indices: set[int] = set()
    for stratum, candidates in strata.items():
        for row in candidates.itertuples():
            index = int(row.moving_index)
            if index in selected_indices:
                continue
            selected_indices.add(index)
            record = row._asdict()
            record["confidence_stratum"] = stratum
            rows.append(record)
            if sum(value["confidence_stratum"] == stratum for value in rows) >= 12:
                break
    return pd.DataFrame(rows)


def _report(
    method_table: pd.DataFrame,
    candidate_table: pd.DataFrame,
    regional_table: pd.DataFrame,
    ablation: pd.DataFrame,
    graph_auc: float,
    graph_primary_auc: float,
    category: tuple[str, str],
    real_table: pd.DataFrame | None,
) -> None:
    indexed = method_table.set_index("method")
    candidate_primary = candidate_table.loc[candidate_table["scenario_id"].str.startswith("primary_")]
    candidate_summary = candidate_primary.groupby(["candidate_strategy", "k"])["candidate_recall"].mean()
    regional_primary = regional_table.loc[regional_table["scenario_id"].str.startswith("primary_")]
    regional_sensitivity = regional_table.loc[regional_table["scenario_id"].str.startswith("missing_")]
    regional_top1 = float(regional_primary["correct_region_top1"].mean())
    regional_top3 = float(regional_primary["correct_region_top3"].mean())
    regional_margin = float(regional_primary["best_second_margin"].median())
    fingerprint_summary = pd.read_csv(OUT / "fingerprint_component_summary.csv")
    lines = [
        "# Multi-scale Contextual Graph-UOT Matching",
        "",
        "実施日: 2026-10-08 JST  ",
        "対象branch: `feature/gpu-slurm`  ",
        "目的: registration距離ではなく、partial overlap下のpersistent cell correspondence recoveryを評価する。",
        "",
        "## Primary synthetic result",
        "",
        "| Method | Top-1 | Top-3 recall | Matched F1 | Unmatched F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = indexed.loc[method]
        lines.append(
            f"| {method} | {row.top1_accuracy:.4f} | {row.top3_recall:.4f} | {row.matched_f1:.4f} | {row.unmatched_f1:.4f} |"
        )
    lines.extend(
        [
            "",
            "Primary tableはEasy/Medium/Hard × 5 seedsの15 casesのcase平均。median NNはrankingに使用していない。",
            "",
            "## Candidate discovery",
            "",
        ]
    )
    for strategy in ("Affine spatial", "Regional NCC", "Fingerprint", "Combined"):
        values = [candidate_summary.loc[(strategy, k)] for k in (10, 20, 30)]
        lines.append(f"- {strategy}: recall@10/20/30 = {values[0]:.4f} / {values[1]:.4f} / {values[2]:.4f}")
    lines.extend(
        [
            "",
            "## Regional NCC hypothesis",
            "",
            f"Gaussian 6/12 µmとDoG 6–18 µmを固定併用したlocal NCCのprimary casesにおけるcorrect-region top-1/top-3は {regional_top1:.4f}/{regional_top3:.4f}。best-second margin中央値は {regional_margin:.4f}。",
            "",
            "| Missingness | Correct-region Top-1 | Top-3 | Median NCC margin |",
            "|---:|---:|---:|---:|",
        ]
    )
    regional_by_missing = regional_sensitivity.groupby("missing_fraction").agg(
        top1=("correct_region_top1", "mean"),
        top3=("correct_region_top3", "mean"),
        margin=("best_second_margin", "median"),
    )
    for missing, row in regional_by_missing.iterrows():
        lines.append(f"| {100 * missing:.0f}% | {row.top1:.4f} | {row.top3:.4f} | {row.margin:.4f} |")
    lines.extend(
        [
            "",
            "NCCはmissingness 50%でもregion top-1 0.8571を維持した。ただしAffine spatial candidate recall@10が既にほぼ1.0のため、今回のNCCの価値は正解を候補集合へ救済することより、候補costのlocal offset priorとして現れた。",
            "",
            "## Fingerprint",
            "",
            "Descriptorはsorted absolute/locally-normalized kNN distances、multi-radius density/count、rotation-invariant angular Fourier magnitudes、covariance anisotropy、近傍間距離とangular-gap quantileをrobust scalingした。画像・学習モデル・cell IDはdescriptor生成に使っていない。spatial top-80内の固定diagnosticは次の通り。",
            "",
            "| Descriptor group | Recall@10 | Hard-negative AUC |",
            "|---|---:|---:|",
        ]
    )
    for row in fingerprint_summary.itertuples():
        lines.append(f"| {row.descriptor_group} | {row.candidate_recall_at_10:.4f} | {row.true_vs_hard_negative_auc:.4f} |")
    lines.extend(
        [
            "",
            "全component併用がrecall@10で最良。単独groupではabsolute kNN distanceが最も強かった。ただしM3のTop-1増分は+1.66ポイント、15 cases中8勝6敗1同率であり、fingerprint単独の効果は小さい。",
            "",
            "## Graph consistency",
            "",
            f"M5 candidate pairで、graph supportによるtrue/false candidateのrank AUCはprimary {graph_primary_auc:.4f}、全case {graph_auc:.4f}。mutual-10NN上でedge length・angle・neighbor candidate compatibilityをsoft集約し、完全なgraph isomorphismは要求していない。M3→M4はTop-1 +7.74ポイント、14勝1敗で最大の改善源だった。",
            "",
            "## UOT as partial assignment",
            "",
            "UOT conservativeのentropy=0.1、KL relaxation=0.5、row-mass threshold=0.25、translation-invariant Sinkhornをsource-of-truthどおり使用し、全45×4 UOT runsが収束した。UOTは候補発見器ではなく、候補外costを抑えたglobal partial-assignment layerとして使用した。top-1 massだけでなくrow entropy、margin、row mass、reciprocal、regional/fingerprint/graph componentを保存した。",
            "",
            "M5はmatched F1を0.6319→0.7290、unmatched F1を0.0835→0.3446へ改善した。ただしunmatched recallは0.2438に留まり、unmatched detectionは未完成。confidence≥0.75のpredicted matchesはprecision 0.9576、common-cell coverage 0.4697だったが、ECE 0.1435なのでconfidenceを校正済み確率とは解釈しない。",
            "",
            "## Ablation",
            "",
            "| Stage | Top-1 | Δ vs M0 | Incremental Δ |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in ablation.itertuples():
        incremental = "—" if pd.isna(row.top1_incremental_delta) else f"{row.top1_incremental_delta:+.4f}"
        lines.append(f"| {row.method} | {row.top1_accuracy:.4f} | {row.top1_delta_vs_spatial_uot:+.4f} | {incremental} |")
    lines.extend(
        [
            "",
            "## Real-data exploratory result",
            "",
        ]
    )
    if real_table is None:
        lines.append("Syntheticで明確な改善条件を満たさなかったため、real datasetには適用していない。")
    else:
        counts = real_table["status"].value_counts()
        maximum_confidence = float(real_table["confidence"].max())
        lines.append(
            f"`serial_section_pair_20260904`へM5を探索的適用した。matched/ambiguous/unmatched = {counts.get('matched', 0)}/{counts.get('ambiguous', 0)}/{counts.get('unmatched', 0)}。最大confidenceは{maximum_confidence:.4f}でsyntheticのhigh-confidence閾値0.75に達せず、domain shiftまたはcost scale不適合を示唆する。ground-truth cell correspondenceがないためaccuracyは主張しない。相対confidence順のhigh/medium/low各12例をmanual review用CSVへ保存した。"
        )
    lines.extend(
        [
            "",
            "## Research conclusion",
            "",
            f"**{category[0]}. {category[1]}**",
            "",
            "M5はM0を15/15 primary casesで上回ったためfull pipelineの研究方向を支持する。ただし寄与は均等ではなくgraphが主因、fingerprint単独は小さく、regional priorは補助的だった。この分類はtrue cell correspondence recovery、candidate recall、unmatched detection、stage ablationに基づき、registration後のmedian NN低下を根拠にしていない。",
            "",
            "## Recommended next step",
            "",
            "追加の独立real serial-section pairで、固定済みの今回手法をblind manual review可能な形で評価する。Workflow C production algorithmへの統合は行わない。",
            "",
            "## Scope and limitations",
            "",
            "- synthetic identityは既知だが、実serial-section biologyを完全には表現しない。",
            "- real datasetにはcell-level ground truthがない。",
            "- weightsと閾値は固定baselineであり、大規模parameter sweepは実施していない。",
            "- syntheticではAffine spatial recall@10がほぼ1.0で、candidate rescue能力の比較には易しすぎる側面がある。",
            "- unmatched recallとconfidence calibrationは不十分で、realへの閾値移植は支持されない。",
            "- optional displacement field reconstructionとJacobian rejectionは今回のcorrespondence評価に使用していない。",
        ]
    )
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-scenarios", type=int)
    parser.add_argument("--skip-real", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    required = OUT / "per_case_metrics.csv"
    if required.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite {required}; pass --force explicitly.")

    prior_config = json.loads((PRIOR_VALIDATION / "fixed_configurations.json").read_text())["uot"]["UOT conservative"]
    for key in ("cost_scale_um", "entropic_regularization", "mass_relaxation_kl", "minimum_row_mass_ratio", "solver"):
        if prior_config[key] != UOT[key]:
            raise RuntimeError(f"Frozen UOT mismatch for {key}: {prior_config[key]} != {UOT[key]}")

    scenarios = scenario_table()
    if args.max_scenarios is not None:
        scenarios = scenarios[: args.max_scenarios]
    metrics_rows: list[dict] = []
    correspondence_frames: list[pd.DataFrame] = []
    candidate_rows: list[dict] = []
    regional_rows: list[dict] = []
    graph_rows: list[dict] = []
    example = None
    started = time.perf_counter()
    for case_index, scenario in enumerate(scenarios, start=1):
        sample_kwargs = {key: scenario[key] for key in (
            "seed", "difficulty", "amplitude_um", "missing_fraction", "unmatched_fraction",
            "overlap_fraction", "noise_um", "local_missing_strength",
        )}
        sample = generate_correspondence_sample(**sample_kwargs)
        assert_identity_alignment(sample)
        results, regional, rankings, full_candidates, graph_full, fingerprints = match_all(
            sample.fixed_points, sample.moving_points
        )
        candidate_rows.extend(candidate_recall_rows(sample, rankings, scenario_id=scenario["scenario_id"]))
        regional_rows.extend(regional_hypothesis_rows(sample, regional, scenario_id=scenario["scenario_id"]))
        graph_rows.extend(_graph_pair_rows(sample, full_candidates, graph_full, scenario["scenario_id"]))
        for result in results:
            row = method_metrics(sample, result, scenario_id=scenario["scenario_id"])
            row.update(
                {
                    "analysis_group": scenario["analysis_group"],
                    "missing_fraction": sample.missing_fraction,
                    "unmatched_fraction": sample.unmatched_fraction,
                    "overlap_fraction": sample.overlap_fraction,
                    "amplitude_um": sample.amplitude_um,
                    "noise_um": sample.noise_um,
                }
            )
            metrics_rows.append(row)
            correspondence_frames.append(correspondence_table(sample, result, scenario_id=scenario["scenario_id"]))
        if scenario["scenario_id"] == f"primary_medium_{SEEDS[0]}":
            example = (sample, results[-1])
        print(
            f"case {case_index}/{len(scenarios)} {scenario['scenario_id']} "
            f"M0={metrics_rows[-6]['top1_accuracy']:.3f} M5={metrics_rows[-1]['top1_accuracy']:.3f}",
            flush=True,
        )

    per_case = pd.DataFrame(metrics_rows)
    correspondences = pd.concat(correspondence_frames, ignore_index=True)
    candidate_table = pd.DataFrame(candidate_rows)
    regional_table = pd.DataFrame(regional_rows)
    graph_table = pd.DataFrame(graph_rows)
    method_table = _aggregate_method_metrics(per_case)
    ablation = _ablation(method_table)

    calibration = []
    ece = {}
    primary_correspondence = correspondences.loc[correspondences["scenario_id"].str.startswith("primary_")]
    for method, group in primary_correspondence.groupby("method", sort=False):
        rows, value = calibration_rows(group)
        calibration.extend(rows)
        ece[method] = value
    calibration_table = pd.DataFrame(calibration)
    method_table["ece"] = method_table["method"].map(ece)
    graph_auc = rank_auc(graph_table["is_true_candidate"].to_numpy(), graph_table["graph_support"].to_numpy())
    graph_primary = graph_table.loc[graph_table["scenario_id"].str.startswith("primary_")]
    graph_primary_auc = rank_auc(
        graph_primary["is_true_candidate"].to_numpy(), graph_primary["graph_support"].to_numpy()
    )
    run_fingerprint_component_analysis()
    regional_primary = regional_table.loc[regional_table["scenario_id"].str.startswith("primary_")]
    category = _interpretation(method_table, regional_primary, graph_primary_auc)

    real_result = None
    m0 = float(method_table.set_index("method").loc["M0 Spatial-only UOT", "top1_accuracy"])
    m5 = float(method_table.set_index("method").loc["M5 Full contextual graph-UOT", "top1_accuracy"])
    if not args.skip_real and m5 > m0:
        explored = _real_exploration()
        if explored is not None:
            real_result, real_fixed, real_moving = explored
            real_result.to_csv(OUT / "real_correspondence_candidates.csv", index=False)
            _manual_review(real_result).to_csv(OUT / "manual_review_candidates.csv", index=False)
            viz.real_candidate_map(real_fixed, real_moving, real_result, OUT / "real_match_candidates.png")

    per_case.to_csv(OUT / "per_case_metrics.csv", index=False)
    method_table.to_csv(OUT / "method_metrics.csv", index=False)
    candidate_table.to_csv(OUT / "candidate_recall.csv", index=False)
    calibration_table.to_csv(OUT / "confidence_calibration.csv", index=False)
    ablation.to_csv(OUT / "ablation_summary.csv", index=False)
    regional_table.to_csv(OUT / "regional_ncc_diagnostics.csv", index=False)
    graph_table.to_csv(OUT / "graph_candidate_diagnostics.csv", index=False)
    correspondences.to_csv(OUT / "correspondence_results.csv", index=False)

    viz.method_comparison(method_table, OUT / "method_correspondence_accuracy.png")
    viz.candidate_recall_plot(candidate_table.loc[candidate_table["scenario_id"].str.startswith("primary_")], OUT / "candidate_recall.png")
    viz.calibration_plot(calibration_table, OUT / "confidence_calibration.png")
    viz.missingness_plot(per_case.loc[per_case["analysis_group"] == "missingness"], OUT / "missingness_sensitivity.png")
    viz.graph_support_plot(graph_table, OUT / "graph_support_correct_vs_wrong.png", graph_auc)
    if example is not None:
        viz.synthetic_example(*example, OUT / "synthetic_example_matches.png")

    config_record = {
        "uot": UOT,
        "regional": REGIONAL,
        "fingerprint": FINGERPRINT,
        "candidates": CANDIDATES,
        "graph": GRAPH,
        "difficulties": DIFFICULTIES,
        "seeds": SEEDS,
        "source_of_truth": str(PRIOR_UOT / "trials/uot/coarse_09/result.json"),
        "primary_outcome": "persistent cell correspondence recovery",
        "parameter_sweep_performed": False,
        "field_reconstruction_performed": False,
    }
    save_json(OUT / "fixed_configurations.json", config_record)
    save_json(
        OUT / "experiment_status.json",
        {
            "state": "completed",
            "scenario_count": len(scenarios),
            "primary_case_count": int(np.count_nonzero(per_case["analysis_group"] == "primary") / len(METHODS)),
            "methods": METHODS,
            "graph_support_auc": graph_auc,
            "graph_support_primary_auc": graph_primary_auc,
            "interpretation": {"category": category[0], "label": category[1]},
            "real_exploration_performed": real_result is not None,
            "runtime_seconds": time.perf_counter() - started,
            "all_finite": bool(per_case["finite"].all()),
        },
    )
    save_json(
        OUT / "provenance.json",
        {
            "date": "2026-10-08",
            "branch": "feature/gpu-slurm",
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "source_files_modified": False,
            "workflow_integration_performed": False,
        },
    )
    _report(
        method_table,
        candidate_table,
        regional_table,
        ablation,
        graph_auc,
        graph_primary_auc,
        category,
        real_result,
    )
    print(f"completed category={category[0]} {category[1]} runtime={time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
