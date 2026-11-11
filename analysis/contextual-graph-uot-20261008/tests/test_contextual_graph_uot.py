from __future__ import annotations

import numpy as np

from candidate_generation import build_candidate_rankings, build_candidate_set, candidate_recall
from config import DIFFICULTIES
from graph_consistency import graph_support_scores
from local_fingerprint import compute_fingerprint_pair, pairwise_fingerprint_cost
from regional_pattern import match_regional_patterns
from run_experiment import match_all
from synthetic_correspondence import assert_identity_alignment, generate_correspondence_sample


def sample(difficulty="Easy", seed=17, n_latent=100):
    return generate_correspondence_sample(
        seed=seed,
        difficulty=difficulty,
        n_latent=n_latent,
        **DIFFICULTIES[difficulty],
    )


def test_fingerprint_is_rotation_invariant_and_finite():
    rng = np.random.default_rng(4)
    points = rng.normal(size=(60, 2)) * np.array([25.0, 14.0])
    theta = 0.73
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated = points @ rotation.T + np.array([30.0, -11.0])
    pair = compute_fingerprint_pair(points, rotated)
    np.testing.assert_allclose(pair.fixed, pair.moving, atol=1e-8, rtol=1e-8)
    assert np.isfinite(pair.fixed).all()


def test_persistent_ids_remain_aligned_after_shuffle():
    value = sample()
    assert_identity_alignment(value)
    common = np.flatnonzero(value.common_moving_mask)
    assert np.array_equal(
        value.moving_ids[common], value.fixed_ids[value.true_fixed_index[common]]
    )


def test_candidate_generation_retains_known_matches():
    value = sample(n_latent=120)
    regional = match_regional_patterns(value.fixed_points, value.moving_points)
    fingerprints = compute_fingerprint_pair(value.fixed_points, value.moving_points)
    fp_cost = pairwise_fingerprint_cost(fingerprints)
    rankings = build_candidate_rankings(value.fixed_points, value.moving_points, fp_cost, regional)
    assert candidate_recall(rankings.combined, value.true_fixed_index, 30) >= 0.95


def test_graph_support_prefers_true_structural_candidates():
    value = sample(n_latent=120)
    regional = match_regional_patterns(value.fixed_points, value.moving_points)
    fingerprints = compute_fingerprint_pair(value.fixed_points, value.moving_points)
    fp_cost = pairwise_fingerprint_cost(fingerprints)
    rankings = build_candidate_rankings(value.fixed_points, value.moving_points, fp_cost, regional)
    candidates = build_candidate_set(rankings, fp_cost, regional, include_region=True)
    support = graph_support_scores(value.fixed_points, value.moving_points, candidates)
    true_scores, wrong_scores = [], []
    for moving_index, true_index in enumerate(value.true_fixed_index):
        if true_index < 0:
            continue
        slots = np.flatnonzero(candidates.indices[moving_index] == true_index)
        if slots.size:
            true_scores.append(support[moving_index, slots[0]])
        wrong_scores.extend(support[moving_index, candidates.indices[moving_index] != true_index][:3])
    assert np.median(true_scores) > np.median(wrong_scores)


def test_uot_assigns_less_mass_to_obvious_unmatched_points():
    value = sample("Hard", n_latent=130)
    results, *_ = match_all(value.fixed_points, value.moving_points)
    full = results[-1]
    common = value.common_moving_mask
    assert np.median(full.row_mass_ratio[~common]) < np.median(full.row_mass_ratio[common])


def test_fixed_seed_is_deterministic():
    first = sample(seed=23)
    second = sample(seed=23)
    np.testing.assert_array_equal(first.fixed_points, second.fixed_points)
    np.testing.assert_array_equal(first.moving_points, second.moving_points)
    np.testing.assert_array_equal(first.true_fixed_index, second.true_fixed_index)


def test_easy_case_is_finite_and_has_high_correspondence_accuracy():
    value = sample(n_latent=120)
    results, *_ = match_all(value.fixed_points, value.moving_points)
    full = results[-1]
    common = value.common_moving_mask
    accuracy = np.mean(full.top1_index[common] == value.true_fixed_index[common])
    assert accuracy >= 0.85
    assert np.isfinite(full.confidence).all()
    assert np.isfinite(full.top1_top2_margin).all()
    assert np.isfinite(full.row_mass_ratio).all()

