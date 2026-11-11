"""Residual-structure diagnostics after a global affine (analysis only).

Nothing here estimates, stores or applies a deformation field to Real points. Residual vectors are *proxies*
(mutual-NN pairs, correlogram peaks, B2 matches); they are not true cell displacements. The held-out predictor is a
test statistic evaluated at held-out nuclei and is discarded after scoring.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

LAG_BINS = ((0, 20), (20, 50), (50, 100), (100, 200))
BANDWIDTHS = (20.0, 50.0, 100.0, 200.0)
WINDOWS = (20.0, 50.0, 100.0, 200.0)


# ----------------------------------------------------------------------------- residual sources
def proxy_pairs(mov_w, fixed, r=4.0):
    """Mutual nearest-neighbour pairs within r um. Returns moving idx, fixed idx, vectors u = fixed - moving."""
    d, j = cKDTree(fixed).query(mov_w)
    back = cKDTree(mov_w).query(fixed)[1]
    ok = (d <= r) & (back[j] == np.arange(len(mov_w)))
    i = np.flatnonzero(ok)
    return i, j[ok], fixed[j[ok]] - mov_w[ok]


# ----------------------------------------------------------------------------- spatial statistics
def vector_autocorr(pos, vec, bins=LAG_BINS):
    """rho(r) = mean(u_i . u_j) / mean(|u|^2) over pairs in each lag bin, after removing the mean vector."""
    u = vec - vec.mean(0)
    var = float(np.mean(np.sum(u * u, 1)))
    pairs = cKDTree(pos).query_pairs(max(b[1] for b in bins), output_type="ndarray")
    if len(pairs) == 0:
        return {f"rho_{a}_{b}": np.nan for a, b in bins} | {f"cov_{a}_{b}": np.nan for a, b in bins} | {"var": var}
    dd = np.linalg.norm(pos[pairs[:, 0]] - pos[pairs[:, 1]], axis=1)
    dots = np.sum(u[pairs[:, 0]] * u[pairs[:, 1]], 1)
    out = {"var": var}
    for a, b in bins:
        s = (dd >= a) & (dd < b)
        c = float(dots[s].mean()) if s.any() else np.nan
        out[f"cov_{a}_{b}"] = c
        out[f"rho_{a}_{b}"] = c / var if var > 0 else np.nan
        out[f"npairs_{a}_{b}"] = int(s.sum())
    return out


def direction_consistency(pos, vec, radii=(20.0, 50.0, 100.0), min_mag=0.3):
    """Mean resultant length of neighbours' unit residual vectors (self excluded), averaged over points."""
    u = vec - vec.mean(0)
    mag = np.linalg.norm(u, axis=1)
    keep = mag >= min_mag
    p, e = pos[keep], u[keep] / mag[keep, None]
    tree = cKDTree(p)
    out = {}
    for r in radii:
        L = []
        for i, nb in enumerate(tree.query_ball_point(p, r)):
            nb = [k for k in nb if k != i]
            if len(nb) >= 3:
                L.append(np.linalg.norm(e[nb].mean(0)))
        out[f"dircons_r{r:g}"] = float(np.mean(L)) if L else np.nan
    return out


def window_means(pos, vec, w, min_n=3, origin=None):
    o = pos.min(0) if origin is None else np.asarray(origin)
    key = np.floor((pos - o) / w).astype(int)
    rows = {}
    for k, v in zip(map(tuple, key), vec):
        rows.setdefault(k, []).append(v)
    cent, mean, n = [], [], []
    for k, vs in rows.items():
        if len(vs) >= min_n:
            cent.append(o + (np.array(k) + 0.5) * w); mean.append(np.mean(vs, 0)); n.append(len(vs))
    return np.array(cent).reshape(-1, 2), np.array(mean).reshape(-1, 2), np.array(n)


def window_summary(pos, vec):
    out = {}
    for w in WINDOWS:
        _, m, n = window_means(pos, vec - vec.mean(0), w)
        mag = np.linalg.norm(m, axis=1) if len(m) else np.array([np.nan])
        out[f"win{w:g}_mean_mag_median"] = float(np.median(mag))
        # expected |mean| of n iid vectors with the pooled spread (noise floor)
        sd = np.sqrt(np.mean(np.sum((vec - vec.mean(0)) ** 2, 1)))
        out[f"win{w:g}_noise_floor_median"] = float(np.median(sd * np.sqrt(np.pi / 4) / np.sqrt(np.maximum(n, 1)))) if len(n) else np.nan
        out[f"win{w:g}_n_windows"] = int(len(m))
    return out


# ----------------------------------------------------------------------------- held-out prediction
def _nw(pos_fit, vec_fit, q, h, mean, k0=1.0):
    """Nadaraya-Watson with shrinkage to the fit mean (k0 pseudo-weight)."""
    tree = cKDTree(pos_fit)
    out = np.tile(mean, (len(q), 1)).astype(float)
    for i, nb in enumerate(tree.query_ball_point(q, 3 * h)):
        if nb:
            d2 = np.sum((pos_fit[nb] - q[i]) ** 2, 1)
            w = np.exp(-0.5 * d2 / h**2)
            out[i] = mean + (w[:, None] * (vec_fit[nb] - mean)).sum(0) / (w.sum() + k0)
    return out


def checker_colour(pos, block, origin):
    k = np.floor((pos - origin) / block).astype(int)
    return (k[:, 0] + k[:, 1]) % 2


def heldout_skill(pos, vec, block, colour, origin, rng=None, n_perm=0, bandwidths=BANDWIDTHS):
    """Fit on checker colour != colour, predict held-out colour. Bandwidth chosen by inner 2-fold block CV on fit only."""
    col = checker_colour(pos, block, origin)
    fit, ho = col != colour, col == colour
    pf, vf = pos[fit], vec[fit]
    mean = vf.mean(0)
    # inner CV inside fit: split fit blocks by a coarser parity (2*block) so no held-out nucleus is touched
    inner = (np.floor((pf - origin) / (2 * block)).astype(int).sum(1)) % 2
    cv = {}
    for h in bandwidths:
        sse = 0.0
        for g in (0, 1):
            a, b = inner != g, inner == g
            if a.sum() < 10 or b.sum() < 10:
                continue
            m0 = vf[a].mean(0)
            pr = _nw(pf[a], vf[a], pf[b], h, m0)
            sse += np.sum((vf[b] - pr) ** 2) - np.sum((vf[b] - m0) ** 2)
        cv[h] = sse
    h = min(cv, key=cv.get)
    pred = _nw(pf, vf, pos[ho], h, mean)
    sse_pred = np.sum((vec[ho] - pred) ** 2); sse0 = np.sum((vec[ho] - mean) ** 2)
    out = {"block": block, "colour": int(colour), "bandwidth": h, "n_fit": int(fit.sum()), "n_heldout": int(ho.sum()),
           "skill": float(1 - sse_pred / sse0),
           "cov_pred_obs": float(np.mean(np.sum((pred - mean) * (vec[ho] - mean), 1))),
           "pred_rms": float(np.sqrt(np.mean(np.sum((pred - mean) ** 2, 1))))}
    # window-level agreement (100 um windows of held-out nuclei)
    c, m, n = window_means(pos[ho], vec[ho] - mean, 100.0, min_n=5, origin=origin)
    if len(c) >= 4:
        pw = _nw(pf, vf, c, h, mean) - mean
        out["window_corr"] = float(np.corrcoef(np.r_[pw[:, 0], pw[:, 1]], np.r_[m[:, 0], m[:, 1]])[0, 1])
    if n_perm:
        rng = rng or np.random.default_rng(0)
        null = []
        for _ in range(n_perm):
            vp = vf[rng.permutation(len(vf))]
            pr = _nw(pf, vp, pos[ho], h, mean)
            null.append(1 - np.sum((vec[ho] - pr) ** 2) / sse0)
        out["skill_perm_p95"] = float(np.quantile(null, 0.95)); out["skill_perm_median"] = float(np.median(null))
        out["skill_perm_p_value"] = float((1 + np.sum(np.array(null) >= out["skill"])) / (1 + n_perm))
    return out


def full_diagnostic(pos, vec, origin, rng=None, n_perm=0, blocks=(100.0, 200.0)):
    out = {"n_vectors": int(len(vec)), "res_mag_median": float(np.median(np.linalg.norm(vec, axis=1))),
           "res_mag_p90": float(np.quantile(np.linalg.norm(vec, axis=1), 0.9)),
           "mean_dx": float(vec[:, 0].mean()), "mean_dy": float(vec[:, 1].mean())}
    out.update(vector_autocorr(pos, vec))
    out.update(direction_consistency(pos, vec))
    out.update(window_summary(pos, vec))
    out["coherent_rms_from_cov0_20"] = float(np.sqrt(max(out["cov_0_20"], 0.0))) if np.isfinite(out["cov_0_20"]) else np.nan
    ho = []
    for b in blocks:
        for c in (0, 1):
            ho.append(heldout_skill(pos, vec, b, c, origin, rng=rng, n_perm=n_perm))
    out["heldout"] = ho
    for b in blocks:
        s = [r["skill"] for r in ho if r["block"] == b]
        out[f"heldout_skill_b{b:g}_min"] = float(min(s)); out[f"heldout_skill_b{b:g}_mean"] = float(np.mean(s))
        cv = [r["cov_pred_obs"] for r in ho if r["block"] == b]
        out[f"heldout_coherent_rms_b{b:g}"] = float(np.sqrt(max(np.mean(cv), 0.0)))
    return out


# ----------------------------------------------------------------------------- Real-template semi-synthetic (N3)
def semisynthetic(fixed, region_poly, *, q_pair, sigma_pair, n_moving, amp_um, rng, field_fn=None):
    """Pseudo-moving from Real fixed nuclei: a fraction q_pair are 'shared' (fixed + N(0, sigma_pair) + field), the rest
    are non-shared nuclei placed 6-20 um from a random fixed nucleus (same tissue density). Known displacement field
    (amp_um RMS) is added to every moving nucleus. Returns moving positions and the true field at them."""
    n_sh = int(round(q_pair * n_moving))
    sh = rng.choice(len(fixed), n_sh, replace=False)
    mv = fixed[sh] + rng.normal(0, sigma_pair, (n_sh, 2))
    extra = []
    tree, tree_f = cKDTree(mv), cKDTree(fixed)
    while len(extra) < n_moving - n_sh:
        base = fixed[rng.integers(len(fixed))]
        a = rng.uniform(0, 2 * np.pi); r = rng.uniform(6, 20)
        c = base + r * np.array([np.cos(a), np.sin(a)])
        if tree.query(c)[0] >= 5 and tree_f.query(c)[0] >= 5:
            extra.append(c)
    mv = np.vstack([mv, np.array(extra).reshape(-1, 2)])
    if field_fn is not None and amp_um > 0:
        d = field_fn(mv)
    else:
        d = np.zeros_like(mv)
    return mv + d, d
