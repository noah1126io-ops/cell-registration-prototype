"""candidate_score_distribution.png: every coarse peak, score vs distance to the reference (8_6 manual / 6_2 accepted)."""
import json, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import FIG, probe, probe_dist, side, store_transform
from src.registration_core.search import pose_matrix
M8, t8, _ = store_transform(); d = json.load(open("cache/ref_6_2_accepted.json")); M6, t6 = np.array(d["M"]), np.array(d["t"])
combos = ["base(tissue+nuclear)", "points", "occ+cleft"]
fig, ax = plt.subplots(2, 3, figsize=(18, 9))
for i, (sid, csv, (M, t)) in enumerate((("031-8_6", "landscape2_8_6.csv", (M8, t8)), ("031-6_2", "landscape2_6_2.csv", (M6, t6)))):
    df = pd.read_csv(csv); df = df[(df.kind == "peak") & df.scale.round(2).isin([0.95, 1.0, 1.05])]
    pr = probe(side(sid, "IF", with_image=False).extent)
    for j, c in enumerate(combos):
        g = df[df.combo == c]
        dist = np.array([probe_dist(pose_matrix("proper", a, s), np.array([x, y]), M, t, pr) for a, s, x, y in zip(g.theta, g.scale, g.tx, g.ty)])
        dth = np.abs(((g.theta.values - np.degrees(np.arctan2(M[1, 0], M[0, 0]))) + 180) % 360 - 180)
        sc = ax[i, j].scatter(dist, g.score, c=dth, cmap="viridis", s=6, vmin=0, vmax=180)
        ax[i, j].axvline(40, color="r", lw=0.8, ls="--"); ax[i, j].set_xscale("symlog", linthresh=50)
        ax[i, j].set_title(f"{sid}: {c}  (top-1 dist {dist[np.argmax(g.score.values)]:.0f} um)")
        ax[i, j].set_xlabel("distance to reference (um, probe median)"); ax[i, j].set_ylabel("combined z score")
    fig.colorbar(sc, ax=ax[i, :].tolist(), label="|rotation - reference| (deg)")
fig.suptitle("Every coarse peak (engine scales 0.95-1.05, 4 deg): score vs distance to reference. Red line = 40 um (same region)")
fig.savefig(FIG / "candidate_score_distribution.png", dpi=80, bbox_inches="tight")
