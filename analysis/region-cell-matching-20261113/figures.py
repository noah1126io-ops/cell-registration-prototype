"""Figures from the collected CSVs (no recomputation)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
ORDER = ["A", "B", "B'", "C", "D", "E"]
COL = {"A": "#7f7f7f", "B": "#bcbd22", "B'": "#17becf", "C": "#1f77b4", "D": "#ff7f0e", "E": "#2ca02c"}


def fig_candidates(cand):
    sub = cand[(cand["status"] == "OK") & ~cand["case"].str.startswith("K30_")].copy()   # K30 re-searches the same cases
    groups = [("SYN-C1", "final", "合成（C1 上位 k）"), ("REAL-D", "final", "実データ由来（T_final）"), ("REAL-D", "search", "実データ由来 8_6（T_search）")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, (ds, pose, title) in zip(axes, groups):
        s = sub[(sub["dataset"] == ds) & (sub["pose"] == pose)]
        for k, (lab, col) in enumerate(((False, "#d62728"), (True, "#2ca02c"))):
            v = s.loc[s["correct_region"] == lab, "excess"].to_numpy(float)
            x = k + np.random.default_rng(0).uniform(-0.15, 0.15, len(v))
            ax.scatter(x, v, s=12, c=col, alpha=0.7)
        ax.axhline(0.10, ls="--", c="k", lw=0.8)
        ax.set_xticks([0, 1], ["誤った領域", "正しい領域"]); ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("B2採択率 − decoy採択率（主指標）")
    fig.suptitle("候補ごとのGraph/UOT/B2スコア（破線＝支持の閾値0.10）", fontsize=11)
    fig.tight_layout(); fig.savefig(FIG / "fig1_candidate_scores.png", dpi=130); plt.close(fig)


def fig_known(defo):
    a = defo[defo["experiment"] == "A"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, ds in zip(axes[:2], ("SYN-FIELD", "SYN-RL")):
        s = a[a["dataset"] == ds]
        data = [s.loc[s["method"] == m, "epe_um_median"].to_numpy(float) for m in ORDER]
        bp = ax.boxplot(data, tick_labels=ORDER, patch_artist=True, showfliers=False)
        for p, m in zip(bp["boxes"], ORDER):
            p.set_facecolor(COL[m]); p.set_alpha(0.6)
        ax.set_title(f"{ds}: 変形場の誤差（EPE中央値, µm、適用後）", fontsize=10); ax.set_ylabel("µm")
    s = a[a["dataset"] == "SYN-FIELD"]
    data = [s.loc[s["method"] == m, "independent_landmark_tre_um_median"].to_numpy(float) for m in ORDER]
    bp = axes[2].boxplot(data, tick_labels=ORDER, patch_artist=True, showfliers=False)
    for p, m in zip(bp["boxes"], ORDER):
        p.set_facecolor(COL[m]); p.set_alpha(0.6)
    axes[2].set_title("SYN-FIELD: 独立ランドマークTRE中央値（µm）", fontsize=10)
    fig.suptitle("実験A（Global変換は既知）: A Affine / B 既存TPS / B′ MNN+RBF / C B2+RBF / D UOT conservative / E B2×信頼度+RBF", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "fig2_known_global_deformation.png", dpi=130); plt.close(fig)


def fig_real_deformation(defo):
    r = defo[defo["experiment"].isin(["REAL-D", "REAL-S"]) & defo["case"].isin(["P6_2_orig", "P8_6_orig", "serial_section_pair_20260904"])]
    r = r[r["pick"].isin(["c1_top1", "saved_affine"])]
    if r.empty:
        return
    rows = []
    for (case, pose), s in r.groupby(["case", r["pose"].fillna("saved")]):
        rows.append((f"{case} ({pose})", {m: (bool(x.applied), x.anchors_n) for m, x in s.set_index("method").iterrows()}))
    fig, ax = plt.subplots(figsize=(9, 0.6 + 0.5 * len(rows)))
    for i, (lab, d) in enumerate(rows):
        for j, m in enumerate(ORDER):
            if m in d:
                ap, an = d[m]
                ax.add_patch(plt.Rectangle((j, i), 0.95, 0.9, color="#d62728" if ap and m != "A" else "#dddddd"))
                ax.text(j + 0.47, i + 0.45, f"{'適用' if ap else '不適用'}\n{'' if an is None or an != an else int(an)}", ha="center",
                        va="center", fontsize=7)
    ax.set_xlim(0, len(ORDER)); ax.set_ylim(0, len(rows)); ax.set_xticks(np.arange(len(ORDER)) + 0.47, ORDER)
    ax.set_yticks(np.arange(len(rows)) + 0.45, [l for l, _ in rows], fontsize=8); ax.invert_yaxis()
    ax.set_title("実データで局所変形が適用されたか（赤＝適用、数字＝アンカー数）。8_6は隣接切片で核の一対一対応はない", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "fig3_real_deformation_applied.png", dpi=130); plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "IPAexGothic", "DejaVu Sans"]
    cand = pd.read_csv(HERE / "candidate_comparison.csv")
    defo = pd.read_csv(HERE / "deformation_comparison.csv")
    fig_candidates(cand)
    fig_known(defo)
    fig_real_deformation(defo)


if __name__ == "__main__":
    main()
