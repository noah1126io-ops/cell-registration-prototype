"""Where does the reference rank?  Distinct peaks (engine rule: probe median < 40 um = same) over a pose subset."""
import json, sys
import numpy as np, pandas as pd
from common import probe, probe_dist, store_transform, side
from src.registration_core.search import pose_matrix

def analyse(csv, ref, mext, scales=None, out=None, label=""):
    df = pd.read_csv(csv)
    if scales is not None:
        df = df[df.scale.round(3).isin([round(s, 3) for s in scales])]
    Mr, tr = ref; pr = probe(mext)
    res = []
    for combo, g in df.groupby("combo", sort=False):
        pk = g[g.kind == "peak"].sort_values("score", ascending=False)
        A = [pose_matrix("proper", th, s) for th, s in zip(pk.theta, pk.scale)]
        T = list(zip(pk.tx, pk.ty))
        dref = np.array([probe_dist(a, np.array(t), Mr, tr, pr) for a, t in zip(A, T)])
        kept = []
        for i in range(len(pk)):
            if all(probe_dist(A[i], np.array(T[i]), A[j], np.array(T[j]), pr) >= 40 for j in kept):
                kept.append(i)
            if len(kept) >= 400: break
        sc = pk.score.values
        near = [i for i in kept if dref[i] < 100]
        flip = [i for i in kept if dref[i] >= 100 and abs(((pk.theta.values[i] - params_theta + 180) % 360) - 180) > 150]
        r_near = kept.index(near[0]) + 1 if near else None
        rr = g[g.kind == "ref"]
        best_ref = rr.score.max()
        top = kept[0]
        res.append({"set": label, "combo": combo, "top1_score": round(sc[top], 2), "top1_theta": pk.theta.values[top], "top1_scale": pk.scale.values[top],
                    "top1_dist_to_ref_um": round(dref[top]), "rank_of_ref_region": r_near,
                    "ref_region_score": round(sc[near[0]], 2) if near else None, "best_score_in_ref_window": round(best_ref, 2),
                    "margin_top1_vs_2": round((sc[kept[0]] - sc[kept[1]]) / abs(sc[kept[0]]), 3),
                    "best_flip180_rank": kept.index(flip[0]) + 1 if flip else None})
    return pd.DataFrame(res)

if __name__ == "__main__":
    M, t, _ = store_transform()
    params_theta = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    mext = side("031-8_6", "IF", with_image=False).extent
    a = analyse("landscape_8_6.csv", (M, t), mext, None, label="extended 0.90-1.20")
    b = analyse("landscape_8_6.csv", (M, t), mext, [0.95, 1.0, 1.05], label="engine scales 0.95-1.05")
    out = pd.concat([b, a]); pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
    print(out.to_string(index=False)); out.to_csv("candidate_rank_reference.csv", index=False)


def run_6_2():
    global params_theta
    d = json.load(open("cache/ref_6_2_accepted.json")); M, t = np.array(d["M"]), np.array(d["t"])
    params_theta = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    mext = side("031-6_2", "IF", with_image=False).extent
    a = analyse("landscape_6_2.csv", (M, t), mext, [0.95, 1.0, 1.05], label="6_2 engine scales")
    b = analyse("landscape_6_2.csv", (M, t), mext, None, label="6_2 extended")
    return pd.concat([a, b])
