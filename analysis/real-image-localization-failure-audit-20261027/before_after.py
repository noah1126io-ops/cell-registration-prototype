"""before_after_comparison.csv: the same engine on the same inputs, image_feature tissue+nuclear (10-22) vs occ+cleft (10-27)."""
import json, numpy as np, pandas as pd
from common import store_transform
from src.registration_core import reference as R
M8, t8, _ = store_transform(); d = json.load(open("cache/ref_6_2_accepted.json")); M6, t6 = np.array(d["M"]), np.array(d["t"])
runs = [("031-8_6", "before (tissue+nuclear, 2026-10-22)", "baseline/8_6_auto_artifact_20261022.json", (M8, t8), "manual MANUAL_UNVERIFIED"),
        ("031-8_6", "after (occ+cleft)", "cache/runs/auto_031-8_6_occ+cleft/artifact.json", (M8, t8), "manual MANUAL_UNVERIFIED"),
        ("031-8_6", "assisted: local search from manual reference (occ+cleft)", "cache/runs/init_031-8_6_ref/artifact.json", (M8, t8), "manual MANUAL_UNVERIFIED"),
        ("031-6_2", "before (tissue+nuclear, 2026-10-22)", "baseline/6_2_auto_artifact_20261022.json", (M6, t6), "accepted 10-22"),
        ("031-6_2", "after (occ+cleft)", "cache/runs/auto_031-6_2_occ+cleft/artifact.json", (M6, t6), "accepted 10-22")]
rows = []
for sample, tag, path, (M, t), refkind in runs:
    a = json.load(open(path)); dc = a["decision"]
    for c in a["candidates"]:
        s, Ts, Tf, v = c["search"], c["T_search"], c["T_final"], c["validation"]["fixed_frame"]
        cs, cf = R.compare(Ts["M"], Ts["t"], M, t, (1026, 1026)), R.compare(Tf["M"], Tf["t"], M, t, (1026, 1026))
        ps = dc["per_source"].get(c["source"], {})
        rows.append({"sample": sample, "run": tag, "reference": refkind, "decision": dc["status"], "failure_stage": dc["failure_stage"],
                     "source": c["source"], "search_verdict": ps.get("verdict"), "search_margin": ps.get("margin"), "candidate": c["id"], "rank": c["rank"],
                     "theta_deg": round(s["theta_deg"], 2), "scale": round(s["scale"], 4), "score_z": round(s["score"], 2),
                     "C1_dist_to_ref_um": round(cs["probe_median_um"], 1), "C2_dist_to_ref_um": round(cf["probe_median_um"], 1),
                     "C1_rot_diff_deg": round(cs["d_rotation_deg"], 1), "C1_scale_ratio": round(cs["scale_ratio"], 3),
                     "icp_shift_um": (c["refine"].get("pose_change") or {}).get("median_shift_um"), "flags": "|".join(c["flags"]),
                     "le_3um": round(v["le_3um"], 4), "null_le_3um": round(v["null_le_3um"], 4), "nn_median_um": round(v["nn_median_um"], 3),
                     "geometric_pass": c["geometric_pass"]})
df = pd.DataFrame(rows); df.to_csv("before_after_comparison.csv", index=False)
pd.set_option("display.width", 260); pd.set_option("display.max_columns", 30)
print(df[df["rank"] <= 1][["sample", "run", "decision", "failure_stage", "candidate", "search_verdict", "search_margin", "theta_deg", "scale", "score_z", "C1_dist_to_ref_um", "C2_dist_to_ref_um", "flags", "le_3um", "null_le_3um", "geometric_pass"]].to_string(index=False))
for p in ("baseline/6_2_auto_artifact_20261022.json", "cache/runs/auto_031-6_2_occ+cleft/artifact.json"):
    T = json.load(open(p))["transform"]["forward"]; print(p, np.round(T["M"], 8).tolist(), np.round(T["t"], 6).tolist())
