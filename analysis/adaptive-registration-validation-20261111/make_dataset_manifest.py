"""dataset_manifest.csv: what the server holds (real pairs) + the derived stress cases of this study.
Facts come from the Explorer manifest / file system; inferences are in separate columns and marked as such."""
import csv, json
from common import HERE, REPO, load_cases
V = json.loads((REPO / "analysis/e0x-difficulty-and-web-integration-20261020/viewer_sample_manifest.json").read_text())
S = {s["sample_id"]: s for s in V["samples"]}
def fov(m):
    if m.get("pixel_um") and m.get("shape_px"):
        h, w = m["shape_px"]; return f"{w*m['pixel_um']:.0f}x{h*m['pixel_um']:.0f} um"
    return "UNKNOWN"
real = {
 "031-8_6": dict(he_sec="031-7 (file name 'HE_reference_031-7_XY06'; provider: 031-8 has no HE)", if_sec="031-8 (sample id)",
     rel="DIFFERENT", rel_src="provider note + file name", dist="UNKNOWN (adjacent by numbering; spacing not recorded)",
     tissue="UNKNOWN in metadata (Dystrophin1/Desmoplakin stains suggest striated/cardiac muscle; not confirmed)",
     prov="MEDIUM (provider note; section spacing unknown)", dev="YES: occ+cleft (10-27) and quadrant rule (11-10) were developed on this pair; landmarks drawn after seeing the auto result"),
 "031-6_2": dict(he_sec="031-6 (file name 'HE_031-6_XY02')", if_sec="031-6? ('031-6_2' parsed as section 031-6 by the engine; meaning of '_2' not documented)",
     rel="DIFFERENT (user correction 2026-10-20) - CONFLICT", rel_src="manifest user_correction_2026-10-20 says DIFFERENT; 10-19 report said 'same section'; engine 1.2 file-name rule infers SAME; nuclear <=3um proximity 0.77 (observation, not provenance)",
     dist="UNKNOWN", tissue="UNKNOWN in metadata (same stains as 8_6)", prov="LOW (conflicting records; needs the provider/user to confirm)",
     dev="YES: used to develop/validate the 10-19 nuclear rule and the 10-22..11-10 engine"),
 "serial_section_pair_20260904": dict(he_sec="UNKNOWN", if_sec="UNKNOWN", rel="UNKNOWN", rel_src="manifest same_section=null", dist="UNKNOWN",
     tissue="UNKNOWN", prov="LOW", dev="YES: used in 09-04..10-15 point-registration studies"),
}
rows = []
for sid, r in real.items():
    s = S[sid]; he, iff = s["modalities"]["HE"], s["modalities"]["IF"]
    rows.append({"row_type": "REAL_PAIR", "sample_id": sid, "he_file": he.get("image_path") or "NONE (points only)", "if_file": iff.get("image_path") or "NONE (points only)",
        "nuclei_paths": f"HE: {he['nuclei'].get('path')} | IF: {iff['nuclei'].get('path')}",
        "he_section_id": r["he_sec"], "if_section_id": r["if_sec"], "section_relation": r["rel"], "section_relation_source": r["rel_src"],
        "section_distance": r["dist"], "he_pixel_um": he.get("pixel_um") or "UNKNOWN", "if_pixel_um": iff.get("pixel_um") or "UNKNOWN",
        "he_fov": fov(he), "if_fov": fov(iff), "modality_channels": f"HE: {he.get('channels') or he.get('kind')} | IF: {iff.get('channels') or iff.get('kind')}",
        "tissue_type": r["tissue"], "image_available": "both" if he.get("image_path") and iff.get("image_path") else ("HE only (runs/.../he_image.npy, 1 um/px crop)" if sid.startswith("serial") else "none"),
        "nuclear_points_available": "both", "provenance_confidence": r["prov"], "used_in_past_development": r["dev"],
        "independent_validation_eligible": "NO", "note": "; ".join(s.get("notes") or [])})
for c in load_cases():
    rows.append({"row_type": "DERIVED_CASE (this study)", "sample_id": c["case_id"], "he_file": f"from {c['he_sample']} ops={c['he_ops']}", "if_file": f"from {c['if_sample']} ops={c['if_ops']}",
        "nuclei_paths": "same as source sample (filtered/transformed with the ops)", "he_section_id": "as source", "if_section_id": "as source",
        "section_relation": "DIFFERENT" if c["he_sample"] != c["if_sample"] else real[c["he_sample"]]["rel"], "section_relation_source": "as source sample",
        "section_distance": "UNKNOWN", "he_pixel_um": "as source", "if_pixel_um": "as source", "he_fov": "see profiles.csv", "if_fov": "see profiles.csv",
        "modality_channels": "as source", "tissue_type": "as source", "image_available": "both (derived maps)", "nuclear_points_available": "both",
        "provenance_confidence": "as source", "used_in_past_development": "YES (derived from development data)",
        "independent_validation_eligible": "NO - stress test only", "note": f"expected={c['expected']}; {c['kind']}"})
with open(HERE / "dataset_manifest.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(len(rows), "rows")
