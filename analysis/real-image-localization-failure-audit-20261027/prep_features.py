"""Cache structural feature maps (registration_core.features.extract, unchanged) for 031-8_6 / 031-6_2."""
import sys, time
from common import features
for sid in sys.argv[1:] or ["031-8_6", "031-6_2"]:
    for k in ("HE", "IF"):
        t = time.time(); fm = features(sid, k)
        print(sid, k, fm.shape, round(time.time() - t, 1), "s", {q: fm.quality[q] for q in ("tissue_fraction_fov", "nuclear_map_auc_own_nuclei_vs_tissue", "usable", "source_level", "source_um_per_px")}, flush=True)
