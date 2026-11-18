"""S2: Mesmer (DeepCell) whole-cell segmentation of the IF ROI crops (env: cellseg-mesmer; independent of cell-rg).

Mesmer needs two channels: nuclear and membrane/cytoplasm.  The IF image is display-rendered RGB (uint8), so both
inputs are PROXIES built from it, not raw fluorescence channels:
  nuclear  = clip(B - R)   (Hoechst estimate; B = Hoechst + Desmoplakin, R = Desmoplakin)
  membrane = G + R         (Dystrophin1 + Desmoplakin, membrane-associated markers in cardiomyocytes)
H&E is not a valid Mesmer input (brightfield) and is not run.
Weights are downloaded by deepcell from users.deepcell.org; this needs DEEPCELL_ACCESS_TOKEN.  A failure is recorded.
Usage: python run_mesmer.py <tuning|evaluation|all>
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    log = {"method": "S2_mesmer", "inputs": {"nuclear": "clip(B-R) of display RGB (proxy)", "membrane": "G+R of display RGB (proxy)"},
           "image_mpp": 0.5, "compartment": "whole-cell", "token_present": bool(os.environ.get("DEEPCELL_ACCESS_TOKEN")), "runs": []}
    t0 = time.time()
    try:
        import deepcell
        import tensorflow as tf
        log["deepcell"] = deepcell.__version__
        log["tensorflow"] = tf.__version__
        log["gpu"] = [d.name for d in tf.config.list_physical_devices("GPU")]
        from deepcell.applications import Mesmer
        app = Mesmer()
        log["model_load_s"] = time.time() - t0
    except Exception as exc:
        log["model_error"] = repr(exc)
        log["traceback"] = traceback.format_exc()[-3000:]
        (HERE / f"runs_S2_{which}.json").write_text(json.dumps(log, indent=1))
        print("MODEL ERROR", repr(exc))
        return
    rois = json.loads((HERE / "rois.json").read_text())
    for sample, sv in rois["samples"].items():
        for roi, r in sv["rois"].items():
            if r is None or (which != "all" and r["role"] != which):
                continue
            meta = json.loads((HERE / "cache/crops" / sample / roi / "crop_meta.json").read_text())
            rgb = np.asarray(Image.open(HERE / "cache/crops" / sample / roi / "IF.png").convert("RGB")).astype(np.float32)
            x = np.stack([np.clip(rgb[..., 2] - rgb[..., 0], 0, 255), np.clip(rgb[..., 1] + rgb[..., 0], 0, 255)], -1)[None]
            t1 = time.time()
            rec = {"sample": sample, "roi": roi, "role": r["role"], "key": "IF"}
            try:
                lab = app.predict(x, image_mpp=0.5, compartment="whole-cell")[0, ..., 0].astype(np.int32)
            except Exception as exc:
                rec["error"] = repr(exc)
                log["runs"].append(rec)
                continue
            rec["seconds"] = time.time() - t1
            rec["n_labels"] = int(lab.max())
            out = HERE / "cache/seg/S2_mesmer" / sample / roi
            out.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out / "IF.npz", labels=lab, um_per_px=meta["IF"]["um_per_px"], origin_um=np.asarray(meta["IF"]["origin_um"]))
            log["runs"].append(rec)
            print(sample, roi, rec["n_labels"], round(rec["seconds"], 2), flush=True)
    (HERE / f"runs_S2_{which}.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
