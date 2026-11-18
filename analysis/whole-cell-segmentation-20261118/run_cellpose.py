"""S1: Cellpose-SAM (cpsam) on the ROI crops (env: cellseg-cp; independent of cell-rg).

Inputs : cache/crops/<sample>/<roi>/{HE,IF}.png (0.5 um/px, section frame; see roi_select.py)
Outputs: cache/seg/S1_cpsam[_<variant>]/<sample>/<roi>/<key>.npz  (labels int32, um_per_px, origin_um)
         runs_S1.json  (model, versions, parameters, per-crop runtime, GPU memory, failures)

Variants (input / scale):
  he_rgb            H&E RGB as is
  if_rgb            IF display RGB as is (R=Desmoplakin, G=Dystrophin1, B=Hoechst+Desmoplakin -- display-rendered, not raw channels)
  if_memnuc         IF ch0 = G+R (membrane-associated proxy), ch1 = clip(B-R) (Hoechst estimate), ch2 = 0
  if_mem            IF ch0 = G+R only (no nuclear channel)
  he_eosin          H&E eosin OD from colour deconvolution (cytoplasm) only
  *_x2              the crop upsampled 2x (0.25 um/px) before inference, labels downsampled back (nearest)
Usage: python run_cellpose.py <variant,...> <roi-set: tuning|evaluation|all> [cellprob_threshold] [flow_threshold] [tag]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent


def prep(rgb: np.ndarray, variant: str) -> np.ndarray:
    x = rgb.astype(np.float32)
    if variant.startswith("if_mem") and not variant.startswith("if_memnuc"):     # membrane-associated proxy only
        return np.clip(x[..., 1] + x[..., 0], 0, 255)[..., None].repeat(3, -1) * np.array([1, 0, 0], np.float32)
    if variant.startswith("he_eosin"):                                            # eosin OD (colour deconvolution), cytoplasm
        from skimage.color import rgb2hed
        e = rgb2hed(rgb)[..., 1]
        e = (e - np.percentile(e, 1)) / max(np.percentile(e, 99) - np.percentile(e, 1), 1e-6)
        return np.stack([np.clip(e, 0, 1) * 255, np.zeros_like(e), np.zeros_like(e)], -1).astype(np.float32)
    if variant.startswith("if_memnuc"):
        mem = np.clip(x[..., 1] + x[..., 0], 0, 255)
        nuc = np.clip(x[..., 2] - x[..., 0], 0, 255)
        x = np.stack([mem, nuc, np.zeros_like(mem)], -1)
    return x


def main():
    variants = sys.argv[1].split(",")
    which = sys.argv[2]
    cellprob = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    flow = float(sys.argv[4]) if len(sys.argv) > 4 else 0.4
    tag = sys.argv[5] if len(sys.argv) > 5 else ""
    import torch
    import cellpose
    from cellpose import models
    gpu = torch.cuda.is_available()
    t0 = time.time()
    err = None
    try:
        model = models.CellposeModel(gpu=gpu, pretrained_model="cpsam")
    except Exception as exc:          # weights download / CUDA problems are recorded, never hidden
        err = repr(exc)
        model = None
    rois = json.loads((HERE / "rois.json").read_text())
    log = {"method": "S1_cpsam", "cellpose_version": getattr(cellpose, "version", None) or cellpose.__version__ if hasattr(cellpose, "__version__") else None,
           "torch": torch.__version__, "cuda_available": gpu, "device": torch.cuda.get_device_name(0) if gpu else "cpu",
           "pretrained_model": "cpsam", "cellprob_threshold": cellprob, "flow_threshold": flow, "model_load_s": time.time() - t0,
           "model_error": err, "runs": []}
    try:
        from importlib.metadata import version
        log["cellpose_version"] = version("cellpose")
    except Exception:
        pass
    if model is not None:
        mp = getattr(model, "pretrained_model", None)
        log["weights"] = str(mp)
        try:
            import hashlib
            log["weights_sha256"] = hashlib.sha256(Path(mp).read_bytes()).hexdigest() if mp and Path(str(mp)).is_file() else None
        except Exception:
            log["weights_sha256"] = None
    for sample, sv in rois["samples"].items():
        for roi, r in sv["rois"].items():
            if r is None or (which != "all" and r["role"] != which):
                continue
            meta = json.loads((HERE / "cache/crops" / sample / roi / "crop_meta.json").read_text())
            for variant in variants:
                key = "HE" if variant.startswith("he") else "IF"
                rgb = np.asarray(Image.open(HERE / "cache/crops" / sample / roi / f"{key}.png").convert("RGB"))
                x = prep(rgb, variant)
                scale = 2 if variant.endswith("_x2") else 1
                if scale > 1:
                    x = np.kron(x, np.ones((scale, scale, 1), np.float32))
                rec = {"sample": sample, "roi": roi, "role": r["role"], "variant": variant, "key": key, "shape": list(x.shape)}
                if model is None:
                    rec["error"] = err
                    log["runs"].append(rec)
                    continue
                if gpu:
                    torch.cuda.reset_peak_memory_stats()
                t1 = time.time()
                try:
                    masks, flows, styles = model.eval(x, channel_axis=-1, cellprob_threshold=cellprob, flow_threshold=flow, normalize=True)
                except Exception as exc:
                    rec["error"] = repr(exc)
                    log["runs"].append(rec)
                    continue
                rec["seconds"] = time.time() - t1
                rec["gpu_peak_mb"] = torch.cuda.max_memory_allocated() / 2**20 if gpu else None
                lab = np.asarray(masks).astype(np.int32)
                if scale > 1:
                    lab = lab[scale // 2::scale, scale // 2::scale]
                rec["n_labels"] = int(lab.max())
                name = f"S1_cpsam_{variant}" + (f"_{tag}" if tag else "")
                out = HERE / "cache/seg" / name / sample / roi
                out.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out / f"{key}.npz", labels=lab, um_per_px=meta[key]["um_per_px"], origin_um=np.asarray(meta[key]["origin_um"]))
                rec["out"] = str(out / f"{key}.npz")
                log["runs"].append(rec)
                print(sample, roi, variant, rec.get("n_labels"), round(rec["seconds"], 2), rec["gpu_peak_mb"], flush=True)
    p = HERE / f"runs_S1{('_' + tag) if tag else ''}_{which}.json"
    p.write_text(json.dumps(log, indent=1))
    print("model_error", err)


if __name__ == "__main__":
    main()
