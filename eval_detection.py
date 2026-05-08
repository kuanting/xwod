"""
XWOD detection benchmark runner.

Runs per-weather detection evaluation for YOLOv8 / YOLO11 / YOLO26 using
Ultralytics + PyTorch. Produces a CSV with mAP50, mAP50-95, precision, recall
per (model, scale, weather) cell, plus the aggregate numbers reported in the
paper.

Usage
-----
    conda activate torch
    pip install -U ultralytics pandas  # first time only
    python eval_detection.py \\
        --data ../dataset/data.yaml \\
        --weights yolov8m.pt yolo11m.pt yolo26m.pt \\
        --out results/xwod_det.csv

Notes
-----
* Per-weather evaluation is implemented by building temporary YOLO-format
  data.yaml files that restrict the `val` split to images whose filename
  starts with a given weather prefix (``rain_`` / ``snow_`` / ``fog_`` /
  ``dust_`` / ``flooding_`` / ``tornado_`` / ``wildfire_``). ``heavy_`` in
  the raw files maps to rain.
* No re-training is performed here; this only evaluates pretrained / shipped
  weights. To reproduce the training sweep from the paper, use
  ``train_detection.py`` (also in this directory).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import yaml

WEATHER_PREFIXES = {
    "rain": "heavy",         # raw-file prefix
    "snow": "snow",
    "fog": "fog",
    "dust": "dust",
    "flooding": "flooding",
    "tornado": "tornado",
    "wildfire": "wildfire",
}

CLASSES = ["person", "car", "truck", "motorcycle", "bus", "bike"]


def build_weather_split(root: Path, split: str, prefix: str, out_dir: Path) -> Path:
    """Create a symlink farm of images with the given filename prefix."""
    src_img = root / split / "images"
    src_lab = root / split / "labels"
    out_img = out_dir / split / "images"
    out_lab = out_dir / split / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lab.mkdir(parents=True, exist_ok=True)
    for f in src_img.iterdir():
        if f.name.startswith(f"{prefix}_"):
            (out_img / f.name).symlink_to(f)
            lab = src_lab / (f.stem + ".txt")
            if lab.exists():
                (out_lab / lab.name).symlink_to(lab)
    return out_dir


def make_yaml(root: Path, out: Path) -> Path:
    cfg = {
        "path": str(root),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(CLASSES),
        "names": CLASSES,
    }
    out.write_text(yaml.safe_dump(cfg))
    return out


def run_eval(weights: str, data_yaml: Path, imgsz: int = 640, device: str = "0"):
    from ultralytics import YOLO  # lazy import, keeps --help fast

    model = YOLO(weights)
    metrics = model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        device=device,
        verbose=False,
        plots=False,
    )
    return {
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }


def evaluate_all(data_root: Path, weights_list: list[str], out_csv: Path, imgsz: int, device: str):
    rows = []
    for w_path in weights_list:
        name = Path(w_path).stem  # e.g. yolo11m
        # Aggregate (whole validation set)
        agg_yaml = data_root / "data.yaml"
        if not agg_yaml.exists():
            agg_yaml = make_yaml(data_root, data_root / "data.yaml")
        rows.append({"model": name, "weather": "ALL", **run_eval(w_path, agg_yaml, imgsz, device)})

        # Per-weather
        for canonical, prefix in WEATHER_PREFIXES.items():
            with tempfile.TemporaryDirectory() as td:
                root = Path(td) / "xwod"
                root.mkdir()
                build_weather_split(data_root, "valid", prefix, root)
                build_weather_split(data_root, "test", prefix, root)
                # train split not needed for val-only eval, but ultralytics requires path
                (root / "train" / "images").mkdir(parents=True, exist_ok=True)
                (root / "train" / "labels").mkdir(parents=True, exist_ok=True)
                y = make_yaml(root, root / "data.yaml")
                rows.append({"model": name, "weather": canonical, **run_eval(w_path, y, imgsz, device)})

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved: {out_csv}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True, help="path to dataset root (containing train/ valid/ test/)")
    p.add_argument("--weights", nargs="+", required=True, help="list of Ultralytics .pt weights")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--out", type=Path, default=Path("results/xwod_det.csv"))
    args = p.parse_args()
    evaluate_all(args.data, args.weights, args.out, args.imgsz, args.device)


if __name__ == "__main__":
    main()
