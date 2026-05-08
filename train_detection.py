"""
XWOD training sweep: reproduces the YOLOv8 / YOLO11 / YOLO26 baselines.

Usage
-----
    conda activate torch
    pip install -U ultralytics

    python train_detection.py \\
        --data ../dataset/data.yaml \\
        --families 8 11 26 --scales n s m l x \\
        --epochs 100 --imgsz 640 --device 0

Each run writes to runs/<family>_<scale>/ and appends its final metrics to
results/xwod_training_sweep.csv.

To reproduce XWOD-Gen (leave-one-weather-out):

    python train_detection.py --leave-one-out --model yolo11m --epochs 100
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
from pathlib import Path

import yaml

WEATHER_PREFIXES = {
    "rain": "heavy", "snow": "snow", "fog": "fog", "dust": "dust",
    "flooding": "flooding", "tornado": "tornado", "wildfire": "wildfire",
}
CLASSES = ["person", "car", "truck", "motorcycle", "bus", "bike"]


def fam_to_weights(family: int, scale: str) -> str:
    return {
        8:  f"yolov8{scale}.pt",
        11: f"yolo11{scale}.pt",
        26: f"yolo26{scale}.pt",
    }[family]


def run_one(weights: str, data_yaml: Path, epochs: int, imgsz: int, device: str, project: str, name: str):
    from ultralytics import YOLO
    model = YOLO(weights)
    results = model.train(
        data=str(data_yaml), epochs=epochs, imgsz=imgsz, device=device,
        project=project, name=name, exist_ok=True, verbose=False, plots=False,
    )
    m = model.val(data=str(data_yaml), imgsz=imgsz, device=device, verbose=False, plots=False)
    return {
        "mAP50": float(m.box.map50), "mAP50-95": float(m.box.map),
        "precision": float(m.box.mp), "recall": float(m.box.mr),
    }


def sweep(data: Path, families: list[int], scales: list[str], epochs: int, imgsz: int, device: str, out_csv: Path):
    rows = []
    for fam in families:
        for sc in scales:
            w = fam_to_weights(fam, sc)
            name = f"yolo{fam}{sc}"
            print(f"[train] {name}")
            try:
                res = run_one(w, data, epochs, imgsz, device, "runs/xwod", name)
                rows.append({"model": name, **res})
            except Exception as e:
                rows.append({"model": name, "error": str(e)})
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(json.dumps(rows, indent=2))


def make_lowo_yaml(root: Path, held_out: str, out: Path) -> Path:
    """Build a data.yaml that excludes the held-out weather from train+val
    and restricts test to the held-out weather only. Uses symlink farms so
    we don't copy images."""
    held_prefix = WEATHER_PREFIXES[held_out]
    def link_split(split: str, include_prefixes: set[str]):
        src_img = root / split / "images"
        src_lab = root / split / "labels"
        dst_img = out / split / "images"
        dst_lab = out / split / "labels"
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lab.mkdir(parents=True, exist_ok=True)
        for f in src_img.iterdir():
            if f.name.split("_")[0] in include_prefixes:
                (dst_img / f.name).symlink_to(f)
                lab = src_lab / (f.stem + ".txt")
                if lab.exists():
                    (dst_lab / lab.name).symlink_to(lab)

    non_held = set(WEATHER_PREFIXES.values()) - {held_prefix}
    link_split("train", non_held)
    link_split("valid", non_held)
    link_split("test", {held_prefix})  # only held-out appears in test
    y = out / "data.yaml"
    y.write_text(yaml.safe_dump({
        "path": str(out), "train": "train/images", "val": "valid/images",
        "test": "test/images", "nc": len(CLASSES), "names": CLASSES,
    }))
    return y


def lowo(data_root: Path, weights: str, epochs: int, imgsz: int, device: str, out_csv: Path):
    rows = []
    for weather in WEATHER_PREFIXES:
        print(f"[LOWO] holding out: {weather}")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "xwod_lowo"
            root.mkdir()
            y = make_lowo_yaml(data_root, weather, root)
            res = run_one(weights, y, epochs, imgsz, device, "runs/xwod_lowo", weather)
            rows.append({"held_out": weather, **res})
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    # aggregate
    ms = [r["mAP50"] for r in rows if "mAP50" in r]
    print(f"\nXWOD-Gen mAP50 mean={sum(ms)/len(ms):.4f}  min={min(ms):.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True,
                   help="root data.yaml for normal training; dataset root for --leave-one-out")
    p.add_argument("--families", nargs="+", type=int, default=[8, 11, 26])
    p.add_argument("--scales", nargs="+", default=["n", "s", "m", "l", "x"])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--out", type=Path, default=Path("results/xwod_training_sweep.csv"))
    p.add_argument("--leave-one-out", action="store_true")
    p.add_argument("--model", default="yolo11m.pt", help="weights for --leave-one-out")
    args = p.parse_args()

    if args.leave_one_out:
        lowo(args.data.parent if args.data.is_file() else args.data,
             args.model, args.epochs, args.imgsz, args.device,
             Path("results/xwod_gen_lowo.csv"))
    else:
        sweep(args.data, args.families, args.scales, args.epochs, args.imgsz, args.device, args.out)


if __name__ == "__main__":
    main()
