"""
EXPERIMENTAL: visibility (Vis) probe — NOT in the XWOD paper.

This script ships an extra visibility-bucketing track that was prototyped
during XWOD development but did not make it into the final paper (paper
§4.3 defines only XWOD-LLM-WC). We keep it here for researchers who want
to explore visibility-derived signals without re-implementing the
plumbing.

How visibility is derived
-------------------------
For every image in a split, we sum the normalized YOLO bounding-box area
(``sum(w*h)`` over the YOLO label rows). Images are then bucketed at the
global terciles of that score:

  L0 = bottom third  (few or very small visible objects -> poor visibility)
  L1 = middle third
  L2 = top third     (most of the scene clearly visible)

Buckets are computed per split, so train/valid/test do **not** share
thresholds. This is a *derived proxy*, not a human-annotated label —
treat any number you produce here as a secondary result, not a primary
benchmark figure.

The model is asked to emit JSON of the form::

    {"weather": "rain", "visibility": "L1"}

Visibility predictions are scored against the bucket of the image's
``image_id``. We report accuracy, Spearman rho on the ordinal, and a 3x3
confusion matrix.

Usage
-----
    python experimental/eval_llm_vis.py \\
        --data ../dataset --split test --balanced-n 100 \\
        --backend google --model gemini-3.1-pro-preview \\
        --out experimental/results/vis_gemini.csv

The four backends are inherited from ``eval_llm_wc.py`` so behavior is
identical apart from the prompt and scorer.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

# Reuse plumbing from the main WC script so we stay aligned.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval_llm_wc import (  # noqa: E402
    BACKENDS,
    load_balanced_probe,
    load_dotenv_if_present,
    run_task,
)


VIS_LEVELS = ["L0", "L1", "L2"]  # L0 = worst visibility, L2 = best

VIS_PROMPT = (
    "You are an expert in autonomous-driving scene understanding. For the image, "
    "output a single JSON object with keys:\n"
    "  weather: one of {rain, snow, fog, haze/sand/dust, flooding, tornado, wildfire}\n"
    "  visibility: one of {L0, L1, L2} where\n"
    "    L0 = worst (severe occlusion; few or very small road users visible)\n"
    "    L1 = moderate (some road users visible but partially obscured)\n"
    "    L2 = best  (most of the scene clearly visible, road users well-defined)\n"
    "Reply ONLY with the JSON, no prose."
)


def build_visibility_buckets(root: Path, split: str) -> dict[str, str]:
    """Derive per-image visibility bucket {L0, L1, L2} from YOLO labels."""
    label_dir = root / split / "labels"
    scores: dict[str, float] = {}
    for label_file in sorted(label_dir.iterdir()):
        if label_file.suffix != ".txt":
            continue
        total_area = 0.0
        try:
            with open(label_file) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    w, h = float(parts[3]), float(parts[4])
                    total_area += w * h
        except OSError:
            total_area = 0.0
        scores[label_file.stem] = total_area

    if not scores:
        return {}

    sorted_scores = sorted(scores.values())
    n = len(sorted_scores)
    t1 = sorted_scores[n // 3]
    t2 = sorted_scores[(2 * n) // 3]

    out: dict[str, str] = {}
    for image_id, area in scores.items():
        if area < t1:
            out[image_id] = "L0"
        elif area < t2:
            out[image_id] = "L1"
        else:
            out[image_id] = "L2"
    return out


def parse_json_blob(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def score_vis(rows: list[dict], visibility_map: dict[str, str]) -> dict:
    """Score the experimental Vis task against detection-derived buckets."""
    from sklearn.metrics import accuracy_score, confusion_matrix
    from scipy.stats import spearmanr
    vis_order = {v: i for i, v in enumerate(VIS_LEVELS)}
    y, yhat = [], []
    n_unparsed = 0
    for r in rows:
        gt = visibility_map.get(r["image_id"])
        if gt is None:
            continue
        j = parse_json_blob(r.get("pred", "")) or {}
        pv = str(j.get("visibility") or "").strip().upper()
        if pv not in vis_order:
            n_unparsed += 1
            continue
        y.append(vis_order[gt])
        yhat.append(vis_order[pv])
    out: dict = {"n_scored": len(y), "n_unparsed": n_unparsed}
    if y:
        rho, _ = spearmanr(y, yhat)
        out["visibility_accuracy"] = float(accuracy_score(y, yhat))
        out["visibility_spearman"] = float(rho) if rho == rho else None  # NaN guard
        out["confusion_matrix"] = confusion_matrix(
            y, yhat, labels=list(range(len(VIS_LEVELS)))).tolist()
        out["labels"] = VIS_LEVELS
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--split", default="test", choices=["train", "valid", "test"])
    p.add_argument("--balanced-n", type=int, default=100,
                   help="images per weather in the probe (7*n total)")
    p.add_argument("--backend", choices=list(BACKENDS), required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--env-file", default=None)
    args = p.parse_args()

    if args.out is None:
        safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", args.model).strip("_")
        ts = time.strftime("%Y%m%d-%H%M%S")
        args.out = Path("experimental/results") / f"vis_{args.backend}_{safe_model}_{ts}.csv"
        print(f"Auto-generated output path: {args.out}", file=sys.stderr)

    load_dotenv_if_present(args.env_file)

    probe = load_balanced_probe(args.data, args.split, args.balanced_n)
    print(f"Probe size: {len(probe)} images", file=sys.stderr)

    visibility_map = build_visibility_buckets(args.data, args.split)
    coverage = sum(1 for r in probe if r["image_id"] in visibility_map)
    print(f"Visibility map: {len(visibility_map)} images in split, "
          f"{coverage}/{len(probe)} probe rows have a bucket",
          file=sys.stderr)

    call = BACKENDS[args.backend](args.model)
    rows = run_task(probe, call, VIS_PROMPT, "vis")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df["model"] = args.model
    df["visibility_gt"] = df["image_id"].map(visibility_map)
    df.to_csv(args.out, index=False)

    summary = {
        "model": args.model, "split": args.split, "n": len(probe),
        "vis": score_vis(rows, visibility_map),
        "note": "EXPERIMENTAL — not part of the XWOD paper.",
    }
    (args.out.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
