#!/usr/bin/env python3
"""Compare and visualize XWOD-LLM-WC results from `eval_llm_wc.py`.

Each evaluation run produces a row-level CSV and a `.summary.json`. This
script ingests a set of those CSVs (or auto-discovers them in `results/`),
recomputes per-class metrics from the raw rows, and writes:

  * `leaderboard.csv`        — one row per run with headline metrics
  * `accuracy_comparison.png`— bar chart of WC accuracy + macro F1
  * `per_class_f1.png`       — grouped bars of WC per-class F1 across runs
  * `confusion_matrices.png` — one normalized heatmap per run

Usage
-----
  # all CSVs in results/
  python compare_results.py

  # specific runs
  python compare_results.py results/openai_gpt-5_*.csv \\
                            results/anthropic_claude-opus-4-7_*.csv

  # custom output dir
  python compare_results.py --out-dir results/comparison/v2

This script is **WC-only**. The (experimental) Vis probe in
`experimental/eval_llm_vis.py` writes its own CSVs that this script
ignores.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Re-use canonical label sets from the evaluator so we stay aligned with
# whatever the runs were scored against.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_llm_wc import WEATHERS, _fuzzy_weather  # noqa: E402


# -------------------------- ingestion ---------------------------------- #

def _normalize_wc_pred(raw: str) -> str:
    pred = (raw or "").strip().lower()
    if pred in WEATHERS:
        return pred
    return _fuzzy_weather(pred) or "UNKNOWN"


def _run_label(csv_path: Path, model: str) -> str:
    """Short tag used in legends/columns. Backend prefix from filename if any."""
    stem = csv_path.stem
    m = re.match(r"(anthropic|openai|gemini|google|hf)_", stem)
    backend = m.group(1) if m else None
    base = model or stem
    return f"{backend}/{base}" if backend else base


def load_run(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"{csv_path}: empty CSV")
    model = str(df["model"].iloc[0]) if "model" in df.columns else csv_path.stem
    label = _run_label(csv_path, model)

    out: dict = {"path": csv_path, "model": model, "label": label, "df": df}

    # Companion summary (optional but used for n + sanity).
    sj = csv_path.with_suffix(".summary.json")
    if sj.exists():
        try:
            out["summary"] = json.loads(sj.read_text())
        except Exception:
            out["summary"] = None
    return out


# ---------------------------- metrics ---------------------------------- #

def wc_metrics(df: pd.DataFrame) -> dict | None:
    # Some older CSVs include a `task` column (for the experimental Vis
    # subtask); filter to WC rows only when present.
    if "task" in df.columns:
        df = df[df["task"] == "wc"]
    if df.empty:
        return None
    from sklearn.metrics import (
        accuracy_score, f1_score, confusion_matrix, classification_report,
    )
    y = df["weather"].astype(str).tolist()
    yhat = [_normalize_wc_pred(p) for p in df["pred"].astype(str)]
    labels_full = WEATHERS + ["UNKNOWN"]
    rep = classification_report(
        y, yhat, labels=WEATHERS, output_dict=True, zero_division=0,
    )
    per_class_f1 = {w: rep[w]["f1-score"] for w in WEATHERS}
    return {
        "accuracy": float(accuracy_score(y, yhat)),
        "macro_f1": float(f1_score(y, yhat, labels=WEATHERS,
                                    average="macro", zero_division=0)),
        "per_class_f1": per_class_f1,
        "confusion_matrix": confusion_matrix(y, yhat, labels=labels_full),
        "labels": labels_full,
        "n": len(df),
    }


# ----------------------------- plots ----------------------------------- #

def plot_accuracy_bars(runs: list[dict], out: Path) -> None:
    labels = [r["label"] for r in runs]
    wc_acc = [r["wc"]["accuracy"] if r.get("wc") else np.nan for r in runs]
    wc_f1  = [r["wc"]["macro_f1"] if r.get("wc") else np.nan for r in runs]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(labels) + 3), 5))

    bars_acc = ax.bar(x - width / 2, wc_acc, width, label="WC accuracy")
    bars_f1  = ax.bar(x + width / 2, wc_f1,  width, label="WC macro F1")

    for group in (bars_acc, bars_f1):
        for rect in group:
            h = rect.get_height()
            if np.isnan(h):
                continue
            ax.annotate(f"{h:.2f}", (rect.get_x() + rect.get_width() / 2, h),
                        ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("XWOD-LLM-WC headline metrics across runs")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_per_class_f1(runs: list[dict], out: Path) -> None:
    runs = [r for r in runs if r.get("wc")]
    if not runs:
        return
    categories = list(WEATHERS) + ["ALL"]
    x = np.arange(len(categories))
    width = 0.8 / len(runs)
    fig, ax = plt.subplots(figsize=(max(8, 1.0 * len(categories) + 2), 5))
    for i, r in enumerate(runs):
        f1s = [r["wc"]["per_class_f1"].get(w, 0.0) for w in WEATHERS]
        f1s.append(r["wc"]["macro_f1"])
        ax.bar(x - 0.4 + width * (i + 0.5), f1s, width, label=r["label"])
    ax.axvline(len(WEATHERS) - 0.5, color="gray", linestyle="--",
               linewidth=0.8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1")
    ax.set_title("Per-class F1 (XWOD-LLM-WC); ALL = macro F1")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_confusion_grid(runs: list[dict], out: Path) -> None:
    runs = [r for r in runs if r.get("wc")]
    if not runs:
        return
    n = len(runs)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.6 * rows),
                             squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for idx, r in enumerate(runs):
        ax = axes[idx // cols][idx % cols]
        ax.axis("on")
        cm = r["wc"]["confusion_matrix"].astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)
        labels = r["wc"]["labels"]
        im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(r["label"], fontsize=10)
        for i in range(len(labels)):
            for j in range(len(labels)):
                v = cm_norm[i, j]
                if v < 0.005:
                    continue
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7,
                        color="white" if v > 0.5 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Row-normalized confusion matrices (XWOD-LLM-WC)", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ------------------------------ main ----------------------------------- #

def build_leaderboard(runs: list[dict]) -> pd.DataFrame:
    rec = []
    for r in runs:
        row = {"label": r["label"], "model": r["model"],
               "csv": str(r["path"])}
        if r.get("wc"):
            row["wc_n"] = r["wc"]["n"]
            row["wc_accuracy"] = r["wc"]["accuracy"]
            row["wc_macro_f1"] = r["wc"]["macro_f1"]
        rec.append(row)
    df = pd.DataFrame(rec)
    if "wc_macro_f1" in df.columns:
        df = df.sort_values("wc_macro_f1", ascending=False, na_position="last")
    return df.reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="*", type=Path,
                   help="result CSV paths (default: results/*.csv)")
    p.add_argument("--out-dir", type=Path, default=Path("results/comparison"),
                   help="where to write figures and leaderboard.csv")
    args = p.parse_args()

    paths = list(args.inputs)
    if not paths:
        paths = sorted(Path("results").glob("*.csv"))
        # Skip files we ourselves emit into `comparison/`.
        paths = [p for p in paths if "comparison" not in p.parts]
    if not paths:
        sys.exit("No result CSVs found. Pass paths explicitly or run an "
                 "evaluation first.")

    runs = []
    for path in paths:
        try:
            r = load_run(path)
        except Exception as e:
            print(f"  skip {path}: {e}", file=sys.stderr)
            continue
        r["wc"] = wc_metrics(r["df"])
        if not r["wc"]:
            print(f"  skip {path}: no scorable WC rows", file=sys.stderr)
            continue
        runs.append(r)

    if not runs:
        sys.exit("No scorable runs found.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    lb = build_leaderboard(runs)
    lb_path = args.out_dir / "leaderboard.csv"
    lb.to_csv(lb_path, index=False)
    print(lb.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nLeaderboard: {lb_path}")

    plot_accuracy_bars(runs, args.out_dir / "accuracy_comparison.png")
    plot_per_class_f1(runs, args.out_dir / "per_class_f1.png")
    plot_confusion_grid(runs, args.out_dir / "confusion_matrices.png")
    print(f"Figures:     {args.out_dir}/")


if __name__ == "__main__":
    main()
