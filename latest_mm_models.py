"""
Latest multimodal model test + visualization (April 2026).

Calls the most recent vision-capable models from Anthropic, Google, and OpenAI
on a small balanced probe from the XWOD test split, then plots latency,
success rate, and per-model weather-classification accuracy.

Model IDs are as of April 2026; update MODEL_REGISTRY if the providers ship
new IDs. Each entry carries a `call` factory so the rest of the script stays
backend-agnostic.

Usage
-----
    conda activate torch
    pip install -r requirements.txt

    # Put ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY in ./.env
    python latest_mm_models.py \
        --data ../dataset --split test \
        --n-per-class 2 \
        --out results/latest_mm

A single run produces:
    results/latest_mm.csv          row-level (image, model, latency, pred, ok)
    results/latest_mm_summary.csv  per-model aggregates
    results/latest_mm_latency.png  mean latency bar chart
    results/latest_mm_accuracy.png weather-classification accuracy bar chart
    results/latest_mm_tokens.png   in/out token bars (when provider returns them)
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd
from PIL import Image


# --------------------------- model registry --------------------------- #

# Provider -> list of current multimodal model IDs (April 2026).
# Flagship first, then smaller / faster / cheaper variants.
MODEL_REGISTRY: dict[str, list[str]] = {
  "openai": [
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
        "gpt-4o",
    ],
    "google": [
        "gemini-3.1-pro-preview",
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite",
    ],
      "anthropic": [
        "claude-opus-4-7",
        "claude-sonnet-4-6",
        "claude-haiku-4-5-20251001",
    ],
}


# ------------------------------ prompts ------------------------------- #

WEATHERS = ["rain", "snow", "fog", "haze/sand/dust",
            "flooding", "tornado", "wildfire"]

PREFIX_TO_CANON = {
    "heavy":    "rain",
    "snow":     "snow",
    "fog":      "fog",
    "dust":     "haze/sand/dust",
    "flooding": "flooding",
    "tornado":  "tornado",
    "wildfire": "wildfire",
}

WC_PROMPT = (
    "You are an expert in autonomous-driving scene understanding. Look at the "
    "image and classify the primary extreme weather condition as ONE of:\n"
    "  rain | snow | fog | haze/sand/dust | flooding | tornado | wildfire\n"
    "Reply with only the label, lowercase, no punctuation."
)


# ---------------------------- data loading ---------------------------- #

def load_balanced_probe(root: Path, split: str, n_per_class: int,
                        seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    img_dir = root / split / "images"
    buckets: dict[str, list[Path]] = {c: [] for c in WEATHERS}
    for f in img_dir.iterdir():
        canon = PREFIX_TO_CANON.get(f.name.split("_")[0])
        if canon:
            buckets[canon].append(f)
    probe = []
    for canon, paths in buckets.items():
        rng.shuffle(paths)
        for p in paths[:n_per_class]:
            probe.append({"path": str(p), "weather": canon,
                          "image_id": p.stem})
    rng.shuffle(probe)
    return probe


def img_to_b64(path: str, max_side: int = 1024) -> str:
    img = Image.open(path).convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.standard_b64encode(buf.getvalue()).decode()


# ------------------------------ backends ------------------------------ #

@dataclass
class CallResult:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None


def anthropic_call(model: str, timeout_s: int = 60) -> Callable[[str, str], CallResult]:
    import anthropic
    client = anthropic.Anthropic(timeout=timeout_s)

    def _call(prompt: str, image_path: str) -> CallResult:
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=64,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": img_to_b64(image_path)}},
                    {"type": "text", "text": prompt},
                ]}],
            )
            return CallResult(
                text=msg.content[0].text,
                input_tokens=getattr(msg.usage, "input_tokens", None),
                output_tokens=getattr(msg.usage, "output_tokens", None),
            )
        except Exception as e:
            return CallResult(text="", error=f"{type(e).__name__}: {e}")

    return _call


def openai_call(model: str, timeout_s: int = 60) -> Callable[[str, str], CallResult]:
    import openai
    client = openai.OpenAI(timeout=timeout_s)
    # Newer models (o1/o3/gpt-5/...) reject `max_tokens` and require
    # `max_completion_tokens`; older ones still take `max_tokens`.
    token_kwarg = {"max_tokens": 64}

    def _call(prompt: str, image_path: str) -> CallResult:
        nonlocal token_kwarg
        messages = [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{img_to_b64(image_path)}"}},
        ]}]
        try:
            try:
                resp = client.chat.completions.create(
                    model=model, messages=messages, **token_kwarg,
                )
            except openai.BadRequestError as e:
                if "max_completion_tokens" in str(e) and "max_tokens" in token_kwarg:
                    token_kwarg = {"max_completion_tokens": token_kwarg["max_tokens"]}
                    resp = client.chat.completions.create(
                        model=model, messages=messages, **token_kwarg,
                    )
                else:
                    raise
            usage = getattr(resp, "usage", None)
            return CallResult(
                text=resp.choices[0].message.content or "",
                input_tokens=getattr(usage, "prompt_tokens", None),
                output_tokens=getattr(usage, "completion_tokens", None),
            )
        except Exception as e:
            return CallResult(text="", error=f"{type(e).__name__}: {e}")

    return _call


def google_call(model: str, timeout_s: int = 60) -> Callable[[str, str], CallResult]:
    # New unified SDK — `pip install google-genai`.
    # The old `google-generativeai` package is deprecated.
    # Explicit timeout: without it, preview/unavailable model IDs can hang
    # silently on connect (HttpOptions.timeout is in ms).
    from google import genai
    from google.genai import types
    client = genai.Client(
        api_key=os.environ["GOOGLE_API_KEY"],
        http_options=types.HttpOptions(timeout=timeout_s * 1000),
    )

    def _call(prompt: str, image_path: str) -> CallResult:
        try:
            img = Image.open(image_path).convert("RGB")
            if max(img.size) > 1024:
                img.thumbnail((1024, 1024))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
            r = client.models.generate_content(
                model=model,
                contents=[
                    prompt,
                    types.Part.from_bytes(data=buf.getvalue(),
                                          mime_type="image/jpeg"),
                ],
            )
            meta = getattr(r, "usage_metadata", None)
            return CallResult(
                text=(r.text or ""),
                input_tokens=getattr(meta, "prompt_token_count", None),
                output_tokens=getattr(meta, "candidates_token_count", None),
            )
        except Exception as e:
            return CallResult(text="", error=f"{type(e).__name__}: {e}")

    return _call


def list_google_models() -> list[str]:
    """List Gemini model IDs the current API key can actually see.

    Use this to sanity-check MODEL_REGISTRY["google"] — preview IDs
    (e.g. -preview suffix) are often gated to specific projects/regions,
    and calls to an un-enrolled ID will stall on the server side."""
    from google import genai
    from google.genai import types
    client = genai.Client(
        api_key=os.environ["GOOGLE_API_KEY"],
        http_options=types.HttpOptions(timeout=15_000),
    )
    return [m.name for m in client.models.list()]


BACKENDS: dict[str, Callable[[str], Callable[[str, str], CallResult]]] = {
    "anthropic": anthropic_call,
    "openai":    openai_call,
    "google":    google_call,
}


# ------------------------------ parsing ------------------------------- #

def normalize_weather(pred: str) -> str:
    p = (pred or "").strip().lower()
    if p in WEATHERS:
        return p
    for w in WEATHERS:
        if w in p or w.replace("/", " ") in p:
            return w
    for key, canon in [("haze", "haze/sand/dust"),
                       ("sand", "haze/sand/dust"),
                       ("dust", "haze/sand/dust"),
                       ("flood", "flooding"),
                       ("rainy", "rain"),
                       ("snowy", "snow"),
                       ("foggy", "fog"),
                       ("fire", "wildfire")]:
        if key in p:
            return canon
    return "UNKNOWN"


# ------------------------------- run ---------------------------------- #

@dataclass
class Row:
    provider: str
    model: str
    image_id: str
    weather: str
    pred_raw: str
    pred: str
    correct: bool
    latency_s: float
    input_tokens: int | None
    output_tokens: int | None
    error: str | None = None


def run(probe: list[dict], providers: list[str],
        skip_missing_keys: bool = True) -> list[Row]:
    env_need = {"anthropic": "ANTHROPIC_API_KEY",
                "openai":    "OPENAI_API_KEY",
                "google":    "GOOGLE_API_KEY"}
    rows: list[Row] = []
    for provider in providers:
        if skip_missing_keys and not os.environ.get(env_need[provider]):
            print(f"[skip] {provider}: {env_need[provider]} not set",
                  file=sys.stderr)
            continue
        for model in MODEL_REGISTRY[provider]:
            print(f"[{provider}] {model}", file=sys.stderr)
            try:
                call = BACKENDS[provider](model)
            except Exception as e:
                print(f"  init failed: {e}", file=sys.stderr)
                continue
            for i, rec in enumerate(probe):
                t0 = time.perf_counter()
                res = call(WC_PROMPT, rec["path"])
                dt = time.perf_counter() - t0
                pred = normalize_weather(res.text)
                rows.append(Row(
                    provider=provider, model=model,
                    image_id=rec["image_id"], weather=rec["weather"],
                    pred_raw=res.text, pred=pred,
                    correct=(pred == rec["weather"] and res.error is None),
                    latency_s=dt,
                    input_tokens=res.input_tokens,
                    output_tokens=res.output_tokens,
                    error=res.error,
                ))
                if (i + 1) % 10 == 0:
                    print(f"  {i+1}/{len(probe)}", file=sys.stderr)
    return rows


# ---------------------------- visualization --------------------------- #

def visualize(df: pd.DataFrame, out_prefix: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agg = (df.groupby(["provider", "model"])
             .agg(n=("model", "size"),
                  mean_latency_s=("latency_s", "mean"),
                  accuracy=("correct", "mean"),
                  errors=("error", lambda s: s.notna().sum()),
                  in_tok=("input_tokens", "mean"),
                  out_tok=("output_tokens", "mean"))
             .reset_index()
             .sort_values(["provider", "mean_latency_s"]))
    agg.to_csv(out_prefix.with_name(out_prefix.name + "_summary.csv"),
               index=False)
    print("\n=== Summary ===")
    print(agg.to_string(index=False))

    colors = {"anthropic": "#d97757",
              "openai":    "#10a37f",
              "google":    "#4285f4"}
    labels = [f"{r.provider}:{r.model}" for r in agg.itertuples()]
    cvec = [colors.get(p, "#888") for p in agg["provider"]]

    # latency
    fig, ax = plt.subplots(figsize=(10, max(3, 0.4 * len(agg))))
    ax.barh(labels, agg["mean_latency_s"], color=cvec)
    ax.set_xlabel("Mean latency per image (s)")
    ax.set_title("Latest multimodal models — latency")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_prefix.with_name(out_prefix.name + "_latency.png"), dpi=140)
    plt.close(fig)

    # accuracy
    fig, ax = plt.subplots(figsize=(10, max(3, 0.4 * len(agg))))
    ax.barh(labels, agg["accuracy"], color=cvec)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Weather-classification accuracy")
    ax.set_title("Latest multimodal models — accuracy (XWOD probe)")
    ax.invert_yaxis()
    for i, v in enumerate(agg["accuracy"]):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_prefix.with_name(out_prefix.name + "_accuracy.png"), dpi=140)
    plt.close(fig)

    # tokens (only if any provider returned them)
    has_tok = agg[["in_tok", "out_tok"]].notna().any().any()
    if has_tok:
        fig, ax = plt.subplots(figsize=(10, max(3, 0.4 * len(agg))))
        y = range(len(agg))
        ax.barh([i - 0.2 for i in y], agg["in_tok"].fillna(0),
                height=0.4, label="input tokens", color="#888")
        ax.barh([i + 0.2 for i in y], agg["out_tok"].fillna(0),
                height=0.4, label="output tokens", color="#333")
        ax.set_yticks(list(y))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Mean tokens per call")
        ax.set_title("Latest multimodal models — token usage")
        ax.invert_yaxis()
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_prefix.with_name(out_prefix.name + "_tokens.png"), dpi=140)
        plt.close(fig)

    print(f"\nPlots written next to {out_prefix}")


# -------------------------------- main -------------------------------- #

def _load_dotenv(path: str | None) -> None:
    try:
        from dotenv import load_dotenv, find_dotenv
    except ImportError:
        return
    if path:
        load_dotenv(path, override=False)
        return
    found = find_dotenv(usecwd=True)
    if found:
        load_dotenv(found, override=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True,
                   help="XWOD dataset root (contains test/images)")
    p.add_argument("--split", default="test",
                   choices=["train", "valid", "test"])
    p.add_argument("--n-per-class", type=int, default=2,
                   help="images per weather class (7 classes total)")
    p.add_argument("--providers", nargs="+",
                   default=list(MODEL_REGISTRY),
                   choices=list(MODEL_REGISTRY))
    p.add_argument("--out", type=Path, default=Path("results/latest_mm"),
                   help="output prefix (no extension)")
    p.add_argument("--env-file", default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="print the model list and exit without calling APIs")
    p.add_argument("--list-google-models", action="store_true",
                   help="print Gemini model IDs visible to GOOGLE_API_KEY and exit")
    args = p.parse_args()

    _load_dotenv(args.env_file)

    if args.dry_run:
        for prov, models in MODEL_REGISTRY.items():
            print(f"{prov}:")
            for m in models:
                print(f"  - {m}")
        return

    if args.list_google_models:
        for name in list_google_models():
            print(name)
        return

    probe = load_balanced_probe(args.data, args.split, args.n_per_class)
    print(f"Probe: {len(probe)} images across {len(WEATHERS)} classes",
          file=sys.stderr)

    rows = run(probe, args.providers)
    if not rows:
        print("No rows produced — did you set any API keys?", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame([r.__dict__ for r in rows])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out.with_suffix(".csv"), index=False)

    visualize(df, args.out)


if __name__ == "__main__":
    main()
