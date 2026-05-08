"""
XWOD-LLM-WC: weather classification by vision-capable LLMs.

Implements the protocol described in §4.3 of the XWOD paper: given a
single street-level image, the model predicts ONE of seven canonical
labels — rain, snow, fog, haze/sand/dust, flooding, tornado, wildfire.
Ground truth is read directly from the XWOD filename prefix
(e.g. heavy_* -> rain, dust_* -> haze/sand/dust); no extra annotation.

Supported backends:

  * anthropic  (Claude 4.x Sonnet / Opus / Haiku)
  * openai     (GPT-4o, GPT-5.x family)
  * google     (Gemini 2.5 / 3.x via the new google-genai SDK)
  * hf         (local HuggingFace VLMs: Qwen2-VL, LLaVA-OV, InternVL-2)

Usage
-----
    conda activate torch
    pip install -r requirements.txt

    # API keys can be set in a .env file in this directory (or any
    # parent), or exported as regular shell env vars. Supported vars:
    #   ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY
    # Override which .env to load with --env-file /path/to/.env.

    python eval_llm_wc.py \\
        --data ../dataset \\
        --split test \\
        --balanced-n 100 \\
        --backend anthropic \\
        --model claude-opus-4-7 \\
        --out results/mllm_claude.csv

    # Open-weights VLM via HF
    python eval_llm_wc.py --backend hf \\
        --model Qwen/Qwen2-VL-72B-Instruct

The harness produces:
  * a row-level CSV with one row per (image, model)
  * a summary JSON with accuracy, macro-F1, and an 8x8 confusion matrix
    (rows are GT classes plus an UNKNOWN row for unparseable replies).
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
from pathlib import Path
from typing import Callable

import pandas as pd
from PIL import Image


def load_dotenv_if_present(path: str | None = None) -> None:
    """Load environment variables from a .env file if python-dotenv is available.

    Search order when ``path`` is None:
      1. $XWOD_ENV_FILE if set
      2. ./.env in the current working directory
      3. .env next to this script
      4. walk up parents of the script looking for a .env
    Silently no-ops if python-dotenv is not installed or no file is found.
    """
    try:
        from dotenv import load_dotenv, find_dotenv
    except ImportError:
        if path:
            print(f"[dotenv] python-dotenv not installed; ignoring --env-file {path}",
                  file=sys.stderr)
        return

    if path:
        loaded = load_dotenv(path, override=False)
        print(f"[dotenv] loaded {path} ({'ok' if loaded else 'missing'})", file=sys.stderr)
        return

    env_path = os.environ.get("XWOD_ENV_FILE")
    if env_path and Path(env_path).exists():
        load_dotenv(env_path, override=False)
        print(f"[dotenv] loaded {env_path}", file=sys.stderr)
        return

    found = find_dotenv(usecwd=True)
    if not found:
        here = Path(__file__).resolve().parent
        for parent in [here, *here.parents]:
            candidate = parent / ".env"
            if candidate.exists():
                found = str(candidate)
                break
    if found:
        load_dotenv(found, override=False)
        print(f"[dotenv] loaded {found}", file=sys.stderr)


WEATHERS = ["rain", "snow", "fog", "haze/sand/dust", "flooding", "tornado", "wildfire"]

PREFIX_TO_CANON = {
    "heavy": "rain",
    "snow": "snow",
    "fog": "fog",
    "dust": "haze/sand/dust",
    "flooding": "flooding",
    "tornado": "tornado",
    "wildfire": "wildfire",
}


# ------------------------------- prompts ------------------------------- #

WC_PROMPT = (
    "You are an expert in autonomous-driving scene understanding. Look at the "
    "image and classify the primary extreme weather condition as ONE of:\n"
    "  rain | snow | fog | haze/sand/dust | flooding | tornado | wildfire\n"
    "Reply with only the label, lowercase, no punctuation."
)


# ----------------------------- data loading ---------------------------- #

def load_balanced_probe(root: Path, split: str, n_per_weather: int, seed: int = 0) -> list[dict]:
    """Return a class-balanced list of {path, weather} records."""
    rng = random.Random(seed)
    img_dir = root / split / "images"
    buckets: dict[str, list[Path]] = {c: [] for c in WEATHERS}
    for f in img_dir.iterdir():
        prefix = f.name.split("_")[0]
        canon = PREFIX_TO_CANON.get(prefix)
        if canon:
            buckets[canon].append(f)
    probe = []
    for canon, paths in buckets.items():
        rng.shuffle(paths)
        for p in paths[:n_per_weather]:
            probe.append({"path": str(p), "weather": canon, "image_id": p.stem})
    rng.shuffle(probe)
    return probe


# ----------------------------- backends -------------------------------- #

def img_to_b64(path: str, max_side: int = 1024) -> str:
    img = Image.open(path).convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.standard_b64encode(buf.getvalue()).decode()


def anthropic_backend(model: str, timeout_s: int = 60):
    import anthropic
    client = anthropic.Anthropic(timeout=timeout_s)
    def call(prompt: str, image_path: str) -> str:
        msg = client.messages.create(
            model=model,
            max_tokens=512,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg",
                    "data": img_to_b64(image_path)}},
                {"type": "text", "text": prompt},
            ]}],
        )
        return msg.content[0].text
    return call


def openai_backend(model: str, timeout_s: int = 60):
    import openai
    client = openai.OpenAI(timeout=timeout_s)
    # Newer models (o1/o3/gpt-5/...) reject `max_tokens` and require
    # `max_completion_tokens`; older ones still take `max_tokens`.
    token_kwarg = {"max_tokens": 512}
    def call(prompt: str, image_path: str) -> str:
        nonlocal token_kwarg
        messages = [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{img_to_b64(image_path)}"}},
        ]}]
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
        return resp.choices[0].message.content
    return call


def gemini_backend(model: str, timeout_s: int = 60):
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
    def call(prompt: str, image_path: str) -> str:
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
        return r.text or ""
    return call


def hf_backend(model: str):
    import torch
    from transformers import AutoProcessor, AutoModelForCausalLM
    proc = AutoProcessor.from_pretrained(model, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto"
    )
    def call(prompt: str, image_path: str) -> str:
        img = Image.open(image_path).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = proc.apply_chat_template(msgs, add_generation_prompt=True)
        inputs = proc(text=[text], images=[img], return_tensors="pt").to(mdl.device)
        out = mdl.generate(**inputs, max_new_tokens=512, do_sample=False)
        return proc.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    return call


BACKENDS: dict[str, Callable[[str], Callable]] = {
    "anthropic": anthropic_backend,
    "openai":    openai_backend,
    "google":    gemini_backend,
    "hf":        hf_backend,
}


# ----------------------------- scoring --------------------------------- #

def _fuzzy_weather(pred: str) -> str | None:
    if not pred:
        return None
    for w in WEATHERS:
        if w in pred or w.replace("/", " ") in pred:
            return w
    for key, canon in [("haze", "haze/sand/dust"),
                       ("sand", "haze/sand/dust"),
                       ("dust", "haze/sand/dust"),
                       ("flood", "flooding"),
                       ("rainy", "rain"),
                       ("snowy", "snow"),
                       ("foggy", "fog"),
                       ("fire", "wildfire")]:
        if key in pred:
            return canon
    return None


def score_wc(rows: list[dict]) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    y, yhat = [], []
    for r in rows:
        pred = (r.get("pred") or "").strip().lower()
        pred = pred if pred in WEATHERS else _fuzzy_weather(pred)
        y.append(r["weather"])
        yhat.append(pred or "UNKNOWN")
    acc = accuracy_score(y, yhat)
    f1 = f1_score(y, yhat, labels=WEATHERS, average="macro", zero_division=0)
    cm = confusion_matrix(y, yhat, labels=WEATHERS + ["UNKNOWN"]).tolist()
    return {"accuracy": acc, "macro_f1": f1, "confusion_matrix": cm, "n": len(rows)}


# --------------------------------- run --------------------------------- #

def run_task(probe: list[dict], call, prompt: str, field: str) -> list[dict]:
    out = []
    n = len(probe)
    t0 = time.monotonic()
    for i, rec in enumerate(probe):
        err = None
        try:
            pred = call(prompt, rec["path"])
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            pred = f"ERROR: {err}"
        rec2 = dict(rec)
        rec2["pred"] = pred
        rec2["task"] = field
        out.append(rec2)
        elapsed = time.monotonic() - t0
        rate = (i + 1) / elapsed if elapsed > 0 else 0.0
        eta = (n - (i + 1)) / rate if rate > 0 else float("inf")
        pred_preview = " ".join(str(pred).split())
        if len(pred_preview) > 120:
            pred_preview = pred_preview[:117] + "..."
        print(
            f"  [{field}] {i+1}/{n}  id={rec['image_id']}  "
            f"gt={rec.get('weather','?')}  pred={pred_preview}  "
            f"({rate:.2f} img/s, ETA {eta:.0f}s)",
            file=sys.stderr, flush=True,
        )
        if err:
            print(f"    ERROR: {err}", file=sys.stderr, flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--split", default="test", choices=["train", "valid", "test"])
    p.add_argument("--balanced-n", type=int, default=100,
                   help="images per weather in the probe (7*n total)")
    p.add_argument("--backend", choices=list(BACKENDS), required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", type=Path, default=None,
                   help="output CSV path; if omitted, auto-generated as "
                        "results/{backend}_{model}_{timestamp}.csv")
    p.add_argument("--env-file", default=None,
                   help="path to a .env file with API keys (default: auto-discover)")
    args = p.parse_args()

    if args.out is None:
        safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", args.model).strip("_")
        ts = time.strftime("%Y%m%d-%H%M%S")
        args.out = Path("results") / f"{args.backend}_{safe_model}_{ts}.csv"
        print(f"Auto-generated output path: {args.out}", file=sys.stderr)

    load_dotenv_if_present(args.env_file)

    probe = load_balanced_probe(args.data, args.split, args.balanced_n)
    print(f"Probe size: {len(probe)} images", file=sys.stderr)

    call = BACKENDS[args.backend](args.model)
    rows = run_task(probe, call, WC_PROMPT, "wc")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df["model"] = args.model
    df.to_csv(args.out, index=False)

    summary = {
        "model": args.model, "split": args.split, "n": len(probe),
        "wc": score_wc(rows),
    }
    (args.out.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
