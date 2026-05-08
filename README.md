# XWOD — Extreme Weather Object Detection

Reproducibility code for the **XWOD** benchmark, a real-world traffic
object-detection benchmark for extreme weather. XWOD ships **10,010
images and 42,924 bounding boxes** spanning seven weather conditions —
rain, snow, fog, haze/sand/dust, **flooding**, **tornado**, and
**wildfire** — across six traffic categories (person, car, truck,
motorcycle, bus, bike). It is the first real-image detection benchmark
to cover climate-amplified hazards (flooding, tornado, wildfire) at
scale.

> **Why this repo?** Three protocols, one repo. Detection (`XWOD-Det`),
> leave-one-weather-out generalization (`XWOD-Gen`), and a
> vision-language probe (`XWOD-LLM-WC`). Drop in your model, get the
> paper's numbers.

📊 **Headline results from the paper** are reproduced below in
[Results](#results).

---

## Quick links

- **Dataset (Kaggle):** <https://www.kaggle.com/datasets/kuantinglai/exwod>
- **Paper:** *XWOD: A Real-World Benchmark for Object Detection under
  Extreme Weather Conditions* (NeurIPS 2026 submission)
- **License:** [CC BY-NC 4.0](LICENSE) (research / non-commercial use)

> The Kaggle dataset slug is `kuantinglai/exwod` because Kaggle requires
> dataset names to be **at least 5 characters** — the four-letter `xwod`
> is rejected, so the published slug pads to `exwod`. The benchmark
> itself is XWOD throughout.

---

## Repo layout

```
code/
├── README.md                # this file
├── LICENSE                  # CC BY-NC 4.0
├── requirements.txt         # YOLO + LLM dependencies
├── .env.example             # API-key template for XWOD-LLM-WC
├── eval_detection.py        # XWOD-Det: per-weather YOLO eval
├── train_detection.py       # XWOD-Det sweep + XWOD-Gen leave-one-weather-out
├── eval_llm_wc.py           # XWOD-LLM-WC weather classification
├── compare_results.py       # leaderboard + plots from eval_llm_wc.py outputs
├── latest_mm_models.py      # cross-provider latency × accuracy probe
├── demo_train_yolo.ipynb    # quick-start notebook
├── experimental/            # NOT in the paper — visibility (Vis) probe
│   ├── README.md
│   └── eval_llm_vis.py
└── results/                 # paper Table 8 artifacts (Gemini, Claude, GPT-5.5)
    ├── *.csv  *.summary.json
    └── comparison/          # accuracy_comparison.png, confusion_matrices.png, ...
```

---

## Installation

```bash
# Python 3.10+ recommended
conda create -n xwod python=3.11 -y && conda activate xwod
pip install -r requirements.txt
```

GPU is required for the YOLO paths (`eval_detection.py`,
`train_detection.py`); the LLM paths only need network access plus a
provider API key.

### API keys for `eval_llm_wc.py`

```bash
cp .env.example .env
# Then edit .env and fill in whichever keys you use:
#   ANTHROPIC_API_KEY=sk-ant-...
#   OPENAI_API_KEY=sk-...
#   GOOGLE_API_KEY=AIza...
```

Discovery order: `--env-file <path>` → `$XWOD_ENV_FILE` → `./.env` →
`.env` next to the script → any parent directory. Exported shell
variables still work; `.env` values do **not** override them.

---

## Dataset

XWOD is hosted on Kaggle. With the
[Kaggle CLI](https://github.com/Kaggle/kaggle-api) installed and
authenticated:

```bash
kaggle datasets download -d kuantinglai/exwod
unzip exwod.zip -d ../dataset
```

Layout after extraction (matches `data.yaml` in the dataset root):

```
dataset/
├── data.yaml
├── train/  (images/ + labels/)   6,206 images
├── valid/  (images/ + labels/)   1,744 images
└── test/   (images/ + labels/)   2,060 images
```

Image filenames are prefixed by weather:

| Prefix       | Canonical class    |
|--------------|--------------------|
| `heavy_*`    | rain               |
| `snow_*`     | snow               |
| `fog_*`      | fog                |
| `dust_*`     | haze/sand/dust     |
| `flooding_*` | flooding           |
| `tornado_*`  | tornado            |
| `wildfire_*` | wildfire           |

Six traffic categories: `person, car, truck, motorcycle, bus, bike`.

---

## The three protocols

### 1. XWOD-Det — standard detection

Per-weather mAP / precision / recall on the released test split.

```bash
python eval_detection.py \
  --data ../dataset \
  --weights yolov8m.pt yolo11m.pt yolo26m.pt \
  --out results/xwod_det.csv
```

To reproduce the full training sweep (paper Table 4):

```bash
python train_detection.py \
  --data ../dataset/data.yaml \
  --families 8 11 26 --scales n s m l x \
  --epochs 100 --imgsz 640 --device 0
```

> **Note on YOLO numbers.** The exact numbers in the paper (Tables 4–7)
> were produced on Roboflow training runs and external workstations,
> not via these scripts. Every architecture is stock Ultralytics, so
> running `train_detection.py` on the same released splits reproduces
> equivalent results. Expect run-to-run variance of ~±1–2 mAP₅₀.

### 2. XWOD-Gen — leave-one-weather-out generalization

For each weather *w*, train on the other six and test on *w*. Reports
mean and worst per-weather mAP across the seven runs.

```bash
python train_detection.py \
  --data ../dataset \
  --leave-one-out --model yolo11m.pt --epochs 100
# Writes results/xwod_gen_lowo.csv and prints mean / min mAP50.
```

### 3. XWOD-LLM-WC — weather classification by LLMs

A single zero-shot, single-turn prompt asks any vision-capable LLM to
emit one of `{rain, snow, fog, haze/sand/dust, flooding, tornado,
wildfire}`. Ground truth is the filename prefix. Class-balanced probe:
`--balanced-n 100` ⇒ 7 × 100 = 700 images.

```bash
# Claude
python eval_llm_wc.py --backend anthropic --model claude-opus-4-7 \
  --data ../dataset --balanced-n 100 \
  --out results/mllm_claude.csv

# GPT-5
python eval_llm_wc.py --backend openai --model gpt-5.5 \
  --data ../dataset --balanced-n 100

# Gemini
python eval_llm_wc.py --backend google --model gemini-3.1-pro-preview \
  --data ../dataset --balanced-n 100

# Local open-weights VLM
python eval_llm_wc.py --backend hf \
  --model Qwen/Qwen2-VL-72B-Instruct
```

Each run writes a row-level CSV plus a `.summary.json` with accuracy,
macro-F1, and an 8×8 confusion matrix (the eighth row is `UNKNOWN` for
unparseable replies).

| `--backend` | SDK | Examples for `--model` |
|-------------|-----|------------------------|
| `anthropic` | `anthropic` | `claude-opus-4-7`, `claude-sonnet-4-6`, `claude-haiku-4-5-20251001` |
| `openai`    | `openai`    | `gpt-5.5`, `gpt-5.4`, `gpt-4o` |
| `google`    | `google-genai` (new unified SDK; `google-generativeai` is deprecated) | `gemini-3.1-pro-preview`, `gemini-3-flash-preview`, `gemini-2.5-flash` |
| `hf`        | `transformers` (local) | `Qwen/Qwen2-VL-72B-Instruct`, `OpenGVLab/InternVL2-...`, etc. |

All API clients use a 60 s request timeout so a gated/preview model ID
fails loud instead of stalling.

#### Comparing runs

```bash
# auto-discovers results/*.csv
python compare_results.py
# or pass specific files
python compare_results.py results/google_gemini-3.1-pro-preview_*.csv \
                         results/mllm_opus4.7.csv
```

This writes `results/comparison/leaderboard.csv` plus three figures —
`accuracy_comparison.png`, `per_class_f1.png`, `confusion_matrices.png`.

#### Latest-models latency × accuracy probe

`latest_mm_models.py` is a smaller, **across-provider** probe that runs
the WC prompt against every model in `MODEL_REGISTRY` (a hard-coded list
of current Anthropic / OpenAI / Google IDs as of April 2026) and writes
plots, not just CSVs.

```bash
# Dry-run: print the model registry
python latest_mm_models.py --data ../dataset --dry-run

# Sanity-check which Gemini IDs your GOOGLE_API_KEY can actually see
python latest_mm_models.py --data ../dataset --list-google-models

# Real run (skips providers whose *_API_KEY is unset)
python latest_mm_models.py --data ../dataset --n-per-class 2 \
  --providers anthropic openai google --out results/latest_mm
```

Produces `latest_mm.csv`, `latest_mm_summary.csv`, plus PNGs for
latency, accuracy, and tokens.

---

## Results

All numbers below are reproduced verbatim from the XWOD paper.

### Table 1 — Scale and provenance of real-image extreme-weather datasets

XWOD is the largest by image count and instance count, and the only one
covering tornado, flooding, and wildfire.

| Dataset             | Year | Images  | Instances | # Weather | Source                     |
|---------------------|------|--------:|----------:|----------:|----------------------------|
| **XWOD (ours)**     | 2026 | **10,010** | **42,924** | **7** | Asia, N. America, Europe, U.S. |
| RTTS                | 2018 |   4,322 |       —   |       1   | China                      |
| ACDC                | 2021 |   4,006 |       —   |       4   | Switzerland, central Europe |
| MAS / Foggy Zurich  | 2018 |   3,808 |       —   |       1   | Zurich, Switzerland        |
| WEDGE               | 2023 |   3,360 |    16,513 |      16   | *Generative (DALL-E)*      |
| DAWN                | 2020 |   1,000 |     7,845 |       4   | Web-sourced                |

### Table 2 — Per-weather counts and box statistics

Rare-hazard classes (tornado, flooding, wildfire) contribute 72.7 % of
images and 67.8 % of instances — the first time these conditions have
been available at scale.

| Weather         | Images | Instances | Boxes/Img | Mean Rel. Area |
|-----------------|------:|---------:|---------:|---------------:|
| Rain (heavy)    |   665 |    3,270 |     4.92 |        0.0244  |
| Snow            | 1,203 |    5,599 |     4.65 |        0.0213  |
| Fog             |   306 |    1,634 |     5.34 |        0.0333  |
| Haze/Sand/Dust  |   560 |    3,321 |     5.93 |        0.0263  |
| Flooding        | 5,151 |   21,734 |     4.22 |        0.0368  |
| Tornado         | 1,164 |    4,962 |     4.26 |        0.0187  |
| Wildfire        |   961 |    2,404 |     2.50 |        0.0586  |
| **Total**       | **10,010** | **42,924** | **4.29** | **0.0316** |

### Table 4 — YOLO family on XWOD-Det (test split)

mAP / precision / recall in %; loss values absolute. **Best per
architecture in bold**, overall best <ins>underlined</ins>.

| Family   | Scale | mAP₅₀ (%) | mAP₅₀₋₉₅ (%) | Precision (%) | Recall (%) | Box loss | Cls loss | DFL loss |
|----------|:-----:|----------:|-------------:|--------------:|-----------:|---------:|---------:|---------:|
| YOLOv8   | n     | 49.24     | 28.74        | 63.59         | 46.79      | 0.9470   | 0.5195   | 0.9642   |
| YOLOv8   | s     | 51.53     | 30.60        | 63.42         | 49.23      | 0.7872   | 0.4000   | 0.9099   |
| YOLOv8   | m     | <ins>**54.69**</ins> | 32.21 | 63.05      | 53.51      | 0.7538   | 0.3715   | 0.9390   |
| YOLOv8   | l     | 52.53     | 31.34        | 68.37         | 47.13      | 0.6877   | 0.3341   | **0.9292** |
| YOLOv8   | x     | 52.70     | 30.97        | 63.08         | 50.28      | 0.6795   | 0.3269   | 0.9274   |
| YOLOv11  | n     | 46.91     | 27.36        | 65.09         | 43.77      | 0.9345   | 0.5184   | 0.9562   |
| YOLOv11  | s     | 48.35     | 28.47        | 60.85         | 47.81      | 0.8129   | 0.4210   | 0.9208   |
| YOLOv11  | m     | **53.84** | 31.93        | 63.33         | 53.47      | 0.7642   | 0.3899   | **0.9253** |
| YOLOv11  | l     | 52.06     | 30.86        | 62.33         | 49.75      | 0.7615   | 0.3788   | 0.9511   |
| YOLOv11  | x     | 52.64     | 31.41        | 59.97         | 52.39      | 0.7430   | 0.3702   | 0.9556   |
| YOLOv26  | n     | 46.29     | 26.73        | 57.92         | 45.27      | 1.1891   | 0.6250   | 0.0048   |
| YOLOv26  | s     | 51.92     | 30.49        | 63.54         | 50.18      | 1.0087   | 0.4349   | 0.0039   |
| YOLOv26  | m     | 53.34     | 32.29        | **70.40**     | 48.24      | 0.9066   | 0.3674   | **0.0035** |
| YOLOv26  | l     | **53.95** | <ins>**32.75**</ins> | 66.24 | **51.77** | 0.9070   | 0.3627   | **0.0035** |
| YOLOv26  | x     | 53.61     | 32.55        | 67.59         | 51.85      | 0.8717   | 0.3458   | 0.0035   |

> Larger scales do **not** consistently win. Medium variants are
> Pareto-best across all three families — XWOD's intra-class texture
> variance under weather regularizes smaller models. See paper §5.3.

### Table 5 — Cross-domain transfer (XWOD → unseen domains)

XWOD-trained YOLO detectors zero-shot to RTTS, DAWN, and WEDGE,
substantially outperforming each baseline.

| Setting / Source | Model       | mAP₅₀ (%) | Precision (%) | Recall (%) |
|------------------|-------------|----------:|--------------:|-----------:|
| **In-domain (XWOD-Dataset)** | YOLOv8m  | **54.69** | 63.05 | 53.51 |
| In-domain        | YOLOv11m    | 53.87 | 63.33 | 53.47 |
| In-domain        | YOLOv26m    | 53.34 | **70.40** | 48.24 |
| In-domain        | YOLOv8l     | 52.53 | 68.37 | 47.13 |
| In-domain        | YOLOv11l    | 52.06 | 62.33 | 49.75 |
| In-domain        | YOLOv26l    | 53.95 | 66.24 | 51.77 |
| **Synthetic shift — XWOD → WEDGE** | Faster R-CNN (baseline) | 45.41 | — | — |
| XWOD-Net         | YOLOv8m     | **61.12** | 66.23 | 55.98 |
| XWOD-Net         | YOLOv11m    | 55.02 | 64.33 | 53.68 |
| XWOD-Net         | YOLOv26m    | 53.94 | 71.83 | 49.00 |
| **Cross-weather — XWOD → DAWN** | Ensemble Det. (baseline) | 32.75 | — | — |
| XWOD-Net         | YOLOv8m     | **59.94** | 65.73 | 52.74 |
| XWOD-Net         | YOLOv11m    | 55.17 | 61.81 | 52.18 |
| XWOD-Net         | YOLOv26m    | 54.81 | 66.58 | 52.62 |
| **Real-world shift — XWOD → RTTS** | Standard Det. (baseline) | 40.37 | — | — |
| XWOD-Net         | YOLOv8m     | **63.00** | 69.70 | 56.56 |
| XWOD-Net         | YOLOv11m    | 54.57 | 69.65 | 50.67 |
| XWOD-Net         | YOLOv26m    | 54.95 | 69.70 | 56.56 |

XWOD → RTTS, DAWN, and WEDGE represent **+56.1 %, +83.0 %, +34.6 %**
relative mAP₅₀ improvements over the published baselines on each
target dataset.

### Table 6 — Cross-dataset comparison of best published results

| Dataset           | Best Model        | mAP (%) (All) | Best Per-Weather (%)      | Notes                      |
|-------------------|-------------------|--------------:|---------------------------|----------------------------|
| **XWOD (ours)**   | YOLOv8m           | **54.69**     | 62.93 (Tornado)           | 7 weather                  |
| XWOD              | YOLOv11m          | 53.84         | 68.97 (Tornado)           | 7 weather                  |
| XWOD              | YOLOv26l          | 53.95         | 66.01 (Tornado)           | 7 weather                  |
| RTTS              | —                 | 40.37         | —                         | 1 weather                  |
| DAWN              | Ensemble          | 32.75         | —                         | 4 weather                  |
| WEDGE             | Faster R-CNN      | 22.78         | —                         | 16 labels, generative      |
| WEDGE             | Fine-tuning       | 45.41         | —                         |                            |

### Table 7 — Per-weather performance of YOLOv11m on XWOD-Det

Wildfire and fog are the clear failure modes — wildfire collapses
recall to 15.42 % (only 1 in 6 objects found) and fog halves recall
relative to rain.

| Weather         | mAP₅₀ (%) | mAP₅₀₋₉₅ (%) | Precision (%) | Recall (%) | Box loss | Cls loss | DFL loss |
|-----------------|----------:|-------------:|--------------:|-----------:|---------:|---------:|---------:|
| Rain            | 60.49     | 37.70        | 61.74         | 61.87      | 0.699    | 0.406    | 0.938    |
| Snow            | 38.46     | 25.93        | 69.79         | 32.19      | 0.534    | 0.314    | 0.810    |
| Fog             | 23.45     | 11.97        | 66.08         | 21.92      | 0.976    | 0.564    | 1.168    |
| Haze/Sand/Dust  | 29.86     |  9.95        | 38.83         | 35.10      | 0.650    | 0.390    | 0.888    |
| Flooding        | 42.86     | 26.36        | 53.15         | 43.73      | 0.740    | 0.375    | 0.919    |
| **Tornado**     | **68.97** | 43.47        | **75.38**     | 57.70      | 0.771    | 0.415    | 0.905    |
| Wildfire        | 17.85     | 12.18        | 60.57         | 15.42      | 0.409    | 0.279    | 0.845    |

### Table 8 — XWOD-LLM-WC: weather classification by LLMs

Class-balanced probe of 7 × 100 = 700 images on the test split, zero-
shot, single-turn prompt. **The CSVs that produced these numbers ship
in `results/`** — `python compare_results.py` regenerates the
leaderboard from them.

| Provider  | Model                              | Accuracy | Macro-F1 |
|-----------|------------------------------------|---------:|---------:|
| Google    | gemini-3.1-pro-preview             |  0.7571  |  0.7585  |
| Google    | gemini-3.1-flash-lite-preview      |  0.7500  |  0.7500  |
| Anthropic | claude-opus-4-7                    |  0.7471  |  0.7446  |
| OpenAI    | gpt-5.5                            |  0.7186  |  0.7221  |

> Gemini 3.1 Pro and Flash-Lite take the top two; Claude Opus 4.7 and
> GPT-5.5 follow. All four mainstream models score above 0.71 on this
> probe — strong, but well below the 0.95 + accuracy that simple
> CNN classifiers achieve, suggesting headroom on rarer classes.

---

## Experimental extras (not in the paper)

The `experimental/` directory hosts work that did not make the final
paper. See [`experimental/README.md`](experimental/README.md) for
details. The current resident is `eval_llm_vis.py` — a 3-bucket
visibility (`L0` / `L1` / `L2`) probe whose ground truth is *derived*
from YOLO label area, not human-annotated. Use at your own risk.

---

## Citation

Our paper is currently uder review. For now, please cite the Kaggle dataset:
<https://www.kaggle.com/datasets/kuantinglai/exwod>.

---

## License

Source code and dataset are released under
[CC BY-NC 4.0](LICENSE) — attribution + non-commercial. See `LICENSE`
for the full text.
