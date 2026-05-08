# Experimental — not part of the XWOD paper

Scripts in this folder were prototyped during XWOD development but did
**not** make it into the final benchmark. They are provided for
researchers who want to extend the work; treat any numbers reported here
as preliminary, not as paper results.

## `eval_llm_vis.py` — visibility (Vis) probe

A 3-bucket visibility probe (`L0` / `L1` / `L2`) where ground truth is
*derived* from the released YOLO labels: each image's score is the sum
of normalized bounding-box areas, and the split is bucketed at the
global terciles of that score.

This is a **derived proxy**, not a human-annotated label, which is why
it was excluded from the paper. The paper's only LLM track is
[XWOD-LLM-WC](../README.md#3-xwod-llm-wc-weather-classification-by-llms)
(weather classification).

### Usage

```bash
python experimental/eval_llm_vis.py \
    --data ../dataset --split test --balanced-n 100 \
    --backend google --model gemini-3.1-pro-preview \
    --out experimental/results/vis_gemini.csv
```

The four backends (`anthropic` / `openai` / `google` / `hf`) are inherited
from the main `eval_llm_wc.py` module, so behavior is identical apart
from the prompt and scorer. Output is a per-image CSV plus a summary
JSON containing accuracy, Spearman ρ on the L0/L1/L2 ordinal, and a 3×3
confusion matrix.
