# Pipeline e2e gallery

Historical generations cleared. Ready to receive user-supplied images.

## How to run

```bash
cd backend

# per-stage gallery for one or more images
uv run python scripts/visualize_e2e_pipeline.py \
    --out /path/to/output-dir \
    --final-only /path/to/final-only-dir \
    --skip-features \
    /path/to/image1.jpg /path/to/image2.jpg ...

# all flags:
uv run python scripts/visualize_e2e_pipeline.py --help

# aggregate stats across any gallery dir
uv run python scripts/gallery_analysis.py

# collect just 11_final_composite.png from an existing gallery into a flat folder
uv run python scripts/collect_finals.py <gallery_dir> <flat_out_dir>

# fast iteration on font / layout (reuses cached 07_inpainted + 09_translate_response)
uv run python scripts/refit_final_composites.py
```

Each page dir produced by the visualizer contains 11 per-stage artefacts:

| # | File | Stage |
|---|---|---|
| 01 | `01_original.png` | input |
| 02 | `02_detect_blocks.png` | CTD bubble/block bboxes |
| 03 | `03_detect_lines.png` | CTD text-line bboxes |
| 04 | `04_mask_refined.png` | Koharu block-aware refined mask |
| 05 | `05_inpaint_mask.png` | LaMa erase mask (red overlay) |
| 06 | `06_ocr_crops.png` | per-line crops with normalized OCR text |
| 07 | `07_inpainted.png` | LaMa clean plate |
| 08 | `08_translate_prompt.txt` | batched `[N]`-tagged prompt |
| 09 | `09_translate_response.txt` | raw LLM response |
| 10 | `10_ocr_translate.png` | JP/EN pairs on the page |
| 11 | `11_final_composite.png` | translated text on LaMa plate |

Research corpus (9 oracle rounds, 28-item backlog) is preserved under
`thoughts/koharu-improvements/round{2..9}-research/`.
