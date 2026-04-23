# Pipeline e2e gallery — aggregate analysis

**Gallery size:** 8 images
**Total blocks detected:** 62  (avg 7.8/page)
**Total text lines:** 199

## Stage latency (ms, CPU-only run)

| Stage | p50 | p95 | mean | n |
|---|---:|---:|---:|---:|
| detect | 731 | 1137 | 817 | 8 |
| ocr | 4257 | 31865 | 10392 | 8 |
| inpaint | 16736 | 70112 | 26531 | 8 |
| translate | 28210 | 128629 | 58616 | 7 |

## Block count distribution

- min: 0
- median: 6
- max: 22

## Observed failure rates

| Mode | Count | Rate | Instances |
|---|---:|---:|---|
| zero_detect | 1 | 12.5% | animetext_1104718 |
| raw_jp_fallback | 1 | 12.5% | animetext_1134971 |
| ocr_stuck | 1 | 12.5% | animetext_1178082 |

## Per-image bubble counts

| Image | Blocks |
|---|---:|
| AisazuNihaIrarenai-003 | 15 |
| animetext_1004023 | 7 |
| animetext_1029004 | 22 |
| animetext_1104718 | 0 |
| animetext_1134971 | 6 |
| animetext_1148788 | 7 |
| animetext_1178082 | 2 |
| de | 3 |
