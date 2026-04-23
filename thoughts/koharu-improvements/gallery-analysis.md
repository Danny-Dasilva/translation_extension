# Pipeline e2e gallery — aggregate analysis

**Gallery size:** 37 images
**Total blocks detected:** 180  (avg 4.9/page)
**Total text lines:** 402

## Stage latency (ms, CPU-only run)

| Stage | p50 | p95 | mean | n |
|---|---:|---:|---:|---:|
| detect | 945 | 1145 | 884 | 37 |
| ocr | 3265 | 16458 | 5579 | 37 |
| inpaint | 11377 | 54502 | 16793 | 37 |
| translate | 55142 | 102898 | 49480 | 35 |

## Block count distribution

- min: 0
- median: 3
- max: 15

## Observed failure rates

| Mode | Count | Rate | Instances |
|---|---:|---:|---|
| zero_detect | 1 | 2.7% | animetext_1104718 |
| raw_jp_fallback | 5 | 13.5% | animetext_1105107, animetext_1134971, animetext_1354458, animetext_1397388, animetext_1427603 |
| ocr_stuck | 1 | 2.7% | animetext_1178082 |

## Per-image bubble counts

| Image | Blocks |
|---|---:|
| AisazuNihaIrarenai-003 | 15 |
| animetext_1036053 | 4 |
| animetext_1039745 | 4 |
| animetext_1081878 | 3 |
| animetext_1099099 | 3 |
| animetext_1104598 | 3 |
| animetext_1104718 | 0 |
| animetext_1105107 | 1 |
| animetext_1111191 | 1 |
| animetext_1134971 | 6 |
| animetext_1137752 | 3 |
| animetext_1147215 | 6 |
| animetext_1148788 | 7 |
| animetext_1173904 | 13 |
| animetext_1178082 | 2 |
| animetext_1189390 | 3 |
| animetext_1225347 | 6 |
| animetext_1237113 | 14 |
| animetext_1251286 | 9 |
| animetext_1257125 | 5 |
| animetext_1285388 | 1 |
| animetext_1318337 | 9 |
| animetext_1327736 | 2 |
| animetext_1347000 | 1 |
| animetext_1349618 | 2 |
| animetext_1354458 | 2 |
| animetext_1372777 | 2 |
| animetext_1395951 | 3 |
| animetext_1397388 | 8 |
| animetext_1424183 | 3 |
| animetext_1427603 | 7 |
| animetext_1457982 | 3 |
| animetext_1459164 | 1 |
| animetext_1465722 | 2 |
| de | 3 |
| detection_v3_test | 13 |
| segmentation_v3_test | 10 |
