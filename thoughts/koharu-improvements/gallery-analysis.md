# Pipeline e2e gallery — aggregate analysis

**Gallery size:** 47 images
**Total blocks detected:** 253  (avg 5.4/page)
**Total text lines:** 568

## Stage latency (ms, CPU-only run)

| Stage | p50 | p95 | mean | n |
|---|---:|---:|---:|---:|
| detect | 811 | 1129 | 825 | 47 |
| ocr | 4039 | 16463 | 6104 | 47 |
| inpaint | 11436 | 48087 | 16494 | 47 |
| translate | 50439 | 93330 | 46560 | 45 |

## Block count distribution

- min: 0
- median: 4
- max: 15

## Observed failure rates

| Mode | Count | Rate | Instances |
|---|---:|---:|---|
| zero_detect | 1 | 2.1% | animetext_1104718 |
| raw_jp_fallback | 8 | 17.0% | animetext_1000269, animetext_1000391, animetext_1000413, animetext_1105107, animetext_1134971 … |
| ocr_stuck | 2 | 4.3% | animetext_1000226, animetext_1178082 |

## Per-image bubble counts

| Image | Blocks |
|---|---:|
| AisazuNihaIrarenai-003 | 15 |
| animetext_1000011 | 13 |
| animetext_1000072 | 8 |
| animetext_1000136 | 3 |
| animetext_1000226 | 2 |
| animetext_1000269 | 7 |
| animetext_1000375 | 5 |
| animetext_1000391 | 5 |
| animetext_1000413 | 11 |
| animetext_1000427 | 8 |
| animetext_1000453 | 11 |
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
