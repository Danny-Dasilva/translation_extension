# Pipeline e2e gallery — aggregate analysis

**Gallery size:** 71 images
**Total blocks detected:** 351  (avg 4.9/page)
**Total text lines:** 784

## Stage latency (ms, CPU-only run)

| Stage | p50 | p95 | mean | n |
|---|---:|---:|---:|---:|
| detect | 656 | 1114 | 773 | 71 |
| ocr | 3478 | 16458 | 5635 | 71 |
| inpaint | 11177 | 36000 | 14706 | 71 |
| translate | 33235 | 92592 | 45193 | 65 |

## Block count distribution

- min: 0
- median: 3
- max: 15

## Observed failure rates

| Mode | Count | Rate | Instances |
|---|---:|---:|---|
| zero_detect | 3 | 4.2% | animetext_1001097, animetext_1003016, animetext_1104718 |
| raw_jp_fallback | 10 | 14.1% | animetext_1000269, animetext_1000391, animetext_1000413, animetext_1005089, animetext_1006016 … |
| ocr_stuck | 4 | 5.6% | animetext_1000226, animetext_1003044, animetext_1003114, animetext_1178082 |

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
| animetext_1001097 | 0 |
| animetext_1001104 | 1 |
| animetext_1001105 | 2 |
| animetext_1001150 | 4 |
| animetext_1002113 | 3 |
| animetext_1002116 | 5 |
| animetext_1002204 | 1 |
| animetext_1002292 | 6 |
| animetext_1003016 | 0 |
| animetext_1003044 | 2 |
| animetext_1003104 | 3 |
| animetext_1003114 | 7 |
| animetext_1004023 | 7 |
| animetext_1004039 | 8 |
| animetext_1004062 | 8 |
| animetext_1004065 | 12 |
| animetext_1005036 | 1 |
| animetext_1005043 | 3 |
| animetext_1005053 | 10 |
| animetext_1005089 | 1 |
| animetext_1006016 | 3 |
| animetext_1006041 | 6 |
| animetext_1006052 | 4 |
| animetext_1006070 | 1 |
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
