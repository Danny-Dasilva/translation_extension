# Pipeline e2e gallery — aggregate analysis

**Gallery size:** 59 images
**Total blocks detected:** 287  (avg 4.9/page)
**Total text lines:** 661

## Stage latency (ms, CPU-only run)

| Stage | p50 | p95 | mean | n |
|---|---:|---:|---:|---:|
| detect | 655 | 1116 | 784 | 59 |
| ocr | 3265 | 16449 | 5519 | 59 |
| inpaint | 11177 | 40748 | 14953 | 59 |
| translate | 32380 | 93009 | 44289 | 54 |

## Block count distribution

- min: 0
- median: 3
- max: 15

## Observed failure rates

| Mode | Count | Rate | Instances |
|---|---:|---:|---|
| zero_detect | 3 | 5.1% | animetext_1001097, animetext_1003016, animetext_1104718 |
| raw_jp_fallback | 8 | 13.6% | animetext_1000269, animetext_1000391, animetext_1000413, animetext_1105107, animetext_1134971 … |
| ocr_stuck | 4 | 6.8% | animetext_1000226, animetext_1003044, animetext_1003114, animetext_1178082 |

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
