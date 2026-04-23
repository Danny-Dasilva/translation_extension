# Pipeline e2e gallery — aggregate analysis

**Gallery size:** 135 images
**Total blocks detected:** 667  (avg 4.9/page)
**Total text lines:** 1435

## Stage latency (ms, CPU-only run)

| Stage | p50 | p95 | mean | n |
|---|---:|---:|---:|---:|
| detect | 608 | 1094 | 692 | 135 |
| ocr | 3055 | 16454 | 5250 | 135 |
| inpaint | 10421 | 31377 | 12866 | 135 |
| translate | 28113 | 84965 | 33461 | 123 |

## Block count distribution

- min: 0
- median: 3
- max: 17

## Observed failure rates

| Mode | Count | Rate | Instances |
|---|---:|---:|---|
| zero_detect | 7 | 5.2% | animetext_1001097, animetext_1003016, animetext_1007105, animetext_1018203, animetext_1019120 … |
| raw_jp_fallback | 23 | 17.0% | animetext_1000269, animetext_1000391, animetext_1000413, animetext_1005089, animetext_1006016 … |
| ocr_stuck | 5 | 3.7% | animetext_1000226, animetext_1003044, animetext_1003114, animetext_1018271, animetext_1178082 |

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
| animetext_1007105 | 0 |
| animetext_1007108 | 3 |
| animetext_1007114 | 1 |
| animetext_1007196 | 2 |
| animetext_1008037 | 4 |
| animetext_1008118 | 3 |
| animetext_1008223 | 6 |
| animetext_1008225 | 4 |
| animetext_1009022 | 12 |
| animetext_1009073 | 3 |
| animetext_1009075 | 8 |
| animetext_1009101 | 2 |
| animetext_1010002 | 8 |
| animetext_1010060 | 13 |
| animetext_1010062 | 14 |
| animetext_1010179 | 1 |
| animetext_1011001 | 3 |
| animetext_1011012 | 4 |
| animetext_1011056 | 1 |
| animetext_1011072 | 11 |
| animetext_1012018 | 1 |
| animetext_1012039 | 2 |
| animetext_1012040 | 2 |
| animetext_1012046 | 14 |
| animetext_1013095 | 1 |
| animetext_1013103 | 4 |
| animetext_1013188 | 5 |
| animetext_1013203 | 13 |
| animetext_1014031 | 13 |
| animetext_1014106 | 1 |
| animetext_1014160 | 1 |
| animetext_1014165 | 7 |
| animetext_1015008 | 2 |
| animetext_1015012 | 2 |
| animetext_1015068 | 9 |
| animetext_1015084 | 8 |
| animetext_1016165 | 5 |
| animetext_1016176 | 4 |
| animetext_1016216 | 2 |
| animetext_1016223 | 2 |
| animetext_1017020 | 1 |
| animetext_1017087 | 2 |
| animetext_1017100 | 2 |
| animetext_1017105 | 2 |
| animetext_1018203 | 0 |
| animetext_1018216 | 17 |
| animetext_1018257 | 2 |
| animetext_1018271 | 6 |
| animetext_1019019 | 2 |
| animetext_1019080 | 17 |
| animetext_1019120 | 0 |
| animetext_1019155 | 5 |
| animetext_1020025 | 2 |
| animetext_1020036 | 3 |
| animetext_1020065 | 4 |
| animetext_1020107 | 6 |
| animetext_1021008 | 3 |
| animetext_1021020 | 4 |
| animetext_1021070 | 1 |
| animetext_1021094 | 0 |
| animetext_1022093 | 6 |
| animetext_1022144 | 11 |
| animetext_1022147 | 9 |
| animetext_1022149 | 10 |
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
