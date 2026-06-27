#!/usr/bin/env bash
# Deterministic Ikenie4 MT regression eval.
#
# Given a predictions.jsonl (from a code/model change, in the bubbles.json ->
# {jp, en} shape), this produces a TRUSTWORTHY regression number:
#
#   1. score_jsonl_metrics.py on the ocr_clean gold subset
#      (chrF++/BLEU [+ optional kiwi/metricx/xcomet]) -- the OCR-dirty rows are
#      EXCLUDED so we measure the translation model, not OCR noise.
#   2. paired_bs_metric.py vs a previous run's per-bubble json -- the headline:
#      Δchrf + seeded(12345) CI95 + two-sided p-value.
#   3. probes.py on the gold-seeded probe cases -- deterministic, seedless.
#
# PASS criterion (printed at the end):
#   * CI95 of Δchrf++ excludes 0 (i.e. the change is statistically a win), AND
#   * no probe regresses vs the baseline probe report.
#
# Usage:
#   run_ikenie4_regression.sh \
#       --predictions  preds.jsonl \
#       --label        my_change \
#       [--baseline-per-bubble  scorecards/ikenie4/per_bubble_PREV.json] \
#       [--baseline-probes      scorecards/ikenie4/probes_PREV.json] \
#       [--metrics     chrf,bleu]      # add kiwi,metricx,xcomet for neural (GPU)
#       [--out-dir     scorecards/ikenie4]
#
# If no --baseline-per-bubble is given, the paired bootstrap self-compares the
# run against itself (CI ~ 0) -- useful for a smoke test / determinism check.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# --- defaults ---------------------------------------------------------------
PY="${PY:-python}"
GOLD="${GOLD:-$SCRIPT_DIR/data/ikenie4/gold.jsonl}"
PROBE_CASES="${PROBE_CASES:-$SCRIPT_DIR/data/ikenie4/probes.jsonl}"
METRICS="chrf,bleu"
OUT_DIR="$SCRIPT_DIR/scorecards/ikenie4"
LABEL="ikenie4_run"
PRED_KEY="en"
BASELINE_PER_BUBBLE=""
BASELINE_PROBES=""
PREDICTIONS=""

# --- arg parse --------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --predictions)          PREDICTIONS="$2"; shift 2;;
    --label)                LABEL="$2"; shift 2;;
    --metrics)              METRICS="$2"; shift 2;;
    --out-dir)              OUT_DIR="$2"; shift 2;;
    --gold)                 GOLD="$2"; shift 2;;
    --probe-cases)          PROBE_CASES="$2"; shift 2;;
    --pred-key)             PRED_KEY="$2"; shift 2;;
    --baseline-per-bubble)  BASELINE_PER_BUBBLE="$2"; shift 2;;
    --baseline-probes)      BASELINE_PROBES="$2"; shift 2;;
    -h|--help)              grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

if [[ -z "$PREDICTIONS" ]]; then
  echo "ERROR: --predictions is required" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"

# --- 0. filter gold to the ocr_clean subset --------------------------------
GOLD_CLEAN="$OUT_DIR/gold_ocr_clean.jsonl"
"$PY" - "$GOLD" "$GOLD_CLEAN" <<'PYEOF'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
n=0; nk=0
with open(src) as fi, open(dst, "w") as fo:
    for line in fi:
        line=line.strip()
        if not line: continue
        r=json.loads(line); n+=1
        if r.get("ocr_clean") is True:
            fo.write(json.dumps(r, ensure_ascii=False)+"\n"); nk+=1
print(f"[gold] {nk}/{n} ocr_clean rows -> {dst}")
PYEOF

echo
echo "=== 1. chrF++/BLEU on ocr_clean gold subset ($LABEL) ==="
"$PY" "$SCRIPT_DIR/score_jsonl_metrics.py" \
  --gold-jsonl "$GOLD_CLEAN" \
  --pred-jsonl "$PREDICTIONS" \
  --gold-ref-key en \
  --pred-key "$PRED_KEY" \
  --label "$LABEL" \
  --metrics "$METRICS" \
  --out-dir "$OUT_DIR"

PER_BUBBLE="$OUT_DIR/per_bubble_${LABEL}.json"

# --- 2. paired bootstrap (headline Δchrf + CI95 + p) -----------------------
echo
echo "=== 2. paired bootstrap (seed=12345) ==="
if [[ -z "$BASELINE_PER_BUBBLE" ]]; then
  echo "(no --baseline-per-bubble: self-comparing the run -> CI~0 smoke test)"
  BASELINE_PER_BUBBLE="$PER_BUBBLE"
fi
PAIRED_OUT="$OUT_DIR/paired_bs_${LABEL}.json"
"$PY" "$SCRIPT_DIR/paired_bs_metric.py" \
  --sys-a-per-bubble "$PER_BUBBLE" \
  --sys-b-per-bubble "$BASELINE_PER_BUBBLE" \
  --label-a "$LABEL" --label-b baseline \
  --metric-keys chrf_pp \
  --lower-is-better metricx_24_xl \
  --seed 12345 \
  --out "$PAIRED_OUT"

# --- 3. probes -------------------------------------------------------------
echo
echo "=== 3. probes (deterministic, seedless) ==="
PROBE_PREDS="$OUT_DIR/probe_preds_${LABEL}.jsonl"
"$PY" "$SCRIPT_DIR/prep_probe_predictions.py" \
  --predictions "$PREDICTIONS" \
  --probe-cases "$PROBE_CASES" \
  --pred-key "$PRED_KEY" \
  --out "$PROBE_PREDS"

PROBE_OUT="$OUT_DIR/probes_${LABEL}.json"
PROBE_ARGS=(--predictions "$PROBE_PREDS" --out "$PROBE_OUT")
if [[ -n "$BASELINE_PROBES" && -f "$BASELINE_PROBES" ]]; then
  PROBE_ARGS+=(--baseline "$BASELINE_PROBES")
fi
# probes.py exits non-zero when overall_pass is False; don't abort the run.
set +e
"$PY" -m backend.scripts.eval.probes "${PROBE_ARGS[@]}"
PROBE_EXIT=$?
set -e

# --- 4. verdict ------------------------------------------------------------
echo
echo "=== 4. VERDICT ($LABEL) ==="
"$PY" - "$PAIRED_OUT" "$PROBE_OUT" <<'PYEOF'
import json, sys
paired = json.load(open(sys.argv[1]))
probes = json.load(open(sys.argv[2]))

m = paired["by_metric"].get("chrf_pp", {})
ci_lo = m.get("ci95_low_delta", 0.0)
ci_hi = m.get("ci95_high_delta", 0.0)
delta = m.get("observed_delta", 0.0)
p = m.get("p_value_two_sided", 1.0)
ci_excludes_zero = (ci_lo > 0.0) or (ci_hi < 0.0)

regressions = probes.get("regressions_vs_baseline", {})
no_probe_regression = len(regressions) == 0

print(f"  Δchrf++          : {delta:+.3f}")
print(f"  CI95             : [{ci_lo:+.3f}, {ci_hi:+.3f}]  (excludes 0: {ci_excludes_zero})")
print(f"  p (two-sided)    : {p:.4f}")
print(f"  probe regressions: {regressions if regressions else 'none'}")
print()
if delta == 0.0 and ci_lo == 0.0 and ci_hi == 0.0:
    print("  RESULT: SELF-COMPARE (no baseline) -> determinism smoke test only.")
elif ci_excludes_zero and delta > 0 and no_probe_regression:
    print("  RESULT: PASS  (CI95 excludes 0 in the WIN direction AND no probe regressed)")
else:
    reasons = []
    if not ci_excludes_zero: reasons.append("CI95 includes 0 (not significant)")
    elif delta <= 0:         reasons.append("Δchrf++ not a win")
    if not no_probe_regression: reasons.append(f"probe regression: {list(regressions)}")
    print("  RESULT: FAIL  (" + "; ".join(reasons) + ")")
PYEOF

echo
echo "artifacts in: $OUT_DIR"
echo "  scorecard : $OUT_DIR/score_summary_metrics_v2_${LABEL}.json"
echo "  per-bubble: $PER_BUBBLE"
echo "  paired-bs : $PAIRED_OUT"
echo "  probes    : $PROBE_OUT"
