#!/usr/bin/env bash
# ============================================================================
# run_cert_gate.sh -- one-command certification gate for a candidate MT model.
#
# WHAT THIS IS
# ------------
# Mechanical aggregator over the SIX ship signals that already exist as separate
# CLIs. It runs (or reads cached results for) each, normalizes every signal to a
# PASS / FAIL / SKIP verdict, and prints one summary table + a machine-readable
# out/cert/cert_gate_<candidate>.json. It writes NO new eval logic -- every
# number comes from the underlying tool.
#
#   #  signal            underlying CLI                       gate criterion
#   -  ----------------  -----------------------------------  ------------------------------
#   1  POV (gender)      scripts.eval.pov_probe_v2 --json     axis_A_gender.inversion_rate
#                                                             <= --max-inversion-rate
#   2  refusal           scripts/eval/refusal_eval.py         gate_pass (refusal_rate==0)
#   3  chrF regression   scripts/eval/run_ikenie4_regression  paired_bs chrf_pp CI95
#                        .sh (paired_bs_*.json)               excludes 0 in WIN direction
#   4  page adequacy     scripts/eval/page_adequacy_judge.py  gate_pass (compare_gate)
#   5  image-prefix perf scripts/eval/bench_image_prefix.py   prefix_reuse_confirmed AND
#                                                             garbled_count == 0
#   6  L3 probes         scripts/eval/probes.py               overall_pass (exit 0)
#
# POV METRIC RETIREMENT (part a)
# ------------------------------
# The old pov_probe.py `gendered_rate` conflated gender (he<->she inversion) with
# person/register (pro-drop -> "I/you"), carried a 79% she-class prior, and
# counted label artifacts (see memory feedback_pov_metric_is_broken +
# scripts/eval/pov_probe_v2.py). This gate TRUSTS ONLY the corrected two-axis
# probe's Axis-A GENDER inversion rate for pass/fail. The Axis-B PERSON/REGISTER
# number is carried in the table as an informational column ("person=NN%"), never
# as a gate signal. `gendered_rate` is never read here.
#
# OFFLINE / DRY MODE
# ------------------
# Signals 2-6 need a live serve box (default http://100.64.235.63:8001/v1) and/or
# gitignored run artifacts (a translation inspect-dir, NAS page images). With
# --dry (or when a prerequisite is absent) those signals report SKIP with a
# reason; only the offline-capable POV probe runs against cached predictions. You
# can also feed any signal a cached report to score it without a live box:
#   --refusal-report --adequacy-report --paired-bs --probes-report --bench-report
#
# EXIT CODES
#   0  all evaluated signals PASS (and, unless --allow-skips, none were skipped)
#   1  at least one signal FAILed
#   2  no FAIL, but >=1 signal was SKIPped (certification INCOMPLETE)
#
# USAGE
#   run_cert_gate.sh --candidate v1 [--dry] [options]
#
#   --candidate <name>        model under test (default: v1). Selects cached POV
#                             preds out/pov/<name>__img-off.json and is passed to
#                             the live refusal/bench CLIs.
#   --dry                     offline: run only cache-backed signals; SKIP the rest.
#   --box-url <url>           serve box (default http://100.64.235.63:8001/v1).
#   --max-inversion-rate <f>  POV Axis-A pass threshold (default 0.05).
#   --testset <path>          POV testset (default backend/.bench/pov_ab/testset_large.json).
#   --refusal-report <json>   score cached refusal report instead of live gen.
#   --adequacy-report <json>  score cached page-adequacy compare_gate json.
#   --paired-bs <json>        score cached run_ikenie4 paired_bs_*.json (chrF signal).
#   --probes-report <json>    score cached probes.py report (overall_pass).
#   --bench-report <json>     score cached bench_image_prefix.py json.
#   --inspect-dir <dir>       LIVE chrF+probes: run run_ikenie4_regression.sh on this
#                             translation inspect-dir (drives signals 3 AND 6).
#   --baseline-per-bubble <json>  baseline for the paired bootstrap (chrF signal).
#   --out-dir <dir>           artifact dir (default backend/scripts/eval/out/cert).
#   --allow-skips             exit 0 even if some signals were SKIPped.
#   -h | --help               print this header and exit.
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# --- interpreter: prefer the backend venv, fall back to python3 --------------
if [[ -x "$BACKEND_DIR/.venv/bin/python" ]]; then
  PY="$BACKEND_DIR/.venv/bin/python"
else
  PY="$(command -v python3 || command -v python)"
fi

# --- defaults ----------------------------------------------------------------
CANDIDATE="v1"
DRY=0
BOX_URL="http://100.64.235.63:8001/v1"
MAX_INVERSION_RATE="0.05"
TESTSET="$BACKEND_DIR/.bench/pov_ab/testset_large.json"
OUT_DIR="$SCRIPT_DIR/out/cert"
REFUSAL_REPORT=""
ADEQUACY_REPORT=""
PAIRED_BS=""
PROBES_REPORT=""
BENCH_REPORT=""
INSPECT_DIR=""
BASELINE_PER_BUBBLE=""
ALLOW_SKIPS=0

# --- arg parse ---------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --candidate)            CANDIDATE="$2"; shift 2;;
    --dry|--offline)        DRY=1; shift;;
    --box-url)              BOX_URL="$2"; shift 2;;
    --max-inversion-rate)   MAX_INVERSION_RATE="$2"; shift 2;;
    --testset)              TESTSET="$2"; shift 2;;
    --refusal-report)       REFUSAL_REPORT="$2"; shift 2;;
    --adequacy-report)      ADEQUACY_REPORT="$2"; shift 2;;
    --paired-bs)            PAIRED_BS="$2"; shift 2;;
    --probes-report)        PROBES_REPORT="$2"; shift 2;;
    --bench-report)         BENCH_REPORT="$2"; shift 2;;
    --inspect-dir)          INSPECT_DIR="$2"; shift 2;;
    --baseline-per-bubble)  BASELINE_PER_BUBBLE="$2"; shift 2;;
    --out-dir)              OUT_DIR="$2"; shift 2;;
    --allow-skips)          ALLOW_SKIPS=1; shift;;
    -h|--help)              grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 64;;
  esac
done

mkdir -p "$OUT_DIR"
RESULTS_TSV="$(mktemp)"   # NAME<TAB>STATUS<TAB>METRIC<TAB>DETAIL
trap 'rm -f "$RESULTS_TSV"' EXIT

# --- helpers -----------------------------------------------------------------
# read a nested JSON field: jget <file> <dotted.path> [default]
jget() {
  "$PY" - "$1" "$2" "${3:-}" <<'PYEOF'
import json, sys
path = sys.argv[2].split(".") if sys.argv[2] else []
default = sys.argv[3]
try:
    with open(sys.argv[1]) as f:
        d = json.load(f)
    for k in path:
        d = d[int(k)] if isinstance(d, list) else d[k]
    print(d if d is not None else default)
except Exception:
    print(default)
PYEOF
}

record() {  # record <name> <status> <metric> <detail>
  printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" >> "$RESULTS_TSV"
}

# box reachability (short timeout); returns 0 if the /models endpoint answers.
box_reachable() {
  curl -fsS --max-time 4 "${BOX_URL%/}/models" >/dev/null 2>&1
}

echo "==============================================================================" >&2
echo " CERT GATE  candidate=$CANDIDATE  mode=$([[ $DRY -eq 1 ]] && echo DRY-offline || echo live)" >&2
echo " box=$BOX_URL  python=$PY" >&2
echo "==============================================================================" >&2

LIVE_OK=0
if [[ $DRY -eq 0 ]] && box_reachable; then LIVE_OK=1; fi
if [[ $DRY -eq 0 && $LIVE_OK -eq 0 ]]; then
  echo "[cert] serve box unreachable ($BOX_URL) -> live signals will SKIP unless a cached report is supplied." >&2
fi

# ============================================================================
# SIGNAL 1 -- POV (Axis-A gender inversion; corrected two-axis probe)
# ============================================================================
POV_PREDS="$SCRIPT_DIR/out/pov/${CANDIDATE}__img-off.json"
POV_OUT="$OUT_DIR/pov_v2_${CANDIDATE}.json"
if [[ ! -f "$TESTSET" ]]; then
  record "1.POV(gender)" "SKIP" "-" "testset absent: $TESTSET (gitignored)"
elif [[ ! -f "$POV_PREDS" ]]; then
  record "1.POV(gender)" "SKIP" "-" "cached preds absent: out/pov/${CANDIDATE}__img-off.json"
else
  # pov_probe_v2 is read-only over the cached img-off predictions; --json only.
  # NOTE: the probe reads the canonical .bench/pov_ab/testset_large.json path
  # itself, so --testset is used only for the prerequisite existence check above.
  if ( cd "$BACKEND_DIR" && "$PY" -m scripts.eval.pov_probe_v2 --json ) > "$POV_OUT" 2>"$OUT_DIR/pov_v2_${CANDIDATE}.err"; then
    INV_RATE="$(jget "$POV_OUT" axis_A_gender.inversion_rate 1.0)"
    N_INV="$(jget "$POV_OUT" axis_A_gender.n_inversions 0)"
    N="$(jget "$POV_OUT" axis_A_gender.n 0)"
    PERSON="$(jget "$POV_OUT" axis_B_person.person_rate 0)"
    PASS="$("$PY" -c "import sys;print('1' if float('$INV_RATE')<=float('$MAX_INVERSION_RATE') else '0')" 2>/dev/null || echo 0)"
    PCT="$("$PY" -c "print(f'{float(\"$INV_RATE\")*100:.1f}')" 2>/dev/null || echo '?')"
    PPCT="$("$PY" -c "print(f'{float(\"$PERSON\")*100:.1f}')" 2>/dev/null || echo '?')"
    if [[ "$PASS" == "1" ]]; then STATUS="PASS"; else STATUS="FAIL"; fi
    record "1.POV(gender)" "$STATUS" "inv=${N_INV}/${N} (${PCT}%)" "person/register=${PPCT}% (informational, not gated); thr<=$MAX_INVERSION_RATE"
  else
    record "1.POV(gender)" "SKIP" "-" "pov_probe_v2 errored (see pov_v2_${CANDIDATE}.err)"
  fi
fi

# ============================================================================
# SIGNAL 2 -- refusal (HARD GATE refusal_rate == 0)
# ============================================================================
if [[ -n "$REFUSAL_REPORT" && -f "$REFUSAL_REPORT" ]]; then
  GP="$(jget "$REFUSAL_REPORT" gate_pass false)"
  RR="$(jget "$REFUSAL_REPORT" refusal_rate '?')"
  CONF="$(jget "$REFUSAL_REPORT" confirmed_refusals '?')"
  [[ "$GP" == "True" || "$GP" == "true" ]] && STATUS="PASS" || STATUS="FAIL"
  record "2.refusal" "$STATUS" "rate=${RR} conf=${CONF}" "cached: $(basename "$REFUSAL_REPORT")"
elif [[ $DRY -eq 1 || $LIVE_OK -eq 0 ]]; then
  record "2.refusal" "SKIP" "-" "needs live box or --refusal-report"
else
  ROUT="$OUT_DIR/refusal_${CANDIDATE}.json"
  if ( cd "$BACKEND_DIR" && "$PY" scripts/eval/refusal_eval.py --model "$CANDIDATE" --gen-base-url "$BOX_URL" --judge-base-url "$BOX_URL" --out-dir "$OUT_DIR" ) >"$OUT_DIR/refusal_${CANDIDATE}.log" 2>&1; then
    STATUS="PASS"
  else
    [[ $? -eq 3 ]] && STATUS="FAIL" || STATUS="SKIP"
  fi
  # locate the freshest emitted report for the metric column
  LATEST="$(ls -t "$OUT_DIR"/refusal_report_${CANDIDATE}_*.json 2>/dev/null | head -1)"
  RR="?"; CONF="?"
  [[ -n "$LATEST" ]] && { RR="$(jget "$LATEST" refusal_rate '?')"; CONF="$(jget "$LATEST" confirmed_refusals '?')"; }
  record "2.refusal" "$STATUS" "rate=${RR} conf=${CONF}" "live --model $CANDIDATE"
fi

# ============================================================================
# SIGNAL 3 -- chrF regression (paired bootstrap CI95 excludes 0 in win dir)
# ============================================================================
eval_paired_bs() {  # $1 = paired_bs json
  local f="$1"
  local lo hi delta p
  lo="$(jget "$f" by_metric.chrf_pp.ci95_low_delta 0)"
  hi="$(jget "$f" by_metric.chrf_pp.ci95_high_delta 0)"
  delta="$(jget "$f" by_metric.chrf_pp.observed_delta 0)"
  p="$(jget "$f" by_metric.chrf_pp.p_value_two_sided 1)"
  local verdict
  verdict="$("$PY" -c "
lo,hi,d=float('$lo'),float('$hi'),float('$delta')
if lo==0.0 and hi==0.0 and d==0.0: print('SKIP')          # self-compare / no baseline
elif lo>0.0 and d>0.0: print('PASS')                        # CI excludes 0 in win dir
elif hi<0.0: print('FAIL')                                  # CI excludes 0, regression
else: print('FAIL')                                         # CI includes 0 -> not significant
" 2>/dev/null || echo SKIP)"
  record "3.chrF-regr" "$verdict" "d=${delta} CI[${lo},${hi}]" "p=${p} $([[ "$verdict" == SKIP ]] && echo '(self-compare/no baseline)' || true)"
}
if [[ -n "$PAIRED_BS" && -f "$PAIRED_BS" ]]; then
  eval_paired_bs "$PAIRED_BS"
elif [[ -n "$INSPECT_DIR" && -d "$INSPECT_DIR" && $DRY -eq 0 ]]; then
  # LIVE: run_ikenie4_regression.sh drives BOTH chrF (signal 3) and probes (signal 6).
  RIK_ARGS=(--inspect-dir "$INSPECT_DIR" --label "cert_${CANDIDATE}" --out-dir "$OUT_DIR/ikenie4")
  [[ -n "$BASELINE_PER_BUBBLE" ]] && RIK_ARGS+=(--baseline-per-bubble "$BASELINE_PER_BUBBLE")
  ( cd "$BACKEND_DIR" && PY="$PY" "$SCRIPT_DIR/run_ikenie4_regression.sh" "${RIK_ARGS[@]}" ) >"$OUT_DIR/ikenie4_${CANDIDATE}.log" 2>&1 || true
  RIK_PAIRED="$OUT_DIR/ikenie4/paired_bs_cert_${CANDIDATE}.json"
  RIK_PROBES="$OUT_DIR/ikenie4/probes_cert_${CANDIDATE}.json"
  if [[ -f "$RIK_PAIRED" ]]; then eval_paired_bs "$RIK_PAIRED"; else
    record "3.chrF-regr" "SKIP" "-" "run_ikenie4 produced no paired_bs (see ikenie4_${CANDIDATE}.log)"; fi
  # signal 6 consumes RIK_PROBES below if present.
  PROBES_REPORT="${PROBES_REPORT:-$RIK_PROBES}"
else
  record "3.chrF-regr" "SKIP" "-" "needs --paired-bs or (live) --inspect-dir"
fi

# ============================================================================
# SIGNAL 4 -- page adequacy (compare_gate gate_pass)
# ============================================================================
if [[ -n "$ADEQUACY_REPORT" && -f "$ADEQUACY_REPORT" ]]; then
  # the compare gate is emitted under either the top level or a "gate" key.
  GP="$(jget "$ADEQUACY_REPORT" gate.gate_pass "$(jget "$ADEQUACY_REPORT" gate_pass '?')")"
  DA="$(jget "$ADEQUACY_REPORT" gate.delta_adequacy "$(jget "$ADEQUACY_REPORT" delta_adequacy '?')")"
  if [[ "$GP" == "True" || "$GP" == "true" ]]; then STATUS="PASS"
  elif [[ "$GP" == "False" || "$GP" == "false" ]]; then STATUS="FAIL"
  else STATUS="SKIP"; fi
  record "4.adequacy" "$STATUS" "Δadeq=${DA}" "cached: $(basename "$ADEQUACY_REPORT")"
else
  record "4.adequacy" "SKIP" "-" "needs live VLM judge + page images, or --adequacy-report"
fi

# ============================================================================
# SIGNAL 5 -- image-prefix perf (prefix_reuse_confirmed AND garbled_count==0)
# ============================================================================
eval_bench() {  # $1 = bench json
  local f="$1"
  local reuse garbled mism
  reuse="$(jget "$f" prefix_reuse_confirmed false)"
  garbled="$(jget "$f" correctness.garbled_count 0)"
  mism="$(jget "$f" correctness.concurrent_vs_sequential_mismatches 0)"
  if { [[ "$reuse" == "True" || "$reuse" == "true" ]]; } && [[ "$garbled" == "0" ]]; then
    STATUS="PASS"; else STATUS="FAIL"; fi
  record "5.img-prefix" "$STATUS" "reuse=${reuse} garbled=${garbled}" "mismatches=${mism}"
}
if [[ -n "$BENCH_REPORT" && -f "$BENCH_REPORT" ]]; then
  eval_bench "$BENCH_REPORT"
elif [[ $DRY -eq 1 || $LIVE_OK -eq 0 ]]; then
  record "5.img-prefix" "SKIP" "-" "needs live vLLM + NAS page images, or --bench-report"
else
  BOUT="$OUT_DIR/bench_prefix_${CANDIDATE}.json"
  if ( cd "$BACKEND_DIR" && "$PY" scripts/eval/bench_image_prefix.py --base-url "$BOX_URL" --model "$CANDIDATE" --image on --order concurrent --out "$BOUT" ) >"$OUT_DIR/bench_${CANDIDATE}.log" 2>&1 && [[ -f "$BOUT" ]]; then
    eval_bench "$BOUT"
  else
    record "5.img-prefix" "SKIP" "-" "bench errored (NAS not mounted? see bench_${CANDIDATE}.log)"
  fi
fi

# ============================================================================
# SIGNAL 6 -- L3 probes (overall_pass)
# ============================================================================
if [[ -n "$PROBES_REPORT" && -f "$PROBES_REPORT" ]]; then
  OP="$(jget "$PROBES_REPORT" overall_pass false)"
  REG="$(jget "$PROBES_REPORT" regressions_vs_baseline '{}')"
  if [[ "$OP" == "True" || "$OP" == "true" ]]; then STATUS="PASS"; else STATUS="FAIL"; fi
  record "6.probes" "$STATUS" "overall_pass=${OP}" "regressions=${REG}"
else
  record "6.probes" "SKIP" "-" "needs --probes-report or (live) --inspect-dir"
fi

# ============================================================================
# SUMMARY TABLE + machine-readable json + exit code
# ============================================================================
N_PASS=$(awk -F'\t' '$2=="PASS"' "$RESULTS_TSV" | wc -l | tr -d ' ')
N_FAIL=$(awk -F'\t' '$2=="FAIL"' "$RESULTS_TSV" | wc -l | tr -d ' ')
N_SKIP=$(awk -F'\t' '$2=="SKIP"' "$RESULTS_TSV" | wc -l | tr -d ' ')

echo
printf '%-16s %-6s %-28s %s\n' "SIGNAL" "STATUS" "METRIC" "DETAIL"
printf '%s\n' "--------------------------------------------------------------------------------"
while IFS=$'\t' read -r name status metric detail; do
  printf '%-16s %-6s %-28s %s\n' "$name" "$status" "$metric" "$detail"
done < "$RESULTS_TSV"
printf '%s\n' "--------------------------------------------------------------------------------"
printf 'PASS=%s  FAIL=%s  SKIP=%s   candidate=%s\n' "$N_PASS" "$N_FAIL" "$N_SKIP" "$CANDIDATE"

# emit machine-readable rollup
CERT_JSON="$OUT_DIR/cert_gate_${CANDIDATE}.json"
"$PY" - "$RESULTS_TSV" "$CERT_JSON" "$CANDIDATE" "$N_PASS" "$N_FAIL" "$N_SKIP" <<'PYEOF'
import json, sys
tsv, out, cand, npass, nfail, nskip = sys.argv[1:7]
signals = []
with open(tsv) as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 4: continue
        signals.append({"signal": parts[0], "status": parts[1],
                        "metric": parts[2], "detail": parts[3]})
rollup = {"candidate": cand,
          "n_pass": int(npass), "n_fail": int(nfail), "n_skip": int(nskip),
          "certified": int(nfail) == 0 and int(nskip) == 0,
          "signals": signals}
with open(out, "w") as f:
    json.dump(rollup, f, ensure_ascii=False, indent=2)
print(f"[cert] wrote {out}")
PYEOF

if [[ "$N_FAIL" -gt 0 ]]; then
  echo "VERDICT: FAIL ($N_FAIL signal(s) failed)"
  exit 1
elif [[ "$N_SKIP" -gt 0 && "$ALLOW_SKIPS" -eq 0 ]]; then
  echo "VERDICT: INCOMPLETE ($N_SKIP signal(s) skipped -- not certified; rerun with a live box / cached reports, or --allow-skips)"
  exit 2
else
  echo "VERDICT: PASS (all evaluated signals green)"
  exit 0
fi
