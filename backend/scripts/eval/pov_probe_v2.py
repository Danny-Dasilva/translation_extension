"""POV probe v2 — two-axis (GENDER vs PERSON/REGISTER), curated & artifact-filtered.

Motivation
----------
The original ``pov_probe.py`` fuses two distinct questions into one "gendered
pass" number: (A) did the model commit the WRONG GENDER (he<->she inversion),
and (B) did the model render a Japanese pro-drop line as 1st/2nd person ("I"/
"you") where the human scanlator wrote a 3rd-person NARRATION caption ("she"/
"he").  On the Furube holdout, (B) dominates (a rendering-style choice), a 79%
"she" class prior beats the 48% gate on its own, and ~10% of the "gendered"
rows are label artifacts (laughter tokens, mega-afterword blobs, misaligned
short-JP rows).  So the fused metric hallucinated a gender failure.

This probe separates the axes and scores each where it is well-posed.  It REUSES
the scorer primitives from ``pov_probe.py`` (``detect_families``,
``required_family``, ``pov_pass``) verbatim — it does not re-implement or edit
them.

    Axis A  GENDER      inversion-only, on a curated gender-OBLIGATORY,
                        gender-RECOVERABLE, artifact-filtered, class-balanced set.
    Axis B  PERSON      how often the model renders 1st/2nd person where the gold
                        caption is 3rd-person narration.  Reported separately.

Read-only: consumes cached v1 predictions from
``scripts/eval/out/pov/v1__img-off.json`` (v1 is text-only; the img-off arm is the
faithful serve).  Falls back to the live box only with --generate.

    python -m backend.scripts.eval.pov_probe_v2            # score cached v1
    python -m backend.scripts.eval.pov_probe_v2 --dump     # + per-row detail
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve()
_BACKEND_ROOT = _HERE.parents[2]  # .../backend
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

# REUSE the scorer primitives — do NOT re-implement.
from scripts.eval.pov_probe import (  # noqa: E402
    detect_families,
    required_family,
    pov_pass,
)

TESTSET_PATH = _BACKEND_ROOT / ".bench" / "pov_ab" / "testset_large.json"
V1_PRED_PATH = _BACKEND_ROOT / "scripts" / "eval" / "out" / "pov" / "v1__img-off.json"

# --------------------------------------------------------------------------- #
# Artifact filters (diagnosis §5A)
# --------------------------------------------------------------------------- #

# Laughter / onomatopoeia — JP source side.
_JP_LAUGH_RE = re.compile(r"(ふ{2,}|フ{2,}|ﾌ{2,}|く{3,}|ク{3,}|ｸ{3,}|は{3,}|ハ{3,})")
# Laughter runs in the EN side that trip \bhe\b / \bha\b:  "HE--HE--HE", "HEHEHE",
# "HA-HA", "HE HE HE".  Strip before detecting pronoun families.
_EN_LAUGH_RE = re.compile(
    r"\b(?:(?:he|ha|hee|haa|fu|ku)[\s\-—–_]*){2,}\b", re.IGNORECASE
)

MEGA_CAPTION_CHARS = 300   # diagnosis §5A; the 2 afterword blobs are 1564 / 1631
SHORT_JP_CHARS = 6         # diagnosis §5A misaligned-row heuristic


def strip_laughter_en(text: str) -> str:
    return _EN_LAUGH_RE.sub(" ", text or "")


def has_jp_laughter(jp: str) -> bool:
    return bool(_JP_LAUGH_RE.search(jp or ""))


# --------------------------------------------------------------------------- #
# Gender recoverability (diagnosis §3) — strong, gender-SPECIFIC markers only.
# --------------------------------------------------------------------------- #
# Ambiguous honorifics (さん/様/ちゃん) are intentionally excluded: they do not
# by themselves fix a gender.  くん leans male but is excluded as too weak.
_FEMALE_MARKERS = [
    "彼女", "母", "姉", "妹", "娘", "嫁", "妻", "女の子", "女性", "女子",
    "母親", "少女", "おかあ", "かあさん", "ママ", "お姉", "姫", "婦",
]
_MALE_MARKERS = [
    "父", "兄", "弟", "息子", "夫", "婿", "男の子", "男性", "男子",
    "父親", "少年", "おとう", "とうさん", "パパ", "お兄", "坊", "紳士",
]
# 彼 is male but 彼女 is female — handle 彼女 first, then bare 彼.
_BARE_KARE_RE = re.compile(r"彼(?!女)")


def marker_genders(text: str) -> set[str]:
    """Return the set of strong gender markers present: subset of {'he','she'}."""
    t = text or ""
    g: set[str] = set()
    if any(m in t for m in _FEMALE_MARKERS):
        g.add("she")
    if any(m in t for m in _MALE_MARKERS):
        g.add("he")
    if _BARE_KARE_RE.search(t):
        g.add("he")
    return g


def recoverable_gender(jp_target: str, context_lines: list[str]) -> tuple[set[str], str]:
    """Gender inferable from the JP target line and/or the page context.

    Returns (genders, source) where source in {'target','context','none'}.
    """
    tgt = marker_genders(jp_target)
    if tgt:
        return tgt, "target"
    ctx = marker_genders(" ".join(context_lines or []))
    if ctx:
        return ctx, "context"
    return set(), "none"


# --------------------------------------------------------------------------- #
# Person detection (Axis B) — 1st / 2nd person in EN.
# --------------------------------------------------------------------------- #
_FIRST_RE = re.compile(r"\b(i|i'm|i've|i'll|i'd|me|my|mine|myself|we|our|us)\b", re.IGNORECASE)
_SECOND_RE = re.compile(r"\b(you|you're|your|yours|yourself|yourselves)\b", re.IGNORECASE)


def persons_present(text: str) -> set[str]:
    p: set[str] = set()
    if _FIRST_RE.search(text or ""):
        p.add("1")
    if _SECOND_RE.search(text or ""):
        p.add("2")
    return p


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #


def load_testset() -> list[dict[str, Any]]:
    return json.loads(TESTSET_PATH.read_text(encoding="utf-8"))


def load_v1_predictions() -> dict[str, str]:
    if not V1_PRED_PATH.exists():
        raise SystemExit(f"cached v1 predictions not found: {V1_PRED_PATH}")
    data = json.loads(V1_PRED_PATH.read_text(encoding="utf-8"))
    return {r["src"]: (r.get("prediction") or "") for r in data["predictions"]}


# --------------------------------------------------------------------------- #
# Curation + scoring
# --------------------------------------------------------------------------- #


def curate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the curated GENDER-obligatory subset and record filter drops."""
    drops = Counter()
    curated: list[dict[str, Any]] = []
    gendered_resolvable = 0

    for r in rows:
        human_en = r["human_en"]
        jp = r["jp"]
        ctx = r.get("context", [])

        # required_family on laughter-stripped human_en (kills the HE-HE FP at the
        # resolvability stage too).
        human_clean = strip_laughter_en(human_en)
        req = required_family(human_clean)

        # Also compute the raw req (pre-strip) to detect laughter FPs explicitly.
        req_raw = required_family(human_en)

        if req_raw is not None:
            gendered_resolvable += 1

        # --- artifact filters ---
        if req_raw is not None and req is None:
            # laughter run WAS the only "pronoun" -> it evaporated after stripping
            drops["laughter_fp"] += 1
            continue
        if has_jp_laughter(jp) and req_raw is not None:
            # JP-side laughter source (ふふ etc.) on a "gendered" row -> artifact
            drops["jp_laughter"] += 1
            continue
        if req is None:
            continue  # ungendered / mixed -> not a GENDER-axis row

        if len(human_en) > MEGA_CAPTION_CHARS:
            drops["mega_caption"] += 1
            continue
        if len(jp.strip()) < SHORT_JP_CHARS:
            drops["short_jp_misaligned"] += 1
            continue

        # --- recoverability gate ---
        rec_g, src = recoverable_gender(jp, ctx)
        if not rec_g:
            drops["not_recoverable"] += 1
            continue
        if req not in rec_g:
            # marker gender contradicts the reference family -> ambiguous / bad row
            drops["marker_contradicts_ref"] += 1
            continue

        curated.append(
            {
                "src": r["src"],
                "work": r["work"],
                "jp": jp,
                "human_en": human_en,
                "req_fam": req,
                "rec_source": src,
                "rec_genders": sorted(rec_g),
            }
        )

    return {
        "gendered_resolvable": gendered_resolvable,
        "drops": dict(drops),
        "curated": curated,
    }


def score_gender_axis(curated: list[dict[str, Any]], preds: dict[str, str]) -> dict[str, Any]:
    """Axis A: he<->she inversion-only on the curated obligatory subset."""
    per_class = {"he": Counter(), "she": Counter()}
    inversions: list[dict[str, Any]] = []
    correct: list[dict[str, Any]] = []
    evasions: list[dict[str, Any]] = []

    for row in curated:
        req = row["req_fam"]
        pred_raw = preds.get(row["src"], "")
        pred = strip_laughter_en(pred_raw)  # symmetric laughter strip on prediction
        fams = detect_families(pred)
        opposing = "she" if req == "he" else "he"

        per_class[req]["n"] += 1
        rec = {**row, "prediction": pred_raw}
        if opposing in fams and req not in fams:
            per_class[req]["inversion"] += 1
            inversions.append(rec)
        elif opposing in fams and req in fams:
            # both present — ambiguous; count as inversion-ish (asserted wrong too)
            per_class[req]["both"] += 1
            inversions.append({**rec, "note": "both_families"})
        elif req in fams:
            per_class[req]["correct"] += 1
            correct.append(rec)
        else:
            per_class[req]["evasion"] += 1  # no gendered pronoun -> person choice
            evasions.append(rec)

    n = len(curated)
    n_inv = sum(len(inversions) for _ in [0])
    return {
        "n": n,
        "per_class": {k: dict(v) for k, v in per_class.items()},
        "inversions": inversions,
        "n_inversions": len(inversions),
        "n_correct": len(correct),
        "n_evasion": len(evasions),
        "inversion_rate": len(inversions) / n if n else 0.0,
        "correct_rate": len(correct) / n if n else 0.0,
        "evasion_rate": len(evasions) / n if n else 0.0,
    }


def score_person_axis(rows: list[dict[str, Any]], preds: dict[str, str]) -> dict[str, Any]:
    """Axis B: on 3rd-person NARRATION captions (gold has a gendered pronoun and
    NO 1st/2nd person), how often does the model render 1st/2nd person instead?"""
    narration: list[dict[str, Any]] = []
    for r in rows:
        human_clean = strip_laughter_en(r["human_en"])
        req = required_family(human_clean)
        if req is None:
            continue
        if len(r["human_en"]) > MEGA_CAPTION_CHARS or len(r["jp"].strip()) < SHORT_JP_CHARS:
            continue
        gold_persons = persons_present(r["human_en"])
        if gold_persons:
            continue  # gold already mixes I/you -> not pure narration
        narration.append(r)

    n = len(narration)
    person_rendered = 0
    third_person = 0
    detail: list[dict[str, Any]] = []
    for r in narration:
        pred = strip_laughter_en(preds.get(r["src"], ""))
        pp = persons_present(pred)
        if pp:
            person_rendered += 1
            detail.append({"src": r["src"], "persons": sorted(pp),
                           "human_en": r["human_en"][:80], "prediction": pred[:80]})
        else:
            third_person += 1
    return {
        "narration_n": n,
        "person_rendered": person_rendered,
        "third_person_kept": third_person,
        "person_rate": person_rendered / n if n else 0.0,
        "detail": detail,
    }


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="POV probe v2 (two-axis, curated).")
    ap.add_argument("--dump", action="store_true", help="Print per-row detail.")
    ap.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = ap.parse_args(argv)

    rows = load_testset()
    preds = load_v1_predictions()

    cur = curate(rows)
    curated = cur["curated"]
    gender = score_gender_axis(curated, preds)
    person = score_person_axis(rows, preds)

    # class balance: report per-class n; also a class-balanced inversion rate.
    pc = gender["per_class"]
    bal_terms = []
    for fam in ("he", "she"):
        nn = pc[fam].get("n", 0)
        inv = pc[fam].get("inversion", 0) + pc[fam].get("both", 0)
        bal_terms.append(inv / nn if nn else 0.0)
    balanced_inv = sum(bal_terms) / 2

    report = {
        "source": {
            "testset": str(TESTSET_PATH),
            "v1_preds": str(V1_PRED_PATH),
            "n_testset": len(rows),
        },
        "curation": {
            "gendered_resolvable_raw": cur["gendered_resolvable"],
            "drops": cur["drops"],
            "curated_n": len(curated),
            "curated_class_n": {k: pc[k].get("n", 0) for k in ("he", "she")},
        },
        "axis_A_gender": gender,
        "axis_A_balanced_inversion_rate": balanced_inv,
        "axis_B_person": {k: v for k, v in person.items() if k != "detail"},
    }

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    print("=" * 72)
    print("POV PROBE v2 — v1 (text-only, cached img-off arm)")
    print("=" * 72)
    print(f"testset rows: {len(rows)}   v1 preds: {V1_PRED_PATH.name}")
    print()
    print("--- CURATION (Axis-A gender-obligatory subset) ---")
    print(f"gendered-resolvable (raw, old denominator): {cur['gendered_resolvable']}")
    print("artifact/recoverability drops:")
    for k, v in cur["drops"].items():
        print(f"    {k:>24}: {v}")
    print(f"curated GENDER subset n = {len(curated)}   "
          f"(he={pc['he'].get('n',0)}, she={pc['she'].get('n',0)})")
    print()
    print("--- AXIS A: GENDER (he<->she inversion only) ---")
    print(f"  inversions (wrong gender asserted): {gender['n_inversions']}/{gender['n']} "
          f"= {gender['inversion_rate']*100:.1f}%")
    print(f"  class-balanced inversion rate:      {balanced_inv*100:.1f}%")
    print(f"  correct gender asserted:            {gender['n_correct']}/{gender['n']} "
          f"= {gender['correct_rate']*100:.1f}%")
    print(f"  evasion (no gendered pronoun):      {gender['n_evasion']}/{gender['n']} "
          f"= {gender['evasion_rate']*100:.1f}%")
    for fam in ("he", "she"):
        c = pc[fam]
        print(f"    [{fam}] n={c.get('n',0)} correct={c.get('correct',0)} "
              f"inversion={c.get('inversion',0)} both={c.get('both',0)} "
              f"evasion={c.get('evasion',0)}")
    print()
    if gender["inversions"]:
        print("  GENUINE INVERSIONS (inspect each):")
        for inv in gender["inversions"]:
            print(f"    - {inv['src']} req={inv['req_fam']} {inv.get('note','')}")
            print(f"        JP:   {inv['jp'][:70]}")
            print(f"        gold: {inv['human_en'][:90]}")
            print(f"        v1:   {inv['prediction'][:90]}")
    else:
        print("  GENUINE INVERSIONS: 0")
    print()
    print("--- AXIS B: PERSON / REGISTER (separate; NOT folded into gender) ---")
    print(f"  3rd-person narration captions (gold): {person['narration_n']}")
    print(f"  model rendered 1st/2nd person:        {person['person_rendered']}/"
          f"{person['narration_n']} = {person['person_rate']*100:.1f}%")
    print(f"  model kept 3rd-person:                {person['third_person_kept']}")
    print()

    if args.dump:
        print("--- curated rows ---")
        for row in curated:
            print(f"  {row['src']} req={row['req_fam']} rec={row['rec_source']}"
                  f"{row['rec_genders']}")
            print(f"      gold: {row['human_en'][:90]}")
            print(f"      v1:   {preds.get(row['src'],'')[:90]}")
        print("\n--- Axis-B person-rendered detail ---")
        for d in person["detail"]:
            print(f"  {d['src']} {d['persons']}")
            print(f"      gold: {d['human_en']}")
            print(f"      v1:   {d['prediction']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
