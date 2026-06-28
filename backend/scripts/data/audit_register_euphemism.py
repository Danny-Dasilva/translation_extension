#!/usr/bin/env python3
"""Audit a CPO/DPO preference parquet for euphemistic CHOSEN translations.

See ``thoughts/shared/plans/fix8-register-faithfulness-audit.md`` for the full
methodology and rationale.

WHY THIS EXISTS
---------------
On a hardcore NSFW title the v11 model SOFTENS crude Japanese into polite English
(精液->"the liquid", 喘ぎ声->"calling out", 顔射->"spray", おかわり->"another one,
please", マンコ->euphemism). The direction is ALWAYS softer, never cruder.

Root cause (verified): preference pairs are built with
``chosen = argmax(COMET-Kiwi score)``. COMET-Kiwi rewards fluency/adequacy, NOT
explicit register, so a fluent euphemism can outrank a crude-but-faithful
rendering and win the ``chosen`` slot — training the model toward softening.

This tool FLAGS the suspect rows so they can be repaired (swap chosen<->rejected
to the faithful candidate, or drop the row). It does NOT mutate the dataset.

DO NOT use this to justify re-OVERSAMPLING NSFW data: the prior v12 attempt
oversampled NSFW DPO/chat to ~36% and REGRESSED register (learned euphemism).
The remedy is cleaning the chosen side, not adding volume.

PRECISION OVER RECALL
---------------------
Mirror ``app/services/register_glossary.py``: every explicit term needs
``jp_excludes`` guards. Naive substring match is noisy — e.g. イク matches マイク
(microphone) / ジェイク (Jake); 潮 matches 潮干狩り (clam digging), 紅潮 (blush),
最高潮 (climax/peak), 風潮 (trend), 渦潮 (whirlpool). Better to miss a row than to
mislabel a clean one. Start SMALL (the confirmed Ikenie terms), grow only on
confirmed (jp, euphemism->crude) mappings.

USAGE
-----
    python backend/scripts/data/audit_register_euphemism.py \\
        --parquet backend/scripts/data/cpo/v10_5_preferences.parquet \\
        --out backend/scripts/data/cpo/register_euphemism_flags.jsonl \\
        --report backend/scripts/data/register_audit_findings.md

Schema expected (verified on v10_5_preferences.parquet, 11,901 rows):
    [prompt, chosen, rejected, chosen_score, rejected_score, margin,
     chosen_kind, rejected_kind, src]
The JP source is embedded in ``prompt`` after the literal "Japanese: ".
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

# NOTE: pandas/polars not imported at module top so the file imports cleanly in
# environments without them. Import inside load_preferences().

# Default target (gitignored; lives in the main checkout, not in worktrees).
DEFAULT_PARQUET = "backend/scripts/data/cpo/v10_5_preferences.parquet"

# Pulls the JP source back out of the stored user-message prompt.
_JP_RE = re.compile(r"Japanese:\s*(.*)\Z", re.S)


@dataclass(frozen=True)
class RegisterTerm:
    """One explicit-term audit rule (mirrors register_glossary.RegisterEntry).

    A CHOSEN translation is flagged as EUPHEMISTIC for this term when the source
    contains a ``jp_term`` (and no ``jp_excludes``) AND the chosen text either
    contains a known ``euphemism`` OR contains NONE of the ``crude_en`` faithful
    renderings (the explicit concept was softened/dropped).
    """

    label: str                                   # human label, e.g. "semen"
    jp_terms: List[str]                          # crude source tokens (精液, ザーメン)
    crude_en: List[str]                          # faithful EN (cum, semen, load)
    euphemisms: List[str]                        # softened EN we treat as FAILURE
    jp_excludes: List[str] = field(default_factory=list)  # disarm false positives
    # Pre-compiled whole-"word" patterns for crude_en + euphemisms. Built lazily.
    _crude_pats: List[re.Pattern] = field(default_factory=list, compare=False, repr=False)
    _euph_pats: List[re.Pattern] = field(default_factory=list, compare=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "_crude_pats", [_compile_word(w) for w in self.crude_en])
        object.__setattr__(self, "_euph_pats", [_compile_word(w) for w in self.euphemisms])


def _compile_word(needle: str) -> re.Pattern:
    """Whole-token (phrase-safe, case-insensitive) pattern.

    Mirrors register_glossary: non-alnum bounds so multi-word phrases like
    "the liquid" anchor and "tit" does not match inside "title".
    """
    return re.compile(
        r"(?<![A-Za-z0-9])" + re.escape(needle) + r"(?![A-Za-z0-9])",
        re.IGNORECASE,
    )


# ---------------------------------------------------------------------------
# Glossary — SMALL & HIGH-PRECISION. Seeded from the confirmed Ikenie no Haha 4
# findings and calibrated against the real v10_5_preferences.parquet.
#
# Calibration notes (from a live scan of the 11,901-row parquet):
#   - 潮 is the trap term: only ONE 潮吹き (squirt) row exists and its chosen is
#     already faithful. The other 潮 rows are non-sexual (Heian-era 最高潮,
#     紅潮=blush, 風潮=trend, 渦潮=whirlpool, literal 近づく潮=incoming tide).
#     So 潮 requires aggressive jp_excludes; we only arm on 潮吹.
#   - イク/イッ collides with マイク/ジェイク/バイク/ライク — guarded, but kept
#     OUT of the default glossary because イク is also non-explicit "to go"
#     (low precision); enable only with a confirmed euphemism mapping.
# ---------------------------------------------------------------------------
GLOSSARY: List[RegisterTerm] = [
    RegisterTerm(
        label="semen",
        jp_terms=["精液", "ザーメン", "ザー汁"],
        crude_en=["cum", "semen", "sperm", "load", "jizz", "spunk", "seed"],
        euphemisms=[
            "the liquid", "the fluid", "hot liquid", "the stuff", "bodily fluid",
            "bodily fluids", "fluids", "white liquid", "the substance", "discharge",
        ],
        jp_excludes=[],
    ),
    RegisterTerm(
        label="moaning",
        jp_terms=["喘ぎ声", "喘ぎ", "喘ぐ", "喘い"],
        crude_en=["moan", "moaning", "moaned", "moans"],
        euphemisms=[
            "calling out", "crying out", "cry out", "her voice", "his voice",
            "raised her voice", "let out a voice", "panting", "gasping breath",
        ],
        # 喘息 = asthma (medical, non-sexual). Disarm.
        jp_excludes=["喘息"],
    ),
    RegisterTerm(
        label="facial",
        jp_terms=["顔射"],
        crude_en=["facial", "on her face", "on my face", "cum on", "load on"],
        euphemisms=["spray", "sprayed", "splash", "splashed", "shower"],
        jp_excludes=[],
    ),
    RegisterTerm(
        label="refill",
        jp_terms=["おかわり", "お代わり"],
        crude_en=["refill", "again", "seconds", "more", "another round"],
        euphemisms=["another one, please", "another one please", "one more, please",
                    "may i have", "could i have"],
        # おかわり is overwhelmingly literal food/drink. Require an NSFW src to arm
        # (handled in audit_row via src gate); keep euphemisms tight.
        jp_excludes=[],
    ),
    RegisterTerm(
        label="pussy",
        jp_terms=["マンコ", "まんこ", "おまんこ", "おま●こ", "おま○こ", "おひ○ひん"],
        crude_en=["pussy", "cunt", "snatch", "vagina"],
        euphemisms=["butthole", "down there", "her privates", "private parts",
                    "her parts", "that place", "her hole"],
        jp_excludes=[],
    ),
    RegisterTerm(
        label="squirt",
        jp_terms=["潮吹", "潮まみれ", "潮を吹"],
        crude_en=["squirt", "squirting", "squirted", "squirts"],
        euphemisms=["seawater", "sea water", "salt water", "saltwater", "tide",
                    "gush of water", "water"],
        # Non-sexual 潮 compounds — these never co-occur with 潮吹 but listed for
        # defense in depth. The 潮_terms above are already the narrow sexual form.
        jp_excludes=["潮干狩", "潮風", "潮目", "潮流", "潮汐", "黒潮", "親潮",
                     "満潮", "干潮", "潮位", "潮騒", "紅潮", "風潮", "渦潮",
                     "最高潮", "高潮"],
    ),
    RegisterTerm(
        label="ejaculate",
        jp_terms=["射精"],
        crude_en=["cum", "came", "coming", "ejaculate", "ejaculation",
                  "ejaculating", "ejaculated", "shoot", "shot", "climax", "load"],
        euphemisms=["the outcome", "release", "released", "finish", "finished",
                    "let go", "let it out", "the moment", "the act"],
        jp_excludes=[],
    ),
    RegisterTerm(
        label="boobs",
        jp_terms=["おっぱい", "オッパイ"],
        crude_en=["boob", "boobs", "boobie", "boobies", "booby", "tit", "tits",
                  "titty", "titties", "breast", "breasts", "chest", "nipple",
                  "nipples", "rack", "knockers"],
        euphemisms=["chest area", "her front", "assets", "her body"],
        jp_excludes=[],
    ),
]

# Source-id substrings that mark a row as NSFW manga context. The corpus mixes
# NSFW manga lines with neutral instruction/literary data (Heian-era essays,
# Joyce, etc.). For TRAP terms (潮, おかわり, イク) the literal reading dominates
# outside NSFW context, so we gate those on an NSFW src. Terms whose JP token is
# inherently explicit (精液/射精/マンコ/顔射) do not need this gate.
_NSFW_SRC_HINTS = ("open_mantra", "gemma_anchor", "ikenie", "manga")
# Trap terms: the JP token has a common NON-explicit reading, so arm only in
# NSFW manga context. 潮=tide/peak, おかわり=food refill, 喘ぐ=labored breathing
# (a literary "pant/gasp" of wind etc., not a sexual moan).
_TRAP_LABELS = {"squirt", "refill", "moaning"}


def _src_is_nsfw(src: str) -> bool:
    s = (src or "").lower()
    return any(h in s for h in _NSFW_SRC_HINTS)


def extract_jp(prompt: str) -> str:
    """Recover the JP source line from a stored preference ``prompt``.

    The builders store the bare user-message body
    ("Translate ...\\n\\nJapanese: <jp>"); pull back the <jp> tail.
    """
    if not prompt:
        return ""
    m = _JP_RE.search(prompt)
    return m.group(1).strip() if m else ""


def _jp_armed(jp: str, term: RegisterTerm) -> bool:
    """True when ``jp`` contains a term token and no exclude guard fired."""
    if not any(t in jp for t in term.jp_terms):
        return False
    if any(ex in jp for ex in term.jp_excludes):
        return False
    return True


def _has_crude(text: str, term: RegisterTerm) -> bool:
    return any(p.search(text or "") for p in term._crude_pats)


def _matched_euphemisms(text: str, term: RegisterTerm) -> List[str]:
    return [e for e, p in zip(term.euphemisms, term._euph_pats) if p.search(text or "")]


def is_euphemistic(chosen: str, jp: str, term: RegisterTerm) -> bool:
    """Return True if ``chosen`` softens ``term`` given source ``jp``.

    Flag rule (all must hold):
      1. jp contains a term.jp_terms entry AND no term.jp_excludes entry
      2. chosen contains a term.euphemisms entry, OR chosen contains NONE of
         term.crude_en (explicit concept dropped/softened)
    """
    if not _jp_armed(jp, term):
        return False
    if _matched_euphemisms(chosen, term):
        return True
    if not _has_crude(chosen, term):
        return True
    return False


@dataclass
class Flag:
    """One flagged preference row."""

    term: str
    jp: str
    chosen: str
    rejected: str
    chosen_kind: str
    rejected_kind: str
    score_gap: float
    reason: str
    rejected_was_faithful: bool  # strong-signal subset: crude candidate existed, got demoted
    src: str = ""


def audit_row(row: dict) -> List[Flag]:
    """Run every glossary term against one preference row; return any Flags."""
    jp = extract_jp(row.get("prompt", ""))
    if not jp:
        return []
    chosen = row.get("chosen") or ""
    rejected = row.get("rejected") or ""
    src = row.get("src") or ""
    flags: List[Flag] = []
    for term in GLOSSARY:
        # Trap terms only arm inside NSFW manga context (literal reading wins
        # elsewhere). Inherently-explicit terms arm anywhere.
        if term.label in _TRAP_LABELS and not _src_is_nsfw(src):
            continue
        if not is_euphemistic(chosen, jp, term):
            continue
        euphs = _matched_euphemisms(chosen, term)
        if euphs:
            reason = f"chosen uses euphemism(s): {', '.join(euphs)}"
        else:
            reason = "chosen drops all crude renderings (concept softened)"
        rejected_was_faithful = _has_crude(rejected, term)
        try:
            gap = float(row.get("margin") if row.get("margin") is not None else 0.0)
        except (TypeError, ValueError):
            gap = 0.0
        flags.append(Flag(
            term=term.label,
            jp=jp,
            chosen=chosen,
            rejected=rejected,
            chosen_kind=str(row.get("chosen_kind", "")),
            rejected_kind=str(row.get("rejected_kind", "")),
            score_gap=gap,
            reason=reason,
            rejected_was_faithful=rejected_was_faithful,
            src=src,
        ))
    return flags


def load_preferences(parquet_path: Path) -> List[dict]:
    """Load the preference parquet into a list[dict].

    Tries polars then pandas; validates the expected columns; raises a clear
    error if the parquet is missing (it is gitignored — only on the main
    checkout, not in git worktrees).
    """
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"preference parquet not found: {parquet_path}\n"
            "It is gitignored and lives in the MAIN checkout, not in worktrees. "
            "Point --parquet at the main-checkout absolute path."
        )
    required = {"prompt", "chosen", "rejected", "chosen_kind", "rejected_kind"}
    try:
        import polars as pl  # type: ignore
        df = pl.read_parquet(parquet_path)
        cols = set(df.columns)
        missing = required - cols
        if missing:
            raise ValueError(f"parquet missing columns: {sorted(missing)}")
        return df.to_dicts()
    except ImportError:
        pass
    import pandas as pd  # type: ignore
    df = pd.read_parquet(parquet_path)
    cols = set(df.columns)
    missing = required - cols
    if missing:
        raise ValueError(f"parquet missing columns: {sorted(missing)}")
    return df.to_dict(orient="records")


def _summarize(flags: List[Flag], n_rows: int) -> dict:
    by_term = Counter(f.term for f in flags)
    by_kind = Counter(f.chosen_kind for f in flags)
    rej_faithful = sum(1 for f in flags if f.rejected_was_faithful)
    # per-term chosen_kind breakdown
    term_kind: dict = {}
    for f in flags:
        term_kind.setdefault(f.term, Counter())[f.chosen_kind] += 1
    return {
        "total_rows": n_rows,
        "total_flagged": len(flags),
        "by_term": dict(by_term.most_common()),
        "by_chosen_kind": dict(by_kind.most_common()),
        "rejected_was_faithful": rej_faithful,
        "term_chosen_kind": {t: dict(c.most_common()) for t, c in term_kind.items()},
    }


def write_report(flags: List[Flag], out_path: Path, n_rows: int,
                 report_path: Optional[Path] = None) -> dict:
    """Write flags as JSONL (faithful-rejected rows first) + a markdown report.

    Sort: rejected_was_faithful desc, then |score_gap| desc. Prints a summary.
    Returns the summary dict.
    """
    flags = sorted(flags, key=lambda f: (f.rejected_was_faithful, abs(f.score_gap)),
                   reverse=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for f in flags:
            fh.write(json.dumps(asdict(f), ensure_ascii=False) + "\n")

    summary = _summarize(flags, n_rows)

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(_render_markdown(flags, summary), encoding="utf-8")

    # console summary
    print(f"\n[audit] scanned {n_rows} rows, flagged {len(flags)}")
    print(f"[audit] by term:        {summary['by_term']}")
    print(f"[audit] by chosen_kind: {summary['by_chosen_kind']}")
    print(f"[audit] rejected-was-faithful (cheap swaps): "
          f"{summary['rejected_was_faithful']}")
    print(f"[audit] wrote JSONL -> {out_path}")
    if report_path:
        print(f"[audit] wrote report -> {report_path}")
    return summary


def _trunc(s: str, n: int = 160) -> str:
    s = (s or "").replace("\n", " ").replace("|", "\\|").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _render_markdown(flags: List[Flag], summary: dict) -> str:
    L: List[str] = []
    L.append("# fix8 — Register Euphemism Audit: Findings\n")
    L.append("_Generated by `backend/scripts/data/audit_register_euphemism.py` "
             "(read-only; the parquet is not mutated)._\n")
    L.append("## Headline\n")
    L.append(f"- **Rows scanned:** {summary['total_rows']:,}")
    L.append(f"- **Rows flagged (chosen softens an explicit source term):** "
             f"**{summary['total_flagged']}**")
    L.append(f"- **Flagged rows with a CRUDER candidate already in `rejected` "
             f"(cheap chosen↔rejected swaps):** **{summary['rejected_was_faithful']}**")
    L.append("")
    L.append("## Breakdown by term\n")
    L.append("| term | flagged |")
    L.append("|------|--------:|")
    for t, c in summary["by_term"].items():
        L.append(f"| {t} | {c} |")
    L.append("")
    L.append("## chosen_kind distribution of flagged rows\n")
    L.append("Hypothesis (plan §2): argmax-COMET-Kiwi pumps softened **gold/teacher** "
             "references into the `chosen` slot. A high gold/teacher share among "
             "flagged rows CONFIRMS the QE-softening mechanism; a high `onpolicy` "
             "share would instead implicate the model's own samples.\n")
    L.append("| chosen_kind | flagged |")
    L.append("|-------------|--------:|")
    for k, c in summary["by_chosen_kind"].items():
        L.append(f"| {k} | {c} |")
    L.append("")
    L.append("### Per-term × chosen_kind\n")
    kinds = sorted({k for d in summary['term_chosen_kind'].values() for k in d})
    L.append("| term | " + " | ".join(kinds) + " |")
    L.append("|------|" + "|".join("--:" for _ in kinds) + "|")
    for t, d in summary["term_chosen_kind"].items():
        L.append(f"| {t} | " + " | ".join(str(d.get(k, 0)) for k in kinds) + " |")
    L.append("")
    L.append("## ~15 concrete examples\n")
    L.append("Sorted so **rejected-was-faithful** (the cheap swaps) surface first, "
             "then by score margin. `gap` = chosen_score − rejected_score (COMET-Kiwi).\n")
    L.append("| term | kind | gap | rej=faithful | JP source | CHOSEN (softened) | REJECTED (candidate) |")
    L.append("|------|------|----:|:------------:|-----------|-------------------|----------------------|")
    for f in flags[:15]:
        L.append(
            f"| {f.term} | {f.chosen_kind} | {f.score_gap:+.3f} | "
            f"{'✓' if f.rejected_was_faithful else ''} | {_trunc(f.jp, 70)} | "
            f"{_trunc(f.chosen, 90)} | {_trunc(f.rejected, 90)} |"
        )
    L.append("")
    L.append("## Interpretation\n")
    gold = summary["by_chosen_kind"].get("gold", 0)
    teacher = summary["by_chosen_kind"].get("teacher", 0)
    onpolicy = summary["by_chosen_kind"].get("onpolicy", 0)
    tot = max(summary["total_flagged"], 1)
    ref_share = (gold + teacher) / tot
    L.append(
        f"- Reference-sourced (gold+teacher) share of flagged `chosen`: "
        f"**{ref_share:.0%}** ({gold} gold + {teacher} teacher vs {onpolicy} onpolicy)."
    )
    if ref_share >= 0.5:
        L.append(
            "- **The COMET-Kiwi-softening hypothesis HOLDS:** the majority of "
            "softened `chosen` rows are reference (gold/teacher) candidates that "
            "won the argmax-QE slot — exactly the pump described in plan §2. "
            "Remediation: re-pick `chosen` to the faithful candidate (often the "
            "`rejected` side) or drop the row. Do NOT oversample NSFW data."
        )
    else:
        L.append(
            "- **The COMET-Kiwi-softening hypothesis is only PARTIALLY supported:** "
            "most softened `chosen` rows are `onpolicy` (the model's own samples), "
            "so the softening is at least as much a model behaviour as a "
            "reference-selection artefact. The chosen-side repair still applies, "
            "but expect fewer clean reference swaps."
        )
    L.append(
        f"- **{summary['rejected_was_faithful']} / {summary['total_flagged']}** flagged "
        "rows already have a cruder candidate sitting in `rejected` — these are the "
        "cheapest fixes (a chosen↔rejected swap recovers faithful register with no "
        "new data)."
    )
    L.append("")
    return "\n".join(L)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Audit a preference parquet for euphemistic CHOSEN translations.")
    ap.add_argument("--parquet", default=DEFAULT_PARQUET,
                    help="preference parquet to audit")
    ap.add_argument("--out", default="backend/scripts/data/cpo/register_euphemism_flags.jsonl",
                    help="JSONL report destination")
    ap.add_argument("--report", default="backend/scripts/data/register_audit_findings.md",
                    help="markdown findings report destination")
    ap.add_argument("--dry-run", action="store_true",
                    help="scan and print summary, do not write report")
    args = ap.parse_args(argv)

    if not GLOSSARY:
        print("[warn] GLOSSARY is empty — fill it in before running "
              "(see thoughts/shared/plans/fix8-register-faithfulness-audit.md)",
              file=sys.stderr)
        return 1

    rows = load_preferences(Path(args.parquet))
    flags = [f for r in rows for f in audit_row(r)]

    if args.dry_run:
        summary = _summarize(flags, len(rows))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    write_report(flags, Path(args.out), len(rows), Path(args.report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
