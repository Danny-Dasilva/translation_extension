#!/usr/bin/env python3
"""Audit a CPO/DPO preference parquet for euphemistic CHOSEN translations.

SKELETON — signatures, docstring, and TODOs only. No logic is wired yet.
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
(microphone) / ジェイク (Jake); 潮 matches 潮干狩り (clam digging). Better to miss
a row than to mislabel a clean one. Start SMALL (the 5 confirmed Ikenie terms),
grow only on confirmed (jp, euphemism->crude) mappings.

USAGE (planned)
---------------
    python backend/scripts/data/audit_register_euphemism.py \
        --parquet backend/scripts/data/cpo/v10_5_preferences.parquet \
        --out backend/scripts/data/cpo/register_euphemism_flags.jsonl

Schema expected (verified on v10_5_preferences.parquet, 11,901 rows):
    [prompt, chosen, rejected, chosen_score, rejected_score, margin,
     chosen_kind, rejected_kind, src]
The JP source is embedded in ``prompt`` after the literal "Japanese: ".
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# NOTE: pandas/polars not imported at module top so the file imports cleanly in
# environments without them. Import inside main() / load_preferences().

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


# ---------------------------------------------------------------------------
# Glossary — SMALL & HIGH-PRECISION. Seeded from the confirmed Ikenie no Haha 4
# findings. TODO: fill crude_en / euphemisms / jp_excludes from the human-ref
# comparison; add ザーメン, 喘ぐ variants, etc. only with a confirmed mapping.
# ---------------------------------------------------------------------------
GLOSSARY: List[RegisterTerm] = [
    # TODO RegisterTerm(label="semen",   jp_terms=["精液", "ザーメン"], crude_en=["cum","semen","load"], euphemisms=["the liquid","fluid"], jp_excludes=[]),
    # TODO RegisterTerm(label="moaning", jp_terms=["喘ぎ", "喘ぐ"],     crude_en=["moan","moaning"],      euphemisms=["calling out","crying out"], jp_excludes=[]),
    # TODO RegisterTerm(label="facial",  jp_terms=["顔射"],            crude_en=["facial","on her face"], euphemisms=["spray"], jp_excludes=[]),
    # TODO RegisterTerm(label="refill",  jp_terms=["おかわり"],         crude_en=["refill","again","seconds"], euphemisms=["another one, please"], jp_excludes=[]),  # guard food/drink context
    # TODO RegisterTerm(label="pussy",   jp_terms=["マンコ", "まんこ"],  crude_en=["pussy","cunt"],         euphemisms=["butthole","down there","privates"], jp_excludes=[]),
]


def extract_jp(prompt: str) -> str:
    """Recover the JP source line from a stored preference ``prompt``.

    The builders store the bare user-message body
    ("Translate ...\\n\\nJapanese: <jp>"); pull back the <jp> tail.
    """
    if not prompt:
        return ""
    m = _JP_RE.search(prompt)
    return m.group(1).strip() if m else ""


def _whole_word(text: str, needle: str) -> bool:
    """Whole-token (phrase-safe, case-insensitive) containment check.

    TODO: reuse the boundary pattern from register_glossary (non-alnum bounds so
    multi-word phrases like "the liquid" anchor; avoid 'tide' inside 'untidy').
    """
    raise NotImplementedError  # TODO


def is_euphemistic(chosen: str, jp: str, term: RegisterTerm) -> bool:
    """Return True if ``chosen`` softens ``term`` given source ``jp``.

    Flag rule (all must hold):
      1. jp contains a term.jp_terms entry AND no term.jp_excludes entry
      2. chosen contains a term.euphemisms entry, OR chosen contains NONE of
         term.crude_en (explicit concept dropped/softened)
    TODO: implement using extract_jp()-resolved jp + _whole_word().
    """
    raise NotImplementedError  # TODO


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


def audit_row(row: dict) -> List[Flag]:
    """Run every glossary term against one preference row; return any Flags.

    TODO:
      - jp = extract_jp(row["prompt"])
      - for term in GLOSSARY: if is_euphemistic(row["chosen"], jp, term): build Flag
      - set rejected_was_faithful = any(crude_en whole-word in row["rejected"])
        (these surface first — the faithful rendering was present and demoted)
    """
    raise NotImplementedError  # TODO


def load_preferences(parquet_path: Path):
    """Load the preference parquet into a list[dict].

    TODO: try polars then pandas; validate the expected columns are present
    (prompt, chosen, rejected, chosen_kind, rejected_kind, *_score); raise a
    clear error if the parquet is missing (it is gitignored — only on the main
    checkout, not in git worktrees).
    """
    raise NotImplementedError  # TODO


def write_report(flags: List[Flag], out_path: Path) -> None:
    """Write flags as JSONL, faithful-rejected rows first, and print a summary.

    TODO:
      - sort: rejected_was_faithful desc, then score_gap desc
      - JSONL of dataclasses.asdict(flag)
      - print per-term counts and per chosen_kind counts (gold/onpolicy/teacher
        tells whether softening comes from the human ref, the model, or teacher)
    """
    raise NotImplementedError  # TODO


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", default=DEFAULT_PARQUET,
                    help="preference parquet to audit")
    ap.add_argument("--out", default="backend/scripts/data/cpo/register_euphemism_flags.jsonl",
                    help="JSONL report destination")
    ap.add_argument("--dry-run", action="store_true",
                    help="scan and print summary, do not write report")
    args = ap.parse_args(argv)

    if not GLOSSARY:
        print("[warn] GLOSSARY is empty — fill it in before running "
              "(see thoughts/shared/plans/fix8-register-faithfulness-audit.md)",
              file=sys.stderr)

    # TODO:
    #   rows  = load_preferences(Path(args.parquet))
    #   flags = [f for r in rows for f in audit_row(r)]
    #   if not args.dry_run: write_report(flags, Path(args.out))
    raise NotImplementedError("audit_register_euphemism is a skeleton — see plan doc")


if __name__ == "__main__":
    sys.exit(main())
