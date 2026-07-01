"""Assemble a REFUSAL-STRIPPED, TEXT-ONLY SFT dataset for the translation ship.

Goal
----
Build a text-only SFT set to fine-tune
``huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated`` on the manga JP->EN
translation task. This is TEXT-ONLY SFT: no page image is present at train time
(the image is added only at inference-time serving, which is out of scope here).

Two properties are load-bearing:

1. TRAIN==SERVE PROMPT CONTRACT.  Every emitted ``prompt`` is produced by the
   EXACT v11 page-context builders
   (``scripts/data/v11/build_v11_dataset.build_context_prompt`` /
   ``build_plain_prompt``), which are byte-identical to the SERVE-side builders in
   ``app.services.vllm_openai_translation_service``. A whitespace/marker drift here
   is the documented ~95% chrF++ collapse, so the prompt is NEVER hand-assembled.
   ``verify_builder_parity`` asserts the two builders agree byte-for-byte.

2. REFUSAL STRIP + NSFW CAP.  Machine-EN NSFW targets (mined by an instruct model)
   occasionally leak refusal strings ("I'm sorry, but I can't ..."). We filter the
   EN TARGET side with an auditable refusal classifier (``REFUSAL_RE``). Separately,
   total NSFW is HARD-CAPPED at <= 18% of rows (a prior 36% oversample regressed the
   model into euphemism/coy register).

Data mix (text-only; from the ship recipe)
-------------------------------------------
* v11 page-context backbone   -- register/fluency; subsample of
  ``scripts/data/v11fix8/data_v11fix8_pagecontext.parquet`` (~302k rows).
* Ikenie gold x3              -- in-domain human gold, upweighted 3x, rebuilt into
  the v11 page-context prompt shape from per-bubble gold bboxes.
* NSFW corpus register slice  -- machine-EN mined via the ``corpus_bitext`` pipeline
  (``data_corpus_bitext_pagecontext.parquet``). This is ALSO the refusal-strip
  target source. NOTE: the full mined parquet is produced by a GPU run; if it is
  not present locally the slice is empty (or, with ``--allow-nsfw-placeholder``,
  the ikenie-derived validation sample stands in) and the build is flagged.

HELD OUT: furube (``scripts/eval/data/furube/gold_furube_p*.jsonl``) is the EVAL
set and MUST NOT appear in training (asserted by ``assert_no_furube``).

Schema (output): ``[prompt, en, src, register_tag, gold_flag]`` -- the training
parquet schema. ``prompt`` is the full user message; the SFT config template is a
passthrough ``"{prompt}"`` with ``completion_field: en``.

Modes
-----
    --inspect    Schema/format/parity checks only. NO output written.
    --dry-run    Build a SMALL sample (default 200 rows) + stats. NO full run.
    --full       Build the full parquet (+ sample + stats). Multi-hour; gated.

Outputs (default under ``scripts/data/v13ship/``; the .parquet is gitignored by
``scripts/data/**/*.parquet``, the small sample.jsonl + stats.json are tracked):
    data_v13ship_textsft.parquet         (--full only)
    data_v13ship_textsft.sample.jsonl
    v13ship_stats.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import polars as pl

# --------------------------------------------------------------------------- #
# Paths + sibling/serve imports (reuse the EXACT v11 builders -- never reinvent)
# --------------------------------------------------------------------------- #
BACKEND = Path(__file__).resolve().parents[3]  # .../backend
for _p in (
    BACKEND,
    BACKEND / "scripts" / "data" / "v11",
    BACKEND / "scripts" / "data" / "v11fix6",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from build_v11_dataset import (  # noqa: E402  (sibling module, byte-exact builders)
    CONV_INSTR,
    PAGE_INSTR,
    PLAIN_INSTR,
    build_context_prompt,
    build_plain_prompt,
    manga_reading_order,
)

try:  # ALL-CAPS scanlation -> natural sentence case (proven helper)
    from build_v11fix6_corrective import to_sentence_case  # noqa: E402
except Exception:  # pragma: no cover - fallback keeps the builder self-contained
    def to_sentence_case(text: str) -> str:
        s = (text or "").strip()
        if s and s.isupper():
            return s[:1] + s[1:].lower()
        return s

# SERVE-side prompt builders. Parity with these is the ~95%-collapse guard.
try:
    from app.services.vllm_openai_translation_service import (  # noqa: E402
        V11_PAGE_INSTR,
        V11_PLAIN_INSTR,
        build_v11_context_prompt,
        build_v11_plain_prompt,
    )
    _HAVE_SERVE = True
except Exception:  # pragma: no cover
    V11_PAGE_INSTR, V11_PLAIN_INSTR = PAGE_INSTR, PLAIN_INSTR
    build_v11_context_prompt = build_v11_plain_prompt = None  # type: ignore
    _HAVE_SERVE = False

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
COLS = ["prompt", "en", "src", "register_tag", "gold_flag"]
SEED = 42
MAX_CONTEXT = 12  # mirror v11fix7/format_rows: page context windowed around target
NSFW_CAP = 0.18   # HARD cap: a prior 36% NSFW oversample regressed into euphemism
NSFW_REGISTERS = {"manga_nsfw", "vn_eroge", "nsfw", "eroge", "doujin_nsfw"}

DEFAULT_V11_PARQUET = BACKEND / "scripts/data/v11fix8/data_v11fix8_pagecontext.parquet"
DEFAULT_IKENIE = [
    BACKEND / "scripts/eval/data/ikenie4/gold_q3.jsonl",
    BACKEND / "scripts/eval/data/ikenie5/gold_q3.jsonl",
]
# Full mined NSFW corpus (GPU run output). Merged shard parquet -> not present until
# the corpus_bitext EN-VLM pass runs. The validation sample is an ikenie-derived
# stand-in only.
DEFAULT_NSFW_PARQUET = (
    BACKEND / "scripts/data/corpus_bitext/data_corpus_bitext_pagecontext.parquet"
)
NSFW_PLACEHOLDER = (
    BACKEND / "scripts/data/corpus_bitext/validation_out/sample_ikenie_bitext.parquet"
)
DEFAULT_OUT_DIR = BACKEND / "scripts/data/v13ship"

# --------------------------------------------------------------------------- #
# Refusal classifier (NAMED, AUDITABLE constant)
# --------------------------------------------------------------------------- #
# Drops EN targets that are an instruct-model refusal rather than a translation.
# Concentrated in the machine-EN NSFW slice (an abliterated/instruct model mining
# adult pages occasionally still emits "I'm sorry, but I can't ..."). Expected drop
# rate ~2%, almost entirely in the mined NSFW rows.
#
# DERIVED FROM the task's suggested cues (as an ai | sorry | unable | cannot |
# won't/will not | inappropriate | translate) but the FP-prone bare arms
# ("i can't", "i'm sorry", "inappropriate") are SCOPED to a refusal object so we do
# NOT nuke ubiquitous manga dialogue: "I can't wait!", "I'm sorry I'm late",
# "I won't lose!", "That's inappropriate of you" must all SURVIVE. The refusal
# signal is an apology/inability preamble FOLLOWED BY a refusal-to-perform clause.
# Empirically tuned against the 302k v11 backbone: this arm set fires on 1/302,382
# rows (0.0003%; a lone in-story "I will not help you with the <project>" refusal),
# vs 129/302k for a looser bare-verb version that FP'd on "I can't help but ...",
# "I can't do this", "I'm afraid I can't do that", "I apologize, but please ...",
# and "Goes against my policy". The distinguishing signal is a refusal-to-do-the-
# TASK object (assist/translate/comply/provide a translation/generate content),
# NOT bare "can't/sorry/inappropriate".
_REFUSAL_ALTS = [
    r"\bas an ai(?: language model| assistant)?\b",
    # apology/inability preamble FOLLOWED BY a refusal-to-do-the-task clause
    r"i(?:'m| am) (?:sorry|afraid)[,.\s]+(?:but[,.\s]+)?i (?:can(?:'?t|not)|cannot|won'?t|will not|am unable|am not able) (?:assist|help (?:you|with)|translate|comply|provide|generate|create|produce|fulfil|fulfill|continue with|complete this|process this)",
    r"i(?:'m| am) (?:unable|not able) to (?:assist|help (?:you|with)|translate|comply|provide a translation|process this|complete this|continue|generate|create)",
    r"i (?:can(?:'?t|not)|cannot|won'?t|will not) (?:assist you|assist with|help you with|translate (?:this|that|the|it)|comply with|provide (?:a )?translation|generate (?:that|this|explicit|content)|produce (?:that|this|explicit)|fulfil (?:that|this)|fulfill (?:that|this)|continue with (?:this|that|the))",
    r"i (?:must|have to|will) (?:decline|refuse) to (?:assist|help|translate|provide|comply|continue|do)",
    r"i apologi[sz]e[,.\s]+but i (?:can(?:'?t|not)|cannot|won'?t|will not|am unable|am not able)",
    r"\bcannot (?:assist with|help you with|be translated|translate (?:this|that|it)|comply with|fulfil this|fulfill this)",
    r"unable to (?:assist with|help you with|translate this|comply with|provide a translation|process this)",
    r"not (?:able|allowed|permitted) to (?:assist with|help (?:you )?with|translate this|provide (?:a )?translation|comply with|generate)",
    r"against my (?:guidelines|programming|content policy|content guidelines)",
    r"(?:this|that|the) (?:content|request|material|image) (?:violates|goes against my|is inappropriate and|cannot be (?:translated|assisted|provided))",
    r"i(?:'m| am)? ?not (?:comfortable|willing) (?:translat|assist|help|provid|generat|complet|continu)",
    r"i can(?:'?t|not) (?:in good conscience|provide a translation of)",
    r"inappropriate (?:content|request|material|and i (?:can|cannot|will|am))",
]
REFUSAL_PATTERN = r"(?i)(?:" + "|".join(_REFUSAL_ALTS) + r")"
REFUSAL_RE = re.compile(REFUSAL_PATTERN)


def is_refusal(en: str | None) -> bool:
    """True if an EN target reads as an instruct-model refusal, not a translation."""
    if not en:
        return False
    return REFUSAL_RE.search(en) is not None


def refusal_strip(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Partition rows into (kept, dropped) by the refusal classifier on ``en``."""
    kept, dropped = [], []
    for r in rows:
        (dropped if is_refusal(r.get("en")) else kept).append(r)
    return kept, dropped


# Canonical LLM refusals (MUST drop) and FP-prone benign manga dialogue (MUST keep).
# Used for a stats self-test so the classifier is demonstrated even when the live
# data has ~0 real refusals (the refusal-heavy source is the missing mined NSFW
# parquet).
_SELFTEST_REFUSALS = [
    "I'm sorry, but I can't assist with that request.",
    "I cannot translate this content.",
    "As an AI language model, I won't help with that.",
    "I'm unable to provide a translation of this material.",
    "I apologize, but I can't continue with this request.",
]
_SELFTEST_BENIGN = [
    "I'm sorry I'm late!",
    "I can't wait to see you again.",
    "I won't lose to you!",
    "Put it in deeper ♥",
    "That's so inappropriate of you, hehe.",
]


def refusal_classifier_selftest() -> dict:
    """Demonstrate REFUSAL_RE: canonical refusals drop, benign dialogue survives."""
    dropped = [s for s in _SELFTEST_REFUSALS if is_refusal(s)]
    leaked = [s for s in _SELFTEST_REFUSALS if not is_refusal(s)]
    false_pos = [s for s in _SELFTEST_BENIGN if is_refusal(s)]
    return {
        "refusals_correctly_dropped": dropped,
        "refusals_missed": leaked,
        "benign_false_positives": false_pos,
        "passes": not leaked and not false_pos,
    }


# --------------------------------------------------------------------------- #
# NSFW cap
# --------------------------------------------------------------------------- #
def is_nsfw_row(row: dict) -> bool:
    return (row.get("register_tag") or "") in NSFW_REGISTERS


def enforce_nsfw_cap(
    rows: list[dict], cap: float = NSFW_CAP, seed: int = SEED
) -> tuple[list[dict], int, dict]:
    """Deterministically downsample NSFW rows so NSFW share <= ``cap``.

    Non-NSFW rows are NEVER dropped. Returns ``(kept, dropped_count, info)``.
    Row ORDER of the surviving rows is preserved (a seeded subset of NSFW rows is
    removed in place), so the caller's later shuffle is the only re-order.
    """
    nsfw_idx = [i for i, r in enumerate(rows) if is_nsfw_row(r)]
    non_n = len(rows) - len(nsfw_idx)
    if not rows or not nsfw_idx:
        return list(rows), 0, {"nsfw_before": len(nsfw_idx), "nsfw_after": len(nsfw_idx),
                               "max_nsfw": len(nsfw_idx), "cap": cap}
    # nsfw_keep <= cap*(non_n + nsfw_keep)  ->  nsfw_keep <= cap*non_n/(1-cap)
    if cap >= 1.0:
        max_nsfw = len(nsfw_idx)
    else:
        max_nsfw = int((cap * non_n) / (1.0 - cap))
    if len(nsfw_idx) <= max_nsfw:
        return list(rows), 0, {"nsfw_before": len(nsfw_idx), "nsfw_after": len(nsfw_idx),
                               "max_nsfw": max_nsfw, "cap": cap}
    keep_set = set(random.Random(seed).sample(nsfw_idx, max_nsfw))
    kept = [
        r for i, r in enumerate(rows)
        if (i not in set(nsfw_idx)) or (i in keep_set)
    ]
    dropped = len(rows) - len(kept)
    info = {"nsfw_before": len(nsfw_idx), "nsfw_after": max_nsfw,
            "max_nsfw": max_nsfw, "cap": cap}
    return kept, dropped, info


# --------------------------------------------------------------------------- #
# Furube hold-out guard
# --------------------------------------------------------------------------- #
def assert_no_furube(rows: list[dict]) -> None:
    """Raise if any row's ``src`` references the held-out furube EVAL set."""
    for r in rows:
        if "furube" in str(r.get("src", "")).lower():
            raise ValueError(f"furube is the EVAL set and must not train: {r.get('src')!r}")


# --------------------------------------------------------------------------- #
# Source loaders
# --------------------------------------------------------------------------- #
def load_v11_backbone(parquet_path: Path, n: int | None, seed: int = SEED) -> list[dict]:
    """Subsample the v11 page-context backbone parquet (already in v11 format)."""
    df = pl.read_parquet(parquet_path).select(COLS)
    if n is not None and n < df.height:
        df = df.sample(n=n, seed=seed, shuffle=True)
    rows = df.to_dicts()
    for r in rows:
        r["gold_flag"] = bool(r["gold_flag"])
    return rows


def _window(jp_lines: list[str], k: int) -> tuple[list[str], int]:
    """Window the page context to MAX_CONTEXT around the target (mirror v11fix7)."""
    n = len(jp_lines)
    if n <= MAX_CONTEXT:
        return jp_lines, k
    half = MAX_CONTEXT // 2
    lo = max(0, k - half)
    hi = min(n, lo + MAX_CONTEXT)
    lo = max(0, hi - MAX_CONTEXT)
    return jp_lines[lo:hi], k - lo


_IKENIE_SRC_RE = re.compile(r"^([^:]+):p(\d+):idx(\d+)")
_LATIN_RE = re.compile(r"[A-Za-z]")


def _is_english_target(en: str) -> bool:
    """True if the EN target has at least one Latin letter (drops untranslated
    SFX/JP-only 'translations' like katakana onomatopoeia)."""
    return bool(_LATIN_RE.search(en or ""))


def _load_ikenie_pages(gold_path: Path) -> dict[tuple[str, int], list[dict]]:
    """Group ikenie gold rows by (book, page), deduped by src, bbox-bearing only."""
    by_page: dict[tuple[str, int], list[dict]] = {}
    seen: set[str] = set()
    with gold_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not r.get("bbox"):
                continue
            src = r.get("src", "")
            if src in seen:
                continue
            m = _IKENIE_SRC_RE.match(src)
            if not m:
                continue
            seen.add(src)
            book, page = m.group(1), int(m.group(2))
            by_page.setdefault((book, page), []).append(r)
    return by_page


def build_ikenie_rows(
    gold_paths: list[Path],
    repeat: int = 3,
    also_plain: bool = True,
    recase: bool = True,
    max_pages: int | None = None,
) -> list[dict]:
    """Rebuild ikenie per-bubble gold into v11 PAGE-CONTEXT (+ optional plain) rows.

    In-domain HUMAN gold -> ``gold_flag=True``. Each page's bubbles are ordered by
    the exact training reading order (``manga_reading_order``, RTL column-major),
    then every bubble is emitted as a marked-line page-context prompt via the
    byte-exact ``build_context_prompt(PAGE_INSTR, ...)``. ALL-CAPS scanlation
    targets are recased to sentence case so we never teach shouting. Upweighted
    ``repeat`` x (default 3) to match the ship recipe.
    """
    unit: list[dict] = []
    for gold_path in gold_paths:
        pages = _load_ikenie_pages(Path(gold_path))
        page_keys = sorted(pages.keys())
        if max_pages is not None:
            page_keys = page_keys[:max_pages]
        for (book, page) in page_keys:
            grows = pages[(book, page)]
            # order by reading order using the bbox (map ikenie minX/.. -> xmin/..)
            order_in = []
            for i, r in enumerate(grows):
                bb = r["bbox"]
                order_in.append({
                    "xmin": float(bb["minX"]), "ymin": float(bb["minY"]),
                    "xmax": float(bb["maxX"]), "ymax": float(bb["maxY"]),
                    "_i": i,
                })
            ordered = manga_reading_order(order_in)
            ordered_rows = [grows[o["_i"]] for o in ordered]
            jp_lines = [(r.get("jp") or "").strip() for r in ordered_rows]
            for pos, r in enumerate(ordered_rows):
                jp = jp_lines[pos]
                en = (r.get("en") or "").strip()
                if recase:
                    en = to_sentence_case(en)
                # drop untranslated SFX / JP-only "en" (mirror corpus EN-language filter)
                if not jp or not en or not _is_english_target(en):
                    continue
                ctx_lines, k = _window(list(jp_lines), pos)
                if 0 <= k < len(ctx_lines):
                    ctx_lines[k] = jp  # keep target verbatim at its windowed slot
                reg = r.get("register_tag") or "manga_nsfw"
                base = f"ikenie_gold:{book}:p{page:03d}:idx{pos}"
                unit.append({
                    "prompt": build_context_prompt(PAGE_INSTR, ctx_lines, k),
                    "en": en,
                    "src": f"{base}:pagectx",
                    "register_tag": reg,
                    "gold_flag": True,
                })
                if also_plain:
                    unit.append({
                        "prompt": build_plain_prompt(jp),
                        "en": en,
                        "src": f"{base}:plain",
                        "register_tag": reg,
                        "gold_flag": True,
                    })
    return unit * max(1, int(repeat))


def load_nsfw_corpus(
    nsfw_parquet: Path,
    allow_placeholder: bool = False,
    max_rows: int | None = None,
    seed: int = SEED,
) -> tuple[list[dict], dict]:
    """Load the mined NSFW register slice (v11 format). Returns (rows, info).

    Prefers the full mined parquet. If it is absent (the GPU EN-VLM mine has not
    run), optionally falls back to the ikenie-derived validation sample as a
    clearly-flagged PLACEHOLDER; otherwise returns [] and flags the blocker.
    ``max_rows`` deterministically truncates the slice (used to keep the dry-run
    mix balanced; the 18% cap still applies downstream).
    """

    def _load(path: Path, placeholder: bool, note: str | None):
        df = pl.read_parquet(path).select(COLS)
        if max_rows is not None and max_rows < df.height:
            df = df.sample(n=max_rows, seed=seed, shuffle=True)
        rows = df.to_dicts()
        for r in rows:
            r["gold_flag"] = bool(r["gold_flag"])
        info = {"source": str(path), "placeholder": placeholder, "rows": len(rows)}
        if note:
            info["note"] = note
        return rows, info

    if Path(nsfw_parquet).exists():
        return _load(Path(nsfw_parquet), False, None)
    if allow_placeholder and NSFW_PLACEHOLDER.exists():
        return _load(NSFW_PLACEHOLDER, True,
                     "ikenie-derived validation sample; NOT independent mined data")
    return [], {"source": None, "placeholder": False, "rows": 0,
                "note": "mined NSFW corpus parquet MISSING (needs GPU corpus_bitext mine)"}


# --------------------------------------------------------------------------- #
# Prompt-format parity (train builder == serve builder)
# --------------------------------------------------------------------------- #
_PARITY_JP_POOL = [
    "おはようございます、今日はいい天気ですね",
    "もう朝ごはんの時間か",
    "違う…！！そんなつもりじゃなかった",
    "平然と家族で朝ごはんを食べている",
    "食欲が無いのか？",
    "あ…ちょっと待って",
    "君のことがずっと好きだった",
    "そんなこと言われても困るよ",
    "早く逃げないと捕まってしまう",
    "この街を守るのが俺の使命だ",
]


def verify_builder_parity(n_cases: int = 200, seed: int = SEED) -> dict:
    """Assert the TRAIN builders byte-match the SERVE builders.

    Compares ``build_context_prompt(PAGE_INSTR, lines, k)`` (train) with
    ``build_v11_context_prompt(lines, k)`` (serve) and the plain equivalents. The
    serve builder applies short-utterance normalization + an optional cast anchor
    at call time; both are DISABLED here so the comparison is a pure format check
    (the trained template itself carries no normalization).
    """
    info: dict = {
        "serve_available": _HAVE_SERVE,
        "instr_page_match": PAGE_INSTR == V11_PAGE_INSTR,
        "instr_plain_match": PLAIN_INSTR == V11_PLAIN_INSTR,
        "compared": 0,
        "mismatches": 0,
        "examples": [],
    }
    if not _HAVE_SERVE:
        info["note"] = "serve module unavailable; parity asserted via identical builders"
        # Instructions fall back to the train constants, so they match by identity.
        return info

    # Disable serve-side normalize + cast so the comparison is format-pure.
    from app.config import settings  # local import
    saved = {
        "short_utterance_normalize_enabled": getattr(
            settings, "short_utterance_normalize_enabled", True),
        "translation_cast_anchor": getattr(settings, "translation_cast_anchor", False),
    }
    rng = random.Random(seed)
    try:
        settings.short_utterance_normalize_enabled = False
        settings.translation_cast_anchor = False
        for _ in range(n_cases):
            m = rng.randint(1, len(_PARITY_JP_POOL))
            lines = rng.sample(_PARITY_JP_POOL, m)
            k = rng.randrange(m)
            train_p = build_context_prompt(PAGE_INSTR, lines, k)
            serve_p = build_v11_context_prompt(lines, k)
            info["compared"] += 1
            if train_p != serve_p:
                info["mismatches"] += 1
                if len(info["examples"]) < 3:
                    info["examples"].append({"train": train_p, "serve": serve_p})
            # plain
            jp = lines[k]
            if build_plain_prompt(jp) != build_v11_plain_prompt(jp):
                info["mismatches"] += 1
                if len(info["examples"]) < 3:
                    info["examples"].append(
                        {"train": build_plain_prompt(jp),
                         "serve": build_v11_plain_prompt(jp)})
    finally:
        for k2, v2 in saved.items():
            setattr(settings, k2, v2)
    return info


def sample_rows_match_serve_format(rows: list[dict]) -> dict:
    """Per-row structural check: every prompt begins with a TRAINED v11 head.

    The three v11 instruction heads are PAGE (manga page-context, byte-identical to
    the SERVE ``V11_PAGE_INSTR``), CONV (VN/novel conversation-context; same
    marked-line contract, conversation wording) and PLAIN (single line, byte-identical
    to SERVE ``V11_PLAIN_INSTR``). Serving only emits PAGE/PLAIN, but CONV is a valid
    trained head carried by the VN/novel backbone rows.
    """
    page = conv = plain = other = 0
    for r in rows:
        p = r["prompt"]
        if p.startswith(V11_PAGE_INSTR):
            page += 1
        elif p.startswith(CONV_INSTR):
            conv += 1
        elif p.startswith(V11_PLAIN_INSTR):
            plain += 1
        else:
            other += 1
    return {"pagectx_prompts": page, "convctx_prompts": conv, "plain_prompts": plain,
            "unrecognized_prompts": other, "all_match": other == 0}


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #
def _pctile(vals: list[int], q: float) -> int:
    if not vals:
        return 0
    s = sorted(vals)
    i = min(len(s) - 1, int(round(q * (len(s) - 1))))
    return s[i]


def length_stats(rows: list[dict]) -> dict:
    """Char/word length distribution for en + prompt (token count is a chars/4 proxy)."""
    en_chars = [len(r["en"]) for r in rows]
    en_words = [len(r["en"].split()) for r in rows]
    pr_chars = [len(r["prompt"]) for r in rows]
    approx_tok = [max(1, len(r["prompt"]) // 4) for r in rows]

    def dist(vals):
        return {"p50": _pctile(vals, 0.50), "p90": _pctile(vals, 0.90),
                "p99": _pctile(vals, 0.99), "max": max(vals) if vals else 0,
                "mean": round(sum(vals) / len(vals), 1) if vals else 0.0}
    return {
        "en_chars": dist(en_chars),
        "en_words": dist(en_words),
        "prompt_chars": dist(pr_chars),
        "prompt_tokens_approx_chars_div4": dist(approx_tok),
    }


def _source_of(row: dict) -> str:
    src = row.get("src", "")
    if src.startswith("ikenie_gold:"):
        return "ikenie_gold"
    if src.startswith("corpus_bitext:"):
        return "nsfw_corpus"
    return "v11_backbone"


# --------------------------------------------------------------------------- #
# Native trainer messages-jsonl format (build_conversations schema)
# --------------------------------------------------------------------------- #
# ``sft_qwen3vl_8b_imagectx.build_conversations`` consumes rows whose ``messages``
# is a Qwen3-VL chat list of {role, content:[{type:text,text}|{type:image,image}]}.
# For the TEXT-ONLY v1 set every row is image-absent (has_image=False), so the
# user block is the byte-exact v11 ``prompt`` and the assistant block is ``en``.
# Parity fields (source/register_tag/has_image/image_path/meta) mirror the
# v12vision ``data_poc_imageoff.jsonl`` rows; build_conversations only strictly
# needs ``messages``.
def row_to_message(row: dict) -> dict:
    return {
        "source": _source_of(row),
        "register_tag": row.get("register_tag"),
        "has_image": False,
        "image_path": "",
        "gold_flag": bool(row.get("gold_flag")),
        "messages": [
            {"role": "user",
             "content": [{"type": "text", "text": row["prompt"]}]},
            {"role": "assistant",
             "content": [{"type": "text", "text": row["en"]}]},
        ],
        "meta": {"src": row.get("src")},
    }


def messages_format_check(msgs: list[dict], orig_rows: list[dict]) -> dict:
    """Validate the messages schema + round-trip (user text == the v11 prompt)."""
    valid = roundtrip = img_ok = True
    for m, r in zip(msgs, orig_rows):
        conv = m.get("messages")
        if (not isinstance(conv, list) or len(conv) != 2
                or conv[0].get("role") != "user"
                or conv[1].get("role") != "assistant"):
            valid = False
            continue
        try:
            u = conv[0]["content"][0]
            a = conv[1]["content"][0]
        except (KeyError, IndexError, TypeError):
            valid = False
            continue
        if u.get("type") != "text" or a.get("type") != "text":
            valid = False
        if u.get("text") != r["prompt"] or a.get("text") != r["en"]:
            roundtrip = False
        if m.get("has_image") is not False:
            img_ok = False
    return {"checked": len(msgs), "valid": valid, "roundtrip_ok": roundtrip,
            "has_image_all_false": img_ok}


def _write_messages_jsonl(rows: list[dict], path: Path) -> int:
    """Write rows as native trainer messages JSONL. Returns bytes written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(row_to_message(r), ensure_ascii=False) + "\n")
            n += 1
    return path.stat().st_size


def compose_stats(final_rows: list[dict], per_source_before: dict,
                  refusal_dropped: list[dict], nsfw_cap_info: dict,
                  parity: dict, extra: dict) -> dict:
    by_src: dict[str, int] = {}
    for r in final_rows:
        by_src[_source_of(r)] = by_src.get(_source_of(r), 0) + 1
    nsfw = sum(1 for r in final_rows if is_nsfw_row(r))
    n = len(final_rows)
    reg: dict[str, int] = {}
    for r in final_rows:
        reg[r.get("register_tag") or "?"] = reg.get(r.get("register_tag") or "?", 0) + 1
    return {
        "total_rows": n,
        "rows_per_source_final": by_src,
        "rows_per_source_before_filters": per_source_before,
        "register_distribution_final": dict(sorted(reg.items(), key=lambda kv: -kv[1])),
        "nsfw_rows_final": nsfw,
        "nsfw_frac_final": round(nsfw / n, 4) if n else 0.0,
        "nsfw_cap": NSFW_CAP,
        "nsfw_cap_info": nsfw_cap_info,
        "refusal_dropped_count": len(refusal_dropped),
        "refusal_dropped_examples": [
            {"src": d.get("src"), "en": (d.get("en") or "")[:160]}
            for d in refusal_dropped[:8]
        ],
        "refusal_classifier_selftest": refusal_classifier_selftest(),
        "length_distribution": length_stats(final_rows),
        "prompt_format_serve_parity": parity,
        "sample_serve_format_check": sample_rows_match_serve_format(final_rows),
        **extra,
    }


# --------------------------------------------------------------------------- #
# Assembler
# --------------------------------------------------------------------------- #
def build_mix(
    v11_parquet: Path,
    ikenie_paths: list[Path],
    nsfw_parquet: Path,
    *,
    v11_n: int | None,
    ikenie_repeat: int,
    ikenie_max_pages: int | None,
    nsfw_cap: float,
    seed: int,
    allow_nsfw_placeholder: bool,
    nsfw_max_rows: int | None = None,
) -> tuple[list[dict], dict]:
    """Assemble the full mix and return ``(shuffled_rows, stats)``."""
    backbone = load_v11_backbone(v11_parquet, v11_n, seed)
    ikenie = build_ikenie_rows(
        ikenie_paths, repeat=ikenie_repeat, also_plain=True,
        max_pages=ikenie_max_pages)
    nsfw, nsfw_info = load_nsfw_corpus(
        nsfw_parquet, allow_nsfw_placeholder, max_rows=nsfw_max_rows, seed=seed)

    per_source_before = {
        "v11_backbone": len(backbone),
        "ikenie_gold": len(ikenie),
        "nsfw_corpus": len(nsfw),
        "nsfw_corpus_info": nsfw_info,
    }

    rows = backbone + ikenie + nsfw
    assert_no_furube(rows)  # hard guard: eval set never trains

    # 1) refusal strip on the EN target side
    rows, refusal_dropped = refusal_strip(rows)
    # 2) NSFW hard cap
    rows, nsfw_dropped_n, nsfw_cap_info = enforce_nsfw_cap(rows, nsfw_cap, seed)
    nsfw_cap_info["dropped"] = nsfw_dropped_n
    # 3) deterministic shuffle
    random.Random(seed).shuffle(rows)

    parity = verify_builder_parity(n_cases=300, seed=seed)
    stats = compose_stats(
        rows, per_source_before, refusal_dropped, nsfw_cap_info, parity,
        extra={"seed": seed, "v11_parquet": str(v11_parquet),
               "ikenie_paths": [str(p) for p in ikenie_paths],
               "nsfw_parquet": str(nsfw_parquet)},
    )
    return rows, stats


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _write_sample(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_stats(stats: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def _print_stats(stats: dict) -> None:
    print("\n=== v13ship text-only SFT stats ===")
    print(f"TOTAL rows: {stats['total_rows']:,}")
    print("rows per source (final):")
    for k, v in stats["rows_per_source_final"].items():
        print(f"    {k:16s} {v:>8,}")
    print(f"NSFW rows: {stats['nsfw_rows_final']:,}  "
          f"({stats['nsfw_frac_final']*100:.2f}%  cap={stats['nsfw_cap']*100:.0f}%)")
    print(f"refusal rows dropped: {stats['refusal_dropped_count']:,}")
    for ex in stats["refusal_dropped_examples"][:5]:
        print(f"    - [{ex['src']}] {ex['en']!r}")
    p = stats["prompt_format_serve_parity"]
    print(f"prompt/serve parity: compared={p['compared']} mismatches={p['mismatches']} "
          f"instr_page={p['instr_page_match']} instr_plain={p['instr_plain_match']}")
    sc = stats["sample_serve_format_check"]
    print(f"sample serve-format check: pagectx={sc['pagectx_prompts']} "
          f"convctx={sc.get('convctx_prompts', 0)} plain={sc['plain_prompts']} "
          f"unrecognized={sc['unrecognized_prompts']} all_match={sc['all_match']}")
    ld = stats["length_distribution"]["en_chars"]
    print(f"en length (chars): p50={ld['p50']} p90={ld['p90']} "
          f"p99={ld['p99']} max={ld['max']}")


def _inspect(args) -> int:
    print("=== INSPECT: sources ===")
    checks = {
        "v11_backbone_parquet": Path(args.v11_parquet),
        "ikenie4_gold": DEFAULT_IKENIE[0],
        "ikenie5_gold": DEFAULT_IKENIE[1],
        "nsfw_corpus_parquet": Path(args.nsfw_parquet),
        "nsfw_placeholder": NSFW_PLACEHOLDER,
    }
    for name, p in checks.items():
        print(f"    [{'OK ' if p.exists() else 'MISSING'}] {name}: {p}")
    if Path(args.v11_parquet).exists():
        df = pl.read_parquet(args.v11_parquet)
        print(f"\n  v11 backbone: {df.height:,} rows, cols={df.columns}")
        assert df.columns == COLS, f"schema drift: {df.columns} != {COLS}"
        # scan the existing backbone for leaked refusals (should be ~0: it is human
        # gold + curated bitext, not raw instruct-model machine EN)
        hits = df.filter(pl.col("en").str.contains(REFUSAL_PATTERN)).height
        print(f"  backbone refusal scan: {hits:,}/{df.height:,} EN targets match "
              f"REFUSAL_RE ({hits/df.height*100:.3f}%)")
        nsfw = df.filter(pl.col("register_tag").is_in(list(NSFW_REGISTERS))).height
        print(f"  backbone NSFW register rows: {nsfw:,} ({nsfw/df.height*100:.1f}%)")
    st = refusal_classifier_selftest()
    print(f"\n  refusal-classifier self-test: passes={st['passes']} "
          f"(dropped {len(st['refusals_correctly_dropped'])}/5 refusals, "
          f"{len(st['benign_false_positives'])} benign false-positives)")
    print("\n=== INSPECT: prompt-format parity (train builder == serve builder) ===")
    parity = verify_builder_parity(n_cases=500, seed=args.seed)
    print(json.dumps(parity, ensure_ascii=False, indent=2)[:800])
    ok = parity["mismatches"] == 0 and parity["instr_page_match"] and parity["instr_plain_match"]
    print(f"\n  BYTE-MATCH v11 serve prompt: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def _build_v1_messages(args, ikenie_paths, out_dir: Path) -> int:
    """v1 (NSFW-free) SFT set -> native trainer messages JSONL.

    Mix = v11 backbone subsample (default 65000, seeded) + Ikenie gold x3 (full),
    NO mined NSFW (the corpus parquet is still missing; --allow-nsfw-placeholder is
    forced OFF here). Refusal-strip runs as a tripwire (drops the ~1 residual).
    Output is the trainer's native messages JSONL that drops into
    ``sft_qwen3vl_8b_imagectx.build_conversations``.
    """
    v11_n = args.backbone_subsample if args.backbone_subsample is not None else 65000
    # NSFW-FREE means no MINED corpus slice (parquet absent, placeholder OFF). The
    # v11 backbone still carries its proven ~18% NSFW-register distribution and
    # Ikenie gold is manga_nsfw -- those are kept. The 18% cap is DISABLED here
    # (cap=1.0): it exists to bound the mined machine-EN oversample, not human gold
    # or the shipped prod distribution, and would otherwise drop Ikenie gold x3.
    rows, stats = build_mix(
        Path(args.v11_parquet), ikenie_paths, Path(args.nsfw_parquet),
        v11_n=v11_n, ikenie_repeat=args.ikenie_repeat,
        ikenie_max_pages=args.ikenie_max_pages, nsfw_cap=1.0,
        seed=args.seed, allow_nsfw_placeholder=False, nsfw_max_rows=None,
    )
    # Validate the native messages format + round-trip on a probe slice.
    msgs_probe = [row_to_message(r) for r in rows[:2000]]
    fmt = messages_format_check(msgs_probe, rows[:2000])

    out_messages = out_dir / "data_v13ship_v1_messages.jsonl"       # gitignored (big)
    out_sample = out_dir / "data_v13ship_v1_messages.sample.jsonl"  # tracked (3 rows)
    stats_path = out_dir / "v13ship_v1_stats.json"

    size_bytes = _write_messages_jsonl(rows, out_messages)
    _write_sample([row_to_message(r) for r in rows[:3]], out_sample)

    by_src = stats["rows_per_source_final"]
    # In v1 mode NO separate mined-NSFW slice is added (load returned 0). The
    # "nsfw_corpus"-labelled rows here are `corpus_bitext:` rows ALREADY folded into
    # the v11fix8 backbone parquet (a labeling artifact of _source_of), NOT a slice
    # this build added -> so they all count as BACKBONE for the mix report.
    added_mined = stats["rows_per_source_before_filters"].get("nsfw_corpus", 0)
    embedded_corpus_bitext = by_src.get("nsfw_corpus", 0)
    ikenie_rows = by_src.get("ikenie_gold", 0)
    backbone_rows = len(rows) - ikenie_rows
    gold = sum(1 for r in rows if r.get("gold_flag"))
    stats["mode"] = "build-v1-messages"
    stats["format"] = "native trainer messages-jsonl (build_conversations schema)"
    stats["mix"] = {"backbone_rows": backbone_rows, "ikenie_gold_x3_rows": ikenie_rows,
                    "backbone_embedded_corpus_bitext_rows": embedded_corpus_bitext}
    stats["added_mined_nsfw_slice_rows"] = added_mined
    stats["mined_nsfw_free"] = added_mined == 0
    stats["nsfw_register_rows"] = stats["nsfw_rows_final"]  # backbone+ikenie register
    stats["nsfw_register_frac"] = stats["nsfw_frac_final"]
    stats["nsfw_cap_applied"] = False
    stats["gold_rows"] = gold
    stats["gold_frac"] = round(gold / len(rows), 4) if rows else 0.0
    stats["backbone_subsample"] = v11_n
    stats["messages_format_check"] = fmt
    stats["out_messages_jsonl"] = str(out_messages)
    stats["out_messages_bytes"] = size_bytes
    stats["out_messages_mb"] = round(size_bytes / 1e6, 2)
    _write_stats(stats, stats_path)

    _print_stats(stats)
    print(f"\nformat: native messages-jsonl (build_conversations)  "
          f"mined_nsfw_free={stats['mined_nsfw_free']} (added mined slice rows={added_mined})")
    print(f"messages format-check: {fmt}")
    print(f"mix: backbone={backbone_rows:,} (incl. {embedded_corpus_bitext:,} "
          f"pre-folded corpus_bitext) + ikenie_gold_x3={ikenie_rows:,}")
    print(f"gold rows: {gold:,} ({stats['gold_frac']*100:.1f}%)  "
          f"nsfw-register: {stats['nsfw_register_rows']:,} "
          f"({stats['nsfw_register_frac']*100:.1f}%, cap disabled)")
    print(f"wrote messages -> {out_messages} ({len(rows):,} rows, {stats['out_messages_mb']} MB)")
    print(f"wrote 3-row sample -> {out_sample}")
    print(f"wrote stats -> {stats_path}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inspect", action="store_true",
                      help="schema/format/parity checks only; NO output")
    mode.add_argument("--dry-run", action="store_true",
                      help="build a SMALL sample (+stats); NO full run")
    mode.add_argument("--full", action="store_true",
                      help="build the FULL parquet (multi-hour); gated")
    mode.add_argument("--build-v1-messages", action="store_true",
                      help="build the v1 (NSFW-free) SFT set in the trainer's NATIVE "
                           "messages-jsonl format (v11 backbone subsample + Ikenie "
                           "gold x3, NO mined NSFW); CPU-only, no training")
    ap.add_argument("--backbone-subsample", type=int, default=None,
                    help="v11 backbone subsample size (default keep-all; v1 uses 65000)")
    ap.add_argument("--v11-parquet", default=str(DEFAULT_V11_PARQUET))
    ap.add_argument("--nsfw-parquet", default=str(DEFAULT_NSFW_PARQUET))
    ap.add_argument("--ikenie", nargs="*", default=[str(p) for p in DEFAULT_IKENIE])
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--v11-n", type=int, default=None,
                    help="backbone subsample size (default: dry=180, full=60000)")
    ap.add_argument("--ikenie-repeat", type=int, default=3)
    ap.add_argument("--ikenie-max-pages", type=int, default=None,
                    help="cap ikenie pages/chapter (default: dry=6, full=all)")
    ap.add_argument("--nsfw-cap", type=float, default=NSFW_CAP)
    ap.add_argument("--nsfw-max-rows", type=int, default=None,
                    help="cap loaded NSFW-corpus rows (default: dry=60, full=all)")
    ap.add_argument("--sample-n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--allow-nsfw-placeholder", action="store_true",
                    help="fall back to the ikenie-derived validation sample when "
                         "the mined NSFW parquet is absent (clearly flagged)")
    args = ap.parse_args(argv)

    if args.inspect:
        return _inspect(args)

    ikenie_paths = [Path(p) for p in args.ikenie]
    out_dir = Path(args.out_dir)
    sample_path = out_dir / "data_v13ship_textsft.sample.jsonl"
    stats_path = out_dir / "v13ship_stats.json"

    if args.build_v1_messages:
        return _build_v1_messages(args, ikenie_paths, out_dir)

    if args.dry_run:
        v11_n = args.v11_n if args.v11_n is not None else 240
        ikenie_max = args.ikenie_max_pages if args.ikenie_max_pages is not None else 8
        nsfw_max = args.nsfw_max_rows if args.nsfw_max_rows is not None else 60
        rows, stats = build_mix(
            Path(args.v11_parquet), ikenie_paths, Path(args.nsfw_parquet),
            v11_n=v11_n, ikenie_repeat=args.ikenie_repeat,
            ikenie_max_pages=ikenie_max, nsfw_cap=args.nsfw_cap, seed=args.seed,
            allow_nsfw_placeholder=args.allow_nsfw_placeholder,
            nsfw_max_rows=nsfw_max,
        )
        sample = rows[: args.sample_n]
        # The 18% cap is enforced on the FULL mix; a shuffled head slice can drift
        # above it, so re-apply the cap to the WRITTEN sample too (keeps the
        # deliverable itself <= 18% NSFW).
        sample, _sn, _si = enforce_nsfw_cap(sample, args.nsfw_cap, args.seed)
        # recompute stats over the WRITTEN sample so the report matches the file
        sample_stats = compose_stats(
            sample, stats["rows_per_source_before_filters"],
            [], stats["nsfw_cap_info"], stats["prompt_format_serve_parity"],
            extra={"mode": "dry-run", "sample_rows": len(sample),
                   "full_mix_rows_after_filters": len(rows),
                   "full_mix_nsfw_frac_capped": stats["nsfw_frac_final"],
                   "seed": args.seed})
        sample_stats["refusal_dropped_count"] = stats["refusal_dropped_count"]
        sample_stats["refusal_dropped_examples"] = stats["refusal_dropped_examples"]
        _write_sample(sample, sample_path)
        _write_stats(sample_stats, stats_path)
        _print_stats(sample_stats)
        print(f"\nwrote sample -> {sample_path} ({len(sample)} rows)")
        print(f"wrote stats  -> {stats_path}")
        return 0

    # --full
    v11_n = args.v11_n if args.v11_n is not None else 60000
    ikenie_max = args.ikenie_max_pages  # default all
    rows, stats = build_mix(
        Path(args.v11_parquet), ikenie_paths, Path(args.nsfw_parquet),
        v11_n=v11_n, ikenie_repeat=args.ikenie_repeat,
        ikenie_max_pages=ikenie_max, nsfw_cap=args.nsfw_cap, seed=args.seed,
        allow_nsfw_placeholder=args.allow_nsfw_placeholder,
        nsfw_max_rows=args.nsfw_max_rows,
    )
    out_parquet = out_dir / "data_v13ship_textsft.parquet"
    out_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).select(COLS).write_parquet(out_parquet)
    _write_sample(rows[: args.sample_n], sample_path)
    stats["mode"] = "full"
    stats["out_parquet"] = str(out_parquet)
    _write_stats(stats, stats_path)
    _print_stats(stats)
    print(f"\nwrote parquet -> {out_parquet} ({len(rows):,} rows)")
    print(f"wrote sample  -> {sample_path}")
    print(f"wrote stats   -> {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
