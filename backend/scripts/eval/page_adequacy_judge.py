#!/usr/bin/env python
"""VLM page-adequacy judge (gate signal #4).

WHAT THIS IS
------------
A *visual* adequacy metric for manga MT. A vision LLM is shown the ACTUAL page
image plus, for that page, the numbered Japanese source lines and the numbered
candidate English lines. It counts, per page:

    * OMISSIONS      -- JP meaning silently dropped from the EN
    * HALLUCINATIONS -- EN content with no basis in the JP or the page
    * per-line adequacy verdict -- adequate / minor / major

This is robust to garbled OCR and noisy references because the judge reads the
real page, not just the (often broken) OCR text. It doubles as a visual metric.

It is DELIBERATELY NOT the GEMBA-MQM prompt. MQM scores a single (src, ref, hyp)
triple against a *reference*; this judge scores a whole page against the *image*
and asks the adequacy questions (omission / hallucination / line adequacy) that
the ship gate cares about.

INPUT CONTRACT (decoupled from any generator)
---------------------------------------------
The judge consumes a "candidates JSONL" where **each row is one PAGE**::

    {
      "page_img": "/abs/path/to/002.jpg",   # original source page (JP text)
      "page_id":  "furube_p1:p02",            # opaque page identifier
      "items": [
        {"idx": 0, "jp": "...", "gold_en": "...", "cand_en": "..."},
        {"idx": 1, "jp": "...", "gold_en": "...", "cand_en": "..."},
        ...
      ]
    }

Nothing about how ``cand_en`` was produced leaks into the judge. ``gold_en`` is
carried for reference/debugging only -- the judge grades ``cand_en`` against the
IMAGE, not against ``gold_en``.

ADAPTER (build the page contract from the Furube gold + a flat candidate file)
------------------------------------------------------------------------------
``build_pages_from_gold`` assembles the page contract from:

  (a) one or more ``gold_furube_p*.jsonl`` files. Each row is one bubble::

          {"jp": "...", "en": "<gold EN>", "src": "furube_p1:p02:idx0", ...}

      Rows are grouped into pages by the first two ``:``-separated parts of
      ``src`` (``furube_p1:p02``). The bubble index is the trailing ``idxN``.

  (b) a flat candidate file keyed by ``src`` -- the shape the POV generator
      emits::

          {"src": "furube_p1:p02:idx0", "output": "<candidate EN>"}

      (``translation`` / ``cand_en`` / ``en`` are accepted as aliases for
      ``output``.) A missing ``src`` in the candidate file yields an empty
      ``cand_en`` for that bubble (i.e. the generator omitted it).

The source page image is resolved from ``page_image_root`` by the page number:
``furube_p1:p02`` -> ``<root>/002.jpg`` (zero-padded to 3 digits; extension
configurable). This is verified against the bench log: gold page ``p02`` is the
source file ``002.jpg``.

Pass ``--gold-as-candidate`` to use each bubble's gold EN as the candidate --
this smokes the plumbing (should score ~all-adequate, 0 omission/hallucination).

JUDGE MODEL / ENDPOINT
----------------------
``--judge-model`` (default ``qwen3vl_ablit8b``) + ``--judge-base-url`` (default
the box at ``http://100.64.235.63:8001/v1``). The base URL is threaded through
``OpenAIJudge`` via its existing ``OPENAI_BASE_URL`` plumbing.

SELF-GRADING BIAS: if the judge is served by the SAME endpoint that serves the
model under test, the judge is grading (a sibling of) itself. The gate is a DIFF
of two candidate files under the SAME judge (v1 vs v11fix8), which cancels most
of the shared bias -- but the warning is printed so the reader knows.

GATE (compare mode)
-------------------
``--compare v1.jsonl v11fix8.jsonl`` runs the judge on both and prints a
side-by-side table + deltas. The gate (v1 is the candidate, v11fix8 the
incumbent) PASSES iff:

    * v1 adequacy (adequate fraction) >= v11fix8 adequacy, AND
    * v1 omission rate      <= v11fix8 omission rate, AND
    * v1 hallucination rate <= v11fix8 hallucination rate.

CLI
---
    # dry run (no network; renders messages minus image bytes for 1 page)
    python backend/scripts/eval/page_adequacy_judge.py \
        --gold backend/scripts/eval/data/furube/gold_furube_p1.jsonl \
        --page-image-root "/mnt/nas/.../486860_Haha to Ochite Iku Part 1" \
        --gold-as-candidate --limit-pages 1 --dry-run

    # single-file scoring against the box
    python backend/scripts/eval/page_adequacy_judge.py \
        --gold .../gold_furube_p1.jsonl --candidates cand_v1.jsonl \
        --page-image-root "/mnt/nas/..." --out adequacy_v1.json

    # gate compare
    python backend/scripts/eval/page_adequacy_judge.py \
        --gold .../gold_furube_p1.jsonl \
        --page-image-root "/mnt/nas/..." \
        --compare cand_v1.jsonl cand_v11fix8.jsonl --out gate.json
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from loguru import logger

# Reuse existing, backward-compatible plumbing from the text MQM judge.
from backend.scripts.eval.gemba_mqm_judge import (  # noqa: E402
    DryRunJudge,
    OpenAIJudge,
    _JudgeBase,
    _read_jsonl,
)

# Reuse the image data-URL helper rather than duplicating it (used as the
# no-downscale fast path inside ``_encode_page_image``).
from backend.scripts.eval.transcribe_gt_vision import _image_to_data_url  # noqa: E402

DEFAULT_JUDGE_MODEL = "qwen3vl_ablit8b"
DEFAULT_JUDGE_BASE_URL = "http://100.64.235.63:8001/v1"
# The box that serves the models under test (v1 / v11fix8 / qwen3vl_ablit8b).
DEFAULT_SERVED_BY = "http://100.64.235.63:8001/v1"

MAX_IMAGE_LONG_SIDE = 1024

VERDICTS = ("adequate", "minor", "major")


# ---------------------------------------------------------------------------
# Adequacy prompt (NEW -- not the MQM prompt).
# ---------------------------------------------------------------------------

ADEQUACY_SYSTEM = (
    "You are a bilingual (Japanese-English) manga translation adequacy judge. "
    "You are shown a manga page image, the Japanese source lines read off that "
    "page (numbered), and a candidate English translation (numbered to match). "
    "Judge whether the English adequately conveys what the page actually says. "
    "The Japanese OCR may be garbled; when it is, trust the IMAGE. Be strict but "
    "fair. Respond with ONE JSON object and nothing else."
)

ADEQUACY_USER_TEMPLATE = """Here is a manga page and its lines.

Japanese source lines (numbered):
{jp_block}

Candidate English lines (numbered, same indices):
{en_block}

For EACH numbered line decide:
  - "verdict": "adequate" (meaning preserved), "minor" (understandable but
    imperfect), or "major" (meaning broken/wrong/empty).
  - "omission": true if Japanese meaning present on the page is SILENTLY DROPPED
    from the English (empty/blank English for a line that says something, or a
    clause dropped).
  - "hallucination": true if the English adds content with NO basis in the
    Japanese line or anywhere on the page.
  - "note": short reason (<= 12 words).

Return EXACTLY this JSON schema, no prose, no markdown fences:
{{
  "lines": [
    {{"idx": <int>, "verdict": "adequate|minor|major",
      "omission": <bool>, "hallucination": <bool>, "note": "<string>"}}
  ],
  "page_omissions": <int>,
  "page_hallucinations": <int>
}}
"idx" must match the numbers above. Count "page_omissions"/"page_hallucinations"
as the number of lines with that flag true."""


def _numbered_block(items: list[dict[str, Any]], field: str) -> str:
    lines = []
    for it in items:
        val = it.get(field) or ""
        val = str(val).replace("\n", " ").strip()
        lines.append(f"[{it['idx']}] {val if val else '(blank)'}")
    return "\n".join(lines) if lines else "(none)"


def build_adequacy_messages(
    page: dict[str, Any], image_data_url: str
) -> list[dict[str, Any]]:
    """Multimodal chat messages for one page (same parts-list shape the
    surgical ``gemba_mqm_judge._build_messages`` upgrade produces)."""
    items = page["items"]
    user_text = ADEQUACY_USER_TEMPLATE.format(
        jp_block=_numbered_block(items, "jp"),
        en_block=_numbered_block(items, "cand_en"),
    )
    return [
        {"role": "system", "content": ADEQUACY_SYSTEM},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        },
    ]


# ---------------------------------------------------------------------------
# Defensive JSON parsing
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

# Per-line salvage: one {...} object carrying an "idx" + "verdict". Used when
# the whole-object parse fails (truncation, a stray control char, etc.). Qwen3
# reasoning models occasionally emit malformed wrappers but well-formed rows.
_LINE_OBJ_RE = re.compile(r"\{[^{}]*?\"idx\"\s*:\s*\d+[^{}]*?\}", re.DOTALL)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of one JSON object from an LLM response."""
    if not text:
        return None
    text = _THINK_RE.sub("", text)
    candidates: list[str] = []
    m = _FENCE_RE.search(text)
    if m:
        candidates.append(m.group(1))
    # First-brace .. last-brace slice.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start : end + 1])
    candidates.append(text)
    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            continue
    # Salvage: rebuild {"lines": [...]} from individually-parseable line objects.
    lines: list[dict[str, Any]] = []
    for m in _LINE_OBJ_RE.finditer(text):
        try:
            ln = json.loads(m.group(0))
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(ln, dict) and "idx" in ln:
            lines.append(ln)
    if lines:
        return {"lines": lines, "_salvaged": True}
    return None


def parse_adequacy_response(
    text: str, items: list[dict[str, Any]]
) -> dict[str, Any]:
    """Parse the judge JSON into per-line records + page counts.

    Defensive: unparseable / missing lines are recorded as ``major`` with an
    ``unparsed`` flag so a broken judge response is visible, not silently zeroed.
    """
    obj = _extract_json_object(text)
    by_idx: dict[int, dict[str, Any]] = {}
    if obj and isinstance(obj.get("lines"), list):
        for ln in obj["lines"]:
            if not isinstance(ln, dict):
                continue
            try:
                idx = int(ln.get("idx"))
            except (TypeError, ValueError):
                continue
            verdict = str(ln.get("verdict", "")).strip().lower()
            if verdict not in VERDICTS:
                verdict = "major"
            by_idx[idx] = {
                "idx": idx,
                "verdict": verdict,
                "omission": bool(ln.get("omission")),
                "hallucination": bool(ln.get("hallucination")),
                "note": str(ln.get("note", ""))[:200],
                "unparsed": False,
            }

    per_line: list[dict[str, Any]] = []
    for it in items:
        idx = int(it["idx"])
        rec = by_idx.get(idx)
        if rec is None:
            rec = {
                "idx": idx,
                "verdict": "major",
                "omission": False,
                "hallucination": False,
                "note": "no judge line for idx",
                "unparsed": True,
            }
        per_line.append(rec)

    omissions = sum(1 for r in per_line if r["omission"])
    hallucinations = sum(1 for r in per_line if r["hallucination"])
    dist = {v: sum(1 for r in per_line if r["verdict"] == v) for v in VERDICTS}
    return {
        "per_line": per_line,
        "n_lines": len(per_line),
        "omissions": omissions,
        "hallucinations": hallucinations,
        "verdict_dist": dist,
        "parsed_ok": obj is not None,
        "raw": text,
    }


# ---------------------------------------------------------------------------
# Image encoding (downscale <= 1024px long side)
# ---------------------------------------------------------------------------


def _encode_page_image(path: Path, max_long_side: int = MAX_IMAGE_LONG_SIDE) -> str:
    """Return a base64 data URL for the page, downscaled so the long side is
    <= ``max_long_side``. Small images are passed through the reused
    ``_image_to_data_url`` unchanged."""
    from PIL import Image

    with Image.open(path) as im:
        w, h = im.size
        long_side = max(w, h)
        if long_side <= max_long_side:
            return _image_to_data_url(path)
        scale = max_long_side / float(long_side)
        new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
        im = im.convert("RGB").resize(new_size, Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="WEBP", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/webp;base64,{b64}"


# ---------------------------------------------------------------------------
# Adapter: build the page contract from Furube gold + flat candidate file
# ---------------------------------------------------------------------------

_IDX_RE = re.compile(r"idx(\d+)")

_CAND_OUTPUT_FIELDS = ("output", "translation", "cand_en", "en", "translation_en")


def _page_id_of(src: str) -> str:
    parts = src.split(":")
    return ":".join(parts[:2]) if len(parts) >= 2 else src


def _bubble_idx_of(src: str, fallback: int) -> int:
    m = _IDX_RE.search(src)
    return int(m.group(1)) if m else fallback


def _page_number_of(page_id: str) -> int | None:
    """furube_p1:p02 -> 2 (the source-image page number)."""
    tail = page_id.split(":")[-1]
    m = re.search(r"(\d+)", tail)
    return int(m.group(1)) if m else None


def _resolve_page_image(
    page_id: str, page_image_root: Path, ext: str, pad: int
) -> Path | None:
    n = _page_number_of(page_id)
    if n is None:
        return None
    return page_image_root / f"{n:0{pad}d}{ext}"


def _load_candidate_map(candidate_path: Path) -> dict[str, str]:
    """Flat candidate file keyed by ``src`` -> candidate EN string."""
    out: dict[str, str] = {}
    for row in _read_jsonl(candidate_path):
        src = row.get("src")
        if not src:
            continue
        val = ""
        for f in _CAND_OUTPUT_FIELDS:
            if row.get(f):
                val = str(row[f])
                break
        out[str(src)] = val
    return out


def build_pages_from_gold(
    gold_paths: list[Path],
    *,
    page_image_root: Path,
    candidate_path: Path | None = None,
    gold_as_candidate: bool = False,
    image_ext: str = ".jpg",
    page_pad: int = 3,
    require_image: bool = True,
) -> list[dict[str, Any]]:
    """Assemble the page-level judge contract (see module docstring)."""
    cand_map: dict[str, str] = {}
    if candidate_path is not None:
        cand_map = _load_candidate_map(candidate_path)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for gp in gold_paths:
        for i, row in enumerate(_read_jsonl(gp)):
            src = str(row.get("src", ""))
            page_id = _page_id_of(src)
            grouped[page_id].append(
                {
                    "idx": _bubble_idx_of(src, i),
                    "jp": row.get("jp", ""),
                    "gold_en": row.get("en", "") or row.get("gold_en", ""),
                    "src": src,
                }
            )

    pages: list[dict[str, Any]] = []
    for page_id in sorted(grouped):
        items = sorted(grouped[page_id], key=lambda r: r["idx"])
        for it in items:
            if gold_as_candidate:
                it["cand_en"] = it["gold_en"]
            else:
                it["cand_en"] = cand_map.get(it["src"], "")
            it.pop("src", None)
        img_path = _resolve_page_image(page_id, page_image_root, image_ext, page_pad)
        if img_path is None or not img_path.exists():
            if require_image:
                logger.warning("skip page {} -- image not found: {}", page_id, img_path)
                continue
        pages.append(
            {
                "page_id": page_id,
                "page_img": str(img_path) if img_path else "",
                "items": items,
            }
        )
    return pages


# ---------------------------------------------------------------------------
# Dry-run judge: renders messages (minus image bytes) + returns valid stub JSON
# ---------------------------------------------------------------------------


def _redact_images(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    red: list[dict[str, Any]] = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            new_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    url = (part.get("image_url") or {}).get("url", "")
                    head = url[:32]
                    new_parts.append(
                        {"type": "image_url",
                         "image_url": {"url": f"<{len(url)} bytes: {head}...>"}}
                    )
                else:
                    new_parts.append(part)
            red.append({"role": m["role"], "content": new_parts})
        else:
            red.append(m)
    return red


class _AdequacyDryRunJudge(DryRunJudge):
    """Prints the fully-rendered adequacy messages (image bytes redacted) and
    returns a schema-valid all-adequate stub so the JSON parser is exercised."""

    def chat(self, messages: list[dict[str, Any]]) -> str:
        self._n = getattr(self, "_n", 0) + 1
        base_url = os.environ.get("OPENAI_BASE_URL") or DEFAULT_JUDGE_BASE_URL
        if self._n == 1:
            print("=" * 70, file=sys.stderr)
            print(f"DRY RUN | target={self.target}", file=sys.stderr)
            print(f"        | OPENAI_BASE_URL={base_url}", file=sys.stderr)
            print("        | rendered messages (image bytes redacted):", file=sys.stderr)
            print(
                json.dumps(_redact_images(messages), indent=2, ensure_ascii=False),
                file=sys.stderr,
            )
            print("=" * 70, file=sys.stderr)
        # Build a valid stub keyed to the indices actually present in the prompt.
        idxs = _extract_idxs_from_messages(messages)
        lines = [
            {"idx": i, "verdict": "adequate", "omission": False,
             "hallucination": False, "note": "dry-run stub"}
            for i in idxs
        ]
        return json.dumps(
            {"lines": lines, "page_omissions": 0, "page_hallucinations": 0}
        )


def _extract_idxs_from_messages(messages: list[dict[str, Any]]) -> list[int]:
    text = ""
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            text += c
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and part.get("type") == "text":
                    text += part.get("text", "")
    return [int(x) for x in re.findall(r"\[(\d+)\]", text)]


# ---------------------------------------------------------------------------
# Run + aggregate
# ---------------------------------------------------------------------------


def run_page_adequacy(
    pages: list[dict[str, Any]],
    *,
    judge: _JudgeBase,
    max_long_side: int = MAX_IMAGE_LONG_SIDE,
) -> dict[str, Any]:
    per_page: list[dict[str, Any]] = []
    tot_lines = tot_om = tot_hall = 0
    dist_tot = {v: 0 for v in VERDICTS}

    for page in pages:
        try:
            data_url = _encode_page_image(Path(page["page_img"]), max_long_side)
        except Exception as e:  # noqa: BLE001 - log original, keep going
            logger.exception("image encode failed for {}: {}", page["page_id"], e)
            continue
        messages = build_adequacy_messages(page, data_url)
        try:
            resp = judge.chat(messages)
        except Exception as e:  # noqa: BLE001 - log original, keep going
            logger.exception("judge failed on page {}: {}", page["page_id"], e)
            resp = ""
        parsed = parse_adequacy_response(resp, page["items"])
        per_page.append(
            {
                "page_id": page["page_id"],
                "page_img": page["page_img"],
                "n_lines": parsed["n_lines"],
                "omissions": parsed["omissions"],
                "hallucinations": parsed["hallucinations"],
                "verdict_dist": parsed["verdict_dist"],
                "parsed_ok": parsed["parsed_ok"],
                "per_line": parsed["per_line"],
                "raw": parsed["raw"],
            }
        )
        tot_lines += parsed["n_lines"]
        tot_om += parsed["omissions"]
        tot_hall += parsed["hallucinations"]
        for v in VERDICTS:
            dist_tot[v] += parsed["verdict_dist"][v]

    n = max(tot_lines, 1)
    return {
        "n_pages": len(per_page),
        "n_lines": tot_lines,
        "omissions": tot_om,
        "hallucinations": tot_hall,
        "omission_rate": tot_om / n,
        "hallucination_rate": tot_hall / n,
        "verdict_dist": dist_tot,
        "adequacy": dist_tot["adequate"] / n,       # fraction of lines "adequate"
        "minor_rate": dist_tot["minor"] / n,
        "major_rate": dist_tot["major"] / n,
        "per_page": per_page,
    }


def _summ(agg: dict[str, Any]) -> dict[str, Any]:
    return {
        k: agg[k]
        for k in (
            "n_pages", "n_lines", "omissions", "hallucinations",
            "omission_rate", "hallucination_rate", "verdict_dist",
            "adequacy", "minor_rate", "major_rate",
        )
    }


def compare_gate(v1: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    """v1 = candidate under test, base = incumbent (v11fix8)."""
    d_adeq = v1["adequacy"] - base["adequacy"]
    d_om = v1["omission_rate"] - base["omission_rate"]
    d_hall = v1["hallucination_rate"] - base["hallucination_rate"]
    passed = (d_adeq >= 0) and (d_om <= 0) and (d_hall <= 0)
    return {
        "delta_adequacy": d_adeq,
        "delta_omission_rate": d_om,
        "delta_hallucination_rate": d_hall,
        "gate_pass": passed,
        "criteria": (
            "PASS iff adequacy>=incumbent AND omission_rate<=incumbent "
            "AND hallucination_rate<=incumbent"
        ),
    }


def _print_compare_table(v1: dict[str, Any], base: dict[str, Any], gate: dict[str, Any]) -> None:
    rows = [
        ("adequacy", v1["adequacy"], base["adequacy"], gate["delta_adequacy"]),
        ("omission_rate", v1["omission_rate"], base["omission_rate"], gate["delta_omission_rate"]),
        ("hallucination_rate", v1["hallucination_rate"], base["hallucination_rate"], gate["delta_hallucination_rate"]),
        ("minor_rate", v1["minor_rate"], base["minor_rate"], v1["minor_rate"] - base["minor_rate"]),
        ("major_rate", v1["major_rate"], base["major_rate"], v1["major_rate"] - base["major_rate"]),
    ]
    print("\n" + "=" * 72, file=sys.stderr)
    print(f"{'metric':<22}{'v1':>12}{'v11fix8':>12}{'delta':>12}", file=sys.stderr)
    print("-" * 72, file=sys.stderr)
    for name, a, b, d in rows:
        print(f"{name:<22}{a:>12.4f}{b:>12.4f}{d:>+12.4f}", file=sys.stderr)
    print("-" * 72, file=sys.stderr)
    print(f"GATE: {'PASS' if gate['gate_pass'] else 'FAIL'}  ({gate['criteria']})",
          file=sys.stderr)
    print("=" * 72, file=sys.stderr)


# ---------------------------------------------------------------------------
# Judge construction + self-grading-bias warning
# ---------------------------------------------------------------------------


def _build_adequacy_judge(
    judge_model: str, judge_base_url: str, *, dry_run: bool
) -> _JudgeBase:
    os.environ["OPENAI_BASE_URL"] = judge_base_url
    # vLLM accepts any bearer token; supply a placeholder if none set.
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    if dry_run:
        return _AdequacyDryRunJudge(target=f"openai:{judge_model}")
    return OpenAIJudge(judge_model)


def _warn_self_grading_bias(judge_base_url: str, served_by: str) -> None:
    ju = urlparse(judge_base_url)
    sv = urlparse(served_by)
    if (ju.hostname, ju.port) == (sv.hostname, sv.port):
        logger.warning(
            "SELF-GRADING BIAS: judge endpoint ({}) == the endpoint serving the "
            "model(s) under test. The judge is grading a sibling of itself. The "
            "gate is a DIFF of two candidate files under this SAME judge, which "
            "cancels most shared bias -- but absolute adequacy numbers are "
            "optimistic. Interpret deltas, not absolutes.",
            judge_base_url,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="VLM page-adequacy judge (gate signal #4).")
    p.add_argument("--gold", type=Path, nargs="+", required=True,
                   help="gold_furube_p*.jsonl file(s).")
    p.add_argument("--page-image-root", type=Path, required=True,
                   help="Directory holding the source page images (NNN.jpg).")
    p.add_argument("--candidates", type=Path, default=None,
                   help="Flat candidate file keyed by src ({src, output}).")
    p.add_argument("--gold-as-candidate", action="store_true",
                   help="Use gold EN as the candidate (plumbing smoke).")
    p.add_argument("--compare", type=Path, nargs=2, default=None,
                   metavar=("V1", "V11FIX8"),
                   help="Two flat candidate files -> side-by-side gate.")
    p.add_argument("--image-ext", type=str, default=".jpg")
    p.add_argument("--page-pad", type=int, default=3)
    p.add_argument("--max-long-side", type=int, default=MAX_IMAGE_LONG_SIDE)
    p.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL)
    p.add_argument("--judge-base-url", type=str, default=DEFAULT_JUDGE_BASE_URL)
    p.add_argument("--served-by", type=str, default=DEFAULT_SERVED_BY,
                   help="Endpoint serving the model(s) under test (bias check).")
    p.add_argument("--limit-pages", type=int, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="No network: render messages (image bytes redacted) + stub JSON.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _warn_self_grading_bias(args.judge_base_url, args.served_by)

    def _pages_for(candidate_path: Path | None, gold_as_cand: bool) -> list[dict[str, Any]]:
        pages = build_pages_from_gold(
            args.gold,
            page_image_root=args.page_image_root,
            candidate_path=candidate_path,
            gold_as_candidate=gold_as_cand,
            image_ext=args.image_ext,
            page_pad=args.page_pad,
        )
        if args.limit_pages:
            pages = pages[: args.limit_pages]
        return pages

    judge = _build_adequacy_judge(
        args.judge_model, args.judge_base_url, dry_run=args.dry_run
    )

    report: dict[str, Any]
    if args.compare is not None:
        v1_path, base_path = args.compare
        v1_pages = _pages_for(v1_path, gold_as_cand=False)
        base_pages = _pages_for(base_path, gold_as_cand=False)
        v1_agg = run_page_adequacy(v1_pages, judge=judge, max_long_side=args.max_long_side)
        base_agg = run_page_adequacy(base_pages, judge=judge, max_long_side=args.max_long_side)
        gate = compare_gate(v1_agg, base_agg)
        _print_compare_table(v1_agg, base_agg, gate)
        report = {
            "mode": "compare",
            "judge_model": args.judge_model,
            "judge_base_url": args.judge_base_url,
            "v1": _summ(v1_agg),
            "v11fix8": _summ(base_agg),
            "gate": gate,
            "v1_detail": v1_agg,
            "v11fix8_detail": base_agg,
        }
    else:
        pages = _pages_for(args.candidates, gold_as_cand=args.gold_as_candidate)
        agg = run_page_adequacy(pages, judge=judge, max_long_side=args.max_long_side)
        logger.info(
            "adequacy: pages={} lines={} adequate={:.3f} omit_rate={:.3f} halluc_rate={:.3f}",
            agg["n_pages"], agg["n_lines"], agg["adequacy"],
            agg["omission_rate"], agg["hallucination_rate"],
        )
        report = {
            "mode": "single",
            "judge_model": args.judge_model,
            "judge_base_url": args.judge_base_url,
            "dry_run": bool(args.dry_run),
            **agg,
        }

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        logger.info("wrote {}", args.out)
    else:
        # Print a compact summary to stdout when not writing a file.
        if report["mode"] == "single":
            print(json.dumps({
                k: report[k] for k in (
                    "n_pages", "n_lines", "adequacy", "omission_rate",
                    "hallucination_rate", "verdict_dist")
            }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
