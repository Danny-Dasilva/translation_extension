#!/usr/bin/env python3
"""Phase-1 IMAGE-CONTEXT POC dataset builder (PBP-VIS-NUM format).

Adapts ``build_v12vision_poc.py`` for the Qwen3-VL-8B image-context LoRA POC
(plan: thoughts/shared/plans/2026-06-30_image-context-vlm-finetune.md, §3 + §7).

WHAT THIS PRODUCES
------------------
Two PARALLEL Qwen3-VL chat datasets (pre-mortem #1 — the FT-text control arm):

  data_poc_imageon.jsonl   Ikenie GOLD pages WITH a numbered-bubble image block,
                           + an image-ABSENT register slice (NSFW corpus) + an
                           image-ABSENT v11 text backbone.
  data_poc_imageoff.jsonl  IDENTICAL text, the Ikenie pages have NO image block.
                           The register + text-backbone rows are BYTE-IDENTICAL
                           in both files (only the Ikenie image block differs).

    image value  =  FT(imageon)  −  FT(imageoff)     (isolates the image)

PBP-VIS-NUM row (Ikenie gold page), following Lippmann COLING 2025:

  user turn:
    [IMAGE block] = the raw page with every OCR text-box REDACTED (glyphs filled)
                    and each SUPERVISED (gold) bubble overlaid with its reading-
                    order number 1..N on a small plate at the bubble centroid.
                    Redaction forces the model to use the SCENE, not in-bubble OCR.
    [TEXT block]  = the byte-exact v11 serve page-context prompt with a NUMBERED
                    JP list  "1. <jp>\n2. <jp>\n...\nN. <jp>"  (reading order).
  assistant turn:
    the NUMBERED EN list  "1. <en>\n2. <en>\n...\nN. <en>"  (loss on assistant).

The numbered set = the page's GOLD bubbles (the ones with human EN). Numbers on
the image, the JP list, and the EN list are CO-INDEXED 1..N. POV supervision is
GOLD-only (pre-mortem #3): the register/backbone slices are machine EN and carry
NO POV signal — they are register + refusal-suppression + fluency only.

READING ORDER
-------------
Reading order = the production column-major RTL order. The pipeline froze that
order into each bubble's ``idx`` in ``bubbles.json`` (idx == the output of
``build_v11_dataset.manga_reading_order`` run on the full page), so we sort the
gold subset by ``idx`` to preserve the exact production order. When bubbles.json
is absent we fall back to ``manga_reading_order`` on the gold bboxes directly.

DATA SOURCES (all LOCAL)
------------------------
  * Ikenie GOLD (human EN, correct POV): ikenie4 + ikenie5 gold_q3.jsonl (~220
    pages). Furube is HELD OUT — it is the eval set (37 of the 44 POV cases).
  * NSFW corpus register slice: rows whose ``src`` starts with ``corpus_bitext:``
    in the v11fix8 page-context parquet (machine EN, register_tag manga_nsfw).
    These are the corpus_bitext machine-EN pairs already folded into v11fix8; a
    standalone corpus_bitext parquet is not built locally.  # VERIFY-ON-BOX note
  * v11 text backbone (SFW, image-absent): register_tag in the SFW set
    {manga_dialog, manga, novel, vn, synthetic} from the same parquet.

NSFW GUARDRAIL
--------------
Total NSFW share is capped at ≤18% of rows (the v12 36%-oversample regressed the
model into euphemism; v11/v11fix held ~16%). The NSFW slice is auto-trimmed to
satisfy the cap; the SFW v11 backbone dilutes it.

USAGE
-----
    python build_numbered_poc.py                 # full POC, default paths
    python build_numbered_poc.py --limit 6       # tiny smoke (6 ikenie pages)
    python build_numbered_poc.py --inspect       # build + print a preview, no heavy pull

All writes are LOCAL. Never write under /mnt/nas (the CIFS share reaps files).
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# paths + v11 serve-format contract (byte-identical to the text serving path)
# --------------------------------------------------------------------------- #
# this file lives at <repo>/backend/scripts/data/v12vision/ -> 4 levels up.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
BACKEND = os.path.join(REPO_ROOT, "backend")

# v11 page instruction — copied VERBATIM from the text serving path. Keep this
# byte-identical: a whitespace/marker drift silently degrades translation.
V11_PAGE_INSTR = (
    "Translate the marked line of this manga page from Japanese to English. "
    "Use the page context for speakers, pronouns, and continuity. "
    "Output only the translation of the marked line."
)

# chapter -> (gold_q3.jsonl, box-inspection root holding NNN/{01_source.webp,bubbles.json})
DEFAULT_CHAPTERS: Dict[str, Tuple[str, str]] = {
    "ikenie4": (
        os.path.join(BACKEND, "scripts/eval/data/ikenie4/gold_q3.jsonl"),
        os.path.join(BACKEND, ".bench/ikenie4_v11fix6_box_insp"),
    ),
    "ikenie5": (
        os.path.join(BACKEND, "scripts/eval/data/ikenie5/gold_q3.jsonl"),
        os.path.join(BACKEND, ".bench/ikenie5_v11fix6_box_insp"),
    ),
}

# v11fix8 page-context parquet holds BOTH text slices (corpus_bitext NSFW + SFW).
DEFAULT_V11_PARQUET = os.path.join(BACKEND, "scripts/data/v11fix8/data_v11fix8_pagecontext.parquet")

# register-tag partitions (aligned with corpus_bitext.format_rows.NSFW_TAGS).
NSFW_TAGS = {"manga_nsfw", "vn_eroge"}
SFW_BACKBONE_TAGS = {"manga_dialog", "manga", "novel", "vn", "synthetic"}

# reading order fallback (production column-major RTL) — reused verbatim so
# train == serve. Primary order is bubbles.json ``idx`` (the frozen output of
# exactly this function on the full page).
sys.path.insert(0, os.path.join(BACKEND, "scripts/data/v11"))
sys.path.insert(0, os.path.join(BACKEND, "scripts/data/v11fix6"))
try:
    from build_v11_dataset import manga_reading_order  # type: ignore  # noqa: E402
except Exception:  # noqa: BLE001
    manga_reading_order = None  # type: ignore
try:
    from build_v11fix6_corrective import to_sentence_case  # type: ignore  # noqa: E402
except Exception:  # noqa: BLE001
    def to_sentence_case(text: str) -> str:  # type: ignore
        """Fallback recaser (all-caps -> sentence case) if the helper is absent."""
        t = (text or "").strip()
        return t[:1].upper() + t[1:].lower() if t and t.isupper() else t

_SRC_RE = re.compile(r"^(?P<chapter>\w+):p(?P<page>\d+):idx(?P<idx>\d+)\s*$")

# Hiragana / Katakana / CJK / half-width kana — used to drop gold EN targets that
# are actually untranslated JP (SFX like 'ズキッ') or judge-notes that leaked into
# the ``en`` field ('ごめんなさい = I'm sorry (...)'). ~11% of gold rows. These are
# bad POV supervision, so we skip the bubble entirely (keeps 1..N co-indexing).
_CJK_RE = re.compile(r"[぀-ヿ㐀-鿿ｦ-ﾟ]")
_CJK_MAX_FRAC = 0.2


def _cjk_fraction(s: str) -> float:
    t = re.sub(r"\s", "", s or "")
    return sum(1 for c in t if _CJK_RE.match(c)) / len(t) if t else 0.0

# in-repo + system font candidates for the number plates (first that loads wins).
_FONT_CANDIDATES = [
    os.path.join(REPO_ROOT, "public/fonts/Anton-Regular.ttf"),
    os.path.join(REPO_ROOT, "public/fonts/Bangers-Regular.ttf"),
    "/mnt/ssd/home/danny/.pyenv/versions/3.12.10/lib/python3.12/site-packages/cv2/qt/fonts/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


# --------------------------------------------------------------------------- #
# small io helpers
# --------------------------------------------------------------------------- #
def parse_src(src: str) -> Optional[Tuple[str, int, int]]:
    m = _SRC_RE.match(src or "")
    return (m.group("chapter"), int(m.group("page")), int(m.group("idx"))) if m else None


def load_jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def page_dir(bench_root: str, page: int) -> str:
    return os.path.join(bench_root, f"{page:03d}")


def _bbox_to_xyxy(b: Any) -> Optional[Tuple[float, float, float, float]]:
    """Accept a dict {'minX','minY','maxX','maxY'} or its stringified form."""
    if isinstance(b, str):
        try:
            b = ast.literal_eval(b)
        except Exception:  # noqa: BLE001
            return None
    if not isinstance(b, dict):
        return None
    try:
        return (float(b["minX"]), float(b["minY"]), float(b["maxX"]), float(b["maxY"]))
    except Exception:  # noqa: BLE001
        return None


def load_bubbles(bench_root: str, page: int) -> Optional[List[dict]]:
    bj = os.path.join(page_dir(bench_root, page), "bubbles.json")
    if not os.path.exists(bj):
        return None
    return json.load(open(bj, encoding="utf-8"))


def find_image(bench_root: str, page: int) -> Optional[str]:
    pdir = page_dir(bench_root, page)
    for name in ("01_source.webp", "01_source.png", "01_source.jpg"):
        p = os.path.join(pdir, name)
        if os.path.exists(p):
            return os.path.abspath(p)
    return None


# --------------------------------------------------------------------------- #
# byte-exact v11 page-context prompt with a NUMBERED JP list
# --------------------------------------------------------------------------- #
def build_numbered_prompt(jp_lines: List[str]) -> str:
    """v11 serve-format block: instruction + numbered ``Page:`` list.

    Byte-identical to the text serve contract (``{instr}\\n\\nPage:\\n1. ...``);
    the per-line ``Translate line k`` suffix is dropped (page-level row).
    """
    numbered = "\n".join(f"{i + 1}. {ln}" for i, ln in enumerate(jp_lines))
    return f"{V11_PAGE_INSTR}\n\nPage:\n{numbered}"


def build_numbered_target(en_lines: List[str]) -> str:
    return "\n".join(f"{i + 1}. {ln}" for i, ln in enumerate(en_lines))


# --------------------------------------------------------------------------- #
# numbered-bubble image renderer
# --------------------------------------------------------------------------- #
_FONT_CACHE: Dict[int, Any] = {}


def _load_font(size: int):
    from PIL import ImageFont
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    font = None
    for cand in _FONT_CANDIDATES:
        if os.path.exists(cand):
            try:
                font = ImageFont.truetype(cand, size)
                break
            except Exception:  # noqa: BLE001
                continue
    if font is None:
        font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


def render_numbered_image(
    src_image: str,
    redact_boxes: List[Tuple[float, float, float, float]],
    numbered: List[Tuple[int, Tuple[float, float, float, float]]],
    out_path: str,
) -> Tuple[int, int]:
    """Redact every OCR box (fill glyphs) and draw reading-order numbers.

    redact_boxes : ALL page OCR boxes (xyxy) — filled white to remove glyphs.
    numbered     : (label, xyxy) for each SUPERVISED bubble — a red plate + white
                   number is drawn at the box centroid (over the redacted box).
    Returns the rendered (width, height).
    """
    from PIL import Image, ImageDraw

    img = Image.open(src_image).convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img)

    # 1) redact: fill each OCR text box white so no JP glyphs leak. Pad each box
    # proportional to its own size (+ a floor) so trailing vertical-kana that
    # bleed just past the tight OCR bbox are covered too.
    floor = max(2, round(0.003 * min(W, H)))
    for (x0, y0, x1, y1) in redact_boxes:
        px = max(floor, round(0.08 * (x1 - x0)))
        py = max(floor, round(0.08 * (y1 - y0)))
        draw.rectangle(
            [max(0, x0 - px), max(0, y0 - py), min(W, x1 + px), min(H, y1 + py)],
            fill=(255, 255, 255),
        )

    # 2) draw the reading-order number plate at each supervised bubble centroid.
    r = max(11, round(0.016 * min(W, H)))          # plate radius
    font = _load_font(int(r * 1.5))
    for label, (x0, y0, x1, y1) in numbered:
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        txt = str(label)
        # multi-digit -> widen to a rounded rect so the number always fits.
        half_w = r if len(txt) == 1 else r + int(r * 0.6 * (len(txt) - 1))
        plate = [cx - half_w, cy - r, cx + half_w, cy + r]
        draw.rounded_rectangle(plate, radius=r, fill=(220, 30, 30), outline=(255, 255, 255), width=2)
        tb = draw.textbbox((0, 0), txt, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        draw.text((cx - tw / 2 - tb[0], cy - th / 2 - tb[1]), txt, fill=(255, 255, 255), font=font)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path, "WEBP", quality=90, method=6)
    return W, H


# --------------------------------------------------------------------------- #
# Ikenie GOLD -> numbered PBP-VIS-NUM rows
# --------------------------------------------------------------------------- #
def _reading_order_idxs(
    idxs: List[int], bubbles_by_idx: Dict[int, dict]
) -> List[int]:
    """Order the gold idxs in production reading order.

    Primary: sort by bubbles.json ``idx`` (the frozen manga_reading_order rank).
    Fallback (no bubbles.json): manga_reading_order on the gold bboxes.
    """
    have_geo = all(i in bubbles_by_idx for i in idxs)
    if have_geo:
        return sorted(idxs)  # idx == production reading-order rank
    if manga_reading_order is None:
        return sorted(idxs)
    proxies = []
    for i in idxs:
        xy = _bbox_to_xyxy((bubbles_by_idx.get(i) or {}).get("bbox"))
        if xy is None:
            xy = (0.0, 0.0, 1.0, 1.0)
        proxies.append({"idx": i, "xmin": xy[0], "ymin": xy[1], "xmax": xy[2], "ymax": xy[3]})
    ordered = manga_reading_order(proxies)
    return [p["idx"] for p in ordered]


def build_ikenie_rows(
    chapter: str,
    gold_path: str,
    bench_root: str,
    image_out_dir: str,
    limit: int = 0,
) -> Tuple[List[dict], dict]:
    """One PBP-VIS-NUM record per page. Returns (records, stats).

    Each record carries the fields BOTH dataset variants need; the writer turns
    it into the image-on / image-off chat rows.
    """
    gold = load_jsonl(gold_path)

    # page -> {idx: gold_row}, deduping repeated idx (bubble flagged >1x).
    by_page: Dict[int, Dict[int, dict]] = defaultdict(dict)
    for r in gold:
        parsed = parse_src(r.get("src", ""))
        if parsed is None or parsed[0] != chapter:
            continue
        _, page, idx = parsed
        by_page[page].setdefault(idx, r)

    records: List[dict] = []
    st = defaultdict(int)
    pages = sorted(by_page)
    if limit:
        pages = pages[:limit]

    for page in pages:
        gold_idx_rows = by_page[page]
        bubbles = load_bubbles(bench_root, page)
        bubbles_by_idx = {b.get("idx"): b for b in bubbles} if bubbles else {}

        order = _reading_order_idxs(list(gold_idx_rows), bubbles_by_idx)

        jp_lines: List[str] = []
        en_lines: List[str] = []
        numbered_boxes: List[Tuple[int, Tuple[float, float, float, float]]] = []
        label = 0
        for i in order:
            grow = gold_idx_rows[i]
            en_raw = (grow.get("en") or "").strip()
            if not en_raw:
                st["gold_bubbles_skipped_empty_en"] += 1
                continue
            if _cjk_fraction(en_raw) > _CJK_MAX_FRAC:
                # untranslated JP / SFX / leaked judge-note — bad POV supervision.
                st["gold_bubbles_skipped_cjk_en"] += 1
                continue
            # JP: prefer serve-faithful OCR from bubbles.json; fall back to gold jp.
            blk = bubbles_by_idx.get(i) or {}
            jp = (blk.get("ocr_jp") or grow.get("jp") or "").strip()
            if not jp:
                st["gold_bubbles_skipped_empty_jp"] += 1
                continue
            xy = _bbox_to_xyxy(blk.get("bbox")) or _bbox_to_xyxy(grow.get("bbox"))
            label += 1
            jp_lines.append(jp)
            en_lines.append(to_sentence_case(en_raw))
            if xy is not None:
                numbered_boxes.append((label, xy))

        if not en_lines:
            st["pages_skipped_no_valid_bubbles"] += 1
            continue

        src_img = find_image(bench_root, page)
        numbered_img_path = ""
        img_w = img_h = 0
        if src_img and numbered_boxes:
            # redact EVERY OCR box on the page (glyph-clean), number the gold ones.
            redact = [xy for b in (bubbles or []) if (xy := _bbox_to_xyxy(b.get("bbox")))]
            # ensure gold boxes are covered even if bubbles.json lacked them.
            redact.extend(xy for _, xy in numbered_boxes)
            numbered_img_path = os.path.join(image_out_dir, chapter, f"{page:03d}_numbered.webp")
            img_w, img_h = render_numbered_image(src_img, redact, numbered_boxes, numbered_img_path)
            st["images_rendered"] += 1
        else:
            st["pages_without_renderable_image"] += 1

        records.append({
            "source": "ikenie_gold",
            "register_tag": "manga_nsfw",
            "chapter": chapter,
            "page": page,
            "n_bubbles": len(en_lines),
            "image_path": numbered_img_path,
            "prompt": build_numbered_prompt(jp_lines),
            "en": build_numbered_target(en_lines),
            "image_wh": [img_w, img_h],
        })

    st["pages"] = len(records)
    st["gold_bubbles"] = sum(r["n_bubbles"] for r in records)
    return records, dict(st)


# --------------------------------------------------------------------------- #
# text slices (image-absent) from the v11fix8 parquet
# --------------------------------------------------------------------------- #
def load_text_slices(
    parquet_path: str,
    n_nsfw: int,
    n_text: int,
    seed: int,
) -> Tuple[List[dict], List[dict], dict]:
    """Pull the NSFW-corpus register slice + the SFW v11 text backbone.

    NSFW slice : src starts with ``corpus_bitext:`` (machine-EN NSFW pairs).
    SFW slice  : register_tag in SFW_BACKBONE_TAGS (dialog/prose fluency backbone).
    Returns (nsfw_rows, sfw_rows, stats) as plain dicts {prompt,en,src,register_tag}.
    """
    import polars as pl

    lf = pl.scan_parquet(parquet_path)

    nsfw = (
        lf.filter(pl.col("src").str.starts_with("corpus_bitext:"))
        .select(["prompt", "en", "src", "register_tag"])
        .collect()
    )
    sfw = (
        lf.filter(pl.col("register_tag").is_in(list(SFW_BACKBONE_TAGS)))
        .select(["prompt", "en", "src", "register_tag"])
        .collect()
    )

    n_nsfw = min(n_nsfw, nsfw.height)
    n_text = min(n_text, sfw.height)
    nsfw_s = nsfw.sample(n=n_nsfw, seed=seed) if n_nsfw else nsfw.head(0)
    sfw_s = sfw.sample(n=n_text, seed=seed) if n_text else sfw.head(0)

    def _mk(df, source):
        out = []
        for r in df.iter_rows(named=True):
            out.append({
                "source": source,
                "register_tag": r["register_tag"],
                "image_path": "",
                "prompt": r["prompt"],
                "en": r["en"],
                "src": r["src"],
            })
        return out

    stats = {
        "nsfw_corpus_available": nsfw.height,
        "sfw_backbone_available": sfw.height,
        "nsfw_corpus_taken": n_nsfw,
        "sfw_backbone_taken": n_text,
    }
    return _mk(nsfw_s, "nsfw_corpus"), _mk(sfw_s, "v11_text"), stats


# --------------------------------------------------------------------------- #
# chat-row assembly (Qwen3-VL messages) + writers
# --------------------------------------------------------------------------- #
def _is_nsfw(rec: dict) -> bool:
    return rec.get("register_tag") in NSFW_TAGS


def to_chat_row(rec: dict, with_image: bool) -> dict:
    """Turn a record into a Qwen3-VL chat row.

    Image rows (with_image + a numbered image) carry an image content block whose
    ``image`` is the ABSOLUTE path to the numbered webp (the trainer hydrates it
    to a PIL at load time). Text rows carry a text-only user turn.
    """
    user_content: List[dict] = []
    has_image = bool(with_image and rec.get("image_path"))
    if has_image:
        user_content.append({"type": "image", "image": rec["image_path"]})
    user_content.append({"type": "text", "text": rec["prompt"]})
    return {
        "source": rec["source"],
        "register_tag": rec.get("register_tag", ""),
        "has_image": has_image,
        "image_path": rec["image_path"] if has_image else "",
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": rec["en"]}]},
        ],
        "meta": {k: rec[k] for k in ("chapter", "page", "n_bubbles", "src") if k in rec},
    }


def enforce_nsfw_cap(
    ikenie: List[dict], nsfw: List[dict], text: List[dict], cap: float
) -> Tuple[List[dict], int]:
    """Trim the NSFW-corpus slice so total NSFW share ≤ cap. Returns (nsfw, trimmed)."""
    fixed_nsfw = sum(_is_nsfw(r) for r in ikenie) + sum(_is_nsfw(r) for r in text)
    other = len(ikenie) + len(text)  # non-nsfw-corpus rows
    keep = len(nsfw)
    # (fixed_nsfw + keep) / (other + keep) <= cap
    while keep > 0 and (fixed_nsfw + keep) / (other + keep) > cap:
        keep -= 1
    return nsfw[:keep], len(nsfw) - keep


def write_variants(
    ikenie: List[dict],
    nsfw: List[dict],
    text: List[dict],
    out_dir: str,
    seed: int,
) -> dict:
    """Emit the aligned image-on / image-off jsonl pair + return parity stats.

    Row order is a single seeded permutation applied IDENTICALLY to both files,
    so the image-off file differs ONLY by the Ikenie image block.
    """
    # image-absent slices are byte-identical in both variants.
    text_rows_on = [to_chat_row(r, with_image=False) for r in (nsfw + text)]
    # Ikenie rows differ only by the image block.
    ik_on = [to_chat_row(r, with_image=True) for r in ikenie]
    ik_off = [to_chat_row(r, with_image=False) for r in ikenie]

    combined_on = ik_on + text_rows_on
    combined_off = ik_off + text_rows_on
    perm = list(range(len(combined_on)))
    random.Random(seed).shuffle(perm)
    combined_on = [combined_on[i] for i in perm]
    combined_off = [combined_off[i] for i in perm]

    on_path = os.path.join(out_dir, "data_poc_imageon.jsonl")
    off_path = os.path.join(out_dir, "data_poc_imageoff.jsonl")
    for path, rows in ((on_path, combined_on), (off_path, combined_off)):
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_img_on = sum(r["has_image"] for r in combined_on)
    n_img_off = sum(r["has_image"] for r in combined_off)
    nsfw_total = sum(_is_nsfw(r) for r in ikenie) + sum(_is_nsfw(r) for r in nsfw) + \
        sum(_is_nsfw(r) for r in text)
    total = len(combined_on)
    return {
        "imageon_path": on_path,
        "imageoff_path": off_path,
        "total_rows": total,
        "rows_by_source": {
            "ikenie_gold": len(ikenie),
            "nsfw_corpus": len(nsfw),
            "v11_text": len(text),
        },
        "image_blocks_imageon": n_img_on,
        "image_blocks_imageoff": n_img_off,
        "parity_ok": (len(combined_on) == len(combined_off) and n_img_off == 0 and
                      n_img_on == sum(1 for r in ikenie if r.get("image_path"))),
        "nsfw_rows": nsfw_total,
        "nsfw_pct": round(100.0 * nsfw_total / total, 2) if total else 0.0,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chapters", nargs="*", default=list(DEFAULT_CHAPTERS),
                    choices=list(DEFAULT_CHAPTERS))
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)),
                    help="Output dir for the jsonl + stats (LOCAL ONLY).")
    ap.add_argument("--image-out-dir", default=None,
                    help="Where numbered webps go (default: <out>/numbered_images).")
    ap.add_argument("--v11-parquet", default=DEFAULT_V11_PARQUET)
    ap.add_argument("--n-nsfw", type=int, default=2000, help="NSFW-corpus register rows (target).")
    ap.add_argument("--n-text", type=int, default=10000, help="SFW v11 text-backbone rows.")
    ap.add_argument("--nsfw-cap", type=float, default=0.18, help="Max NSFW share of rows.")
    ap.add_argument("--limit", type=int, default=0, help="Cap Ikenie pages (smoke).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--inspect", action="store_true",
                    help="Build Ikenie + a SMALL text pull, print a preview, EXIT.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    out = os.path.abspath(args.out)
    if "/mnt/nas" in out:
        raise SystemExit("Refusing to write under /mnt/nas (CIFS reaps files).")
    os.makedirs(out, exist_ok=True)
    image_out_dir = os.path.abspath(args.image_out_dir) if args.image_out_dir \
        else os.path.join(out, "numbered_images")

    # ---- Ikenie GOLD -> numbered PBP-VIS-NUM records --------------------------
    ikenie: List[dict] = []
    per_chapter: List[dict] = []
    for chapter in args.chapters:
        gold_path, bench_root = DEFAULT_CHAPTERS[chapter]
        if not os.path.exists(gold_path):
            raise SystemExit(f"gold jsonl not found: {gold_path}")
        recs, st = build_ikenie_rows(chapter, gold_path, bench_root, image_out_dir, args.limit)
        ikenie.extend(recs)
        st["chapter"] = chapter
        per_chapter.append(st)
        print(f"[{chapter}] pages={st['pages']} gold_bubbles={st['gold_bubbles']} "
              f"images_rendered={st.get('images_rendered', 0)}")

    # ---- text slices (image-absent) ------------------------------------------
    n_nsfw = 40 if args.inspect else args.n_nsfw
    n_text = 40 if args.inspect else args.n_text
    if not os.path.exists(args.v11_parquet):
        raise SystemExit(f"v11 parquet not found: {args.v11_parquet}")
    nsfw_rows, text_rows, text_stats = load_text_slices(
        args.v11_parquet, n_nsfw, n_text, args.seed
    )

    # ---- enforce the NSFW cap by trimming the corpus slice --------------------
    nsfw_rows, trimmed = enforce_nsfw_cap(ikenie, nsfw_rows, text_rows, args.nsfw_cap)
    if trimmed:
        print(f"[nsfw-cap] trimmed {trimmed} corpus rows to satisfy cap={args.nsfw_cap}")

    # ---- inspect: preview, no write ------------------------------------------
    if args.inspect:
        preview = None
        for r in ikenie:
            if r.get("image_path"):
                row = to_chat_row(r, with_image=True)
                preview = {
                    "source": row["source"],
                    "has_image": row["has_image"],
                    "image_path": row["image_path"],
                    "user_content_types": [c["type"] for c in row["messages"][0]["content"]],
                    "prompt_head": r["prompt"][:220],
                    "assistant_head": r["en"][:220],
                    "n_bubbles": r["n_bubbles"],
                }
                break
        print("=== --inspect preview (NO WRITE) ===")
        print(json.dumps({
            "ikenie_pages": len(ikenie),
            "nsfw_corpus_rows": len(nsfw_rows),
            "v11_text_rows": len(text_rows),
            "text_slice_stats": text_stats,
            "sample_ikenie_chat": preview,
        }, ensure_ascii=False, indent=2))
        return 0

    # ---- write both variants --------------------------------------------------
    parity = write_variants(ikenie, nsfw_rows, text_rows, out, args.seed)

    stats = {
        "format": "PBP-VIS-NUM (numbered redacted image + numbered JP/EN lists)",
        "base_model_target": "huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated",
        "reading_order": "bubbles.json idx (frozen manga_reading_order rank); "
                         "manga_reading_order fallback when bubbles.json absent",
        "en_target_casing": "to_sentence_case (POV-preserving) on gold EN",
        "held_out": "Furube (eval set — 37 of the 44 POV cases)",
        "image_out_dir": image_out_dir,
        "text_slice_stats": text_stats,
        "nsfw_corpus_trimmed_for_cap": trimmed,
        "nsfw_cap": args.nsfw_cap,
        "per_chapter": per_chapter,
        **parity,
    }
    with open(os.path.join(out, "stats_numbered.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)

    print(f"\nwrote {parity['total_rows']} rows")
    print(f"  image-on : {parity['imageon_path']}  (image blocks={parity['image_blocks_imageon']})")
    print(f"  image-off: {parity['imageoff_path']} (image blocks={parity['image_blocks_imageoff']})")
    print(f"  by source: {parity['rows_by_source']}")
    print(f"  NSFW: {parity['nsfw_rows']} rows ({parity['nsfw_pct']}%)  parity_ok={parity['parity_ok']}")
    print(f"  numbered images -> {image_out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
