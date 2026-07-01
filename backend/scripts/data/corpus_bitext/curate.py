"""Curation: turn aligned (jp, en) bubble pairs into TRAINING-WORTHY rows.

Quality > volume. Each pair is filtered and scored; only pairs above a tunable
quality threshold survive. Filters (precision-favoring):

  * alignment distance     -- drop pairs whose normalized centroid distance is
                              above ``max_match_dist`` (looser matches are noise).
  * OCR confidence + garble -- reuse the production garble gate
                              (``app.utils.ocr_confidence_gate``): drop low-conf
                              or linguistically-implausible JP OCR.
  * SFX / non-dialogue     -- drop bubbles the SFX glossary recognizes
                              (onomatopoeia, not bitext) and JP-glyph-empty lines.
  * length-ratio sanity    -- EN words vs JP chars must be in a plausible band
                              (catches merged/split/mis-paired bubbles).
  * empty / duplicate      -- drop empty sides and exact (jp,en) duplicates.
  * page coverage          -- drop ALL pairs from a page whose alignment coverage
                              is below ``min_coverage`` (redraw mismatch / wrong
                              pairing risk).

Each surviving pair carries a 0..1 ``quality`` score so the threshold is tunable
downstream without re-running OCR/alignment.
"""
from __future__ import annotations

import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[3]

# Production garble gate + SFX glossary (cwd must be backend/ for `app.` imports;
# run_gallery / validate add backend to sys.path).
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app.utils.ocr_confidence_gate import is_garbled_low_conf  # noqa: E402

try:
    from app.services.sfx_glossary import sfx_pre_translate  # noqa: E402
except Exception:  # pragma: no cover
    def sfx_pre_translate(_):  # type: ignore
        return None


def _is_japanese_glyph(ch: str) -> bool:
    o = ord(ch)
    return (
        0x3040 <= o <= 0x309F  # hiragana
        or 0x30A0 <= o <= 0x30FF  # katakana
        or 0xFF65 <= o <= 0xFF9F  # halfwidth katakana
        or 0x4E00 <= o <= 0x9FFF  # CJK
        or 0x3400 <= o <= 0x4DBF
    )


def jp_glyph_count(s: str) -> int:
    return sum(1 for c in unicodedata.normalize("NFC", s or "") if _is_japanese_glyph(c))


def en_word_count(s: str) -> int:
    return len([w for w in (s or "").split() if any(c.isalnum() for c in w)])


@dataclass
class CurationConfig:
    max_match_dist: float = 0.06   # normalized centroid distance ceiling
    min_ocr_conf: float = 0.65     # garble-gate confidence floor
    min_jp_glyphs: int = 2         # below this a JP "line" is a fragment
    len_ratio_lo: float = 0.10     # en_words / jp_glyphs lower bound
    len_ratio_hi: float = 2.50     # en_words / jp_glyphs upper bound
    min_coverage: float = 0.35     # page-level alignment coverage floor
    keep_threshold: float = 0.50   # quality score to keep a row
    # quality-score component weights (sum ~= 1.0)
    w_pos: float = 0.40
    w_conf: float = 0.25
    w_len: float = 0.15
    w_cov: float = 0.20


@dataclass
class CuratedPair:
    jp_src: str
    en_tgt: str
    jp_bbox: dict
    en_bbox: dict
    match_dist: float
    jp_ocr_conf: float
    page: int
    quality: float
    drop_reason: str | None = None  # set when the pair was rejected (for audit)


@dataclass
class CurationStats:
    seen: int = 0
    kept: int = 0
    drop_dist: int = 0
    drop_garble: int = 0
    drop_sfx: int = 0
    drop_lenratio: int = 0
    drop_empty: int = 0
    drop_en_lang: int = 0
    drop_dup: int = 0
    drop_coverage: int = 0
    drop_score: int = 0
    dropped_pages: int = 0
    quality_hist: list = field(default_factory=list)

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "quality_hist"}
        d["quality_hist_buckets"] = _hist(self.quality_hist)
        return d


def _hist(scores: list[float], bins: int = 10) -> dict:
    out = {f"{i/bins:.1f}-{(i+1)/bins:.1f}": 0 for i in range(bins)}
    for s in scores:
        b = min(bins - 1, max(0, int(s * bins)))
        out[f"{b/bins:.1f}-{(b+1)/bins:.1f}"] += 1
    return out


def _len_score(jp: str, en: str, cfg: CurationConfig) -> float:
    jg = jp_glyph_count(jp)
    ew = en_word_count(en)
    if jg == 0:
        return 0.0
    ratio = ew / jg
    # full credit inside a tight central band; linear falloff to the hard bounds.
    if 0.25 <= ratio <= 1.5:
        return 1.0
    if ratio < 0.25:
        return max(0.0, (ratio - cfg.len_ratio_lo) / (0.25 - cfg.len_ratio_lo))
    return max(0.0, (cfg.len_ratio_hi - ratio) / (cfg.len_ratio_hi - 1.5))


def quality_score(match_dist: float, ocr_conf: float, jp: str, en: str,
                  page_coverage: float, cfg: CurationConfig) -> float:
    s_pos = 1.0 - min(1.0, match_dist / max(1e-6, cfg.max_match_dist))
    s_conf = min(1.0, max(0.0, (ocr_conf - cfg.min_ocr_conf) / max(1e-6, 0.95 - cfg.min_ocr_conf)))
    s_len = _len_score(jp, en, cfg)
    s_cov = min(1.0, max(0.0, page_coverage))
    return cfg.w_pos * s_pos + cfg.w_conf * s_conf + cfg.w_len * s_len + cfg.w_cov * s_cov


def curate_pair(jp_text: str, en_text: str, jp_bbox: dict, en_bbox: dict,
                match_dist: float, ocr_conf: float, page: int,
                page_coverage: float, cfg: CurationConfig,
                seen_keys: set, stats: CurationStats) -> CuratedPair:
    """Evaluate ONE aligned pair. Returns a CuratedPair; ``drop_reason`` is set
    (and ``quality`` may be 0) when rejected. The caller keeps only pairs with
    ``drop_reason is None``."""
    stats.seen += 1
    jp = (jp_text or "").strip()
    en = (en_text or "").strip()

    def rej(reason: str, counter: str) -> CuratedPair:
        setattr(stats, counter, getattr(stats, counter) + 1)
        return CuratedPair(jp, en, jp_bbox, en_bbox, match_dist, ocr_conf, page, 0.0, reason)

    if not jp or not en:
        return rej("empty", "drop_empty")
    if jp_glyph_count(jp) < cfg.min_jp_glyphs:
        return rej("jp_too_short", "drop_empty")
    # The EN target must actually BE English: at least one Latin letter and NO
    # residual Japanese glyphs (an untranslated JP/SFX bubble whose "translation"
    # is the JP itself, e.g. "ズキッ", must never become a training target).
    if not any(c.isascii() and c.isalpha() for c in en) or jp_glyph_count(en) > 0:
        return rej("en_not_english", "drop_en_lang")
    if match_dist > cfg.max_match_dist:
        return rej("match_dist", "drop_dist")
    # SFX / onomatopoeia are not dialogue bitext.
    if sfx_pre_translate(jp) is not None:
        return rej("sfx", "drop_sfx")
    # Garble gate: low-confidence or linguistically-implausible OCR.
    if is_garbled_low_conf(jp, ocr_conf, conf_threshold=cfg.min_ocr_conf):
        return rej("garble", "drop_garble")
    # Length-ratio sanity.
    jg = jp_glyph_count(jp)
    ratio = en_word_count(en) / jg if jg else 999.0
    if ratio < cfg.len_ratio_lo or ratio > cfg.len_ratio_hi:
        return rej("len_ratio", "drop_lenratio")
    # Exact duplicate (jp, en).
    key = (jp, en.lower())
    if key in seen_keys:
        return rej("dup", "drop_dup")

    q = quality_score(match_dist, ocr_conf, jp, en, page_coverage, cfg)
    if q < cfg.keep_threshold:
        stats.drop_score += 1
        return CuratedPair(jp, en, jp_bbox, en_bbox, match_dist, ocr_conf, page, q, "low_score")

    seen_keys.add(key)
    stats.kept += 1
    stats.quality_hist.append(q)
    return CuratedPair(jp, en, jp_bbox, en_bbox, match_dist, ocr_conf, page, q, None)
