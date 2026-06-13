"""Parse the UserUnknownFactor SFX gist into a unified-schema parquet.

Source: /tmp/uuf_sfx.md (5934 lines, mixed SFW + NSFW manga onomatopoeia)
Output: backend/training/datasets/filtered/uuf_sfx.parquet

Adapted from oracle agent's parser. Adds:
- pykakasi katakana synthesis when only romaji is in head
- Per-gloss row expansion (numbered senses → individual training pairs)
- NSFW flag via keyword regex + alias propagation
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "data"))
from unify_schema import make_row, write_parquet  # noqa: E402

NSFW = re.compile(
    r"\b(sex|sexual|orgasm|penis|vagin|anal|cum|ejaculat|fuck|titty|moan|seductive|"
    r"arous|hentai|nipple|breast|naughty|erotic|masturb|climax|grope|horny|"
    r"pussy|cock|suck|lick|wet|squelch|squirt|thrust|nipple)",
    re.I,
)
HEAD = re.compile(r"^\*([^*]+)\*\s*:\s*(.*)$")
KANA_PAREN = re.compile(r"\(\s*([぀-ヿー\s,，、]+)\s*\)")
NUM_PREFIX = re.compile(r"^\s*\d+\s*[\.\)]\s*")
NOTE_PREFIX = re.compile(r"^\s*\*?Note\*?\s*:\s*", re.I)
GLOSS_QUOTE = re.compile(r'"([^"]+)"')


def parse(path: Path) -> list[dict]:
    entries: list[dict] = []
    cur: dict | None = None
    for raw in open(path, encoding="utf-8"):
        line = raw.rstrip()
        if not line.strip() or line.startswith("#"):
            if cur:
                entries.append(cur)
                cur = None
            continue
        m = HEAD.match(line)
        if m:
            if cur:
                entries.append(cur)
            head, rest = m.group(1).strip(), m.group(2).strip()
            kana = ""
            km = KANA_PAREN.search(head)
            if km:
                kana = km.group(1).strip()
                head = KANA_PAREN.sub("", head).strip()
            aliases = [a.strip() for a in head.split(",") if a.strip()]
            cur = {
                "romaji": aliases[0] if aliases else "",
                "aliases": aliases[1:],
                "kana": kana,
                "glosses": [],
                "notes": "",
                "nsfw": False,
                "raw": [rest],
            }
            if rest and not NUM_PREFIX.match(rest):
                cur["glosses"].append(rest)
        elif cur:
            s = line.strip()
            cur["raw"].append(s)
            if NOTE_PREFIX.match(s):
                cur["notes"] += " " + NOTE_PREFIX.sub("", s)
            elif NUM_PREFIX.match(s):
                cur["glosses"].append(NUM_PREFIX.sub("", s))
            elif cur["glosses"]:
                cur["glosses"][-1] += " " + s
            else:
                cur["glosses"].append(s)
    if cur:
        entries.append(cur)
    for e in entries:
        body = " ".join(e["raw"]) + " " + e.get("notes", "")
        e["nsfw"] = bool(NSFW.search(body))
        # Clean glosses
        clean_glosses = []
        for g in e["glosses"]:
            g = g.strip(" ;.")
            if g:
                clean_glosses.append(g)
        e["glosses"] = clean_glosses
    return entries


def synth_kana(romaji: str) -> str:
    """Try pykakasi to synthesize katakana from romaji. Fallback: empty."""
    try:
        import pykakasi  # type: ignore[import-not-found]
        kks = pykakasi.kakasi()
        result = kks.convert(romaji)
        return "".join(r.get("kana", "") for r in result)
    except Exception:
        return ""


def extract_short_glosses(gloss_str: str) -> list[str]:
    """A gloss line might be 'exclamation of surprise: "Oh!", "Ack!", "Ah!"'.
    Extract the quoted short glosses if present, else use the whole line.
    """
    quotes = GLOSS_QUOTE.findall(gloss_str)
    if quotes:
        return [q for q in quotes if 1 <= len(q) <= 30]
    # Strip leading "1. ", "category: ", etc.
    cleaned = re.sub(r"^[A-Za-z][^:]{0,40}:\s*", "", gloss_str).strip()
    if 1 <= len(cleaned) <= 60:
        return [cleaned]
    return []


def main() -> int:
    src = Path("/tmp/uuf_sfx.md")
    out = REPO / "training" / "datasets" / "filtered" / "uuf_sfx.parquet"

    if not src.exists():
        print(f"missing source {src}; fetch the UUF gist first.")
        return 1

    entries = parse(src)
    print(f"parsed {len(entries)} headwords")

    rows = []
    n_nsfw = 0
    n_kana = 0
    for e in entries:
        # JP form: prefer kana, then synthesize from romaji
        kana = e["kana"]
        if not kana:
            kana = synth_kana(e["romaji"])
        if kana:
            n_kana += 1
        # If still no kana, use romaji written in latin
        jp_for_train = kana if kana else e["romaji"]
        if not jp_for_train:
            continue

        if e["nsfw"]:
            n_nsfw += 1

        for gloss in e["glosses"]:
            shorts = extract_short_glosses(gloss)
            for en in shorts:
                rows.append(make_row(
                    jp=jp_for_train,
                    en=en,
                    src=f"uuf_sfx:{e['romaji']}{':nsfw' if e['nsfw'] else ''}",
                    register_tag="sfx",
                    gold_flag=True,
                ))

    out.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(iter(rows), out)
    print(f"headwords: {len(entries)} (nsfw {n_nsfw}, kana {n_kana})")
    print(f"training pairs: {len(rows)}")
    print(f"wrote -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
