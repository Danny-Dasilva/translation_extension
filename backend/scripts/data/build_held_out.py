"""Assemble held-out evaluation JSONL files + MANIFEST.

Emits under ``backend/training/eval_held_out/``:
  - vntl128.jsonl             — 128 JP lines from VNTL-v3.1-1k val split
                                (first 128 aligned pairs as parsed from the
                                packed ``text`` column, per VNTL leaderboard protocol).
  - flores_ja_en.jsonl        — FLORES-200 devtest jpn_Jpan↔eng_Latn (1,012 lines).
                                Downloaded via ``datasets.load_dataset(
                                "facebook/flores", "jpn_Jpan-eng_Latn", split="devtest")``
                                on first run.
  - open_mantra_test.jsonl    — 2 held-out volumes (boureisougi, rasetugari).
  - probes.jsonl              — stub w/ ~30 seed adversarial examples
                                (8 categories). TODO: expand to 300.
  - regression_canary.jsonl   — 500 rows from Helsinki-NLP/news_commentary
                                (ja-en). Logs a clear error if unavailable,
                                does NOT fabricate.
  - MANIFEST.json             — file list w/ row counts, source, license.

Explicitly SKIPS ``custom_manga_1500.jsonl`` — flagged in stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

from _cli_common import configure_logging, logger, parse_vntl_packed_text


DEFAULT_OUT_DIR = "backend/training/eval_held_out"

VNTL_V31_VAL = (
    "backend/training/datasets/translation/vn-ln-manga/"
    "lmg-anon__VNTL-v3.1-1k/data/val-00000-of-00001-51ab569c62b2bbc8.parquet"
)
OPEN_MANTRA_ANNOT = (
    "backend/training/datasets/translation/vn-ln-manga/"
    "open-mantra-dataset/annotation.json"
)
OPEN_MANTRA_HELD_OUT_TITLES = ("boureisougi", "rasetugari")


# --- probe seeds (30 examples across 8 categories) -----------------------------

PROBE_SEEDS: list[dict[str, str]] = [
    # names — must not transliterate erroneously
    {"category": "names", "jp": "桜井さん、おはよう。", "en_ref": "Good morning, Sakurai.", "note": "surname + ~さん"},
    {"category": "names", "jp": "ミカは学校に行った。", "en_ref": "Mika went to school.", "note": "katakana given name"},
    {"category": "names", "jp": "田中先輩、お待たせしました。", "en_ref": "Senpai Tanaka, sorry to keep you waiting.", "note": "surname + senpai"},
    {"category": "names", "jp": "俺、山田ケンタだ。", "en_ref": "I'm Kenta Yamada.", "note": "full name"},
    # honorifics — keep ~さん/~くん/~ちゃん or naturalize
    {"category": "honorifics", "jp": "田中さん、ありがとう。", "en_ref": "Thank you, Tanaka-san.", "note": "keep -san"},
    {"category": "honorifics", "jp": "先生、お元気ですか？", "en_ref": "How are you, sensei?", "note": "sensei"},
    {"category": "honorifics", "jp": "お嬢様、お茶をどうぞ。", "en_ref": "Here is your tea, my lady.", "note": "ojou-sama"},
    {"category": "honorifics", "jp": "ご主人様！", "en_ref": "Master!", "note": "goshujin-sama"},
    # curly quotes / manga punctuation — must preserve bubble punctuation
    {"category": "curly_quotes", "jp": "「おはよう」", "en_ref": "“Good morning.”", "note": "jp corner brackets -> curly quotes"},
    {"category": "curly_quotes", "jp": "『重要』", "en_ref": "“Important”", "note": "double corner brackets"},
    {"category": "curly_quotes", "jp": "それって……本当？", "en_ref": "Is that... really true?", "note": "leading ellipsis"},
    {"category": "curly_quotes", "jp": "やめて！！", "en_ref": "Stop it!!", "note": "full-width bang"},
    # repetition — 繰り返し, don't loop
    {"category": "repetition", "jp": "えええええええ！？", "en_ref": "What?!", "note": "elongated exclamation"},
    {"category": "repetition", "jp": "ずっとずっと待ってた。", "en_ref": "I've been waiting for so, so long.", "note": "intensifier"},
    {"category": "repetition", "jp": "ドキドキドキドキ…", "en_ref": "*thump-thump-thump...*", "note": "heartbeat sfx"},
    # refusal — must not trigger LLM refusal; these are benign-looking prompts
    {"category": "refusal", "jp": "殺してやる！", "en_ref": "I'll kill you!", "note": "villain line; must not refuse"},
    {"category": "refusal", "jp": "裸で走るなんてバカね。", "en_ref": "Running around naked is so stupid.", "note": "mild content"},
    {"category": "refusal", "jp": "武器を下ろせ。", "en_ref": "Lower your weapon.", "note": "instruction in fiction"},
    # length — very short + very long handling
    {"category": "length", "jp": "はい。", "en_ref": "Yes.", "note": "minimal"},
    {"category": "length", "jp": "えっ？", "en_ref": "Huh?", "note": "single particle"},
    {
        "category": "length",
        "jp": "今日は朝から晩までずっと本を読んでいて、気がついたら外が暗くなっていて、お腹もペコペコだ。",
        "en_ref": "I spent all day from morning till night reading, and before I knew it the sky was dark and I was starving.",
        "note": "long sentence",
    },
    # SFX — should map to English SFX convention
    {"category": "sfx", "jp": "ドキドキ", "en_ref": "*thump-thump*", "note": "heartbeat"},
    {"category": "sfx", "jp": "ゴクッ", "en_ref": "*gulp*", "note": "swallow"},
    {"category": "sfx", "jp": "パチパチ", "en_ref": "*clap clap*", "note": "applause"},
    {"category": "sfx", "jp": "ザーザー", "en_ref": "*whoosh* (rain)", "note": "rain"},
    # idioms — don't literalize
    {"category": "idiom", "jp": "猿も木から落ちる。", "en_ref": "Even monkeys fall from trees.", "note": "lit translation OK as proverb"},
    {"category": "idiom", "jp": "顔が広い。", "en_ref": "He knows a lot of people.", "note": "face is wide -> has many connections"},
    {"category": "idiom", "jp": "耳が痛い。", "en_ref": "That hits close to home.", "note": "ear hurts"},
    {"category": "idiom", "jp": "油を売る。", "en_ref": "Slacking off.", "note": "selling oil"},
    {"category": "idiom", "jp": "腹を割って話す。", "en_ref": "Let's have a heart-to-heart.", "note": "splitting belly to talk"},
]


# -------------------------------------------------------------------------------


def write_jsonl(path: Path, rows: list[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def build_vntl128(vntl_val_parquet: Path, out: Path, n: int = 128) -> int | None:
    if not vntl_val_parquet.exists():
        logger.error(f"VNTL val parquet missing: {vntl_val_parquet}")
        return None
    df = pl.read_parquet(vntl_val_parquet)
    if "text" not in df.columns:
        logger.error(f"unexpected schema for {vntl_val_parquet}: {df.columns}")
        return None
    rows: list[dict] = []
    for row_idx, rec in enumerate(df.iter_rows(named=True)):
        blob = rec.get("text") or ""
        for turn_idx, (jp, en) in enumerate(parse_vntl_packed_text(blob)):
            rows.append(
                {
                    "jp": jp,
                    "en": en,
                    "src": f"vntl_v31_1k_val:row{row_idx}:turn{turn_idx}",
                }
            )
            if len(rows) >= n:
                return write_jsonl(out, rows)
    logger.warning(
        f"VNTL val produced only {len(rows)} pairs (< {n} requested)"
    )
    return write_jsonl(out, rows)


def build_flores(out: Path, limit: int | None = None) -> int | None:
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError:
        logger.error(
            "FLORES build skipped: `datasets` not installed. "
            "Install with `uv add --project backend datasets`."
        )
        return None
    try:
        ds = load_dataset("facebook/flores", "jpn_Jpan-eng_Latn", split="devtest")
    except Exception as e:  # noqa: BLE001
        logger.error(f"FLORES build skipped: load_dataset failed: {e}")
        return None
    rows: list[dict] = []
    for i, r in enumerate(ds):
        rec: dict = r if isinstance(r, dict) else {}  # pyright-friendly
        jp = rec.get("sentence_jpn_Jpan") or rec.get("sentence_jpn") or ""
        en = rec.get("sentence_eng_Latn") or rec.get("sentence_eng") or ""
        if jp and en:
            rows.append({"jp": jp, "en": en, "src": f"flores_devtest:{i}"})
        if limit is not None and len(rows) >= limit:
            break
    return write_jsonl(out, rows)


def build_open_mantra_test(annotation: Path, out: Path) -> int | None:
    if not annotation.exists():
        logger.error(f"open-mantra annotation missing: {annotation}")
        return None
    with annotation.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    want = set(OPEN_MANTRA_HELD_OUT_TITLES)
    rows: list[dict] = []
    for book in data:
        title = book.get("book_title")
        if title not in want:
            continue
        for page in book.get("pages", []):
            page_idx = page.get("page_index")
            for t_idx, entry in enumerate(page.get("text", [])):
                jp = (entry.get("text_ja") or "").strip()
                en = (entry.get("text_en") or "").strip()
                if jp and en:
                    rows.append(
                        {
                            "jp": jp,
                            "en": en,
                            "src": f"open_mantra_test:{title}:p{page_idx}:t{t_idx}",
                        }
                    )
    return write_jsonl(out, rows)


def build_probes(out: Path) -> int:
    """Emit seed probe set with TODO to expand to 300."""
    rows: list[dict] = []
    for i, s in enumerate(PROBE_SEEDS):
        rows.append(
            {
                "jp": s["jp"],
                "en_ref": s["en_ref"],
                "category": s["category"],
                "note": s.get("note", ""),
                "src": f"probe_seed:{s['category']}:{i}",
            }
        )
    # sentinel TODO row
    rows.append(
        {
            "jp": "__TODO__",
            "en_ref": "",
            "category": "_meta",
            "note": (
                f"{len(PROBE_SEEDS)} seed examples across 8 categories. "
                "Expand to 300 adversarial probes: ~40 per category. See "
                "`PROBE_SEEDS` list in build_held_out.py for starting point."
            ),
            "src": "probe_seed:_meta:todo",
        }
    )
    return write_jsonl(out, rows)


def build_regression_canary(out: Path, n: int = 500) -> int | None:
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError:
        logger.error(
            "regression_canary skipped: `datasets` not installed. "
            "Install with `uv add --project backend datasets`."
        )
        return None
    # Try news_commentary ja-en first.
    for config_id in ("Helsinki-NLP/news_commentary", "news_commentary"):
        for cfg in ("ja-en", "en-ja"):
            try:
                ds = load_dataset(config_id, cfg, split="train", streaming=True)
                rows: list[dict] = []
                for i, r in enumerate(ds):
                    rec: dict = r if isinstance(r, dict) else {}
                    pair = rec.get("translation") or {}
                    if not isinstance(pair, dict):
                        pair = {}
                    jp = pair.get("ja") or ""
                    en = pair.get("en") or ""
                    if jp and en:
                        rows.append(
                            {"jp": jp, "en": en, "src": f"news_commentary:{i}"}
                        )
                    if len(rows) >= n:
                        break
                if rows:
                    return write_jsonl(out, rows)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"news_commentary {config_id} {cfg} failed: {e}")
    logger.error(
        "regression_canary: no source produced rows; flagged and NOT fabricated."
    )
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--vntl-val", default=VNTL_V31_VAL)
    parser.add_argument("--open-mantra-annot", default=OPEN_MANTRA_ANNOT)
    parser.add_argument("--skip-flores", action="store_true")
    parser.add_argument("--skip-canary", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []

    def record(name: str, n: int | None, source: str, license_: str) -> None:
        if n is None:
            logger.warning(f"{name}: NOT written (see errors above)")
            manifest.append(
                {
                    "file": name,
                    "row_count": 0,
                    "source": source,
                    "license": license_,
                    "status": "MISSING",
                }
            )
        else:
            manifest.append(
                {
                    "file": name,
                    "row_count": n,
                    "source": source,
                    "license": license_,
                    "status": "OK",
                }
            )

    # vntl128
    vntl_out = out_dir / "vntl128.jsonl"
    if args.dry_run:
        print(f"[dry-run] would build {vntl_out}")
    else:
        n = build_vntl128(Path(args.vntl_val), vntl_out, n=128)
        record("vntl128.jsonl", n, "lmg-anon/VNTL-v3.1-1k val split", "unspecified (research use)")

    # flores
    flores_out = out_dir / "flores_ja_en.jsonl"
    if args.skip_flores:
        logger.info("skipping FLORES per --skip-flores")
        record("flores_ja_en.jsonl", None, "facebook/flores jpn_Jpan-eng_Latn devtest", "CC-BY-SA 4.0")
    elif args.dry_run:
        print(f"[dry-run] would build {flores_out}")
    else:
        n = build_flores(flores_out)
        record("flores_ja_en.jsonl", n, "facebook/flores jpn_Jpan-eng_Latn devtest", "CC-BY-SA 4.0")

    # open mantra test
    om_out = out_dir / "open_mantra_test.jsonl"
    if args.dry_run:
        print(f"[dry-run] would build {om_out}")
    else:
        n = build_open_mantra_test(Path(args.open_mantra_annot), om_out)
        record("open_mantra_test.jsonl", n, "mantra-inc/open-mantra-dataset (2 held-out volumes)", "CC-BY-NC-SA 4.0")

    # probes
    probes_out = out_dir / "probes.jsonl"
    if args.dry_run:
        print(f"[dry-run] would build {probes_out}")
    else:
        n = build_probes(probes_out)
        record("probes.jsonl", n, "hand-built (stub; expand to 300)", "project-internal")

    # regression canary
    canary_out = out_dir / "regression_canary.jsonl"
    if args.skip_canary:
        logger.info("skipping regression_canary per --skip-canary")
        record("regression_canary.jsonl", None, "Helsinki-NLP/news_commentary ja-en", "CC-BY-SA 3.0")
    elif args.dry_run:
        print(f"[dry-run] would build {canary_out}")
    else:
        n = build_regression_canary(canary_out, n=500)
        record(
            "regression_canary.jsonl",
            n,
            "Helsinki-NLP/news_commentary ja-en",
            "CC-BY-SA 3.0",
        )

    # Manifest + explicit flag on custom_manga_1500
    manifest.append(
        {
            "file": "custom_manga_1500.jsonl",
            "row_count": 0,
            "source": "OCR pipeline output (not yet produced)",
            "license": "project-internal",
            "status": "ABSENT_BY_DESIGN",
        }
    )
    msg = (
        "NOTE: `custom_manga_1500.jsonl` is intentionally absent. The plan "
        "requires 1,500 manga bubbles from the OCR pipeline; those do not exist "
        "yet. Held-out target is reduced accordingly (see plan §6)."
    )
    logger.warning(msg)
    print(msg, file=sys.stderr)

    if args.dry_run:
        print(json.dumps({"dry_run": True, "manifest_preview": manifest}, indent=2))
        return

    manifest_path = out_dir / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps({"files": manifest}, indent=2, ensure_ascii=False)
    )
    print(f"wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
