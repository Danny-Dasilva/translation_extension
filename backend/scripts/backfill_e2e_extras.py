"""Backfill translations.txt + originals/ into an existing pipeline-e2e gallery.

Takes an existing --out directory that was produced by visualize_e2e_pipeline.py
(before the txt/originals features landed) and populates:

  <out_root>/originals/<original-filename>     (copies of source images)
  <out_root>/<slug>/translations.txt           (JP OCR + EN translations per page)
  <out_root>/translations.txt                  (aggregate across all pages)

Input is inferred from each slug's stats.json (image filename) plus a user-supplied
--source directory where the originals live. Existing files are not overwritten
unless --force is passed.

Usage:
    uv run python scripts/backfill_e2e_extras.py \
        --out ~/manga-output/644289 \
        --source "/mnt/.../644289__Ryou_ .../"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def _parse_prompt_sources(prompt_text: str) -> list[str]:
    """Recover the JP source strings from an 08_translate_prompt.txt body.

    The prompt written by the pipeline has the form:
        <BATCHED_SYSTEM_PROMPT>

        [1] 日本語...
        [2] ...
    We return the list of per-bubble strings in order, or [] if unparseable.
    """
    out: list[str] = []
    current: list[str] | None = None
    for raw in prompt_text.splitlines():
        line = raw.rstrip()
        stripped = line.lstrip()
        if stripped.startswith("[") and "]" in stripped:
            idx_close = stripped.index("]")
            inner = stripped[1:idx_close]
            if inner.isdigit():
                if current is not None:
                    out.append("\n".join(current).strip())
                current = [stripped[idx_close + 1:].lstrip()]
                continue
        if current is not None:
            current.append(line)
    if current is not None:
        out.append("\n".join(current).strip())
    return out


def _parse_translate_response(resp_text: str) -> list[str]:
    """Mirror of _parse_prompt_sources for the response file (same [N] format)."""
    return _parse_prompt_sources(resp_text)


def _write_translations_txt(path: Path, image_name: str,
                            jp_texts: list[str], translations: list[str]) -> None:
    lines = [f"# {image_name}",
             f"# {len(jp_texts)} bubble(s)",
             ""]
    for i, jp in enumerate(jp_texts):
        en = translations[i] if i < len(translations) else ""
        lines.append(f"[{i + 1}]")
        lines.append(f"  JP: {jp}")
        lines.append(f"  EN: {en}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _slug_to_source(slug_dir: Path, source_dir: Path) -> Path | None:
    """Locate the original source image for a given slug dir via its stats.json."""
    stats_p = slug_dir / "stats.json"
    if not stats_p.exists():
        return None
    try:
        data = json.loads(stats_p.read_text(encoding="utf-8"))
    except Exception:
        return None
    name = data.get("image")
    if not name:
        return None
    candidate = source_dir / name
    if candidate.exists():
        return candidate
    # Fallback: search by stem (extension may have changed)
    stem = Path(name).stem
    for p in source_dir.iterdir():
        if p.stem == stem and p.is_file():
            return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True,
                    help="Gallery root produced by visualize_e2e_pipeline.py")
    ap.add_argument("--source", type=Path, required=True,
                    help="Directory containing the original source images")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing translations.txt / originals/")
    args = ap.parse_args()

    out_root: Path = args.out
    source_dir: Path = args.source
    if not out_root.is_dir():
        print(f"error: --out {out_root} is not a directory", file=sys.stderr)
        return 2
    if not source_dir.is_dir():
        print(f"error: --source {source_dir} is not a directory", file=sys.stderr)
        return 2

    originals_dir = out_root / "originals"
    originals_dir.mkdir(parents=True, exist_ok=True)

    slug_dirs = sorted(p for p in out_root.iterdir()
                       if p.is_dir() and (p / "stats.json").exists())
    print(f"found {len(slug_dirs)} slug directories under {out_root}")

    aggregate: list[tuple[str, str, list[str], list[str]]] = []
    copied = 0
    wrote = 0
    for slug_dir in slug_dirs:
        stats = json.loads((slug_dir / "stats.json").read_text(encoding="utf-8"))
        image_name = stats.get("image", slug_dir.name)

        # 1) Copy original into originals/
        src = _slug_to_source(slug_dir, source_dir)
        if src is not None:
            dst = originals_dir / src.name
            if args.force or not dst.exists():
                try:
                    dst.write_bytes(src.read_bytes())
                    copied += 1
                except Exception as exc:
                    print(f"  [{slug_dir.name}] copy failed: {exc}")
        else:
            print(f"  [{slug_dir.name}] no matching source for image={image_name!r}")

        # 2) Build JP/EN lists. Prefer the expanded fields; fall back to the
        #    08/09 txt files if an older run only saved 8-item samples.
        jp_all = stats.get("ocr_all")
        en_all = stats.get("translations_all")
        if not jp_all:
            prompt_p = slug_dir / "08_translate_prompt.txt"
            if prompt_p.exists():
                jp_all = _parse_prompt_sources(prompt_p.read_text(encoding="utf-8"))
        if not en_all:
            resp_p = slug_dir / "09_translate_response.txt"
            if resp_p.exists():
                en_all = _parse_translate_response(resp_p.read_text(encoding="utf-8"))
        jp_all = jp_all or stats.get("ocr_samples", [])
        en_all = en_all or stats.get("translations", [])

        # 3) Write per-slug translations.txt
        txt_p = slug_dir / "translations.txt"
        if args.force or not txt_p.exists():
            _write_translations_txt(txt_p, image_name, jp_all, en_all)
            wrote += 1

        aggregate.append((slug_dir.name, image_name, jp_all, en_all))

    # 4) Aggregate translations.txt at the root
    agg_lines = ["# Aggregate OCR + translations",
                 f"# {len(aggregate)} page(s)",
                 ""]
    for slug, image, jps, ens in aggregate:
        agg_lines.append(f"## {slug}  ({image})")
        for i, jp in enumerate(jps):
            en = ens[i] if i < len(ens) else ""
            agg_lines.append(f"  [{i + 1}] JP: {jp}")
            agg_lines.append(f"      EN: {en}")
        agg_lines.append("")
    (out_root / "translations.txt").write_text("\n".join(agg_lines), encoding="utf-8")

    print(f"copied {copied} originals into {originals_dir}")
    print(f"wrote {wrote} per-slug translations.txt")
    print(f"aggregate → {out_root / 'translations.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
