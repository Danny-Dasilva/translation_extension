#!/usr/bin/env python3
"""Flag / drop hamming=0 art-only pollution from a page-level bitext manifest.

~11% of the 375k export (41,037 / 374,688 page rows = 10.95%) have a JP<->EN pHash
Hamming distance of exactly 0. The distance histogram is bimodal -- a spike at h=0,
then near-zero at h=1 (only 2 rows), then the real translated-page mass from h=2 up:

    h=0 : 41037     <- pollution: EN "translation" page is pixel-identical to JP page
    h=1 : 2
    h=2 : 90748
    h=4 : 98555
    ...

A distance of 0 means the two page images are effectively identical, i.e. the page
carries no translated text to learn from: covers, splash art, credits, or an
untranslated raw page copied verbatim into the EN gallery. These rows add no bitext
signal and dilute training, so they should be excluded from the next train.

This filter is manifest-schema-agnostic about extra fields: it only needs a numeric
``hamming`` per row. It works on manifest_pages_375k.jsonl and on
manifest_pages_backlog.jsonl alike.

LOCAL-ONLY, read-only on the input, stdlib-only. Does not modify the input manifest.

Outputs (default; override with flags):
  <stem>.clean.jsonl        rows with hamming > --min-hamming (default keep h>=1, i.e. drop h==0)
  <stem>.pollution.jsonl    the dropped rows (audit trail)               [--emit-dropped]
  <stem>.pollution.json     report: totals, per-bucket, whole-gallery-pollution pairs
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest", type=Path, help="page-level manifest jsonl (has a 'hamming' field)")
    ap.add_argument("--min-hamming", type=int, default=1,
                    help="keep rows with hamming >= this; default 1 drops only exact h==0 (default)")
    ap.add_argument("--out-clean", type=Path, default=None)
    ap.add_argument("--out-report", type=Path, default=None)
    ap.add_argument("--emit-dropped", action="store_true",
                    help="also write the dropped (polluted) rows to <stem>.pollution.jsonl")
    ap.add_argument("--flag-only", action="store_true",
                    help="do not split; write one manifest with an added 'art_only' bool per row")
    ap.add_argument("--limit", type=int, default=0, help="only read first N rows (sampling / smoke)")
    args = ap.parse_args()

    stem = args.manifest.with_suffix("")  # strips .jsonl
    out_clean = args.out_clean or Path(f"{stem}.clean.jsonl")
    out_dropped = Path(f"{stem}.pollution.jsonl")
    out_report = args.out_report or Path(f"{stem}.pollution.json")
    out_flagged = Path(f"{stem}.flagged.jsonl")

    total = kept = dropped = missing_h = 0
    hist: Counter = Counter()
    per_bucket_total: Counter = Counter()
    per_bucket_dropped: Counter = Counter()
    # per gallery-pair: (total_pages, art_only_pages) to find whole-gallery pollution
    pair_total: dict = defaultdict(int)
    pair_art: dict = defaultdict(int)

    fclean = open(out_clean, "w") if not args.flag_only else None
    fflag = open(out_flagged, "w") if args.flag_only else None
    fdrop = open(out_dropped, "w") if (args.emit_dropped and not args.flag_only) else None

    with open(args.manifest) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if args.limit and total >= args.limit:
                break
            r = json.loads(line)
            total += 1
            h = r.get("hamming")
            bucket = r.get("source_bucket", "?")
            per_bucket_total[bucket] += 1
            key = (r.get("en_gid"), r.get("jp_gid"))
            pair_total[key] += 1
            if not isinstance(h, (int, float)):
                missing_h += 1
                # treat missing distance as non-pollution (keep)
                if fflag is not None:
                    fflag.write(json.dumps({**r, "art_only": False}) + "\n")
                elif fclean is not None:
                    fclean.write(line + "\n")
                kept += 1
                continue
            hist[min(int(h), 20)] += 1
            is_pollution = h < args.min_hamming
            if is_pollution:
                pair_art[key] += 1
            if args.flag_only:
                fflag.write(json.dumps({**r, "art_only": is_pollution}) + "\n")
                kept += 1  # flag-only keeps everything
                if is_pollution:
                    dropped += 1  # count as flagged
                    per_bucket_dropped[bucket] += 1
                continue
            if is_pollution:
                dropped += 1
                per_bucket_dropped[bucket] += 1
                if fdrop is not None:
                    fdrop.write(line + "\n")
            else:
                kept += 1
                fclean.write(line + "\n")

    for fh in (fclean, fflag, fdrop):
        if fh is not None:
            fh.close()

    # whole-gallery pollution: pairs where every aligned page is art-only
    fully_polluted = [
        {"en_gid": k[0], "jp_gid": k[1], "pages": pair_total[k]}
        for k in pair_total if pair_art.get(k, 0) == pair_total[k] and pair_total[k] > 0
    ]
    fully_polluted.sort(key=lambda d: -d["pages"])

    report = {
        "input": str(args.manifest),
        "min_hamming": args.min_hamming,
        "mode": "flag" if args.flag_only else "split",
        "total_rows": total,
        "kept_rows": kept if not args.flag_only else total - dropped,
        "dropped_rows": dropped,
        "dropped_pct": round(100 * dropped / total, 2) if total else 0.0,
        "missing_hamming_rows": missing_h,
        "per_bucket_total": dict(per_bucket_total),
        "per_bucket_dropped": dict(per_bucket_dropped),
        "hamming_histogram_0to20": {str(k): hist[k] for k in sorted(hist)},
        "fully_polluted_pairs": len(fully_polluted),
        "fully_polluted_top20": fully_polluted[:20],
    }
    out_report.write_text(json.dumps(report, indent=2))

    print(f"[hamming_filter] {args.manifest.name}: total={total} "
          f"dropped(h<{args.min_hamming})={dropped} ({report['dropped_pct']}%) kept={report['kept_rows']}")
    print(f"[hamming_filter] fully-polluted gallery pairs (all pages h<{args.min_hamming}): "
          f"{len(fully_polluted)}")
    if args.flag_only:
        print(f"[hamming_filter] flagged manifest -> {out_flagged}")
    else:
        print(f"[hamming_filter] clean manifest -> {out_clean}")
        if fdrop is not None:
            print(f"[hamming_filter] dropped rows -> {out_dropped}")
    print(f"[hamming_filter] report -> {out_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
