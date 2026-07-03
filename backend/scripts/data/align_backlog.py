#!/usr/bin/env python3
"""Recover the strict JP<->EN alignment BACKLOG: pairs whose images are already on
disk but which never made it into the 375k training export.

Background (audit sec.10 + memory reference_corpus_pov_mine_engine_gate):
  * ``galleries.sqlite::pair_strict`` is the strict-quality candidate universe
    (56,561 EN<->JP gallery pairs).
  * ``mctl_index.sqlite::disk_index(gid, side, n_files)`` is the on-disk inventory
    (35,689 EN + 34,700 JP gallery dirs actually extracted).
  * ``export/manifest_pairs_375k.jsonl`` is what the current export already shipped
    (9,310 gallery pairs -> 374,688 page pairs).

The BACKLOG is therefore:

    strict pairs  WHERE  en side on disk (n_files>0)
                  AND    jp side on disk (n_files>0)
                  AND    (en_gid, jp_gid) NOT already in the 375k export

This is ~4x the current training corpus and needs NO downloading. This script
computes that set and, for the pairs whose per-page pHashes are already cached in
``viewer/review.sqlite::page_phash``, reproduces the SAME page alignment the export
uses (best_match_per_en + lis_constrained) and emits a page-level manifest that is
schema-identical to ``manifest_pages_375k.jsonl`` -- i.e. it can be fed straight into
the existing ``corpus_bitext/run_gallery.py`` via ``--manifest``.

Pairs that are strict + on-disk but have NO cached page_phash are emitted separately
as ``needs_scoring`` -- they must go through the pHash page-scoring step first
(compute_full_alignment) before the OCR pipeline can consume them.

LOCAL-ONLY, read-only, stdlib-only. Never writes under /mnt/nas. Does not touch any
existing corpus_bitext / export file. The page alignment functions below are copied
verbatim from ``data/manga_datasets/merged/viewer/compute_full_alignment.py`` so the
emitted alignment is byte-identical to the production export; keep them in sync.

Outputs (under --out-dir):
  backlog_pairs.jsonl        one strict-on-disk-unexported gallery pair per line
                             {en_gid, jp_gid, en_dir, jp_dir, pair_pop,
                              on_disk_en_files, on_disk_jp_files, scored,
                              aligned_count, aligned_avg, coverage_en, coverage_jp}
  manifest_pages_backlog.jsonl  page-level rows for the SCORED backlog pairs, in the
                             exact manifest_pages_375k.jsonl schema (feeds run_gallery)
  backlog_needs_scoring.jsonl   backlog pairs lacking cached page_phash (need pHash step)
  backlog_stats.json         counts + reconciliation vs the audit's ~34,274 estimate
"""
from __future__ import annotations

import argparse
import bisect
import json
import sqlite3
import time
from pathlib import Path
from typing import Optional

# --- default paths (main checkout; overridable) -----------------------------
_MERGED = Path("/home/danny/Documents/personal/extension/data/manga_datasets/merged")
DEFAULT_GALLERIES = _MERGED / "galleries.sqlite"
DEFAULT_MCTL = _MERGED / "mctl_index.sqlite"
DEFAULT_REVIEW = _MERGED / "viewer/review.sqlite"
DEFAULT_EXPORT_PAIRS = _MERGED / "export/manifest_pairs_375k.jsonl"
DEFAULT_NAS = "/mnt/nas/drive_2/manga-ml/ehentai_corpus/galleries"

MATCH_THRESHOLD = 22   # hamming distance considered "aligned" (matches production)
DUP_THRESHOLD = 3      # below this, image is essentially identical


# --- alignment algorithm (verbatim from viewer/compute_full_alignment.py) ----
def hamming(a: int, b: int) -> int:
    return bin((a & 0xFFFFFFFFFFFFFFFF) ^ (b & 0xFFFFFFFFFFFFFFFF)).count("1")


def load_hashes(conn: sqlite3.Connection, gid: int, side: str) -> list[Optional[int]]:
    rows = conn.execute(
        "SELECT page_idx, hash_bits FROM page_phash WHERE gid=? AND side=? ORDER BY page_idx",
        (gid, side)).fetchall()
    if not rows:
        return []
    max_idx = max(r[0] for r in rows)
    out: list[Optional[int]] = [None] * (max_idx + 1)
    for idx, h in rows:
        out[idx] = h
    return out


def best_match_per_en(en_hashes, jp_hashes, threshold: int = MATCH_THRESHOLD):
    matches = []
    for i, h_en in enumerate(en_hashes):
        if h_en is None:
            continue
        best_j = -1
        best_d = threshold + 1
        for j, h_jp in enumerate(jp_hashes):
            if h_jp is None:
                continue
            d = hamming(h_en, h_jp)
            if d < best_d:
                best_d = d
                best_j = j
        if best_j >= 0 and best_d <= threshold:
            matches.append((i, best_j, best_d))
    return matches


def lis_constrained(matches):
    if not matches:
        return []
    matches = sorted(matches, key=lambda x: (x[0], x[1]))
    tails: list[int] = []
    tail_match_idx: list[int] = []
    parent: list[int] = [-1] * len(matches)
    for k, (_, j, _) in enumerate(matches):
        pos = bisect.bisect_left(tails, j)
        if pos == len(tails):
            tails.append(j)
            tail_match_idx.append(k)
        else:
            tails[pos] = j
            tail_match_idx[pos] = k
        parent[k] = tail_match_idx[pos - 1] if pos > 0 else -1
    seq: list[int] = []
    cur = tail_match_idx[-1] if tail_match_idx else -1
    while cur != -1:
        seq.append(cur)
        cur = parent[cur]
    seq.reverse()
    return [matches[k] for k in seq]


# --- backlog selection ------------------------------------------------------
def load_exported_pairs(path: Path) -> set[tuple[int, int]]:
    exp: set[tuple[int, int]] = set()
    if not path.exists():
        return exp
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            exp.add((r["en_gid"], r["jp_gid"]))
    return exp


def compute_backlog(gconn: sqlite3.Connection, mctl_path: Path,
                    exported: set[tuple[int, int]], min_files: int) -> list[dict]:
    """strict pairs with both sides on disk (n_files>=min_files) and NOT exported."""
    gconn.execute("ATTACH ? AS m", (str(mctl_path),))
    rows = gconn.execute(
        """
        SELECT ps.en_gid, ps.jp_gid, ps.pair_pop, de.n_files AS en_files, dj.n_files AS jp_files
        FROM pair_strict ps
        JOIN m.disk_index de ON de.gid = ps.en_gid AND de.side='en' AND de.n_files >= ?
        JOIN m.disk_index dj ON dj.gid = ps.jp_gid AND dj.side='jp' AND dj.n_files >= ?
        """,
        (min_files, min_files),
    ).fetchall()
    out = []
    for en, jp, pop, enf, jpf in rows:
        if (en, jp) in exported:
            continue
        out.append({"en_gid": en, "jp_gid": jp, "pair_pop": pop,
                    "on_disk_en_files": enf, "on_disk_jp_files": jpf})
    out.sort(key=lambda d: (d["en_gid"], d["jp_gid"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--galleries-db", type=Path, default=DEFAULT_GALLERIES)
    ap.add_argument("--mctl-db", type=Path, default=DEFAULT_MCTL)
    ap.add_argument("--review-db", type=Path, default=DEFAULT_REVIEW)
    ap.add_argument("--export-pairs", type=Path, default=DEFAULT_EXPORT_PAIRS,
                    help="manifest_pairs_375k.jsonl = the already-exported pair set")
    ap.add_argument("--nas-prefix", default=DEFAULT_NAS,
                    help="path prefix written into en_path/jp_path (run_gallery remaps to staging)")
    ap.add_argument("--out-dir", type=Path, default=Path("."))
    ap.add_argument("--min-files", type=int, default=1,
                    help="min n_files on each side to count a dir as present (default 1)")
    ap.add_argument("--limit-pairs", type=int, default=0,
                    help="only emit page rows for the first N scored pairs (sampling / smoke)")
    ap.add_argument("--count-only", action="store_true",
                    help="print the backlog count + breakdown and exit (no files written)")
    args = ap.parse_args()

    t0 = time.time()
    gconn = sqlite3.connect(f"file:{args.galleries_db}?mode=ro", uri=True)
    exported = load_exported_pairs(args.export_pairs)
    backlog = compute_backlog(gconn, args.mctl_db, exported, args.min_files)
    print(f"[align_backlog] strict-on-disk-unexported pairs: {len(backlog)}  "
          f"(exported set={len(exported)}, min_files={args.min_files})")

    # split scored (has cached page_phash) vs needs_scoring
    rconn = sqlite3.connect(f"file:{args.review_db}?mode=ro", uri=True)
    scored_pairs = set(
        (e, j) for e, j in rconn.execute("SELECT en_gid, jp_gid FROM pair_phash"))
    for b in backlog:
        b["scored"] = (b["en_gid"], b["jp_gid"]) in scored_pairs
    n_scored = sum(1 for b in backlog if b["scored"])
    print(f"[align_backlog] of those: scored(page_phash cached)={n_scored}  "
          f"needs_scoring={len(backlog) - n_scored}")

    if args.count_only:
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    nas = args.nas_prefix.rstrip("/")

    backlog_path = args.out_dir / "backlog_pairs.jsonl"
    pages_path = args.out_dir / "manifest_pages_backlog.jsonl"
    needs_path = args.out_dir / "backlog_needs_scoring.jsonl"

    total_pages = 0
    aligned_pairs = 0
    n_scored_emit = 0
    with open(backlog_path, "w") as fbl, open(pages_path, "w") as fpg, \
            open(needs_path, "w") as fns:
        for b in backlog:
            en, jp = b["en_gid"], b["jp_gid"]
            rec = {
                "en_gid": en, "jp_gid": jp,
                "en_dir": f"{nas}/{en}_en", "jp_dir": f"{nas}/{jp}_jp",
                "pair_pop": b["pair_pop"],
                "on_disk_en_files": b["on_disk_en_files"],
                "on_disk_jp_files": b["on_disk_jp_files"],
                "scored": b["scored"],
            }
            if not b["scored"]:
                fns.write(json.dumps(rec) + "\n")
                fbl.write(json.dumps(rec) + "\n")
                continue

            if args.limit_pairs and n_scored_emit >= args.limit_pairs:
                # still record the pair in the master list, just skip page emission
                fbl.write(json.dumps({**rec, "aligned_count": None}) + "\n")
                continue

            en_h = load_hashes(rconn, en, "en")
            jp_h = load_hashes(rconn, jp, "jp")
            aligned = lis_constrained(best_match_per_en(en_h, jp_h))
            n_scored_emit += 1
            if not aligned:
                rec["aligned_count"] = 0
                fbl.write(json.dumps(rec) + "\n")
                continue
            dists = [d for _, _, d in aligned]
            n_en = sum(1 for h in en_h if h is not None)
            n_jp = sum(1 for h in jp_h if h is not None)
            rec.update({
                "aligned_count": len(aligned),
                "aligned_avg": sum(dists) / len(dists),
                "coverage_en": len(aligned) / n_en if n_en else 0.0,
                "coverage_jp": len(aligned) / n_jp if n_jp else 0.0,
            })
            fbl.write(json.dumps(rec) + "\n")
            aligned_pairs += 1
            for ei, ji, d in aligned:
                fpg.write(json.dumps({
                    "en_gid": en, "jp_gid": jp,
                    "en_path": f"{nas}/{en}_en/{ei + 1:04d}.jpg",
                    "jp_path": f"{nas}/{jp}_jp/{ji + 1:04d}.jpg",
                    "hamming": d,
                    "status": "backlog_strict",
                    "source_bucket": "backlog_strict",
                }) + "\n")
                total_pages += 1

    stats = {
        "backlog_pairs": len(backlog),
        "scored_pairs": n_scored,
        "needs_scoring_pairs": len(backlog) - n_scored,
        "scored_pairs_emitted": n_scored_emit,
        "aligned_pairs_emitted": aligned_pairs,
        "page_pairs_emitted": total_pages,
        "exported_pairs_excluded": len(exported),
        "min_files": args.min_files,
        "limit_pairs": args.limit_pairs,
        "audit_estimate": 34274,
        "delta_vs_audit": len(backlog) - 34274,
        "elapsed_s": round(time.time() - t0, 2),
        "note": ("page_pairs_emitted counts only the SCORED subset actually emitted; "
                 "full page yield requires page-scoring the needs_scoring pairs first "
                 "(viewer/compute_full_alignment) then re-running without --limit-pairs."),
    }
    (args.out_dir / "backlog_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"[align_backlog] wrote {backlog_path.name} ({len(backlog)} pairs), "
          f"{pages_path.name} ({total_pages} page rows from {aligned_pairs} pairs), "
          f"{needs_path.name} ({len(backlog) - n_scored} pairs)")
    print(f"[align_backlog] stats -> {(args.out_dir / 'backlog_stats.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
