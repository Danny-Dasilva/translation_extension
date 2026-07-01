# Restore Plan: e-hentai/nhentai page-images from the 3.3 TB drive_1 tar

Date: 2026-06-28
Author: research/characterization pass (bounded I/O, NO extraction performed)
Archive: `/mnt/nas/drive_1/manga-ml/ehentai_corpus.tar` (3,554,874,664,960 bytes ≈ 3.232 TiB / 3.55 TB)

---

## TL;DR

- The training manifests point at `…/galleries/<eh_gid>_<en|jp>/NNNN.jpg` on the
  **reaped** drive_2. Those derived galleries are gone.
- The surviving drive_1 tar, **within the scanned first ~10–20 GB**, contains the
  *upstream sources* — `jabbany_exhentai/` (jsonl metadata), `nhentai/` (only TWO
  sample galleries), and `nhentai_cache/` (a HuggingFace `datasets` cache of
  `infinity-blackhole/nhentai`). It does **NOT** contain a `galleries/` subtree in
  the scanned region, and `galleries/` would have to be **rebuilt**, not just
  extracted.
- **CRITICAL CAVEAT (flagged):** the tar is in **creation/append order, not
  alphabetical**, and CIFS header-skip throughput is only ~10 MB/s, so a full
  `tar -tf` index is a multi-hour-to-overnight job. We could **not** prove whether
  `galleries/`, `pre_downloaded/DMCA*`, or `archive_ubuca_v5_*` exist *later* in
  the 3.3 TB tail within bounded I/O. **Step 0 below builds the definitive index.**
- The needed subset is small enough to extract to **local disk** (~120 GB for the
  current 134k manifest; ~335 GB projected for a 375k manifest; local free = 503 GB).
  The full 3.3 TB does **not** fit locally.

---

## 1. Layout verdict

**Manifest layout (target we must reproduce):**
```
/mnt/nas/drive_2/manga-ml/ehentai_corpus/galleries/<eh_gid>_en/NNNN.jpg   (4-digit, 1-indexed)
/mnt/nas/drive_2/manga-ml/ehentai_corpus/galleries/<eh_gid>_jp/NNNN.jpg
```
(verified: `manifest_pages.jsonl` rows, `manifest_pairs.jsonl` `en_dir`/`jp_dir`)

**Tar contents (VERIFIED, first ~10–20 GB only):**
```
manga-ml/ehentai_corpus/
  jabbany_exhentai/no_cg_cosplay_all.jsonl        (547 MB metadata)
  jabbany_exhentai/no_cg_cosplay_all.jsonl.bz2    (60 MB)
  jabbany_exhentai/sample-100.jsonl
  nhentai/__pycache__/nhentai.cpython-311.pyc
  nhentai/sample_367568/1.jpg .. N.jpg            (ONE sample gallery, ~30 pp)
  nhentai/sample_367569/…                          (ONE sample gallery)
  nhentai_cache/infinity-blackhole___nhentai/nhentai/0.0.0/   (HF datasets cache)
  …                                                (3.3 TB unscanned tail)
```
Page filenames in the tar's `nhentai/` are **unpadded** (`1.jpg`, `10.jpg`), unlike
the manifest's zero-padded `0001.jpg`.

**Verdict: NO `galleries/<gid>_<en|jp>/` subtree was found in the scanned region.**
A bounded `grep -m5 '/galleries/'` over the tar ran the full 120 s budget and
returned **zero** matches (TIMEOUT — see Flags). The galleries layout must be
**rebuilt**, not merely extracted.

**Why galleries/ is a *derived* tree (read the scripts — confirmed):**
- `data/manga_datasets/merged/build_merged.py` builds **only `galleries.sqlite`
  metadata** (gallery/tag tables). It does **not** create any image directory.
- `data/manga_datasets/merged/extract_ready_pairs.py` populates
  `galleries/<eh_gid>_<side>/NNNN.<ext>` by **extracting/copying** image members
  out of zip/dir sources under `NAS_ROOT/pre_downloaded/DMCA*` and
  `archive_ubuca_v5_p1|p2`, renaming them to sequential `0001.jpg…`. (a renamed
  **copy**, not a symlink.)
- `data/manga_datasets/merged/download_pairs_offline_map.py` is the alternative
  producer: it downloads each side **directly from the i.nhentai.net CDN** using
  the `eh_nh_map` (eh_gid→nh_id) join and writes the same
  `galleries/<eh_gid>_<en|jp>/NNNN.<ext>` layout.

So `galleries/` images come from EITHER (a) DMCA/ubuca zips, OR (b) the nhentai
CDN keyed by `nh_id`. The HF `nhentai_cache` in the tar is the offline mirror of
that nhentai source.

---

## 2. gid → tar-path mapping

Manifest gids are **e-hentai gids (`eh_gid`)**; the tar's nhentai dirs are keyed by
**nhentai id (`nh_id`)** (sample dir = `sample_<nh_id>`, e.g. `sample_367568`).
There is **no direct path correspondence**. The join is the local
`eh_nh_map` table in `galleries.sqlite` (2.6 GB, present locally):

| manifest eh_gid (side) | → nh_id | confidence | match_type |
|---|---|---|---|
| 16972 (en) | 5223 | 1.0 | title_artist_page |
| 532 (jp) | 126076 | 1.0 | title_artist_page |
| 6788 (en) | 2528 | 1.0 | title_artist_page |
| 1458 (jp) | 362 | 1.0 | title_artist_page |

Chain: `manifest eh_gid` → `eh_nh_map.nh_id` → nhentai source page (CDN dir / HF
dataset record keyed by `nh_id`/`media_id`) → renamed to `galleries/<eh_gid>_<side>/NNNN.jpg`.

**Membership hit/miss (bounded):**
- `/galleries/` membership: **MISS within budget** (no hit in 120 s; TIMEOUT — see Flags).
- Direct `nhentai/<nh_id>/` membership for real nh_ids (5223, 2528, 362): **NOT
  bounded-testable** — the tar's `nhentai/` held only the two `sample_*` dirs, then
  switched to the `nhentai_cache/` HF arrow cache (real pages are inside the
  dataset cache, not browsable as `nhentai/<nh_id>/N.jpg`). Confirming requires the
  Step-0 index or loading the HF dataset.

---

## 3. Disk math

**Manifest scale (current `manifest_pages.jsonl`, VERIFIED):**
- 134,016 page-pair rows
- 3,591 distinct `en_gid`, 2,886 distinct `jp_gid` → **6,477 gallery dirs**
- **239,002 unique image paths** referenced (en+jp aligned pages)
- pairs: 3,503 `good` + 88 `partial` = 3,591 (the current manifest is *already*
  all good+partial)

**Page byte size (VERIFIED, sampled from tar `nhentai/sample_367568/`):**
~313 KB – ~794 KB, **mean ≈ 605 KB** (nhentai full-res). DMCA/ubuca scanlation
pages tend smaller; plan with a **400–600 KB** band.

**Subset size estimates:**

| Subset | Unique imgs | @400 KB | @500 KB | @600 KB |
|---|---|---|---|---|
| Current 134k manifest (aligned pages only) | 239,002 | ~95 GB | ~119 GB | ~143 GB |
| Whole gallery dirs (6,477 × ~30 pp) | ~194,000 | ~78 GB | ~97 GB | ~117 GB |
| Projected 375k manifest (~2.8× pairs) | ~668,000 | ~267 GB | ~335 GB | ~401 GB |

**Free space:**
- Repo/local disk `/dev/nvme1n1p2`: 1.8 TB, **503 GB free** (72% used)
- drive_1 (CIFS): 22 TB, **16 TB free** — holds the tar; **reaping behavior UNKNOWN**
- drive_2 (CIFS): 22 TB, 5.6 TB free — **REAPS** (do not stage here)

**Feasibility:**
- Full 3.3 TB extraction to local: **NO** (3.3 TB ≫ 503 GB free).
- Selective extraction of the **134k** subset to local: **YES** (~120 GB).
- Selective extraction of the **375k** subset to local: **YES but tight**
  (~335 GB of 503 GB free) → use the phased option (§4.5).

---

## 4. Restore plan

### Step 0 (PREREQUISITE — settle the layout question definitively)

Build the full member index ONCE as a background/overnight job to **local** disk
(NOT /mnt/nas). This is the only way to know whether `galleries/`,
`pre_downloaded/DMCA*`, or `archive_ubuca_v5_*` exist in the 3.3 TB tail:

```bash
# overnight; ~hours at the observed ~10 MB/s CIFS header-skip rate
nohup tar -tvf /mnt/nas/drive_1/manga-ml/ehentai_corpus.tar \
  > /home/danny/Documents/personal/extension/data/restore/tar_index.txt 2> tar_index.err &
```
Then classify the top-level layout:
```bash
grep -oE 'manga-ml/ehentai_corpus/[^/]+/' data/restore/tar_index.txt | sort -u
grep -m1 '/galleries/' data/restore/tar_index.txt && echo "GALLERIES PRESENT"
```

The branch taken below depends on Step 0's result.

### 4.A — IF `galleries/<gid>_<en|jp>/` IS present in the tar (extract path)

1. Build the exact member list from the manifest (local, fast):
   ```bash
   python3 - <<'PY'
   import json, pathlib
   PREFIX = "/mnt/nas/drive_2/manga-ml/ehentai_corpus/"      # manifest abs prefix
   TARPREFIX = "manga-ml/ehentai_corpus/"                    # member prefix in tar
   out = pathlib.Path("data/restore/members.txt"); out.parent.mkdir(parents=True, exist_ok=True)
   seen = set()
   with open("data/manga_datasets/merged/export/manifest_pages.jsonl") as f, out.open("w") as w:
       for line in f:
           d = json.loads(line)
           for k in ("en_path","jp_path"):
               m = d[k].replace(PREFIX, TARPREFIX)
               if m not in seen:
                   seen.add(m); w.write(m + "\n")
   print(len(seen), "members ->", out)
   PY
   ```
   (To restore WHOLE galleries instead of aligned pages only, emit
   `--wildcards` dir patterns `…/galleries/<gid>_en/*` from the distinct gids.)

2. Extract to LOCAL staging using the member list (single sequential pass —
   tar streams the whole archive once but only writes matched members):
   ```bash
   DEST=/home/danny/Documents/personal/extension/data/restore/galleries
   mkdir -p "$DEST"
   tar -xvf /mnt/nas/drive_1/manga-ml/ehentai_corpus.tar \
       -C "$DEST" --strip-components=2 \
       --files-from=data/restore/members.txt
   ```
   - `--strip-components=2` drops the `manga-ml/ehentai_corpus/` prefix so the
     result is `…/data/restore/galleries/<gid>_en/NNNN.jpg`.
   - NOTE: `--files-from` with an exact path list is the fast/correct form.
     `--wildcards` works too but is per-pattern. Either way tar makes ONE
     sequential pass over the 3.3 TB; budget the full read time.

### 4.B — IF `galleries/` is ABSENT (rebuild path — current evidence favors this)

The galleries layout must be regenerated from the nhentai source the tar *does*
hold (`nhentai_cache/infinity-blackhole___nhentai/…`) plus the local
`eh_nh_map`:

1. Extract just the HF nhentai dataset cache to local staging:
   ```bash
   DEST=/home/danny/Documents/personal/extension/data/restore/nhentai_cache
   mkdir -p "$DEST"
   tar -xvf /mnt/nas/drive_1/manga-ml/ehentai_corpus.tar -C "$DEST" \
       --strip-components=2 \
       --wildcards 'manga-ml/ehentai_corpus/nhentai_cache/*'
   # (size unknown until Step 0; verify it fits in 503 GB before running)
   ```
2. Determine the nh_ids needed (join manifest gids → eh_nh_map):
   ```bash
   python3 - <<'PY'
   import json, sqlite3
   con = sqlite3.connect("data/manga_datasets/merged/galleries.sqlite")
   gids=set()
   with open("data/manga_datasets/merged/export/manifest_pages.jsonl") as f:
       for line in f:
           d=json.loads(line); gids.add(d["en_gid"]); gids.add(d["jp_gid"])
   q="SELECT eh_gid,nh_id FROM eh_nh_map WHERE eh_gid IN (%s)" % ",".join("?"*len(gids))
   m=dict(con.execute(q, tuple(gids)).fetchall())
   print("gids:",len(gids),"mapped nh_ids:",len(set(m.values())))
   json.dump(m, open("data/restore/eh2nh.json","w"))
   PY
   ```
3. Materialize `galleries/<eh_gid>_<side>/NNNN.jpg` from the extracted nhentai
   pages keyed by nh_id. Reuse the existing renaming/ordering logic — adapt
   `download_pairs_offline_map.py`’s layout writer to read pages from the **local
   extracted cache** instead of the CDN (offline mode), writing to local
   `data/restore/galleries/`. If the HF dataset proves awkward to address
   per-nh_id, the fallback is to re-download via the existing
   `download_pairs_offline_map.py` (runs inside the airvpn netns — see project
   memory `project_exsafe_restart.md`).

### 4.5 — Phasing (recommended for the 375k case)

- **Phase 1 — good+partial first (~120 GB, fits easily):** the current
  `manifest_pages.jsonl` (134k pairs, 3,503 good + 88 partial) is already exactly
  the good+partial set. Restore it first.
- **Phase 2 — unreviewed_hq remainder:** when `manifest_pages_375k.jsonl` lands
  (being built by another agent; **does not exist yet** as of this writing —
  Flag), diff its image paths against Phase 1’s restored set and extract only the
  delta (~+215 GB). Watch local free space (503 GB total) — if Phase 1+2 exceeds
  ~450 GB, stage Phase 2 on drive_1 (16 TB free) **only after** confirming
  drive_1 does not reap (run a 15-min canary write/read on drive_1 first).

### Target directory

Stage to **local disk**:
`/home/danny/Documents/personal/extension/data/restore/galleries/`
- drive_2 **REAPS** — never stage output there.
- drive_1 reaping is **UNKNOWN**; the tar lives there but treat drive_1 writes as
  unverified until a canary test passes. Prefer local (503 GB free) for the
  134k/Phase-1 set.

---

## 5. Re-point / validate the manifest against the restored layout

1. **Re-point** paths from the reaped drive_2 prefix to the local restore root:
   ```bash
   python3 - <<'PY'
   import json
   SRC="/mnt/nas/drive_2/manga-ml/ehentai_corpus/galleries"
   DST="/home/danny/Documents/personal/extension/data/restore/galleries"
   with open("data/manga_datasets/merged/export/manifest_pages.jsonl") as f, \
        open("data/manga_datasets/merged/export/manifest_pages.local.jsonl","w") as w:
       for line in f:
           d=json.loads(line)
           d["en_path"]=d["en_path"].replace(SRC,DST)
           d["jp_path"]=d["jp_path"].replace(SRC,DST)
           w.write(json.dumps(d)+"\n")
   PY
   ```
2. **Validate** every referenced path now exists + is a decodable image:
   ```bash
   python3 - <<'PY'
   import json, os
   miss=bad=ok=0
   for line in open("data/manga_datasets/merged/export/manifest_pages.local.jsonl"):
       d=json.loads(line)
       for k in ("en_path","jp_path"):
           p=d[k]
           if not os.path.exists(p): miss+=1
           elif os.path.getsize(p)==0: bad+=1
           else: ok+=1
   print(f"ok={ok} missing={miss} empty={bad}")
   PY
   ```
   (Optionally add a `PIL.Image.open(p).verify()` pass to catch truncated JPEGs.)
3. Cross-check page COUNT per restored gallery vs `manifest_pairs.jsonl`
   `en_pages`/`jp_pages`; mismatches flag galleries that rebuilt with the wrong
   page ordering (the unpadded→padded `N.jpg`→`NNNN.jpg` rename is the most likely
   failure point in the rebuild path 4.B).

---

## Flags (probes that hit their timeout / could not be settled within bounds)

- **`/galleries/` membership = TIMEOUT.** `grep -m5 '/galleries/'` over `tar -tf`
  consumed its full 120 s budget with **zero** matches. This is strong-but-not-
  conclusive evidence of absence: the tar is **creation-order, not alphabetical**
  (root order observed: jabbany → nhentai → nhentai_cache; `archive_*`/`galleries`
  which sort earlier never appeared first), so `galleries/` could still live in the
  unscanned ~3.3 TB tail. **Only Step 0’s full index settles this.**
- **CIFS enumeration throughput ≈ 10 MB/s** (60 s of `tar -tvf` reached only
  ~608 MB in). Full `tar -tf` index ≈ many hours / overnight. Selective
  `tar -x --files-from` likewise makes ONE full sequential pass over 3.3 TB.
- **`nhentai/<nh_id>/` direct membership = NOT bounded-testable.** Real pages are
  inside the `nhentai_cache` HF arrow cache, not a browsable `nhentai/<id>/` tree.
- **`manifest_pages_375k.jsonl` does NOT exist yet** (only `manifest_pages.jsonl`
  134k and `manifest_pairs.jsonl` 3,591 are present). All 375k numbers here are
  projections (×~2.8 on the 134k pair count).

## Key files
- `data/manga_datasets/merged/export/manifest_pages.jsonl` — 134,016 rows
- `data/manga_datasets/merged/export/manifest_pairs.jsonl` — 3,591 rows
- `data/manga_datasets/merged/galleries.sqlite` — has `eh_nh_map` (eh_gid→nh_id)
- `data/manga_datasets/merged/extract_ready_pairs.py` — galleries/ builder (zip→layout)
- `data/manga_datasets/merged/download_pairs_offline_map.py` — galleries/ builder (CDN→layout)
- `data/manga_datasets/merged/export_manifest.py` — manifest generator (reads galleries/ on drive_2)
- `/mnt/nas/drive_1/manga-ml/ehentai_corpus.tar` — the 3.3 TB archive
