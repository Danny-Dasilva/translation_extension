"""
Live tests for the wrapper. Hits the real e-hentai API + HTML listing.

Run:  python3 -m scripts.ehentai_api.test_live
(from the repo root) or just:  python3 scripts/ehentai_api/test_live.py
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

# Allow running as a script without -m.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ehentai_api import EHentaiClient  # noqa: E402

# A known, stable public gallery used in the ehwiki API examples.
KNOWN_GID = 618395
KNOWN_TOKEN = "0439fa3666"


def banner(msg: str) -> None:
    print(f"\n{'=' * 8} {msg} {'=' * 8}")


def ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def test_parse_url(client: EHentaiClient) -> bool:
    banner("parse_gallery_url")
    try:
        gid, tok = client.parse_gallery_url(
            f"https://e-hentai.org/g/{KNOWN_GID}/{KNOWN_TOKEN}/"
        )
        assert (gid, tok) == (KNOWN_GID, KNOWN_TOKEN), (gid, tok)
        ok(f"parsed -> ({gid}, {tok})")
        return True
    except Exception as e:
        fail(f"{e}")
        return False


def test_gdata(client: EHentaiClient) -> bool:
    banner("gdata (single)")
    try:
        results = client.gdata([(KNOWN_GID, KNOWN_TOKEN)])
        if not results:
            fail("no gallery returned")
            return False
        g = results[0]
        print(f"  title:    {g.title[:80]}")
        print(f"  title_jp: {g.title_jpn[:80]}")
        print(f"  cat:      {g.category}  uploader={g.uploader}")
        print(f"  files:    {g.filecount}  size={g.filesize}")
        print(f"  rating:   {g.rating}")
        print(f"  langs:    {g.languages}")
        print(f"  tags[:5]: {g.tags[:5]}")
        assert g.gid == KNOWN_GID
        assert g.token == KNOWN_TOKEN
        assert g.filecount > 0
        ok("metadata looks valid")
        return True
    except Exception as e:
        traceback.print_exc()
        fail(f"{e}")
        return False


def test_gdata_batched(client: EHentaiClient) -> bool:
    banner("gdata_batched (2 galleries)")
    try:
        # Two arbitrary recent galleries from the front page, fetched fresh.
        pairs = client.search("", page=0)[:2]
        if len(pairs) < 2:
            fail("couldn't sample 2 galleries from front page")
            return False
        results = client.gdata_batched(pairs)
        for g in results:
            print(f"  {g.gid}/{g.token[:8]}  {g.category:12s}  {g.title[:60]}")
        assert len(results) == len(pairs)
        ok(f"fetched {len(results)} metadata entries")
        return True
    except Exception as e:
        traceback.print_exc()
        fail(f"{e}")
        return False


def test_search(client: EHentaiClient) -> bool:
    banner("search (query='')")
    try:
        pairs = client.search("", page=0)
        print(f"  front-page hits: {len(pairs)}")
        print(f"  first 3: {pairs[:3]}")
        assert len(pairs) > 5, "expected at least a few results from front page"
        ok(f"got {len(pairs)} gallery pairs")
        return True
    except Exception as e:
        traceback.print_exc()
        fail(f"{e}")
        return False


def test_search_query(client: EHentaiClient) -> bool:
    banner("search (text query)")
    try:
        pairs = client.search("language:japanese", page=0)
        print(f"  hits: {len(pairs)}; first: {pairs[:2]}")
        assert len(pairs) > 0, "expected hits for language:japanese"
        ok("non-empty result set")
        return True
    except Exception as e:
        traceback.print_exc()
        fail(f"{e}")
        return False


def _find_seed_with_parent(client: EHentaiClient, max_pages: int = 3):
    """Scan language:english pages until a gallery with a Parent link is found."""
    for page in range(max_pages):
        for gid, tok in client.search("language:english", page=page):
            info = client.get_gallery_page(gid, tok)
            if info["parent"]:
                return gid, tok, info
    return None


def test_gallery_page_scrape(client: EHentaiClient) -> bool:
    banner("get_gallery_page (parent + newer_versions)")
    try:
        seed = _find_seed_with_parent(client)
        if seed is None:
            print("  (no parent-linked gallery found; skipping)")
            ok("skipped — no sample data")
            return True
        gid, tok, info = seed
        print(f"  seed:           {gid}/{tok}")
        print(f"  parent:         {info['parent']}")
        print(f"  newer_versions: {info['newer_versions'][:3]}")
        assert info["parent"] is not None
        ok("parent link parsed from HTML")
        return True
    except Exception as e:
        traceback.print_exc()
        fail(f"{e}")
        return False


def test_find_translations(client: EHentaiClient) -> bool:
    banner("find_translations (parent chain + language buckets)")
    try:
        seed = _find_seed_with_parent(client)
        if seed is None:
            print("  (no parent-linked gallery found; skipping)")
            ok("skipped — no sample data")
            return True
        gid, tok, _ = seed
        print(f"  seed: {gid}/{tok}")

        chain = client.walk_parent_chain(gid, tok)
        print(f"  parent chain ({len(chain)} hops): {chain}")

        by_lang = client.find_translations(gid, tok)
        for lang, galleries in sorted(by_lang.items()):
            key = lang or "(untagged)"
            print(f"  {key}: {len(galleries)}")
            for g in galleries[:2]:
                print(f"      - {g.gid}/{g.token[:8]}  {g.title[:60]}")
        assert by_lang, "expected at least one language bucket"
        assert len(chain) >= 2, f"expected parent chain of 2+, got {len(chain)}"
        ok(f"walked {len(chain)}-deep chain; {len(by_lang)} language buckets")
        return True
    except Exception as e:
        traceback.print_exc()
        fail(f"{e}")
        return False


def main() -> int:
    client = EHentaiClient(rate_limit=1.5)
    tests = [
        test_parse_url,
        test_gdata,
        test_search,
        test_search_query,
        test_gdata_batched,
        test_gallery_page_scrape,
        test_find_translations,
    ]
    results = [t(client) for t in tests]
    banner("summary")
    passed = sum(results)
    print(f"  {passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
