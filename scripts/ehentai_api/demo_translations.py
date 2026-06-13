"""
Demo: pick an English-tagged gallery, walk to its Japanese source.

Run:  python3 scripts/ehentai_api/demo_translations.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ehentai_api import EHentaiClient  # noqa: E402


def main() -> int:
    client = EHentaiClient(rate_limit=1.5)

    for page in range(5):
        for gid, tok in client.search("language:english", page=page):
            info = client.get_gallery_page(gid, tok)
            if not info["parent"]:
                continue

            # Resolve metadata for seed + parent to compare languages.
            both = client.gdata_batched([(gid, tok), info["parent"]])
            by_gid = {g.gid: g for g in both}
            en = by_gid.get(gid)
            parent = by_gid.get(info["parent"][0])
            if not (en and parent):
                continue

            en_langs = set(en.languages)
            p_langs = set(parent.languages)

            # We want a true EN -> JP crossover (parent has no 'translated' tag
            # and is implicitly/explicitly Japanese).
            if "translated" in p_langs:
                continue
            if "english" not in en_langs:
                continue

            print("=" * 70)
            print("English version:")
            print(f"  {en.url}")
            print(f"  title:    {en.title[:80]}")
            print(f"  title_jp: {en.title_jpn[:80]}")
            print(f"  langs:    {en.languages}")
            print(f"  tags:     {en.tags[:6]}...")
            print()
            print("Japanese (parent):")
            print(f"  {parent.url}")
            print(f"  title:    {parent.title[:80]}")
            print(f"  title_jp: {parent.title_jpn[:80]}")
            print(f"  langs:    {parent.languages or '(untagged -> likely Japanese)'}")
            print()

            # Show the whole translation family grouped by language.
            by_lang = client.find_translations(gid, tok)
            print("All known versions, grouped by language:")
            for lang, galleries in sorted(by_lang.items()):
                key = lang or "(untagged)"
                print(f"  [{key}]  {len(galleries)} gallery(ies)")
                for g in galleries[:4]:
                    print(f"    - {g.gid}/{g.token[:8]}  {g.title[:70]}")
            return 0

    print("no EN->JP crossover in first 5 search pages")
    return 1


if __name__ == "__main__":
    sys.exit(main())
