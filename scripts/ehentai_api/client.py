"""
Minimal e-hentai / ExHentai API wrapper.

Covers:
  * gdata  - batch gallery metadata (JSON API)
  * gtoken - convert page image links into (gid, token)
  * search - scrape /?f_search= listing
  * parent chain + sibling translations (English <-> Japanese)

Docs: https://ehwiki.org/wiki/API
Rate limit: ~5 sequential gdata calls, then sleep ~5 seconds.
"""

from __future__ import annotations

import html
import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

import requests

EH_HOST = "e-hentai.org"
EX_HOST = "exhentai.org"
API_PATH = "/api.php"

GALLERY_RE = re.compile(r"/g/(\d+)/([0-9a-f]{10})/?")


@dataclass
class Gallery:
    """Subset of gdata fields we care about, plus a couple of parsed extras."""

    gid: int
    token: str
    title: str
    title_jpn: str
    category: str
    uploader: str
    posted: int  # unix epoch
    filecount: int
    filesize: int
    rating: float
    tags: list[str]
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    # Convenience: language tags derived from `tags`
    @property
    def languages(self) -> list[str]:
        return [t.split(":", 1)[1] for t in self.tags if t.startswith("language:")]

    @property
    def is_translated(self) -> bool:
        return "translated" in self.languages

    @property
    def url(self) -> str:
        return f"https://e-hentai.org/g/{self.gid}/{self.token}/"


class EHentaiError(RuntimeError):
    pass


class EHentaiClient:
    """
    Thin wrapper around the e-hentai JSON API + listing scraper.

    Pass `cookies` with keys ipb_member_id / ipb_pass_hash (and igneous for ExHentai)
    to access ExHentai / adult content.
    """

    def __init__(
        self,
        cookies: dict[str, str] | None = None,
        *,
        exhentai: bool = False,
        rate_limit: float = 1.2,
        timeout: float = 20.0,
        user_agent: str = "Mozilla/5.0 (eh-py-wrapper/0.1)",
    ):
        self.host = EX_HOST if exhentai else EH_HOST
        self.timeout = timeout
        self.rate_limit = rate_limit
        self._last_call = 0.0
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        if cookies:
            self.session.cookies.update(cookies)

    # ------------------------------------------------------------------ core

    def _throttle(self) -> None:
        if self.rate_limit <= 0:
            return
        wait = self.rate_limit - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    def _api(self, payload: dict) -> dict:
        self._throttle()
        url = f"https://api.{EH_HOST}{API_PATH}"  # api.e-hentai.org works for both
        r = self.session.post(url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            raise EHentaiError(data["error"])
        return data

    # ----------------------------------------------------------------- gdata

    def gdata(
        self,
        gidlist: Iterable[tuple[int, str]],
        *,
        namespace: int = 1,
    ) -> list[Gallery]:
        """Batch fetch metadata for up to 25 galleries per call."""
        pairs = [[int(gid), str(token)] for gid, token in gidlist]
        if not pairs:
            return []
        if len(pairs) > 25:
            raise ValueError("gdata accepts at most 25 entries per call")

        data = self._api(
            {"method": "gdata", "gidlist": pairs, "namespace": namespace}
        )
        return [_gallery_from_gmetadata(g) for g in data.get("gmetadata", [])]

    def gdata_batched(
        self, gidlist: Iterable[tuple[int, str]], *, namespace: int = 1
    ) -> list[Gallery]:
        """gdata with automatic 25-per-call batching."""
        out: list[Gallery] = []
        batch: list[tuple[int, str]] = []
        for gid, token in gidlist:
            batch.append((gid, token))
            if len(batch) == 25:
                out.extend(self.gdata(batch, namespace=namespace))
                batch = []
        if batch:
            out.extend(self.gdata(batch, namespace=namespace))
        return out

    # ---------------------------------------------------------------- gtoken

    def gtoken(
        self, pagelist: Iterable[tuple[int, str, int]]
    ) -> list[tuple[int, str]]:
        """Convert [(gid, page_token, page_num), ...] -> [(gid, gallery_token), ...]."""
        pages = [[int(g), str(t), int(n)] for g, t, n in pagelist]
        if not pages:
            return []
        data = self._api({"method": "gtoken", "pagelist": pages})
        return [(int(e["gid"]), e["token"]) for e in data.get("tokenlist", [])]

    # ------------------------------------------------------------ URL helper

    @staticmethod
    def parse_gallery_url(url: str) -> tuple[int, str]:
        m = GALLERY_RE.search(url)
        if not m:
            raise ValueError(f"not a gallery URL: {url!r}")
        return int(m.group(1)), m.group(2)

    # ---------------------------------------------------------------- search

    def search(
        self,
        query: str = "",
        *,
        page: int = 0,
        categories: int | None = None,
        advanced: dict[str, str] | None = None,
    ) -> list[tuple[int, str]]:
        """
        Scrape the listing page and return a list of (gid, token) for each hit.

        Call `gdata(...)` on the returned pairs to get full metadata.

        `categories` is a bitmask; see https://ehwiki.org/wiki/Gallery_Searching.
        `advanced` lets you pass any extra f_* params (e.g. {"f_srdd": "4"} for min 4-star rating).
        """
        self._throttle()
        params: dict[str, str | int] = {"f_search": query, "page": page}
        if categories is not None:
            params["f_cats"] = categories
        if advanced:
            params.update(advanced)

        url = f"https://{self.host}/"
        r = self.session.get(url, params=params, timeout=self.timeout)
        r.raise_for_status()

        # Uniquify while preserving order
        seen: set[tuple[int, str]] = set()
        out: list[tuple[int, str]] = []
        for gid_str, tok in GALLERY_RE.findall(r.text):
            pair = (int(gid_str), tok)
            if pair in seen:
                continue
            seen.add(pair)
            out.append(pair)
        return out

    def search_detailed(self, query: str = "", **kwargs) -> list[Gallery]:
        """Search + resolve to full metadata via gdata (auto-batched)."""
        pairs = self.search(query, **kwargs)
        return self.gdata_batched(pairs)

    # ----------------------------------------------------- HTML page scrape

    def get_gallery_page(self, gid: int, token: str) -> dict[str, Any]:
        """
        Fetch the gallery HTML page and extract:
          * parent: (gid, token) or None
          * newer_versions: [(gid, token), ...]
          * all_linked: every other gallery link found on the page

        Rate limiting applies.
        """
        self._throttle()
        url = f"https://{self.host}/g/{gid}/{token}/"
        r = self.session.get(url, timeout=self.timeout)
        r.raise_for_status()
        return _parse_gallery_html(r.text, self_pair=(gid, token))

    # ---------------------------------------------------- translation lookup

    def walk_parent_chain(
        self, gid: int, token: str, *, max_hops: int = 10
    ) -> list[tuple[int, str]]:
        """
        Walk `Parent:` links upward until there is no parent.

        Returns [start, parent, grandparent, ...]; the last entry is the root
        upload (typically the raw-language original).
        """
        chain: list[tuple[int, str]] = []
        cur: tuple[int, str] | None = (gid, token)
        seen: set[tuple[int, str]] = set()
        for _ in range(max_hops):
            if cur is None or cur in seen:
                break
            seen.add(cur)
            chain.append(cur)
            info = self.get_gallery_page(*cur)
            cur = info.get("parent")
        return chain

    def find_translations(
        self, gid: int, token: str
    ) -> dict[str, list[Gallery]]:
        """
        High-level: for a given gallery, walk up to the root and then collect
        all "newer versions" of the root. Returns galleries grouped by language
        tag (empty string key for untagged).
        """
        chain = self.walk_parent_chain(gid, token)
        if not chain:
            return {}
        root_gid, root_tok = chain[-1]
        root_info = self.get_gallery_page(root_gid, root_tok)

        pairs: list[tuple[int, str]] = [(root_gid, root_tok)]
        for p in root_info.get("newer_versions", []):
            if p not in pairs:
                pairs.append(p)
        # Include any in-between galleries we already visited.
        for p in chain:
            if p not in pairs:
                pairs.append(p)

        galleries = self.gdata_batched(pairs)
        by_lang: dict[str, list[Gallery]] = {}
        for g in galleries:
            langs = g.languages or [""]
            for lang in langs:
                by_lang.setdefault(lang, []).append(g)
        return by_lang


# ---------------------------------------------------------------- helpers

def _gallery_from_gmetadata(m: dict) -> Gallery:
    return Gallery(
        gid=int(m["gid"]),
        token=m["token"],
        title=html.unescape(m.get("title") or ""),
        title_jpn=html.unescape(m.get("title_jpn") or ""),
        category=m.get("category", ""),
        uploader=m.get("uploader", ""),
        posted=int(m.get("posted", 0)),
        filecount=int(m.get("filecount", 0)),
        filesize=int(m.get("filesize", 0)),
        rating=float(m.get("rating", 0) or 0),
        tags=list(m.get("tags", []) or []),
        raw=m,
    )


_PARENT_ROW_RE = re.compile(
    r'Parent:</td>\s*<td class="gdt2">(.*?)</td>', re.S
)
_GND_BLOCK_RE = re.compile(
    r'<div id="gnd">(.*?)</div>', re.S
)


def _parse_gallery_html(
    page: str, *, self_pair: tuple[int, str]
) -> dict[str, Any]:
    """Pull parent link + newer-versions list out of a gallery page."""
    parent: tuple[int, str] | None = None
    m = _PARENT_ROW_RE.search(page)
    if m:
        pm = GALLERY_RE.search(m.group(1))
        if pm:
            parent = (int(pm.group(1)), pm.group(2))

    newer: list[tuple[int, str]] = []
    g = _GND_BLOCK_RE.search(page)
    if g:
        seen: set[tuple[int, str]] = {self_pair}
        for gid_str, tok in GALLERY_RE.findall(g.group(1)):
            pair = (int(gid_str), tok)
            if pair in seen:
                continue
            seen.add(pair)
            newer.append(pair)

    # Every gallery link referenced anywhere on the page (minus self).
    all_linked: list[tuple[int, str]] = []
    seen_all: set[tuple[int, str]] = {self_pair}
    for gid_str, tok in GALLERY_RE.findall(page):
        pair = (int(gid_str), tok)
        if pair in seen_all:
            continue
        seen_all.add(pair)
        all_linked.append(pair)

    return {"parent": parent, "newer_versions": newer, "all_linked": all_linked}
