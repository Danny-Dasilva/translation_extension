#!/usr/bin/env python3
"""Image-prefix latency / prefix-reuse bench (gate signal #6) — DIRECT vLLM.

Measures the serving shape v1 was trained for: N *marked-line* chat calls per
manga page that share a BYTE-IDENTICAL prefix

    [optional image][V11_PAGE_INSTR + "\n\nPage:\n1. ...\nN. ..." + "\n\nTranslate line "]

and differ only in the trailing "{k}: {target}". Under vLLM automatic prefix
caching (APC) calls 2..N should re-use the KV of the shared prefix (image +
instruction + numbered page), so only call 1 pays the image prefill.

This harness talks to vLLM's OpenAI-compatible API DIRECTLY (no app pipeline
import). It:
  1. Builds the N marked prompts per page (identical prefix, varying k).
  2. Times call 1 (cold / image prefill) separately from calls 2..N (warm).
     Modes: --image on|off, --order sequential|concurrent, --model, --pages,
     --repeats.
  3. Scrapes /metrics before/after each page and reports prefix-cache
     query/hit token deltas plus a computed `prefix_reuse_confirmed` boolean
     (hit-token delta must cover the expected shared-prefix tokens across the
     re-using calls — not merely be > 0).
  4. CONCURRENCY CORRECTNESS: in --order concurrent it first runs a
     deterministic sequential reference pass, then the concurrent pass, and
     (a) scans every output for the multimodal-APC corruption failure mode
         (a char repeated >6x, a short token looped, or non-English garbage)
     (b) diffs concurrent outputs against the sequential reference for the same
         inputs and reports mismatches. This probes vLLM issue #20261 on our
         real workload.
  5. Emits JSON (bench_e2e.py conventions: mean_ms/median_ms/p95_ms/n) with
     per-call latency arrays (call-1 vs calls-2..N separated) and the cache
     metrics, to stdout and optionally --out.

The V11_PAGE_INSTR text + prompt layout below are a BYTE-FOR-BYTE mirror of
backend/app/services/vllm_openai_translation_service.py::build_v11_context_prompt
(V11_PAGE_INSTR at line 124, builder at line 282). Kept inline so the harness
has zero app/config/DB import surface. The trailing target-line normalization
the service applies only affects the *varying* suffix, never the shared prefix,
so it is irrelevant to the reuse measurement and deliberately omitted.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import re
import statistics
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import httpx

# --------------------------------------------------------------------------- #
# Config / defaults
# --------------------------------------------------------------------------- #
DEFAULT_BASE_URL = "http://100.64.235.63:8001/v1"
DEFAULT_MODEL = "v1"
DEFAULT_TESTSET = str(
    Path(__file__).resolve().parents[2] / ".bench" / "pov_ab" / "testset_large.json"
)

# Byte-identical mirror of vllm_openai_translation_service.V11_PAGE_INSTR.
V11_PAGE_INSTR = (
    "Translate the marked line of this manga page from Japanese to English. "
    "Use the page context for speakers, pronouns, and continuity. "
    "Output only the translation of the marked line."
)

# vLLM KV block size (default). Prefix caching is block-aligned, so the number
# of *hit* tokens is rounded down to a multiple of this.
VLLM_BLOCK_SIZE = 16

_MIME_BY_SUFFIX = {
    ".webp": "image/webp",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}

# Prometheus counter lines we care about (names discovered from a live scrape;
# matched by regex so a version rename is tolerated).
_METRIC_PATTERNS = {
    "prefix_cache_queries": re.compile(r"^vllm:prefix_cache_queries_total\b"),
    "prefix_cache_hits": re.compile(r"^vllm:prefix_cache_hits_total\b"),
    "mm_cache_queries": re.compile(r"^vllm:mm_cache_queries_total\b"),
    "mm_cache_hits": re.compile(r"^vllm:mm_cache_hits_total\b"),
}
_MODEL_LABEL_RE = re.compile(r'model_name="([^"]+)"')


# --------------------------------------------------------------------------- #
# Corruption / garbage detectors (multimodal-APC failure mode, issue #20261)
# --------------------------------------------------------------------------- #
# A single char repeated >6 times, e.g. "aaaaaaa" or "|||||||".
_RUNAWAY_CHAR_RE = re.compile(r"(.)\1{6,}")
# A short token (2-4 chars) looped many times, e.g. "lololololo", "の の の の".
_RUNAWAY_TOK_RE = re.compile(r"(.{2,4}?)\1{5,}")


def looks_garbled(text: str) -> tuple[bool, str]:
    """True + reason if an output shows the corruption failure mode.

    (a) a char repeated >6x, (b) a short token looped >=6x, or (c) mostly
    non-English garbage (the model is supposed to emit English; a high ratio
    of CJK / non-ASCII letters in a non-trivial output is a corruption tell).
    """
    s = (text or "").strip()
    if not s:
        return False, ""
    if _RUNAWAY_CHAR_RE.search(s):
        return True, "char_repeat>6"
    m = _RUNAWAY_TOK_RE.search(s)
    if m and (m.end() - m.start()) >= 12:
        return True, "token_loop"
    letters = [c for c in s if c.isalpha()]
    if len(letters) >= 8:
        non_ascii = sum(1 for c in letters if ord(c) > 0x24F)  # beyond Latin-Ext-A
        if non_ascii / len(letters) > 0.30:
            return True, "non_english_garbage"
    return False, ""


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #
def shared_prefix(lines: list[str]) -> str:
    """The BYTE-IDENTICAL prefix every marked-line call on a page shares.

    Everything up to and including "Translate line " — k and the target line
    are the only bytes that vary between the N calls.
    """
    numbered = "\n".join(f"{i + 1}. {ln}" for i, ln in enumerate(lines))
    return f"{V11_PAGE_INSTR}\n\nPage:\n{numbered}\n\nTranslate line "


def marked_prompt(lines: list[str], k_idx: int) -> str:
    """Full marked-line user text for 0-based target k_idx (k is 1-based)."""
    return f"{shared_prefix(lines)}{k_idx + 1}: {lines[k_idx]}"


def image_to_data_url(image_path: Path) -> str:
    raw = image_path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    mime = _MIME_BY_SUFFIX.get(image_path.suffix.lower(), "image/webp")
    return f"data:{mime};base64,{b64}"


def build_messages(prompt: str, image_data_url: str | None) -> list[dict[str, Any]]:
    """Chat messages. Image FIRST so image tokens sit in the shared prefix."""
    if image_data_url is None:
        return [{"role": "user", "content": prompt}]
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data_url}},
                {"type": "text", "text": prompt},
            ],
        }
    ]


# --------------------------------------------------------------------------- #
# Metrics scraping
# --------------------------------------------------------------------------- #
def metrics_url(base_url: str) -> str:
    # base_url is ".../v1"; metrics live at the server root "/metrics".
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    return f"{root}/metrics"


async def scrape_metrics(client: httpx.AsyncClient, url: str, model: str) -> dict[str, float]:
    """Return {logical_name: value} for our counters, filtered to `model`.

    The v1 LoRA is served under model_name="v1"; if a line lacks the exact
    model label we still fall back to summing all series so the delta is never
    silently zero on a naming mismatch.
    """
    resp = await client.get(url, timeout=30.0)
    resp.raise_for_status()
    out: dict[str, float] = {name: 0.0 for name in _METRIC_PATTERNS}
    matched_model: dict[str, bool] = {name: False for name in _METRIC_PATTERNS}
    for line in resp.text.splitlines():
        if line.startswith("#") or not line:
            continue
        for name, pat in _METRIC_PATTERNS.items():
            if not pat.match(line):
                continue
            try:
                value = float(line.rsplit(" ", 1)[1])
            except (ValueError, IndexError):
                continue
            m = _MODEL_LABEL_RE.search(line)
            this_model = m.group(1) if m else None
            if this_model == model:
                if not matched_model[name]:
                    out[name] = 0.0
                    matched_model[name] = True
                out[name] += value
            elif not matched_model[name]:
                out[name] += value  # fallback until an exact-model series is seen
    return out


def metric_deltas(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {k: after.get(k, 0.0) - before.get(k, 0.0) for k in before}


# --------------------------------------------------------------------------- #
# Single call
# --------------------------------------------------------------------------- #
async def call_one(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,  # deterministic: required for the concurrent-vs-seq diff
        "max_tokens": max_tokens,
    }
    t0 = time.perf_counter()
    resp = await client.post(
        f"{base_url.rstrip('/')}/chat/completions", json=payload, timeout=300.0
    )
    ms = (time.perf_counter() - t0) * 1000.0
    resp.raise_for_status()
    obj = resp.json()
    return {
        "ms": ms,
        "text": obj["choices"][0]["message"]["content"],
        "prompt_tokens": obj.get("usage", {}).get("prompt_tokens"),
        "completion_tokens": obj.get("usage", {}).get("completion_tokens"),
    }


# --------------------------------------------------------------------------- #
# Per-page runs
# --------------------------------------------------------------------------- #
async def run_page_sequential(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    lines: list[str],
    image_data_url: str | None,
    max_tokens: int,
) -> list[dict[str, Any]]:
    """N calls in strict order: call 1 (cold prefill) then 2..N (warm reuse)."""
    results = []
    for k in range(len(lines)):
        msgs = build_messages(marked_prompt(lines, k), image_data_url)
        r = await call_one(client, base_url, model, msgs, max_tokens)
        r["k"] = k
        results.append(r)
    return results


async def run_page_concurrent(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    lines: list[str],
    image_data_url: str | None,
    max_tokens: int,
) -> list[dict[str, Any]]:
    """All N calls fired together via asyncio.gather (contention + APC race)."""

    async def one(k: int) -> dict[str, Any]:
        msgs = build_messages(marked_prompt(lines, k), image_data_url)
        r = await call_one(client, base_url, model, msgs, max_tokens)
        r["k"] = k
        return r

    return await asyncio.gather(*(one(k) for k in range(len(lines))))


# --------------------------------------------------------------------------- #
# Reuse confirmation
# --------------------------------------------------------------------------- #
def compute_prefix_reuse(
    calls: list[dict[str, Any]],
    hits_delta: float,
    reusing_calls: int,
    confirm_fraction: float = 0.85,
) -> dict[str, Any]:
    """Decide `prefix_reuse_confirmed` from the hit-token delta.

    The shared prefix is P tokens; each of the `reusing_calls` should hit ~P
    cached tokens (block-aligned). We estimate P conservatively from the
    per-call prompt_tokens usage: prefix ≈ min(prompt_tokens) minus a suffix
    slack (the "k: target" tail, generously 64 tokens), block-aligned down.
    Confirmed iff hits_delta covers `confirm_fraction` of reusing_calls * P.
    """
    prompt_tokens = [c["prompt_tokens"] for c in calls if c.get("prompt_tokens")]
    if not prompt_tokens or reusing_calls <= 0:
        return {
            "prefix_reuse_confirmed": False,
            "prefix_tokens_est": None,
            "expected_min_hit_tokens": None,
            "hits_delta": hits_delta,
            "reason": "no prompt_tokens usage or no reusing calls",
        }
    suffix_slack = 64
    prefix_est = max(0, min(prompt_tokens) - suffix_slack)
    prefix_est = (prefix_est // VLLM_BLOCK_SIZE) * VLLM_BLOCK_SIZE  # block-aligned
    expected = reusing_calls * prefix_est * confirm_fraction
    confirmed = prefix_est > 0 and hits_delta >= expected
    return {
        "prefix_reuse_confirmed": bool(confirmed),
        "prefix_tokens_est": prefix_est,
        "reusing_calls": reusing_calls,
        "expected_min_hit_tokens": round(expected, 1),
        "hits_delta": hits_delta,
        "confirm_fraction": confirm_fraction,
    }


def latency_stats(samples: list[float]) -> dict[str, Any]:
    if not samples:
        return {"n": 0, "mean_ms": None, "median_ms": None, "p95_ms": None,
                "min_ms": None, "max_ms": None}
    s = sorted(samples)
    return {
        "n": len(s),
        "mean_ms": round(statistics.mean(s), 1),
        "median_ms": round(statistics.median(s), 1),
        "p95_ms": round(s[min(len(s) - 1, int(len(s) * 0.95))], 1),
        "min_ms": round(s[0], 1),
        "max_ms": round(s[-1], 1),
    }


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_pages(testset: str, limit: int) -> list[dict[str, Any]]:
    rows = json.load(open(testset))
    pages: "OrderedDict[str, dict]" = OrderedDict()
    for r in rows:
        pages.setdefault(r["page_img"], r)  # first row per page
    out = []
    for page_img, r in list(pages.items())[:limit]:
        out.append({"page_img": page_img, "lines": list(r["context"])})
    if not out:
        raise SystemExit(f"no pages loaded from {testset}")
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--testset", default=DEFAULT_TESTSET)
    ap.add_argument("--pages", type=int, default=1, help="number of pages to bench")
    ap.add_argument("--repeats", type=int, default=1, help="repeat each page N times")
    ap.add_argument("--image", choices=["on", "off"], default="on")
    ap.add_argument("--order", choices=["sequential", "concurrent"], default="sequential")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--label", default="image_prefix_bench")
    ap.add_argument("--out", default=None, help="also write JSON here")
    ap.add_argument("--no-ref-check", action="store_true",
                    help="skip the concurrent-vs-sequential correctness diff")
    args = ap.parse_args()

    pages = load_pages(args.testset, args.pages)
    murl = metrics_url(args.base_url)
    print(f"[bench] {len(pages)} page(s) x{args.repeats} repeats | image={args.image} "
          f"| order={args.order} | model={args.model} | metrics={murl}",
          file=sys.stderr)

    # Pre-load image data URLs once (byte-identical across the page's N calls).
    image_urls: dict[str, str | None] = {}
    for p in pages:
        if args.image == "on":
            path = Path(p["page_img"])
            if not path.exists():
                raise SystemExit(f"image not found (NAS mounted?): {path}")
            image_urls[p["page_img"]] = image_to_data_url(path)
        else:
            image_urls[p["page_img"]] = None

    call1_ms: list[float] = []
    rest_ms: list[float] = []
    per_page_report: list[dict[str, Any]] = []
    reuse_flags: list[bool] = []
    garbled_hits: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []

    async with httpx.AsyncClient() as client:
        for rep in range(args.repeats):
            for pi, page in enumerate(pages):
                lines = page["lines"]
                img = image_urls[page["page_img"]]
                n = len(lines)

                before = await scrape_metrics(client, murl, args.model)

                ref_calls = None
                if args.order == "concurrent":
                    # Deterministic reference for the correctness diff (its
                    # latency is NOT recorded — this is a correctness pass).
                    if not args.no_ref_check:
                        ref_calls = await run_page_sequential(
                            client, args.base_url, args.model, lines, img, args.max_tokens
                        )
                    # Re-baseline metrics AFTER the reference pass so the
                    # reported deltas describe only the concurrent pass.
                    before = await scrape_metrics(client, murl, args.model)
                    calls = await run_page_concurrent(
                        client, args.base_url, args.model, lines, img, args.max_tokens
                    )
                else:
                    calls = await run_page_sequential(
                        client, args.base_url, args.model, lines, img, args.max_tokens
                    )

                after = await scrape_metrics(client, murl, args.model)
                deltas = metric_deltas(before, after)

                calls.sort(key=lambda c: c["k"])
                call1_ms.append(calls[0]["ms"])
                rest_ms.extend(c["ms"] for c in calls[1:])

                # For sequential, calls 2..N re-use. For concurrent, they race;
                # count all-but-one as potential re-users (best case).
                reusing = n - 1
                reuse = compute_prefix_reuse(
                    calls, deltas.get("prefix_cache_hits", 0.0), reusing
                )
                reuse_flags.append(reuse["prefix_reuse_confirmed"])

                # Corruption scan on every output.
                for c in calls:
                    bad, why = looks_garbled(c["text"])
                    if bad:
                        garbled_hits.append({
                            "page": page["page_img"], "k": c["k"],
                            "reason": why, "text": c["text"][:200],
                        })

                # Concurrent-vs-sequential output diff.
                if ref_calls is not None:
                    ref_by_k = {c["k"]: c["text"].strip() for c in ref_calls}
                    for c in calls:
                        cur = c["text"].strip()
                        ref = ref_by_k.get(c["k"])
                        if ref is not None and cur != ref:
                            mismatches.append({
                                "page": page["page_img"], "k": c["k"],
                                "sequential": ref[:200], "concurrent": cur[:200],
                            })

                per_page_report.append({
                    "repeat": rep,
                    "page_index": pi,
                    "page_img": page["page_img"],
                    "n_calls": n,
                    "call1_ms": round(calls[0]["ms"], 1),
                    "calls_2toN_ms": [round(c["ms"], 1) for c in calls[1:]],
                    "prompt_tokens_call1": calls[0].get("prompt_tokens"),
                    "cache_deltas": {k: round(v, 1) for k, v in deltas.items()},
                    "reuse": reuse,
                })
                print(f"[bench] rep{rep} page{pi} n={n} "
                      f"call1={calls[0]['ms']:.0f}ms "
                      f"warm_med={statistics.median([c['ms'] for c in calls[1:]]):.0f}ms "
                      f"hits_delta={deltas.get('prefix_cache_hits', 0):.0f} "
                      f"reuse={reuse['prefix_reuse_confirmed']}", file=sys.stderr)

    all_ms = call1_ms + rest_ms
    result = {
        "label": args.label,
        "config": {
            "base_url": args.base_url, "model": args.model,
            "pages": args.pages, "repeats": args.repeats,
            "image": args.image, "order": args.order,
            "max_tokens": args.max_tokens,
        },
        "metric_names": {
            "prefix_cache_queries": "vllm:prefix_cache_queries_total",
            "prefix_cache_hits": "vllm:prefix_cache_hits_total",
            "mm_cache_queries": "vllm:mm_cache_queries_total",
            "mm_cache_hits": "vllm:mm_cache_hits_total",
        },
        # bench_e2e.py conventions over ALL calls:
        "n": len(all_ms),
        "mean_ms": latency_stats(all_ms)["mean_ms"],
        "median_ms": latency_stats(all_ms)["median_ms"],
        "p95_ms": latency_stats(all_ms)["p95_ms"],
        # Call-1 (cold, image prefill) vs calls 2..N (warm, prefix reuse):
        "call1_stats": latency_stats(call1_ms),
        "calls_2toN_stats": latency_stats(rest_ms),
        "call1_ms": [round(x, 1) for x in call1_ms],
        "calls_2toN_ms": [round(x, 1) for x in rest_ms],
        "prefix_reuse_confirmed": bool(reuse_flags) and all(reuse_flags),
        "prefix_reuse_per_page": reuse_flags,
        "correctness": {
            "garbled_count": len(garbled_hits),
            "garbled": garbled_hits,
            "concurrent_vs_sequential_mismatches": len(mismatches),
            "mismatches": mismatches,
        },
        "per_page": per_page_report,
    }
    out_json = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(out_json)
        print(f"[bench] wrote {args.out}", file=sys.stderr)
    print(out_json)


if __name__ == "__main__":
    asyncio.run(main())
