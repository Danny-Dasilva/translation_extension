"""Benchmark v10-it served by vLLM with the official Gemma 4 MTP drafter.

Mirrors `bench_llamacpp_v9c.py` — same 25 manga JP->EN prompts, same single-batch
/ single-concurrency cadence — so the resulting ``decode_tps_median`` is directly
comparable to the v9c baseline (77.8 tok/s @ Q8_0+FA, llama.cpp).

The path under test is **Path A**: vLLM serving the merged v10-it weights
plus Google's official assistant model `google/gemma-4-E4B-it-assistant`
running as the MTP drafter (vLLM PR #41745, merged 2026-05-06).

------------------------------------------------------------------------------
EVAL-SCRIPT FIX (chat-template) — see also `diag_v10it_chat_template.py`
------------------------------------------------------------------------------
The original bench used /v1/completions with a raw "Japanese: ...\\nEnglish:"
template. That format is OOD for v10-it (Gemma-4-E4B-IT + LoRA) which expects
chat-template framing. Diagnosis: 20/20 clean English with chat template,
~6/20 with raw. Switching to /v1/chat/completions makes vLLM apply the
served model's chat template server-side, so the model sees the same framing
it was trained with. This also tends to lengthen outputs (proper completion
+ <end_of_turn>), giving MTP more headroom to amortize draft cost.

The previous broken bench reported 35.3% acceptance — expect higher here.
------------------------------------------------------------------------------

What this script does NOT do
----------------------------
* Spawn the server. Start it externally via ``serve_v10it_vllm.sh`` and pass
  ``--base-url`` (default ``http://127.0.0.1:8000/v1``).

Usage
-----

    # Terminal A: bash backend/scripts/eval/serve_v10it_vllm.sh
    # Terminal B:
    uv run python backend/scripts/eval/bench_vllm_v10it.py \\
        --out-dir backend/scripts/eval/bench_out/v10it_vllm_mtp \\
        --base-url http://127.0.0.1:8000/v1 \\
        --model v10it \\
        --prompts backend/scripts/eval/bench_prompts.jsonl

Outputs (under --out-dir):
    outputs.jsonl  — per-prompt decoded text + timing
    summary.json   — aggregate medians/p95 + speedup vs v9c baseline
    config.json    — argparse snapshot
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests
from loguru import logger


# Chat user message — matches translate_manga_unsloth.py and
# diag_v10it_chat_template.py character-for-character. NO trailing
# "\nEnglish:" — the chat template provides the assistant turn marker.
USER_MSG_TEMPLATE = (
    "Translate the following Japanese to English. Output only the translation.\n\n"
    "Japanese: {jp}"
)

# Production per-bubble prompt (VLLMOpenAITranslationService.translate_single).
# A drafter trained on this format MUST be benched with it, or spec-decode
# acceptance collapses on the prompt-distribution mismatch alone.
PRODUCTION_MSG_TEMPLATE = (
    "Translate the following segment into English, "
    "without additional explanation.\n\n{jp}"
)

_TEMPLATES = {"eval": USER_MSG_TEMPLATE, "production": PRODUCTION_MSG_TEMPLATE}
_ACTIVE_TEMPLATE = USER_MSG_TEMPLATE  # overridden by --prompt-style in main()

# v9c llama.cpp baseline (Q8_0 + flash-attn, vanilla decode, 25 prompts, single concurrency).
# Source: backend/scripts/eval/bench_out/q8_0-fa-vanilla/summary.json
V9C_BASELINE_TPS_MEDIAN = 77.81427720617002


@dataclass
class RequestResult:
    idx: int
    jp: str
    en_pred: str
    n_prompt_tokens: int
    n_decoded_tokens: int
    wall_seconds: float
    decode_tps: float                          # n_decoded_tokens / wall_seconds
    server_total_tokens: int | None            # usage.total_tokens (vLLM OpenAI shim)
    server_time_to_first_token_s: float | None # if streaming used; else None


def build_messages(jp: str) -> list[dict[str, str]]:
    """Construct the OpenAI chat messages list for a single JP prompt.

    Exposed at module level so the smoke test (and any future debug harness)
    can import + inspect without spinning up a server.
    """
    return [{"role": "user", "content": _ACTIVE_TEMPLATE.format(jp=jp)}]


def _load_prompts(path: Path, limit: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append({"jp": row["jp"], "en_ref": row.get("en", row.get("en_ref", ""))})
            if limit and len(rows) >= limit:
                break
    return rows


def _wait_ready(base_url: str, timeout: float = 600.0) -> None:
    """Poll the OpenAI /v1/models endpoint until vLLM is up."""
    deadline = time.time() + timeout
    last: Any = None
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url.rstrip('/')}/models", timeout=2.0)
            if r.status_code == 200:
                ids = [m.get("id") for m in r.json().get("data", [])]
                logger.info("vLLM ready, models: {}", ids)
                return
        except Exception as e:  # noqa: BLE001
            last = e
        time.sleep(1.0)
    raise RuntimeError(f"vLLM did not become ready at {base_url}: {last}")


def _chat_completion(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    api_key: str,
    timeout: float = 600.0,
) -> tuple[dict[str, Any], float]:
    """Issue a /v1/chat/completions request. Returns (response_json, wall_seconds).

    vLLM applies the served model's chat template server-side so the request
    body just needs the role/content messages — exactly what /v1/completions
    could not do (it bypasses the chat template entirely).
    """
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.9 if temperature > 0 else 1.0,
        "stream": False,
        "n": 1,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    t0 = time.perf_counter()
    r = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        json=body, headers=headers, timeout=timeout,
    )
    dt = time.perf_counter() - t0
    r.raise_for_status()
    return r.json(), dt


def _fetch_metrics(base_url: str) -> str | None:
    """Return raw Prometheus text from /metrics, or None if not exposed.

    vLLM exposes spec-decode acceptance counters at ``/metrics`` as
    ``vllm:spec_decode_*``. We snapshot before/after to compute MTP
    acceptance rate when available.
    """
    if "/v1" in base_url:
        root = base_url.split("/v1", 1)[0]
    else:
        root = base_url.rstrip("/")
    try:
        r = requests.get(f"{root}/metrics", timeout=5.0)
        if r.status_code == 200:
            return r.text
    except Exception as e:  # noqa: BLE001
        logger.warning("metrics fetch failed: {}", e)
    return None


def _parse_spec_metrics(text: str) -> dict[str, float]:
    """Pull out vllm:spec_decode_* gauges/counters from /metrics text."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        if "vllm:spec_decode" not in line:
            continue
        try:
            name_and_labels, value = line.rsplit(" ", 1)
        except ValueError:
            continue
        try:
            out[name_and_labels] = float(value)
        except ValueError:
            continue
    return out


def _diff_spec_metrics(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {k: after[k] - before.get(k, 0.0) for k in after if k.startswith("vllm:")}


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    return statistics.quantiles(values, n=100, method="inclusive")[max(0, min(99, int(pct) - 1))]


def _summarize(results: list[RequestResult]) -> dict[str, Any]:
    decoded = [r.n_decoded_tokens for r in results]
    walls = [r.wall_seconds for r in results]
    tps = [r.decode_tps for r in results if r.n_decoded_tokens > 0]
    median_tps = statistics.median(tps) if tps else 0.0
    return {
        "n_requests": len(results),
        "total_wall_s": sum(walls),
        "tokens_decoded_total": sum(decoded),
        "tokens_decoded_per_request_median": statistics.median(decoded) if decoded else 0,
        "decode_tps_median": median_tps,
        "decode_tps_mean": statistics.mean(tps) if tps else 0.0,
        "decode_tps_p05": _percentile(tps, 5) if tps else 0.0,
        "decode_tps_p95": _percentile(tps, 95) if tps else 0.0,
        "wall_p50_s": statistics.median(walls) if walls else 0.0,
        "wall_p95_s": _percentile(walls, 95) if walls else 0.0,
        "v9c_baseline_tps_median": V9C_BASELINE_TPS_MEDIAN,
        "speedup_vs_v9c_llamacpp": (median_tps / V9C_BASELINE_TPS_MEDIAN) if median_tps else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1",
                    help="OpenAI-compatible base URL (default: http://127.0.0.1:8000/v1)")
    ap.add_argument("--model", default="v10it",
                    help="served model name as registered with --served-model-name (default: v10it)")
    ap.add_argument("--api-key", default="",
                    help="API key if vLLM was started with one (default: empty / disabled)")
    ap.add_argument("--prompts", default=str(Path(__file__).parent / "bench_prompts.jsonl"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=25,
                    help="cap on prompts (matches v9c bench default)")
    ap.add_argument("--max-tokens", type=int, default=64,
                    help="max output tokens (matches v9c bench default)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=2,
                    help="warmup requests not counted (matches v9c bench default)")
    ap.add_argument("--ready-timeout", type=float, default=600.0,
                    help="how long to wait for /v1/models to come up (sec)")
    ap.add_argument("--label", default="v10it_vllm_mtp_chat",
                    help="label written into summary.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print first request body and exit. No server needed.")
    ap.add_argument("--prompt-style", choices=sorted(_TEMPLATES), default="eval",
                    help="Prompt template. Use 'production' to match a drafter "
                         "trained on the per-bubble production prompt — otherwise "
                         "spec-decode acceptance is measured on a mismatched "
                         "distribution.")
    args = ap.parse_args()

    global _ACTIVE_TEMPLATE
    _ACTIVE_TEMPLATE = _TEMPLATES[args.prompt_style]

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "outputs.jsonl"
    summary_path = out_dir / "summary.json"
    config_path = out_dir / "config.json"
    metrics_before_path = out_dir / "metrics_before.prom"
    metrics_after_path = out_dir / "metrics_after.prom"

    prompts = _load_prompts(Path(args.prompts), args.limit)
    if not prompts:
        raise RuntimeError(f"no prompts loaded from {args.prompts}")
    logger.info("loaded {} prompts (warmup={})", len(prompts), args.warmup)

    if args.dry_run:
        msgs = build_messages(prompts[0]["jp"])
        body = {
            "model": args.model,
            "messages": msgs,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": 0.9 if args.temperature > 0 else 1.0,
            "stream": False,
            "n": 1,
        }
        print("=== dry-run: first request body ===")
        print(json.dumps(body, ensure_ascii=False, indent=2))
        print("=== endpoint: {}/chat/completions ===".format(args.base_url.rstrip("/")))
        return 0

    logger.info("waiting for vLLM at {} (timeout {}s)...", args.base_url, args.ready_timeout)
    _wait_ready(args.base_url, timeout=args.ready_timeout)

    # Warmup
    for i in range(args.warmup):
        try:
            _chat_completion(
                args.base_url, args.model,
                build_messages(prompts[i % len(prompts)]["jp"]),
                max_tokens=args.max_tokens, temperature=args.temperature,
                api_key=args.api_key,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("warmup {} failed: {}", i, e)
    logger.info("warmup done")

    metrics_before_text = _fetch_metrics(args.base_url)
    if metrics_before_text:
        metrics_before_path.write_text(metrics_before_text)
    metrics_before = _parse_spec_metrics(metrics_before_text or "")

    results: list[RequestResult] = []
    with open(raw_path, "w", encoding="utf-8") as raw_f:
        for i, row in enumerate(prompts):
            msgs = build_messages(row["jp"])
            resp, wall = _chat_completion(
                args.base_url, args.model, msgs,
                max_tokens=args.max_tokens, temperature=args.temperature,
                api_key=args.api_key,
            )
            choice = (resp.get("choices") or [{}])[0]
            # Chat-completions returns {"message": {"role": "assistant", "content": "..."}}
            msg = choice.get("message") or {}
            text = msg.get("content", "") or ""
            usage = resp.get("usage") or {}
            n_dec = int(usage.get("completion_tokens", 0) or 0)
            n_p = int(usage.get("prompt_tokens", 0) or 0)
            tot = usage.get("total_tokens")
            decode_tps = (n_dec / wall) if wall > 0 and n_dec else 0.0
            rr = RequestResult(
                idx=i, jp=row["jp"], en_pred=text.strip(),
                n_prompt_tokens=n_p, n_decoded_tokens=n_dec,
                wall_seconds=wall, decode_tps=decode_tps,
                server_total_tokens=int(tot) if tot is not None else None,
                server_time_to_first_token_s=None,
            )
            results.append(rr)
            raw_f.write(json.dumps({**asdict(rr), "en_ref": row.get("en_ref", "")},
                                   ensure_ascii=False) + "\n")
            logger.info("[{}/{}] {} tok in {:.2f}s = {:.1f} tps | {}",
                        i + 1, len(prompts), n_dec, wall, decode_tps,
                        text[:50].replace("\n", " "))

    metrics_after_text = _fetch_metrics(args.base_url)
    if metrics_after_text:
        metrics_after_path.write_text(metrics_after_text)
    metrics_after = _parse_spec_metrics(metrics_after_text or "")
    spec_delta = _diff_spec_metrics(metrics_before, metrics_after)

    summary = _summarize(results)
    summary["label"] = args.label
    summary["base_url"] = args.base_url
    summary["model"] = args.model
    summary["spec_decode_metrics_delta"] = spec_delta
    accept_rate = None
    for k, v in spec_delta.items():
        if "acceptance_rate" in k or "draft_acceptance" in k:
            accept_rate = v
            break
    if accept_rate is not None:
        summary["mtp_acceptance_rate"] = accept_rate

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)

    logger.info("=== {} ===", args.label)
    logger.info("median decode tps: {:.1f}  ({:.2f}x v9c llama.cpp baseline of {:.1f})",
                summary["decode_tps_median"], summary["speedup_vs_v9c_llamacpp"],
                V9C_BASELINE_TPS_MEDIAN)
    logger.info("p95 wall:          {:.2f}s", summary["wall_p95_s"])
    logger.info("total wall:        {:.1f}s", summary["total_wall_s"])
    if spec_delta:
        logger.info("spec-decode metrics delta keys: {}",
                    list(spec_delta.keys())[:5])
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
