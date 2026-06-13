"""Benchmark v9c (Gemma 4 E4B + LoRA) running under llama.cpp.

Spawns a local `llama-server`, sends manga-style JP -> EN translation prompts
in series (single batch, single concurrency), measures decode tok/s and end-
to-end latency for each request.

Supports:
  - vanilla decode  (no draft model)
  - speculative decode  (--draft-model passed through to llama-server)
  - flash-attn on/off  (-fa flag)
  - quant sweep (Q4_K_M, Q8_0, fp16)

Usage example::

    uv run python backend/scripts/eval/bench_llamacpp_v9c.py \
        --llama-bin /home/danny/llama.cpp/build/bin/llama-server \
        --model /home/danny/llama.cpp/models/gemma-4-E4B-pt.Q4_K_M.gguf \
        --lora  /path/to/v9c-adapter.gguf \
        --prompts backend/scripts/eval/bench_prompts.jsonl \
        --out-dir backend/scripts/eval/bench_out/v9c-q4km-fa \
        --quant Q4_K_M --ctx 2048 --port 8765 --flash-attn

The script writes:
  - raw outputs jsonl (per-prompt)
  - summary json (median/p50/p95 latency, median tok/s, total wall)
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests
from loguru import logger


# Default prompt template — same wording the v9c training data used.
PROMPT_TEMPLATE = (
    "Translate the following Japanese to English. "
    "Output only the translation.\n\nJapanese: {jp}\nEnglish:"
)


@dataclass
class RequestResult:
    idx: int
    jp: str
    en_pred: str
    n_prompt_tokens: int
    n_decoded_tokens: int
    wall_seconds: float
    decode_tps: float
    prompt_tps: float
    server_predicted_per_token_ms: float | None  # llama-server timings.predicted_per_token_ms


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_ready(port: int, timeout: float = 180.0) -> None:
    """Poll /health until ready or timeout."""
    deadline = time.time() + timeout
    last_err: Any = None
    while time.time() < deadline:
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=2.0)
            if r.status_code == 200:
                return
        except Exception as e:  # noqa: BLE001
            last_err = e
        time.sleep(0.5)
    raise RuntimeError(f"llama-server did not become ready on port {port}: {last_err}")


def _spawn_server(args: argparse.Namespace, port: int, log_path: Path) -> subprocess.Popen:
    cmd: list[str] = [
        args.llama_bin,
        "-m", str(args.model),
        "--port", str(port),
        "--host", "127.0.0.1",
        "-c", str(args.ctx),
        "-ngl", str(args.n_gpu_layers),
        "--no-mmap",
        "-t", str(args.threads),
    ]
    if args.lora:
        cmd += ["--lora", str(args.lora)]
    # llama.cpp: -fa now takes on|off|auto (default auto). Be explicit.
    cmd += ["-fa", "on" if args.flash_attn else "off"]
    if args.draft_model:
        cmd += [
            "--model-draft", str(args.draft_model),
            "--draft-max", str(args.draft_max),
            "--draft-min", str(args.draft_min),
        ]
        if args.draft_n_gpu_layers is not None:
            cmd += ["--n-gpu-layers-draft", str(args.draft_n_gpu_layers)]
    if args.spec_type and args.spec_type != "none":
        cmd += [
            "--spec-type", args.spec_type,
            "--draft-max", str(args.draft_max),
            "--draft-min", str(args.draft_min),
        ]
    if args.extra_server_args:
        cmd += args.extra_server_args.split()

    logger.info("starting llama-server: {}", " ".join(cmd))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "w", buffering=1)
    proc = subprocess.Popen(
        cmd,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        env={**os.environ, "LLAMA_NUMA": "isolate"},
    )
    return proc


def _completion(port: int, prompt: str, max_tokens: int, temperature: float) -> tuple[dict[str, Any], float]:
    """Issue a /completion request, return (response_json, wall_seconds)."""
    body = {
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": temperature,
        "top_p": 0.9,
        "min_p": 0.1,
        "stream": False,
        "cache_prompt": False,
    }
    t0 = time.perf_counter()
    r = requests.post(f"http://127.0.0.1:{port}/completion", json=body, timeout=600.0)
    dt = time.perf_counter() - t0
    r.raise_for_status()
    return r.json(), dt


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


def _summarize(results: list[RequestResult]) -> dict[str, Any]:
    decoded = [r.n_decoded_tokens for r in results]
    walls = [r.wall_seconds for r in results]
    tps = [r.decode_tps for r in results]
    return {
        "n_requests": len(results),
        "total_wall_s": sum(walls),
        "tokens_decoded_total": sum(decoded),
        "tokens_decoded_per_request_median": statistics.median(decoded) if decoded else 0,
        "decode_tps_median": statistics.median(tps) if tps else 0,
        "decode_tps_mean": statistics.mean(tps) if tps else 0,
        "decode_tps_p05": _pct(tps, 5) if tps else 0,
        "decode_tps_p95": _pct(tps, 95) if tps else 0,
        "wall_p50_s": statistics.median(walls) if walls else 0,
        "wall_p95_s": _pct(walls, 95) if walls else 0,
    }


def _pct(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    return statistics.quantiles(values, n=100, method="inclusive")[max(0, min(99, int(pct) - 1))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--llama-bin", default="/home/danny/llama.cpp/build/bin/llama-server")
    ap.add_argument("--model", required=True, help="path to base GGUF (e.g. gemma-4-E4B-pt.Q4_K_M.gguf)")
    ap.add_argument("--lora", default=None, help="path to v9c LoRA GGUF (optional, for base-only runs)")
    ap.add_argument("--prompts", default=str(Path(__file__).parent / "bench_prompts.jsonl"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--quant", default="unknown", help="label for the run (e.g. Q4_K_M)")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--n-gpu-layers", type=int, default=999, dest="n_gpu_layers")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--port", type=int, default=0, help="0 = pick free port")
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--flash-attn", action="store_true", help="pass -fa to llama-server")
    ap.add_argument("--draft-model", default=None, help="path to draft GGUF for spec decode")
    ap.add_argument("--draft-max", type=int, default=8)
    ap.add_argument("--draft-min", type=int, default=2)
    ap.add_argument("--draft-n-gpu-layers", type=int, default=None, dest="draft_n_gpu_layers")
    ap.add_argument("--spec-type", default="none",
                    choices=["none", "ngram-cache", "ngram-simple", "ngram-map-k", "ngram-map-k4v", "ngram-mod"],
                    help="server-side n-gram speculative decoding (no draft model required)")
    ap.add_argument("--warmup", type=int, default=2, help="number of warmup requests not counted")
    ap.add_argument("--extra-server-args", default="", help="extra args appended to llama-server")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "llama-server.log"
    raw_path = out_dir / "outputs.jsonl"
    summary_path = out_dir / "summary.json"
    config_path = out_dir / "config.json"

    port = args.port or _free_port()
    proc = _spawn_server(args, port, log_path)
    try:
        _wait_ready(port)
        prompts = _load_prompts(Path(args.prompts), args.limit)
        if not prompts:
            raise RuntimeError(f"no prompts loaded from {args.prompts}")
        logger.info("ready: {} prompts to send (warmup={})", len(prompts), args.warmup)

        # Warmup
        for i in range(args.warmup):
            _completion(port, PROMPT_TEMPLATE.format(jp=prompts[i % len(prompts)]["jp"]),
                        max_tokens=args.max_tokens, temperature=args.temperature)
        logger.info("warmup done, starting timed runs")

        results: list[RequestResult] = []
        with open(raw_path, "w", encoding="utf-8") as raw_f:
            for i, row in enumerate(prompts):
                prompt = PROMPT_TEMPLATE.format(jp=row["jp"])
                resp, wall = _completion(port, prompt, max_tokens=args.max_tokens, temperature=args.temperature)
                content = resp.get("content", "")
                timings = resp.get("timings", {}) or {}
                n_dec = int(timings.get("predicted_n", resp.get("tokens_predicted", 0)) or 0)
                n_p = int(timings.get("prompt_n", resp.get("tokens_evaluated", 0)) or 0)
                ms_per_tok = timings.get("predicted_per_token_ms")
                decode_tps = (n_dec / wall) if wall > 0 and n_dec else 0.0
                prompt_tps = (n_p / wall) if wall > 0 and n_p else 0.0
                rr = RequestResult(
                    idx=i, jp=row["jp"], en_pred=content.strip(),
                    n_prompt_tokens=n_p, n_decoded_tokens=n_dec,
                    wall_seconds=wall, decode_tps=decode_tps, prompt_tps=prompt_tps,
                    server_predicted_per_token_ms=ms_per_tok,
                )
                results.append(rr)
                raw_f.write(json.dumps({**asdict(rr), "en_ref": row.get("en_ref", "")}, ensure_ascii=False) + "\n")
                logger.info("[{}/{}] {} tok in {:.2f}s = {:.1f} tps | {}",
                            i + 1, len(prompts), n_dec, wall, decode_tps, content[:50].replace("\n", " "))

        summary = _summarize(results)
        summary["quant"] = args.quant
        summary["flash_attn"] = bool(args.flash_attn)
        summary["draft_model"] = str(args.draft_model) if args.draft_model else None
        summary["spec_type"] = args.spec_type
        summary["model"] = str(args.model)
        summary["lora"] = str(args.lora) if args.lora else None
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, default=str)

        logger.info("=== {} (-fa={}) draft={} ===",
                    args.quant, args.flash_attn, args.draft_model)
        logger.info("median decode tps: {:.1f}", summary["decode_tps_median"])
        logger.info("p95 wall:          {:.2f}s", summary["wall_p95_s"])
        logger.info("total wall:        {:.1f}s", summary["total_wall_s"])
        print(json.dumps(summary, indent=2))
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
