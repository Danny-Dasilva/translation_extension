"""Batch-translate Manga109 bubbles JP -> EN with a teacher model.

Spawns a local ``llama-server`` (default Gemma 4 31B Q4_K_XL on the NAS)
and POSTs JSON ``/completion`` requests in concurrent batches.  Saves a
partial parquet every ``--save-every`` examples so the run is resume-safe.

The output prompt template MIRRORS ``backend/scripts/eval/translate_manga_gemma4.py``
so v10 trains on data that matches the eval-time distribution:

    Translate the following Japanese to English. Output only the translation.

    Japanese: <jp>
    English:

When ``--include-speaker-context`` is True, an extra preface is added when
a speaker name is known:

    The following is dialogue spoken by <speaker> in a Japanese manga.
    Translate to natural English.

    Japanese: <jp>
    English:

Output parquet columns:
    [book, page, text_id, jp_text, en_text, speaker, xmin, ymin, xmax, ymax,
     teacher_model, teacher_temp]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl
import requests
from requests.adapters import HTTPAdapter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli_common import configure_logging, logger  # noqa: E402


def _make_session(pool_size: int) -> requests.Session:
    """A requests.Session sized for high-concurrency local llama-server calls.

    The default urllib3 pool is 10 connections, which deadlocks with
    ThreadPoolExecutor(max_workers > 10) when the server-side closes
    keepalive sockets between bursts -- threads pile up in CLOSE_WAIT
    and never time out. Explicit pool sizing + `Connection: close` per
    request makes each call grab a fresh socket and release it cleanly.
    """
    sess = requests.Session()
    adapter = HTTPAdapter(
        pool_connections=pool_size,
        pool_maxsize=pool_size,
        max_retries=0,
        pool_block=True,
    )
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    sess.headers.update({"Connection": "close"})
    return sess


DEFAULT_LLAMA_BIN = "/home/danny/llama.cpp/build/bin/llama-server"
DEFAULT_TEACHER = "/mnt/nas/drive_2/ml-models/gemma-4-31B-it-GGUF/gemma-4-31B-it-UD-Q4_K_XL.gguf"
DEFAULT_INPUT = Path(
    "/home/danny/Documents/personal/extension/backend/scripts/data/manga109/bubbles.parquet"
)
DEFAULT_OUTPUT = Path(
    "/home/danny/Documents/personal/extension/backend/scripts/data/manga109/bubbles_translated.parquet"
)


# Two prompt formats:
#  - PROMPT_*  : completion-style.  Used at eval time (translate_manga_gemma4.py)
#                via `Japanese: ... \nEnglish:` cloze.  This is what v10 will be
#                trained on so the SFT prompt matches the eval distribution.
#  - SYSTEM_*  : chat-style.  We send the completion-style prompt as the user
#                message under a strict system prompt.  Gemma 4 31B-it is heavily
#                instruction-tuned and refuses to behave correctly with raw
#                completion (it emits "<channel|>thought" reasoning blocks); the
#                chat envelope plus the strict system instruction force a clean
#                single-line answer.  The user-message body still uses the
#                Japanese/English cloze, so the resulting (jp, en) pair is
#                directly usable as v10 training data.
PROMPT_NOSPEAKER = (
    "Translate the following Japanese to English. Output only the translation.\n\n"
    "Japanese: {jp}\nEnglish:"
)
PROMPT_SPEAKER = (
    "The following is dialogue spoken by {speaker} in a Japanese manga. "
    "Translate to natural English.\n\n"
    "Japanese: {jp}\nEnglish:"
)
SYSTEM_PROMPT = (
    "You are a professional Japanese-to-English manga translator. "
    "Translate the bubble verbatim into natural English. "
    "Output ONLY the translation -- no commentary, no options, no notes, "
    "no quotation marks, no explanation, single line."
)


# ---- Cleaning regex (lifted from eval/translate_manga_gemma4.py) ----
# We compile everything as third-party `regex` so we can pass timeout= per
# match call. The Python stdlib `re` engine has no timeout and the back-
# referenced patterns below (REPEAT_PHRASE_RE, REPEAT_NGRAM_RE,
# LONG_TOKEN_REPEAT_RE) can catastrophically backtrack on adversarial
# teacher outputs (long repetitive babble). A timeout turns those into a
# clean fallback path instead of a permanent thread hang.
import regex as _re  # type: ignore[import-not-found]
NEWLINE_RE = _re.compile(r"[\r\n]")
NEXT_PROMPT_RE = _re.compile(r"\s*(?:Japanese:|JP:|English:|EN:).*$", _re.S)
TRAILING_PAREN_LOOP_RE = _re.compile(r"(?:\s*[\(\[][^\)\]]{0,25}[\)\]]){2,}\s*$")
TRAILING_NOISE_RE = _re.compile(r"(?:\s*[.!?\"'~_\-•・]+){4,}\s*$")
TRAILING_CHAR_REP_RE = _re.compile(r"(?:\s*(\S))\s*(?:\1\s*){3,}$")
LONG_TOKEN_REPEAT_RE = _re.compile(r"\b(\w{3,15}?)\1{3,}\w*\b", _re.I)
REPEAT_PHRASE_RE = _re.compile(r"(\b[^.!?]{3,80}[.!?]+)\s*(?:\1\s*)+", _re.I)
REPEAT_NGRAM_RE = _re.compile(r"(\b.{2,40}?\b)(?:\s*\1){1,}")
_REGEX_TIMEOUT_S = 1.0  # plenty for 500-char inputs; pathological hits abort

def _safe_sub(pattern, repl, text):
    """re.sub but bounded by _REGEX_TIMEOUT_S; on timeout return text unchanged."""
    try:
        return pattern.sub(repl, text, timeout=_REGEX_TIMEOUT_S)
    except TimeoutError:
        return text


def clean(text: str) -> str:
    if not text:
        return ""
    # Strip leading whitespace BEFORE splitting on newline -- Gemma 4 often
    # emits a leading "\n" or " " which would otherwise truncate to empty.
    text = text.lstrip()
    text = NEWLINE_RE.split(text, 1)[0]
    # Hard cap to bound regex worst-case. Valid manga translations are
    # < 250 chars; max_tokens=60 produces ~240 chars max. 500 = safety margin.
    if len(text) > 500:
        text = text[:500]
    text = _safe_sub(NEXT_PROMPT_RE, "", text)
    text = _safe_sub(TRAILING_PAREN_LOOP_RE, "", text)
    text = _safe_sub(TRAILING_NOISE_RE, "", text)
    text = _safe_sub(TRAILING_CHAR_REP_RE, "", text)
    text = _safe_sub(LONG_TOKEN_REPEAT_RE, r"\1", text)
    prev = None
    iters = 0
    while prev != text and iters < 10:
        prev = text
        text = _safe_sub(REPEAT_PHRASE_RE, r"\1 ", text)
        iters += 1
    prev = None
    iters = 0
    while prev != text and iters < 10:
        prev = text
        text = _safe_sub(REPEAT_NGRAM_RE, r"\1", text)
        iters += 1
    text = re.sub(r"\.{4,}", "...", text)
    text = re.sub(r"([!?])\1{3,}", r"\1\1\1", text)
    return text.strip()


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _wait_ready(port: int, timeout: float = 300.0) -> None:
    deadline = time.time() + timeout
    last_err = None
    while time.time() < deadline:
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=2.0)
            if r.status_code == 200:
                return
        except Exception as e:  # noqa: BLE001
            last_err = e
        time.sleep(1.0)
    raise RuntimeError(f"llama-server did not become ready on :{port}: {last_err}")


def _spawn_server(args: argparse.Namespace, port: int, log_path: Path) -> subprocess.Popen:
    cmd = [
        args.llama_bin,
        "-m", str(args.teacher),
        "--port", str(port),
        "--host", "127.0.0.1",
        "-c", str(args.ctx),
        "-ngl", str(args.n_gpu_layers),
        "-t", str(args.threads),
        "-fa", "on" if args.flash_attn else "off",
        "--parallel", str(args.parallel),
        "-np", str(args.parallel),
        # Gemma 4 31B-it is a thinking/reasoning model; disable so /v1/chat
        # returns the answer in `content` instead of `reasoning_content`.
        "--reasoning", args.reasoning,
    ]
    if args.reasoning_budget is not None:
        cmd += ["--reasoning-budget", str(args.reasoning_budget)]
    if args.no_mmap:
        cmd.append("--no-mmap")
    if args.extra_server_args:
        cmd += args.extra_server_args.split()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "w", buffering=1)
    logger.info("starting llama-server: {}", " ".join(cmd))
    return subprocess.Popen(
        cmd,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        env={**os.environ, "LLAMA_NUMA": "isolate"},
    )


def _build_prompt(jp: str, speaker: str | None, include_speaker: bool) -> str:
    if include_speaker and speaker:
        return PROMPT_SPEAKER.format(speaker=speaker, jp=jp)
    return PROMPT_NOSPEAKER.format(jp=jp)


def _completion_one(
    port: int,
    user_prompt: str,
    *,
    max_tokens: int,
    temperature: float,
    timeout: float,
    use_chat: bool = True,
    session: requests.Session | None = None,
) -> tuple[str, int, float]:
    """Call llama-server.

    With ``use_chat=True`` (default) we hit ``/v1/chat/completions`` so the
    server applies the model's chat template -- required for Gemma 4 31B-it
    which otherwise emits multi-option "thought" blocks.  The user message
    is the same cloze prompt eval uses, so the (jp, en) pair is directly
    suitable as v10 SFT data.
    """
    t0 = time.perf_counter()
    poster = session.post if session is not None else requests.post
    if use_chat:
        body = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "stream": False,
        }
        r = poster(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json=body,
            timeout=timeout,
        )
        dt = time.perf_counter() - t0
        r.raise_for_status()
        j = r.json()
        msg = j["choices"][0]["message"]
        # If --reasoning off didn't take effect (older llama.cpp builds,
        # different chat template), the answer may land in reasoning_content
        # while content is empty. Fall back to the reasoning channel and
        # let clean() strip thought-style noise downstream.
        raw = msg.get("content") or ""
        if not raw.strip():
            raw = msg.get("reasoning_content") or ""
        n_decoded = j.get("usage", {}).get("completion_tokens") or 0
    else:
        body = {
            "prompt": user_prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "min_p": 0.05,
            "stream": False,
            "cache_prompt": False,
            "stop": ["\nJapanese:", "\n\nJapanese:", "\nJP:"],
        }
        r = poster(
            f"http://127.0.0.1:{port}/completion",
            json=body,
            timeout=timeout,
        )
        dt = time.perf_counter() - t0
        r.raise_for_status()
        j = r.json()
        raw = j.get("content") or ""
        n_decoded = j.get("tokens_predicted") or j.get("timings", {}).get("predicted_n") or 0
    return raw, int(n_decoded), dt


def _save_partial(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    df = pl.DataFrame(rows)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--teacher", type=str, default=DEFAULT_TEACHER)
    p.add_argument("--llama-bin", type=str, default=DEFAULT_LLAMA_BIN)
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--n-gpu-layers", type=int, default=999, dest="n_gpu_layers")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--flash-attn", action="store_true", default=True)
    p.add_argument("--no-mmap", action="store_true", default=False)
    p.add_argument("--parallel", type=int, default=8,
                   help="server slots; client concurrency is capped to this")
    p.add_argument("--batch-size", type=int, default=8,
                   help="concurrent in-flight requests (<= --parallel)")
    p.add_argument("--max-tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--include-speaker-context",
                   type=lambda v: v.lower() not in {"0", "false", "no"},
                   default=True)
    p.add_argument("--limit", type=int, default=0,
                   help="if >0, translate only N rows (debug). With --seed, random sample.")
    p.add_argument("--seed", type=int, default=None,
                   help="if set with --limit, take a deterministic random sample of N rows")
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--resume", action="store_true", default=True,
                   help="if --output exists, load completed text_ids and skip them")
    p.add_argument("--no-resume", action="store_false", dest="resume")
    p.add_argument("--server-log", type=Path, default=None)
    p.add_argument("--request-timeout", type=float, default=180.0)
    p.add_argument("--extra-server-args", default="")
    p.add_argument("--server-only", action="store_true",
                   help="reuse a running server on --port; do NOT spawn one")
    p.add_argument("--no-chat", action="store_true",
                   help="use raw /completion endpoint instead of /v1/chat/completions")
    p.add_argument("--reasoning", default="off", choices=["on", "off", "auto"],
                   help="llama-server --reasoning flag (Gemma 4 31B-it must be 'off')")
    p.add_argument("--reasoning-budget", type=int, default=None,
                   help="optional --reasoning-budget; -1=unrestricted, 0=immediate end")
    return p.parse_args()


def main() -> int:
    configure_logging()
    args = parse_args()

    if not args.input.exists():
        logger.error(f"input parquet not found: {args.input}")
        return 2
    df_in = pl.read_parquet(args.input)
    logger.info(f"loaded {len(df_in):,} rows from {args.input}")

    # Resume support: skip text_ids already present in --output.
    completed: dict[str, dict] = {}
    if args.resume and args.output.exists():
        try:
            existing = pl.read_parquet(args.output)
            for row in existing.iter_rows(named=True):
                # Use (book, text_id) as key since text_ids are only unique per book.
                key = f"{row['book']}::{row['text_id']}"
                completed[key] = dict(row)
            logger.info(f"resume: {len(completed):,} rows already translated")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"resume read failed ({e}); starting fresh")
            completed = {}

    # Build the work list.
    work: list[dict] = []
    for row in df_in.iter_rows(named=True):
        key = f"{row['book']}::{row['text_id']}"
        if key in completed:
            continue
        work.append(dict(row))
    if args.limit and args.limit > 0:
        if args.seed is not None:
            import random as _random
            rng = _random.Random(args.seed)
            if args.limit < len(work):
                work = rng.sample(work, args.limit)
            logger.info(f"random subsample: limit={args.limit} seed={args.seed}")
        else:
            work = work[: args.limit]
    logger.info(f"to translate: {len(work):,} rows")

    # Spawn server (unless --server-only and a port is given).
    port = args.port or _free_port()
    proc: subprocess.Popen | None = None
    if not args.server_only:
        log_path = args.server_log or args.output.parent / "manga109-llama-server.log"
        proc = _spawn_server(args, port, log_path)
    try:
        if not args.server_only:
            _wait_ready(port, timeout=600.0)
            logger.info(f"llama-server ready on :{port}")
        else:
            logger.info(f"using existing server on :{port}")

        # Pre-load output rows (already-completed first), then translate.
        out_rows: list[dict] = list(completed.values())
        # Shared HTTP session sized for the threadpool. Default urllib3 pool
        # is 10, which deadlocks ThreadPoolExecutor(>10) once keepalive sockets
        # accumulate in CLOSE_WAIT (server idle, client stuck reading).
        session = _make_session(pool_size=max(args.batch_size * 2, 32))
        # Estimate via warmup of 1 prompt.
        if work:
            warm = work[0]
            warm_prompt = _build_prompt(
                warm["jp_text"], warm.get("speaker"), args.include_speaker_context
            )
            try:
                _, _, _ = _completion_one(
                    port, warm_prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    timeout=args.request_timeout,
                    use_chat=not args.no_chat,
                    session=session,
                )
                logger.info("warmup ok")
            except Exception as e:  # noqa: BLE001
                logger.error(f"warmup failed: {e}")
                raise

        teacher_label = Path(args.teacher).name if "/" in args.teacher else args.teacher

        total_decoded = 0
        t_all = time.perf_counter()
        save_lock = threading.Lock()
        last_save_count = 0

        def translate_row(row: dict) -> dict:
            prompt = _build_prompt(
                row["jp_text"], row.get("speaker"), args.include_speaker_context
            )
            raw, n_dec, _dt = _completion_one(
                port, prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout=args.request_timeout,
                use_chat=not args.no_chat,
                session=session,
            )
            return {
                **row,
                "en_text": clean(raw),
                "en_text_raw": raw,
                "teacher_model": teacher_label,
                "teacher_temp": args.temperature,
                "n_decoded": n_dec,
            }

        # Concurrent dispatch -- llama-server handles the slot scheduling.
        completed_count = 0
        with ThreadPoolExecutor(max_workers=args.batch_size) as pool:
            futures = {pool.submit(translate_row, r): r for r in work}
            for fut in as_completed(futures):
                src_row = futures[fut]
                try:
                    res = fut.result()
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        f"translate failed for {src_row['book']}::{src_row['text_id']}: {e}"
                    )
                    continue
                with save_lock:
                    out_rows.append(res)
                    total_decoded += res["n_decoded"]
                    completed_count += 1
                    if completed_count % 50 == 0:
                        elapsed = time.perf_counter() - t_all
                        tps = total_decoded / elapsed if elapsed > 0 else 0
                        rps = completed_count / elapsed if elapsed > 0 else 0
                        remaining = len(work) - completed_count
                        eta_s = remaining / rps if rps > 0 else 0
                        logger.info(
                            f"  {completed_count:,}/{len(work):,} "
                            f"({rps:.1f} req/s, {tps:.0f} tok/s, "
                            f"ETA {eta_s/60:.1f} min)"
                        )
                    if completed_count - last_save_count >= args.save_every:
                        _save_partial(out_rows, args.output)
                        last_save_count = completed_count
                        logger.info(
                            f"  partial save: {len(out_rows):,} rows -> {args.output}"
                        )

        # Final save.
        _save_partial(out_rows, args.output)
        wall = time.perf_counter() - t_all
        logger.info(
            f"DONE: translated {completed_count:,} rows in {wall/60:.1f} min "
            f"({total_decoded} tokens, {total_decoded/max(wall,1e-9):.1f} tok/s); "
            f"output: {args.output}"
        )

        # Print 10 samples.
        df_out = pl.read_parquet(args.output)
        sample = df_out.sample(min(10, len(df_out)), seed=0) if len(df_out) > 0 else df_out
        print("\n=== 10 sample translations ===", file=sys.stderr)
        for r in sample.iter_rows(named=True):
            spk = f" <{r['speaker']}>" if r.get("speaker") else ""
            print(f"  [{r['book']} p{r['page']}{spk}]", file=sys.stderr)
            print(f"    JP: {r['jp_text']}", file=sys.stderr)
            print(f"    EN: {r['en_text']}", file=sys.stderr)
        return 0
    finally:
        if proc is not None:
            logger.info("stopping llama-server")
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    sys.exit(main())
