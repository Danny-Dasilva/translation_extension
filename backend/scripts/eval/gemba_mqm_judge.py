"""GEMBA-MQM judge (Kocmi & Federmann 2023).

Given (source, reference, translation) triples, prompt a large instruction
LLM to identify MQM-style errors and emit a score.

Error categories (major/minor):

    - Accuracy
    - Fluency
    - Style
    - Terminology
    - Locale

Scoring (standard GEMBA-MQM):

    score = -(5 * num_major + 1 * num_minor)

Higher (closer to 0) is better.  We also emit the per-row parsed error list
for qualitative review.

Judges:

- Local 4-bit Qwen2.5-72B-Instruct via ``transformers`` + ``bitsandbytes``.
- OpenAI: ``--judge openai:<model>`` (e.g. ``openai:gpt-4o``).  Reads the
  API key from ``OPENAI_API_KEY``.

CLI:
    python -m backend.scripts.eval.gemba_mqm_judge \
        --predictions pred.jsonl \
        --references ref.jsonl \
        [--judge qwen25-72b-4bit | openai:gpt-4o] \
        --out mqm.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from loguru import logger

# ---------------------------------------------------------------------------
# GEMBA-MQM prompt (from Kocmi & Federmann 2023, reproduced from the
# MicrosoftTranslator/GEMBA reference implementation).
# ---------------------------------------------------------------------------

GEMBA_MQM_FEW_SHOT_SYSTEM = (
    "You are an annotator for the quality of machine translation. Your task is to "
    "identify errors and assess the quality of the translation."
)

GEMBA_MQM_FEW_SHOT_USER_TEMPLATE = """{source_lang} source:
```{source_seg}```
{target_lang} translation:
```{target_seg}```

Based on the source segment and machine translation surrounded with triple backticks, identify error types in the translation and classify them. The categories of errors are: accuracy (addition, mistranslation, omission, untranslated text), fluency (character encoding, grammar, inconsistency, punctuation, register, spelling), style (awkward), terminology (inappropriate for context, inconsistent use), non-translation, other, or no-error.
Each error is classified as one of two categories: major or minor. Major errors disrupt the flow and make the understandability of text difficult or impossible. Minor errors are errors that do not disrupt the flow significantly and what the text is trying to say is still understandable.""".strip()


def _build_messages(source_seg: str, target_seg: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": GEMBA_MQM_FEW_SHOT_SYSTEM},
        {
            "role": "user",
            "content": GEMBA_MQM_FEW_SHOT_USER_TEMPLATE.format(
                source_lang="Japanese",
                source_seg=source_seg,
                target_lang="English",
                target_seg=target_seg,
            ),
        },
    ]


# Output lines look like: "Major errors: accuracy/mistranslation - '...' ".
_ERROR_LINE_RE = re.compile(
    r"(?i)\b(?P<sev>major|minor)\s+errors?\b\s*:\s*(?P<body>.*)"
)


def parse_mqm_response(text: str) -> tuple[int, int, list[dict[str, str]]]:
    """Return (num_major, num_minor, details)."""
    if not text:
        return (0, 0, [])
    major = 0
    minor = 0
    details: list[dict[str, str]] = []

    for line in text.splitlines():
        m = _ERROR_LINE_RE.search(line)
        if not m:
            continue
        sev = m.group("sev").lower()
        body = m.group("body").strip()
        if "no-error" in body.lower() or "no error" in body.lower() or not body:
            continue
        # Multiple errors on one line might be comma-separated.
        for piece in re.split(r";|,(?=\s*[A-Za-z])", body):
            piece = piece.strip().strip("-").strip()
            if not piece:
                continue
            if sev == "major":
                major += 1
            else:
                minor += 1
            details.append({"severity": sev, "description": piece})
    return major, minor, details


def mqm_score(num_major: int, num_minor: int) -> float:
    return -(5.0 * num_major + 1.0 * num_minor)


# ---------------------------------------------------------------------------
# Judge backends
# ---------------------------------------------------------------------------


class _JudgeBase:
    def chat(self, messages: list[dict[str, str]]) -> str:  # pragma: no cover - abstract
        raise NotImplementedError


class LocalQwenJudge(_JudgeBase):
    """Qwen2.5-72B-Instruct loaded at 4-bit via bitsandbytes."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-72B-Instruct") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        logger.info("Loading {} in 4-bit...", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_cfg,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    def chat(self, messages: list[dict[str, str]]) -> str:
        import torch

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(
                **enc,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        gen = out[0, enc["input_ids"].shape[1] :]
        return self.tokenizer.decode(gen, skip_special_tokens=True)


class OpenAIJudge(_JudgeBase):
    """OpenAI-compatible chat judge.

    Honors ``OPENAI_BASE_URL`` so this same code can talk to Groq
    (``https://api.groq.com/openai/v1``), DeepInfra, OpenRouter, vLLM, etc.
    For Groq specifically use ``--judge openai:llama-3.3-70b-versatile``.
    """

    def __init__(self, model: str) -> None:
        from openai import OpenAI  # lazy

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set.")
        base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get(
            "OPENAI_API_BASE"
        )
        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
            logger.info("OpenAIJudge using base_url={}", base_url)
        self.client = OpenAI(**kwargs)
        self.model = model

    def chat(self, messages: list[dict[str, str]]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=0.0,
        )
        return resp.choices[0].message.content or ""


class DryRunJudge(_JudgeBase):
    """Print the request that *would* be sent and return a stub MQM response.

    Useful for verifying the prompt template + endpoint configuration without
    burning Groq quota. The stub response is parsed correctly by
    ``parse_mqm_response`` (yields 0 major / 0 minor — i.e. "no-error").
    """

    def __init__(self, target: str = "openai:dry-run") -> None:
        self.target = target
        self._n = 0

    def chat(self, messages: list[dict[str, str]]) -> str:
        self._n += 1
        api_key_set = bool(os.environ.get("OPENAI_API_KEY"))
        base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get(
            "OPENAI_API_BASE"
        ) or "https://api.openai.com/v1"
        # Print exactly once for the first row, then a compact summary line
        # for subsequent rows (so a 600-row holdout doesn't flood the log).
        if self._n == 1:
            print("=" * 70, file=sys.stderr)
            print(f"DRY RUN | target={self.target}", file=sys.stderr)
            print(f"        | OPENAI_API_KEY={'<set>' if api_key_set else '<unset>'}",
                  file=sys.stderr)
            print(f"        | OPENAI_BASE_URL={base_url}", file=sys.stderr)
            print("        | first-row payload:", file=sys.stderr)
            print(json.dumps({
                "model": self.target,
                "messages": messages,
                "temperature": 0.0,
            }, indent=2, ensure_ascii=False), file=sys.stderr)
            print("=" * 70, file=sys.stderr)
        else:
            print(f"DRY RUN [{self._n}] {messages[-1]['content'][:80]!r}",
                  file=sys.stderr)
        return "Major errors: no-error\nMinor errors: no-error"


def _build_judge(name: str, *, dry_run: bool = False) -> _JudgeBase:
    if dry_run:
        return DryRunJudge(target=name)
    if name.startswith("openai:"):
        return OpenAIJudge(name.split(":", 1)[1])
    if name in {"qwen25-72b-4bit", "qwen2.5-72b-4bit"}:
        return LocalQwenJudge()
    # Allow passing a raw HF id too.
    return LocalQwenJudge(model_name=name)


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _align(preds: list[dict[str, Any]], refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(preds) != len(refs):
        logger.warning("pred/ref length mismatch ({} vs {})", len(preds), len(refs))
    n = min(len(preds), len(refs))
    out: list[dict[str, Any]] = []
    for i in range(n):
        out.append(
            {
                "jp": refs[i].get("jp") or preds[i].get("jp", ""),
                "en_ref": refs[i].get("en_ref") or refs[i].get("en", ""),
                "en_pred": preds[i].get("en_pred") or preds[i].get("en", ""),
            }
        )
    return out


def run_gemba_mqm(
    rows: list[dict[str, Any]],
    *,
    judge: _JudgeBase,
) -> dict[str, Any]:
    per_row: list[dict[str, Any]] = []
    total_major = 0
    total_minor = 0

    for idx, row in enumerate(rows):
        messages = _build_messages(row["jp"], row["en_pred"])
        try:
            resp = judge.chat(messages)
        except Exception as e:
            logger.exception("judge failed on row {}: {}", idx, e)
            resp = ""
        major, minor, details = parse_mqm_response(resp)
        per_row.append(
            {
                "idx": idx,
                "major": major,
                "minor": minor,
                "score": mqm_score(major, minor),
                "details": details,
                "raw": resp,
            }
        )
        total_major += major
        total_minor += minor

    n = max(len(per_row), 1)
    return {
        "n": len(per_row),
        "total_major": total_major,
        "total_minor": total_minor,
        "mean_score": mqm_score(total_major, total_minor) / n,
        "per_row": per_row,
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GEMBA-MQM judge.")
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--references", type=Path, required=True)
    p.add_argument(
        "--judge",
        type=str,
        default="qwen25-72b-4bit",
        help="'qwen25-72b-4bit', a HF id, or 'openai:<model>'.",
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None, help="Subsample N rows for dev.")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "CPU-only mode: prints the chat request that *would* be sent to the "
            "judge and returns a stub no-error response. Use this to verify the "
            "prompt template + endpoint config before burning API quota."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    preds = _read_jsonl(args.predictions)
    refs = _read_jsonl(args.references)
    rows = _align(preds, refs)
    if args.limit:
        rows = rows[: args.limit]

    judge = _build_judge(args.judge, dry_run=args.dry_run)
    report = run_gemba_mqm(rows, judge=judge)
    report["judge"] = args.judge
    report["dry_run"] = bool(args.dry_run)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    logger.info(
        "GEMBA-MQM done: n={} major={} minor={} mean_score={:.3f}",
        report["n"],
        report["total_major"],
        report["total_minor"],
        report["mean_score"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
