"""vLLM (OpenAI-compatible) translation service.

Talks to a local vLLM `vllm serve` instance over the OpenAI Chat Completions
API. Designed for the v10-it Gemma 4 E4B + MTP setup launched by
``backend/scripts/eval/serve_v10it_vllm.sh`` (default port 8000,
served-model-name "v10it").

Drop-in translation service — same `translate_single` /
`translate_batched` async surface used by the e2e visualizer and routers.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import List

import httpx

from app.config import settings
from app.services.sfx_glossary import sfx_pre_translate
from app.services.translation_text_utils import (
    LIGHT_SYSTEM_PROMPT,
    clean_translation_output,
    parse_tagged_blocks,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# v11 page-context prompt format
# --------------------------------------------------------------------------- #
# These strings MUST stay byte-for-byte identical to the templates the v11 LoRA
# was trained on. See backend/scripts/data/v11/build_v11_dataset.py
# (PAGE_INSTR / PLAIN_INSTR / build_context_prompt / build_plain_prompt). A
# whitespace/marker mismatch silently degrades translation quality, so do NOT
# "tidy" these — they are a contract with the training data.
V11_PAGE_INSTR = (
    "Translate the marked line of this manga page from Japanese to English. "
    "Use the page context for speakers, pronouns, and continuity. "
    "Output only the translation of the marked line."
)
V11_PLAIN_INSTR = "Translate the following Japanese to English. Output only the translation."


import re as _nre  # noqa: E402  (local alias for short-utterance normalize)

# Hiragana, katakana, the long-vowel mark, and the katakana middle dot.
_KANA_CLASS = "぀-ゟ゠-ヿー"
# A separator (interpunct / dot / ascii-or-fullwidth space) WEDGED between two
# kana: バ.カ / わ け. We keep `・` unless it sits between two kana (handled by
# including it in the separator class only for the between-kana rule).
_BETWEEN_KANA_SEP_RE = _nre.compile(
    rf"(?<=[{_KANA_CLASS}])[\.・‧·\s　]+(?=[{_KANA_CLASS}])"
)
# A single char repeated >=4 times (runaway kana: ですですですです / ーーーーー).
_KANA_RUNAWAY_RE = _nre.compile(r"(.)\1{3,}")


def normalize_short_utterance(jp: str, max_len: int = 8) -> str:
    """Normalize a SHORT Japanese utterance before translation.

    Strips interpunct/dot/space separators wedged BETWEEN two kana
    (バ.カ -> バカ; わ け -> わけ) and collapses runaway repeated kana
    ((.)\\1{3,} -> two copies). Longer lines (> max_len chars) are left
    untouched. `・` is only removed when it sits between two kana on a short
    line; otherwise it survives.
    """
    if not jp:
        return jp
    if len(jp) > max_len:
        return jp
    out = _BETWEEN_KANA_SEP_RE.sub("", jp)
    out = _KANA_RUNAWAY_RE.sub(r"\1\1", out)
    return out.strip()


# --------------------------------------------------------------------------- #
# 笑 (net-slang "lol"/"haha") sentence-final marker
# --------------------------------------------------------------------------- #
# The Japanese net-slang sentence-final 笑 means "lol"/"haha", NOT the verb
# "to laugh". The small model systematically renders おばさん笑 -> "Laugh, lady!",
# カレー笑 -> "Curry laughter", etc. We strip a TRAILING standalone 笑 (incl.
# 笑笑) from the JP before prompting, translate the remainder unchanged, then
# append ", haha" to the cleaned English. A bare 笑-only bubble -> "haha".
#
# CONSERVATIVE GUARD: 笑 is only stripped when it is a trailing run that is NOT
# preceded by a kanji — so it never touches 笑 inside a word: 笑顔, 笑う/笑った/
# 笑える, 爆笑, 微笑, 苦笑, 嘲笑 all keep their 笑. Trailing punctuation / emoji /
# whitespace AFTER the 笑 run, and a few common emphatic marks wedged between
# the body and the 笑 (! ! ？ ～ ♪ ♡ ☆ w), are tolerated.
_WARAI_MARK_RE = _nre.compile(
    r"(?<![一-鿿])"            # NOT preceded by a kanji (guards 爆笑/微笑/苦笑/嘲笑)
    r"[\s!！?？~〜ｗw♪♡☆、。,.…・]*"  # optional emphatic glue before the marker
    r"(笑+)"                    # the trailing 笑 run itself
    r"[\s!！?？~〜♪♡☆、。,.…]*$"   # optional trailing punctuation / emoji
)


def strip_warai_marker(jp: str) -> tuple[str, bool]:
    """Split a trailing net-slang 笑 marker off a Japanese line.

    Returns ``(body, had_marker)``. ``body`` is the JP with the trailing 笑
    run (and any trailing punctuation after it) removed; ``had_marker`` is True
    when a marker was stripped. 笑 that is part of a word (笑顔, 笑う, 爆笑, ...)
    is NEVER stripped — the regex requires the 笑 run to NOT be preceded by a
    kanji, and 笑う/笑った/笑顔 have either a kanji-adjacent okurigana or are
    word-initial-followed-by-more-text (not a *trailing* run).
    """
    if not jp or "笑" not in jp:
        return jp, False
    m = _WARAI_MARK_RE.search(jp)
    if not m:
        return jp, False
    body = jp[: m.start()].rstrip()
    return body, True


def append_haha(en: str) -> str:
    """Append the net-slang ", haha" tail to a cleaned English translation.

    A bare 笑-only bubble (empty body -> empty translation) becomes "haha".
    An already-present trailing "haha"/"lol" is not duplicated.
    """
    s = (en or "").strip()
    if not s:
        return "haha"
    if _nre.search(r"\b(haha|lol)\b\W*$", s, _nre.IGNORECASE):
        return s
    # Place the tag naturally relative to trailing terminal punctuation:
    #   "Curry."   -> "Curry, haha"      (drop a lone sentence period)
    #   "Stop it!" -> "Stop it, haha!"   (keep !/?, tag goes inside)
    #   "Really?"  -> "Really, haha?"
    #   "Wait..."  -> "Wait, haha..."    (keep ellipsis feel)
    m = _nre.search(r"[.!?…]+$", s)
    if m:
        punct = m.group(0)
        body = s[: m.start()].rstrip()
        if body:
            if punct == ".":
                return f"{body}, haha"
            return f"{body}, haha{punct}"
    return f"{s}, haha"


def build_v11_context_prompt(lines: List[str], k_idx: int) -> str:
    """CONTEXT-AUGMENTED single-line user message (page translation).

    Mirrors build_v11_dataset.build_context_prompt(PAGE_INSTR, lines, k_idx):
        {PAGE_INSTR}\n\nPage:\n1. {jp1}\n...\nN. {jpN}\n\nTranslate line {k}: {jpk}
    `lines` is the full ordered page (reading order); k_idx is the 0-based
    target line. k in the prompt is 1-based.
    """
    numbered = "\n".join(f"{i + 1}. {ln}" for i, ln in enumerate(lines))
    k = k_idx + 1
    # Normalize ONLY the target line (context lines stay verbatim).
    target = lines[k_idx]
    if getattr(settings, "short_utterance_normalize_enabled", True):
        target = normalize_short_utterance(target)
    return (
        f"{V11_PAGE_INSTR}\n\n"
        f"Page:\n{numbered}\n\n"
        f"Translate line {k}: {target}"
    )


def build_v11_plain_prompt(jp: str) -> str:
    """PLAIN single-line user message (no page context).

    Mirrors build_v11_dataset.build_plain_prompt:
        {PLAIN_INSTR}\n\nJapanese: {jp}
    """
    if getattr(settings, "short_utterance_normalize_enabled", True):
        jp = normalize_short_utterance(jp)
    return f"{V11_PLAIN_INSTR}\n\nJapanese: {jp}"


import re as _re

_LEADING_IDX_RE = _re.compile(r"^\s*\[?\d+\]?[.):\-]?\s*")
# A "preamble" is meta/chatter rather than a translation: ends with a colon,
# or starts with a common lead-in. Translations rarely end in a colon.
_PREAMBLE_RE = _re.compile(
    r"^(here|sure|okay|ok|certainly|translations?|output|the\s+following)\b",
    _re.IGNORECASE,
)


def _strip_leading_index(line: str) -> str:
    """Drop a leading ``[3]`` / ``3.`` / ``3)`` index from a plain line."""
    return _LEADING_IDX_RE.sub("", line, count=1).strip()


# A run of >=6 identical chars, or a short token (<=3 chars) repeated >=4 times:
# the degenerate `||||...` / `aaaa` / `lololo` loops the small model emits on
# garbled SFX OCR. We cut the line at the start of the runaway tail.
_RUNAWAY_CHAR_RE = _re.compile(r"(.)\1{5,}")
_RUNAWAY_TOK_RE = _re.compile(r"(.{1,3}?)\1{3,}")


def _strip_runaway_repeat(line: str) -> str:
    """Cut a degenerate repetition tail off a line, keeping the real prefix."""
    if not line:
        return line
    cut = len(line)
    m = _RUNAWAY_CHAR_RE.search(line)
    if m:
        cut = min(cut, m.start())
    m = _RUNAWAY_TOK_RE.search(line)
    if m and (m.end() - m.start()) >= 8:  # only long token-loops, not "haha"
        cut = min(cut, m.start())
    return line[:cut].strip()


def _looks_like_preamble(line: str) -> bool:
    """True if a line reads as meta/chatter rather than a translation."""
    s = line.strip()
    if not s:
        return True
    if s.endswith(":"):
        return True
    return bool(_PREAMBLE_RE.match(s))


class VLLMOpenAITranslationService:
    def __init__(
        self,
        base_url: str | None = None,
        model_name: str | None = None,
        api_key: str = "EMPTY",
        request_timeout_s: float = 120.0,
        concurrency: int = 8,
    ):
        self.base_url = (base_url or os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")).rstrip("/")
        self.model_name = model_name or os.environ.get("VLLM_MODEL", "v10it")
        self.api_key = api_key
        self.timeout = request_timeout_s
        self._client = httpx.AsyncClient(timeout=request_timeout_s)
        self._sem = asyncio.Semaphore(max(1, concurrency))
        self._healthy = False
        logger.info(f"vLLM client targeting {self.base_url} model={self.model_name}")

    async def _ensure_healthy(self) -> None:
        if self._healthy:
            return
        try:
            r = await self._client.get(f"{self.base_url}/models")
            r.raise_for_status()
            ids = [m.get("id") for m in r.json().get("data", [])]
            if self.model_name not in ids:
                logger.warning(
                    f"vLLM /v1/models returned {ids}; expected {self.model_name}"
                )
            self._healthy = True
        except Exception as e:
            raise RuntimeError(
                f"vLLM server at {self.base_url} not reachable. "
                f"Start it with: bash backend/scripts/eval/serve_v10it_vllm.sh "
                f"(orig error: {e!r})"
            )

    async def _chat(
        self,
        messages: List[dict],
        max_tokens: int,
        temperature: float = 0.0,
        repetition_penalty: float | None = None,
    ) -> str:
        await self._ensure_healthy()
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if repetition_penalty is not None:
            # vLLM-specific sampler param: suppresses the degenerate `||||`/char
            # repetition loops the small model falls into on garbled SFX OCR
            # (those loops eat the token budget and truncate the tail -> mismatch).
            payload["repetition_penalty"] = repetition_penalty
        async with self._sem:
            r = await self._client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
            )
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"] or ""

    async def translate_single(
        self, text: str, target_language: str = "English"
    ) -> str:
        if not text.strip():
            return ""
        # PRE-LLM GATE 1: glossary-matched SFX bypass the model entirely
        # (ぬちょ -> "Squelch", ビクン -> "Twitch", ガバガバに -> "so loose").
        sfx = sfx_pre_translate(text)
        if sfx is not None:
            return sfx
        # PRE-LLM GATE 2: net-slang 笑 marker. Strip a trailing standalone 笑,
        # translate the remainder with a BYTE-IDENTICAL prompt, append ", haha".
        body, had_warai = strip_warai_marker(text)
        if had_warai:
            if not body.strip():
                return "haha"  # bare 笑 / 笑笑 bubble
            return append_haha(await self.translate_single(body, target_language))
        if settings.translation_v11_pagecontext:
            # v11 PLAIN single-line format (byte-for-byte the trained template).
            # v11 is JP->EN only; target_language is ignored on this path (the
            # model was not trained on other targets).
            content = build_v11_plain_prompt(text)
        else:
            # Mirror the prompt format used by the dedicated translation models.
            content = (
                f"Translate the following segment into {target_language}, "
                f"without additional explanation.\n\n{text}"
            )
        msg = [{"role": "user", "content": content}]
        try:
            raw = await self._chat(
                msg, max_tokens=settings.translate_max_tokens, temperature=0.0
            )
        except Exception as e:
            logger.warning(f"vLLM translate_single failed: {e!r}")
            return ""
        return clean_translation_output(raw)

    async def translate_batched(
        self, texts: List[str], target_language: str = "English"
    ) -> List[str]:
        """Fan out to per-bubble translate_single concurrently.

        vLLM with continuous batching handles concurrent requests well, so
        N parallel single-bubble calls is typically faster than one giant
        tagged-block prompt — and it avoids the small-model regression
        observed with the few-shot tagged prompt on dedicated translation
        models.
        """
        if not texts:
            return []
        return await asyncio.gather(
            *(self.translate_single(t, target_language) for t in texts)
        )

    async def translate_page_context(
        self, texts: List[str], target_language: str = "English"
    ) -> List[str]:
        """v11 CONTEXT-AUGMENTED page translation: N marked-line calls.

        For a page of N bubbles, issue N independent chat calls. Each call sends
        the FULL numbered page as context and asks for ONE marked line:

            Translate the marked line of this manga page from Japanese to English.
            Use the page context for speakers, pronouns, and continuity. Output
            only the translation of the marked line.

            Page:
            1. {jp1}
            ...
            N. {jpN}

            Translate line {k}: {jpk}

        and the model returns just that one line's English. This is the EXACT
        format the v11 LoRA was trained on (see build_v11_dataset.py), so there
        is no N-in/N-out alignment risk: every call yields exactly one output
        mapped 1:1 to its bubble. The "Page:\n1. …\nN. …" prefix is byte-identical
        across the N calls, so vLLM prefix-caching amortizes the shared context.

        Returns a list of length N (one translation per input bubble). Empty
        inputs map to "". On a per-call failure the slot is left "" so the caller
        can gap-fill or fall back; the whole list is never silently dropped.

        This is the "all lines are targets" case of
        ``translate_page_context_marked``: every input line is BOTH context AND a
        translation target. Use ``translate_page_context_marked`` when the page
        context (all detected dialogue) is wider than the set of lines you want
        translated back (e.g. dropped/garbled dialogue lines that should still
        inform pronouns/speakers but are not rendered).
        """
        if not texts:
            return []
        page_lines = [t if t is not None else "" for t in texts]
        return await self.translate_page_context_marked(
            page_lines, list(range(len(page_lines))), target_language
        )

    async def _translate_one_marked(
        self, page_lines: List[str], k_idx: int, target_language: str
    ) -> str:
        """Translate ONE marked line (``k_idx``) of a fully-specified page.

        ``page_lines`` is the COMPLETE numbered context (every detected dialogue
        line, in reading order); ``k_idx`` is the 0-based marked target within
        it. The numbered "Page:" block is byte-compatible with
        ``build_v11_context_prompt`` so the served prompt matches the v11 LoRA's
        training template exactly. Pre-LLM SFX-glossary and net-slang 笑 gates
        run on the MARKED line only (context lines stay verbatim).
        """
        src = page_lines[k_idx]
        if not src.strip():
            return ""
        # PRE-LLM GATE 1: glossary-matched SFX bypass the model entirely.
        sfx = sfx_pre_translate(src)
        if sfx is not None:
            return sfx
        # PRE-LLM GATE 2: net-slang 笑 marker on the MARKED line. Strip the
        # trailing 笑 from the target only (context lines stay verbatim), so the
        # v11 template/context is byte-identical for non-笑 targets.
        body, had_warai = strip_warai_marker(src)
        append_warai = had_warai
        if had_warai and not body.strip():
            return "haha"  # bare 笑 / 笑笑 bubble — no model call needed
        if had_warai:
            # Substitute ONLY the marked line with its 笑-stripped body; the
            # numbered context list is otherwise unchanged.
            lines_for_prompt = list(page_lines)
            lines_for_prompt[k_idx] = body
        else:
            lines_for_prompt = page_lines
        prompt = build_v11_context_prompt(lines_for_prompt, k_idx)
        msg = [{"role": "user", "content": prompt}]
        try:
            raw = await self._chat(
                msg,
                max_tokens=settings.translate_max_tokens,
                temperature=0.0,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"vLLM translate_page_context line {k_idx + 1} failed: {e!r}")
            return ""
        # One line out; clean exactly as the single-line path does.
        cleaned = clean_translation_output(raw)
        return append_haha(cleaned) if append_warai else cleaned

    async def translate_page_context_marked(
        self,
        page_lines: List[str],
        target_indices: List[int],
        target_language: str = "English",
    ) -> List[str]:
        """v11 page-context translation over a WIDER context than the targets.

        ``page_lines`` is the WHOLE page's dialogue, in reading order — EVERY
        detected dialogue line, including ones dropped downstream (garbled / low
        OCR-conf dialogue) that must still inform speaker/pronoun/continuity but
        are NOT rendered. Pure-SFX boxes are NOT dialogue and must be excluded by
        the caller before building ``page_lines``.

        ``target_indices`` are the 0-based positions in ``page_lines`` to
        actually translate and return (the KEPT lines). For each target we issue
        ONE marked-line call whose numbered "Page:" context is the FULL
        ``page_lines`` — so the model sees the same page it was trained on, with
        no gaps where dropped dialogue used to be. Context-only lines are never
        marked / never requested.

        Returns a list aligned 1:1 with ``target_indices`` (same order). Empty
        target lines map to "". The shared full-page prefix is byte-identical
        across calls, so vLLM prefix-caching amortizes it.
        """
        if not target_indices:
            return []
        page_lines = [t if t is not None else "" for t in page_lines]
        results = await asyncio.gather(
            *(
                self._translate_one_marked(page_lines, k, target_language)
                for k in target_indices
            )
        )
        return list(results)

    async def translate_numbered_block(
        self, texts: List[str], target_language: str = "English"
    ) -> List[str]:
        """Page-level translation entry point used by the router.

        When ``settings.translation_v11_pagecontext`` is True (default), this
        delegates to ``translate_page_context`` — the v11 N-marked-line format.
        When False, it falls back to the prior numbered-block ([N]/tagged)
        single-call path below (kept intact, dormant behind the flag).
        """
        if settings.translation_v11_pagecontext:
            return await self.translate_page_context(texts, target_language)
        return await self._translate_numbered_block_legacy(texts, target_language)

    async def _translate_numbered_block_legacy(
        self, texts: List[str], target_language: str = "English"
    ) -> List[str]:
        """TRUE single-call page-level translation with a system prompt.

        Packs all of a page's bubbles into ONE generate call as `[N]text`
        tagged blocks, sending the strong BATCHED_SYSTEM_PROMPT as a `system`
        message (intra-page context + target-language lock + romanization/
        full-width punctuation bans) and the tagged source as the `user`
        message.

        Parsing accepts EITHER the `[N]`-tagged output the prompt requests OR a
        plain one-translation-per-line response (the v10it fine-tune emits the
        latter): tags are preferred, else lines are split and matched 1:1.
        Returns [] on any count mismatch so the caller can fall back to the
        per-bubble path (preserves the existing safety contract).
        """
        if not texts:
            return []
        n = len(texts)

        # IMPORTANT: the v10it fine-tune was NOT trained on the heavy few-shot
        # BATCHED_SYSTEM_PROMPT / `[N]text` tagged format — given that prompt it
        # collapses a whole page to a single confused line. It DOES reliably do a
        # plain numbered list ("1. text" -> "1. translation", one per line) from
        # a short user instruction. We use that here; the garble gate upstream
        # keeps low-confidence SFX out of the batch (they poison the generation),
        # and the parser/salvage/gap-fill below recover partial responses so we
        # hold page context whenever the model produces a usable list.
        instr = (
            f"Translate each numbered line below into {target_language}. "
            f"Output the SAME numbers, one translation per line, in order, "
            f"nothing else. Keep every line — do not merge, drop, or add lines."
        )
        body = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
        user_src = f"{instr}\n\n{body}"

        # A/B-gated LIGHT system message. Default OFF (settings flag False):
        # v10it collapses on the heavy BATCHED_SYSTEM_PROMPT, so we only opt into
        # a short genre/self-reference primer when explicitly enabled. _msgs()
        # builds [system?, user] so the first call and the strict retry share it.
        sys_msg = (
            {"role": "system",
             "content": LIGHT_SYSTEM_PROMPT.format(target=target_language)}
            if settings.translation_system_prompt_enabled
            else None
        )

        def _msgs(user_content: str) -> List[dict]:
            msgs: List[dict] = []
            if sys_msg is not None:
                msgs.append(sys_msg)
            msgs.append({"role": "user", "content": user_content})
            return msgs

        # Token budget MUST scale with N so long pages aren't truncated (a
        # truncated tail drops lines -> count mismatch -> isolation fallback).
        budget = max(256, settings.translate_max_tokens * n + 32 * n + 64)

        def _parse(raw: str) -> List[str] | None:
            # Prefer the strict 'k.' / 'k)' numbered parser (matches the format
            # we asked for); fall back to the robust tagged/plain parser.
            return self._parse_numbered_output(raw, n) or self._parse_page_output(raw, n)

        parsed: List[str] | None = None
        try:
            raw = await self._chat(
                _msgs(user_src),
                max_tokens=budget,
                temperature=0.0,
            )
            parsed = _parse(raw)
        except Exception as e:
            logger.warning(f"vLLM translate_numbered_block failed: {e!r}")
            parsed = None

        # Bounded retry: ONE stricter attempt before any isolation fallback.
        if parsed is None:
            logger.info(
                "Page-level output did not parse to %d lines; retrying once strict", n
            )
            strict = (
                f"Output EXACTLY {n} lines, numbered 1. to {n}., one "
                f"{target_language} translation per line, in order, nothing "
                f"else — no preamble, no blank lines, no extra lines.\n\n{body}"
            )
            try:
                raw = await self._chat(
                    _msgs(strict),
                    max_tokens=budget,
                    temperature=0.0,
                )
                parsed = _parse(raw)
            except Exception as e:
                logger.warning(f"vLLM translate_numbered_block retry failed: {e!r}")
                parsed = None

        if parsed is None:
            logger.warning(
                "Page-level output still unparseable after retry for %d lines; "
                "caller will fall back per-bubble",
                n,
            )
            return []

        cleaned = [clean_translation_output(p) for p in parsed]

        # NEVER let a non-empty source bubble collapse to "..." or "" just
        # because the batch dropped/blanked its line. Individually translate
        # only the gaps so page context holds for the rest of the page.
        gaps = [
            i
            for i, (src, out) in enumerate(zip(texts, cleaned))
            if src.strip() and (not out.strip() or out.strip() == "...")
        ]
        if gaps:
            logger.info("Page-level: filling %d empty/ellipsis gap(s) individually", len(gaps))
            fills = await asyncio.gather(
                *(self.translate_single(texts[i], target_language) for i in gaps)
            )
            for i, fill in zip(gaps, fills):
                if fill.strip():
                    cleaned[i] = fill
        return cleaned

    @staticmethod
    def _parse_page_output(raw: str, n: int) -> List[str] | None:
        """Parse a page-level translation response into n ordered lines.

        Robust, in priority order:
          1. `[N]`-tagged output -> align by tag index (handles reorder, blanks,
             and a few missing tags: the gap stays "" and is filled per-bubble
             by the caller, never rendered as "...").
          2. Plain one-line-per-item output (what the v10it fine-tune emits):
             strip blank lines; if a short preamble made it n+1 lines, drop the
             leading non-translation line; accept when the count reconciles to n.

        Returns None ONLY when the count genuinely cannot be reconciled, so the
        caller's per-bubble isolation fallback is a true last resort.
        """
        # 1) Tagged path. Accept when at least half the tags are present; the
        #    blocks list is length-n with "" for any missing tag.
        tagged = parse_tagged_blocks(raw, n)
        if tagged is not None:
            present = sum(1 for p in tagged if p.strip())
            if present == n or present >= max(1, (n + 1) // 2):
                return tagged

        # 2) Plain-line path. Strip blanks; tolerate a single preamble/trailer.
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        # Strip stray `[N]` prefixes + degenerate repetition tails (the small
        # model loops `||||...`/repeated chars on garbled SFX, which both eats
        # the budget — truncating later lines — and corrupts the looped line).
        lines = [_strip_runaway_repeat(_strip_leading_index(ln)) for ln in lines]
        lines = [ln for ln in lines if ln.strip()]
        if len(lines) == n:
            return lines
        if len(lines) == n + 1:
            # One extra line is almost always a leading preamble ("Sure! ...")
            # or a trailing note. Prefer dropping the leading one.
            head, *rest = lines
            if _looks_like_preamble(head):
                return rest
            return lines[:n]
        if n < len(lines) <= n + 2:
            # A couple of extra chatter lines: keep the last n (translations
            # follow any preamble). Conservative best-effort.
            return lines[-n:]
        # TRUNCATION SALVAGE: the model emitted the first K (<n) lines then a
        # repetition loop ate the budget before the tail. Keep the K we have
        # (page context preserved) and pad to n with "" — the caller fills the
        # missing tail bubbles INDIVIDUALLY rather than dropping the whole page
        # to per-bubble isolation. Only salvage when we got a real majority so a
        # genuinely broken response still returns None.
        if 0 < len(lines) < n and len(lines) >= max(1, (n + 1) // 2):
            return lines + [""] * (n - len(lines))
        return None

    @staticmethod
    def _parse_numbered_output(raw: str, n: int) -> List[str] | None:
        """Parse 'k. text' lines back into an ordered list of n items.

        Aligns by the emitted number (handles reorder / stray blank lines) and
        strips any runaway repetition tail. Salvages a MAJORITY response (gaps
        left "" for the caller to fill individually); returns None only when
        too few numbered lines are recovered. Tolerates 'k)' / 'k.' separators.
        """
        import re

        out: dict[int, str] = {}
        pat = re.compile(r"^\s*(\d+)[.)]\s*(.*)$")
        for line in raw.splitlines():
            m = pat.match(line)
            if not m:
                continue
            k = int(m.group(1))
            if 1 <= k <= n:
                out[k] = _strip_runaway_repeat(m.group(2).strip())
        present = sum(1 for v in out.values() if v.strip())
        if present == 0 or present < max(1, (n + 1) // 2):
            return None
        return [out.get(i + 1, "") for i in range(n)]

    async def warmup(self) -> dict:
        t0 = time.perf_counter()
        try:
            await self.translate_single("テスト", "English")
        except Exception as e:
            logger.warning(f"vLLM warmup failed: {e!r}")
        return {"warmup_ms": (time.perf_counter() - t0) * 1000}

    @property
    def num_instances(self) -> int:
        return 1
