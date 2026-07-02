"""POV eval harness — gate signal #1 (pronoun / point-of-view resolution).

Two subcommands over the A/B testset (backend/.bench/pov_ab/testset_large.json):

    generate  Run the 4 arms {v1, qwen3vl_ablit8b} x {image off, image on} against
              the live vLLM box and write per-arm prediction JSON.
    score     Score every arm + the free ``our_en_v11fix8`` baseline column with a
              PRESENCE-of-correct-family pronoun probe and emit a comparison table.

Also:

    --selftest  Run synthetic-row unit checks for the scorer (no network, no pytest;
                matches probes.py conventions).

Byte-match note
---------------
v1 was trained on marked-line manga-page rows whose user text is produced by
``build_context_prompt`` (build_v11_dataset.py:109) — the RAW builder, i.e. the
target line is verbatim, NOT run through ``normalize_short_utterance``.  We reuse
``build_v11_context_prompt`` (vllm_openai_translation_service.py:282) but force
``settings.short_utterance_normalize_enabled = False`` so the rendered serve prompt
byte-matches the training rows (verified 29467/29467 manga-page rows in
data_v13ship_v1_messages.jsonl).  ``translation_cast_anchor`` is likewise forced off.

CLI examples
------------
    # smoke: 3 rows, v1 + qwen, image on, against the live box
    python -m backend.scripts.eval.pov_probe generate \
        --arms v1:on qwen3vl_ablit8b:on --limit 3

    # full 4-arm run
    python -m backend.scripts.eval.pov_probe generate --arms all

    # score whatever is on disk
    python -m backend.scripts.eval.pov_probe score

    # scorer self-test
    python -m backend.scripts.eval.pov_probe --selftest
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Paths / constants
# --------------------------------------------------------------------------- #

_HERE = Path(__file__).resolve()
_BACKEND_ROOT = _HERE.parents[2]  # .../backend
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

TESTSET_PATH = _BACKEND_ROOT / ".bench" / "pov_ab" / "testset_large.json"
OUT_DIR = _BACKEND_ROOT / "scripts" / "eval" / "out" / "pov"

VLLM_BASE_URL = "http://100.64.235.63:8001/v1"
BASE_MODEL = "qwen3vl_ablit8b"
LORA_MODEL = "v1"

FURUBE_37 = {"furube_p1", "furube_p2", "furube_p3"}  # 26 + 5 + 6 = 37 rows

MAX_IMAGE_LONG_SIDE = 1024
DEFAULT_CONCURRENCY = 4
DEFAULT_MAX_TOKENS = 256
DEFAULT_RETRIES = 3
REQUEST_TIMEOUT_S = 180.0

_MIME_BY_SUFFIX = {
    ".webp": "image/webp",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}

# --------------------------------------------------------------------------- #
# Pronoun family detection  (PRESENCE semantics — see scorer docstring)
# --------------------------------------------------------------------------- #
# Same word-boundary regex shape as probes.py:86-87 (_HE_RE / _SHE_RE) so "the"
# never trips "he".  These detect the *presence* of a family, not its absence.
_HE_RE = re.compile(r"\b(he|him|his|himself)\b", re.IGNORECASE)
_SHE_RE = re.compile(r"\b(she|her|hers|herself)\b", re.IGNORECASE)


def detect_families(text: str) -> set[str]:
    """Return the set of gendered pronoun families present in ``text``.

    A subset of {"he", "she"}.  Empty set == no gendered pronoun present.
    """
    fams: set[str] = set()
    if _HE_RE.search(text or ""):
        fams.add("he")
    if _SHE_RE.search(text or ""):
        fams.add("she")
    return fams


def required_family(human_en: str) -> str | None:
    """The gender family a correct translation MUST contain, per the human ref.

    A row is "gendered-resolvable" iff ``human_en`` contains exactly one family.
    If the human reference is ungendered (0 families) or mixed (both he AND she),
    there is no single required family -> return None (row is not resolvable).
    """
    fams = detect_families(human_en)
    if len(fams) == 1:
        return next(iter(fams))
    return None


def pov_pass(prediction: str, req_fam: str) -> bool:
    """Pass == PRESENCE of the required family AND ABSENCE of the opposing one.

    This is the deliberate inversion of probes.check_pronoun_gender, whose
    absence-of-wrong logic lets a pronoun-evading baseline score ~100% falsely.
    Here an evasive prediction (no gendered pronoun) FAILS a gendered row.
    """
    fams = detect_families(prediction)
    opposing = "she" if req_fam == "he" else "he"
    return (req_fam in fams) and (opposing not in fams)


# --------------------------------------------------------------------------- #
# Prompt builder (byte-matches v1 training rows — see module docstring)
# --------------------------------------------------------------------------- #


def _get_prompt_builder():
    """Import & configure ``build_v11_context_prompt`` for byte-exact serve prompts.

    Forces normalization + cast-anchor OFF so the rendered text byte-matches the
    raw marked-line training rows (build_v11_dataset.build_context_prompt).
    """
    from app.config import settings
    from app.services.vllm_openai_translation_service import build_v11_context_prompt

    settings.short_utterance_normalize_enabled = False
    settings.translation_cast_anchor = False
    return build_v11_context_prompt


def build_marked_line_prompt(row: dict[str, Any], builder) -> str:
    return builder(row["context"], row["target_idx"])


# --------------------------------------------------------------------------- #
# Image encoding
# --------------------------------------------------------------------------- #


def image_to_data_url(image_path: Path, *, max_long_side: int = MAX_IMAGE_LONG_SIDE) -> str:
    """Downscale to <= ``max_long_side`` on the long edge, then base64 data-URL.

    Mirrors transcribe_gt_vision._image_to_data_url / ocr_adapters, but adds a
    PIL downscale to bound upload + prefill cost.  Re-encodes as JPEG (quality 90)
    unless the source is PNG (kept lossless).
    """
    from PIL import Image

    with Image.open(image_path) as im:
        im = im.convert("RGB") if im.mode not in ("RGB", "L") else im
        w, h = im.size
        long_side = max(w, h)
        if long_side > max_long_side:
            scale = max_long_side / float(long_side)
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        buf = io.BytesIO()
        suffix = image_path.suffix.lower()
        if suffix == ".png":
            im.save(buf, format="PNG")
            mime = "image/png"
        else:
            im.save(buf, format="JPEG", quality=90)
            mime = "image/jpeg"
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:{mime};base64,{b64}"


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #


@dataclass
class Arm:
    model: str
    image: bool  # True == image-on

    @property
    def key(self) -> str:
        return f"{self.model}__img-{'on' if self.image else 'off'}"

    @property
    def out_path(self) -> Path:
        return OUT_DIR / f"{self.key}.json"


def parse_arms(tokens: list[str]) -> list[Arm]:
    """Parse ``--arms`` tokens.

    Accepts ``all`` (the 4-arm cross product) or ``model:on|off`` tokens, e.g.
    ``v1:on qwen3vl_ablit8b:off``.
    """
    if tokens == ["all"] or tokens == ["ALL"]:
        return [
            Arm(LORA_MODEL, False),
            Arm(LORA_MODEL, True),
            Arm(BASE_MODEL, False),
            Arm(BASE_MODEL, True),
        ]
    arms: list[Arm] = []
    for tok in tokens:
        if ":" not in tok:
            raise SystemExit(f"bad --arms token {tok!r} (want model:on|off or 'all')")
        model, img = tok.rsplit(":", 1)
        img = img.strip().lower()
        if img not in ("on", "off"):
            raise SystemExit(f"bad image flag in {tok!r} (want on|off)")
        arms.append(Arm(model, img == "on"))
    return arms


def load_testset(furube_only: bool = False) -> list[dict[str, Any]]:
    rows = json.loads(TESTSET_PATH.read_text(encoding="utf-8"))
    if furube_only:
        rows = [r for r in rows if r["work"] in FURUBE_37]
    return rows


def _load_existing(path: Path) -> dict[str, dict]:
    """Load prior per-arm predictions keyed by src (resumable/mergeable)."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return {rec["src"]: rec for rec in data.get("predictions", [])}


async def _one_request(
    client,
    *,
    arm: Arm,
    prompt: str,
    data_url: str | None,
    max_tokens: int,
    retries: int,
) -> tuple[str | None, str | None]:
    """Return (prediction, error).  Deterministic (temperature 0)."""
    content: Any
    if data_url is not None:
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
    else:
        content = prompt
    payload = {
        "model": arm.model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    last_err: str | None = None
    for attempt in range(retries):
        try:
            resp = await client.post(
                f"{VLLM_BASE_URL}/chat/completions",
                json=payload,
                timeout=REQUEST_TIMEOUT_S,
            )
            if resp.status_code >= 500:
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                await asyncio.sleep(1.5 * (attempt + 1))
                continue
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            return (text.strip() if text else ""), None
        except Exception as e:  # noqa: BLE001 — record raw error, then retry/surface
            last_err = f"{type(e).__name__}: {e}"
            await asyncio.sleep(1.5 * (attempt + 1))
    return None, last_err


async def generate_arm(
    arm: Arm,
    rows: list[dict[str, Any]],
    *,
    concurrency: int,
    max_tokens: int,
    retries: int,
    limit: int | None,
    resume: bool,
) -> Path:
    import httpx

    builder = _get_prompt_builder()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if limit is not None:
        rows = rows[:limit]

    existing = _load_existing(arm.out_path) if resume else {}

    # Pre-encode images once per unique page (image-on only).
    img_cache: dict[str, str] = {}

    todo = [r for r in rows if r["src"] not in existing]
    sem = asyncio.Semaphore(concurrency)
    results: dict[str, dict] = dict(existing)

    print(
        f"[{arm.key}] rows={len(rows)} todo={len(todo)} "
        f"(resume-skipped={len(rows) - len(todo)})"
    )

    async with httpx.AsyncClient() as client:

        async def worker(row: dict[str, Any]) -> None:
            async with sem:
                prompt = build_marked_line_prompt(row, builder)
                data_url = None
                if arm.image:
                    pimg = row["page_img"]
                    if pimg not in img_cache:
                        img_cache[pimg] = image_to_data_url(Path(pimg))
                    data_url = img_cache[pimg]
                pred, err = await _one_request(
                    client,
                    arm=arm,
                    prompt=prompt,
                    data_url=data_url,
                    max_tokens=max_tokens,
                    retries=retries,
                )
                results[row["src"]] = {
                    "src": row["src"],
                    "work": row["work"],
                    "jp": row["jp"],
                    "human_en": row["human_en"],
                    "our_en_v11fix8": row["our_en_v11fix8"],
                    "prediction": pred,
                    "error": err,
                    "prompt_chars": len(prompt),
                    "image": arm.image,
                }
                if err:
                    print(f"  ! {row['src']}: {err}")

        await asyncio.gather(*(worker(r) for r in todo))

    # Preserve testset order.
    order = {r["src"]: i for i, r in enumerate(load_testset())}
    ordered = sorted(results.values(), key=lambda rec: order.get(rec["src"], 1 << 30))
    n_ok = sum(1 for r in ordered if r.get("prediction") is not None)
    payload = {
        "arm": arm.key,
        "model": arm.model,
        "image": arm.image,
        "n": len(ordered),
        "n_ok": n_ok,
        "predictions": ordered,
    }
    arm.out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[{arm.key}] wrote {arm.out_path}  (ok={n_ok}/{len(ordered)})")
    return arm.out_path


async def cmd_generate(args: argparse.Namespace) -> int:
    arms = parse_arms(args.arms)
    rows = load_testset(furube_only=args.furube_only)
    for arm in arms:
        await generate_arm(
            arm,
            rows,
            concurrency=args.concurrency,
            max_tokens=args.max_tokens,
            retries=args.retries,
            limit=args.limit,
            resume=not args.no_resume,
        )
    return 0


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


@dataclass
class ArmScore:
    label: str
    # gendered-only
    gendered_n: int = 0
    gendered_pass: int = 0
    # all-rows (ungendered pass-by-default)
    all_n: int = 0
    all_pass: int = 0
    # flips / gains vs baseline
    flips: int = 0  # baseline had WRONG family, candidate correct (strict he<->she)
    gains: int = 0  # baseline had NO pronoun, candidate correct
    errors: int = 0

    @property
    def gendered_rate(self) -> float:
        return self.gendered_pass / self.gendered_n if self.gendered_n else 0.0

    @property
    def all_rate(self) -> float:
        return self.all_pass / self.all_n if self.all_n else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "gendered_n": self.gendered_n,
            "gendered_pass": self.gendered_pass,
            "gendered_rate": round(self.gendered_rate, 4),
            "all_n": self.all_n,
            "all_pass": self.all_pass,
            "all_rate": round(self.all_rate, 4),
            "flips_he_she": self.flips,
            "gains_from_evasion": self.gains,
            "errors": self.errors,
        }


def _classify_baseline(baseline_en: str, req_fam: str) -> str:
    """Baseline family status for a gendered row: 'correct' | 'wrong' | 'none'."""
    fams = detect_families(baseline_en)
    if not fams:
        return "none"
    opposing = "she" if req_fam == "he" else "he"
    if req_fam in fams and opposing not in fams:
        return "correct"
    return "wrong"  # has opposing (possibly both) — a POV error


def score_rows(
    rows: list[dict[str, Any]],
    *,
    label: str,
    pred_key: str,
    baseline_key: str | None = "our_en_v11fix8",
) -> ArmScore:
    """Score a set of prediction rows.

    ``pred_key`` selects the candidate text field.  ``baseline_key`` (if given and
    != pred_key) drives flip/gain accounting relative to the v11fix8 column.
    """
    s = ArmScore(label=label)
    for row in rows:
        pred = row.get(pred_key)
        if pred is None:
            s.errors += 1
            continue
        req = required_family(row["human_en"])
        s.all_n += 1
        if req is None:
            # ungendered row: pass-by-default in the all-rows denominator
            s.all_pass += 1
            continue
        # gendered-resolvable row
        s.gendered_n += 1
        ok = pov_pass(pred, req)
        if ok:
            s.gendered_pass += 1
            s.all_pass += 1
        # flip / gain vs baseline
        if baseline_key and baseline_key != pred_key and ok:
            base_status = _classify_baseline(row.get(baseline_key, ""), req)
            if base_status == "wrong":
                s.flips += 1
            elif base_status == "none":
                s.gains += 1
    return s


def _md_table(scores: list[ArmScore], title: str) -> str:
    hdr = (
        "| arm | gendered pass | gendered rate | all rate | flips (he↔she) | "
        "gains | errors |"
    )
    sep = "|---|---|---|---|---|---|---|"
    lines = [f"### {title}", "", hdr, sep]
    for s in scores:
        lines.append(
            f"| {s.label} | {s.gendered_pass}/{s.gendered_n} | "
            f"{s.gendered_rate*100:.1f}% | {s.all_rate*100:.1f}% | "
            f"{s.flips} | {s.gains} | {s.errors} |"
        )
    return "\n".join(lines)


def _merge_predictions_by_src(arm_files: list[Path]) -> dict[str, dict[str, dict]]:
    """Return {arm_key: {src: record}} for each on-disk arm file."""
    out: dict[str, dict[str, dict]] = {}
    for p in arm_files:
        data = json.loads(p.read_text(encoding="utf-8"))
        out[data["arm"]] = {rec["src"]: rec for rec in data["predictions"]}
    return out


def cmd_score(args: argparse.Namespace) -> int:
    arm_files = sorted(OUT_DIR.glob("*.json"))
    arm_files = [p for p in arm_files if p.name != "score_report.json"]
    if not arm_files:
        print(f"no arm prediction files in {OUT_DIR}", file=sys.stderr)
        return 2

    by_arm = _merge_predictions_by_src(arm_files)
    testset = load_testset()
    by_src = {r["src"]: r for r in testset}

    def build_scope(furube: bool) -> tuple[list[ArmScore], dict]:
        scope_srcs = [
            r["src"] for r in testset if (not furube or r["work"] in FURUBE_37)
        ]
        scores: list[ArmScore] = []

        # Free baseline column (our_en_v11fix8) — scored on the testset rows directly.
        base_rows = [by_src[s] for s in scope_srcs]
        scores.append(
            score_rows(
                base_rows,
                label="our_en_v11fix8 (baseline col)",
                pred_key="our_en_v11fix8",
                baseline_key=None,
            )
        )

        # Each arm.
        for arm_key, recs in sorted(by_arm.items()):
            rows = [recs[s] for s in scope_srcs if s in recs]
            scores.append(
                score_rows(
                    rows,
                    label=arm_key,
                    pred_key="prediction",
                    baseline_key="our_en_v11fix8",
                )
            )
        meta = {"scope_row_count": len(scope_srcs)}
        return scores, meta

    furube_scores, furube_meta = build_scope(furube=True)
    all_scores, all_meta = build_scope(furube=False)

    md = "\n\n".join(
        [
            _md_table(furube_scores, f"Furube-37 (n={furube_meta['scope_row_count']})"),
            _md_table(all_scores, f"All-148 (n={all_meta['scope_row_count']})"),
        ]
    )
    print(md)

    report = {
        "arms_scored": sorted(by_arm.keys()),
        "furube37": {
            "scope_row_count": furube_meta["scope_row_count"],
            "scores": [s.to_dict() for s in furube_scores],
        },
        "all148": {
            "scope_row_count": all_meta["scope_row_count"],
            "scores": [s.to_dict() for s in all_scores],
        },
        "gate_note": (
            "≥48% gate is measured on the gendered-only denominator; all_rate "
            "counts ungendered rows as pass-by-default and is reported for context."
        ),
    }
    out_path = OUT_DIR / "score_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")
    return 0


# --------------------------------------------------------------------------- #
# Scorer self-test  (synthetic rows; runnable via --selftest, no pytest)
# --------------------------------------------------------------------------- #


def _selftest() -> int:
    failures: list[str] = []

    def check(name: str, cond: bool) -> None:
        status = "ok " if cond else "FAIL"
        print(f"  [{status}] {name}")
        if not cond:
            failures.append(name)

    # --- family detection ---
    check("detect he", detect_families("He went home.") == {"he"})
    check("detect she", detect_families("She left, it's hers.") == {"she"})
    check("detect both", detect_families("He gave her his book.") == {"he", "she"})
    check("detect none", detect_families("The cat sat on the mat.") == set())
    check("word-boundary: 'the' not 'he'", detect_families("the theater") == set())
    check("word-boundary: 'sherpa' not 'she'", detect_families("A sherpa climbed.") == set())
    check("himself counts as he", detect_families("He hurt himself.") == {"he"})

    # --- required family (resolvability) ---
    check("req: single she", required_family("She loves her son.") == "she")
    check("req: single he", required_family("He is his own boss.") == "he")
    check("req: mixed -> None", required_family("He told her the news.") is None)
    check("req: ungendered -> None", required_family("The dog barked.") is None)

    # --- pov_pass: PRESENCE of correct + ABSENCE of opposing ---
    # correct family present -> pass
    check("pass: correct she present", pov_pass("She did it.", "she") is True)
    # WRONG family (the classic v11 POV inversion) -> fail
    check("fail: wrong family (he for she)", pov_pass("He did it.", "she") is False)
    # EVASION (no pronoun) -> FAIL under presence semantics (the whole point)
    check(
        "fail: evasion scores as fail (not pass)",
        pov_pass("The person did it.", "she") is False,
    )
    # mixed (both families) -> fail because opposing present
    check("fail: both families present", pov_pass("He gave her it.", "she") is False)

    # --- scorer accounting: flips vs gains vs pass-by-default ---
    synth = [
        # gendered row: baseline WRONG (he), candidate correct (she) -> FLIP + pass
        {
            "src": "s1",
            "human_en": "She hugged her child.",
            "our_en_v11fix8": "He hugged his child.",
            "prediction": "She hugged her child.",
        },
        # gendered row: baseline EVADES (no pronoun), candidate correct -> GAIN + pass
        {
            "src": "s2",
            "human_en": "He fixed his bike.",
            "our_en_v11fix8": "The bike got fixed.",
            "prediction": "He fixed his bike.",
        },
        # gendered row: candidate WRONG -> fail, no flip/gain
        {
            "src": "s3",
            "human_en": "She smiled.",
            "our_en_v11fix8": "She smiled.",
            "prediction": "He smiled.",
        },
        # ungendered row: pass-by-default in all_rate, not counted in gendered
        {
            "src": "s4",
            "human_en": "The rain fell.",
            "our_en_v11fix8": "It rained.",
            "prediction": "Rain fell down.",
        },
    ]
    sc = score_rows(synth, label="synthetic", pred_key="prediction")
    check("scorer: gendered_n == 3", sc.gendered_n == 3)
    check("scorer: gendered_pass == 2", sc.gendered_pass == 2)
    check("scorer: all_n == 4", sc.all_n == 4)
    check("scorer: all_pass == 3 (2 gendered + 1 ungendered default)", sc.all_pass == 3)
    check("scorer: flips == 1 (s1)", sc.flips == 1)
    check("scorer: gains == 1 (s2)", sc.gains == 1)

    # baseline column self-scores with no flips/gains
    sc_base = score_rows(
        synth, label="baseline", pred_key="our_en_v11fix8", baseline_key=None
    )
    check("baseline: gendered_pass == 1 (only s3 baseline correct)", sc_base.gendered_pass == 1)
    check("baseline: flips == 0", sc_base.flips == 0)
    check("baseline: gains == 0", sc_base.gains == 0)

    print()
    if failures:
        print(f"SELFTEST FAILED: {len(failures)} check(s): {failures}")
        return 1
    print("SELFTEST PASSED (all checks green)")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="POV eval harness (gate signal #1).")
    p.add_argument("--selftest", action="store_true", help="Run scorer self-test and exit.")
    sub = p.add_subparsers(dest="cmd")

    g = sub.add_parser("generate", help="Run arms against the live vLLM box.")
    g.add_argument(
        "--arms",
        nargs="+",
        default=["all"],
        help="'all' or model:on|off tokens, e.g. v1:on qwen3vl_ablit8b:off",
    )
    g.add_argument("--limit", type=int, default=None, help="First N testset rows (smoke).")
    g.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    g.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, dest="max_tokens")
    g.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    g.add_argument("--furube-only", action="store_true", dest="furube_only")
    g.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing per-arm output (recompute all rows).",
    )

    sub.add_parser("score", help="Score on-disk arm outputs + baseline column.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.selftest:
        return _selftest()
    if args.cmd == "generate":
        return asyncio.run(cmd_generate(args))
    if args.cmd == "score":
        return cmd_score(args)
    _build_parser().print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
