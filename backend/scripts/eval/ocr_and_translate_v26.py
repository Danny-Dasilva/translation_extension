"""Consume v26 detection JSONs from /home/danny/manga-output/644289-v26-detect/,
OCR each block via PARSeq, translate via v7, save two output dirs:

  /home/danny/manga-output/644289-v26-with-sfx/<slug>/
  /home/danny/manga-output/644289-v26-no-sfx/<slug>/

Each contains stats.json (compatible with refit_final_composites + render_manga_finals)
and translations.txt.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

BACKEND = Path(__file__).resolve().parents[2]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.config import settings  # noqa: E402
from app.services.parseq_ocr_service import ParseqOCRService  # noqa: E402
from app.utils.japanese_text_filter import is_japanese_text  # noqa: E402

V26 = Path("/home/danny/manga-output/644289-v26-detect")
ORIG = Path("/home/danny/manga-output/644289/originals")

PROMPT_TEMPLATE = (
    "Translate the following Japanese to English. "
    "Output only the translation.\n\nJapanese: {jp}\nEnglish:"
)


async def ocr_pages(ocr: ParseqOCRService, batch_size: int):
    """Yield (slug, [{bbox, lang, type, jp}]) per page after OCR."""
    pages = []
    slugs = sorted(p.name for p in V26.iterdir() if p.is_dir())
    for slug in slugs:
        det_p = V26 / slug / "v26_detect.json"
        if not det_p.exists():
            continue
        with open(det_p) as f:
            d = json.load(f)
        img_path = Path(d["image"])
        if not img_path.exists():
            # try originals dir
            img_path = ORIG / f"{slug}.webp"
            if not img_path.exists():
                img_path = ORIG / f"{slug}.jpg"
        if not img_path.exists():
            continue
        img = np.array(Image.open(img_path).convert("RGB"))
        h, w = img.shape[:2]
        crops = []
        valid_blocks = []
        for b in d["blocks"]:
            x0, y0, x1, y1 = b["bbox"]
            x0 = max(0, x0); y0 = max(0, y0); x1 = min(w, x1); y1 = min(h, y1)
            if x1 <= x0 or y1 <= y0:
                continue
            crops.append(img[y0:y1, x0:x1])
            valid_blocks.append(b)
        ocr_texts = await ocr.recognize_text_batch(crops, batch_size=batch_size) if crops else []
        for b, t in zip(valid_blocks, ocr_texts):
            b["jp"] = (t or "").strip()
        pages.append((slug, valid_blocks, str(img_path)))
    return pages


def translate_batch(model, tok, jp_list: list[str], device: str, batch_size: int = 32) -> list[str]:
    if not jp_list:
        return []
    from transformers import StoppingCriteria, StoppingCriteriaList

    class StopOnSubstring(StoppingCriteria):
        def __init__(self, stop_ids: list[list[int]], start_len: int):
            self.stop_ids = stop_ids
            self.start = start_len
        def __call__(self, input_ids, scores, **kw):
            for b in range(input_ids.shape[0]):
                gen = input_ids[b, self.start:].tolist()
                if not any(self._has(gen, s) for s in self.stop_ids):
                    return False
            return True
        @staticmethod
        def _has(gen, s):
            if not s or len(gen) < len(s):
                return False
            for i in range(len(gen) - len(s) + 1):
                if gen[i:i+len(s)] == s:
                    return True
            return False

    stop_strings = ["\n\n", "\nJapanese:", "\nEnglish:"]
    stop_ids = [tok(s, add_special_tokens=False).input_ids for s in stop_strings]

    results = []
    for i in range(0, len(jp_list), batch_size):
        batch_jp = jp_list[i:i+batch_size]
        prompts = [PROMPT_TEMPLATE.format(jp=jp) for jp in batch_jp]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        start_len = enc["input_ids"].shape[1]
        stopping = StoppingCriteriaList([StopOnSubstring(stop_ids, start_len)])
        with torch.inference_mode():
            out = model.generate(
                **enc,
                do_sample=True,
                temperature=0.2, top_p=0.9, min_p=0.1,
                max_new_tokens=60,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
                stopping_criteria=stopping,
            )
        for j in range(out.shape[0]):
            gen = out[j, start_len:]
            results.append(tok.decode(gen, skip_special_tokens=True).strip())
    return results


def clean(text: str) -> str:
    """Same post-processor as translate_manga_batch.py."""
    import re
    if not text:
        return ""
    text = re.split(r"[\r\n]", text, 1)[0]
    text = re.sub(r"\s*(?:Japanese:|JP:|English:|EN:).*$", "", text, flags=re.S)
    text = re.sub(r"(?:\s*[\(\[][^\)\]]{0,25}[\)\]]){2,}\s*$", "", text)
    text = re.sub(r"(?:\s*[.!?\"'~_\-•・]+){4,}\s*$", "", text)
    text = re.sub(r"(?:\s*(\S))\s*(?:\1\s*){3,}$", "", text)
    text = re.sub(r"\b(\w{3,15}?)\1{3,}\w*\b", r"\1", text, flags=re.I)
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r"(\b[^.!?]{3,80}[.!?]+)\s*(?:\1\s*)+", r"\1 ", text, flags=re.I)
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r"(\b.{2,40}?\b)(?:\s*\1){1,}", r"\1", text)
    text = re.sub(r"\.{4,}", "...", text)
    text = re.sub(r"([!?])\1{3,}", r"\1\1\1", text)
    return text.strip()


def write_page(out_dir: Path, slug: str, image_path: str, blocks: list[dict], translations: list[str]):
    out_dir.mkdir(parents=True, exist_ok=True)
    cleaned = [clean(t) for t in translations]
    with open(out_dir / "translations.txt", "w", encoding="utf-8") as f:
        f.write(f"# {Path(image_path).name}\n# {len(blocks)} bubble(s)\n\n")
        for i, (b, en) in enumerate(zip(blocks, cleaned), 1):
            f.write(f"[{i}] type={b.get('type','?')}\n  JP: {b.get('jp','')}\n  EN: {en}\n\n")
    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump({
            "image": Path(image_path).name,
            "image_path": image_path,
            "num_bubbles": len(blocks),
            "ocr_samples": [b.get("jp", "") for b in blocks],
            "translations": cleaned,
            "translations_raw": translations,
            "blocks": blocks,
            "model": "qwen3-mt-v7",
        }, f, ensure_ascii=False, indent=2)


async def main_async(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("loading PARSeq OCR...")
    ocr = ParseqOCRService(model_path=settings.parseq_model_path)
    print("OCR'ing all pages...")
    pages = await ocr_pages(ocr, batch_size=settings.parseq_batch_size)
    print(f"OCR'd {len(pages)} pages")

    print(f"loading translator: {args.model}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map=device)
    model.eval()

    out_with = Path(args.out_with_sfx)
    out_no = Path(args.out_no_sfx)
    out_with.mkdir(parents=True, exist_ok=True)
    out_no.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for slug, blocks, img_path in pages:
        # Filter to JP-text-only blocks for both variants
        kept = [b for b in blocks if b.get("jp") and is_japanese_text(
            b["jp"],
            settings.japanese_filter_min_ratio,
            settings.japanese_filter_katakana_max_length
        )]
        # with-sfx: keep all kept blocks
        # no-sfx: drop blocks classified as 'sfx'
        with_sfx_blocks = kept
        no_sfx_blocks = [b for b in kept if b.get("type") != "sfx"]

        # Translate the union (any JP that appears in either variant) once
        all_jp = list({b["jp"] for b in with_sfx_blocks})
        if all_jp:
            ens = translate_batch(model, tok, all_jp, device, batch_size=32)
            jp_to_en = dict(zip(all_jp, ens))
        else:
            jp_to_en = {}

        with_sfx_ens = [jp_to_en.get(b["jp"], "") for b in with_sfx_blocks]
        no_sfx_ens = [jp_to_en.get(b["jp"], "") for b in no_sfx_blocks]

        write_page(out_with / slug, slug, img_path, with_sfx_blocks, with_sfx_ens)
        write_page(out_no / slug, slug, img_path, no_sfx_blocks, no_sfx_ens)
        print(f"  [{slug}] with_sfx={len(with_sfx_blocks)} no_sfx={len(no_sfx_blocks)}")

    print(f"DONE in {time.time()-t0:.1f}s. with_sfx -> {out_with}, no_sfx -> {out_no}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="backend/training/weights/qwen3-mt-v7-merged")
    ap.add_argument("--out-with-sfx", default="/home/danny/manga-output/644289-v26-with-sfx")
    ap.add_argument("--out-no-sfx", default="/home/danny/manga-output/644289-v26-no-sfx")
    args = ap.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
