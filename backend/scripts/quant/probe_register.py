"""Probe a vLLM server with register-sensitive / NSFW / SFX JP lines to check for
INT4 register drift vs bf16. Greedy (temperature=0) for determinism.

Usage: probe_register.py --base-url http://127.0.0.1:8002/v1 --label int4
"""
from __future__ import annotations

import argparse
import json
import sys

import requests

# v11 plain prompt format (matches bench 'eval' style)
TMPL = ("Translate the following Japanese to English. Output only the translation.\n\n"
        "Japanese: {jp}")

PROBES = [
    # (id, jp) — register-heavy VN dialogue, casual/rude, NSFW-leaning, SFX
    ("undertaker_isolated", "葬儀屋さん"),
    ("rude_male", "うるせえ、黙って言うこと聞きやがれ"),
    ("polite_fem", "あの……よろしければ、お茶でもいかがですか？"),
    ("nsfw_moan", "んっ……ああっ、そこ、ダメぇ……"),
    ("nsfw_explicit", "もっと奥まで挿れて、はやく……"),
    ("nsfw_crude", "こんなにびしょびしょに濡らしやがって、このド変態が"),
    ("sfx_dokun", "ドクン"),
    ("sfx_gusha", "グシャッ"),
    ("sfx_zawa", "ザワザワ"),
    ("emph_refuse", "絶対に許さない……お前だけは"),
    ("childish", "やだやだ！ぼく、おうちかえるーっ！"),
    ("archaic", "拙者、これにて御免つかまつる"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default="v10it")
    ap.add_argument("--label", default="sys")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results = []
    for pid, jp in PROBES:
        body = {
            "model": args.model,
            "messages": [{"role": "user", "content": TMPL.format(jp=jp)}],
            "temperature": 0.0,
            "max_tokens": 64,
        }
        r = requests.post(f"{args.base_url.rstrip('/')}/chat/completions",
                          json=body, timeout=60)
        r.raise_for_status()
        txt = r.json()["choices"][0]["message"]["content"].strip()
        results.append({"id": pid, "jp": jp, "pred": txt})
        print(f"[{args.label}] {pid}: {jp}\n   -> {txt!r}\n")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
