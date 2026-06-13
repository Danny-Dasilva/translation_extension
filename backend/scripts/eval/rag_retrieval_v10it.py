"""BGE-M3 RAG index of training corpus JP→EN pairs for v10-it inference.

Build phase
-----------
    /home/danny/.venvs/vllm/bin/python rag_retrieval_v10it.py build \
        --corpus backend/training/datasets/unified/data_v10.parquet \
        --jp-col jp --en-col en \
        --out backend/scripts/eval/rag_index_v10 \
        [--max-rows 200000] [--manga-only]

Query phase (used by inference_v10it_quality.py via RAGIndex.load(...).topk(...))
-------------------------------------------------------------------------------
    idx = RAGIndex.load("backend/scripts/eval/rag_index_v10")
    pairs = idx.topk("えぇ!?", k=3)  # -> [(jp, en), ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


class RAGIndex:
    def __init__(self, jp_texts: list[str], en_texts: list[str], embeddings: np.ndarray, model_id: str):
        self.jp_texts = jp_texts
        self.en_texts = en_texts
        self.embeddings = embeddings.astype(np.float32)
        # Normalize for cosine via dot product
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        self.embeddings = self.embeddings / norms
        self.model_id = model_id
        self._encoder = None

    def size(self) -> int:
        return len(self.jp_texts)

    def _ensure_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.model_id, device="cuda")
        return self._encoder

    def topk(self, jp: str, k: int = 3) -> list[tuple[str, str]]:
        enc = self._ensure_encoder()
        q = enc.encode([jp], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        sims = (q @ self.embeddings.T)[0]
        # Block exact matches (edge case if query was in corpus)
        idxs = np.argsort(-sims)[: k * 4]
        out: list[tuple[str, str]] = []
        seen: set[str] = set()
        for i in idxs:
            jp_i = self.jp_texts[int(i)]
            if jp_i.strip() == jp.strip():
                continue
            if jp_i in seen:
                continue
            seen.add(jp_i)
            out.append((jp_i, self.en_texts[int(i)]))
            if len(out) >= k:
                break
        return out

    @classmethod
    def load(cls, path: str | Path) -> "RAGIndex":
        p = Path(path)
        meta = json.loads((p / "meta.json").read_text())
        jp = json.loads((p / "jp.json").read_text())
        en = json.loads((p / "en.json").read_text())
        emb = np.load(p / "embeddings.npy")
        return cls(jp, en, emb, meta["model_id"])

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        (p / "meta.json").write_text(json.dumps({"model_id": self.model_id, "n": len(self.jp_texts)}))
        (p / "jp.json").write_text(json.dumps(self.jp_texts, ensure_ascii=False))
        (p / "en.json").write_text(json.dumps(self.en_texts, ensure_ascii=False))
        np.save(p / "embeddings.npy", self.embeddings)


def cmd_build(args) -> int:
    import polars as pl
    from sentence_transformers import SentenceTransformer

    corpus = Path(args.corpus)
    if not corpus.exists():
        print(f"ERROR: corpus not found: {corpus}", file=sys.stderr)
        return 2

    df = pl.read_parquet(corpus)
    if args.manga_only and "register_tag" in df.columns:
        df = df.filter(pl.col("register_tag") == "manga")
    if args.gold_only and "gold_flag" in df.columns:
        df = df.filter(pl.col("gold_flag"))
    if args.max_rows and len(df) > args.max_rows:
        df = df.sample(args.max_rows, seed=42)

    jp = df[args.jp_col].to_list()
    en = df[args.en_col].to_list()
    pairs = [(j, e) for j, e in zip(jp, en) if j and e]
    jp_texts = [p[0] for p in pairs]
    en_texts = [p[1] for p in pairs]

    print(f"[build] encoding {len(jp_texts)} JP segments with {args.model_id}")
    model = SentenceTransformer(args.model_id, device="cuda")
    emb = model.encode(
        jp_texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=args.batch_size,
        show_progress_bar=True,
    ).astype(np.float32)

    idx = RAGIndex(jp_texts, en_texts, emb, args.model_id)
    idx.save(args.out)
    print(f"[build] wrote {idx.size()} pairs to {args.out}")
    return 0


def cmd_query(args) -> int:
    idx = RAGIndex.load(args.index)
    print(f"loaded {idx.size()} pairs from {args.index}")
    for q in args.queries:
        print(f"\n=== query: {q!r} ===")
        for jp, en in idx.topk(q, k=args.k):
            print(f"  JP: {jp[:60]!r}")
            print(f"  EN: {en[:60]!r}")
            print()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--corpus", required=True)
    b.add_argument("--jp-col", default="jp")
    b.add_argument("--en-col", default="en")
    b.add_argument("--out", required=True)
    b.add_argument("--model-id", default="BAAI/bge-m3")
    b.add_argument("--max-rows", type=int, default=200000)
    b.add_argument("--batch-size", type=int, default=64)
    b.add_argument("--manga-only", action="store_true")
    b.add_argument("--gold-only", action="store_true")
    b.set_defaults(fn=cmd_build)

    q = sub.add_parser("query")
    q.add_argument("--index", required=True)
    q.add_argument("--k", type=int, default=3)
    q.add_argument("queries", nargs="+")
    q.set_defaults(fn=cmd_query)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
