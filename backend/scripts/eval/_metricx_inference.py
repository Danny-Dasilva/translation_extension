"""Minimal MetricX-24 PyTorch inference (mirrors google-research/metricx).

Reproduces the public ``MT5ForRegression`` class and prompt format so we can
score JP→EN translations without depending on the (T5X / JAX) reference code.

Score interpretation: error in [0, 25]. **Lower is better.**

Implementation strategy: subclass ``MT5ForConditionalGeneration`` and reuse its
forward path so the encoder/decoder attention-mask preparation is correct on
modern (4.5x+) transformers. We then override the score extraction to read off
the trained regression head at vocab id 250089 (=<extra_id_10>), exactly like
google-research/metricx's MT5ForRegression.
"""

from __future__ import annotations

import torch
from transformers import MT5ForConditionalGeneration


class MetricXForRegression(MT5ForConditionalGeneration):
    """Same MT5 weights, different score extraction.

    The HF MT5ForConditionalGeneration class handles all the encoder/decoder
    attention-mask shape gymnastics correctly via _get_extended_attention_mask
    + the cross-attention's own mask — we just reuse it.
    """

    @torch.inference_mode()
    def score(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor) -> torch.Tensor:
        batch = input_ids.size(0)
        # Single-step decoder input: pad token (id 0) — same as official metricx.
        decoder_input_ids = torch.zeros(
            (batch, 1), dtype=torch.long, device=input_ids.device
        )
        out = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
            return_dict=True,
        )
        logits = out.logits  # (batch, 1, vocab)
        # 250089 == <extra_id_10>: the regression head slot.
        scores = logits[:, 0, 250089]
        return torch.clamp(scores, 0.0, 25.0)


def score_metricx(
    triples: list[tuple[str, str, str]],  # (src, hyp, ref)
    *,
    model_id: str = "google/metricx-24-hybrid-xl-v2p6",
    tokenizer_id: str = "google/mt5-xl",
    qe_mode: bool = False,
    batch_size: int = 8,
    max_input_length: int = 1536,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> list[float]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=False, legacy=False)
    model = MetricXForRegression.from_pretrained(model_id, torch_dtype=dtype)
    model = model.to(device).eval()

    def fmt(src: str, hyp: str, ref: str) -> str:
        if qe_mode:
            return f"source: {src} candidate: {hyp}"
        return f"source: {src} candidate: {hyp} reference: {ref}"

    out: list[float] = []
    for i in range(0, len(triples), batch_size):
        batch = triples[i : i + batch_size]
        texts = [fmt(*t) for t in batch]
        # Tokenize each example individually so we can drop the trailing EOS
        # (per the official `_remove_eos` step) before padding.
        ids_list: list[list[int]] = []
        for t in texts:
            e = tok(
                t,
                truncation=True,
                max_length=max_input_length,
                add_special_tokens=True,
            )
            seq = e["input_ids"]
            if seq and seq[-1] == tok.eos_token_id:
                seq = seq[:-1]
            ids_list.append(seq)
        max_len = max(len(s) for s in ids_list)
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
        padded = [s + [pad_id] * (max_len - len(s)) for s in ids_list]
        am_padded = [[1] * len(s) + [0] * (max_len - len(s)) for s in ids_list]
        ids = torch.tensor(padded, dtype=torch.long, device=device)
        am = torch.tensor(am_padded, dtype=torch.long, device=device)
        scores = model.score(ids, am)
        out.extend(scores.float().cpu().tolist())
    del model
    torch.cuda.empty_cache()
    return out
