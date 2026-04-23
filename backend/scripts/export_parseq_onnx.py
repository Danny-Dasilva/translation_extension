"""Export the trained PARSeq-large manga OCR checkpoint to ONNX.

Loads parseq_manga_best_5p16.pt (model_size=large, NAR, refine_iters=1,
img_size=128x512, charset=4400 JP chars), builds the PARSeq model, strips the
torch.compile `_orig_mod.` prefix, wraps it in an export-friendly module that
returns (log-softmax-ready) logits, and exports to ONNX opset 17.

Run:
    backend/.venv/bin/python backend/scripts/export_parseq_onnx.py \
        --ckpt /home/danny/Documents/personal/manga/comic-text-detector-parseq-v2/output/parseq_large/parseq_manga_best_5p16.pt \
        --out  backend/models/parseq_manga_large_5p16.onnx
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

PARSEQ_REPO = Path("/home/danny/Documents/personal/manga/comic-text-detector-parseq-v2/parseq_repo")
sys.path.insert(0, str(PARSEQ_REPO))

from strhub.models.parseq.model import PARSeq  # noqa: E402
from strhub.data.utils import Tokenizer  # noqa: E402


class ParseqExport(nn.Module):
    """Wraps PARSeq so it takes only images and returns logits.

    Implements the NAR (decode_ar=False) forward path with one refine
    iteration, matching the training-time test-time configuration used for
    this checkpoint. Output shape: (B, max_label_length+1, num_classes).
    """

    def __init__(self, model: PARSeq, bos_id: int, eos_id: int, refine_iters: int = 1):
        super().__init__()
        self.model = model
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.refine_iters = refine_iters
        self.num_steps = model.max_label_length + 1

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        bs = images.shape[0]
        num_steps = self.num_steps
        device = images.device

        memory = self.model.encode(images)
        pos_queries = self.model.pos_queries[:, :num_steps].expand(bs, -1, -1)
        tgt_mask = torch.triu(torch.ones((num_steps, num_steps), dtype=torch.bool, device=device), 1)

        # NAR initial decode: only <bos> as context, query all positions.
        tgt_in = torch.full((bs, 1), self.bos_id, dtype=torch.long, device=device)
        tgt_out = self.model.decode(tgt_in, memory, tgt_query=pos_queries)
        logits = self.model.head(tgt_out)

        # Iterative refinement with cloze mask (training used refine_iters=1).
        if self.refine_iters > 0:
            cloze_mask = tgt_mask.clone()
            cloze_mask[torch.triu(torch.ones(num_steps, num_steps, dtype=torch.bool, device=device), 2)] = False
            L = logits.shape[1]
            bos = torch.full((bs, 1), self.bos_id, dtype=torch.long, device=device)
            for _ in range(self.refine_iters):
                tgt_in = torch.cat([bos, logits[:, :-1].argmax(-1)], dim=1)
                tgt_padding_mask = (tgt_in == self.eos_id).int().cumsum(-1) > 0
                tgt_out = self.model.decode(
                    tgt_in,
                    memory,
                    tgt_mask[:L, :L],
                    tgt_padding_mask,
                    pos_queries[:, :L],
                    cloze_mask[:L, : tgt_in.shape[1]],
                )
                logits = self.model.head(tgt_out)

        return logits


def strip_prefix(sd: dict, prefix: str = "_orig_mod.") -> dict:
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in sd.items()}


def build_model(charset: str, img_size, max_label_length: int, decode_ar: bool) -> PARSeq:
    tok = Tokenizer(charset)
    # parseq-large config from scripts/train_parseq_manga.py
    model = PARSeq(
        num_tokens=len(tok),
        max_label_length=max_label_length,
        img_size=list(img_size),
        patch_size=[4, 8],
        embed_dim=384,
        enc_num_heads=6,
        enc_mlp_ratio=4,
        enc_depth=12,
        dec_num_heads=12,
        dec_mlp_ratio=4,
        dec_depth=3,
        decode_ar=decode_ar,
        refine_iters=1,
        dropout=0.0,
    )
    return model, tok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--batch", type=int, default=1, help="static export batch size (use dynamic axis too)")
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    charset: str = ckpt["charset"]
    img_size = tuple(ckpt["img_size"])
    max_label_length = int(ckpt["max_label_length"])
    decode_ar = bool(ckpt["decode_ar"])
    print(f"Checkpoint: model_size={ckpt.get('model_size')} img_size={img_size} "
          f"max_label={max_label_length} decode_ar={decode_ar} "
          f"epoch={ckpt.get('epoch')} val_cer={ckpt.get('val_cer')}")
    print(f"Charset: {len(charset)} chars")

    model, tok = build_model(charset, img_size, max_label_length, decode_ar)
    sd = strip_prefix(ckpt["model_state_dict"])
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"WARN missing keys: {missing[:5]}... ({len(missing)} total)")
    if unexpected:
        print(f"WARN unexpected keys: {unexpected[:5]}... ({len(unexpected)} total)")
    model.eval()

    wrapper = ParseqExport(model, tok.bos_id, tok.eos_id, refine_iters=1).eval()

    # Sanity forward
    dummy = torch.zeros(args.batch, 3, img_size[0], img_size[1])
    with torch.no_grad():
        out = wrapper(dummy)
    print(f"Sanity forward OK. Output shape: {tuple(out.shape)} (expect "
          f"({args.batch}, {max_label_length+1}, {len(tok)-2}))")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Use the legacy TorchScript-based exporter (dynamo=False). The new dynamo
    # exporter bakes `batch=1` into internal reshape ops even with
    # dynamic_axes, so runtime batches >1 fail.
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(out_path),
        input_names=["images"],
        output_names=["logits"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,
    )
    print(f"ONNX exported: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    # Persist tokenizer metadata alongside the ONNX file.
    meta = {
        "charset": charset,
        "img_size": list(img_size),
        "max_label_length": max_label_length,
        "decode_ar": decode_ar,
        "refine_iters": 1,
        "num_tokens": len(tok),
        "bos_id": tok.bos_id,
        "eos_id": tok.eos_id,
        "pad_id": tok.pad_id,
        "head_dim": len(tok) - 2,
        "specials_first": ["[E]"],
        "specials_last": ["[B]", "[P]"],
        "normalize_mean": [0.5, 0.5, 0.5],
        "normalize_std": [0.5, 0.5, 0.5],
        "source_ckpt": str(args.ckpt),
        "model_size": ckpt.get("model_size"),
        "val_cer": ckpt.get("val_cer"),
        "epoch": ckpt.get("epoch"),
    }
    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"Metadata written: {meta_path}")


if __name__ == "__main__":
    main()
