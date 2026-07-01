"""INT4 W4A16 quantization of the gemma4_e4b_v11_pagecontext translation model.

Quantizes ONLY the 343 language_model transformer Linear layers to 4-bit
(group_size 128, symmetric). ALL embeddings (per-layer + tied 262k vocab), the
tied lm_head, the PLE per-layer projections, and the unused vision/audio towers
are kept high-precision.

Methods
-------
  rtn  (default) : QuantizationModifier, weight-only round-to-nearest. DATA-FREE
                   (no calibration forward), so it sidesteps BOTH gemma4 issues
                   that GPTQ hits on this arch:
                     1. cross-layer KV sharing breaks the *sequential* GPTQ
                        pipeline (shared_kv_states KeyError 'sliding_attention')
                     2. the *basic* GPTQ pipeline holds ~23GB of Hessians at once
                        (down_proj in=10240 -> 419MB each x42) -> CUDA OOM on 32GB
                   RTN W4A16 @ group128 is a strong weight-only baseline.
  gptq           : GPTQModifier (kept for reference; OOMs/KV-breaks on this arch,
                   see notes above). Requires --calib + a working pipeline.

De-risk: pass --n 6 for a tiny smoke run; full run uses --n 384.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import torch


def default_ignore() -> list[str]:
    # Verified against this exact checkpoint (inspect_modules.py):
    return [
        "lm_head",
        "re:.*embed_tokens.*",       # embed_tokens + embed_tokens_per_layer
        "re:.*embed_vision.*",
        "re:.*embed_audio.*",
        "re:model\\.vision_tower\\..*",
        "re:model\\.audio_tower\\..*",
        "re:.*multi_modal_projector.*",
        # PLE per-layer-input projections feed the per-layer embeddings (kept
        # high-precision); tiny (->256), bypass the main residual stream.
        "re:.*per_layer_model_projection.*",
        "re:.*per_layer_projection.*",
        "re:.*per_layer_input_gate.*",
    ]


def load_calib_texts(path: Path, n: int) -> list[str]:
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(json.loads(line)["text"])
            if len(texts) >= n:
                break
    return texts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", choices=["rtn", "gptq"], default="rtn")
    ap.add_argument("--calib", type=Path, default=None,
                    help="calibration jsonl (required for gptq, ignored for rtn)")
    ap.add_argument("--n", type=int, default=384)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--pipeline", default="basic")
    ap.add_argument("--device-map", default="cuda:0",
                    help="bf16 load placement. Use 'cpu' for the sequential pipeline "
                         "so weights stay on CPU and only the active layer onloads to "
                         "GPU (peak GPU ~= one decoder layer + its Hessian).")
    # Gemma4-specific sequential-tracing knobs (the fix for the KeyError/TraceError):
    #  * sequential_targets pins each decoder layer as ONE opaque graph leaf so the
    #    cross-layer KV-sharing + sliding/full layer_types logic never enters the FX
    #    graph (that was the source of the 'sliding_attention' KeyError when the
    #    partitioner tried to trace THROUGH layers).
    #  * tracing_ignore autowraps project_per_layer_inputs (PLE projection) which does
    #    `*inputs_embeds.shape[:-1]` Proxy-shape iteration -> TraceError. (This is
    #    already in llmcompressor 0.12.0's default tracing_ignore, but we set it
    #    explicitly to be version-robust.)
    ap.add_argument("--sequential-targets", nargs="*",
                    default=["Gemma4TextDecoderLayer"])
    ap.add_argument("--tracing-ignore", nargs="*", default=None,
                    help="override tracing autowrap ignore (default: add "
                         "project_per_layer_inputs to llmcompressor defaults)")
    ap.add_argument("--offload-device", default="cpu",
                    help="device for cached inter-layer activations (sequential)")
    ap.add_argument("--keep-embeds-on-cpu", action="store_true",
                    help="pin embed_tokens[_per_layer] to CPU during sequential "
                         "calibration (avoids 5.6GB GPU onload in head subgraph)")
    ap.add_argument("--fix-kv-sharing", action="store_true",
                    help="route gemma4 shared_kv_states through a persistent "
                         "per-batch store so KV sharing survives the sequential "
                         "partitioner (fixes KeyError 'sliding_attention')")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
    from compressed_tensors.quantization import (
        QuantizationArgs, QuantizationScheme, QuantizationStrategy,
    )

    ignore = default_ignore()

    print(f"[quant] method={args.method} loading model {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map=args.device_map,
        low_cpu_mem_usage=True,
    )
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # --- Gemma4 PLE memory fix for the sequential pipeline ---
    # The head subgraph's calibration forward onloads `embed_tokens_per_layer`
    # ([262144, 10752] bf16 = 5.6GB) to GPU just to do a gather in
    # get_per_layer_inputs(). That single 5.6GB onload is the GPTQ OOM on a
    # contended card. These embeddings are in `ignore` (never quantized), so we
    # pin them (and the 1.34GB tied embed_tokens) to stay on CPU during
    # calibration: the gather runs on CPU and only the tiny result moves to GPU.
    # We do this by patching set_onload_device to skip these modules so the
    # sequential pipeline's internal call cannot move them to GPU.
    if args.method == "gptq" and args.pipeline == "sequential" and args.keep_embeds_on_cpu:
        import compressed_tensors.offload.dispatch as _ct_dispatch
        from llmcompressor.pipelines.sequential import pipeline as _seq_pipeline

        # Pin ONLY embed_tokens_per_layer (5.6GB, used solely for the isolated
        # gather in get_per_layer_inputs). embed_tokens (1.34GB) must onload to
        # GPU normally because its output flows directly into GPU compute
        # (project_per_layer_inputs scaling), else we hit a device mismatch.
        _big_embed_substrings = (
            "embed_tokens_per_layer",
        )
        _orig_set_onload = _ct_dispatch.set_onload_device

        def _set_onload_skip_embeds(m, onload_device):
            # map module -> its qualified name once
            name_by_mod = {mod: name for name, mod in m.named_modules()}
            from compressed_tensors.offload.cache import OffloadCache
            from compressed_tensors.offload.module import offload_module
            from compressed_tensors.offload.utils import get_module_device
            for module in m.modules():
                nm = name_by_mod.get(module, "")
                pin_cpu = any(s in nm for s in _big_embed_substrings)
                dev = "cpu" if pin_cpu else onload_device
                if isinstance(module._parameters, OffloadCache):
                    module._parameters.onload_device = dev
                    module._buffers.onload_device = dev
                else:
                    offload_device = get_module_device(module, torch.device("cpu"))
                    offload_module(module, dev, offload_device)
            return m

        _ct_dispatch.set_onload_device = _set_onload_skip_embeds
        _seq_pipeline.set_onload_device = _set_onload_skip_embeds

        # The pinned embed_tokens_per_layer gather runs on CPU and returns a CPU
        # tensor. Its (tiny) result is consumed on GPU inside
        # project_per_layer_inputs, so patch get_per_layer_inputs on the
        # Gemma4TextModel instance to move the result to the compute device.
        # (We patch this MODEL METHOD rather than the embedding module's forward,
        # because offload_module later does `module.forward.__func__` and would
        # choke on a plain-function replacement; get_per_layer_inputs is
        # autowrapped/opaque so its body runs eagerly and is never offloaded.)
        _onload = torch.device(args.device_map if args.device_map != "cpu"
                               else "cuda:0")
        _txt_model = None
        for _n, _m in model.named_modules():
            if type(_m).__name__ == "Gemma4TextModel":
                _txt_model = _m
                break
        if _txt_model is not None:
            import types as _types
            _orig_gpli = _txt_model.get_per_layer_inputs.__func__

            def _gpli(self, input_ids, inputs_embeds, _orig=_orig_gpli,
                      _dev=_onload):
                out = _orig(self, input_ids, inputs_embeds)
                return out.to(_dev) if out is not None else out

            _txt_model.get_per_layer_inputs = _types.MethodType(_gpli, _txt_model)
        print("[quant] PLE fix: embed_tokens_per_layer weight pinned to CPU; "
              "gather runs on CPU, result -> GPU (avoids 5.6GB GPU onload)")

    # --- Gemma4 KV-sharing fix for the sequential pipeline ---
    # The sequential partitioner bakes `shared_kv_states = {}` (a FRESH empty dict)
    # into EVERY subgraph's compiled forward, so the in-place KV sharing across
    # layers is severed: producer layers (e.g. 6, 17) write KV into their own
    # discarded {}, and consumer layers 24-41 read from a fresh {} -> the
    # `KeyError: 'sliding_attention'` at modeling_gemma4.py:1253.
    #
    # Fix: patch Gemma4TextAttention.forward to route shared_kv_states reads/writes
    # through a PERSISTENT store on the text model, keyed by (current_batch_idx,
    # layer_type). Because the sequential pipeline runs subgraphs IN ORDER and sets
    # session.state.current_batch_idx per batch, the producer layer's KV for a
    # given batch survives until the matching consumer layer runs for that same
    # batch. This reproduces the true forward-pass KV sharing exactly.
    if args.method == "gptq" and args.pipeline == "sequential" and args.fix_kv_sharing:
        import types as _types
        import transformers.models.gemma4.modeling_gemma4 as _g4

        from llmcompressor.core import active_session as _active_session

        _persist: dict = {}  # (batch_idx, layer_type) -> (k, v)
        model._calib_kv_store = _persist

        def _cur_batch():
            try:
                st = _active_session().state
                return getattr(st, "current_batch_idx", 0)
            except Exception:
                return 0

        _orig_attn_fwd = _g4.Gemma4TextAttention.forward

        def _patched_attn_forward(self, hidden_states, position_embeddings,
                                  attention_mask, shared_kv_states,
                                  past_key_values=None, **kwargs):
            # Wrap the passed (empty) dict with a view onto the persistent store so
            # the unmodified original forward's `shared_kv_states[...]` ops hit the
            # persistent KV. We build a small dict proxy keyed by layer_type.
            bidx = _cur_batch()

            class _KVView(dict):
                def __getitem__(_s, key):
                    return _persist[(bidx, key)]

                def __setitem__(_s, key, val):
                    _persist[(bidx, key)] = val

                def __contains__(_s, key):
                    return (bidx, key) in _persist

            return _orig_attn_fwd(
                self, hidden_states, position_embeddings, attention_mask,
                _KVView(), past_key_values=past_key_values, **kwargs)

        _g4.Gemma4TextAttention.forward = _patched_attn_forward
        print("[quant] KV-sharing fix: shared_kv_states routed through a "
              "persistent per-batch store (survives subgraph partitioning)")

    # Build the W4A16 scheme (4-bit int, symmetric, group quant).
    w_args = QuantizationArgs(
        num_bits=4, type="int", symmetric=True,
        strategy=QuantizationStrategy.GROUP, group_size=args.group_size,
    )
    scheme = QuantizationScheme(targets=["Linear"], weights=w_args)

    oneshot_kwargs = dict(
        model=model,
        processor=tok,
        output_dir=args.out,
    )

    if args.method == "rtn":
        recipe = QuantizationModifier(
            config_groups={"group_0": scheme}, ignore=ignore,
        )
        # data-free: no dataset needed
        print(f"[quant] RTN W4A16 group_size={args.group_size} (data-free)")
        oneshot(recipe=recipe, **oneshot_kwargs)
    else:
        from datasets import Dataset

        if not args.calib:
            print("ERROR: --calib required for gptq")
            return 2
        texts = load_calib_texts(args.calib, args.n)

        def tokenize(s):
            return tok(s["text"], truncation=True, max_length=args.max_seq_len,
                       add_special_tokens=False)

        ds = Dataset.from_dict({"text": texts}).map(tokenize, remove_columns=["text"])
        recipe = GPTQModifier(config_groups={"group_0": scheme}, ignore=ignore)
        print(f"[quant] GPTQ W4A16 group_size={args.group_size} n={len(texts)} "
              f"pipeline={args.pipeline} seq_targets={args.sequential_targets}")

        # tracing_ignore: ensure project_per_layer_inputs is autowrapped (PLE path).
        if args.tracing_ignore is not None:
            tracing_ignore = args.tracing_ignore
        else:
            from llmcompressor.args.dataset_arguments import DatasetArguments
            tracing_ignore = list(DatasetArguments().tracing_ignore)
            if "project_per_layer_inputs" not in tracing_ignore:
                tracing_ignore.append("project_per_layer_inputs")

        seq_kwargs = {}
        if args.pipeline == "sequential":
            seq_kwargs = dict(
                sequential_targets=args.sequential_targets,
                tracing_ignore=tracing_ignore,
                sequential_offload_device=args.offload_device,
            )
        oneshot(recipe=recipe, dataset=ds, pipeline=args.pipeline,
                max_seq_length=args.max_seq_len,
                num_calibration_samples=len(texts),
                **seq_kwargs, **oneshot_kwargs)

    print(f"[quant] SUCCESS -> {args.out}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        print("\n========== QUANT FAILED ==========", file=sys.stderr)
        traceback.print_exc()
        print("==================================", file=sys.stderr)
        sys.exit(1)
