#!/usr/bin/env python3
"""Benchmark translation speed for HY-MT1.5-1.8B-Q4_K_M model."""

import time
import sys
from pathlib import Path

sys.path.insert(0, '.')

from llama_cpp import Llama

# Test configuration - multiple different texts to simulate real API usage
TEST_TEXTS = [
    "また私の勝ち〜!",
    "何か賭けたら張り合い出て本気出るかもよ?",
    "ゴール!!",
    "なんでも……？",
    "負けた方が勝った方の言うこと何でもひとつ聞くとか!",
    "くそっ…どうしても勝てない…",
    "お前には絶対負けない!",
    "これで終わりだ!",
    "まだまだこれからだ!",
    "信じられない…",
]
TEST_TEXT = TEST_TEXTS[1]  # Default for single-text mode
NUM_ITERATIONS = 10
WARMUP_ITERATIONS = 3
TEST_DIFFERENT_TEXTS = True  # Toggle between same-text and different-text modes

# Model configuration
WEIGHTS_DIR = Path(__file__).parent.parent / "app" / "weights"
MODEL_PATH = WEIGHTS_DIR / "HY-MT1.5-1.8B-Q8_0.gguf"


def translate(llm: Llama, text: str) -> str:
    """Translate using HY-MT model."""
    prompt = f"Translate the following segment into English, without additional explanation.\n\n{text}"
    messages = [{"role": "user", "content": prompt}]
    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=256,
        temperature=0.7,
        top_k=20,
        top_p=0.6,
        repeat_penalty=1.05
    )
    result = response["choices"][0]["message"]["content"].strip()
    if result.startswith("Assistant:"):
        result = result[len("Assistant:"):].strip()
    return result


def main():
    print("Translation Model Benchmark")
    print(f"Model: HY-MT1.5-1.8B-Q4_K_M")
    mode = "DIFFERENT texts" if TEST_DIFFERENT_TEXTS else "SAME text repeated"
    print(f"Mode: {mode}")
    print(f"Iterations: {NUM_ITERATIONS} (+ {WARMUP_ITERATIONS} warmup)")

    if not MODEL_PATH.exists():
        print(f"\nModel not found at {MODEL_PATH}")
        print("Run: uv run python scripts/download_models.py --translation")
        sys.exit(1)

    # Load model
    print("\nLoading model...")
    load_start = time.perf_counter()
    llm = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=2048,
        n_gpu_layers=-1,
        n_threads=4,
        verbose=False,
    )
    load_time = (time.perf_counter() - load_start) * 1000
    print(f"Loaded in {load_time:.0f}ms")

    # Warmup with first text
    warmup_text = TEST_TEXTS[0] if TEST_DIFFERENT_TEXTS else TEST_TEXT
    print(f"\nWarmup ({WARMUP_ITERATIONS} iterations)...")
    for _ in range(WARMUP_ITERATIONS):
        translate(llm, warmup_text)

    # Benchmark
    print(f"Benchmarking ({NUM_ITERATIONS} iterations)...")
    times = []
    last_result = ""
    for i in range(NUM_ITERATIONS):
        # Use different text each iteration or same text
        text = TEST_TEXTS[i % len(TEST_TEXTS)] if TEST_DIFFERENT_TEXTS else TEST_TEXT
        start = time.perf_counter()
        result = translate(llm, text)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        last_result = result
        preview = text[:20] + "..." if len(text) > 20 else text
        print(f"  {i+1}: {elapsed:.1f}ms - '{preview}' → '{result[:40]}...'")

    # Calculate stats
    avg = sum(times) / len(times)
    min_t = min(times)
    max_t = max(times)

    print(f"\nResults:")
    print(f"  Avg: {avg:.1f}ms | Min: {min_t:.1f}ms | Max: {max_t:.1f}ms")
    print(f"  Load time: {load_time:.0f}ms")
    print(f"  Translation: '{last_result}'")


if __name__ == "__main__":
    main()
