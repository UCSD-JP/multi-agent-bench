#!/usr/bin/env python3
"""Workload generator for memory-profile experiments.

Generates a JSONL file with requests spanning a memory-pressure scan grid:
  input_len  × [8192, 16384, 32768]
  output_len × [64, 128]
  concurrency× [8, 16, 32, 64]
  prefix_reuse × [low, high]

Each condition produces n=1 request (default). Requests are grouped by
concurrency_group so the runner sends them at the right parallelism level.

Prompt construction:
  - "low" prefix reuse: unique random-ish prompt per request
  - "high" prefix reuse: shared prefix + unique suffix (simulates prefix cache hits)
"""

import json
import hashlib
from pathlib import Path
from typing import Optional


# Default grid axes
DEFAULT_INPUT_LENS = [8192, 16384, 32768]
DEFAULT_OUTPUT_LENS = [64, 128]
DEFAULT_CONCURRENCIES = [8, 16, 32, 64]
DEFAULT_PREFIX_MODES = ["low", "high"]
DEFAULT_N_PER_CONDITION = 1

# Shared prefix for "high" reuse mode (~90% of input_len is shared)
SHARED_PREFIX_TEMPLATE = (
    "You are a helpful AI assistant. Below is a long document that you need "
    "to analyze carefully. Please read the entire document and answer the "
    "question at the end.\n\n"
    "--- BEGIN DOCUMENT ---\n"
    "{padding}\n"
    "--- END DOCUMENT ---\n\n"
)

UNIQUE_SUFFIX_TEMPLATE = (
    "Question {req_id}: Based on section {section} of the document above, "
    "provide a detailed summary of the key findings. "
    "Include specific data points and conclusions. "
)


_PADDING_WORDS = [
    "the", "system", "performance", "analysis", "report", "shows", "that",
    "memory", "utilization", "increased", "during", "batch", "processing",
    "with", "concurrent", "requests", "reaching", "peak", "allocation",
    "when", "input", "sequences", "exceeded", "threshold", "values",
    "observed", "across", "multiple", "experiments", "conducted", "on",
    "hardware", "configurations", "including", "various", "parallelism",
    "strategies", "and", "optimization", "techniques", "applied", "to",
    "model", "inference", "workloads", "generating", "output", "tokens",
    "while", "maintaining", "acceptable", "latency", "throughput", "levels",
]


def _make_padding(target_chars: int, seed: str) -> str:
    """Generate deterministic padding text to reach target character count.

    Uses common English words for predictable tokenization (~1 token per word).
    """
    # Use seed to pick starting offset for variety
    h = int(hashlib.sha256(seed.encode()).hexdigest()[:8], 16)
    n_words = len(_PADDING_WORDS)
    chunks = []
    total = 0
    i = 0
    while total < target_chars:
        word = _PADDING_WORDS[(h + i) % n_words]
        chunks.append(word)
        total += len(word) + 1  # +1 for space
        i += 1
    return " ".join(chunks)[:target_chars]


def _estimate_chars_for_tokens(n_tokens: int) -> int:
    """Estimate characters needed for n_tokens.

    Calibrated against Qwen3-Next tokenizer with our padding vocabulary.
    Measured: ~8.2 chars/token (common English words tokenize efficiently).
    """
    return int(n_tokens * 8.2)


def make_prompt(
    input_len: int,
    prefix_mode: str,
    req_id: str,
    shared_prefix_cache: dict,
) -> str:
    """Construct a prompt targeting `input_len` tokens.

    Args:
        input_len: Target input length in tokens.
        prefix_mode: "low" (unique per request) or "high" (shared prefix).
        req_id: Unique request identifier for suffix variation.
        shared_prefix_cache: Cache dict for shared prefixes (mutated).
    """
    target_chars = _estimate_chars_for_tokens(input_len)

    if prefix_mode == "low":
        # Fully unique prompt
        padding = _make_padding(target_chars - 200, seed=req_id)
        return (
            f"Request {req_id}: Analyze the following data thoroughly.\n\n"
            f"{padding}\n\n"
            f"Provide a comprehensive summary."
        )
    else:
        # High prefix reuse: shared prefix + unique suffix
        cache_key = (input_len, "shared")
        if cache_key not in shared_prefix_cache:
            prefix_chars = int(target_chars * 0.9)
            padding = _make_padding(prefix_chars - 200, seed=f"shared_{input_len}")
            shared_prefix_cache[cache_key] = SHARED_PREFIX_TEMPLATE.format(
                padding=padding
            )
        shared = shared_prefix_cache[cache_key]

        # Unique suffix (~10% of total)
        section = hashlib.md5(req_id.encode()).hexdigest()[:6]
        suffix = UNIQUE_SUFFIX_TEMPLATE.format(req_id=req_id, section=section)

        # Pad suffix to fill remaining chars
        remaining = target_chars - len(shared) - len(suffix)
        if remaining > 0:
            extra = _make_padding(remaining, seed=f"suffix_{req_id}")
            suffix += f" Additional context: {extra}"

        return shared + suffix


def generate_workload(
    output_path: Path,
    input_lens: Optional[list[int]] = None,
    output_lens: Optional[list[int]] = None,
    concurrencies: Optional[list[int]] = None,
    prefix_modes: Optional[list[str]] = None,
    n_per_condition: int = DEFAULT_N_PER_CONDITION,
) -> int:
    """Generate workload JSONL file.

    Returns the number of requests written.
    """
    input_lens = input_lens or DEFAULT_INPUT_LENS
    output_lens = output_lens or DEFAULT_OUTPUT_LENS
    concurrencies = concurrencies or DEFAULT_CONCURRENCIES
    prefix_modes = prefix_modes or DEFAULT_PREFIX_MODES

    shared_prefix_cache: dict = {}
    output_path.parent.mkdir(parents=True, exist_ok=True)

    req_count = 0
    with open(output_path, "w") as f:
        for conc in concurrencies:
            for ilen in input_lens:
                for olen in output_lens:
                    for pmode in prefix_modes:
                        for rep in range(n_per_condition):
                            req_id = (
                                f"r_{conc:03d}c_{ilen}i_{olen}o"
                                f"_{pmode}_{rep:02d}"
                            )
                            prompt = make_prompt(
                                ilen, pmode, req_id, shared_prefix_cache
                            )
                            record = {
                                "req_id": req_id,
                                "prompt": prompt,
                                "input_len": ilen,
                                "output_len": olen,
                                "concurrency_group": conc,
                                "extra": {
                                    "prefix_mode": pmode,
                                    "rep": rep,
                                },
                            }
                            f.write(json.dumps(record) + "\n")
                            req_count += 1

    return req_count


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Memory-profile workload generator")
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("workload.jsonl"),
        help="Output JSONL path (default: workload.jsonl)",
    )
    parser.add_argument(
        "--input-lens", type=str, default="8192,16384,32768",
        help="Comma-separated input lengths (default: 8192,16384,32768)",
    )
    parser.add_argument(
        "--output-lens", type=str, default="64,128",
        help="Comma-separated output lengths (default: 64,128)",
    )
    parser.add_argument(
        "--concurrencies", type=str, default="8,16,32,64",
        help="Comma-separated concurrency levels (default: 8,16,32,64)",
    )
    parser.add_argument(
        "--prefix-modes", type=str, default="low,high",
        help="Comma-separated prefix modes (default: low,high)",
    )
    parser.add_argument(
        "--n-per-condition", type=int, default=1,
        help="Requests per condition (default: 1)",
    )
    args = parser.parse_args()

    n = generate_workload(
        args.output,
        input_lens=[int(x) for x in args.input_lens.split(",")],
        output_lens=[int(x) for x in args.output_lens.split(",")],
        concurrencies=[int(x) for x in args.concurrencies.split(",")],
        prefix_modes=args.prefix_modes.split(","),
        n_per_condition=args.n_per_condition,
    )
    print(f"Generated {n} requests → {args.output}")

    # Print grid summary
    grid = (
        len(args.input_lens.split(","))
        * len(args.output_lens.split(","))
        * len(args.concurrencies.split(","))
        * len(args.prefix_modes.split(","))
        * args.n_per_condition
    )
    print(f"Grid: {grid} conditions (should match {n})")


if __name__ == "__main__":
    main()
