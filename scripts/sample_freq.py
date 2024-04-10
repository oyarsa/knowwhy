#!/usr/bin/env python3
# pyright: basic
import argparse
import json
import random
from pathlib import Path
from typing import Any, Counter


def sample_preserving_frequencies(
    data: list[dict[str, Any]], k: int
) -> list[dict[str, Any]]:
    "Samples K elements preserving the frequency of valid values."
    freq = Counter(item["valid"] for item in data)

    total = freq[True] + freq[False]
    true_to_sample = round(k * (freq[True] / total))
    false_to_sample = k - true_to_sample

    true_data = [item for item in data if item["valid"]]
    false_data = [item for item in data if not item["valid"]]

    sampled_true = random.sample(true_data, true_to_sample)
    sampled_false = random.sample(false_data, false_to_sample)
    sampled_data = sampled_true + sampled_false

    random.shuffle(sampled_data)
    return sampled_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample data from dataset preserving frequencies."
    )
    parser.add_argument("input", type=Path, help="Input file")
    parser.add_argument("k", type=int, help="Number of samples")
    parser.add_argument("--output", type=Path, help="Output file")
    parser.add_argument("--seed", type=int, help="Random seed", default=2)
    args = parser.parse_args()

    random.seed(args.seed)
    data = json.loads(args.input.read_text())
    sampled = sample_preserving_frequencies(data, args.k)

    output = args.output or args.input.with_name(
        f"{args.input.stem}_sampled_{args.k}.json"
    )
    output.write_text(json.dumps(sampled))


if __name__ == "__main__":
    main()
