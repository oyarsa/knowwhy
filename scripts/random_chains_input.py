"""Select K elements from training data. They will be used to generate random CoT chains.

K must be even, such that K/2 elements have a high score and K/2 have low score. This
will be used to show the model how to evaluate both good and bad results.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any


def sample_data(data: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    random.shuffle(data)
    return data[:k]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter and randomly select items from a JSON file."
    )
    parser.add_argument("file", type=Path, help="JSON file containing data.")
    parser.add_argument(
        "output_file",
        type=Path,
        help="Output file to write the selected items. If not specified, prints to stdout.",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=8,
        help="Number of random items to select (default: %(default)s). It must be an"
        " even number. As half will be invalid and half will be valid.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed (default: %(default)s)."
    )
    args = parser.parse_args()

    if args.k % 2 != 0:
        parser.error("k must be an even number.")

    random.seed(args.seed)
    data = json.loads(args.file.read_text())

    high_score = [item for item in data if item.get("score") >= 4]
    low_score = [item for item in data if item.get("score") <= 2]

    high_selected = sample_data(high_score, args.k // 2)
    low_selected = sample_data(low_score, args.k // 2)
    selected = high_selected + low_selected
    random.shuffle(selected)

    args.output_file.write_text(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
