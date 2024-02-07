"Splits the converted dataset into train, dev, and test sets."
import argparse
import json
import random
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Input file path.")
    parser.add_argument("output_dir", type=Path, help="Output directory path.")
    parser.add_argument(
        "--split", type=float, default=0.7, help="Train split percentage."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for shuffling."
    )
    args = parser.parse_args()

    random.seed(args.seed)
    data: list[dict[str, Any]] = json.loads(args.input.read_text())

    inputs = sorted({d["input"] for d in data})
    print(f"Total inputs: {len(data)}")
    print(f"Unique inputs: {len(inputs)}")

    train_size = int(len(inputs) * args.split)
    train_inputs, rest = inputs[:train_size], inputs[train_size:]

    dev_size = len(rest) // 2
    dev_inputs, test_inputs = rest[:dev_size], rest[dev_size:]

    train = [d for d in data if d["input"] in train_inputs]
    dev = [d for d in data if d["input"] in dev_inputs]
    test = [d for d in data if d["input"] in test_inputs]

    # Shuffle the datasets so that partial data isn't dominated by particular questions
    random.shuffle(train)
    random.shuffle(dev)
    random.shuffle(test)

    print(f"Train size: {len(train)} ({len(train) / len(data):.2%})")
    print(f"Dev size: {len(dev)} ({len(dev) / len(data):.2%})")
    print(f"Test size: {len(test)} ({len(test) / len(data):.2%})")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "train.json").write_text(json.dumps(train, indent=2))
    (args.output_dir / "dev.json").write_text(json.dumps(dev, indent=2))
    (args.output_dir / "test.json").write_text(json.dumps(test, indent=2))


if __name__ == "__main__":
    main()
