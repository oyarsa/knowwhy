"""
Converts the human cache evaluations into a JSON file to train an evaluation model.
"""

import argparse
import json
import os
import pickle
import re
from collections.abc import Mapping
from typing import Any, TypedDict

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=str, help="Human cache evaluations")
    parser.add_argument("dataset", type=str, help="Original dataset")
    parser.add_argument("output", type=str, help="Output file")
    return parser.parse_args()


class CacheValue(TypedDict):
    """Cache value for a tuple of (question, answer, narrative).

    val_annotations: list of Likert annotation value for the respective source.
    annotation_sources: name of the source model from which the evaluated is.

    Both lists are always of the same size, but that size varies.
    """

    val_annotations: list[int]
    annotation_sources: list[str]


Cache = dict[tuple[str, str, str], CacheValue]
"""Cache of human evaluations.

Keys are tuple of (question, answer, narrative). They are always lower cased.
"""


def retrieve_from_cache(
    cache: Cache, question: str, answer: str, narrative: str
) -> CacheValue:
    """
    Use question, answer and narrative to retrieve all associated values
    Return failure if key not found
    """

    key = (question.lower(), answer.lower(), narrative.lower())
    return cache[key]


def remove_punctuation(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text)


def get_likert_scores(row: Mapping[str, str], cache: Cache) -> list[int]:
    answer = row["predicted_answer"] if "predicted_answer" in row else row["answer"]
    answer = remove_punctuation(answer)

    question = row["question"]
    narrative = row["narrative"]

    info = retrieve_from_cache(cache, question, answer, narrative)
    return info["val_annotations"]


def overall_avg_likert(df: pd.DataFrame, cache: Cache) -> tuple[float, float]:
    likerts: list[float] = []
    for _, row in df.iterrows():
        likertscores = get_likert_scores(row.to_dict(), cache)
        likerts.append(sum(likertscores) / len(likertscores))

    return round(sum(likerts) / len(likerts), 2), round(
        len(likerts) * 100 / df.shape[0], 2
    )


def overall_avg_binary_likert(df: pd.DataFrame, cache: Cache) -> tuple[float, float]:
    likerts: list[float] = []
    for _, row in df.iterrows():
        likertscores = get_likert_scores(row.to_dict(), cache)
        binary_likertscores = [0 if x < 1 else 1 for x in likertscores]
        likerts.append(sum(binary_likertscores) / len(binary_likertscores))

    return round(sum(likerts) / len(likerts), 2), round(
        len(likerts) * 100 / df.shape[0], 2
    )


class DataEntry(TypedDict):
    input: str
    output: str
    gold: str
    valid: bool


def entry_to_input(question: str, narrative: str) -> str:
    return f"question: {question}\ncontext: {narrative}"


def convert_cache(cache: Cache) -> list[DataEntry]:
    "Convert the cache into a list of DataEntry (classifier input) objects."
    new_cache: list[DataEntry] = []
    for (question, answer, narrative), value in cache.items():
        likerts = value["val_annotations"]
        avg_likert = sum(likerts) / len(likerts)
        bin_likert = avg_likert < 1

        new_cache.append(
            {
                "input": entry_to_input(question, narrative),
                "output": answer,
                "gold": "not found",
                "valid": bin_likert,
            }
        )

    return new_cache


def index_dataset(dataset: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        entry_to_input(item["question"].lower(), item["narrative"]).lower(): item
        for item in dataset
    }


def add_gold_answers(
    cache: list[DataEntry], dataset: list[dict[str, Any]]
) -> list[DataEntry]:
    indexed_dataset = index_dataset(dataset)
    new_dataset: list[DataEntry] = []

    for entry in cache:
        if data := indexed_dataset.get(entry["input"].lower()):
            entry["input"] = entry_to_input(data["question"], data["narrative"])
            entry["gold"] = data["answer"]
            new_dataset.append(entry)

    return new_dataset


def main(args: argparse.Namespace) -> None:
    with open(args.cache, "rb") as f:
        original_cache: Cache = pickle.load(f)

    with open(args.dataset) as f:
        dataset: list[dict[str, Any]] = json.load(f)

    cache = convert_cache(original_cache)
    new_dataset = add_gold_answers(cache, dataset)

    print("Cache size:", len(cache))
    print(
        "Cache with gold answers size:",
        sum(entry["gold"] != "not found" for entry in new_dataset),
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(new_dataset, f, indent=2)


if __name__ == "__main__":
    args = parse_args()
    main(args)
