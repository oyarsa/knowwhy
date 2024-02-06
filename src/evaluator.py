"""
This file runs evaluation of model outputs using the cache of human judgments available
"""

import argparse
import json
import pickle
import re
from collections.abc import Mapping
from typing import TypedDict

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions-file-path", type=str, required=True, help="Predictions file path"
    )
    parser.add_argument(
        "--answers-file-path", type=str, required=True, help="Cache file"
    )
    parser.add_argument(
        "--output-file-path",
        type=str,
        required=True,
        help="Name of JSON metrics file that should be produced",
    )
    args, _ = parser.parse_known_args()
    return args


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


def main(args: argparse.Namespace) -> None:
    with open(args.answers_file_path, "rb") as fp:
        cache: Cache = pickle.load(fp)

    predictions_df = pd.read_csv(args.predictions_file_path)
    avg_likert, coverage = overall_avg_likert(predictions_df, cache)
    binary_likert, coverage = overall_avg_binary_likert(predictions_df, cache)

    impl_predictions_df = predictions_df.query("is_ques_answerable == 'Not Answerable'")
    impl_avg_likert, impl_coverage = overall_avg_likert(impl_predictions_df, cache)
    impl_binary_likert, impl_coverage = overall_avg_binary_likert(
        impl_predictions_df, cache
    )

    results = {
        "Avg Likert": avg_likert,
        "Binary Accuracy": binary_likert,
        "Coverage": coverage,
        "Avg Likert (IMPL)": impl_avg_likert,
        "Binary Accuracy (IMPL)": impl_binary_likert,
        "Binary Coverage (IMPL)": impl_coverage,
    }

    with open(args.output_file_path, "w") as fp:
        json.dump(results, fp)


if __name__ == "__main__":
    args = parse_args()
    main(args)
