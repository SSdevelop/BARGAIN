import csv
import math
import os
from typing import Iterable

import numpy as np
import pandas as pd


TAIL_FRACTIONS = (0.01, 0.05, 0.10)
RESULTS_CSV_HEADER = [
    "script",
    "metric",
    "score",
    "tail_accuracy_1pct",
    "tail_accuracy_5pct",
    "tail_accuracy_10pct",
    "proxy_fraction",
    "time_seconds",
    "proxy_seconds",
    "oracle_seconds",
    "judge_seconds",
    "judge_eval_seconds",
    "judge_calls",
    "proxy_total_tokens",
    "oracle_total_tokens",
    "judge_total_tokens",
    "judge_eval_tokens",
]


def compute_tail_accuracies(
    values: Iterable[float], tail_fracs: tuple[float, float, float] = TAIL_FRACTIONS
) -> tuple[float, float, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return 0.0, 0.0, 0.0

    arr = np.sort(arr)
    tail_scores: list[float] = []
    for frac in tail_fracs:
        tail_count = max(1, int(math.ceil(arr.size * frac)))
        tail_scores.append(float(np.mean(arr[:tail_count])))
    return tail_scores[0], tail_scores[1], tail_scores[2]


def save_outputs_df(df: pd.DataFrame, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)


def append_results_row(
    script_name: str,
    metric_name: str,
    score: float,
    proxy_fraction: float,
    elapsed_seconds: float,
    proxy_timing: dict,
    oracle_timing: dict,
    judge_eval_seconds: float,
    judge_total_tokens: int,
    judge_eval_tokens: int,
    tail_accuracy_1pct: float,
    tail_accuracy_5pct: float,
    tail_accuracy_10pct: float,
    results_csv: str = "results.csv",
) -> None:
    write_header = not os.path.exists(results_csv)
    with open(results_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(RESULTS_CSV_HEADER)
        writer.writerow(
            [
                script_name,
                metric_name,
                f"{score}",
                f"{tail_accuracy_1pct}",
                f"{tail_accuracy_5pct}",
                f"{tail_accuracy_10pct}",
                f"{proxy_fraction:.2f}",
                f"{elapsed_seconds}",
                f"{proxy_timing['proxy_seconds']}",
                f"{oracle_timing['oracle_seconds']}",
                f"{oracle_timing['judge_seconds']}",
                f"{judge_eval_seconds}",
                f"{oracle_timing['judge_calls']}",
                f"{proxy_timing['proxy_total_tokens']}",
                f"{oracle_timing['oracle_total_tokens']}",
                f"{judge_total_tokens}",
                f"{judge_eval_tokens}",
            ]
        )
