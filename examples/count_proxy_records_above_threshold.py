import argparse
import json
from pathlib import Path


def count_records_above_threshold(
    cache_path: Path,
    threshold: float,
    score_column: str,
) -> tuple[int, int, int]:
    total_rows = 0
    passing_rows = 0
    skipped_rows = 0

    with cache_path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            payload = line.strip()
            if not payload:
                continue

            total_rows += 1
            try:
                row = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {cache_path}:{line_num}") from exc

            raw_score = row.get(score_column)
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                skipped_rows += 1
                continue

            if score >= threshold:
                passing_rows += 1

    return total_rows, passing_rows, skipped_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Count records in a task's proxy cache whose score meets/exceeds "
            "a given threshold."
        )
    )
    parser.add_argument(
        "--task",
        required=True,
        help="Task name (expects cache file at <proxy-cache-dir>/<task>.jsonl).",
    )
    parser.add_argument(
        "--threshold",
        required=True,
        type=float,
        help="Threshold value used for pass/fail comparison (score >= threshold).",
    )
    parser.add_argument(
        "--proxy-cache-dir",
        default="cache/proxy",
        help="Directory containing per-task proxy JSONL cache files. Default: cache/proxy",
    )
    parser.add_argument(
        "--score-column",
        default="score_bargain",
        help="Score column name in each JSONL row. Default: score_bargain",
    )
    args = parser.parse_args()

    cache_path = Path(args.proxy_cache_dir) / f"{args.task}.jsonl"
    if not cache_path.exists():
        raise FileNotFoundError(f"Proxy cache file not found: {cache_path}")

    total_rows, passing_rows, skipped_rows = count_records_above_threshold(
        cache_path=cache_path,
        threshold=args.threshold,
        score_column=args.score_column,
    )

    print(
        f"task={args.task} \nthreshold={args.threshold} "
        f"\nscore_column={args.score_column} \npassing_records={passing_rows}"
    )
    print(f"total_rows={total_rows} skipped_rows_non_numeric={skipped_rows}")


if __name__ == "__main__":
    main()
