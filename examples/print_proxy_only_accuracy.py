import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            payload = line.strip()
            if not payload:
                continue
            try:
                rows.append(json.loads(payload))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_num}") from exc
    return rows


def parse_csv_list(raw: str) -> list[str]:
    return [token.strip() for token in raw.split(",") if token.strip()]


def infer_tasks(judge_cache_dir: Path) -> list[str]:
    if not judge_cache_dir.exists():
        return []
    return sorted(path.stem for path in judge_cache_dir.glob("*.jsonl"))


def parse_tasks_arg(raw: str, available: list[str]) -> list[str]:
    raw = raw.strip()
    if raw.lower() == "all":
        return available
    tasks = parse_csv_list(raw)
    if not tasks:
        raise ValueError("No task names provided.")
    invalid = [task for task in tasks if task not in available]
    if invalid:
        raise ValueError(
            f"Unknown task(s): {invalid}. Available tasks: {', '.join(available)}"
        )
    return list(dict.fromkeys(tasks))


def compute_cache_proxy_only_mean(judge_cache_file: Path, judge_column: str) -> tuple[float, int]:
    rows = load_jsonl(judge_cache_file)
    if not rows:
        return 0.0, 0
    df = pd.DataFrame(rows)
    if judge_column not in df.columns:
        raise ValueError(f"Missing '{judge_column}' in {judge_cache_file}")
    values = pd.to_numeric(df[judge_column], errors="coerce").dropna()
    if values.empty:
        return 0.0, 0
    return float(values.mean()), int(len(values))


def compute_routed_proxy_mean(outputs_file: Path, judge_column: str) -> tuple[float | None, int, float | None]:
    if not outputs_file.exists():
        return None, 0, None
    df = pd.read_csv(outputs_file)
    if "used_oracle" not in df.columns or judge_column not in df.columns:
        return None, 0, None
    used_oracle = df["used_oracle"].astype(bool)
    proxy_values = pd.to_numeric(df.loc[~used_oracle, judge_column], errors="coerce").dropna()
    if proxy_values.empty:
        return None, 0, float(1.0 - used_oracle.mean())
    return (
        float(proxy_values.mean()),
        int(len(proxy_values)),
        float(1.0 - used_oracle.mean()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Print proxy-only accuracy to terminal using average judge F1 from caches."
        )
    )
    parser.add_argument(
        "--tasks",
        default="all",
        help="Comma-separated task names or 'all'. Default: all tasks in judge cache dir.",
    )
    parser.add_argument(
        "--judge-cache-dir",
        default="cache/judge",
        help="Directory containing per-task judge JSONL files.",
    )
    parser.add_argument(
        "--judge-column",
        default="semantic_f1",
        help="Judge score column. Default: semantic_f1",
    )
    parser.add_argument(
        "--outputs-dir",
        default="results",
        help="Optional outputs CSV directory for routed-proxy stats. Default: results",
    )
    args = parser.parse_args()

    judge_cache_dir = Path(args.judge_cache_dir)
    outputs_dir = Path(args.outputs_dir)
    available_tasks = infer_tasks(judge_cache_dir)
    if not available_tasks:
        raise FileNotFoundError(f"No judge cache JSONL files found in {judge_cache_dir}")
    tasks = parse_tasks_arg(args.tasks, available_tasks)

    print("Proxy-only accuracy (average judge score)")
    print(f"- judge_column: {args.judge_column}")
    print(f"- tasks: {tasks}")
    print("")

    for task in tasks:
        judge_cache_file = judge_cache_dir / f"{task}.jsonl"
        cache_mean, cache_n = compute_cache_proxy_only_mean(judge_cache_file, args.judge_column)
        outputs_file = outputs_dir / f"{task}_outputs.csv"
        routed_mean, routed_n, proxy_fraction = compute_routed_proxy_mean(
            outputs_file,
            args.judge_column,
        )

        print(f"[{task}]")
        print(
            f"  cache proxy-only mean ({args.judge_column}): "
            f"{cache_mean:.6f} (n={cache_n})"
        )
        if routed_mean is None:
            print("  routed proxy mean: N/A (missing or incompatible outputs CSV)")
        else:
            print(
                f"  routed proxy mean ({args.judge_column}): "
                f"{routed_mean:.6f} (n={routed_n}, proxy_fraction={proxy_fraction:.6f})"
            )
        print("")


if __name__ == "__main__":
    main()

