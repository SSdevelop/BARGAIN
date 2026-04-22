import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_quantiles_arg(raw: str) -> list[float]:
    values: list[float] = []
    for token in [part.strip() for part in raw.split(",") if part.strip()]:
        q = float(token)
        if q > 1.0:
            q /= 100.0
        if q < 0.0 or q > 1.0:
            raise ValueError(
                f"Invalid quantile '{token}'. Use values in [0,1] or percentages in [0,100]."
            )
        values.append(q)
    if not values:
        raise ValueError("No quantiles provided.")
    return sorted(set(values))


def parse_tasks_arg(tasks_arg: str, default_tasks: list[str]) -> list[str]:
    raw = tasks_arg.strip()
    if raw.lower() == "all":
        return default_tasks

    requested = [name.strip() for name in raw.split(",") if name.strip()]
    if not requested:
        raise ValueError("No task names provided.")
    return list(dict.fromkeys(requested))


def parse_tail_fracs(raw: str) -> list[float]:
    values: list[float] = []
    for token in [part.strip() for part in raw.split(",") if part.strip()]:
        frac = float(token)
        if frac > 1.0:
            frac /= 100.0
        if frac <= 0.0 or frac > 1.0:
            raise ValueError(
                f"Invalid tail fraction '{token}'. Use values in (0,1] or percentages in (0,100]."
            )
        values.append(frac)
    if not values:
        raise ValueError("No tail fractions provided.")
    return sorted(set(values))


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


def compute_tail_metrics(
    confidence: np.ndarray, correctness: np.ndarray, tail_frac: float
) -> dict[str, float]:
    n = len(confidence)
    if n == 0:
        raise ValueError("Cannot compute tail metrics on empty arrays.")

    k = max(1, int(math.ceil(n * tail_frac)))
    order = np.argsort(confidence, kind="mergesort")
    tail_idx = order[:k]
    threshold = float(confidence[tail_idx[-1]])

    return {
        "n_records": float(n),
        "tail_count": float(k),
        "overall_accuracy": float(np.mean(correctness)),
        "tail_accuracy": float(np.mean(correctness[tail_idx])),
        "tail_threshold_max_confidence": threshold,
    }


def compute_tail_latency_metrics(
    rank_values: np.ndarray, latency_values: np.ndarray, tail_frac: float, quantiles: list[float]
) -> dict[str, float]:
    n = len(rank_values)
    if n == 0:
        raise ValueError("Cannot compute tail metrics on empty arrays.")

    k = max(1, int(math.ceil(n * tail_frac)))
    order = np.argsort(rank_values, kind="mergesort")
    tail_idx = order[:k]
    threshold = float(rank_values[tail_idx[-1]])

    out: dict[str, float] = {
        "n_records": float(n),
        "tail_count": float(k),
        "tail_threshold_max_rank_value": threshold,
        "overall_latency_mean": float(np.mean(latency_values)),
        "tail_latency_mean": float(np.mean(latency_values[tail_idx])),
    }
    for q in quantiles:
        key = f"latency_p{int(round(100 * q)):02d}"
        out[f"overall_{key}"] = float(np.quantile(latency_values, q))
        out[f"tail_{key}"] = float(np.quantile(latency_values[tail_idx], q))
    return out


def build_proxy_eval_df(
    task: str,
    proxy_cache_dir: Path,
    judge_cache_dir: Path,
    proxy_columns: set[str],
    judge_columns: set[str],
) -> pd.DataFrame:
    judge_path = judge_cache_dir / f"{task}.jsonl"
    if not judge_path.exists():
        return pd.DataFrame()

    judge_rows = load_jsonl(judge_path)
    if not judge_rows:
        return pd.DataFrame()

    judge_df = pd.DataFrame(judge_rows)
    if proxy_columns:
        proxy_path = proxy_cache_dir / f"{task}.jsonl"
        if not proxy_path.exists():
            return pd.DataFrame()
        proxy_rows = load_jsonl(proxy_path)
        if not proxy_rows:
            return pd.DataFrame()
        proxy_df = pd.DataFrame(proxy_rows)
    else:
        proxy_df = pd.DataFrame()

    required_proxy_cols = {"record_key", "record_index"} | proxy_columns
    required_judge_cols = {"record_key", "record_index"} | judge_columns
    missing_proxy = (required_proxy_cols - set(proxy_df.columns)) if proxy_columns else set()
    missing_judge = required_judge_cols - set(judge_df.columns)
    if missing_proxy or missing_judge:
        return pd.DataFrame()

    proxy_keep = ["record_key", "record_index"] + sorted(proxy_columns)
    judge_keep = ["record_key", "record_index"] + sorted(judge_columns)
    judge_df = judge_df[judge_keep].copy()
    for col in judge_columns:
        judge_df[col] = pd.to_numeric(judge_df[col], errors="coerce")

    if proxy_columns:
        proxy_df = proxy_df[proxy_keep].copy()
        for col in proxy_columns:
            proxy_df[col] = pd.to_numeric(proxy_df[col], errors="coerce")
        merged = proxy_df.merge(judge_df, on="record_key", how="inner")
        merged = merged.dropna(subset=sorted(proxy_columns | judge_columns))
    else:
        merged = judge_df.dropna(subset=sorted(judge_columns)).copy()
    return merged


def build_system_eval_df(
    task: str, outputs_dir: Path, proxy_df: pd.DataFrame, rank_column: str
) -> pd.DataFrame | None:
    output_path = outputs_dir / f"{task}_outputs.csv"
    if not output_path.exists():
        return None

    out_df = pd.read_csv(output_path)
    if "used_oracle" not in out_df.columns:
        return None

    row_index = (
        pd.to_numeric(out_df["record_index"], errors="coerce")
        if "record_index" in out_df.columns
        else pd.Series(np.arange(len(out_df)), index=out_df.index, dtype=float)
    )
    used_oracle = out_df["used_oracle"].astype(bool)

    proxy_for_join = (
        proxy_df[["record_index", rank_column, "semantic_f1"]]
        .groupby("record_index", as_index=False)
        .first()
    )

    system_df = pd.DataFrame(
        {
            "record_index": row_index,
            "used_oracle": used_oracle.astype(float),
        }
    )
    system_df = system_df.dropna(subset=["record_index"])
    system_df["record_index"] = system_df["record_index"].astype(int)
    merged = system_df.merge(proxy_for_join, on="record_index", how="inner")
    merged = merged.dropna(subset=[rank_column, "semantic_f1"])
    merged["system_score"] = np.where(merged["used_oracle"].astype(bool), 1.0, merged["semantic_f1"])
    return merged


def infer_tasks(proxy_cache_dir: Path, judge_cache_dir: Path) -> list[str]:
    task_names: set[str] = set()
    if proxy_cache_dir.exists():
        task_names.update(path.stem for path in proxy_cache_dir.glob("*.jsonl"))
    if judge_cache_dir.exists():
        task_names.update(path.stem for path in judge_cache_dir.glob("*.jsonl"))
    return sorted(task_names)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Report tail metrics from proxy/judge caches: accuracy by confidence and/or latency by judge score."
        )
    )
    parser.add_argument(
        "--tasks",
        default="all",
        help="Comma-separated task names or 'all'. Default: all tasks found in proxy cache dir.",
    )
    parser.add_argument(
        "--tail-fracs",
        default="0.01,0.05,0.10",
        help="Comma-separated fractions (e.g. 0.1) or percentages (e.g. 10). Default: 0.01,0.05,0.10",
    )
    parser.add_argument(
        "--proxy-cache-dir",
        default="cache/proxy",
        help="Directory containing per-task proxy JSONL cache files.",
    )
    parser.add_argument(
        "--judge-cache-dir",
        default="cache/judge",
        help="Directory containing per-task judge JSONL cache files.",
    )
    parser.add_argument(
        "--outputs-dir",
        default="results",
        help="Directory containing BARGAIN run output CSV files (e.g. task_outputs.csv).",
    )
    parser.add_argument(
        "--score-column",
        default="score_bargain",
        help="Proxy confidence column in proxy cache to use for tail ranking.",
    )
    parser.add_argument(
        "--accuracy-tail-by",
        default="semantic_f1",
        help=(
            "Column used to rank rows when defining accuracy tails. "
            "Use semantic_f1 to rank only by judge score (default)."
        ),
    )
    parser.add_argument(
        "--report-mode",
        choices=["accuracy", "latency", "both"],
        default="accuracy",
        help="Which report to generate. Default: accuracy.",
    )
    parser.add_argument(
        "--latency-tail-by",
        default="semantic_f1",
        help="Column used to rank rows when defining latency tail. Default: semantic_f1",
    )
    parser.add_argument(
        "--latency-col",
        default="judge_seconds",
        help="Latency column to aggregate for latency report. Default: judge_seconds",
    )
    parser.add_argument(
        "--latency-quantiles",
        default="0.95",
        help="Latency quantiles to report (fractions or percentages). Default: 0.95",
    )
    parser.add_argument(
        "--report-csv",
        default="results/tail_accuracy_report.csv",
        help="Output CSV path for the tail report.",
    )
    args = parser.parse_args()

    proxy_cache_dir = Path(args.proxy_cache_dir)
    judge_cache_dir = Path(args.judge_cache_dir)
    outputs_dir = Path(args.outputs_dir)
    report_csv_path = Path(args.report_csv)
    tail_fracs = parse_tail_fracs(args.tail_fracs)
    latency_quantiles = parse_quantiles_arg(args.latency_quantiles)

    default_tasks = infer_tasks(proxy_cache_dir, judge_cache_dir)
    if not default_tasks:
        raise FileNotFoundError(
            f"No cache files found in proxy dir {proxy_cache_dir} or judge dir {judge_cache_dir}"
        )
    tasks = parse_tasks_arg(args.tasks, default_tasks)

    proxy_columns: set[str] = set()
    judge_columns: set[str] = set()
    if args.report_mode in {"accuracy", "both"}:
        judge_columns.add("semantic_f1")
        if args.accuracy_tail_by == "semantic_f1":
            pass
        elif args.accuracy_tail_by == args.score_column:
            proxy_columns.add(args.accuracy_tail_by)
        else:
            judge_columns.add(args.accuracy_tail_by)
    if args.report_mode in {"latency", "both"}:
        if args.latency_tail_by == "record_index":
            raise ValueError("--latency-tail-by cannot be 'record_index'. Use a score-like numeric column.")
        if args.latency_tail_by == args.score_column:
            proxy_columns.add(args.latency_tail_by)
        else:
            judge_columns.add(args.latency_tail_by)
        judge_columns.add(args.latency_col)

    rows: list[dict[str, Any]] = []
    for task in tasks:
        proxy_eval = build_proxy_eval_df(
            task=task,
            proxy_cache_dir=proxy_cache_dir,
            judge_cache_dir=judge_cache_dir,
            proxy_columns=proxy_columns,
            judge_columns=judge_columns,
        )
        if proxy_eval.empty:
            print(f"[{task}] skipped: no overlapping records between proxy and judge caches.")
            continue

        if args.report_mode in {"accuracy", "both"}:
            if args.accuracy_tail_by not in proxy_eval.columns:
                print(f"[{task}] skipped accuracy: tail column '{args.accuracy_tail_by}' not found.")
                continue

            proxy_conf = proxy_eval[args.accuracy_tail_by].to_numpy(dtype=float)
            proxy_correct = proxy_eval["semantic_f1"].to_numpy(dtype=float)

            system_eval = build_system_eval_df(
                task=task,
                outputs_dir=outputs_dir,
                proxy_df=proxy_eval,
                rank_column=args.accuracy_tail_by,
            )
            has_system = system_eval is not None and not system_eval.empty
            if has_system and system_eval is not None:
                system_conf = system_eval[args.accuracy_tail_by].to_numpy(dtype=float)
                system_correct = system_eval["system_score"].to_numpy(dtype=float)
                system_oracle_flags = system_eval["used_oracle"].to_numpy(dtype=float)
            else:
                system_conf = np.array([])
                system_correct = np.array([])
                system_oracle_flags = np.array([])

            for frac in tail_fracs:
                proxy_metrics = compute_tail_metrics(proxy_conf, proxy_correct, frac)
                rows.append(
                    {
                        "task": task,
                        "variant": "proxy",
                        "report_mode": "accuracy",
                        "tail_frac": frac,
                        "tail_by_column": args.accuracy_tail_by,
                        "n_records": int(proxy_metrics["n_records"]),
                        "tail_count": int(proxy_metrics["tail_count"]),
                        "overall_accuracy": proxy_metrics["overall_accuracy"],
                        "tail_accuracy": proxy_metrics["tail_accuracy"],
                        "tail_threshold_max_rank_value": proxy_metrics["tail_threshold_max_confidence"],
                        "oracle_fraction_in_tail": np.nan,
                    }
                )

                if has_system:
                    system_metrics = compute_tail_metrics(system_conf, system_correct, frac)
                    k = int(system_metrics["tail_count"])
                    order = np.argsort(system_conf, kind="mergesort")
                    tail_idx = order[:k]
                    rows.append(
                        {
                            "task": task,
                            "variant": "system",
                            "report_mode": "accuracy",
                            "tail_frac": frac,
                            "tail_by_column": args.accuracy_tail_by,
                            "n_records": int(system_metrics["n_records"]),
                            "tail_count": int(system_metrics["tail_count"]),
                            "overall_accuracy": system_metrics["overall_accuracy"],
                            "tail_accuracy": system_metrics["tail_accuracy"],
                            "tail_threshold_max_rank_value": system_metrics["tail_threshold_max_confidence"],
                            "oracle_fraction_in_tail": float(np.mean(system_oracle_flags[tail_idx])),
                        }
                    )

        if args.report_mode in {"latency", "both"}:
            if args.latency_tail_by not in proxy_eval.columns:
                print(f"[{task}] skipped latency: tail column '{args.latency_tail_by}' not found.")
                continue
            if args.latency_col not in proxy_eval.columns:
                print(f"[{task}] skipped latency: latency column '{args.latency_col}' not found.")
                continue

            rank_vals = proxy_eval[args.latency_tail_by].to_numpy(dtype=float)
            latency_vals = proxy_eval[args.latency_col].to_numpy(dtype=float)

            for frac in tail_fracs:
                latency_metrics = compute_tail_latency_metrics(
                    rank_values=rank_vals,
                    latency_values=latency_vals,
                    tail_frac=frac,
                    quantiles=latency_quantiles,
                )
                row: dict[str, Any] = {
                    "task": task,
                    "variant": "judge_latency",
                    "report_mode": "latency",
                    "tail_frac": frac,
                    "tail_by_column": args.latency_tail_by,
                    "latency_column": args.latency_col,
                    "n_records": int(latency_metrics["n_records"]),
                    "tail_count": int(latency_metrics["tail_count"]),
                    "tail_threshold_max_rank_value": latency_metrics[
                        "tail_threshold_max_rank_value"
                    ],
                    "overall_latency_mean": latency_metrics["overall_latency_mean"],
                    "tail_latency_mean": latency_metrics["tail_latency_mean"],
                }
                for q in latency_quantiles:
                    key = f"latency_p{int(round(100 * q)):02d}"
                    row[f"overall_{key}"] = latency_metrics[f"overall_{key}"]
                    row[f"tail_{key}"] = latency_metrics[f"tail_{key}"]
                rows.append(row)

    if not rows:
        raise ValueError("No report rows generated. Check task names and cache files.")

    report_df = pd.DataFrame(rows)
    report_df = report_df.sort_values(["task", "report_mode", "variant", "tail_frac"]).reset_index(
        drop=True
    )

    # Keep final CSV focused on core metrics.
    drop_cols = {
        "variant",
        "report_mode",
        "tail_threshold_max_rank_value",
        "oracle_fraction_in_tail",
    }
    final_report_df = report_df.drop(columns=[c for c in drop_cols if c in report_df.columns])

    report_csv_path.parent.mkdir(parents=True, exist_ok=True)
    final_report_df.to_csv(report_csv_path, index=False)

    print("\nTail report")
    print(f"- tasks: {tasks}")
    print(f"- tail_fracs: {tail_fracs}")
    print(f"- report_mode: {args.report_mode}")
    if args.report_mode in {"accuracy", "both"}:
        print(f"- accuracy_tail_by: {args.accuracy_tail_by}")
    if args.report_mode in {"latency", "both"}:
        print(f"- latency_tail_by: {args.latency_tail_by}")
        print(f"- latency_col: {args.latency_col}")
        print(f"- latency_quantiles: {latency_quantiles}")
    print(f"- saved: {report_csv_path}")
    print("\n" + final_report_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
