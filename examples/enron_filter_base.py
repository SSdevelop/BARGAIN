import csv
import os
import time

import pandas as pd

from BARGAIN import BARGAIN_A, OpenAIOracle, OpenAIProxy
from benchmark_metrics import append_results_row, compute_tail_accuracies, save_outputs_df


# --- Model config ---
oracle_model = "Qwen/Qwen3.5-122B-A10B"
oracle_api_base = "http://localhost:8011/v1"
oracle_api_key = "EMPTY"

proxy_model = "Qwen/Qwen3.5-35B-A3B"
proxy_api_base = "http://localhost:8012/v1"
proxy_api_key = "EMPTY"

no_think = {"chat_template_kwargs": {"enable_thinking": False}}
oracle_cache_path = "cache/oracle/enron_filter.jsonl"
oracle_cache_task_name = "enron_filter"

# --- Define task ---
task = """
I will give you an email from the Enron corpus.
Classify the sender into one of these two labels:

- C-suite
- VP-level executive

Here is the email: {}

Respond with ONLY one label from the list above.
"""

# --- Load data (subset for testing) ---
df = pd.read_csv("examples/enron.csv").head(1000)

# --- Define oracle and proxy (no judge) ---
proxy = OpenAIProxy(
    task,
    model=proxy_model,
    base_url=proxy_api_base,
    api_key=proxy_api_key,
    max_tokens=10000,
    extra_body=no_think,
    max_workers=8,
)
oracle = OpenAIOracle(
    task,
    model=oracle_model,
    base_url=oracle_api_base,
    api_key=oracle_api_key,
    max_tokens=10000,
    extra_body=no_think,
    cache_path=oracle_cache_path,
    cache_task_name=oracle_cache_task_name,
    max_workers=8,
)

# --- Run BARGAIN map query ---
bargain = BARGAIN_A(proxy, oracle, target=0.9, delta=0.1, seed=0)
t0 = time.time()
output, used_oracle = bargain.process(df["text"].astype(str).tolist(), return_oracle_usage=True)
elapsed = time.time() - t0
proxy_timing = proxy.get_timing_stats()
oracle_timing = oracle.get_timing_stats()
df["output"] = output
df["used_oracle"] = pd.Series(used_oracle, index=df.index, dtype=bool)
df["oracle_record_seconds"] = [
    float(sec) if (used and sec is not None) else 0.0
    for used, sec in zip(
        df["used_oracle"],
        (oracle.get_cached_record_seconds(text) for text in df["text"].astype(str)),
    )
]

# --- Evaluate with exact oracle match (cache first, live fallback on miss) ---
df["output_oracle"] = df["output"]
proxy_mask = ~df["used_oracle"]
if bool(proxy_mask.any()):
    proxy_records = df.loc[proxy_mask, "text"].astype(str).tolist()
    cached_outputs = oracle.get_cached_record_outputs(proxy_records)
    missing_indices = [i for i, output in enumerate(cached_outputs) if output is None]
    if missing_indices:
        print(f"Cache miss for {len(missing_indices)} records during evaluation; using live oracle fallback")
        missing_records = [proxy_records[i] for i in missing_indices]
        live_outputs = oracle.get_pred(missing_records).tolist()
        for idx, live_output in zip(missing_indices, live_outputs):
            cached_outputs[idx] = live_output
    df.loc[proxy_mask, "output_oracle"] = [str(output) if output is not None else "" for output in cached_outputs]
df["is_correct"] = df["output"] == df["output_oracle"]

save_outputs_df(df, "results/enron_filter_base_outputs.csv")
accuracy = df["is_correct"].mean()
tail_accuracy_1pct, tail_accuracy_5pct, tail_accuracy_10pct = compute_tail_accuracies(df["is_correct"].astype(float).tolist())
proxy_fraction = 1 - df["used_oracle"].mean()

append_results_row(
    script_name="enron_filter_base",
    metric_name="accuracy",
    score=accuracy,
    proxy_fraction=proxy_fraction,
    elapsed_seconds=elapsed,
    proxy_timing=proxy_timing,
    oracle_timing=oracle_timing,
    judge_eval_seconds=0.0,
    judge_total_tokens=0,
    judge_eval_tokens=0,
    tail_accuracy_1pct=tail_accuracy_1pct,
    tail_accuracy_5pct=tail_accuracy_5pct,
    tail_accuracy_10pct=tail_accuracy_10pct,
)
print(f"Accuracy: {accuracy:.3f}, Used Proxy: {proxy_fraction:.2f}")
