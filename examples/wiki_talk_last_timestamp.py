import csv
import os
import time

import dspy
import pandas as pd

from BARGAIN import BARGAIN_A
from BARGAIN import DSPySemanticJudge, OpenAIOracle, OpenAIProxy
from benchmark_metrics import append_results_row, compute_tail_accuracies, save_outputs_df


# --- Model config ---
oracle_model = "Qwen/Qwen3.5-122B-A10B"
oracle_api_base = "http://localhost:8011/v1"
oracle_api_key = "EMPTY"

proxy_model = "Qwen/Qwen3.5-35B-A3B"
proxy_api_base = "http://localhost:8012/v1"
proxy_api_key = "EMPTY"

judge_model = f"openai/{oracle_model}"
judge_api_base = oracle_api_base

no_think = {"chat_template_kwargs": {"enable_thinking": False}}
oracle_cache_path = "cache/oracle/wiki_talk_last_timestamp.jsonl"
oracle_cache_task_name = "wiki_talk_last_timestamp"

# --- Configure DSPy LM for the semantic judge ---
dspy.configure(lm=dspy.LM(judge_model, api_base=judge_api_base, api_key=oracle_api_key, extra_body=no_think))

# --- Define task ---
task = """
I will give you a Wikipedia talk page conversation.
Each message starts with a header like:
[YYYY-MM-DD HH:MM:SS] Username: message text

Extract the timestamp of the last message in the conversation.
Return ONLY the timestamp in exactly this format: YYYY-MM-DD HH:MM:SS
If no timestamp is present, return ONLY: None

Here is the conversation: {}
"""

# --- Load data ---
df = pd.read_csv("examples/wiki_talk.csv").head(1000)

# --- Define oracle, proxy, and judge ---
judge = DSPySemanticJudge()
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
    judge=judge,
    max_tokens=10000,
    extra_body=no_think,
    max_workers=8,
    cache_path=oracle_cache_path,
    cache_task_name=oracle_cache_task_name,
)

# --- Run BARGAIN ---
bargain = BARGAIN_A(proxy, oracle, target=0.9, delta=0.1, seed=0)
t0 = time.time()
documents = df["document"].astype(str).tolist()
output, used_oracle = bargain.process(documents, return_oracle_usage=True)
elapsed = time.time() - t0
proxy_timing = proxy.get_timing_stats()
oracle_timing = oracle.get_timing_stats()
judge_stats_after_process = judge.get_timing_stats()
judge_seconds_after_process = judge_stats_after_process["total_seconds"]
judge_tokens_after_process = judge_stats_after_process["total_tokens"]
df["last_timestamp"] = output
df["used_oracle"] = pd.Series(used_oracle, index=df.index, dtype=bool)
df["oracle_record_seconds"] = [
    float(sec) if (used and sec is not None) else 0.0
    for used, sec in zip(
        df["used_oracle"],
        (oracle.get_cached_record_seconds(text) for text in documents),
    )
]
proxy_fraction = 1 - df["used_oracle"].mean()

# --- Evaluate: use cached oracle outputs for evaluation (live fallback on cache miss) ---
print("Resolving oracle outputs for evaluation from cache")
df["output_oracle"] = df["last_timestamp"]
proxy_mask = ~df["used_oracle"]
if bool(proxy_mask.any()):
    proxy_records = df.loc[proxy_mask, "document"].astype(str).tolist()
    cached_outputs = oracle.get_cached_record_outputs(proxy_records)
    missing_indices = [i for i, output in enumerate(cached_outputs) if output is None]
    if missing_indices:
        print(f"Cache miss for {len(missing_indices)} records during evaluation; using live oracle fallback")
        missing_records = [proxy_records[i] for i in missing_indices]
        live_outputs = oracle.get_pred(missing_records).tolist()
        for idx, live_output in zip(missing_indices, live_outputs):
            cached_outputs[idx] = live_output
    df.loc[proxy_mask, "output_oracle"] = [str(output) if output is not None else "" for output in cached_outputs]

scores = []
for _, row in df.iterrows():
    score = judge(task, str(row["document"]), str(row["output_oracle"]), str(row["last_timestamp"]))
    scores.append(score)
df["semantic_f1"] = scores
judge_stats_total = judge.get_timing_stats()
judge_seconds_total = judge_stats_total["total_seconds"]
judge_total_tokens = judge_stats_total["total_tokens"]
judge_eval_seconds = max(0.0, judge_seconds_total - judge_seconds_after_process)
judge_eval_tokens = max(0, judge_total_tokens - judge_tokens_after_process)

save_outputs_df(df, "results/wiki_talk_last_timestamp_outputs.csv")

mean_f1 = df["semantic_f1"].mean()
tail_accuracy_1pct, tail_accuracy_5pct, tail_accuracy_10pct = compute_tail_accuracies(df["semantic_f1"].astype(float).tolist())
append_results_row(
    script_name="wiki_talk_last_timestamp",
    metric_name="semantic_f1",
    score=mean_f1,
    proxy_fraction=proxy_fraction,
    elapsed_seconds=elapsed,
    proxy_timing=proxy_timing,
    oracle_timing=oracle_timing,
    judge_eval_seconds=judge_eval_seconds,
    judge_total_tokens=judge_total_tokens,
    judge_eval_tokens=judge_eval_tokens,
    tail_accuracy_1pct=tail_accuracy_1pct,
    tail_accuracy_5pct=tail_accuracy_5pct,
    tail_accuracy_10pct=tail_accuracy_10pct,
)
