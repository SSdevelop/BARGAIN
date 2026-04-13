import csv
import os
import time
import numpy as np
import pandas as pd
import dspy

from BARGAIN import OpenAIProxy, OpenAIOracle, DSPySemanticJudge
from BARGAIN import BARGAIN_A

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

# --- Configure DSPy LM for the semantic judge ---
dspy.configure(lm=dspy.LM(judge_model, api_base=judge_api_base, api_key=oracle_api_key))

# --- Define task ---
task = '''
I will give you a Supreme Court opinion.
Your task is to provide a concise summary of the opinion, capturing the key legal issue,
the court's reasoning, and the final holding.

Here is the opinion: {}

Provide only the summary, nothing else:
'''

# --- Load data (use a subset for testing) ---
df = pd.read_csv('examples/court_opinion.csv').head(1000)

# --- Define oracle, proxy, and judge ---
proxy = OpenAIProxy(task, model=proxy_model, base_url=proxy_api_base, api_key=proxy_api_key,
                    max_tokens=50000, extra_body=no_think, max_workers=8)
oracle = OpenAIOracle(task, model=oracle_model, base_url=oracle_api_base, api_key=oracle_api_key,
                      max_tokens=50000, judge=DSPySemanticJudge(), extra_body=no_think, max_workers=8)

# --- Run BARGAIN ---
bargain = BARGAIN_A(proxy, oracle, target=0.9, delta=0.1, seed=0)
t0 = time.time()
output, used_oracle = bargain.process(df['opinion_text'].to_numpy(), return_oracle_usage=True)
elapsed = time.time() - t0
df['output'] = output
df['used_oracle'] = used_oracle
proxy_fraction = 1 - df['used_oracle'].mean()

# --- Evaluate: run oracle on all records and compute semantic F1 ---
print('Running oracle on all records for evaluation')
df['output_oracle'] = df['output']
proxy_mask = ~df['used_oracle']
if proxy_mask.any():
    df.loc[proxy_mask, 'output_oracle'] = oracle.get_pred(
        df.loc[proxy_mask, 'opinion_text'].to_numpy()
    )

judge = DSPySemanticJudge()
scores = []
for _, row in df.iterrows():
    score = judge(task, row['opinion_text'], row['output_oracle'], row['output'])
    scores.append(score)
df['semantic_f1'] = scores

os.makedirs('results', exist_ok=True)
df.to_csv('results/court_opinion_summarization_outputs.csv', index=False)

mean_f1 = df['semantic_f1'].mean()
results_csv = 'results.csv'
write_header = not os.path.exists(results_csv)
with open(results_csv, 'a', newline='') as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow(['script', 'metric', 'score', 'proxy_fraction', 'time_seconds'])
    writer.writerow(['court_opinion_summarization', 'semantic_f1', f'{mean_f1:.3f}', f'{proxy_fraction:.2f}', f'{elapsed:.1f}'])
