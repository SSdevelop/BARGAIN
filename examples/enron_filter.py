import csv
import os
import time
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

# --- Define task (EnronExecutiveLevel) ---
task = '''
I will give you an email from the Enron corpus.
Classify the sender into one of these two labels:

- C-suite 
- VP-level executive

Here is the email: {}

Respond with ONLY one label from the list above.
'''

# --- Load data ---
# Expects a CSV with a 'document' column containing email text
df = pd.read_csv('examples/enron.csv').head(1000)

# --- Define oracle, proxy, and judge ---
proxy = OpenAIProxy(task, model=proxy_model, base_url=proxy_api_base, api_key=proxy_api_key,
                    max_tokens=16, extra_body=no_think, max_workers=8)
oracle = OpenAIOracle(task, model=oracle_model, base_url=oracle_api_base, api_key=oracle_api_key,
                      judge=DSPySemanticJudge(), max_tokens=16, extra_body=no_think, max_workers=8)

# --- Run BARGAIN ---
bargain = BARGAIN_A(proxy, oracle, target=0.9, delta=0.1, seed=0)
t0 = time.time()
documents = df['text'].astype(str).tolist()
output, used_oracle = bargain.process(documents, return_oracle_usage=True)
elapsed = time.time() - t0
df['sender_level'] = output
df['used_oracle'] = pd.Series(used_oracle, index=df.index, dtype=bool)
proxy_fraction = 1 - (sum(bool(v) for v in used_oracle) / len(used_oracle))

# --- Evaluate: run oracle on all records and compute semantic F1 ---
print('Running oracle on all records for evaluation')
df['output_oracle'] = df['sender_level']
proxy_mask = ~df['used_oracle'].astype(bool)
if bool(proxy_mask.any()):
    df.loc[proxy_mask, 'output_oracle'] = oracle.get_pred(
        df.loc[proxy_mask, 'text'].astype(str).tolist()
    )

judge = DSPySemanticJudge()
scores = []
for _, row in df.iterrows():
    score = judge(task, str(row['text']), str(row['output_oracle']), str(row['sender_level']))
    scores.append(score)
df['semantic_f1'] = scores

os.makedirs('results', exist_ok=True)
df.to_csv('results/enron_filter_outputs.csv', index=False)

mean_f1 = (sum(scores) / len(scores)) if scores else 0.0
results_csv = 'results.csv'
write_header = not os.path.exists(results_csv)
with open(results_csv, 'a', newline='') as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow(['script', 'metric', 'score', 'proxy_fraction', 'time_seconds'])
    writer.writerow(['enron_filter', 'semantic_f1', f'{mean_f1:.3f}', f'{proxy_fraction:.2f}', f'{elapsed:.1f}'])
