import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import dspy
import pandas as pd
from openai import OpenAI

from BARGAIN import DSPySemanticJudge


NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}


@dataclass(frozen=True)
class TaskConfig:
    name: str
    csv_path: str
    text_column: str
    prompt_template: str
    max_tokens: int


TASKS: dict[str, TaskConfig] = {
    "enron_filter": TaskConfig(
        name="enron_filter",
        csv_path="examples/enron.csv",
        text_column="text",
        prompt_template="""
I will give you an email from the Enron corpus.
Classify the sender into one of these two labels:

- C-suite
- VP-level executive

Here is the email: {}

Respond with ONLY one label from the list above.
""",
        max_tokens=10000,
    ),
    "review_extract_praised_games": TaskConfig(
        name="review_extract_praised_games",
        csv_path="examples/review_per_row_top10_filtered.csv",
        text_column="review_text",
        prompt_template="""
I will give you a video game review. The review is for a specific game.
Your task is to extract the names of any OTHER games that the reviewer praises
or speaks of more favorably than the game being reviewed.

If the reviewer does not praise any other game more than the one being reviewed,
respond with ONLY: None

Otherwise, respond with ONLY a comma-separated list of the praised game names.

Here is the review: {}

Your response:
""",
        max_tokens=10000,
    ),
    "court_reverse": TaskConfig(
        name="court_reverse",
        csv_path="examples/court_opinion.csv",
        text_column="opinion_text",
        prompt_template="""
I will give you a U.S. Supreme Court opinion.
Explain why or why not this opinion reverses the lower-court ruling.

Here is the opinion: {}

Provide only your explanation, nothing else:
""",
        max_tokens=10000,
    ),
    "legal_doc_extract": TaskConfig(
        name="legal_doc_extract",
        csv_path="examples/legal_cuad.csv",
        text_column="document",
        prompt_template="""
I will give you a license agreement or legal document.
Your task is to extract any covenants not to sue or IP no-challenge clauses.

If no such clauses are found, respond with ONLY: None

Otherwise, respond with ONLY the extracted clause text, separated by newlines.

Here is the document: {}

Your response:
""",
        max_tokens=10000,
    ),
    "court_opinion_summarization": TaskConfig(
        name="court_opinion_summarization",
        csv_path="examples/court_opinion.csv",
        text_column="opinion_text",
        prompt_template="""
I will give you a Supreme Court opinion.
Your task is to provide a concise summary of the opinion, capturing the key legal issue,
the court's reasoning, and the final holding.

Here is the opinion: {}

Provide only the summary, nothing else:
""",
        max_tokens=10000,
    ),
    "enron_extract_emails": TaskConfig(
        name="enron_extract_emails",
        csv_path="examples/enron.csv",
        text_column="text",
        prompt_template="""
I will give you an Enron email document.
Extract all email addresses that appear in the document.

Return ONLY a comma-separated list of unique email addresses in order of first appearance.
If no email address is present, return ONLY: None

Here is the document: {}
""",
        max_tokens=10000,
    ),
    "wiki_talk_extract_usernames": TaskConfig(
        name="wiki_talk_extract_usernames",
        csv_path="examples/wiki_talk.csv",
        text_column="document",
        prompt_template="""
I will give you a Wikipedia talk page conversation.
Each message starts with a header like:
[YYYY-MM-DD HH:MM:SS] Username: message text

Extract all unique speaker usernames from these message headers only.
Return ONLY a comma-separated list of unique usernames in order of first appearance.
If no username is present, return ONLY: None

Here is the conversation: {}
""",
        max_tokens=10000,
    ),
    "wiki_talk_first_timestamp": TaskConfig(
        name="wiki_talk_first_timestamp",
        csv_path="examples/wiki_talk.csv",
        text_column="document",
        prompt_template="""
I will give you a Wikipedia talk page conversation.
Each message starts with a header like:
[YYYY-MM-DD HH:MM:SS] Username: message text

Extract the timestamp of the first message in the conversation.
Return ONLY the timestamp in exactly this format: YYYY-MM-DD HH:MM:SS
If no timestamp is present, return ONLY: None

Here is the conversation: {}
""",
        max_tokens=10000,
    ),
    "wiki_talk_last_timestamp": TaskConfig(
        name="wiki_talk_last_timestamp",
        csv_path="examples/wiki_talk.csv",
        text_column="document",
        prompt_template="""
I will give you a Wikipedia talk page conversation.
Each message starts with a header like:
[YYYY-MM-DD HH:MM:SS] Username: message text

Extract the timestamp of the last message in the conversation.
Return ONLY the timestamp in exactly this format: YYYY-MM-DD HH:MM:SS
If no timestamp is present, return ONLY: None

Here is the conversation: {}
""",
        max_tokens=10000,
    ),
    "random_pubmed_articles_classification": TaskConfig(
        name="random_pubmed_articles_classification",
        csv_path="examples/random-pubmed-articles.csv",
        text_column="article",
        prompt_template="""
I will give you a biomedical article.
Classify it into one of the following categories:

- RCT
- Observational
- Meta-analysis
- Bench/Lab
- Computational
- Review

Here is the article: {}

Respond with ONLY one label from the list above.
""",
        max_tokens=10000,
    ),
}


def sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parse_tasks_arg(tasks_arg: str) -> list[str]:
    raw = tasks_arg.strip()
    if raw.lower() == "all":
        return list(TASKS.keys())

    requested = [name.strip() for name in raw.split(",") if name.strip()]
    if not requested:
        raise ValueError("No task names provided.")

    invalid = [name for name in requested if name not in TASKS]
    if invalid:
        valid = ", ".join(TASKS.keys())
        raise ValueError(f"Unknown tasks: {invalid}. Valid task names: {valid}")

    return list(dict.fromkeys(requested))


def load_existing_cache(cache_file: Path) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    if not cache_file.exists():
        return entries

    with cache_file.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            payload = line.strip()
            if not payload:
                continue
            try:
                entry = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Failed to parse JSON at {cache_file}:{line_num}"
                ) from exc
            record_key = entry.get("record_key")
            if not isinstance(record_key, str) or not record_key:
                raise ValueError(
                    f"Missing/invalid record_key in {cache_file}:{line_num}"
                )
            entries[record_key] = entry
    return entries


def write_cache(cache_file: Path, entries_by_key: dict[str, dict[str, Any]]) -> None:
    with cache_file.open("w", encoding="utf-8") as f:
        for entry in entries_by_key.values():
            f.write(json.dumps(entry, ensure_ascii=False))
            f.write("\n")


def extract_usage_totals(usage: Any) -> tuple[int, int, int]:
    if usage is None:
        return 0, 0, 0

    def _get_value(obj: Any, *names: str) -> int:
        for name in names:
            if isinstance(obj, dict) and name in obj:
                val = obj.get(name)
            else:
                val = getattr(obj, name, None)
            if val is not None:
                try:
                    return int(val)
                except (TypeError, ValueError):
                    continue
        return 0

    input_tokens = _get_value(usage, "prompt_tokens", "input_tokens")
    output_tokens = _get_value(usage, "completion_tokens", "output_tokens")
    total_tokens = _get_value(usage, "total_tokens")
    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens
    return input_tokens, output_tokens, total_tokens


def _safe_exp(log_value: float) -> float:
    # exp(logprob) converts back to probability space in [0, 1].
    if log_value <= -745:
        return 0.0
    return float(math.exp(log_value))


def compute_logprob_metrics(token_logprobs: list[float]) -> dict[str, Any]:
    if not token_logprobs:
        return {
            "token_count": 0,
            "positive_logprob_count_clamped": 0,
            "sum_logprob": None,
            "mean_logprob": None,
            "min_logprob": None,
            "max_logprob": None,
            "score_bargain": None,
            "score_avg": None,
            "score_min": None,
            "score_max": None,
        }

    clamped_logprobs: list[float] = []
    positive_count = 0
    for logprob in token_logprobs:
        lp = float(logprob)
        if lp > 0:
            positive_count += 1
            lp = 0.0
        clamped_logprobs.append(lp)

    sum_logprob = float(sum(clamped_logprobs))
    mean_logprob = sum_logprob / len(clamped_logprobs)
    min_logprob = float(min(clamped_logprobs))
    max_logprob = float(max(clamped_logprobs))

    return {
        "token_count": len(clamped_logprobs),
        "positive_logprob_count_clamped": int(positive_count),
        "sum_logprob": sum_logprob,
        "mean_logprob": mean_logprob,
        "min_logprob": min_logprob,
        "max_logprob": max_logprob,
        # BARGAIN-style: product of token probabilities = exp(sum logprob).
        "score_bargain": _safe_exp(sum_logprob),
        # Length-normalized confidence: geometric mean token probability.
        "score_avg": _safe_exp(mean_logprob),
        # Weakest/strongest token probabilities.
        "score_min": _safe_exp(min_logprob),
        "score_max": _safe_exp(max_logprob),
    }


def call_proxy(
    client: OpenAI,
    model: str,
    prompt_template: str,
    record_text: str,
    max_tokens: int,
    extra_body: dict[str, Any] | None,
) -> dict[str, Any]:
    prompt = [
        {"role": "system", "content": "You are a helpful assistant that is good at processing data."},
        {"role": "user", "content": prompt_template.format(record_text)},
    ]

    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=cast(Any, prompt),
        logprobs=True,
        seed=0,
        temperature=0,
        max_tokens=max_tokens,
        extra_body=extra_body,
    )
    elapsed = time.perf_counter() - t0

    output_text = response.choices[0].message.content
    if output_text is None:
        output_text = ""

    usage_input_tokens, usage_output_tokens, usage_total_tokens = extract_usage_totals(response.usage)

    token_logprobs: list[float] = []
    response_logprobs = response.choices[0].logprobs
    if response_logprobs is not None and response_logprobs.content is not None:
        for token_info in response_logprobs.content:
            token_lp = getattr(token_info, "logprob", None)
            if token_lp is None:
                continue
            token_logprobs.append(float(token_lp))

    metrics = compute_logprob_metrics(token_logprobs)
    response_model = getattr(response, "model", None) or model

    return {
        "proxy_output": output_text,
        "proxy_seconds": float(elapsed),
        "proxy_input_tokens": int(usage_input_tokens),
        "proxy_output_tokens": int(usage_output_tokens),
        "proxy_total_tokens": int(usage_total_tokens),
        "proxy_model": str(response_model),
        **metrics,
    }


def process_task(
    config: TaskConfig,
    limit: int,
    proxy_client: OpenAI,
    proxy_model: str,
    oracle_cache_dir: Path,
    proxy_cache_dir: Path,
    judge_cache_dir: Path,
    judge: DSPySemanticJudge,
    judge_model: str,
    overwrite_proxy: bool,
    overwrite_judge: bool,
    resume: bool,
    disable_thinking: bool,
) -> None:
    proxy_cache_dir.mkdir(parents=True, exist_ok=True)
    judge_cache_dir.mkdir(parents=True, exist_ok=True)

    oracle_cache_path = oracle_cache_dir / f"{config.name}.jsonl"
    if not oracle_cache_path.exists():
        raise FileNotFoundError(
            f"Oracle cache not found for task '{config.name}': {oracle_cache_path}"
        )
    oracle_by_key = load_existing_cache(oracle_cache_path)

    proxy_cache_path = proxy_cache_dir / f"{config.name}.jsonl"
    judge_cache_path = judge_cache_dir / f"{config.name}.jsonl"

    proxy_by_key = {} if overwrite_proxy else load_existing_cache(proxy_cache_path)
    judge_by_key = {} if overwrite_judge else load_existing_cache(judge_cache_path)

    df = pd.read_csv(config.csv_path).head(limit)
    texts = df[config.text_column].astype(str).tolist()

    print(
        f"[{config.name}] records={len(texts)} proxy_cached={len(proxy_by_key)} "
        f"judge_cached={len(judge_by_key)}"
    )

    missing_oracle = 0
    for record_text in texts:
        record_key = sha256_hex(f"{config.name}\n{record_text}")
        if record_key not in oracle_by_key:
            missing_oracle += 1
    if missing_oracle > 0:
        raise ValueError(
            f"[{config.name}] oracle cache missing {missing_oracle} records for requested limit={len(texts)}."
        )

    prompt_hash = sha256_hex(config.prompt_template)
    extra_body = NO_THINK if disable_thinking else None

    proxy_built = 0
    proxy_skipped = 0
    for record_index, record_text in enumerate(texts):
        record_key = sha256_hex(f"{config.name}\n{record_text}")
        if resume and record_key in proxy_by_key:
            proxy_skipped += 1
            continue

        record_text_hash = sha256_hex(record_text)
        proxy_result = call_proxy(
            client=proxy_client,
            model=proxy_model,
            prompt_template=config.prompt_template,
            record_text=record_text,
            max_tokens=config.max_tokens,
            extra_body=extra_body,
        )
        proxy_by_key[record_key] = {
            "task_name": config.name,
            "task_prompt_hash": prompt_hash,
            "record_index": int(record_index),
            "record_key": record_key,
            "record_text_hash": record_text_hash,
            **proxy_result,
        }
        proxy_built += 1
        if proxy_built % 10 == 0:
            print(
                f"[{config.name}] proxy processed={proxy_built} skipped={proxy_skipped} "
                f"cached_total={len(proxy_by_key)}"
            )

    write_cache(proxy_cache_path, proxy_by_key)
    print(
        f"[{config.name}] proxy done. wrote={proxy_built} skipped={proxy_skipped} "
        f"total_cached={len(proxy_by_key)} file={proxy_cache_path}"
    )

    judge_built = 0
    judge_skipped = 0
    for record_index, record_text in enumerate(texts):
        record_key = sha256_hex(f"{config.name}\n{record_text}")
        if resume and record_key in judge_by_key:
            judge_skipped += 1
            continue
        if record_key not in proxy_by_key:
            raise ValueError(
                f"[{config.name}] proxy cache missing record_key={record_key}; cannot score judge."
            )
        if record_key not in oracle_by_key:
            raise ValueError(
                f"[{config.name}] oracle cache missing record_key={record_key}; cannot score judge."
            )

        proxy_output = proxy_by_key[record_key].get("proxy_output", "")
        oracle_output = oracle_by_key[record_key].get("oracle_output", "")

        before = judge.get_timing_stats()
        score = judge(
            query=config.prompt_template,
            document=record_text,
            oracle_output=str(oracle_output),
            proxy_output=str(proxy_output),
        )
        after = judge.get_timing_stats()

        judge_by_key[record_key] = {
            "task_name": config.name,
            "task_prompt_hash": prompt_hash,
            "record_index": int(record_index),
            "record_key": record_key,
            "record_text_hash": sha256_hex(record_text),
            "proxy_model": proxy_by_key[record_key].get("proxy_model", proxy_model),
            "judge_model": judge_model,
            "semantic_f1": float(score),
            "judge_seconds": float(max(0.0, after["total_seconds"] - before["total_seconds"])),
            "judge_input_tokens": int(max(0, after["input_tokens"] - before["input_tokens"])),
            "judge_output_tokens": int(max(0, after["output_tokens"] - before["output_tokens"])),
            "judge_total_tokens": int(max(0, after["total_tokens"] - before["total_tokens"])),
        }
        judge_built += 1
        if judge_built % 10 == 0:
            print(
                f"[{config.name}] judge processed={judge_built} skipped={judge_skipped} "
                f"cached_total={len(judge_by_key)}"
            )

    write_cache(judge_cache_path, judge_by_key)
    print(
        f"[{config.name}] judge done. wrote={judge_built} skipped={judge_skipped} "
        f"total_cached={len(judge_by_key)} file={judge_cache_path}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect proxy logprob metrics (BARGAIN/avg/min/max variants) and semantic F1 "
            "judge scores against oracle caches for benchmark tasks."
        )
    )
    parser.add_argument(
        "--tasks",
        default="all",
        help="Comma-separated task names or 'all'. Default: all.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Max rows per task. Default: 1000.",
    )
    parser.add_argument(
        "--oracle-cache-dir",
        default="cache/oracle",
        help="Input oracle cache directory. Default: cache/oracle.",
    )
    parser.add_argument(
        "--proxy-cache-dir",
        default="cache/proxy",
        help="Output proxy metrics cache directory. Default: cache/proxy.",
    )
    parser.add_argument(
        "--judge-cache-dir",
        default="cache/judge",
        help="Output judge score cache directory. Default: cache/judge.",
    )
    parser.add_argument(
        "--proxy-model",
        default="Qwen/Qwen3.5-35B-A3B",
        help="Proxy model name.",
    )
    parser.add_argument(
        "--proxy-api-base",
        default="http://localhost:8012/v1",
        help="Proxy API base URL.",
    )
    parser.add_argument(
        "--proxy-api-key",
        default="EMPTY",
        help="Proxy API key.",
    )
    parser.add_argument(
        "--judge-model",
        default="openai/Qwen/Qwen3.5-122B-A10B",
        help="DSPy judge model ID.",
    )
    parser.add_argument(
        "--judge-api-base",
        default="http://localhost:8011/v1",
        help="Judge API base URL.",
    )
    parser.add_argument(
        "--judge-api-key",
        default="EMPTY",
        help="Judge API key.",
    )
    parser.add_argument(
        "--overwrite-proxy",
        action="store_true",
        help="Rebuild proxy cache files from scratch.",
    )
    parser.add_argument(
        "--overwrite-judge",
        action="store_true",
        help="Rebuild judge cache files from scratch.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip records already present in cache files. Default: true.",
    )
    parser.add_argument(
        "--disable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass no-thinking extra_body to proxy calls. Default: true.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be a positive integer.")

    task_names = parse_tasks_arg(args.tasks)
    oracle_cache_dir = Path(args.oracle_cache_dir)
    proxy_cache_dir = Path(args.proxy_cache_dir)
    judge_cache_dir = Path(args.judge_cache_dir)

    dspy.configure(
        lm=dspy.LM(
            args.judge_model,
            api_base=args.judge_api_base,
            api_key=args.judge_api_key,
            extra_body=NO_THINK if args.disable_thinking else None
        )
    )
    judge = DSPySemanticJudge()

    proxy_client = OpenAI(
        api_key=args.proxy_api_key,
        base_url=args.proxy_api_base,
    )

    print(
        "Collecting proxy metrics + semantic F1 caches "
        f"for tasks={task_names}, limit={args.limit}, "
        f"oracle_cache_dir={oracle_cache_dir}, proxy_cache_dir={proxy_cache_dir}, "
        f"judge_cache_dir={judge_cache_dir}, resume={args.resume}"
    )

    for task_name in task_names:
        process_task(
            config=TASKS[task_name],
            limit=args.limit,
            proxy_client=proxy_client,
            proxy_model=args.proxy_model,
            oracle_cache_dir=oracle_cache_dir,
            proxy_cache_dir=proxy_cache_dir,
            judge_cache_dir=judge_cache_dir,
            judge=judge,
            judge_model=args.judge_model,
            overwrite_proxy=args.overwrite_proxy,
            overwrite_judge=args.overwrite_judge,
            resume=args.resume,
            disable_thinking=args.disable_thinking,
        )


if __name__ == "__main__":
    main()
