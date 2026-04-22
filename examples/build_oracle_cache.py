import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd

from BARGAIN import OpenAIOracle


ORACLE_MODEL = "Qwen/Qwen3.5-122B-A10B"
ORACLE_API_BASE = "http://localhost:8011/v1"
ORACLE_API_KEY = "EMPTY"
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

    # De-duplicate while preserving input order.
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


def call_oracle(
    oracle: "OpenAIOracle", prompt_template: str, record_text: str, max_tokens: int
) -> tuple[str, float, int, int, int, str]:
    prompt = [
        {"role": "system", "content": "You are a helpful assistant that is good at processing data."},
        {"role": "user", "content": prompt_template.format(record_text)},
    ]

    t0 = time.perf_counter()
    response = oracle.client.chat.completions.create(
        model=oracle.model,
        messages=cast(Any, prompt),
        logprobs=False,
        seed=0,
        temperature=0,
        max_tokens=max_tokens,
        extra_body=oracle.extra_body,
    )
    oracle_seconds = time.perf_counter() - t0

    output_text = response.choices[0].message.content
    if output_text is None:
        output_text = ""

    input_tokens, output_tokens, total_tokens = extract_usage_totals(response.usage)
    response_model = getattr(response, "model", None) or oracle.model
    return (
        output_text,
        float(oracle_seconds),
        int(input_tokens),
        int(output_tokens),
        int(total_tokens),
        str(response_model),
    )


def build_task_cache(
    config: TaskConfig, limit: int, out_dir: Path, overwrite: bool, resume: bool
) -> None:

    out_dir.mkdir(parents=True, exist_ok=True)
    cache_file = out_dir / f"{config.name}.jsonl"
    task_prompt_hash = sha256_hex(config.prompt_template)

    oracle = OpenAIOracle(
        config.prompt_template,
        model=ORACLE_MODEL,
        base_url=ORACLE_API_BASE,
        api_key=ORACLE_API_KEY,
        max_tokens=config.max_tokens,
        extra_body=NO_THINK,
        max_workers=8,
    )

    existing_entries: dict[str, dict[str, Any]] = {}
    if cache_file.exists() and not overwrite:
        existing_entries = load_existing_cache(cache_file)

    df = pd.read_csv(config.csv_path).head(limit)
    texts = df[config.text_column].astype(str).tolist()

    built_count = 0
    skipped_count = 0

    for row_idx, record_text in enumerate(texts):
        record_text_hash = sha256_hex(record_text)
        record_key = sha256_hex(f"{config.name}\n{record_text}")

        if resume and record_key in existing_entries:
            skipped_count += 1
            continue

        (
            oracle_output,
            oracle_seconds,
            oracle_input_tokens,
            oracle_output_tokens,
            oracle_total_tokens,
            oracle_model,
        ) = call_oracle(
            oracle=oracle,
            prompt_template=config.prompt_template,
            record_text=record_text,
            max_tokens=config.max_tokens,
        )

        existing_entries[record_key] = {
            "task_name": config.name,
            "task_prompt_hash": task_prompt_hash,
            "record_index": int(row_idx),
            "record_key": record_key,
            "record_text_hash": record_text_hash,
            "oracle_output": oracle_output,
            "oracle_seconds": oracle_seconds,
            "oracle_input_tokens": oracle_input_tokens,
            "oracle_output_tokens": oracle_output_tokens,
            "oracle_total_tokens": oracle_total_tokens,
            "oracle_model": oracle_model,
        }
        built_count += 1

        if built_count % 10 == 0:
            print(
                f"[{config.name}] processed={built_count} skipped={skipped_count} "
                f"cached_total={len(existing_entries)}"
            )

    write_cache(cache_file, existing_entries)
    print(
        f"[{config.name}] done. wrote={built_count} skipped={skipped_count} "
        f"total_cached={len(existing_entries)} file={cache_file}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build per-record oracle output + latency caches for benchmark tasks."
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
        "--out-dir",
        default="cache/oracle",
        help="Output directory for per-task JSONL cache files. Default: cache/oracle.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, rebuild task cache files from scratch.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip records already cached by record_key. Default: true.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be a positive integer.")

    task_names = parse_tasks_arg(args.tasks)
    out_dir = Path(args.out_dir)

    print(
        f"Building oracle cache for tasks={task_names}, limit={args.limit}, "
        f"out_dir={out_dir}, overwrite={args.overwrite}, resume={args.resume}"
    )

    for task_name in task_names:
        build_task_cache(
            config=TASKS[task_name],
            limit=args.limit,
            out_dir=out_dir,
            overwrite=args.overwrite,
            resume=args.resume,
        )


if __name__ == "__main__":
    main()
