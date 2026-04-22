from openai import OpenAI
import os
import json
import hashlib
import threading
import time
from typing import Any, cast
import numpy as np

from BARGAIN.models.AbstractModels import Oracle, Proxy

def _extract_usage_totals(usage: Any) -> tuple[int, int, int]:
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

def get_bool_val_prob(res, logprobs=None):
    if logprobs is None:
        output=False
        if 'true' in res.lower() and 'false' not in res.lower():
            output=True
        return output
    
    true_prob = 0
    false_prob = 0
    for toplogprob in logprobs[0].top_logprobs:
        if toplogprob.token == 'True':
            true_prob = np.exp(toplogprob.logprob)
        if toplogprob.token == 'False':
            false_prob = np.exp(toplogprob.logprob)
    if true_prob == 0 and false_prob == 0:
        return False, 0
    norm = true_prob+false_prob
    true_prob = true_prob/norm
    false_prob = false_prob/norm
    if true_prob>false_prob:
        return True, true_prob 
    return False, false_prob




class OpenAIProxy(Proxy):
    def __init__(
                self,
                task:str,
                is_binary:bool=False,
                model:str='gpt-4o-mini',
                verbose:bool=True,
                base_url:str|None=None,
                api_key:str|None=None,
                max_tokens:int|None=None,
                extra_body:dict|None=None,
                max_workers:int=1
            ) -> None :
        '''
        Args: 
            task: prompt to perform on data records. `task` must be a templatized string: `task.format(data_record)` is passed to `model` to process a `data_record`
            is_binary: Set to `True` if the task is a binary classifiction task. **WARNING** If `True`, `task` should have directions to ensure `model` outputs only True or False
            model: Name of OpenAI model
            verbose: provide progress updates
            base_url: Base URL for an OpenAI-compatible API server (e.g. vLLM). Defaults to the OpenAI API when None.
            api_key: API key for authentication. Falls back to the OPENAI_API_KEY environment variable when None.
            max_tokens: Maximum number of tokens to generate for non-binary tasks. None means no limit.
            extra_body: Extra parameters to pass in the request body (e.g. for vLLM chat_template_kwargs).
            max_workers: Number of parallel threads for API calls. Defaults to 1 (sequential).

        '''
        super().__init__(verbose=verbose, max_workers=max_workers)
        self.task = task
        self.is_binary=is_binary
        self.model = model
        self.max_tokens = max_tokens
        self.extra_body = extra_body

        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url,
        )
        self._timing_lock = threading.Lock()
        self.proxy_seconds = 0.0
        self.proxy_calls = 0
        self.proxy_input_tokens = 0
        self.proxy_output_tokens = 0
        self.proxy_total_tokens = 0

    def reset(self) -> None:
        super().reset()
        with self._timing_lock:
            self.proxy_seconds = 0.0
            self.proxy_calls = 0
            self.proxy_input_tokens = 0
            self.proxy_output_tokens = 0
            self.proxy_total_tokens = 0

    def _record_proxy_time(self, elapsed_seconds: float) -> None:
        with self._timing_lock:
            self.proxy_seconds += elapsed_seconds
            self.proxy_calls += 1

    def _record_proxy_tokens(self, usage: Any) -> None:
        input_tokens, output_tokens, total_tokens = _extract_usage_totals(usage)
        with self._timing_lock:
            self.proxy_input_tokens += input_tokens
            self.proxy_output_tokens += output_tokens
            self.proxy_total_tokens += total_tokens

    def get_timing_stats(self) -> dict:
        with self._timing_lock:
            return {
                "proxy_seconds": float(self.proxy_seconds),
                "proxy_calls": int(self.proxy_calls),
                "proxy_input_tokens": int(self.proxy_input_tokens),
                "proxy_output_tokens": int(self.proxy_output_tokens),
                "proxy_total_tokens": int(self.proxy_total_tokens),
            }

    def proxy_func_general(self, data_record):
        task_with_data = self.task.format(data_record)
        prompt=[
                    {"role": "system", "content": "You are a helpful assistant that is good at processing data."},
                    {"role": "user", "content": task_with_data}
                    ]
        t0 = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model, messages=cast(Any, prompt), logprobs=True, seed=0, temperature=0,
            max_tokens=self.max_tokens, extra_body=self.extra_body)
        self._record_proxy_time(time.perf_counter() - t0)
        self._record_proxy_tokens(response.usage)
        if response.choices[0].logprobs is None:
            print(f"Logprobs are None")
            prob = 0
        else:
            logprobs = response.choices[0].logprobs.content or []
            all_logprobs = 0
            for t in logprobs:
                all_logprobs+= t.logprob
            prob = np.exp(all_logprobs)

        return response.choices[0].message.content, prob

    def proxy_func_binary(self, data_record):
        task_with_data = self.task.format(data_record)
        prompt=[
                    {"role": "system", "content": "You are a helpful assistant that is good at processing data."},
                    {"role": "user", "content": task_with_data}
                    ]
        t0 = time.perf_counter()
        response = self.client.chat.completions.create( model=self.model, messages=cast(Any, prompt), logprobs=True, seed=0, temperature=0, max_tokens=2, top_logprobs=10, extra_body=self.extra_body)
        self._record_proxy_time(time.perf_counter() - t0)
        self._record_proxy_tokens(response.usage)
        res =response.choices[0].message.content
        response_logprobs = response.choices[0].logprobs
        logprobs = response_logprobs.content if (response_logprobs is not None and response_logprobs.content is not None) else []
        return get_bool_val_prob(res, logprobs)

    def proxy_func(self, data_record):
        if self.is_binary:
            return self.proxy_func_binary(data_record)
        else:
            return self.proxy_func_general(data_record)

class OpenAIOracle(Oracle):
    def __init__(
        self,
        task:str,
        is_binary:bool=False,
        model:str='gpt-4o',
        verbose:bool=True,
        base_url:str|None=None,
        api_key:str|None=None,
        judge=None,
        max_tokens:int|None=None,
        extra_body:dict|None=None,
        cache_path:str|None=None,
        cache_task_name:str|None=None,
        max_workers:int=1
    ):
        '''
        Args: 
            task: prompt to perform on data records. `task` must be a templatized string: `task.format(data_record)` is passed to `model` to process a `data_record`
            is_binary: Set to `True` if the task is a binary classifiction task. **WARNING** If `True`, `task` should have directions to ensure `model` outputs only True or False
            model: Name of OpenAI model
            verbose: provide progress updates
            base_url: Base URL for an OpenAI-compatible API server (e.g. vLLM). Defaults to the OpenAI API when None.
            api_key: API key for authentication. Falls back to the OPENAI_API_KEY environment variable when None.
            judge: Optional callable with signature (query, document, oracle_output, proxy_output) -> float.
                `query` is the task template, `document` is the raw data record.
                When provided, used in place of exact-match comparison for open-ended (non-binary) tasks.
                Should return a score between 0.0 and 1.0 (e.g. semantic F1).
                Has no effect on binary tasks. Example: DSPySemanticJudge().
            max_tokens: Maximum number of tokens to generate for non-binary tasks. None means no limit.
            extra_body: Extra parameters to pass in the request body (e.g. for vLLM chat_template_kwargs).
            cache_path: Optional JSONL oracle cache file path built with examples/build_oracle_cache.py.
                Cache key format is sha256(task_name + "\\n" + record_text).
            cache_task_name: Optional task name used in cache key generation. Including task_name in
                the key prevents collisions when identical record text appears in different tasks.
            max_workers: Number of parallel threads for API calls. Defaults to 1 (sequential).

        '''
        super().__init__(verbose=verbose, max_workers=max_workers)
        self.task = task
        self.is_binary=is_binary
        self.model = model
        self.judge = judge
        self.max_tokens = max_tokens
        self.extra_body = extra_body
        self.cache_path = cache_path
        self.cache_task_name = cache_task_name

        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url,
        )
        self.oracle_cache_by_key = self._load_oracle_cache(cache_path)
        self._timing_lock = threading.Lock()
        self.oracle_seconds = 0.0
        self.oracle_calls = 0
        self.judge_seconds = 0.0
        self.judge_calls = 0
        self.oracle_input_tokens = 0
        self.oracle_output_tokens = 0
        self.oracle_total_tokens = 0
        self.oracle_cache_hits = 0
        self.oracle_cache_misses = 0
        self.oracle_cached_seconds = 0.0

    def reset(self) -> None:
        super().reset()
        with self._timing_lock:
            self.oracle_seconds = 0.0
            self.oracle_calls = 0
            self.judge_seconds = 0.0
            self.judge_calls = 0
            self.oracle_input_tokens = 0
            self.oracle_output_tokens = 0
            self.oracle_total_tokens = 0
            self.oracle_cache_hits = 0
            self.oracle_cache_misses = 0
            self.oracle_cached_seconds = 0.0

    def _load_oracle_cache(self, cache_path: str | None) -> dict[str, dict[str, Any]]:
        if cache_path is None:
            return {}
        if not os.path.exists(cache_path):
            if self.verbose:
                print(f"[OpenAIOracle] cache file not found at {cache_path}; using live oracle calls.")
            return {}

        cache_entries: dict[str, dict[str, Any]] = {}
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, start=1):
                    payload = line.strip()
                    if not payload:
                        continue
                    try:
                        entry = json.loads(payload)
                    except json.JSONDecodeError:
                        if self.verbose:
                            print(f"[OpenAIOracle] skipping invalid JSON at {cache_path}:{line_num}")
                        continue
                    record_key = entry.get("record_key")
                    if isinstance(record_key, str) and record_key:
                        cache_entries[record_key] = entry
            if self.verbose:
                print(f"[OpenAIOracle] loaded {len(cache_entries)} cache entries from {cache_path}")
        except OSError as exc:
            if self.verbose:
                print(f"[OpenAIOracle] failed to read cache file {cache_path}: {exc}")
            return {}
        return cache_entries

    def _get_cache_key_for_record(self, data_record: str) -> str | None:
        if self.cache_path is None:
            return None
        task_name = self.cache_task_name if self.cache_task_name is not None else self.task
        payload = f"{task_name}\n{str(data_record)}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _record_oracle_tokens_from_totals(
        self, input_tokens: int, output_tokens: int, total_tokens: int
    ) -> None:
        with self._timing_lock:
            self.oracle_input_tokens += int(input_tokens)
            self.oracle_output_tokens += int(output_tokens)
            self.oracle_total_tokens += int(total_tokens)

    def _extract_cached_metrics(self, cache_entry: dict[str, Any]) -> tuple[float, int, int, int]:
        cached_seconds = cache_entry.get("oracle_seconds", 0.0)
        cached_input_tokens = cache_entry.get("oracle_input_tokens", 0)
        cached_output_tokens = cache_entry.get("oracle_output_tokens", 0)
        cached_total_tokens = cache_entry.get("oracle_total_tokens", 0)

        try:
            seconds_value = float(cached_seconds)
        except (TypeError, ValueError):
            seconds_value = 0.0
        try:
            input_tokens = int(cached_input_tokens)
        except (TypeError, ValueError):
            input_tokens = 0
        try:
            output_tokens = int(cached_output_tokens)
        except (TypeError, ValueError):
            output_tokens = 0
        try:
            total_tokens = int(cached_total_tokens)
        except (TypeError, ValueError):
            total_tokens = input_tokens + output_tokens

        if total_tokens == 0:
            total_tokens = input_tokens + output_tokens
        return seconds_value, input_tokens, output_tokens, total_tokens

    def _record_cache_hit(self, cache_entry: dict[str, Any]) -> None:
        cached_seconds, input_tokens, output_tokens, total_tokens = self._extract_cached_metrics(cache_entry)
        self._record_oracle_time(cached_seconds)
        self._record_oracle_tokens_from_totals(input_tokens, output_tokens, total_tokens)
        with self._timing_lock:
            self.oracle_cache_hits += 1
            self.oracle_cached_seconds += cached_seconds

    def _record_cache_miss(self) -> None:
        with self._timing_lock:
            self.oracle_cache_misses += 1

    def _lookup_cache_entry(self, data_record: str) -> dict[str, Any] | None:
        cache_key = self._get_cache_key_for_record(data_record)
        if cache_key is None:
            return None
        cache_entry = self.oracle_cache_by_key.get(cache_key)
        if cache_entry is None:
            self._record_cache_miss()
            return None
        self._record_cache_hit(cache_entry)
        return cache_entry

    def _to_binary_output(self, oracle_output: Any) -> bool:
        if isinstance(oracle_output, bool):
            return oracle_output
        if oracle_output is None:
            return False
        return bool(get_bool_val_prob(str(oracle_output)))

    def get_cached_record_seconds(self, data_record: str) -> float | None:
        cache_key = self._get_cache_key_for_record(data_record)
        if cache_key is None:
            return None
        cache_entry = self.oracle_cache_by_key.get(cache_key)
        if cache_entry is None:
            return None
        cached_seconds, _, _, _ = self._extract_cached_metrics(cache_entry)
        return cached_seconds

    def get_cached_record_outputs(self, data_records: list[str]) -> list[str | None]:
        outputs: list[str | None] = []
        for data_record in data_records:
            cache_key = self._get_cache_key_for_record(data_record)
            if cache_key is None:
                outputs.append(None)
                continue
            cache_entry = self.oracle_cache_by_key.get(cache_key)
            if cache_entry is None:
                outputs.append(None)
                continue
            oracle_output = cache_entry.get("oracle_output")
            if oracle_output is None:
                outputs.append("")
            else:
                outputs.append(str(oracle_output))
        return outputs

    def _record_oracle_time(self, elapsed_seconds: float) -> None:
        with self._timing_lock:
            self.oracle_seconds += elapsed_seconds
            self.oracle_calls += 1

    def _record_oracle_tokens(self, usage: Any) -> None:
        input_tokens, output_tokens, total_tokens = _extract_usage_totals(usage)
        with self._timing_lock:
            self.oracle_input_tokens += input_tokens
            self.oracle_output_tokens += output_tokens
            self.oracle_total_tokens += total_tokens

    def _record_judge_time(self, elapsed_seconds: float) -> None:
        with self._timing_lock:
            self.judge_seconds += elapsed_seconds
            self.judge_calls += 1

    def get_timing_stats(self) -> dict:
        with self._timing_lock:
            return {
                "oracle_seconds": float(self.oracle_seconds),
                "oracle_calls": int(self.oracle_calls),
                "judge_seconds": float(self.judge_seconds),
                "judge_calls": int(self.judge_calls),
                "oracle_input_tokens": int(self.oracle_input_tokens),
                "oracle_output_tokens": int(self.oracle_output_tokens),
                "oracle_total_tokens": int(self.oracle_total_tokens),
                "oracle_cache_hits": int(self.oracle_cache_hits),
                "oracle_cache_misses": int(self.oracle_cache_misses),
                "oracle_cached_seconds": float(self.oracle_cached_seconds),
            }

    def oracle_func_binary(self, data_record, proxy_output):
        cache_entry = self._lookup_cache_entry(data_record)
        if cache_entry is not None:
            oracle_output = self._to_binary_output(cache_entry.get("oracle_output"))
            return oracle_output == proxy_output, oracle_output

        task_with_data = self.task.format(data_record)
        prompt=[
                    {"role": "system", "content": "You are a helpful assistant that is good at processing data."},
                    {"role": "user", "content": task_with_data}
                ]
        t0 = time.perf_counter()
        response = self.client.chat.completions.create( model=self.model, messages=cast(Any, prompt), logprobs=False, seed=0, temperature=0, max_tokens=2, extra_body=self.extra_body)
        self._record_oracle_time(time.perf_counter() - t0)
        self._record_oracle_tokens(response.usage)
        res=response.choices[0].message.content
        oracle_output = get_bool_val_prob(res)
        return oracle_output == proxy_output, oracle_output

    def oracle_func_general(self, data_record, proxy_output):
        cache_entry = self._lookup_cache_entry(data_record)
        if cache_entry is not None:
            oracle_output = cache_entry.get("oracle_output")
            if oracle_output is None:
                oracle_output = ""
            else:
                oracle_output = str(oracle_output)
            if self.judge is not None:
                judge_t0 = time.perf_counter()
                is_correct = self.judge(self.task, data_record, oracle_output, proxy_output)
                self._record_judge_time(time.perf_counter() - judge_t0)
            else:
                is_correct = (proxy_output == oracle_output)
            return is_correct, oracle_output

        task_with_data = self.task.format(data_record)
        prompt=[
                    {"role": "system", "content": "You are a helpful assistant that is good at processing data."},
                    {"role": "user", "content": task_with_data}
                ]
        t0 = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model, messages=cast(Any, prompt), logprobs=False, seed=0, temperature=0,
            max_tokens=self.max_tokens, extra_body=self.extra_body)
        self._record_oracle_time(time.perf_counter() - t0)
        self._record_oracle_tokens(response.usage)
        oracle_output=response.choices[0].message.content
        if self.judge is not None:
            judge_t0 = time.perf_counter()
            is_correct = self.judge(self.task, data_record, oracle_output, proxy_output)
            self._record_judge_time(time.perf_counter() - judge_t0)
        else:
            is_correct = (proxy_output == oracle_output)
        return is_correct, oracle_output
    
    def oracle_func(self, data_record, proxy_output):
        if self.is_binary:
            return self.oracle_func_binary(data_record, proxy_output)
        else:
            return self.oracle_func_general(data_record, proxy_output)
