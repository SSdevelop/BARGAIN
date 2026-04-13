from openai import OpenAI
import os
import json
import numpy as np

from BARGAIN.models.AbstractModels import Oracle, Proxy

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

    def proxy_func_general(self, data_record):
        task_with_data = self.task.format(data_record)
        prompt=[
                    {"role": "system", "content": "You are a helpful assistant that is good at processing data."},
                    {"role": "user", "content": task_with_data}
                    ]
        response = self.client.chat.completions.create(
            model=self.model, messages=prompt, logprobs=True, seed=0, temperature=0,
            max_tokens=self.max_tokens, extra_body=self.extra_body)
        if response.choices[0].logprobs is None:
            print(f"Logprobs are None")
            prob = 0
        else:
            logprobs = response.choices[0].logprobs.content
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
        response = self.client.chat.completions.create( model=self.model, messages=prompt, logprobs=True, seed=0, temperature=0, max_tokens=2, top_logprobs=10, extra_body=self.extra_body)
        res =response.choices[0].message.content
        logprobs = response.choices[0].logprobs.content
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
            max_workers: Number of parallel threads for API calls. Defaults to 1 (sequential).

        '''
        super().__init__(verbose=verbose, max_workers=max_workers)
        self.task = task
        self.is_binary=is_binary
        self.model = model
        self.judge = judge
        self.max_tokens = max_tokens
        self.extra_body = extra_body

        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url,
        )

    def oracle_func_binary(self, data_record, proxy_output):
        task_with_data = self.task.format(data_record)
        prompt=[
                    {"role": "system", "content": "You are a helpful assistant that is good at processing data."},
                    {"role": "user", "content": task_with_data}
                ]
        response = self.client.chat.completions.create( model=self.model, messages=prompt, logprobs=False, seed=0, temperature=0, max_tokens=2, extra_body=self.extra_body)
        res=response.choices[0].message.content
        oracle_output = get_bool_val_prob(res)
        return oracle_output == proxy_output, oracle_output

    def oracle_func_general(self, data_record, proxy_output):
        task_with_data = self.task.format(data_record)
        prompt=[
                    {"role": "system", "content": "You are a helpful assistant that is good at processing data."},
                    {"role": "user", "content": task_with_data}
                ]
        response = self.client.chat.completions.create(
            model=self.model, messages=prompt, logprobs=False, seed=0, temperature=0,
            max_tokens=self.max_tokens, extra_body=self.extra_body)
        oracle_output=response.choices[0].message.content
        if self.judge is not None:
            is_correct = self.judge(self.task, data_record, oracle_output, proxy_output)
        else:
            is_correct = (proxy_output == oracle_output)
        return is_correct, oracle_output
    
    def oracle_func(self, data_record, proxy_output):
        if self.is_binary:
            return self.oracle_func_binary(data_record, proxy_output)
        else:
            return self.oracle_func_general(data_record, proxy_output)
