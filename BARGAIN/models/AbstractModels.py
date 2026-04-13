import numpy as np
from typing import Tuple, List, Any
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

class Proxy():
    def __init__(
        self,
        verbose:bool=True,
        max_workers:int=1
    ):
        self.preds_dict={}
        self.verbose=verbose
        self.max_workers=max_workers

    def proxy_func(self, input: str) -> Tuple[Any, float]:
        '''
        Must extend Proxy class and specifiy this function. This function processes `input` with Proxy. 
        It returns a tuple, with the first element denoting the output of proxy, and the second element the proxy score
    
        Args:
            input: Data record to be processed by the oracle

        Returns:
            Tuple[Any, float]:
                -- Any: The output for `input` computed by proxy model
                -- float: The proxy score computed for the model
        '''
        assert False << "SUBCLASS MUST IMPLEMENT"

    def reset(self) -> None:
        self.preds_dict={}

    def get_preds_and_scores(self, indxs:List, data_records:List) -> Tuple[np.ndarray, np.ndarray]:
        preds = [None] * len(indxs)
        scores = [None] * len(indxs)

        uncached = []
        for i, x in enumerate(indxs):
            if x in self.preds_dict:
                preds[i], scores[i] = self.preds_dict[x]
            else:
                uncached.append(i)

        if uncached:
            show_bar = len(uncached) > 20 and self.verbose
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self.proxy_func, data_records[i]): i
                    for i in uncached
                }
                for future in tqdm(as_completed(futures), disable=not show_bar, total=len(uncached)):
                    i = futures[future]
                    pred, score = future.result()
                    preds[i] = pred
                    scores[i] = score
                    self.preds_dict[indxs[i]] = (pred, score)

        return np.array(preds), np.array(scores)


class Oracle():
    def __init__(
        self,
        verbose:bool=True,
        max_workers:int=1
    ):
        self.cached_validations={}
        self.preds_dict={}
        self.verbose=verbose
        self.max_workers=max_workers

    def get_pred(self, data_records:List, indxs:List=None) -> np.ndarray:
        preds = [None] * len(data_records)

        uncached = []
        for i in range(len(data_records)):
            if indxs is not None and indxs[i] in self.preds_dict:
                preds[i] = self.preds_dict[indxs[i]]
            else:
                uncached.append(i)

        if uncached:
            show_bar = len(uncached) > 20 and self.verbose
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self.oracle_func, data_records[i], ""): i
                    for i in uncached
                }
                for future in tqdm(as_completed(futures), disable=not show_bar, total=len(uncached)):
                    i = futures[future]
                    _, oracle_output = future.result()
                    if indxs is not None:
                        self.preds_dict[indxs[i]] = oracle_output
                    preds[i] = oracle_output

        return np.array(preds)

    def oracle_func(self, input: str, proxy_output: Any) -> Tuple[bool, Any]:
        '''
        Must extend Oracle class and specifiy this function. This function checks if a given `proxy_output` is correct for a given `input`. 
        It returns a tuple, with the first element denoting whether the `proxy_output` is correct, and the second element denotes the correct answer for `input`
    
        Args:
            input: Data record to be processed by the oracle
            proxy_output: Output provided by the Proxy on `input`. Oracle needs to validate if `proxy_output` is correct

        Returns:
            Tuple[bool, Any]:
                -- bool: Whether `proxy_output` is correct for `input` 
                -- Any: The correct output for `input` (can be the same as `proxy_output`)
        '''
        assert False << "MUST IMPLEMENT"


    def get_number_preds(self) -> int:
        return len(self.preds_dict)

    def reset(self) -> None:
        self.cached_validations={}
        self.preds_dict={}

    def is_answer_correct(self, data_indxs:List, data_records:List, proxy_output_at_indxs:List) -> np.ndarray:
        validations = [None] * len(data_indxs)

        uncached = []
        for i, x in enumerate(data_indxs):
            proxy_output = proxy_output_at_indxs[i]
            if (x, proxy_output) in self.cached_validations:
                validations[i] = self.cached_validations[(x, proxy_output)]
            else:
                uncached.append(i)

        if uncached:
            show_bar = len(uncached) > 20 and self.verbose

            def _validate(pos):
                x = data_indxs[pos]
                proxy_output = proxy_output_at_indxs[pos]
                is_correct, oracle_output = self.oracle_func(data_records[pos], proxy_output)
                return pos, x, proxy_output, is_correct, oracle_output

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(_validate, i): i for i in uncached}
                for future in tqdm(as_completed(futures), disable=not show_bar, total=len(uncached)):
                    pos, x, proxy_output, is_correct, oracle_output = future.result()
                    self.preds_dict[x] = oracle_output
                    self.cached_validations[(x, proxy_output)] = is_correct
                    validations[pos] = is_correct

        return np.array(validations)
