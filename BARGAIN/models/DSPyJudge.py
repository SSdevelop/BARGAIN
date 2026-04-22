"""Semantic F1 judge using DSPy's ChainOfThought for LLM-based precision/recall.

The caller must configure a DSPy language model globally before using
DSPySemanticJudge, e.g.:

    dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
"""

import dspy
import threading
import time

def _extract_usage_totals(obj) -> tuple[int, int, int]:
    def _as_int(value) -> int:
        if value is None:
            return 0
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def _extract_direct(candidate) -> tuple[int, int, int]:
        if candidate is None:
            return 0, 0, 0

        if isinstance(candidate, dict):
            in_tok = _as_int(candidate.get("prompt_tokens", candidate.get("input_tokens")))
            out_tok = _as_int(candidate.get("completion_tokens", candidate.get("output_tokens")))
            total_tok = _as_int(candidate.get("total_tokens"))
            if total_tok == 0:
                total_tok = in_tok + out_tok
            return in_tok, out_tok, total_tok

        in_tok = _as_int(getattr(candidate, "prompt_tokens", getattr(candidate, "input_tokens", None)))
        out_tok = _as_int(getattr(candidate, "completion_tokens", getattr(candidate, "output_tokens", None)))
        total_tok = _as_int(getattr(candidate, "total_tokens", None))
        if total_tok == 0:
            total_tok = in_tok + out_tok
        return in_tok, out_tok, total_tok

    # Try common direct and nested usage containers.
    in_tok, out_tok, total_tok = _extract_direct(obj)
    if total_tok > 0:
        return in_tok, out_tok, total_tok

    usage = obj.get("usage") if isinstance(obj, dict) else getattr(obj, "usage", None)
    in_tok, out_tok, total_tok = _extract_direct(usage)
    if total_tok > 0:
        return in_tok, out_tok, total_tok

    response = obj.get("response") if isinstance(obj, dict) else getattr(obj, "response", None)
    in_tok, out_tok, total_tok = _extract_direct(response)
    if total_tok > 0:
        return in_tok, out_tok, total_tok

    return 0, 0, 0


class SemanticRecallPrecision(dspy.Signature):
    """
    Compare a system's response to the ground truth to compute its recall and precision.
    If asked to reason, enumerate key ideas in each response, and whether they are present in the other response.
    """

    query: str = dspy.InputField(desc="The task or question that was asked")
    document: str = dspy.InputField(desc="The source document the task was applied to")
    ground_truth: str = dspy.InputField()
    system_response: str = dspy.InputField()
    precision: float = dspy.OutputField(desc="Fraction of the system response that is supported by the ground truth, between 0.0 and 1.0")
    recall: float = dspy.OutputField(desc="Fraction of the ground truth that is covered by the system response, between 0.0 and 1.0")
    feedback: str = dspy.OutputField(desc="Brief explanation of the scores")


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


class SemanticF1(dspy.Module):
    """Computes semantic F1 between a prediction and ground truth via LLM-based precision/recall."""

    def __init__(self):
        self.module = dspy.ChainOfThought(SemanticRecallPrecision)

    def forward(self, example, pred, trace=None, pred_name=None, pred_trace=None):
        scores = self.module(
            query=example.query,
            document=example.document,
            ground_truth=example.answer,
            system_response=pred.answer,
        )
        score = f1_score(scores.precision, scores.recall)
        usage_input_tokens, usage_output_tokens, usage_total_tokens = _extract_usage_totals(scores)
        return dspy.Prediction(score=score, precision=scores.precision,
                               recall=scores.recall, feedback=scores.feedback,
                               usage_input_tokens=usage_input_tokens,
                               usage_output_tokens=usage_output_tokens,
                               usage_total_tokens=usage_total_tokens)


class DSPySemanticJudge:
    """Wraps SemanticF1 into a callable that returns the F1 score for BARGAIN's oracle interface."""

    def __init__(self):
        self.metric = SemanticF1()
        self._timing_lock = threading.Lock()
        self.total_seconds = 0.0
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0

    def reset_timing(self) -> None:
        with self._timing_lock:
            self.total_seconds = 0.0
            self.calls = 0
            self.input_tokens = 0
            self.output_tokens = 0
            self.total_tokens = 0

    def get_timing_stats(self) -> dict:
        with self._timing_lock:
            return {
                "total_seconds": float(self.total_seconds),
                "calls": int(self.calls),
                "input_tokens": int(self.input_tokens),
                "output_tokens": int(self.output_tokens),
                "total_tokens": int(self.total_tokens),
            }

    def __call__(self, query: str, document: str, oracle_output: str, proxy_output: str) -> float:
        t0 = time.perf_counter()
        example = dspy.Example(query=query, document=document, answer=oracle_output)
        pred = dspy.Example(answer=proxy_output)
        result = self.metric(example=example, pred=pred)
        elapsed = time.perf_counter() - t0
        input_tokens, output_tokens, total_tokens = _extract_usage_totals(result)
        if total_tokens == 0:
            input_tokens = int(getattr(result, "usage_input_tokens", 0))
            output_tokens = int(getattr(result, "usage_output_tokens", 0))
            total_tokens = int(getattr(result, "usage_total_tokens", 0))
        with self._timing_lock:
            self.total_seconds += elapsed
            self.calls += 1
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.total_tokens += total_tokens
        return result.score
