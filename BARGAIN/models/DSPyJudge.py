"""Semantic F1 judge using DSPy's ChainOfThought for LLM-based precision/recall.

The caller must configure a DSPy language model globally before using
DSPySemanticJudge, e.g.:

    dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
"""

import dspy


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
        return dspy.Prediction(score=score, precision=scores.precision,
                               recall=scores.recall, feedback=scores.feedback)


class DSPySemanticJudge:
    """Wraps SemanticF1 into a callable that returns the F1 score for BARGAIN's oracle interface."""

    def __init__(self):
        self.metric = SemanticF1()

    def __call__(self, query: str, document: str, oracle_output: str, proxy_output: str) -> float:
        example = dspy.Example(query=query, document=document, answer=oracle_output)
        pred = dspy.Example(answer=proxy_output)
        result = self.metric(example=example, pred=pred)
        return result.score
