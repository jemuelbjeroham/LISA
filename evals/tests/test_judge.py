from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

test_case = LLMTestCase(
    input="What is Retrieval-Augmented Generation (RAG)?",

    actual_output=(
        "Retrieval-Augmented Generation is a technique "
        "where an LLM retrieves relevant information from "
        "an external knowledge source and uses that context "
        "to generate a response."
    )
)

metric = AnswerRelevancyMetric(
    threshold=0.7,
    include_reason=True
)

evaluate(
    test_cases=[test_case],
    metrics=[metric]
)