from deepeval import assert_test
from deepeval.test_case import LLMTestCase

from evals.metrics.generation import get_generation_metrics


def test_generation_quality():

    test_case = LLMTestCase(
        input="How do I troubleshoot encoder alarms?",

        actual_output=(
            "Check the encoder alarm status and "
            "follow the approved troubleshooting procedure."
        ),

        retrieval_context=[
            "The encoder troubleshooting procedure requires ",
            "checking the alarm status before escalating."
        ]
    )

    metrics = get_generation_metrics()

    assert_test(
        test_case=test_case,
        metrics=metrics
    )