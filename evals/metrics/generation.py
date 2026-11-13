from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric


def get_generation_metrics():

    faithfulness = FaithfulnessMetric(
        threshold=0.8,
        include_reason=True,
    )

    answer_relevancy = AnswerRelevancyMetric(
        threshold=0.8,
        include_reason=True,
    )

    return [
        faithfulness,
        answer_relevancy
    ]