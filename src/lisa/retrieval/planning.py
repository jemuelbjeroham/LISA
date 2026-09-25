import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class RetrievalDecision(BaseModel):
    retrieval_query: str
    use_hyde: bool
    reasoning: str


class RetrievalPlanner:

    def __init__(
        self,
        model: BaseChatModel,
        prompt: str,
    ):
        self.model = model.with_structured_output(
            RetrievalDecision
        )
        self.prompt = prompt

    async def plan(
        self,
        messages: list,
    ) -> RetrievalDecision:

        planner_messages = [
            SystemMessage(content=self.prompt),
            *messages,
        ]

        logger.info("Generating retrieval plan")

        decision = await self.model.ainvoke(
            planner_messages
        )

        logger.info(
            "Retrieval plan: query=%r use_hyde=%s reason=%r",
            decision.retrieval_query,
            decision.use_hyde,
            decision.reasoning,
        )

        return decision