"""Intent routing with TypeSafe's Jev.

Jev answers narrow yes/no judgments about the user's latest message; the
decision table in `decide_route` turns those judgments into a route.

The question wording and the background facts sent as state are a versioned
prompt in `prompts/orchestrator/jev_routing_v*.json`, loaded by the
composition root and parsed with `parse_jev_routing_prompt`. The decision
table and thresholds are decision logic and stay here, in code.

Decision table (Y = probability above YES_THRESHOLD, N = below
NO_THRESHOLD, ? = in between, - = not read). Rows are checked in order,
mirroring the precedence rules of the LLM routing prompt:

    ticket   problem  internal_docs | outcome
    Y        -        -             | general_enquiry
    ?        -        -             | unsettled -> LLM fallback
    N        Y        -             | troubleshooting_agent
    N        ?        -             | unsettled -> LLM fallback
    N        N        Y             | knowledge_assistant
    N        N        ?             | unsettled -> LLM fallback
    N        N        N             | general_enquiry
"""

import json
import logging
from collections.abc import Mapping, Sequence
from typing import Any, Self

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, ConfigDict, model_validator
from typesafe_sdk import Noul, TypeSafeClient

from lisa.routing import Route, RoutingDecision

logger = logging.getLogger(__name__)

YES_THRESHOLD = 0.7
NO_THRESHOLD = 0.3

MAX_EARLIER_MESSAGES = 6
MAX_MESSAGE_CHARS = 4000

WANTS_TICKET = "wants_ticket"
WANTS_PROBLEM_SOLVED = "wants_problem_solved"
NEEDS_INTERNAL_DOCS = "needs_internal_docs"

# Read in this order by the decision table; a clear yes wins.
ROUTE_RULES: tuple[tuple[str, Route], ...] = (
    (WANTS_TICKET, Route.GENERAL_ENQUIRY),
    (WANTS_PROBLEM_SOLVED, Route.TROUBLESHOOTING_AGENT),
    (NEEDS_INTERNAL_DOCS, Route.KNOWLEDGE_ASSISTANT),
)

# State keys the router fills from the conversation; a prompt may not set them.
CONVERSATION_KEYS = frozenset({"earlier_conversation", "latest_user_message"})


class NoulCriteriaPrompt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    true: str
    false: str


class NoulPrompt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instructions: str
    criteria: NoulCriteriaPrompt


class JevRoutingPrompt(BaseModel):
    """A versioned Jev routing prompt: background state and the questions."""

    model_config = ConfigDict(extra="forbid")

    state: dict[str, str]
    questions: dict[str, NoulPrompt]

    @model_validator(mode="after")
    def _matches_decision_table(self) -> Self:
        expected = {name for name, _ in ROUTE_RULES}
        if set(self.questions) != expected:
            raise ValueError(
                f"Jev routing prompt must define exactly the questions "
                f"{sorted(expected)}; got {sorted(self.questions)}"
            )
        reserved = CONVERSATION_KEYS & set(self.state)
        if reserved:
            raise ValueError(
                f"Jev routing prompt state may not set {sorted(reserved)}; "
                f"the router fills them from the conversation"
            )
        return self

    def to_questions(self) -> dict[str, Noul]:
        return {
            name: Noul(
                instructions=question.instructions,
                criteria={
                    "true": question.criteria.true,
                    "false": question.criteria.false,
                },
            )
            for name, question in self.questions.items()
        }


def parse_jev_routing_prompt(text: str) -> JevRoutingPrompt:
    return JevRoutingPrompt.model_validate(json.loads(text))


def build_routing_state(
    messages: Sequence[BaseMessage],
    background: Mapping[str, str],
) -> dict[str, Any] | None:
    """Build Jev's state from the conversation, or None when there is no user message."""
    turns = [
        turn
        for message in messages
        if (turn := _conversation_turn(message)) is not None
    ]
    latest_index = next(
        (index for index in range(len(turns) - 1, -1, -1) if turns[index]["from"] == "user"),
        None,
    )
    if latest_index is None:
        return None

    state: dict[str, Any] = dict(background)
    earlier = turns[:latest_index][-MAX_EARLIER_MESSAGES:]
    if earlier:
        state["earlier_conversation"] = earlier
    state["latest_user_message"] = turns[latest_index]["text"]
    return state


def decide_route(probabilities: Mapping[str, float]) -> Route | None:
    """Apply the decision table; None means the answers do not settle the route."""
    for question, route in ROUTE_RULES:
        probability = probabilities[question]
        if probability > YES_THRESHOLD:
            return route
        if probability >= NO_THRESHOLD:
            return None
    return Route.GENERAL_ENQUIRY


class JevRouter:
    def __init__(self, client: TypeSafeClient, prompt: JevRoutingPrompt) -> None:
        self.client = client
        self.background = dict(prompt.state)
        self.questions = prompt.to_questions()

    def classify(self, messages: Sequence[BaseMessage]) -> RoutingDecision | None:
        """Return Jev's route, or None when Jev is unsure. Raises on API errors."""
        state = build_routing_state(messages, self.background)
        if state is None:
            logger.info("Jev routing skipped: no user message in state")
            return None

        logger.debug("Jev routing request: state=%s", state)
        response = self.client.system_one(state, self.questions)
        probabilities = {
            name: response.nouls[name].noul for name in self.questions
        }
        route = decide_route(probabilities)

        logger.info(
            "Jev routing answers: model=%s, probabilities=%s, route=%s",
            response.model,
            {name: round(value, 3) for name, value in probabilities.items()},
            route.value if route else "unsettled",
        )
        return RoutingDecision(route=route) if route else None


def _conversation_turn(message: BaseMessage) -> dict[str, str] | None:
    if isinstance(message, HumanMessage):
        speaker = "user"
    elif isinstance(message, AIMessage):
        speaker = "LISA"
    else:
        return None

    text = message.text.strip()
    if not text:
        return None
    return {"from": speaker, "text": text[:MAX_MESSAGE_CHARS]}
