from typing import Any

from langchain_core.messages import messages_from_dict, messages_to_dict

from lisa.routing import Route
from lisa.state import LISAState


class ConversationStateSerializer:

    @staticmethod
    def serialize(state: LISAState) -> dict[str, Any]:
        return {
            "messages": messages_to_dict(state["messages"]),
            "route": (
                state["route"].value
                if state["route"] is not None
                else None
            ),
            "active_route": (
                state["active_route"].value
                if state["active_route"] is not None
                else None
            ),
            "agent_handoff": state["agent_handoff"],
            "knowledge_context": state["knowledge_context"],
            "enable_thinking": state["enable_thinking"],
        }

    @staticmethod
    def deserialize(data: dict[str, Any]) -> LISAState:
        messages = messages_from_dict(data["messages"])

        return {
            "messages": messages,
            "route": (
                Route(data["route"])
                if data["route"] is not None
                else None
            ),
            "active_route": (
                Route(data["active_route"])
                if data["active_route"] is not None
                else None
            ),
            "agent_handoff": data["agent_handoff"],
            "knowledge_context": data["knowledge_context"],
            "enable_thinking": data["enable_thinking"],
        }