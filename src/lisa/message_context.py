from datetime import UTC, datetime

from langchain_core.messages import BaseMessage, HumanMessage

RECEIVED_AT = "lisa_received_at_utc"
TIME_CONTEXT = "lisa_time_context_applied"


def user_message(content: str, *, received_at: datetime | None = None) -> HumanMessage:
    timestamp = (received_at or datetime.now(UTC)).astimezone(UTC).isoformat().replace("+00:00", "Z")
    return HumanMessage(content=content, additional_kwargs={RECEIVED_AT: timestamp})


def with_system_time(messages: list[BaseMessage]) -> list[BaseMessage]:
    now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    prepared = []
    for message in messages:
        if not isinstance(message, HumanMessage) or message.additional_kwargs.get(TIME_CONTEXT):
            prepared.append(message)
            continue
        stored = message.additional_kwargs.get(RECEIVED_AT)
        try:
            parsed = datetime.fromisoformat(stored) if isinstance(stored, str) else None
            timestamp = parsed.astimezone(UTC).isoformat().replace("+00:00", "Z") if parsed and parsed.tzinfo else None
        except ValueError:
            timestamp = None
        label = timestamp or f"{now} (context construction; message receipt time unavailable)"
        prefix = f"at system time: {label}\n"
        content = prefix + message.content if isinstance(message.content, str) else [
            {"type": "text", "text": prefix}, *message.content]
        prepared.append(message.model_copy(update={"content": content,
            "additional_kwargs": {**message.additional_kwargs, TIME_CONTEXT: True}}))
    return prepared