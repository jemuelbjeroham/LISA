from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol

from openai import AsyncOpenAI


@dataclass(frozen=True)
class ModerationResult:
    allowed: bool
    reason: str | None = None


class OutputModerator(Protocol):
    async def moderate(self, text: str) -> ModerationResult:
        """Determine whether generated text is allowed."""
        ...


class OpenAIOutputModerator:
    """Moderate generated text using OpenAI's Moderation API."""

    def __init__(self, client: AsyncOpenAI) -> None:
        self.client = client

    async def moderate(self, text: str) -> ModerationResult:
        if not text.strip():
            return ModerationResult(allowed=True)

        # API errors intentionally propagate. Callers must not
        # interpret a failed moderation request as approval.
        response = await self.client.moderations.create(
            model="omni-moderation-latest",
            input=text,
        )

        result = response.results[0]

        if result.flagged:
            categories = [
                name
                for name, flagged in result.categories.model_dump().items()
                if flagged
            ]

            return ModerationResult(
                allowed=False,
                reason=(
                    "Output moderation flagged categories: "
                    + ", ".join(categories)
                ),
            )

        return ModerationResult(allowed=True)


class BufferedOutputModerator:
    """Buffer streamed text and moderate it before releasing it."""

    def __init__(
        self,
        moderator: OutputModerator,
        max_buffer_chars: int = 200,
    ) -> None:
        if max_buffer_chars < 1:
            raise ValueError("max_buffer_chars must be at least 1")

        self.moderator = moderator
        self.max_buffer_chars = max_buffer_chars

    async def filter_stream(
        self,
        chunks: AsyncIterator[str],
    ) -> AsyncIterator[str]:
        buffer = ""

        async for chunk in chunks:
            if not chunk:
                continue

            buffer += chunk

            while len(buffer) >= self.max_buffer_chars:
                boundary = self._find_boundary(
                    buffer,
                    self.max_buffer_chars,
                )

                if boundary is None:
                    boundary = self.max_buffer_chars

                candidate = buffer[:boundary]
                buffer = buffer[boundary:]

                result = await self.moderator.moderate(candidate)

                if not result.allowed:
                    raise OutputModerationBlocked(
                        result.reason or "Output moderation blocked the response."
                    )

                yield candidate

        if buffer:
            result = await self.moderator.moderate(buffer)

            if not result.allowed:
                raise OutputModerationBlocked(
                    result.reason or "Output moderation blocked the response."
                )

            yield buffer

    @staticmethod
    def _find_boundary(text: str, limit: int) -> int | None:
        minimum = max(1, limit // 2)

        for index in range(min(limit, len(text)) - 1, minimum - 1, -1):
            if text[index] in ".!?\n":
                return index + 1

        return None


class OutputModerationBlocked(Exception):
    """Raised when generated output fails moderation."""