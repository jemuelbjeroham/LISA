
import logging
from dataclasses import dataclass
from typing import Protocol

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

INPUT_MODERATION_MODEL = "omni-moderation-latest"

INPUT_BLOCKED_MESSAGE = (
    "I'm sorry, but I can't help with that request. "
    "Please rephrase your message and try again."
)

INPUT_MODERATION_ERROR_MESSAGE = (
    "I'm sorry, but I couldn't validate your message right now. "
    "Please try again shortly."
)


@dataclass(frozen=True)
class InputModerationResult:
    flagged: bool
    categories: tuple[str, ...] = ()


class InputModerator(Protocol):
    async def moderate(self, text: str) -> InputModerationResult:
        """Check user-provided text before agent processing."""
        ...


class InputModerationBlocked(Exception):
    """Raised when user input is flagged by the moderation service."""

    def __init__(self, result: InputModerationResult):
        self.result = result
        super().__init__("User input did not pass moderation.")


class InputModerationUnavailable(Exception):
    """Raised when the moderation service cannot validate user input."""


class OpenAIInputModerator:
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str = INPUT_MODERATION_MODEL,
    ):
        self.client = client
        self.model = model

    async def moderate(self, text: str) -> InputModerationResult:
        if not text or not text.strip():
            return InputModerationResult(flagged=False)

        try:
            response = await self.client.moderations.create(
                model=self.model,
                input=text,
            )
        except Exception as exc:
            logger.exception("Input moderation API call failed")
            raise InputModerationUnavailable(
                "Unable to validate user input."
            ) from exc

        if not response.results:
            logger.error("Input moderation returned no results")
            raise InputModerationUnavailable(
                "The moderation service returned no result."
            )

        result = response.results[0]

        flagged_categories = tuple(
            category
            for category, is_flagged in result.categories.model_dump().items()
            if is_flagged
        )

        moderation_result = InputModerationResult(
            flagged=result.flagged,
            categories=flagged_categories,
        )

        if moderation_result.flagged:
            logger.warning(
                "User input blocked by moderation: categories=%s",
                moderation_result.categories,
            )

        return moderation_result
