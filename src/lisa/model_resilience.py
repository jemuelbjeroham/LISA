import asyncio
import logging
from collections.abc import AsyncIterator

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)


class ModelResilience:

    def __init__(
        self,
        primary: BaseChatModel,
        fallback: BaseChatModel,
        inactivity_timeout_seconds: float = 10.0,
    ):
        self.primary = primary
        self.fallback = fallback
        self.inactivity_timeout_seconds = inactivity_timeout_seconds

    async def stream(
        self,
        messages: list[BaseMessage],
    ) -> AsyncIterator:

        try:
            logger.info(
                "Starting primary model stream: model=%s",
                type(self.primary).__name__,
            )

            async for chunk in self._stream_with_timeout(
                self.primary,
                messages,
            ):
                yield chunk

            logger.info(
                "Primary model stream completed successfully: model=%s",
                type(self.primary).__name__,
            )

        except asyncio.TimeoutError:
            logger.warning(
                "Primary model timed out after %.1f seconds. "
                "Switching to fallback model: %s",
                self.inactivity_timeout_seconds,
                type(self.fallback).__name__,
            )

            logger.info(
                "Starting fallback model stream: model=%s",
                type(self.fallback).__name__,
            )

            async for chunk in self.fallback.astream(messages):
                logger.debug(
                    "Received fallback model chunk: content=%r tool_calls=%s",
                    chunk.content,
                    chunk.tool_calls,
                )
                yield chunk

            logger.info(
                "Fallback model stream completed successfully: model=%s",
                type(self.fallback).__name__,
            )

        except Exception:
            logger.exception(
                "Primary model failed unexpectedly. "
                "Switching to fallback model: %s",
                type(self.fallback).__name__,
            )

            logger.info(
                "Starting fallback model stream: model=%s",
                type(self.fallback).__name__,
            )

            async for chunk in self.fallback.astream(messages):
                logger.debug(
                    "Received fallback model chunk: content=%r tool_calls=%s",
                    chunk.content,
                    chunk.tool_calls,
                )
                yield chunk

            logger.info(
                "Fallback model stream completed successfully: model=%s",
                type(self.fallback).__name__,
            )

    async def _stream_with_timeout(
        self,
        model: BaseChatModel,
        messages: list[BaseMessage],
    ) -> AsyncIterator:

        stream = model.astream(messages).__aiter__()

        while True:
            try:
                chunk = await asyncio.wait_for(
                    stream.__anext__(),
                    timeout=self.inactivity_timeout_seconds,
                )

            except StopAsyncIteration:
                break

            yield chunk