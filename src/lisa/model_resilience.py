
import asyncio
import logging
from collections.abc import AsyncIterator

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)

MAX_PRIMARY_ATTEMPTS = 2
RETRY_DELAY_SECONDS = 0.0


class ModelResilience:
    def __init__(
        self,
        primary: BaseChatModel,
        fallback: BaseChatModel,
        inactivity_timeout_seconds: float = 10.0,
    ) -> None:
        if inactivity_timeout_seconds <= 0:
            raise ValueError("inactivity_timeout_seconds must be positive")

        self.primary = primary
        self.fallback = fallback
        self.inactivity_timeout_seconds = inactivity_timeout_seconds

    async def stream(
        self,
        messages: list[BaseMessage],
    ) -> AsyncIterator:
        """
        Stream from the primary model with bounded retries.

        Retry and fallback are permitted only before the primary emits
        its first chunk. Once streaming begins, errors propagate rather
        than restarting the response and potentially duplicating output.
        """
        for attempt in range(1, MAX_PRIMARY_ATTEMPTS + 1):
            emitted_chunk = False

            try:
                logger.info(
                    "Starting primary stream: attempt=%d/%d model=%s",
                    attempt,
                    MAX_PRIMARY_ATTEMPTS,
                    type(self.primary).__name__,
                )

                async for chunk in self._stream_with_timeout(
                    self.primary,
                    messages,
                ):
                    emitted_chunk = True
                    yield chunk

                logger.info("Primary model stream completed successfully.")
                return

            except asyncio.CancelledError:
                raise

            except Exception:
                logger.exception(
                    "Primary stream failed: attempt=%d/%d emitted_chunk=%s",
                    attempt,
                    MAX_PRIMARY_ATTEMPTS,
                    emitted_chunk,
                )

                if emitted_chunk:
                    # Output has already reached the caller. Restarting
                    # could duplicate text or tool calls.
                    raise

                if attempt < MAX_PRIMARY_ATTEMPTS:
                    if RETRY_DELAY_SECONDS > 0:
                        await asyncio.sleep(RETRY_DELAY_SECONDS)
                    continue

                logger.warning(
                    "Primary retries exhausted before output; "
                    "switching to fallback model=%s",
                    type(self.fallback).__name__,
                )
                break

        # The fallback is streamed immediately as its chunks arrive.
        # Its errors propagate because there is no further fallback.
        try:
            async for chunk in self._stream_with_timeout(
                self.fallback,
                messages,
            ):
                yield chunk

            logger.info("Fallback model stream completed successfully.")

        except asyncio.CancelledError:
            raise

        except Exception:
            logger.exception("Fallback model stream failed.")
            raise

    async def _stream_with_timeout(
        self,
        model: BaseChatModel,
        messages: list[BaseMessage],
    ) -> AsyncIterator:
        """Yield chunks immediately with a per-chunk inactivity timeout."""
        stream = model.astream(messages).__aiter__()

        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(
                        stream.__anext__(),
                        timeout=self.inactivity_timeout_seconds,
                    )
                except StopAsyncIteration:
                    break

                yield chunk

        finally:
            close = getattr(stream, "aclose", None)
            if close is not None:
                try:
                    await close()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.debug(
                        "Failed to close model stream cleanly.",
                        exc_info=True,
                    )
