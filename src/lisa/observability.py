import logging
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from langfuse import Langfuse

logger = logging.getLogger(__name__)

def create_langfuse_client(settings) -> Langfuse | None:
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        logger.warning("Langfuse credentials are missing; tracing is disabled.")
        return None

    try:
        return Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            base_url=settings.langfuse_base_url,
        )

    except Exception:
        logger.exception("Failed to initialize Langfuse.")
        return None

class LISAObservability:
    def __init__(self, client: Langfuse | None) -> None:
        self.client = client

    @contextmanager
    def span(
        self,
        name: str,
        *,
        input: dict[str, Any] | None = None,
    ) -> Generator[Any, None, None]:  # noqa: UP043
        if self.client is None:
            yield None
            return

        with self.client.start_as_current_observation(
            as_type="span",
            name=name,
            input=input,
        ) as observation:
            yield observation

    def flush(self) -> None:
        if self.client is not None:
            try:
                self.client.flush()
            except Exception:
                logger.exception("Failed to flush Langfuse observations")
