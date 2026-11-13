import logging
from collections.abc import Generator
from contextlib import contextmanager
from functools import wraps
from inspect import isasyncgenfunction, iscoroutinefunction
from typing import Any

from langchain_core.runnables.config import (
    ensure_config,
    merge_configs,
    var_child_runnable_config,
)
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

logger = logging.getLogger(__name__)


def create_langfuse_handler(public_key: str | None = None) -> CallbackHandler:
    """Create a handler for LangChain/LangGraph tracing."""
    return CallbackHandler(public_key=public_key)

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

    def instrument(self, agent: Any, method_name: str, handler: Any) -> None:
        if self.client is None:
            return
        method = getattr(agent, method_name)
        name = f"lisa.agent.{type(agent).__name__}.{method_name}"

        @contextmanager
        def scope():
            config = ensure_config()
            if handler is not None:
                config = merge_configs(config, {"callbacks": [handler]})
            token = var_child_runnable_config.set(config)
            try:
                with self.span(name):
                    yield
            finally:
                var_child_runnable_config.reset(token)

        if isasyncgenfunction(method):
            @wraps(method)
            async def traced(*args, **kwargs):
                with scope():
                    async for event in method(*args, **kwargs):
                        yield event
        elif iscoroutinefunction(method):
            @wraps(method)
            async def traced(*args, **kwargs):
                with scope():
                    return await method(*args, **kwargs)
        else:
            @wraps(method)
            def traced(*args, **kwargs):
                with scope():
                    return method(*args, **kwargs)

        setattr(agent, method_name, traced)

    @contextmanager
    def span(
        self,
        name: str,
        *,
        input: dict[str, Any] | None = None,
        as_type: str = "span",
        **attributes: Any,
    ) -> Generator[Any, None, None]:  # noqa: UP043
        """Open a Langfuse observation nested under the current one.

        `as_type="generation"` records a model call; extra `attributes`
        (for example `model` or `metadata`) are passed to Langfuse as-is.
        Yields None when tracing is disabled.
        """
        if self.client is None:
            yield None
            return

        with self.client.start_as_current_observation(
            as_type=as_type,
            name=name,
            input=input,
            **attributes,
        ) as observation:
            yield observation

    def flush(self) -> None:
        if self.client is not None:
            try:
                self.client.flush()
            except Exception:
                logger.exception("Failed to flush Langfuse observations")
