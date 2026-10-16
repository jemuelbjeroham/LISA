import logging

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
        logger.exception("Failder to initialize Langfuse.")
        return None
    