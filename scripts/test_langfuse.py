from langfuse import Langfuse

from lisa.config import Settings

settings = Settings()

if not settings.langfuse_public_key or not settings.langfuse_secret_key:
    raise RuntimeError(
        "Langfuse credentials were not loaded. "
        "Check LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in .env."
    )

langfuse = Langfuse(
    public_key=settings.langfuse_public_key,
    secret_key=settings.langfuse_secret_key,
    base_url=settings.langfuse_base_url,
)

with langfuse.start_as_current_observation(
    as_type="span",
    name="lisa-connection-test",
    input={"query": "Can Langfuse observe LISA?"},
) as span:
    span.update(
        output={
            "status": "success",
            "message": "LISA is connected to Langfuse.",
        }
    )

langfuse.flush()
print("Test trace submitted.")