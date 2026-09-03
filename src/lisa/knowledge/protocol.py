from typing import Protocol


class KnowledgeRetriever(Protocol):
    async def retrieve(self, query: str) -> list[str]:
        ...

