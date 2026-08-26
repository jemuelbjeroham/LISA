from typing import Protocol


class KnowledgeRetriever(Protocol):
    def retrieve(self, query: str) -> list[str]:
        pass

