from datetime import UTC, datetime
from uuid import UUID

from rank_bm25 import BM25Okapi

from lisa.memory.models import Memory
from lisa.memory.service import MemoryService


class MemoryRetriever:
    def __init__(self, service: MemoryService):
        self.service = service

    async def retrieve(
        self,
        user_id: UUID,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[Memory]:
        memories = await self.service.list_by_user(user_id)

        if not memories or not query.strip():
            return []

        now = datetime.now(UTC)

        valid_memories = [
            memory
            for memory in memories
            if self._is_valid(memory, now)
        ]

        if not valid_memories:
            return []

        documents = [
            self._tokenize(memory.content)
            for memory in valid_memories
        ]

        bm25 = BM25Okapi(documents)

        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        scores = bm25.get_scores(query_tokens)

        ranked_memories = sorted(
            zip(valid_memories, scores),
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            memory
            for memory, score in ranked_memories[:top_k]
            if score > 0
        ]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.casefold().split()

    @staticmethod
    def _is_valid(memory: Memory, now: datetime) -> bool:
        if memory.valid_from is not None and memory.valid_from > now:
            return False

        return not (
            memory.valid_to is not None
            and memory.valid_to <= now
        )