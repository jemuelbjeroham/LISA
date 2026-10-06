from lisa.memory.models import Memory


class MemoryContextBuilder:
    def build(self, memories: list[Memory]) -> str:
        if not memories:
            return ""

        lines = [
            "Relevant context about the user:",
            *[
                f"- {memory.content}"
                for memory in memories
            ],
        ]

        return "\n".join(lines)