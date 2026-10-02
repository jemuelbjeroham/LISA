from dataclasses import dataclass
from uuid import UUID

from lisa.infrastructure.postgres.memory_dependencies import MemoryDependencies


@dataclass
class LISAContext:
    user_id: UUID
    memory: MemoryDependencies