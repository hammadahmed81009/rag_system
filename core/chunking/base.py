from abc import ABC, abstractmethod
from typing import TypedDict


class Chunk(TypedDict):
    text: str
    metadata: dict


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        pass