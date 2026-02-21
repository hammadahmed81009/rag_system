from abc import ABC, abstractmethod
from typing import List, Dict

class BaseRetriever(ABC):
    @abstractmethod
    async def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        pass