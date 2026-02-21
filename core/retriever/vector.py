from .base import BaseRetriever

class VectorRetriever(BaseRetriever):
    def __init__(self, embedding_model, vector_store):
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    async def retrieve(self, query: str, k: int = 5):
        query_vector = (await self.embedding_model.embed([query]))[0]
        results = await self.vector_store.search(query_vector, k=k)
        return results