from .base import BaseEmbedding


class OpenAIEmbedding(BaseEmbedding):
    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None):
        self.model = model
        self._api_key = api_key or ""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        from openai import AsyncOpenAI

        if not texts:
            return []
        client = AsyncOpenAI(api_key=self._api_key)
        response = await client.embeddings.create(input=texts, model=self.model)
        return [item.embedding for item in response.data]
