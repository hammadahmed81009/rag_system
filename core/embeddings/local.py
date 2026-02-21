from sentence_transformers import SentenceTransformer
from .base import BaseEmbedding

class LocalEmbedding(BaseEmbedding):
    def __init__(self, model_name="BAAI/bge-small-en"):
        self.model = SentenceTransformer(model_name)

    async def embed(self, texts):
        return self.model.encode(texts, show_progress_bar=False).tolist()