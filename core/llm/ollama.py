import httpx
from app.config import settings
from .base import BaseLLM


class OllamaLLM(BaseLLM):
    def __init__(self, model: str = settings.ollama_model, base_url: str = settings.ollama_base_url, timeout: int = settings.ollama_timeout):
        self.model = model
        self.url = f"{base_url.rstrip('/')}/api/generate"
        self.timeout = timeout

    async def generate(self, prompt: str) -> str:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                self.url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                }
            )
        return response.json()["response"]