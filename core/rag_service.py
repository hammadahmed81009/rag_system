from core.prompt.templates import format_answer_prompt


class RAGService:
    def __init__(self, retriever, llm, top_k: int = 5):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k

    async def answer(self, query: str):
        docs = await self.retriever.retrieve(query, k=self.top_k)
        context = "\n\n".join(
            [doc.payload.get("text", "") for doc in docs]
        )
        prompt = format_answer_prompt(context, query)
        response = await self.llm.generate(prompt)
        return {
            "answer": response,
            "sources": docs
        }