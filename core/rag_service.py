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

        prompt = f"""
        You must answer only from the provided context.

        Context:
        {context}

        Question:
        {query}
        """

        response = await self.llm.generate(prompt)

        return {
            "answer": response,
            "sources": docs
        }