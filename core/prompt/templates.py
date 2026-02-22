ANSWER_FROM_CONTEXT = """You must answer only from the provided context.

Context:
{context}

Question:
{query}"""


def format_answer_prompt(context: str, query: str) -> str:
    return ANSWER_FROM_CONTEXT.format(context=context, query=query)
