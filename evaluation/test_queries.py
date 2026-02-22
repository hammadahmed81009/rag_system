from dataclasses import dataclass


@dataclass
class TestCase:
    query: str
    expected_source_substr: str
    expected_phrase_in_text: str | None = None


TEST_CASES: list[TestCase] = [
    TestCase(
        query="What is Qdrant?",
        expected_source_substr="readme.txt",
        expected_phrase_in_text="vector database",
    ),
    TestCase(
        query="What is chunking?",
        expected_source_substr="chunking.md",
        expected_phrase_in_text="splits long texts",
    ),
    TestCase(
        query="How does the RAG system work?",
        expected_source_substr="readme.txt",
        expected_phrase_in_text="retrieval-augmented",
    ),
]