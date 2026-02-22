import pytest

from core.chunking.paragraph_chunker import ParagraphChunker
from core.chunking.smart_chunker import SmartChunker


class TestSmartChunker:
    def test_empty_returns_empty(self):
        chunker = SmartChunker(chunk_size=512, overlap=50)
        assert chunker.chunk("") == []
        assert chunker.chunk("   \n  ") == []

    def test_short_text_one_chunk(self):
        chunker = SmartChunker(chunk_size=512, overlap=50)
        text = "One short sentence."
        chunks = chunker.chunk(text, metadata={"source": "test.txt"})
        assert len(chunks) == 1
        assert chunks[0]["text"] == text
        assert chunks[0]["metadata"]["source"] == "test.txt"
        assert chunks[0]["metadata"]["chunk_index"] == 0

    def test_large_text_multiple_chunks(self):
        chunker = SmartChunker(chunk_size=100, overlap=20)
        sentences = ["First sentence here.", "Second one here.", "Third."] * 15
        text = " ".join(sentences)
        chunks = chunker.chunk(text, metadata={"source": "doc.md"})
        assert len(chunks) >= 2
        for i, c in enumerate(chunks):
            assert "text" in c and "metadata" in c
            assert c["metadata"]["chunk_index"] == i
            assert c["metadata"]["source"] == "doc.md"

    def test_metadata_preserved(self):
        chunker = SmartChunker(chunk_size=512, overlap=0)
        chunks = chunker.chunk("Hello world.", metadata={"source": "x", "custom": "value"})
        assert len(chunks) == 1
        assert chunks[0]["metadata"]["source"] == "x"
        assert chunks[0]["metadata"]["custom"] == "value"
        assert "chunk_index" in chunks[0]["metadata"]


class TestParagraphChunker:
    def test_empty_returns_empty(self):
        chunker = ParagraphChunker(chunk_size=512, overlap=0)
        assert chunker.chunk("") == []
        assert chunker.chunk("   \n\n  ") == []

    def test_single_paragraph_one_chunk(self):
        chunker = ParagraphChunker(chunk_size=512, overlap=0)
        text = "One paragraph of text."
        chunks = chunker.chunk(text, metadata={"source": "test.txt"})
        assert len(chunks) == 1
        assert chunks[0]["text"] == text
        assert chunks[0]["metadata"]["source"] == "test.txt"
        assert chunks[0]["metadata"]["chunk_index"] == 0

    def test_multiple_paragraphs_respected(self):
        chunker = ParagraphChunker(chunk_size=10, overlap=0)
        text = "Para one.\n\nPara two.\n\nPara three."
        chunks = chunker.chunk(text, metadata={})
        assert len(chunks) == 3
        assert "Para one" in chunks[0]["text"]
        assert "Para two" in chunks[1]["text"]
        assert "Para three" in chunks[2]["text"]
        for i, c in enumerate(chunks):
            assert c["metadata"]["chunk_index"] == i

    def test_metadata_preserved(self):
        chunker = ParagraphChunker(chunk_size=512, overlap=0)
        chunks = chunker.chunk("Hello.", metadata={"source": "y"})
        assert chunks[0]["metadata"]["source"] == "y"
        assert "chunk_index" in chunks[0]["metadata"]
