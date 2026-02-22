import re
from .base import BaseChunker, Chunk


class SmartChunker(BaseChunker):
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        meta = metadata or {}
        if not text or not text.strip():
            return []
        segments = self._split_sentences(text)
        chunks: list[Chunk] = []
        buffer: list[str] = []
        current_len = 0
        for seg in segments:
            seg_len = len(seg) + 1
            if current_len + seg_len > self.chunk_size and buffer:
                chunk_text = " ".join(buffer).strip()
                if chunk_text:
                    chunks.append({"text": chunk_text, "metadata": {**meta}})
                overlap_start = max(0, len(buffer) - self._count_overlap_segments(buffer))
                buffer = buffer[overlap_start:]
                current_len = sum(len(s) + 1 for s in buffer)
            buffer.append(seg)
            current_len += seg_len
        if buffer:
            chunk_text = " ".join(buffer).strip()
            if chunk_text:
                chunks.append({"text": chunk_text, "metadata": {**meta}})
        for i, c in enumerate(chunks):
            c["metadata"]["chunk_index"] = i
        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        text = text.replace("\n", " ").strip()
        if not text:
            return []
        parts = re.split(r"(?<=[.!?])\s+", text)
        out: list[str] = []
        for p in parts:
            if len(p) <= self.chunk_size:
                out.append(p)
            else:
                for i in range(0, len(p), self.chunk_size - self.overlap):
                    out.append(p[i : i + self.chunk_size])
        return out

    def _count_overlap_segments(self, buffer: list[str]) -> int:
        if self.overlap <= 0:
            return 0
        count = 0
        total = 0
        for s in reversed(buffer):
            total += len(s) + 1
            if total > self.overlap:
                break
            count += 1
        return count