from .base import BaseChunker, Chunk


class ParagraphChunker(BaseChunker):
    def __init__(self, chunk_size: int = 512, overlap: int = 0):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        meta = metadata or {}
        if not text or not text.strip():
            return []
        raw_paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not raw_paras:
            raw_paras = [text.strip()]
        merged: list[str] = []
        buf: list[str] = []
        buf_len = 0
        for p in raw_paras:
            need = len(p) + (2 if buf else 0)
            if buf_len + need > self.chunk_size and buf:
                merged.append("\n\n".join(buf))
                if self.overlap > 0 and buf:
                    overlap_paras = []
                    o_len = 0
                    for x in reversed(buf):
                        o_len += len(x) + 2
                        if o_len > self.overlap:
                            break
                        overlap_paras.append(x)
                    buf = list(reversed(overlap_paras))
                    buf_len = sum(len(x) + 2 for x in buf)
                else:
                    buf = []
                    buf_len = 0
            buf.append(p)
            buf_len += need
        if buf:
            merged.append("\n\n".join(buf))
        out: list[Chunk] = []
        for i, t in enumerate(merged):
            out.append({"text": t, "metadata": {**meta, "chunk_index": i}})
        return out