from pathlib import Path
from typing import Protocol


class LoadedDoc:
    def __init__(self, content: str, path: str):
        self.content = content
        self.path = path


class BaseLoader(Protocol):
    def load(self, path: Path) -> LoadedDoc | None:
        ...


def load_text(path: Path) -> LoadedDoc | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        return LoadedDoc(text, str(path))
    except Exception:
        return None


def load_markdown(path: Path) -> LoadedDoc | None:
    return load_text(path)


EXTENSION_LOADERS: dict[str, BaseLoader] = {
    ".txt": load_text,
    ".md": load_markdown,
    ".markdown": load_markdown,
}


def load_file(path: Path) -> LoadedDoc | None:
    suffix = path.suffix.lower()
    loader = EXTENSION_LOADERS.get(suffix)
    if loader is None:
        return None
    return loader(path)


def discover_files(root: Path, extensions: set[str] | None = None) -> list[Path]:
    exts = extensions or set(EXTENSION_LOADERS.keys())
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]