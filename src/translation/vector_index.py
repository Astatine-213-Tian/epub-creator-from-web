from __future__ import annotations

import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

from src.core.output import repo_root
from src.crawl.snapshot import clean_text, write_json


def cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    letters = sum(1 for ch in text if ch.isalpha() or "\u4e00" <= ch <= "\u9fff")
    return cjk / max(letters, 1)


def resolve_repo_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else repo_root() / value


def chapter_members(zf: zipfile.ZipFile) -> list[str]:
    names = zf.namelist()
    chapters = [name for name in names if "/chap_" in name and name.endswith((".xhtml", ".html"))]
    if chapters:
        return sorted(chapters)
    return sorted(
        name
        for name in names
        if name.endswith((".xhtml", ".html"))
        and not name.endswith(("nav.xhtml", "cover.xhtml"))
        and "toc" not in name.lower()
    )


def iter_reference_paragraphs(book: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(book) as zf:
        for member in chapter_members(zf):
            soup = BeautifulSoup(zf.read(member), "xml")
            body = soup.find("body")
            if body is None:
                continue
            for elem in body.find_all(["p", "li", "blockquote"], recursive=True):
                if elem.find_parent("nav") is not None:
                    continue
                text = clean_text(elem.get_text(" ", strip=True))
                if not 16 <= len(text) <= 260:
                    continue
                if cjk_ratio(text) < 0.65:
                    continue
                if text.startswith(("上一章", "下一章", "返回", "目录")):
                    continue
                rows.append({"book": book.name, "member": member, "text": text})
    return rows


def build_reference_corpus(reference_books: list[str], output: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    books = [resolve_repo_path(path) for path in reference_books]
    missing = [str(path) for path in books if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing reference books:\n" + "\n".join(missing))

    for book in books:
        for row in iter_reference_paragraphs(book):
            if row["text"] in seen:
                continue
            seen.add(row["text"])
            row["id"] = f"r{len(entries) + 1:06d}"
            row["source_path"] = str(book.relative_to(repo_root())) if book.is_relative_to(repo_root()) else str(book)
            entries.append(row)

    payload = {
        "source_books": reference_books,
        "entry_count": len(entries),
        "entries": entries,
    }
    write_json(output, payload)
    return payload


def load_sentence_transformer(model_name: str) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "sentence-transformers is required for vector indexing/search.\n"
            "Run with: uv run --with sentence-transformers --with torch --with numpy ..."
        ) from exc
    return SentenceTransformer(model_name)


def encode_texts(model: Any, texts: list[str], *, batch_size: int, prefix: str) -> Any:
    prefixed = [prefix + text for text in texts]
    return model.encode(
        prefixed,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )


def build_vector_index(
    *,
    corpus_path: Path,
    vectors_path: Path,
    metadata_path: Path,
    reference_books: list[str],
    model_name: str,
    batch_size: int,
    passage_prefix: str,
    query_prefix: str,
    float16: bool = True,
) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as exc:
        raise SystemExit(
            "numpy is required for vector indexing/search.\n"
            "Run with: uv run --with sentence-transformers --with torch --with numpy ..."
        ) from exc

    corpus = (
        json.loads(corpus_path.read_text(encoding="utf-8"))
        if corpus_path.exists()
        else build_reference_corpus(reference_books, corpus_path)
    )
    texts = [entry["text"] for entry in corpus["entries"]]
    model = load_sentence_transformer(model_name)
    vectors = encode_texts(model, texts, batch_size=batch_size, prefix=passage_prefix)
    vectors = np.asarray(vectors, dtype="float16" if float16 else "float32")
    vectors_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(vectors_path, vectors)
    metadata = {
        "model": model_name,
        "passage_prefix": passage_prefix,
        "query_prefix": query_prefix,
        "dtype": str(vectors.dtype),
        "shape": list(vectors.shape),
        "corpus": str(corpus_path),
        "vectors": str(vectors_path),
        "source_books": reference_books,
    }
    write_json(metadata_path, metadata)
    return metadata


class VectorSearcher:
    def __init__(self, metadata_path: Path, *, batch_size: int = 64) -> None:
        try:
            import numpy as np
        except ImportError as exc:
            raise SystemExit(
                "numpy is required for vector indexing/search.\n"
                "Run with: uv run --with sentence-transformers --with torch --with numpy ..."
            ) from exc

        self.np = np
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        corpus_path = Path(self.metadata["corpus"])
        self.corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
        self.entries = self.corpus["entries"]
        self.entry_by_id = {entry["id"]: entry for entry in self.entries}
        self.vectors = np.load(self.metadata["vectors"], mmap_mode="r")
        self.model = load_sentence_transformer(self.metadata["model"])
        self.batch_size = batch_size

    def search_texts(self, texts: list[str], *, top_k: int) -> list[list[tuple[str, float]]]:
        query_vectors = encode_texts(
            self.model,
            texts,
            batch_size=self.batch_size,
            prefix=self.metadata["query_prefix"],
        )
        return [self._search_vector(vector, top_k=top_k) for vector in query_vectors]

    def select_reference_bank(
        self,
        per_item_refs: dict[int, list[tuple[str, float]]],
        *,
        refs_per_paragraph: int,
        bank_size: int,
        max_refs_per_book: int,
    ) -> tuple[list[dict[str, Any]], dict[int, list[str]]]:
        aggregate: Counter[str] = Counter()
        for refs in per_item_refs.values():
            for ref_id, score in refs[:refs_per_paragraph]:
                aggregate[ref_id] += score

        selected: list[dict[str, Any]] = []
        book_counts: Counter[str] = Counter()
        for ref_id, _score in aggregate.most_common():
            entry = self.entry_by_id[ref_id]
            if book_counts[entry["book"]] >= max_refs_per_book:
                continue
            selected.append(entry)
            book_counts[entry["book"]] += 1
            if len(selected) >= bank_size:
                break

        selected_ids = {entry["id"] for entry in selected}
        paragraph_refs = {
            idx: [ref_id for ref_id, _score in refs if ref_id in selected_ids][:refs_per_paragraph]
            for idx, refs in per_item_refs.items()
        }
        return selected, paragraph_refs

    def _search_vector(self, query_vector: Any, *, top_k: int) -> list[tuple[str, float]]:
        matrix = self.vectors.astype("float32", copy=False)
        query = self.np.asarray(query_vector, dtype="float32")
        scores = matrix @ query
        if top_k >= len(scores):
            order = self.np.argsort(-scores)
        else:
            candidates = self.np.argpartition(-scores, top_k)[:top_k]
            order = candidates[self.np.argsort(-scores[candidates])]
        return [(self.entries[int(index)]["id"], float(scores[index])) for index in order[:top_k]]
