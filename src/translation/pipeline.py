from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.core.output import repo_root
from src.crawl.snapshot import load_manifest
from src.translation.epub_builder import build_bilingual_epub
from src.translation.glossary import load_glossary, merge_glossary_candidates
from src.translation.prompt_builder import prepare_prompts
from src.translation.validation import validate_and_merge
from src.translation.vector_index import VectorSearcher, build_vector_index, resolve_repo_path


def load_config(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["_config_path"] = str(path)
    return data


def config_path(config: dict[str, Any], key: str, default: str | None = None) -> Path | None:
    raw = config.get(key, default)
    if not raw:
        return None
    return resolve_repo_path(str(raw))


def default_run_dir(snapshot_dir: Path, config: dict[str, Any]) -> Path:
    if config.get("run_dir"):
        return resolve_repo_path(str(config["run_dir"]))
    return repo_root() / "generated" / "translation_runs" / snapshot_dir.name


def build_configured_index(config: dict[str, Any], *, rebuild_corpus: bool = False) -> dict[str, Any]:
    index_config = config.get("index") or {}
    reference_books = list(config.get("reference_books") or [])
    if not reference_books:
        raise ValueError("translation config must define reference_books")
    corpus_path = resolve_repo_path(index_config.get("corpus_path") or "generated/vector_indexes/default/corpus.json")
    vectors_path = resolve_repo_path(index_config.get("vectors_path") or "generated/vector_indexes/default/vectors.npy")
    metadata_path = resolve_repo_path(index_config.get("metadata_path") or "generated/vector_indexes/default/metadata.json")
    if rebuild_corpus and corpus_path.exists():
        corpus_path.unlink()
    return build_vector_index(
        corpus_path=corpus_path,
        vectors_path=vectors_path,
        metadata_path=metadata_path,
        reference_books=reference_books,
        model_name=str(index_config.get("model") or "intfloat/multilingual-e5-small"),
        batch_size=int(index_config.get("batch_size") or 64),
        passage_prefix=str(index_config.get("passage_prefix") or "passage: "),
        query_prefix=str(index_config.get("query_prefix") or "query: "),
        float16=bool(index_config.get("float16", True)),
    )


def load_vector_searcher(config: dict[str, Any], *, enabled: bool = True) -> VectorSearcher | None:
    if not enabled:
        return None
    index_config = config.get("index") or {}
    metadata_path = resolve_repo_path(index_config.get("metadata_path") or "generated/vector_indexes/default/metadata.json")
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"vector metadata not found: {metadata_path}\n"
            "Run `book-translate index --config ...` first, or use prepare --no-vector."
        )
    return VectorSearcher(metadata_path, batch_size=int(index_config.get("query_batch_size") or 64))


def prepare_translation_run(
    *,
    snapshot_dir: Path,
    config: dict[str, Any],
    run_dir: Path | None = None,
    use_vector: bool = True,
) -> Path:
    snapshot_dir = snapshot_dir.expanduser().resolve()
    run_dir = (run_dir or default_run_dir(snapshot_dir, config)).expanduser()
    run_dir.mkdir(parents=True, exist_ok=True)
    glossary = load_glossary(config_path(config, "glossary_path"))
    searcher = load_vector_searcher(config, enabled=use_vector)
    prepare_prompts(
        snapshot_dir=snapshot_dir,
        run_dir=run_dir,
        config=config,
        glossary=glossary,
        vector_searcher=searcher,
    )
    return run_dir


def validate_translation_run(run_dir: Path, *, allow_missing: bool = False) -> dict[str, Any]:
    return validate_and_merge(run_dir.expanduser(), allow_missing=allow_missing)


def update_glossary_from_run(config: dict[str, Any], run_dir: Path) -> dict[str, int] | None:
    glossary_path = config_path(config, "glossary_path")
    if glossary_path is None:
        return None
    candidates_path = run_dir.expanduser() / "glossary_candidates.json"
    if not candidates_path.exists():
        raise FileNotFoundError(
            f"glossary candidates not found: {candidates_path}\n"
            "Run `book-translate validate ...` before updating the glossary."
        )
    return merge_glossary_candidates(
        glossary_path=glossary_path,
        candidates_path=candidates_path,
    )


def build_epub_from_run(
    *,
    snapshot_dir: Path,
    run_dir: Path,
    config: dict[str, Any],
    output: Path | None,
) -> Path:
    manifest = load_manifest(snapshot_dir)
    return build_bilingual_epub(
        snapshot_dir=snapshot_dir,
        translations_dir=run_dir / "translations",
        output=output,
        title=str(config.get("title") or manifest.get("title") or ""),
        author=str(config.get("author") or manifest.get("author") or ""),
    )
