from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.runtime.paths import repo_root
from src.crawler.snapshot import load_manifest
from src.translation.output import build_bilingual_epub
from src.translation.glossary import load_glossary
from src.translation.prompt_builder import prepare_prompts
from src.translation.semantic_compression import semantic_qa_summary_is_current
from src.translation.sentence_translations import apply_sentence_translation_overrides
from src.translation.style_transfer import (
    is_author_style_transfer_run,
    prepare_style_transfer_run,
    validate_style_transfer_provenance,
)
from src.translation.validation import validate_and_merge


def load_config(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["_config_path"] = str(path)
    return data


def config_path(config: dict[str, Any], key: str, default: str | None = None) -> Path | None:
    raw = config.get(key, default)
    if not raw:
        return None
    path = Path(str(raw)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (repo_root() / path).resolve()


def default_run_dir(snapshot_dir: Path, config: dict[str, Any]) -> Path:
    if config.get("run_dir"):
        configured = config_path(config, "run_dir")
        if configured is not None:
            return configured
    return repo_root() / "generated" / "translation_runs" / snapshot_dir.name


def prepare_translation_run(
    *,
    snapshot_dir: Path,
    config: dict[str, Any],
    run_dir: Path | None = None,
) -> Path:
    snapshot_dir = snapshot_dir.expanduser().resolve()
    run_dir = (run_dir or default_run_dir(snapshot_dir, config)).expanduser()
    run_dir.mkdir(parents=True, exist_ok=True)
    glossary = load_glossary(config_path(config, "glossary_path"))
    prepare_prompts(
        snapshot_dir=snapshot_dir,
        run_dir=run_dir,
        config=config,
        glossary=glossary,
    )
    return run_dir


def prepare_author_style_transfer_run(
    *,
    semantic_run_dir: Path,
    config: dict[str, Any],
    style_run_dir: Path | None = None,
    block_size: int | None = None,
) -> Path:
    semantic_run_dir = semantic_run_dir.expanduser().resolve()
    style_run_dir = (
        style_run_dir or (semantic_run_dir / "author_style_transfer")
    ).expanduser().resolve()
    prepare_style_transfer_run(
        semantic_run_dir=semantic_run_dir,
        style_run_dir=style_run_dir,
        config=config,
        block_size=block_size,
    )
    return style_run_dir


def validate_translation_run(
    run_dir: Path,
    *,
    allow_missing: bool = False,
    config: dict[str, Any] | None = None,
    snapshot_dir: Path | None = None,
) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    summary = validate_and_merge(run_dir, allow_missing=allow_missing)
    if config is not None:
        sentence_summary = protect_sentence_translations(
            run_dir=run_dir,
            config=config,
            snapshot_dir=snapshot_dir,
        )
        summary["sentence_translations"] = {
            "entry_count": sentence_summary["entry_count"],
            "occurrence_count": sentence_summary["occurrence_count"],
            "changed_occurrence_count": sentence_summary[
                "changed_occurrence_count"
            ],
            "unresolved_count": len(sentence_summary["unresolved"]),
        }
    return summary


def protect_sentence_translations(
    *,
    run_dir: Path,
    config: dict[str, Any],
    snapshot_dir: Path | None = None,
) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    glossary = load_glossary(config_path(config, "glossary_path"))
    if snapshot_dir is None:
        run_manifest = json.loads(
            (run_dir / "run_manifest.json").read_text(encoding="utf-8")
        )
        raw_snapshot_dir = run_manifest.get("snapshot_dir")
        if not raw_snapshot_dir:
            raise ValueError(
                f"translation run has no snapshot_dir: {run_dir / 'run_manifest.json'}"
            )
        snapshot_dir = Path(str(raw_snapshot_dir)).expanduser()
        if not snapshot_dir.is_absolute():
            snapshot_dir = repo_root() / snapshot_dir
    return apply_sentence_translation_overrides(
        run_dir=run_dir,
        snapshot_dir=snapshot_dir,
        glossary=glossary,
    )


def build_epub_from_run(
    *,
    snapshot_dir: Path,
    run_dir: Path,
    config: dict[str, Any],
    output: Path | None,
) -> Path:
    run_dir = run_dir.expanduser().resolve()
    validate_and_merge(run_dir, allow_missing=False)
    if is_author_style_transfer_run(run_dir):
        validate_style_transfer_provenance(run_dir)
        semantic_config = config.get("semantic_compression") or {}
        semantic_summary_path = (
            run_dir.expanduser()
            / "semantic_compression"
            / "semantic_compression_summary.json"
        )
        if not bool(semantic_config.get("auto_review")):
            raise ValueError(
                "author style-transfer EPUB build requires semantic_compression.auto_review"
            )
        if not semantic_qa_summary_is_current(
            run_dir, semantic_summary_path, config
        ):
            raise ValueError(
                "author style-transfer output has no current semantic QA summary; "
                "run `book-translate validate --config ...` before building"
            )
    protect_sentence_translations(
        run_dir=run_dir,
        config=config,
        snapshot_dir=snapshot_dir,
    )
    manifest = load_manifest(snapshot_dir)
    return build_bilingual_epub(
        snapshot_dir=snapshot_dir,
        translations_dir=run_dir / "translations",
        output=output,
        title=str(config.get("title") or manifest.get("title") or ""),
        author=str(config.get("author") or manifest.get("author") or ""),
    )
