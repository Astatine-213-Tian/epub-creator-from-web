#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.runtime.progress import ProgressLogger, configure_progress
from src.translation.codex_cli import run_missing_prompts_with_model_fallback
from src.translation.comments import write_glossary_comment_evidence
from src.translation.pipeline import (
    build_epub_from_run,
    default_run_dir,
    load_config,
    prepare_author_style_transfer_run,
    prepare_translation_run,
    validate_translation_run,
)
from src.translation.semantic_compression import (
    DEFAULT_BATCH_SIZE as DEFAULT_COMPRESSION_BATCH_SIZE,
)
from src.translation.semantic_compression import (
    DEFAULT_MAX_CANDIDATES as DEFAULT_COMPRESSION_MAX_CANDIDATES,
)
from src.translation.semantic_compression import (
    DEFAULT_MIN_CONFIDENCE as DEFAULT_COMPRESSION_MIN_CONFIDENCE,
)
from src.translation.semantic_compression import (
    run_semantic_compression_qa,
    semantic_qa_summary_is_current,
)
from src.translation.style_transfer import (
    configured_model_order,
    is_author_style_transfer_run,
    run_style_transfer,
    validate_style_transfer_provenance,
)


def _add_model_order_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help=(
            "Codex model in capacity-fallback order; repeat to override the "
            "configured order. Models switch only on explicit capacity or "
            "availability failures."
        ),
    )


def _codex_timeout(config: dict[str, Any], override: int | None) -> int | None:
    if override is not None:
        return override
    value = (config.get("codex") or {}).get("timeout_seconds")
    return int(value) if value is not None else None


def _log_adaptive_summary(progress: ProgressLogger, summary: dict[str, Any]) -> None:
    progress.info(
        "completed neutral translation "
        f"(chunks {summary['completed']}, "
        f"skipped {summary['skipped']}, "
        f"chunk fallbacks {summary['chunk_failures']}, "
        f"local windows {summary['local_window_attempts']}, "
        f"paragraph attempts {summary['paragraph_attempts']}, "
        f"recovered {summary['recovered']}, "
        f"still missing {summary['still_missing']}, "
        f"model {summary.get('effective_model') or 'default'})"
    )
    unavailable = summary.get("unavailable_models") or {}
    if unavailable:
        progress.info("capacity fallback skipped " + ", ".join(unavailable))


def _run_neutral(
    *,
    run_dir: Path,
    config: dict[str, Any],
    models: list[str] | None,
    codex_bin: str,
    timeout_seconds: int | None,
    chunk_ids: set[str] | None,
    overwrite: bool,
    progress: ProgressLogger,
) -> dict[str, Any]:
    summary = run_missing_prompts_with_model_fallback(
        run_dir=run_dir,
        models=configured_model_order(config, models),
        codex_bin=codex_bin,
        chunk_ids=chunk_ids,
        overwrite=overwrite,
        timeout_seconds=_codex_timeout(config, timeout_seconds),
    )
    _log_adaptive_summary(progress, summary)
    return summary


def _run_style(
    *,
    run_dir: Path,
    config: dict[str, Any],
    models: list[str] | None,
    codex_bin: str,
    reasoning_effort: str | None,
    timeout_seconds: int | None,
    max_attempts: int | None,
    workers: int | None,
    chunk_ids: set[str] | None,
    overwrite: bool,
    progress: ProgressLogger,
) -> dict[str, Any]:
    summary = run_style_transfer(
        style_run_dir=run_dir,
        config=config,
        models=models,
        codex_bin=codex_bin,
        reasoning_effort=reasoning_effort,
        timeout_seconds=timeout_seconds,
        max_attempts=max_attempts,
        workers=workers,
        chunk_ids=chunk_ids,
        overwrite=overwrite,
    )
    progress.info(
        "completed author style transfer "
        f"(completed {summary['completed_count']}, "
        f"skipped {summary['skipped_count']}, "
        f"failures {summary['failure_count']}, "
        f"models {summary['model_chunk_counts']})"
    )
    if int(summary["failure_count"]) > 0:
        raise RuntimeError(
            f"author style transfer has {summary['failure_count']} failed chunk(s)"
        )
    return summary


def _log_glossary_candidates(
    progress: ProgressLogger,
    run_dir: Path,
    summary: dict[str, Any],
) -> None:
    count = int(summary.get("glossary_candidate_count") or 0)
    progress.info(
        f"wrote {count} glossary candidate(s) to "
        f"{run_dir.expanduser().resolve() / 'glossary_candidates.json'} for review; "
        "maintained glossary unchanged"
    )


def _run_semantic_qa(
    *,
    run_dir: Path,
    config: dict[str, Any],
    models: list[str] | None,
    codex_bin: str,
    timeout_seconds: int | None,
    output_dir: Path | None = None,
    max_candidates: int | None = None,
    batch_size: int | None = None,
    auto_repair: bool | None = None,
    min_confidence: str | None = None,
    dry_run: bool = False,
    overwrite: bool = False,
    progress: ProgressLogger,
) -> dict[str, Any]:
    semantic_config = config.get("semantic_compression") or {}
    codex_config = config.get("codex") or {}
    effective_timeout = (
        timeout_seconds
        if timeout_seconds is not None
        else semantic_config.get("timeout_seconds", codex_config.get("timeout_seconds"))
    )
    summary = run_semantic_compression_qa(
        run_dir=run_dir,
        config=config,
        output_dir=output_dir,
        max_candidates=(
            max_candidates
            if max_candidates is not None
            else int(
                semantic_config.get("max_candidates")
                or DEFAULT_COMPRESSION_MAX_CANDIDATES
            )
        ),
        batch_size=(
            batch_size
            if batch_size is not None
            else int(
                semantic_config.get("batch_size") or DEFAULT_COMPRESSION_BATCH_SIZE
            )
        ),
        models=configured_model_order(config, models),
        codex_bin=codex_bin,
        timeout_seconds=int(effective_timeout)
        if effective_timeout is not None
        else None,
        auto_repair=(
            bool(semantic_config.get("auto_repair"))
            if auto_repair is None
            else auto_repair
        ),
        min_auto_apply_confidence=str(
            min_confidence
            or semantic_config.get("min_auto_apply_confidence")
            or DEFAULT_COMPRESSION_MIN_CONFIDENCE
        ),
        dry_run=dry_run,
        overwrite=overwrite,
    )
    result = summary["result_summary"]
    progress.info(
        "semantic QA complete "
        f"(candidates {summary['candidate_summary']['candidate_count']}, "
        f"reviewed {summary['review_summary']['reviewed_count']}, "
        f"applied {result['applied_count']}, "
        f"skipped {result['skipped_count']}, "
        f"rejected {result['rejected_count']}, "
        f"unresolved losses {result.get('unresolved_true_loss_count', 0)}, "
        f"failures {result['failure_count']})"
    )
    if int(result["failure_count"]) > 0:
        raise RuntimeError(
            f"semantic QA has {result['failure_count']} failed review batch(es)"
        )
    if int(result["rejected_count"]) > 0:
        raise RuntimeError(
            f"semantic QA has {result['rejected_count']} rejected candidate(s)"
        )
    if int(result.get("unresolved_true_loss_count") or 0) > 0:
        raise RuntimeError(
            "semantic QA has "
            f"{result['unresolved_true_loss_count']} unresolved true-loss candidate(s)"
        )
    return summary


def _validate_with_optional_semantic_qa(
    *,
    run_dir: Path,
    config: dict[str, Any] | None,
    allow_missing: bool,
    models: list[str] | None,
    codex_bin: str,
    timeout_seconds: int | None,
    skip_semantic_qa: bool,
    semantic_overwrite: bool,
    progress: ProgressLogger,
) -> dict[str, Any]:
    if is_author_style_transfer_run(run_dir):
        provenance = validate_style_transfer_provenance(run_dir)
        progress.info(
            f"verified {provenance['validated_chunk_count']} style artifact binding(s)"
        )
    summary = validate_translation_run(
        run_dir,
        allow_missing=allow_missing,
        config=config,
    )
    progress.info(f"validated {len(summary['chunks'])} chunk(s)")
    if config and not skip_semantic_qa and is_author_style_transfer_run(run_dir):
        if not bool((config.get("semantic_compression") or {}).get("auto_review")):
            raise ValueError(
                "author style-transfer validation requires "
                "semantic_compression.auto_review"
            )
        qa_path = run_dir / "semantic_compression" / "semantic_compression_summary.json"
        qa_is_current = semantic_qa_summary_is_current(run_dir, qa_path, config)
        if qa_is_current and not semantic_overwrite:
            progress.info(f"skipping existing semantic QA summary {qa_path}")
        else:
            if qa_path.exists() and not qa_is_current:
                progress.info("styled outputs changed; rerunning semantic QA")
            progress.section("Semantic QA")
            qa_summary = _run_semantic_qa(
                run_dir=run_dir,
                config=config,
                models=models,
                codex_bin=codex_bin,
                timeout_seconds=timeout_seconds,
                overwrite=semantic_overwrite or not qa_is_current,
                progress=progress,
            )
            if int(qa_summary["result_summary"]["applied_count"]) > 0:
                progress.section("Validate Repaired")
                summary = validate_translation_run(
                    run_dir,
                    allow_missing=allow_missing,
                    config=config,
                )
                progress.info(
                    f"validated repaired run {len(summary['chunks'])} chunk(s)"
                )
    sentence_summary = summary.get("sentence_translations")
    if sentence_summary:
        progress.info(
            "protected authoritative sentence translations "
            f"({sentence_summary['occurrence_count']} occurrence(s), "
            f"{sentence_summary['changed_occurrence_count']} changed)"
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Translate a crawl snapshot and build a bilingual EPUB."
    )
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_comments = sub.add_parser(
        "comment-evidence",
        help="Extract authoritative reply threads for glossary review",
    )
    p_comments.add_argument("snapshot", type=Path)
    p_comments.add_argument("--config", type=Path, required=True)
    p_comments.add_argument("--output", type=Path)

    p_prepare = sub.add_parser("prepare", help="Prepare neutral Chinese prompts")
    p_prepare.add_argument("snapshot", type=Path)
    p_prepare.add_argument("--config", type=Path, required=True)
    p_prepare.add_argument("--run-dir", type=Path)

    p_run = sub.add_parser("run", help="Run neutral Chinese prompts with Codex")
    p_run.add_argument("run_dir", type=Path)
    p_run.add_argument("--config", type=Path)
    _add_model_order_argument(p_run)
    p_run.add_argument("--codex-bin", default="codex")
    p_run.add_argument("--timeout-seconds", type=int)
    p_run.add_argument("--chunk-id", action="append", dest="chunk_ids")
    p_run.add_argument("--overwrite", action="store_true")

    p_transfer = sub.add_parser(
        "transfer-style",
        help="Prepare and optionally run content-plan author style transfer",
    )
    p_transfer.add_argument(
        "run_dir", type=Path, help="Validated neutral run directory"
    )
    p_transfer.add_argument("--config", type=Path, required=True)
    p_transfer.add_argument("--style-run-dir", type=Path)
    p_transfer.add_argument("--block-size", type=int)
    p_transfer.add_argument("--run-codex", action="store_true")
    _add_model_order_argument(p_transfer)
    p_transfer.add_argument("--codex-bin", default="codex")
    p_transfer.add_argument("--reasoning-effort")
    p_transfer.add_argument("--timeout-seconds", type=int)
    p_transfer.add_argument("--max-attempts", type=int)
    p_transfer.add_argument("--workers", type=int)
    p_transfer.add_argument("--chunk-id", action="append", dest="chunk_ids")
    p_transfer.add_argument("--overwrite", action="store_true")

    p_validate = sub.add_parser("validate", help="Validate and assemble run outputs")
    p_validate.add_argument("run_dir", type=Path)
    p_validate.add_argument("--config", type=Path)
    p_validate.add_argument("--allow-missing", action="store_true")
    _add_model_order_argument(p_validate)
    p_validate.add_argument("--codex-bin", default="codex")
    p_validate.add_argument("--timeout-seconds", type=int)
    p_validate.add_argument(
        "--skip-semantic-qa",
        action="store_true",
        help="Run structural validation only; EPUB build still requires current QA.",
    )
    p_validate.add_argument("--semantic-qa-overwrite", action="store_true")

    p_semantic = sub.add_parser(
        "semantic-qa",
        help="Review and optionally repair style-transfer semantic losses",
    )
    p_semantic.add_argument("run_dir", type=Path, help="Author style-transfer run")
    p_semantic.add_argument("--config", type=Path, required=True)
    p_semantic.add_argument("--output-dir", type=Path)
    p_semantic.add_argument("--max-candidates", type=int)
    p_semantic.add_argument("--batch-size", type=int)
    _add_model_order_argument(p_semantic)
    p_semantic.add_argument("--codex-bin", default="codex")
    p_semantic.add_argument("--timeout-seconds", type=int)
    p_semantic.add_argument("--auto-repair", action="store_true")
    p_semantic.add_argument("--min-auto-apply-confidence")
    p_semantic.add_argument("--dry-run", action="store_true")
    p_semantic.add_argument("--overwrite", action="store_true")

    p_build = sub.add_parser("build-epub", help="Build a bilingual EPUB")
    p_build.add_argument("snapshot", type=Path)
    p_build.add_argument("--run-dir", type=Path, required=True)
    p_build.add_argument("--config", type=Path, required=True)
    p_build.add_argument("-o", "--output", type=Path)

    p_all = sub.add_parser(
        "all",
        help="Run neutral translation, author style transfer, semantic QA, and EPUB build",
    )
    p_all.add_argument("snapshot", type=Path)
    p_all.add_argument("--config", type=Path, required=True)
    p_all.add_argument("--run-dir", type=Path, help="Neutral translation run directory")
    p_all.add_argument("--style-run-dir", type=Path)
    p_all.add_argument("--run-codex", action="store_true")
    _add_model_order_argument(p_all)
    p_all.add_argument("--codex-bin", default="codex")
    p_all.add_argument("--reasoning-effort")
    p_all.add_argument("--timeout-seconds", type=int)
    p_all.add_argument("--max-attempts", type=int)
    p_all.add_argument("--workers", type=int)
    p_all.add_argument("--block-size", type=int)
    p_all.add_argument("--overwrite", action="store_true")
    p_all.add_argument("--semantic-qa-overwrite", action="store_true")
    p_all.add_argument("-o", "--output", type=Path)
    sub.metavar = (
        "{comment-evidence,prepare,run,transfer-style,validate,semantic-qa,"
        "build-epub,all}"
    )

    args = parser.parse_args(argv)
    configure_progress(debug=args.verbose)
    progress = ProgressLogger()

    try:
        if args.command == "comment-evidence":
            config = load_config(args.config)
            output_path = args.output or (
                default_run_dir(args.snapshot, config)
                / "glossary_comment_evidence.json"
            )
            progress.section("Authoritative Comment Evidence")
            evidence = write_glossary_comment_evidence(
                snapshot_dir=args.snapshot,
                output_path=output_path,
            )
            progress.info(
                f"wrote {evidence['thread_count']} thread(s) with "
                f"{evidence['authoritative_reply_count']} authoritative replies "
                f"to {output_path.expanduser().resolve()} for glossary review"
            )
            return 0

        if args.command == "prepare":
            config = load_config(args.config)
            progress.section("Prepare Neutral")
            run_dir = prepare_translation_run(
                snapshot_dir=args.snapshot,
                config=config,
                run_dir=args.run_dir,
            )
            progress.info(f"wrote neutral translation run {run_dir}")
            return 0

        if args.command == "run":
            config = load_config(args.config) if args.config else {}
            progress.section("Neutral Translation")
            _run_neutral(
                run_dir=args.run_dir,
                config=config,
                models=args.models,
                codex_bin=args.codex_bin,
                timeout_seconds=args.timeout_seconds,
                chunk_ids=set(args.chunk_ids) if args.chunk_ids else None,
                overwrite=args.overwrite,
                progress=progress,
            )
            return 0

        if args.command == "transfer-style":
            config = load_config(args.config)
            progress.section("Prepare Author Style Transfer")
            style_run_dir = prepare_author_style_transfer_run(
                semantic_run_dir=args.run_dir,
                config=config,
                style_run_dir=args.style_run_dir,
                block_size=args.block_size,
            )
            progress.info(f"wrote author style-transfer run {style_run_dir}")
            if args.run_codex:
                progress.section("Author Style Transfer")
                _run_style(
                    run_dir=style_run_dir,
                    config=config,
                    models=args.models,
                    codex_bin=args.codex_bin,
                    reasoning_effort=args.reasoning_effort,
                    timeout_seconds=args.timeout_seconds,
                    max_attempts=args.max_attempts,
                    workers=args.workers,
                    chunk_ids=set(args.chunk_ids) if args.chunk_ids else None,
                    overwrite=args.overwrite,
                    progress=progress,
                )
            return 0

        if args.command == "validate":
            config = load_config(args.config) if args.config else None
            progress.section("Validate")
            summary = _validate_with_optional_semantic_qa(
                run_dir=args.run_dir,
                config=config,
                allow_missing=args.allow_missing,
                models=args.models,
                codex_bin=args.codex_bin,
                timeout_seconds=args.timeout_seconds,
                skip_semantic_qa=args.skip_semantic_qa,
                semantic_overwrite=args.semantic_qa_overwrite,
                progress=progress,
            )
            if not is_author_style_transfer_run(args.run_dir):
                _log_glossary_candidates(progress, args.run_dir, summary)
            return 0

        if args.command == "semantic-qa":
            config = load_config(args.config)
            progress.section("Semantic QA")
            _run_semantic_qa(
                run_dir=args.run_dir,
                config=config,
                models=args.models,
                codex_bin=args.codex_bin,
                timeout_seconds=args.timeout_seconds,
                output_dir=args.output_dir,
                max_candidates=args.max_candidates,
                batch_size=args.batch_size,
                auto_repair=args.auto_repair
                or bool((config.get("semantic_compression") or {}).get("auto_repair")),
                min_confidence=args.min_auto_apply_confidence,
                dry_run=args.dry_run,
                overwrite=args.overwrite,
                progress=progress,
            )
            return 0

        if args.command == "build-epub":
            config = load_config(args.config)
            progress.section("Build EPUB")
            out_path = build_epub_from_run(
                snapshot_dir=args.snapshot,
                run_dir=args.run_dir,
                config=config,
                output=args.output,
            )
            progress.info(f"wrote {out_path}")
            return 0

        if args.command == "all":
            config = load_config(args.config)
            progress.section("Prepare Neutral")
            semantic_run_dir = prepare_translation_run(
                snapshot_dir=args.snapshot,
                config=config,
                run_dir=args.run_dir,
            )
            progress.info(f"wrote neutral translation run {semantic_run_dir}")
            if args.run_codex:
                progress.section("Neutral Translation")
                _run_neutral(
                    run_dir=semantic_run_dir,
                    config=config,
                    models=args.models,
                    codex_bin=args.codex_bin,
                    timeout_seconds=args.timeout_seconds,
                    chunk_ids=None,
                    overwrite=args.overwrite,
                    progress=progress,
                )
            progress.section("Validate Neutral")
            neutral_summary = validate_translation_run(
                semantic_run_dir,
                allow_missing=False,
                config=config,
                snapshot_dir=args.snapshot,
            )
            _log_glossary_candidates(
                progress,
                semantic_run_dir,
                neutral_summary,
            )

            progress.section("Prepare Author Style Transfer")
            style_run_dir = prepare_author_style_transfer_run(
                semantic_run_dir=semantic_run_dir,
                config=config,
                style_run_dir=args.style_run_dir,
                block_size=args.block_size,
            )
            progress.info(f"wrote author style-transfer run {style_run_dir}")
            if args.run_codex:
                progress.section("Author Style Transfer")
                _run_style(
                    run_dir=style_run_dir,
                    config=config,
                    models=args.models,
                    codex_bin=args.codex_bin,
                    reasoning_effort=args.reasoning_effort,
                    timeout_seconds=args.timeout_seconds,
                    max_attempts=args.max_attempts,
                    workers=args.workers,
                    chunk_ids=None,
                    overwrite=args.overwrite,
                    progress=progress,
                )
            progress.section("Validate Styled Chinese")
            _validate_with_optional_semantic_qa(
                run_dir=style_run_dir,
                config=config,
                allow_missing=False,
                models=args.models,
                codex_bin=args.codex_bin,
                timeout_seconds=args.timeout_seconds,
                skip_semantic_qa=False,
                semantic_overwrite=args.semantic_qa_overwrite,
                progress=progress,
            )
            progress.section("Build EPUB")
            out_path = build_epub_from_run(
                snapshot_dir=args.snapshot,
                run_dir=style_run_dir,
                config=config,
                output=args.output,
            )
            progress.info(f"wrote {out_path}")
            return 0
    except Exception as exc:  # noqa: BLE001
        progress.warning(str(exc))
        return 1

    parser.error(f"unsupported command: {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
