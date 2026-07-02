#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.runtime.progress import ProgressLogger, configure_progress
from src.translation.codex_cli import run_missing_prompts
from src.translation.pipeline import (
    build_configured_index,
    build_epub_from_run,
    load_config,
    prepare_translation_run,
    update_glossary_from_run,
    validate_translation_run,
)
from src.translation.retry import retry_failed_translations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Translate a crawl snapshot and build a bilingual EPUB.")
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Build the configured local vector index")
    p_index.add_argument("--config", type=Path, required=True)
    p_index.add_argument("--rebuild-corpus", action="store_true")

    p_prepare = sub.add_parser("prepare", help="Prepare chunk prompts for a crawl snapshot")
    p_prepare.add_argument("snapshot", type=Path)
    p_prepare.add_argument("--config", type=Path, required=True)
    p_prepare.add_argument("--run-dir", type=Path)
    p_prepare.add_argument("--no-vector", action="store_true")

    p_run = sub.add_parser("run", help="Run missing prepared prompts with Codex CLI")
    p_run.add_argument("run_dir", type=Path)
    p_run.add_argument("--config", type=Path)
    p_run.add_argument("--model")
    p_run.add_argument("--codex-bin", default="codex")
    p_run.add_argument("--chunk-id", action="append", dest="chunk_ids")
    p_run.add_argument("--overwrite", action="store_true")
    p_run.add_argument("--retry-empty", action="store_true")

    p_retry = sub.add_parser(
        "retry-empty",
        help="Retry empty/refusal translations one paragraph at a time",
    )
    p_retry.add_argument("run_dir", type=Path)
    p_retry.add_argument("--config", type=Path, required=True)
    p_retry.add_argument("--model")
    p_retry.add_argument("--codex-bin", default="codex")
    p_retry.add_argument("--max-items", type=int)
    p_retry.add_argument("--overwrite", action="store_true")

    p_validate = sub.add_parser("validate", help="Validate chunk outputs and merge chapter translations")
    p_validate.add_argument("run_dir", type=Path)
    p_validate.add_argument("--config", type=Path)
    p_validate.add_argument("--allow-missing", action="store_true")

    p_build = sub.add_parser("build-epub", help="Build a bilingual EPUB from merged translations")
    p_build.add_argument("snapshot", type=Path)
    p_build.add_argument("--run-dir", type=Path, required=True)
    p_build.add_argument("--config", type=Path, required=True)
    p_build.add_argument("-o", "--output", type=Path)

    p_all = sub.add_parser("all", help="Prepare, optionally run Codex, validate, and build EPUB")
    p_all.add_argument("snapshot", type=Path)
    p_all.add_argument("--config", type=Path, required=True)
    p_all.add_argument("--run-dir", type=Path)
    p_all.add_argument("--no-vector", action="store_true")
    p_all.add_argument("--run-codex", action="store_true")
    p_all.add_argument("--model")
    p_all.add_argument("--codex-bin", default="codex")
    p_all.add_argument("--no-retry-empty", action="store_true")
    p_all.add_argument("--allow-missing", action="store_true")
    p_all.add_argument("-o", "--output", type=Path)

    args = parser.parse_args(argv)
    configure_progress(debug=args.verbose)
    progress = ProgressLogger()

    try:
        if args.command == "index":
            config = load_config(args.config)
            progress.section("Index")
            metadata = build_configured_index(config, rebuild_corpus=args.rebuild_corpus)
            progress.info(f"wrote vector metadata {metadata['vectors']}")
            return 0

        if args.command == "prepare":
            config = load_config(args.config)
            progress.section("Prepare")
            run_dir = prepare_translation_run(
                snapshot_dir=args.snapshot,
                config=config,
                run_dir=args.run_dir,
                use_vector=not args.no_vector,
            )
            progress.info(f"wrote translation run {run_dir}")
            return 0

        if args.command == "run":
            progress.section("Codex")
            config = load_config(args.config) if args.config else {}
            model = args.model or (config.get("codex") or {}).get("model")
            count = run_missing_prompts(
                run_dir=args.run_dir,
                model=model,
                codex_bin=args.codex_bin,
                chunk_ids=set(args.chunk_ids) if args.chunk_ids else None,
                overwrite=args.overwrite,
            )
            progress.info(f"completed {count} chunk(s)")
            if args.retry_empty:
                if not args.config:
                    raise ValueError("--retry-empty requires --config")
                retry_summary = retry_failed_translations(
                    run_dir=args.run_dir,
                    config=config,
                    model=model,
                    codex_bin=args.codex_bin,
                )
                progress.info(
                    "retried empty/refusal translations "
                    f"(attempted {retry_summary['attempted']}, "
                    f"recovered {retry_summary['recovered']}, "
                    f"still missing {retry_summary['still_missing']})"
                )
            return 0

        if args.command == "retry-empty":
            progress.section("Retry Empty")
            config = load_config(args.config)
            model = args.model or (config.get("codex") or {}).get("model")
            retry_summary = retry_failed_translations(
                run_dir=args.run_dir,
                config=config,
                model=model,
                codex_bin=args.codex_bin,
                max_items=args.max_items,
                overwrite=args.overwrite,
            )
            progress.info(
                "retried empty/refusal translations "
                f"(attempted {retry_summary['attempted']}, "
                f"recovered {retry_summary['recovered']}, "
                f"still missing {retry_summary['still_missing']})"
            )
            return 0

        if args.command == "validate":
            progress.section("Validate")
            summary = validate_translation_run(args.run_dir, allow_missing=args.allow_missing)
            progress.info(f"validated {len(summary['chunks'])} chunk(s)")
            if args.config:
                config = load_config(args.config)
                glossary_summary = update_glossary_from_run(config, args.run_dir)
                if glossary_summary is not None:
                    progress.info(
                        "updated glossary "
                        f"(added {glossary_summary['added']}, kept {glossary_summary['kept']}, "
                        f"skipped {glossary_summary['skipped']})"
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
            progress.section("Prepare")
            run_dir = prepare_translation_run(
                snapshot_dir=args.snapshot,
                config=config,
                run_dir=args.run_dir,
                use_vector=not args.no_vector,
            )
            progress.info(f"wrote translation run {run_dir}")
            if args.run_codex:
                progress.section("Codex")
                model = args.model or (config.get("codex") or {}).get("model")
                count = run_missing_prompts(
                    run_dir=run_dir,
                    model=model,
                    codex_bin=args.codex_bin,
                )
                progress.info(f"completed {count} chunk(s)")
                if not args.no_retry_empty:
                    retry_summary = retry_failed_translations(
                        run_dir=run_dir,
                        config=config,
                        model=model,
                        codex_bin=args.codex_bin,
                    )
                    progress.info(
                        "retried empty/refusal translations "
                        f"(attempted {retry_summary['attempted']}, "
                        f"recovered {retry_summary['recovered']}, "
                        f"still missing {retry_summary['still_missing']})"
                    )
            progress.section("Validate")
            validate_translation_run(run_dir, allow_missing=args.allow_missing)
            glossary_summary = update_glossary_from_run(config, run_dir)
            if glossary_summary is not None:
                progress.info(
                    "updated glossary "
                    f"(added {glossary_summary['added']}, kept {glossary_summary['kept']}, "
                    f"skipped {glossary_summary['skipped']})"
                )
            progress.section("Build EPUB")
            out_path = build_epub_from_run(
                snapshot_dir=args.snapshot,
                run_dir=run_dir,
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
