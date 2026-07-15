---
name: book-ingest-pipeline
description: Use when ingesting a web novel/book in epub-creator-from-web for reading, training, or both; choosing epub/txt/both output modes; updating dataset manifests; avoiding broad TXT exports; and validating or reformatting generated EPUBs.
---

# Book Ingest Pipeline

Use the repo CLI as the source of truth. Do not hand-roll a one-off EPUB/TXT
export unless the source is unsupported by `book-ingest`.

## Modes

- Reading only: `uv run book-ingest <target> --mode epub`
- Reading and training: `uv run book-ingest <target>` or `--mode both`
- Training only: `uv run book-ingest <target> --mode txt`

Default to `both` unless the user explicitly asks for only EPUB or only TXT.

## Output Rules

- EPUBs belong under `books/<author>/<title>.epub`.
- Dataset TXT files belong under `research/datasets/raw/<author>/<title>.txt`.
- `both` and `txt` modes must update `research/datasets/dataset_manifest.json` through the targeted upsert path.
- Do not run broad `book-dataset export-txt` without an explicit `--books-root`.
- Prefer `book-dataset upsert --txt ...` for existing TXT files.

## Metadata

Dataset metadata should use maintained Jinjiang crawl data/fetching when present,
then Codex classification as fallback. Manual `--time-area` and `--genre`
overrides are acceptable when the user provides them.

## EPUB Validation And Reformatting

`book-ingest` validates EPUB ZIP integrity and chapter numbering. If validation
or visual inspection shows reader-visible structure issues, use the
`epub-reformatter` skill and patch the EPUB archive with synchronized
`nav.xhtml`, `toc.ncx`, chapter XHTML, and `content.opf` changes.

## Standalone Commands

```bash
uv run book-dataset upsert --txt research/datasets/raw/black_di/替罪羊.txt --author black_di --title 替罪羊
uv run book-dataset export-txt --epub books/black_di/替罪羊.epub
uv run book-dataset export-txt --books-root books
```
