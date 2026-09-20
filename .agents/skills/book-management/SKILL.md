---
name: book-management
description: Add, find, update, or repair books in epub-creator-from-web. Use for choosing sources, crawling to local EPUB, Notion drafts and/or TXT, targeted training-dataset exports, adding or fixing crawler providers, and correcting book formatting or metadata. Dataset export is an output choice in the shared book workflow.
---

# Book Management

Use the maintained CLI for book acquisition and output. One crawl can produce
local EPUB, a Notion draft, TXT, or any combination of these.

## Choose the Task

- **Add or reacquire a book:** follow the acquisition steps below.
- **Source unsupported, incomplete or broken:** read [providers.md](references/providers.md),
  implement or repair the provider, then resume acquisition if a book output was requested.
  A provider-development-only task ends with its fixture and CLI checks.
- **Correct existing content, formatting, TOC or metadata:** read
  [formatting.md](references/formatting.md). Edit uploaded content in Notion;
  repair a local archive when requested.
- **Export an existing book for training:** use the targeted dataset command below.
  Training output uses the same collection pipeline when a fresh source is needed.

For Eternal Gate updates, follow the dedicated reuse/glossary/translation workflow
in `AGENTS.md`; preserve its established local bilingual output.

## Acquire a Book

1. Determine the output destinations from the user's request and established
   instructions. If they are missing, ask which they want: **local EPUB, Notion
   draft, TXT, or a combination**. Explain that this chooses where the book will
   be written. Do not assume Notion or add a training export without a request.
   A supplied EPUB/TXT path or “add to the training dataset” already specifies
   that destination. Ask only about unresolved choices; source research can
   continue while awaiting the answer.
2. Find or verify the source and completeness. Use `book-ingest --search` for
   supported sites, then compare previews before choosing a version. If the
   source needs development, follow the provider reference above.
3. Invoke `book-ingest` with each selected `--mode`. Crawl once; shared source
   preparation handles wording, structure and formatting before Notion/EPUB output.
   All selected destinations must finish before reporting the book complete.
4. Verify the selected outputs: EPUB ZIP/navigation, title/order and thumbnail;
   Notion's verified upload checkpoint; TXT content and its targeted dataset
   manifest entry. Report every actual path or Notion destination, with failures
   distinguished from completed outputs.

```bash
uv run book-ingest "<url>" --mode epub
uv run book-ingest "<url>" --mode notion
uv run book-ingest "<url>" --mode txt
uv run book-ingest "<url>" --mode epub --mode notion --mode txt
uv run book-ingest --search "<title>" --mode epub --mode txt
```

`--mode` and `--output-format` are aliases on `book-ingest`. The lower-level
`book-to-epub` command accepts repeated `--output-format` options.
`both` remains an alias for EPUB + TXT; spell out destinations for new commands.
The CLI requires a format choice, or infers EPUB/TXT from explicit output paths.

## Destinations and Recovery

- Local EPUB: default path `books/<author>/<title>.epub`; override with `-o`.
- TXT: default path `research/datasets/raw/<author>/<title>.txt`; override with
  `--txt-output` or `--dataset-root`. `book-ingest` upserts only that book in the
  selected dataset's `dataset_manifest.json` whenever TXT is requested.
- Notion: upload an editable draft and verify its readback. Authentication,
  cover upload, schema and checkpoint recovery are documented in
  [notion-books.md](../../../docs/notion-books.md). Shared extras use
  [fanwai-notion.md](../../../docs/fanwai-notion.md).

A combination is processed sequentially. On failure, inspect the completed
files and upload checkpoint before retrying; do not recreate an uncertain
Notion write. Use `--overwrite` only for output replacement within the request.

## Existing Training Data

Training is a purpose for TXT output, not a separate crawler. Use targeted
exports and manifest updates:

```bash
uv run book-dataset export-txt --epub books/<author>/<title>.epub
uv run book-dataset upsert --txt research/datasets/raw/<author>/<title>.txt
```

Respect the requested dataset path. Classification uses maintained Jinjiang
metadata, then the configured fallback; apply explicit user category overrides.
Bulk export requires an explicitly requested `--books-root` scope. Keep book
text, generated EPUBs, upload checkpoints and credentials out of Git.
