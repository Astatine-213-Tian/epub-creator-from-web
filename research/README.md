# Author Style Transfer Research

This nested project owns the dataset-first research for Chinese authorship
identification and English-to-Chinese author-style transfer. The repository root
owns crawling, dataset ingestion, production translation, and EPUB construction.

## Layout

```text
datasets/       local raw text, derived chunk views, and challenge corpora
docs/           research plan, numbered reports, and literature notes
generated/      cleaned book copies, corpus metadata, and experimental evidence
workflows/      maintained corpus, benchmark, verification, and handoff commands
experiments/    numbered transfer studies plus small shared infrastructure
tests/          portable workflow and experiment contract tests
```

Start with [`docs/README.md`](docs/README.md). It states the current decision and
links the dataset report, authorship-meter report, four numbered transfer
iterations, post-transfer validation, and literature notes. Use
[`workflows/README.md`](workflows/README.md) for current commands and
[`experiments/README.md`](experiments/README.md) to audit an older iteration or
the retained post-transfer validation work.

## Environment

Run research commands from this directory. Canonical reports use paths relative
to this project root, so this working-directory rule also keeps their recorded
reproduction commands valid:

```bash
cd research
uv sync
uv run author-style-research verify
```

`verify` rebuilds the current clean and masked corpus before running the portable
checks. Use `make benchmark-authorship` for the complete normalized-corpus rebuild,
canonical classifier rerun, and report regeneration.

Superseded prototypes are removed once their conclusions are captured in a
canonical report. Frozen generated evidence may retain an older source path or
source hash; current reproduction code is organized for readability and is not
required to recreate nondeterministic model output byte for byte.

## Production Handoff

The production translator consumes one minimized, hash-locked bundle. Rebuild it
from the frozen research asset with:

```bash
uv run author-style-research export-production --overwrite
```

The exporter writes only
`../generated/author_styles/feitianyexiang/content_plan_combined.v1.json` in the
production project. Corpus text and experiment outputs remain here.

## Maintenance Boundary

The parent project owns crawling, targeted dataset ingest, translation, semantic
QA, and EPUB generation. Research may import its shared `src.*` primitives. The
parent production package must never import `research.*` or depend on the
scientific Python stack used here.
