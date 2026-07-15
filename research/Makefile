.PHONY: help sync verify test rebuild-corpus audit-masks audit-mask-artifacts benchmark-authorship export-production

help:
	uv run author-style-research --help

sync:
	uv sync

verify:
	uv run author-style-research verify

test:
	uv run author-style-research test

rebuild-corpus:
	uv run author-style-research corpus-build --stage all

audit-masks:
	uv run author-style-research mask-audit

audit-mask-artifacts:
	uv run author-style-research mask-artifact-audit --jobs -1

benchmark-authorship: rebuild-corpus
	uv run author-style-research authorship-supervised --jobs -1
	uv run author-style-research authorship-report \
		--evaluation-file generated/style_research/benchmarks/author_style_supervised_50authors_cleaned/evaluator_review.md

export-production:
	uv run author-style-research export-production --overwrite
