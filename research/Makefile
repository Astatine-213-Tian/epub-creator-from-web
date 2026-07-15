.PHONY: help sync verify test rebuild-corpus audit-masks benchmark-authorship export-production

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

benchmark-authorship:
	uv run author-style-research authorship-supervised \
		--views clean,entity_masked_v3 \
		--masked-view entity_masked_v3 \
		--methods char_ngrams \
		--classifiers sgd_hinge,sgd_hinge_unbalanced \
		--max-char-features 80000 \
		--char-min-df 20 \
		--output-dir generated/style_research/benchmarks/author_style_supervised_50authors_iter3_exact_hinge_mindf20 \
		--jobs -1 --max-iter 3000

export-production:
	uv run author-style-research export-production --overwrite
