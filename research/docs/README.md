# Author Style Research Index

This directory is the maintained written record for the Eternal Gate
author-style research. Start with the research plan, then read the numbered
reports in order. Generated JSON, model outputs, and private benchmark packets
remain under `../generated/`; reports contain the human-readable procedure,
results, limitations, and independent evaluations.

## Current Decision

The production pipeline uses `content_plan_combined_full_regeneration` because it
showed the strongest relative behavior among the retained prompt methods and was
preferred in direct Eternal Gate review. Iteration 4 did not pass its formal
academic gate, so this remains an engineering choice rather than a claim of
validated transfer efficacy.

## Research Plan

- [Research plan](research_plan.md): questions, data roles, leakage controls,
  evaluation design, advancement gates, and production handoff rules.

## Reports

| Order | Report | Main result |
| ---: | --- | --- |
| 1 | [Dataset curation and masking](reports/01_dataset_curation_and_masking.md) | 50-author corpus, strict book-level splits, train-fitted global masking, and a book-local diagnostic view |
| 2 | [Authorship style meter](reports/02_authorship_style_meter.md) | mask-stripped character 2-4 gram TF-IDF plus unweighted SGD hinge; 89.8% test accuracy and 87.9% balanced accuracy |
| 3 | [Transfer Iteration 1](reports/03_transfer_iteration1_prompt_methods.md) | generic cards, retrieval, and close-reading prompt families; NO-GO |
| 4 | [Transfer Iteration 2](reports/04_transfer_iteration2_aligned_pairs.md) | aligned neutral-to-author pairs; NO-GO |
| 5 | [Transfer Iteration 3](reports/05_transfer_iteration3_constrained_rerank.md) | constrained edits, microcards, and reranking; NO-GO |
| 6 | [Transfer Iteration 4](reports/06_transfer_iteration4_full_regeneration.md) | full regeneration and content-plan methods; formal development NO-GO |

Each transfer iteration has exactly one canonical report. Earlier drafts and the
abandoned single-author profile are not part of this report sequence.

Post-transfer meter and application diagnostics are documented with their code
under [`../experiments/validation/`](../experiments/validation/README.md). They
are supporting evidence, not a fifth transfer iteration.

## Literature Notes

- [01: Chinese stylometry](literature/01_stylometry_chinese_author_profile.md)
- [02: Corpus stylistics and evidence cards](literature/02_corpus_stylistics_evidence_cards.md)
- [03: Back-translation and style transfer](literature/03_textless_back_translation_and_style_transfer.md)
- [04: Retrieval and evaluation](literature/04_retrieval_and_evaluation.md)
- [05: Historical single-author profile methodology](literature/05_author_profile_methodology.md)
- [06: Authorship benchmarks and LLM methods](literature/06_author_style_benchmark_and_llm_methods.md)
- [07: Content masking and content control](literature/07_content_masking_and_content_control.md)

## Reproduction

Commands in the reports assume the current working directory is `research/`:

```bash
cd research
uv sync
make benchmark-authorship
```

`make benchmark-authorship` first rebuilds every clean and masked view from
`datasets/raw/`, then runs the canonical classifier configuration and report. The
classifier rejects chunks that do not carry the current punctuation-normalization
version, so an older generated corpus cannot be used accidentally.

Use [the project README](../README.md) for the short workflow,
[`workflows/README.md`](../workflows/README.md) for maintained commands, and
[`experiments/README.md`](../experiments/README.md) for the numbered experiment map.
