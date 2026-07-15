# Author Style Research Index

This directory is the maintained written record for the Eternal Gate
author-style research. Start with the research plan, then read the numbered
reports in order. Generated JSON, model outputs, and private benchmark packets
remain under `../generated/`; reports contain the human-readable procedure,
results, limitations, and independent evaluations.

## Current Decision

The production pipeline uses `content_plan_combined_full_regeneration` because it
was the strongest prompt method in the final developmental benchmark and was
preferred in direct Eternal Gate review. It achieved 6/8 joint source successes,
below the preregistered 7/8 gate, so this is an engineering choice rather than a
claim that the academic promotion criterion passed.

The 4B LoRA run proved local training and inference plumbing only. It achieved
0/3 full book successes and is not a production candidate. A serious LoRA study
requires a new, clean paired corpus and a separate preregistration.

## Research Plan

- [Research plan](research_plan.md): questions, data roles, leakage controls,
  evaluation design, advancement gates, and production handoff rules.

## Reports

| Order | Report | Main result |
| ---: | --- | --- |
| 1 | [Dataset curation and masking](reports/01_dataset_curation_and_masking.md) | 50-author corpus, strict book-level splits, clean and `entity_masked_v3` views |
| 2 | [Authorship style meter](reports/02_authorship_style_meter.md) | exact character 2-4 gram TF-IDF plus SGD hinge; 88.2% masked test accuracy |
| 3 | [Transfer Iteration 1](reports/03_transfer_iteration1_prompt_methods.md) | generic cards, retrieval, and close-reading prompt families; NO-GO |
| 4 | [Transfer Iteration 2](reports/04_transfer_iteration2_aligned_pairs.md) | aligned neutral-to-author pairs; NO-GO |
| 5 | [Transfer Iteration 3](reports/05_transfer_iteration3_constrained_rerank.md) | constrained edits, microcards, and reranking; NO-GO |
| 6 | [Transfer Iteration 4](reports/06_transfer_iteration4_full_regeneration.md) | full regeneration and content-plan methods; formal development NO-GO |
| 7 | [Transfer Iteration 5](reports/07_transfer_iteration5_prompt_vs_lora.md) | reference-anchored prompt comparison and 4B LoRA smoke; prompt 6/8, LoRA 0/3 |

Each transfer iteration has exactly one canonical report. Earlier drafts and the
abandoned single-author profile are not part of this report sequence.

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
make verify
```

Use [the project README](../README.md) for the short workflow,
[`workflows/README.md`](../workflows/README.md) for maintained commands, and
[`experiments/README.md`](../experiments/README.md) for the numbered experiment map.
