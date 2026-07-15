# Transfer Experiments

This directory contains the maintained code for four numbered transfer studies
and one unnumbered post-transfer validation package. Read the matching canonical
report before rerunning a numbered experiment: the report defines its data,
frozen decisions, observed results, and limitations.

Run modules from `research/` using their package path, for example:

```bash
uv run python -m experiments.iteration4.prepare --help
uv run python -m experiments.validation.meter.benchmark_content_resistant_meter --help
```

## Study Map

| Package | Research question | Canonical report |
| --- | --- | --- |
| [`iteration1/`](iteration1/README.md) | Do generic prompts, style cards, examples, retrieval, or close reading transfer the target style? | [Iteration 1 report](../docs/reports/03_transfer_iteration1_prompt_methods.md) |
| [`iteration2/`](iteration2/README.md) | Do aligned neutral-to-author pairs improve transfer? | [Iteration 2 report](../docs/reports/04_transfer_iteration2_aligned_pairs.md) |
| [`iteration3/`](iteration3/README.md) | Do constrained edits, microcards, or reranking avoid fidelity loss? | [Iteration 3 report](../docs/reports/05_transfer_iteration3_constrained_rerank.md) |
| [`iteration4/`](iteration4/README.md) | Does English-grounded full regeneration work better? | [Iteration 4 report](../docs/reports/06_transfer_iteration4_full_regeneration.md) |
| [`validation/`](validation/README.md) | Does the transfer meter survive content controls, and how did the selected prompt method behave in application? | Diagnostic evidence; not a numbered transfer iteration |
| [`shared/`](shared/README.md) | Stable path and hashing primitives with no experiment policy | Not a study |

Tests are kept separately under `../tests/experiments/iterationN/`; production
and corpus workflows are under `../workflows/`.

## Code Ownership

- Put prompts, sample allocation, generation, rating, and decision rules in the
  numbered iteration that registered them.
- Put later measurement and application audits in `validation/`; these audits
  must not be presented as an additional transfer-method iteration.
- Put only policy-free infrastructure used across iterations in `shared/`.
- Put evolving corpus construction, classifier benchmarks, verification, and
  production export commands in `../workflows/`.
- Start a new numbered iteration when changing an observed transfer method.

Similar-looking runners are not automatically duplicates. Iterations 2-4 encode
different prompt contracts and evaluation rules, so merging them would obscure
the method actually tested. Small stable primitives are shared; protocol logic
remains local and readable.

## Evidence And Reproduction

Generated preregistrations, model outputs, ratings, charts, and independent
evaluations remain under `../generated/`. Some retained snapshots record the old
`scripts/...` module layout or hashes of an earlier source file. Those fields
describe the historical run; they are not current import paths.

The maintained modules reproduce the experiment procedure and produce a
reasonable current artifact format. Nondeterministic model results and obsolete
source-file hashes are not expected to match byte for byte. Immutable input and
artifact content hashes are still checked where they define the scientific
sample or observed evidence.

## Verification

```bash
uv run author-style-research verify
uv run author-style-research test
```

The full command runs all portable tests. One native-Codex command-template
audit is reported as environment-bound because it requires the installation
captured by the original preregistration.
