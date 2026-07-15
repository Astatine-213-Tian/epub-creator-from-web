# Iteration 3: Constrained Rerank

Tests lighter aligned-pair edits, microcards, rule-linked microcards, and a
candidate-rerank diagnostic. `prepare.py` owns the frozen design;
`run_style_transfer_block_generation.py`, `build_candidate_rerank.py`, and
`evaluate_style_transfer_methods.py` own execution and scoring.

Canonical report: [Iteration 3 constrained rerank](../../docs/reports/05_transfer_iteration3_constrained_rerank.md).

Contract tests are under `tests/experiments/iteration3/`. Run modules from
`research/`, for example:

```bash
uv run python -m experiments.iteration3.prepare --help
```
