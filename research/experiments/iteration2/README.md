# Iteration 2: Aligned Pairs

Tests aligned neutral-Chinese to target-author-Chinese examples, with and without
style cards or an edit plan. `prepare.py` freezes the cohort and methods,
`run_style_transfer_generation.py` generates candidates,
`evaluate_style_transfer_methods.py` scores them, and
`analyze_family_rerank.py` runs the registered diagnostic.

Canonical report: [Iteration 2 aligned pairs](../../docs/reports/04_transfer_iteration2_aligned_pairs.md).

Run modules from `research/`, for example:

```bash
uv run python -m experiments.iteration2.prepare --help
```
