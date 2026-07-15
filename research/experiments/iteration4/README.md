# Iteration 4: Full Regeneration

Tests English-grounded full regeneration with aligned pairs, a multidimensional
style definition, masked examples, a paragraph content plan, and an independent
candidate selector. `prepare.py` builds the frozen package;
`rehearse_downstream_pipeline.py` validates contracts before model calls; the
remaining builders, runners, evaluators, and tests preserve the registered
execution path.

Canonical report: [Iteration 4 full regeneration](../../docs/reports/06_transfer_iteration4_full_regeneration.md).

Contract tests are under `tests/experiments/iteration4/`. Run modules from
`research/`, for example:

```bash
uv run python -m experiments.iteration4.prepare --help
```
