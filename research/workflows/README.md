# Maintained Research Workflows

This is the operational surface for new research work. Use the unified command
instead of searching through historical experiment files:

```bash
uv run author-style-research --help
uv run author-style-research verify
```

| Task | Command | Implementation |
| --- | --- | --- |
| Clean and rebuild the corpus | `author-style-research corpus-build --stage all` | `audit_style_dataset.py` |
| Expand the ranked author corpus | `author-style-research corpus-expand <stage>` | `expand_jjwxc_author_dataset.py` |
| Audit masking quality | `author-style-research mask-audit` | `mask_quality_report.py` |
| Run interpretable baselines | `author-style-research authorship-baseline` | `benchmark_author_style.py` |
| Run the selected classifier benchmark | `author-style-research authorship-supervised ...` | `benchmark_author_style_supervised.py` |
| Render authorship results | `author-style-research authorship-report ...` | `report_author_classifier_retest.py` |
| Export the production style bundle | `author-style-research export-production --overwrite` | `export_production_style_transfer_assets.py` |
| Run the quick frozen gate | `author-style-research verify` | unified verifier |
| Run all portable tests | `author-style-research test` | `../tests/run_research_tests.py` |

The versioned transfer studies are under `../experiments/`. Current code is
organized for readable reproduction; frozen outputs may retain historical
source hashes from the exact run that produced them. New iterations should reuse
maintained workflow code where appropriate and bind their inputs and protocol at
preregistration.

## Code Ownership

Use this package for code that should evolve across future studies: corpus
construction, masking audits, authorship benchmarks, report rendering, portable
verification, and the production handoff. Keep method-specific prompts,
candidate generation, rating, and decision logic in the numbered experiment
that registered them.

`audit_style_dataset.py`, `benchmark_author_style.py`, and
`benchmark_author_style_supervised.py` produced retained evidence. Treat
behavioral changes to them as protocol changes and rerun the affected gates.
Prefer shared helpers when they remove real duplication without hiding the
feature definitions or split policy being evaluated.
