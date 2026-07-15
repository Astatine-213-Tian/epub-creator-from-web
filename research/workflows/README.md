# Maintained Research Workflows

This is the operational surface for new research work. Use the unified command
instead of searching through historical experiment files:

```bash
uv run author-style-research --help
uv run author-style-research verify
```

| Task | Command | Implementation |
| --- | --- | --- |
| Clean and rebuild every derived corpus view | `author-style-research corpus-build --stage all` | `audit_style_dataset.py` |
| Rebuild chunks from an existing clean corpus and mask plan | `author-style-research corpus-build --stage chunks` | `audit_style_dataset.py` |
| Normalize manifest paths without rebuilding text | `author-style-research corpus-build --stage paths` | `audit_style_dataset.py` |
| Expand the ranked author corpus | `author-style-research corpus-expand <stage>` | `expand_jjwxc_author_dataset.py` |
| Audit masking quality | `author-style-research mask-audit` | `mask_quality_report.py` |
| Test mask-token artifacts | `author-style-research mask-artifact-audit` | `audit_mask_artifacts.py` |
| Run the selected classifier benchmark | `author-style-research authorship-supervised` | `benchmark_author_style_supervised.py` |
| Render authorship results | `author-style-research authorship-report ...` | `report_author_style_meter.py` |
| Build interpretable author-profile figures | `author-style-research interpretable-profiles` | `analyze_interpretable_author_profiles.py` |
| Export the production style bundle | `author-style-research export-production --overwrite` | `export_production_style_transfer_assets.py` |
| Rebuild current data and run quick checks | `author-style-research verify` | unified verifier |
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

`audit_style_dataset.py` and `benchmark_author_style_supervised.py` produce the
current retained evidence. `benchmark_author_style.py` contains exploratory
interpretable feature implementations but is not a maintained CLI entry point. Treat
behavioral changes to them as protocol changes and rerun the affected gates.
Prefer shared helpers when they remove real duplication without hiding the
feature definitions or split policy being evaluated.

The maintained benchmark consumes only the current punctuation-normalized,
cross-book-decontaminated corpus and reports three aligned views: clean text,
training-fit global masking, and training-fit global plus label-blind per-book
masking. Historical experiment scripts may describe an older frozen protocol;
they are not an alternative way to rebuild the current meter.
