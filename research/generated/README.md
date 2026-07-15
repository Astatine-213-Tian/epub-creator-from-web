# Generated Research Evidence

This directory is local and ignored by Git. It contains reproducible derived
corpora plus frozen model outputs and evaluations that cannot be recreated
bit-for-bit from nondeterministic model calls.

```text
corpus_acquisition/             retained ranking, final selection, and compact crawl history
style_research/corpus/          canonical cleaned corpus and book splits
style_research/benchmarks/      authorship and masking evaluations
style_research/style_transfer_experiments/
                                frozen transfer protocols, outputs, ratings,
                                charts, and independent audits
```

Retention policy:

- Keep manifests, preregistrations, prompts, schemas, model outputs, ratings,
  final tables/graphs, and independent evaluations referenced by a report.
- Remove temporary downloads, browser traces, caches, empty debug directories,
  duplicate cleaned-corpus builds, and abandoned single-author-profile outputs.
- Do not rewrite retained evidence merely to match a current code path. Frozen
  records may contain historical `scripts/...` paths or old source-file hashes;
  those values describe the completed run and are not active import targets.
- Current reproduction output may use a clearer layout and need not match a
  nondeterministic snapshot byte for byte. Preserve immutable sample, input,
  model-output, rating, and decision evidence referenced by canonical reports.

Notable external-challenge evidence lives at
`style_research/style_transfer_experiments/iterations/content_resistant_v1/`
`cr_fysm_v4/external_confirmation_v2/`. It binds
`datasets/external_author_challenge_v2/` and records a failed preregistered gate;
it is retained as negative evidence, not as production input.
