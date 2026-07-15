# Research Project Guidelines

This project contains local copyrighted corpora and reproducible author-style
experiments. Keep `datasets/` and `generated/` untracked. Do not copy source text
into tracked reports, fixtures, or production code.

Run `uv sync` and `uv run author-style-research --help` from the `research/`
directory. Maintained commands live in `workflows/`; hash-bound historical code
and maintained reproductions live in numbered packages under `experiments/`.
Preserve frozen iteration artifacts and canonical reports. New experiments
belong in a new iteration directory with an explicit manifest, fixed seed,
hash-bound inputs, a human-readable report, and an independent evaluator result.

The production project is the parent directory. Research may import shared
`src.*` primitives through the editable local dependency, but production code
must not import `research.*`.
