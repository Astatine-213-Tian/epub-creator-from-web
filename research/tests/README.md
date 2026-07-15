# Research Tests

`test_dataset_manifest_paths.py` covers the maintained dataset path resolver.
Experiment contract tests are grouped under `experiments/iteration3/`,
`experiments/iteration4/`, and `experiments/validation/` to mirror the source
packages. `run_research_tests.py` discovers every portable `test_*.py` module.

Run the quick gate with `uv run author-style-research verify` or the full
portable suite with `uv run author-style-research test`.
