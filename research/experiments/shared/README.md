# Shared Experiment Infrastructure

This package contains small, policy-free helpers used by multiple numbered
experiments:

- `paths.py` exposes the research root independently of package depth.
- `hashing.py` provides canonical JSON and SHA-256 primitives.

Do not move prompt construction, sample selection, feature definitions,
evaluation thresholds, or decision rules here. Those choices define an
experiment and belong in its numbered package.
