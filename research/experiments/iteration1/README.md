# Iteration 1: Prompt Methods

Tests neutral Chinese, generic author instructions, corpus statistics, style
cards, retrieved examples, and LLM close-reading descriptions. This package also
contains the original calibration, evaluation, leakage, provenance, and final
validation machinery used by later iterations.

## Workflow Map

| Phase | Main modules |
| --- | --- |
| Cohort and protocol | `style_transfer_research.py` |
| Style evidence and model payloads | `generate_style_close_reading.py`, `style_transfer_payloads.py` |
| Candidate generation | `run_style_transfer_generation.py` |
| Calibration and scoring | `calibrate_style_meter.py`, `evaluate_style_transfer_methods.py` |
| Independent semantic judgments | `build_style_semantic_judgments.py` |
| Promotion and final validation | `build_style_refinement_contract.py`, `select_style_transfer_promotions.py`, `prepare_style_final_validation.py` |
| Provenance and gates | `style_analysis_lock.py`, `style_experiment_provenance.py`, `verify_style_analysis_gates.py` |
| Construct audit and reports | `audit_style_meter_construct.py`, `report_style_transfer_experiment.py` |

Inspect the protocol command surface with:

```bash
uv run python -m experiments.iteration1.style_transfer_research --help
```

Canonical report: [Iteration 1 prompt methods](../../docs/reports/03_transfer_iteration1_prompt_methods.md).
