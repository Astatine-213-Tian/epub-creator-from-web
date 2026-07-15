# Iteration 5: Meter Audit, LoRA, And Application

Read the [Iteration 5 canonical report](../../docs/reports/07_transfer_iteration5_prompt_vs_lora.md)
first. Iteration 5
contains several related studies, so its code is grouped by research role rather
than left as one flat directory.

## Package Map

| Package | Purpose | Result |
| --- | --- | --- |
| [`meter/`](meter/README.md) | Develop and externally challenge CR-FYSM, the content-resistant style meter | CR-FYSM-v4 rejected as a transfer-selection endpoint |
| [`construct/`](construct/README.md) | Diagnose the meter with semantic, sham, and style-positive controls | Confirmed construct-validity limitations |
| [`lora/`](lora/README.md) | Build paired data and run the local Qwen3-4B MLX LoRA smoke | Infrastructure passed; production LoRA rejected |
| [`decision/`](decision/README.md) | Blind, reference-anchored comparison of prompt and LoRA candidates | Best prompt reached 6/8; formal gate missed |
| [`application/`](application/README.md) | Apply and audit the selected prompt method on Eternal Gate | Historical application evidence; production now owns execution |
| `report_style_transfer_iteration5.py` | Render retained Iteration 5 charts | Reporting only |

The `v1` through `v4` meter and construct files are successive measurement
repairs, not competing production implementations. Start a new iteration rather
than changing the conclusion of this chain after observing its results.

## Quick Verification

From `research/`:

```bash
uv run python -m experiments.iteration5.report_style_transfer_iteration5
uv run python -m unittest \
  tests.experiments.iteration5.test_transfer_lora_reference_benchmark_v2 \
  tests.experiments.iteration5.test_apply_content_plan_combined_to_translation_run \
  tests.experiments.iteration5.test_evaluate_method4_application
```

Rerunning model ratings or MLX training requires the frozen inputs and model
access documented in the canonical report. Use the parent `book-translate` CLI
for new Eternal Gate translations; this package preserves research procedures
and evidence, not the maintained production path.
