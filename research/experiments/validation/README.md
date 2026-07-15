# Post-Transfer Validation

This package contains diagnostic work performed after the four numbered
transfer experiments. It does not define a fifth transfer method. The code is
grouped by research role so a new researcher can distinguish measurement audits
from downstream application checks.

## Package Map

| Package | Purpose | Result |
| --- | --- | --- |
| [`meter/`](meter/README.md) | Develop and externally challenge CR-FYSM, the content-resistant style meter | CR-FYSM-v4 rejected as a transfer-selection endpoint |
| [`construct/`](construct/README.md) | Diagnose the meter with semantic, sham, and style-positive controls | Confirmed construct-validity limitations |
| [`application/`](application/README.md) | Apply and audit the selected prompt method on Eternal Gate | Historical application evidence; production now owns execution |

The `v1` through `v4` meter and construct files are successive measurement
repairs, not competing production implementations. Their negative result is
retained because it constrains how classifier scores may be interpreted.

## Quick Verification

From `research/`:

```bash
uv run python -m unittest \
  tests.experiments.validation.test_cr_fysm_v4_construct_v2 \
  tests.experiments.validation.test_apply_content_plan_combined_to_translation_run \
  tests.experiments.validation.test_evaluate_method4_application
```

Rerunning model ratings requires the frozen inputs and model access documented
by each preregistration snapshot. Use the parent `book-translate` CLI for new
Eternal Gate translations; this package preserves research procedures and
evidence, not the maintained production path.
