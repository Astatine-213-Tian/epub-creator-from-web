# Meter Construct Validation

These modules test what CR-FYSM-v4 measures using English semantic validation,
style-positive controls, sham controls, blinded ratings, and amendment records.
`construct_v2_protocol.py` defines the final protocol helpers; `prepare_*`,
`run_*`, `close_*`, and `analyze_*` follow the execution order.

Prompt and schema assets are under `assets/`. Amendment files are retained
because they document bounded repairs to failed generation or adjudication, not
new transfer methods.
