# LoRA Smoke Study

`prepare_lora_paired_reconstruction_smoke_v1.py` builds the clean paired corpus,
`prepare_lora_mlx_smoke_run_v1.py` writes the MLX configuration, and
`run_lora_mlx_smoke_inference_v1.py` records base and adapter inference.

This was an infrastructure smoke on a small 4B model, not an adequately powered
training study. The adapter failed the semantic and complete-success endpoints;
the canonical report records a production NO-GO.
