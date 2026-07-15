# Eternal Gate Style-Transfer Research: Iteration 5

## Executive Status

> **Academic decision: NO-GO for both formal prompt promotion and production
> LoRA. Operational decision: retain `content_plan_combined_full_regeneration`
> as the production prompt method because it was the strongest tested method and
> subsequently performed well in direct Eternal Gate review.**

Iteration 5 compared the frozen Iteration 4 prompt methods with a local 4B LoRA
in a reference-anchored structural benchmark. The best prompt method achieved
6/8 joint source successes (75.0%), below the registered 7/8 threshold. The LoRA
smoke achieved 0/3 full book successes. A separate evaluator confirmed the
NO-GO interpretation.

## Research Question

1. Does any frozen full-regeneration prompt method preserve meaning while
   reconstructing passage-level target-author structure on unseen books?
2. Does a small paired-reconstruction LoRA show enough contract and semantic
   reliability to justify production training?
3. Which result, if any, is strong enough to guide the Eternal Gate production
   style pass?

This was a developmental comparison. It was not a confirmatory efficacy study,
a human-equivalent imitation claim, or a generic authorship-identification test.

## Data

### Prompt benchmark

The prompt arm used eight book-disjoint target-author development sources and 22
passage blocks drawn from `天宝伏妖录`, `山有木兮`, `清平梦华录`, `相见欢`,
`万物风华录`, `骑士之歌`, `图灵密码`, and `乱世为王`. Candidate generation
was frozen before the reference-anchored rerating.

The style panel saw a clean Chinese reference and anonymous candidates but no
English. The semantic panel saw the English semantic source and anonymous
candidates but no Chinese reference. This separation prevents a style rater from
rewarding English alignment and a semantic rater from copying the reference.

### LoRA smoke

The paired corpus contained 18 source passages from 18 disjoint books: 12 fit,
3 validation, and 3 internal test books. Ten other books were excluded because
their neutral and target placeholder multisets did not match. Deterministic
blocking produced 35 training, 9 validation, and 9 internal-test examples; the
12 fit passages contained only 5,877 unique target Han characters. Augmentation
did not add unique text.

The model was `Qwen/Qwen3-4B-MLX-4bit` with MLX-LM 0.31.3, eight LoRA layers,
rank 8, scale 20, learning rate `1e-5`, assistant-only loss, and 120 iterations.
This sample size authorized an infrastructure smoke only.

## Methods Tested

| Method | Meaning |
| --- | --- |
| `neutral_only` | unchanged neutral Chinese control |
| `generic_full_regeneration` | regeneration from a generic target-author instruction without structured evidence |
| `aligned_pairs_full_regeneration` | regeneration using retrieved English/neutral/target-author aligned examples |
| `style_definition_examples_full_regeneration` | multidimensional style definition plus masked examples |
| `aligned_pairs_style_definition_full_regeneration` | aligned pairs combined with the style definition |
| `content_plan_combined_full_regeneration` | English-grounded paragraph plan plus aligned pairs, style definition, and masked examples |
| `independent_candidate_selector` | blind selection among generated candidates |
| `frozen_base_4b` | unadapted 4B base-model control |
| `lora_smoke_4b` | the locally trained paired-reconstruction LoRA |

## Evaluation

Three style-rater calls scored passage structure against clean Chinese
references. Two semantic-rater calls scored fidelity against English. All calls
used `gpt-5.5` with high reasoning, so the panel is not model-family independent.

A source succeeded only when every block satisfied the output contract, at least
80% of its blocks passed both semantic validators, no block had a majority
high-severity error, structural score improved over neutral in both mean and
paired-majority terms, and median naturalness was at least 3/5. The prompt gate
was at least 7 successful sources out of 8. The LoRA result was descriptive and
required all three books to be contract-valid before style efficacy could be
interpreted.

All three style raters passed all four positive-versus-decoy calibration blocks,
each with a mean separation of +3.750. The uncertainty unit was the source or
book, not individual ratings.

## Prompt Results

![Reference-anchored source success](../../generated/style_research/style_transfer_experiments/iterations/paired_reconstruction_decision_v1/blind_benchmark_v2_reference_anchored/method_success.svg)

| Method | Joint success | Contract | Semantic | Structure over neutral | Naturalness | Mean delta, cluster 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `content_plan_combined_full_regeneration` | **6/8** | 8/8 | 7/8 | 6/8 | 8/8 | +0.556 [+0.306, +0.778] |
| `aligned_pairs_style_definition_full_regeneration` | 5/8 | 8/8 | 6/8 | 6/8 | 8/8 | +0.556 [+0.264, +0.833] |
| `style_definition_examples_full_regeneration` | 5/8 | 8/8 | 6/8 | 6/8 | 8/8 | +0.514 [+0.250, +0.722] |
| `aligned_pairs_full_regeneration` | 4/8 | 8/8 | 5/8 | 6/8 | 8/8 | +0.569 [+0.389, +0.750] |
| `independent_candidate_selector` | 4/8 | 8/8 | 7/8 | 4/8 | 8/8 | +0.347 [+0.083, +0.597] |
| `generic_full_regeneration` | 2/8 | 8/8 | 5/8 | 3/8 | 8/8 | +0.090 [-0.201, +0.361] |
| `neutral_only` | 0/8 | 8/8 | 7/8 | 0/8 | 8/8 | +0.000 [+0.000, +0.000] |

`content_plan_combined_full_regeneration` was the strongest joint method but
missed the formal gate by one source. `山有木兮` failed the structural endpoint;
`万物风华录` failed semantic and structural endpoints. No prompt method was
formally promotable.

## LoRA Results

![LoRA smoke loss](../../generated/style_research/style_transfer_experiments/iterations/paired_reconstruction_decision_v1/iteration5_report_assets/lora_loss.svg)

Training completed in about five minutes without an out-of-memory error. Train
loss fell from 1.967 to 0.621; best validation loss was 1.064; test loss was
0.786 with perplexity 2.194; peak memory was 4.899 GB. These values establish
that the local MLX training path works, not that transfer quality is adequate.

| Method | Full success | Contract-complete | Semantic | Structure over neutral | Mean delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `lora_smoke_4b` | 0/3 | 1/3 | 0/3 | 2/3 | +0.685 |
| `frozen_base_4b` | 0/3 | 0/3 | 0/3 | 0/3 | +0.000 |
| `neutral_only` | 0/3 | 3/3 | 3/3 | 0/3 | +0.000 |

The adapter learned some structural behavior and output formatting, but failed
the semantic and full-contract endpoints. It is not evidence for a production
LoRA.

## Independent Evaluation

A separate preregistration auditor first issued a conditional GO, required the
LoRA failure status and v1/v2 denominator assertions to be made explicit, then
issued GO to rate after both changes. A different post-outcome evaluator found
the locked NO-GO correct.

The evaluator highlighted two limits: pairwise style-rater Spearman correlations
were only 0.265, 0.530, and 0.418, and all automated raters used one model family.
Semantic binary-pass agreement was 0.918. Independent Chinese human or
different-family confirmation remains necessary before any efficacy claim.

## Decision

Do not train or deploy a production LoRA from this corpus. A serious LoRA study
requires 1,500-2,000 accepted pairs, at least 400,000 target tokens, multiple
seeds, a new preregistration, and independent confirmation.

For production, use `content_plan_combined_full_regeneration` with deterministic
fidelity gates and independent English-grounded QA. This operational selection
reflects its rank in this benchmark and direct reader preference on Eternal Gate;
it does not change the formal 6/8 NO-GO result.

## Reproduction

Run from `research/`. Existing rating files can be revalidated with `--resume`;
new rating calls require model access.

```bash
uv run python -m experiments.iteration5.lora.prepare_lora_paired_reconstruction_smoke_v1
uv run python -m experiments.iteration5.lora.prepare_lora_mlx_smoke_run_v1
uvx --from mlx-lm==0.31.3 mlx_lm.lora \
  --config generated/style_research/style_transfer_experiments/iterations/lora_paired_reconstruction_v1/runs/qwen3_4b_mlx_smoke_v1/train_config.yaml

uv run python -m experiments.iteration5.decision.prepare_transfer_lora_reference_benchmark_v2
uv run python -m experiments.iteration5.decision.run_transfer_lora_blind_ratings_v1 \
  style --benchmark v2 --jobs 3 --resume
uv run python -m experiments.iteration5.decision.run_transfer_lora_blind_ratings_v1 \
  semantic --benchmark v2 --jobs 2 --resume
uv run python -m experiments.iteration5.report_style_transfer_iteration5
```

The frozen analyzer refuses to overwrite existing results. Primary evidence is
under `generated/style_research/style_transfer_experiments/iterations/` in
`lora_paired_reconstruction_v1/` and
`paired_reconstruction_decision_v1/blind_benchmark_v2_reference_anchored/`.
