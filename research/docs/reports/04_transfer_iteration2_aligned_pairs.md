# Eternal Gate Style-Transfer Research: Iteration 2

Generated from frozen artifacts at `2026-07-13T00:19:37+00:00`.

## Executive Result

Iteration 2 finished source construction, neutral translation, three aligned-pair style-transfer arms, deterministic scoring, and an independent outcome audit. All three style methods produced positive mean target-author margin lift in both benchmark arms, but **none passed the preregistered joint gate** because own-author hard-fidelity failure rates were 25.0%-29.2%, above the 10% limit. Confirmation was therefore not opened.

The reused binary style-success threshold is also invalid for this iteration because its allocation and evaluation-protocol hashes still bind iteration 1. The report therefore does not claim a threshold-based accuracy or an 80% result. This threshold defect cannot be repaired post hoc after viewing outputs; it must be corrected before iteration 3.

| Stage | Data | Result | Status |
| --- | --- | --- | --- |
| Aligned-pair evidence | 29 excerpts from 29 target-author train books | 29 English -> neutral -> original-target pairs | complete |
| Fresh screen | 24 own-author + 12 cross-author rows | Zero overlap with iteration-1 screen | complete |
| Style generation | 3 methods x 36 rows | 108/108 final outputs valid | complete |
| Deterministic evaluation | Frozen exact character n-gram style meter plus fidelity/copy gates | Positive relative style lift; no copy failures; every method fails joint promotion | complete |
| Independent evaluator | Separate `gpt-5.5` close reading and audit | No base method qualifies for confirmation; Independent English-grounded close reading covered 13 method-output cells. | complete |
| Confirmation | 116 untouched rows | Not generated because no base method qualified | locked |

## Research Question

Do retrieved aligned neutral-to-target demonstrations teach the target author's transformations more effectively than iteration-1 unaligned examples while preserving the English source meaning and Chinese surface structure?

## Experimental Design

### Data roles

| Role | Construction | Exposure |
| --- | --- | --- |
| In-context pair pool | One masked excerpt from each of 29 非天夜翔 train books; scenes: action conflict: 9, dialogue: 9, internal reflection: 7, interpersonal care: 1, travel transition: 3 | Available only as retrieved style evidence; no development or test books |
| Pair English source | `gpt-5.4` reconstruction from each masked Chinese train excerpt | Used to create production-direction neutral Chinese |
| Pair neutral Chinese | `gpt-5.4` using the same `eternal_gate_pass_1` neutral prompt contract | Retrieval query and neutral side of each aligned demonstration |
| Pair target Chinese | Original `entity_masked_v2` train excerpt | Target side of an in-context pair; never an evaluation target |
| Iteration-2 screen | 36 fresh rows: 3 per target-author proxy book and 1 per comparison author | Used once for method screening |
| Confirmation | 116 rows: 80 own-author and 36 cross-author | Unopened by style generators |
| Final validation | Four target-author books reserved in the corpus protocol | Unopened; no row IDs created |

This is **in-context style transfer, not model fine-tuning**. The phrase `training pairs` means prompt evidence retrieved from train-split books. Model weights were not updated.

### Why English and neutral Chinese were generated

Production starts from the English Eternal Gate translation. A Chinese-to-Chinese neutralization benchmark would give the transfer method an easier and different input distribution. Iteration 2 therefore reconstructs English semantic sources and applies the same frozen neutral-translation prompt intended for production before any style rewrite.

### Source QA and repair

A repair round regenerated only English sources still rejected by semantic QA. It did not edit style outputs, did not repeatedly change already approved rows, and did not use style scores. The pair pool required three repair rounds; the fresh screen required six.

| Dataset | QA point | Rows checked | Still unresolved |
| --- | --- | ---: | ---: |
| Aligned-pair pool | Initial QA | 29 | 15 |
| Aligned-pair pool | Repair QA round 1 | 15 | 6 |
| Aligned-pair pool | Repair QA round 2 | 6 | 3 |
| Aligned-pair pool | Repair QA round 3 | 3 | 0 |
| Fresh screen | Initial QA | 36 | 23 |
| Fresh screen | Repair QA round 1 | 23 | 11 |
| Fresh screen | Repair QA round 2 | 11 | 8 |
| Fresh screen | Repair QA round 3 | 8 | 2 |
| Fresh screen | Repair QA round 4 | 2 | 1 |
| Fresh screen | Repair QA round 5 | 1 | 1 |
| Fresh screen | Repair QA round 6 | 1 | 0 |

### Models and frozen execution

| Component | Model | Reasoning | Notes |
| --- | --- | --- | --- |
| English reconstruction, QA, repair, neutral Chinese | `gpt-5.4` | high | Same neutral prompt contract as planned production pass 1 |
| All style-transfer arms | `gpt-5.5` | high | One model across arms; no model mixing |
| Independent methodology and outcome reviews | `gpt-5.5` | high | Separate Codex sessions; no style generation |

`gpt-5.6` was tried first and returned `unsupported_with_chatgpt_account`. `gpt-5.5` returned `available` and was used for style transfer.

The active pre-generation analysis lock is `protocols/pre_style_analysis_lock.v1.e97a085379b0cefe15bd60bd7d9d539c73f23b283e8c2d8c19aa805e209a7d4f.json`. A separate methodology reviewer issued **GO** only after stale source bindings, sample hashes, evaluator geometry, and promotion wording were corrected.

### What the pair rebuild changed

The audit-triggered `prepare.py` rebuild did **not** call an LLM and did not replace the 29 English, neutral, or target texts. It deterministically rewrote manifests, role labels, paths, hashes, method assets, source bindings, and the active lock so every artifact referenced the iteration-2 allocation. Rebuilding those records was required because the first frozen package still contained iteration-1 paths and geometry.

## Methods Tested

| Method | Meaning | Evidence visible to `gpt-5.5` | Hypothesis |
| --- | --- | --- | --- |
| `neutral_only` | No rewrite; score the neutral Chinese unchanged as the paired control. | `english_semantic_source`, `neutral_zh` | Negative control for paired margin lift. |
| `aligned_pairs_only` | Retrieve three semantically similar neutral-to-target pairs, with at most one pair from each source book. | `english_semantic_source`, `neutral_zh`, `aligned_target_pairs` | Aligned neutral-to-target examples teach transformations that unaligned prose examples do not. |
| `aligned_pairs_plus_cards` | Use the same three aligned pairs plus compact corpus style cards only when cards agree with pair evidence. | `english_semantic_source`, `neutral_zh`, `aligned_target_pairs`, `global_style_cards` | Corpus cards improve pair generalization when both evidence sources agree. |
| `aligned_pairs_edit_plan` | Internally infer recurring edits in clause order, sentence boundaries, dialogue timing, function words, and reaction beats; apply only patterns supported by at least two pairs. | `english_semantic_source`, `neutral_zh`, `aligned_target_pairs`, `edit_plan_contract` | An explicit repeated-edit inference contract improves application of paired demonstrations. |

Every method received byte-identical English and neutral Chinese. The style runner could not read original evaluation Chinese, author/book labels, target-derived strata, or scores. Retrieval used only target-author train books and capped evidence at one pair per book.

## Generation Reliability

All arms ultimately produced 36 schema-valid outputs. Long samples with 46-53 paragraphs caused paragraph-ID omissions, duplicates, and occasional timeouts, so the request count is materially larger than the final output count.

| Method | Final outputs | Total requests | Failed attempts | Samples needing retry | Timeouts |
| --- | ---: | ---: | ---: | ---: | ---: |
| `aligned_pairs_only` | 36/36 | 64 | 28 | 7 | 3 |
| `aligned_pairs_plus_cards` | 36/36 | 42 | 6 | 4 | 0 |
| `aligned_pairs_edit_plan` | 36/36 | 50 | 14 | 8 | 1 |

This is an execution-quality result, not a style result. Iteration 3 should freeze a paragraph-batched long-form execution method before generation so output-schema retries do not depend on repeated whole-chunk calls.

## Deterministic Results

The frozen Stage-1 style meter is the class-balanced exact 2-4 character n-gram SGD hinge classifier on `entity_masked_v3`. `Mean paired lift` is candidate target-author decision margin minus the same row's neutral margin. Positive lift means movement toward the target-author side of this proxy; it is not a probability or human style score.

![Mean paired margin lift](../../generated/style_research/style_transfer_experiments/iterations/aligned_pairs_v1/evaluations/screening_v1_initial/charts/mean_paired_lift.svg)

| Method | Arm | Mean target margin | Mean paired lift | Cluster bootstrap 95% CI | Positive rows | Target top-class share | Fidelity failures | Copy failures |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `neutral_only` | Own-author reconstruction | -0.397 | +0.000 | [+0.000, +0.000] | 0/24 | 91.7% | 1/24 (4.2%) | 0/24 |
| `neutral_only` | Cross-author transfer | -1.322 | +0.000 | [+0.000, +0.000] | 0/12 | 0.0% | 1/12 (8.3%) | 0/12 |
| `aligned_pairs_only` | Own-author reconstruction | -0.224 | +0.173 | [+0.101, +0.250] | 19/24 | 91.7% | 7/24 (29.2%) | 0/24 |
| `aligned_pairs_only` | Cross-author transfer | -1.187 | +0.135 | [+0.005, +0.320] | 8/12 | 0.0% | 0/12 (0.0%) | 0/12 |
| `aligned_pairs_plus_cards` | Own-author reconstruction | -0.203 | +0.194 | [+0.132, +0.264] | 21/24 | 91.7% | 7/24 (29.2%) | 0/24 |
| `aligned_pairs_plus_cards` | Cross-author transfer | -1.175 | +0.147 | [+0.037, +0.281] | 9/12 | 0.0% | 4/12 (33.3%) | 0/12 |
| `aligned_pairs_edit_plan` | Own-author reconstruction | -0.247 | +0.149 | [+0.038, +0.255] | 19/24 | 87.5% | 6/24 (25.0%) | 0/24 |
| `aligned_pairs_edit_plan` | Cross-author transfer | -1.183 | +0.139 | [+0.005, +0.315] | 7/12 | 8.3% | 1/12 (8.3%) | 0/12 |

### Threshold status

The threshold artifact contains the historical value `0.218174636`, but its iteration-2 status is **`invalid`** with errors `threshold_allocation_sha256_mismatch, threshold_protocol_sha256_mismatch`. Binary deterministic style success is therefore undefined in this screen. The generated `deterministic_style_success.svg` is retained for artifact completeness but must not be interpreted as a measured 0% or 80% result.

The error is a binding failure, not evidence that the numeric threshold itself is scientifically wrong. Rebinding or recalibrating it now would be post-hoc because the method outputs are visible. Iteration 3 must generate a fresh threshold artifact bound to its allocation and protocol before style generation.

### Promotion gate

Screening required positive mean lift in both arms, no copy failure, and at most 10% hard-fidelity failures in each arm. The 24-row own-author arm permits at most two failures; the 12-row cross-author arm permits at most one.

| Method | Lift positive in both arms | Own fidelity <=10% | Cross fidelity <=10% | No copy failures | Joint result |
| --- | --- | --- | --- | --- | --- |
| `aligned_pairs_only` | pass | fail | pass | pass | **does not qualify** |
| `aligned_pairs_plus_cards` | pass | fail | fail | pass | **does not qualify** |
| `aligned_pairs_edit_plan` | pass | fail | pass | pass | **does not qualify** |

### Method-level interpretation

#### `aligned_pairs_only`

Best evidence of stable lift: own-author +0.173 and cross-author +0.135, with both cluster bootstrap intervals above zero. It also has the strongest single cross-author gain, but 7/24 own-author rows fail the frozen fidelity screen.

Recorded paragraph-level surface flags: `dialogue_turn_surface_mismatch` (25), `latin_token_mismatch` (6).

![Aligned pairs only paired lift](../../generated/style_research/style_transfer_experiments/iterations/aligned_pairs_v1/evaluations/screening_v1_initial/methods/aligned_pairs_only/medium/charts/paired_margin_lift.svg)

#### `aligned_pairs_plus_cards`

Largest mean lift in both arms (+0.194 own, +0.147 cross), but cards do not solve control: 7/24 own and 4/12 cross rows fail fidelity. This arm is the strongest style mover and the weakest cross-arm fidelity result.

Recorded paragraph-level surface flags: `dialogue_turn_surface_mismatch` (22), `latin_token_mismatch` (6).

![Aligned pairs plus cards paired lift](../../generated/style_research/style_transfer_experiments/iterations/aligned_pairs_v1/evaluations/screening_v1_initial/methods/aligned_pairs_plus_cards/medium/charts/paired_margin_lift.svg)

#### `aligned_pairs_edit_plan`

Lowest own-author fidelity failure rate (6/24) and an admissible 1/12 cross rate, but it still exceeds the own-arm gate. Its lift is positive and cluster-stable, yet smaller than the cards arm.

Recorded paragraph-level surface flags: `dialogue_turn_surface_mismatch` (10), `latin_token_mismatch` (4).

![Aligned-pair edit plan paired lift](../../generated/style_research/style_transfer_experiments/iterations/aligned_pairs_v1/evaluations/screening_v1_initial/methods/aligned_pairs_edit_plan/medium/charts/paired_margin_lift.svg)

## Registered Family-Rerank Diagnostic

The preregistered diagnostic selects, per row, the fidelity-passing candidate with the highest target margin, then paired lift, then lexical method ID. It is not a generated method arm. Unresolved rows contribute zero lift rather than being dropped.

| Arm | Rows | Selected | Unresolved | Conservative lift | Selected-only lift | Positive selected rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Own-author reconstruction | 24 | 23 | 1 | +0.223 | +0.232 | 20 |
| Cross-author transfer | 12 | 12 | 0 | +0.193 | +0.193 | 9 |

This suggests that candidate-level selection may outperform any fixed prompt family. It does **not** establish accuracy, semantic safety, or confirmation performance. Iteration 3 must register candidate generation and reranking as the tested pipeline before output generation, with its own valid threshold and independent judgments.

## Independent Evaluation

A separate `gpt-5.5` evaluator reviewed frozen aggregate files and performed English-grounded stratified close reading. Its recorded verdict is **No base method qualifies for confirmation**. Independent English-grounded close reading covered 13 method-output cells.

The 13 reviewed method-output cells covered every method, both arms, high- and negative-lift rows, and hard-fidelity failures. Most inspected dialogue flags were quote-attribution relocations that preserved proposition and speaker, so they look conservative for semantics. They remain valid protocol failures. The reviewer also found literal `},{` residue in some failed outputs, which is a real readability and generation-artifact defect. Inspected Latin-token mismatches generally preserved event meaning but still violated the frozen token-consistency rule.

This close reading is an independent outcome audit, not the preregistered blinded semantic-judge layer. No independent judgment artifact was supplied, so semantic and readability noninferiority remain untested and cannot be claimed.

The evaluator was instructed to include all three methods, both arms, high- and negative-lift cases, and deterministic fidelity failures; distinguish likely semantic risk from conservative surface false positives; and leave the preregistered gate unchanged. The full audit is at `generated/style_research/style_transfer_experiments/iterations/aligned_pairs_v1/audits/iteration2_outcome_audit.md`.

## Conclusion

Aligned pseudo-parallel evidence materially improves the proxy style margin compared with iteration 1's unaligned examples, and the lift is positive in both own- and cross-author arms. However, no fixed method is eligible for confirmation because its own-author fidelity failure rate exceeds the frozen 10% limit. The invalid threshold binding independently prevents any threshold-based 80% claim. **No method from iteration 2 is approved for production.**

## Iteration 3 Requirements

1. Freeze a correctly bound generated-domain threshold before any method output exists.
2. Test the registered multi-candidate family reranker as an actual pipeline, not a post-output diagnostic.
3. Add paragraph-batched long-form generation with deterministic reassembly to reduce schema retries on 46-53 paragraph chunks.
4. Freeze a sanitizer that rejects literal JSON residue and a dialogue rule that preserves speech-tag placement unless explicitly allowed.
5. Freeze a Latin-name/token policy and test lighter style intensity before adding more style evidence.
6. Preserve the same production-compatible English-to-neutral prompt and English semantic authority.
7. Run blind independent semantic/readability judgments on screened candidates before promotion.
8. Open the 116-row confirmation set only if the entire frozen pipeline passes screening; require at least 80% on both registered endpoints.
9. If prompting plus reranking still fails, preregister a learned adapter or preference-optimization study using the aligned pairs, then evaluate it on the same separation rules.

## Reproduction

Validate the frozen iteration-two package:

```bash
uv run python experiments/iteration2/prepare.py validate
```

Re-run deterministic scoring from frozen outputs:

```bash
uv run python experiments/iteration2/evaluate_style_transfer_methods.py evaluate \
  --experiment-root generated/style_research/style_transfer_experiments/iterations/aligned_pairs_v1 \
  --sample-set development_proxy_v1 \
  --run-id iteration2_aligned_gpt55_v1 \
  --analysis-lock generated/style_research/style_transfer_experiments/iterations/aligned_pairs_v1/protocols/pre_style_analysis_lock.v1.e97a085379b0cefe15bd60bd7d9d539c73f23b283e8c2d8c19aa805e209a7d4f.json \
  --selection-file generated/style_research/style_transfer_experiments/iterations/aligned_pairs_v1/sample_sets/development_proxy_v1.screening_v1_ids.json \
  --scorer-mode load \
  --screening-phase initial \
  --output-dir generated/style_research/style_transfer_experiments/iterations/aligned_pairs_v1/evaluations/screening_v1_initial
```

Rebuild the family-rerank diagnostic and this report:

```bash
uv run python experiments/iteration2/analyze_family_rerank.py
uv run python experiments/iteration2/report_style_transfer_iteration2.py
```

## Primary Artifacts

- Preregistration: `generated/style_research/style_transfer_experiments/iterations/aligned_pairs_v1/protocols/iteration2_preregistration.v1.json`
- Active lock: `generated/style_research/style_transfer_experiments/iterations/aligned_pairs_v1/protocols/pre_style_analysis_lock.v1.e97a085379b0cefe15bd60bd7d9d539c73f23b283e8c2d8c19aa805e209a7d4f.json`
- Pair-pool summary: `generated/style_research/style_transfer_experiments/iterations/aligned_pairs_v1/sample_sets/iteration2_aligned_pair_pool_v1.summary.json`
- Proxy summary: `generated/style_research/style_transfer_experiments/iterations/aligned_pairs_v1/sample_sets/development_proxy_v1.summary.json`
- Method registry: `generated/style_research/style_transfer_experiments/iterations/aligned_pairs_v1/method_registry/style_methods.v1.json`
- Screening evaluation: `generated/style_research/style_transfer_experiments/iterations/aligned_pairs_v1/evaluations/screening_v1_initial/evaluation_summary.json`
- Per-method tables and graphs: `generated/style_research/style_transfer_experiments/iterations/aligned_pairs_v1/evaluations/screening_v1_initial/methods/`
- Family rerank: `generated/style_research/style_transfer_experiments/iterations/aligned_pairs_v1/evaluations/screening_v1_initial/diagnostics/family_rerank/`
- Pre-generation audit: `generated/style_research/style_transfer_experiments/iterations/aligned_pairs_v1/audits/pre_style_generation_methodology_audit.md`
- Independent outcome audit: `generated/style_research/style_transfer_experiments/iterations/aligned_pairs_v1/audits/iteration2_outcome_audit.md`

## Method Sources

- Krishna, Wieting, and Iyyer (2020), STRAP: https://aclanthology.org/2020.emnlp-main.55/
- Patel, Andrews, and Callison-Burch (2024 revision), Styll: https://arxiv.org/abs/2212.08986
- Suzgun, Melas-Kyriazi, and Jurafsky (2022), Prompt-and-Rerank: https://aclanthology.org/2022.emnlp-main.141/
- Horvitz et al. (2024), TinyStyler: https://aclanthology.org/2024.findings-emnlp.781/
- Liu and May (2025), iterative preference optimization with pseudo-parallel data: https://aclanthology.org/2025.naacl-long.135/
