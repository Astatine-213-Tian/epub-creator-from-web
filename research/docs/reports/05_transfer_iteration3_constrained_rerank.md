# Eternal Gate Style-Transfer Research: Iteration 3

Generated from frozen local artifacts last refreshed at
`2026-07-13T06:42:44Z`.

## Executive Status

> **COMPLETE - NO METHOD QUALIFIED. ITERATION 3 IS A NO-GO.**

Iteration 3 completed its preregistered initial screen under the active
content-addressed lock. Four fixed `gpt-5.5` generators each produced all 36
required outputs, for 144/144 valid final method samples. The frozen style meter
then found **0/24 own-author and 0/12 cross-author deterministic style
successes for every fixed method**. The selection-conditioned rerank diagnostic
also produced 0/24 and 0/12 successes. No method entered the provisional
shortlist, so confirmation, final validation, and production use remain closed.

| Stage | Data or artifact | Current result | Status |
| --- | --- | --- | --- |
| Corpus evidence | 29 masked target-author train books plus 49 comparison authors | 29 one-book-per-pair aligned excerpts and two statistically supported microcards | complete |
| Proxy cohort | 32 calibration + 36 screening + 116 confirmation rows | 184 opaque samples, zero overlap with prior transfer rows | complete |
| English semantic source | 68 calibration/screening rows | 68/68 approved after shrinking repair rounds and one independent adjudication | complete |
| Neutral Chinese | Same 68 rows | 68/68 generated with the production pass-1 prompt contract | complete |
| Style-meter calibration | 32 calibration rows only | threshold `0.354900062084198`; leave-one safety check changes it by `0.0` | complete and bound to the active lock |
| Method registration | Control + four fixed generators + one rerank diagnostic | Exact prompts, schemas, payloads, and hypotheses frozen | complete |
| Execution validation | Superseded pre-outcome attempts | Request geometry, provenance, and source-ID defects were corrected; all superseded outputs were excluded | complete |
| ID-safe readiness audit | Active `d086827f...` lock | Prompt-hash recheck issued GO for the exact lock | complete |
| Admissible generation | Four generators x 36 rows | 144/144 final outputs; 18 failed block attempts recovered by deterministic split/retry | complete |
| Deterministic screening | 24 own-author + 12 cross-author rows per method | 0% style success in both arms for every fixed method | complete; no method qualified |
| Candidate rerank | 36 derived selections | Best descriptive lift, but 0% success and selection-conditioned on the same meter | diagnostic only |
| Independent semantic/readability judging | Provisional-shortlist rows only | No shortlist existed, so the registered judging stage was not entered | not applicable after screening rejection |
| Independent outcome audit | Frozen aggregate and row-level artifacts | 64 method-arm cells close-read; 11/11 fallacy checks; NO-GO | complete |
| Confirmation | 116 untouched rows | Not opened because zero methods qualified | locked |
| Final validation | 80 rows from four reserved books | Not opened | locked |
| Production decision | Eternal Gate pass 2 | No Iteration 3 method selected | no-go |

This file is the sole canonical human-readable report for Iteration 3. It
contains the design, tested methods, frozen results, separate-agent evaluation,
failure analysis, and next-iteration decision.

## Research Question

Can a light Chinese rewrite grounded in entity-masked target-author training
evidence move production-compatible neutral Chinese toward Feitian Yexiang's
measurable author signal while preserving the English semantic source?

The experiment separates six questions:

1. Do retrieved **aligned neutral-to-target pairs** teach recurring author
   transformations better than unaligned prose examples?
2. Does an explicit, low-risk **edit-plan contract** improve the consistency of
   pair application?
3. Can two small, statistically supported **rule-linked microcards** move author
   signal without exposing retrieved target passages?
4. Are aligned pairs and microcards **complementary**, or does the larger prompt
   reduce instruction and paragraph-ID adherence?
5. Can deterministic candidate reranking find complementary successes without
   being mistaken for an independently evaluated generator?
6. Does any apparent style gain survive **English-grounded semantic and
   readability judgment**, book/author separation, confirmation, and one-time
   final validation?

### Preregistered hypotheses

| ID | Hypothesis | Required evidence |
| --- | --- | --- |
| H1 | Aligned pairs produce positive target-author margin lift in both own- and cross-author arms. | Positive paired mean lift in each arm under the frozen meter |
| H2 | A constrained edit plan lowers hard-fidelity failures without eliminating pair-driven lift. | `aligned_pairs_edit_plan_light` versus `aligned_pairs_light` |
| H3 | Microcards alone preserve fidelity better than full retrieved-pair prompts. | Lower hard-failure rate with positive lift |
| H4 | Pair-plus-microcard evidence is complementary. | Better lift/success than either evidence source alone without crossing the 10% fidelity ceiling |
| H5 | Candidate methods are complementary at the row level. | Diagnostic rerank selects multiple families and improves conservative lift, reported as selection-conditioned only |
| H6 | At least one fixed generator reaches the final 80% endpoint without semantic/readability regression. | Judged confirmation and one-time final validation; screening cannot establish H6 |

## Relation to Iterations 1 and 2

Iteration 3 does not restart method selection from an empty baseline. It tests
specific failure diagnoses from the earlier frozen studies.

| Earlier result | Evidence | Iteration 3 response |
| --- | --- | --- |
| Iteration 1 unaligned examples did not teach a usable transformation. | Retrieved and contrastive examples moved mean style margin backward. | Use neutral-to-original pseudo-parallel pairs from target-author train books. |
| Iteration 1 structured rules often raised surface-fidelity failures. | Dialogue, Latin-token, and quote topology failures dominated. | Use light local edits, explicit invariants, counterexamples, and at most one operation per changed paragraph. |
| Iteration 2 aligned methods produced positive mean lift in both arms. | All three pair families improved relative margin. | Retain aligned evidence but reduce prompt size and directly test lower-risk variants. |
| Iteration 2 failed its <=10% hard-fidelity gate. | Own-author failures were 25.0%-29.2%. | Add stricter dialogue/token guardrails and independent blind critique before promotion. |
| Iteration 2's absolute threshold was invalidly bound. | Threshold hashes still referenced the Iteration 1 allocation/protocol. | Recalibrate on 32 Iteration 3 rows before any admissible style output. |
| Iteration 2's family rerank was post-output diagnostic only. | It improved conservative lift but was not an independent method. | Register candidate rerank in advance and preserve its non-promotable, selection-conditioned label. |
| Long chunks caused repeated schema failures. | Whole-request retries omitted IDs on 46-53 paragraph chunks. | Freeze paragraph blocks and deterministic reassembly before admissible generation. |

No Iteration 1 or 2 style output is reused as an Iteration 3 outcome. The
historical reports remain:

- [Iteration 1 report](03_transfer_iteration1_prompt_methods.md)
- [Iteration 2 report](04_transfer_iteration2_aligned_pairs.md)

This document is the only report for Iteration 3. Superseded execution attempts
are summarized only where they explain the validity of the final method contract;
they are not research methods or outcome runs.

## Experimental Design and Data Lineage

### Data roles

| Role | Construction | Rows or sources | Exposure |
| --- | --- | ---: | --- |
| Target pair pool | One `entity_masked_v3` train excerpt from each selected target-author book | 29 books / 29 pairs | Retrieval evidence only; never an evaluation target |
| Pair English source | `gpt-5.4` semantic reconstruction from each masked target excerpt | 29 | Used to build production-direction neutral Chinese |
| Pair neutral Chinese | Same frozen neutral prompt as planned Eternal Gate pass 1 | 29 | Neutral side of aligned demonstration and retrieval query |
| Pair target Chinese | Original masked train excerpt | 29 | Target side of demonstration only |
| Calibration | Held-out original target Chinese versus its neutral reconstruction | 32 | Threshold selection only; no method generation |
| Screening, own-author | Held-out chunks from eight target-author development books | 24 | Method-family screening |
| Screening, cross-author | Held-out chunks from 12 comparison authors | 12 | Tests whether transfer is more than reconstruction of already target-like prose |
| Confirmation | Same book/author families, disjoint chunks | 116 | Unopened until judged promotion |
| Final validation | Four reserved target-author books, 20 chunks each | 80 planned | Created once only after confirmation success |

This is **in-context style transfer**, not model fine-tuning. The term
`training pair` refers to prompt evidence sampled from train-split books. Model
weights are unchanged.

### Frozen cohort geometry

| Partition | Own-author rows | Cross-author rows | Total | Selection state |
| --- | ---: | ---: | ---: | --- |
| Calibration | 32 | 0 | 32 | Used before style generation |
| Screening | 24 | 12 | 36 | Frozen; next active stage |
| Confirmation | 80 | 36 | 116 | Unopened |
| Total | 136 | 48 | 184 | Fixed |

- Sampling seed: `20260714`.
- Minimum distance from every previously allocated chunk: five chunks.
- Overlap with prior iteration transfer rows: zero.
- Eternal Gate source text is excluded from proxy construction.
- The four final-validation books are excluded from transfer development.
- One safety-sensitive `相见欢` chunk was replaced one-for-one with another
  chunk from the same book before calibration or style generation.
- Public runner rows contain opaque IDs and source text only. Author, book,
  stratum, and original Chinese target fields are evaluator-only.

### Screening composition

Own-author reconstruction uses exactly three rows from each book:

| Target-author book | Screening rows | Confirmation rows |
| --- | ---: | ---: |
| 万物风华录 | 3 | 10 |
| 乱世为王 | 3 | 10 |
| 图灵密码 | 3 | 10 |
| 天宝伏妖录 | 3 | 10 |
| 山有木兮 | 3 | 10 |
| 清平梦华录 | 3 | 10 |
| 相见欢 | 3 | 10 |
| 骑士之歌 | 3 | 10 |

Cross-author transfer uses one screening row and three confirmation rows from
each comparison author:

| Comparison author | Screen | Confirmation | Comparison author | Screen | Confirmation |
| --- | ---: | ---: | --- | ---: | ---: |
| priest | 1 | 3 | 唐酒卿 | 1 | 3 |
| 墨香铜臭 | 1 | 3 | 巫哲 | 1 | 3 |
| 木苏里 | 1 | 3 | 梦溪石 | 1 | 3 |
| 淮上 | 1 | 3 | 漫漫何其多 | 1 | 3 |
| 稚楚 | 1 | 3 | 莫晨欢 | 1 | 3 |
| 西子绪 | 1 | 3 | 酱子贝 | 1 | 3 |

### Scene and length coverage

| Scene stratum | Screening | Confirmation |
| --- | ---: | ---: |
| Action/conflict | 3 | 22 |
| Dialogue | 4 | 20 |
| Internal reflection | 5 | 15 |
| Interpersonal care | 9 | 21 |
| Travel/transition | 8 | 18 |
| Worldbuilding/exposition | 7 | 20 |

| Measurement | Screening | Confirmation |
| --- | ---: | ---: |
| Paragraph count, minimum | 28 | 18 |
| Paragraph count, maximum | 64 | 86 |
| Paragraph count, mean | 44.67 | 43.92 |
| Clean CJK count, minimum | 1,386 | 1,275 |
| Clean CJK count, maximum | 1,500 | 1,500 |
| Clean CJK count, mean | 1,474.22 | 1,473.74 |

The screen deliberately spans dialogue density, sentence flow, punctuation,
position in book, and scene type. It is not a prevalence sample of all Chinese
webnovel scenes; its purpose is controlled method comparison.

### Information available at each stage

| Stage | English | Neutral Chinese | Train evidence | Original evaluation Chinese | Author/book label | Style score |
| --- | --- | --- | --- | --- | --- | --- |
| English reconstruction | No; produces it | No | No | Yes, in isolated source request | No | No |
| English QA/repair | Candidate English | No | No | Yes, in isolated QA request | No | No |
| Neutral translation | Yes | Produces it | No | No | No | No |
| Style transfer | Yes | Yes | Method-specific masked train evidence | No | No | No |
| Candidate rerank | No new generation | Yes | Frozen candidates only | No | Evaluator metadata only for masking | Target margin used for selection |
| Independent critique | Yes | Yes | No retrieved evidence | No | No | No |
| Deterministic evaluator | Yes through provenance | Yes | Frozen scorer only | Evaluator-only benchmark target | Yes | Computes score |

The style model cannot read hidden original targets, evaluator allocation,
book/author labels, threshold scores, or promotion results.

## English Semantic Source and Neutral Chinese

### Why the benchmark passes through English

Production begins from the published English Eternal Gate translation. A
Chinese-to-Chinese neutralization task would be easier and distributionally
different: it could preserve Chinese syntax and lexical choices that are absent
from English. Each proxy row therefore follows the production direction:

```text
held-out Chinese passage
  -> isolated English semantic reconstruction
  -> independent English QA and shrinking repair
  -> production-compatible neutral Chinese
  -> style-transfer methods
  -> compare against English and hidden original Chinese
```

The original Chinese is never supplied to neutral translation or style transfer.

### Neutral prompt compatibility

The neutral prompt is the same contract intended for Eternal Gate pass 1:

- translate only the supplied English meaning;
- use fluent, restrained modern Chinese;
- do not imitate a named author;
- do not add imagery, motive, emotion, lore, action, or causal relations;
- preserve paragraph IDs and all semantic distinctions needed by later QA.

Changing this prompt after seeing method outcomes would alter every paired
baseline and requires a new study.

### Source QA and repair

Only English rows rejected by the prior QA stage entered the next repair round.
Already approved rows were not repeatedly rewritten.

| QA stage | Rows checked | Approved in that stage | Rows remaining afterward |
| --- | ---: | ---: | ---: |
| Initial QA | 68 | 21 | 47 |
| Repair QA 1 | 47 | 26 | 21 |
| Repair QA 2 | 21 | 12 | 9 |
| Repair QA 3 | 9 | 5 | 4 |
| Repair QA 4 | 4 | 0 | 4 |
| Repair QA 5 | 4 | 3 | 1 |
| Repair QA 6 | 1 | 0 | 1 |
| Independent adjudication / QA 7 | 1 | 1 | 0 |

The final row oscillated around one narrow aspectual interpretation. A separate
`gpt-5.5` adjudication froze the conservative reading and an audit that binds the
source repair, contradictory QA decisions, final repair, final approval, agent,
and hashes. The adjudication explicitly applies to that ambiguity only; it does
not waive other English-source checks.

Final source status:

| Artifact | Required rows | Present and approved |
| --- | ---: | ---: |
| English semantic source | 68 | 68 |
| Effective approved English after repair/adjudication | 68 | 68 |
| Neutral Chinese | 68 | 68 |

## Target-Author Evidence Construction

### Aligned pair pool

The pair pool contains one masked excerpt from each of 29 target-author train
books. At retrieval time, no two selected pairs may come from the same book.

| Property | Value |
| --- | ---: |
| Pairs | 29 |
| Distinct source books | 29 |
| Pair source view | `entity_masked_v3` |
| Paragraphs per full pair, minimum | 4 |
| Paragraphs per full pair, maximum | 27 |
| Paragraphs per full pair, mean | 13.17 |

| Pair scene | Count |
| --- | ---: |
| Action/conflict | 9 |
| Dialogue | 9 |
| Internal reflection | 7 |
| Interpersonal care | 1 |
| Travel/transition | 3 |

Each pair is:

```text
masked target-author train passage
  -> English semantic reconstruction
  -> production-compatible neutral Chinese
  -> aligned demonstration: neutral Chinese -> original masked target Chinese
```

Retrieval uses a frozen semantic-and-structural signature and returns four pairs
with at most one pair per book. Evaluation targets and final books cannot enter
this pool.

### Compact reference representation

The four retrieved pair identities and ranks are unchanged, but each pair is
represented by one deterministic contiguous three-paragraph before/after window.
The window maximizes the frozen sum of paragraph change scores within that pair.

| Property | Full-pair pilot | Compact active contract |
| --- | ---: | ---: |
| Retrieved pair identities | 4 | Same 4 |
| Exposed paragraphs per pair | 4-27 | Exactly 3 |
| Total exposed before paragraphs | 58-96 in observed requests | 12 |
| Total exposed after paragraphs | 58-96 in observed requests | 12 |
| Window selection | Full chunk | Deterministic maximum-change contiguous window |
| Outcome-driven choice | No | No |

In a measured representative payload, aligned-reference JSON fell from 12,408
bytes to 3,626 bytes, a 70.8% reduction. Across all 36 screening rows, a
metadata-only audit verified four references, ranks 1-4, three neutral and three
target paragraphs per reference, identical pair/window identity across aligned
arms, and zero aligned references in the microcards-only arm.

### Rule-linked microcards

Only two rules met the support, coverage, and isolated-example requirements.

| Card | Direction in target | Target value | 49-author mean | z-score | Target-book coverage | Guardrail |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Connective function-word rate | Lower | 2.786660 | 4.525882 | -1.114579 | 29/29 | Do not remove a connective when it carries source causality, contrast, modality, sequence, or deixis. |
| Simple speech-tag rate | Higher | 9.761639 | 2.925336 | +3.100656 | 29/29 | Use a plain tag only when attribution is needed; never invent tone, motive, or emotion. |

Each microcard contains the corpus statistic, a trigger, semantic invariants, two
isolated observed before/after examples from different pair books, and one
leave-unchanged counterexample. At most two cards and one local operation per
changed paragraph are allowed.

## Models, Prompts, Schemas, and Frozen Execution

### Models

| Component | Model | Reasoning | Sampling role |
| --- | --- | --- | --- |
| English reconstruction, QA, repair, neutral Chinese | `gpt-5.4` | high | Shared source/control construction |
| Independent English adjudication | `gpt-5.5` | high | One unresolved source ambiguity |
| All fixed style generators | `gpt-5.5` | high | Same model across method arms |
| Blind semantic/readability critiques | `gpt-5.5` | high | Separate calls after shortlist freeze |
| Methodology and outcome audits | `gpt-5.5` | high | Separate Codex sessions; no generation |

English and neutral artifacts were frozen before style generation. Cross-run
provenance is explicit: generation, critique, evaluation, reranking, and judgment
freezing each record and verify both the source-artifact identity and the
style-output identity. Run-directory suffixes are implementation provenance only.

### Frozen prompt and schema identities

| Artifact | SHA-256 |
| --- | --- |
| English semantic-source prompt | `03a79e0b0de1df5b9721f81eb967aee49f9f0b13a482647fefe15dae115b21fa` |
| English-source QA prompt | `c8243b5e39361e79466484a90c68746011ff6b78a3fa7acf3ef11e59f32648d5` |
| English-source repair prompt | `13bd8b500ea1ba9f590dfa81b3acb328304800fc7e4d8ee57900b6639d57d061` |
| Neutral-translation prompt | `600cc82d5f40ec70b667b918f5c5cfebad1b8378212e010b49c4ebf2847ad43b` |
| Style-transfer prompt | `90f709addfde27d087e83f78f9f61f0078f81a238fa3484e4a47ea1c47d6091f` |
| Independent critique prompt | `0ece968624c110e05bc5fcd63be5e5e0d5c9b1d2f1d897ec44b41fdf12349bd8` |
| Style output schema | `8b19b962f2e333ab9ec2a5a34955190e8832fef01397f554aa7436ea3bc31be4` |
| Critique output schema | `69d6513444945239025519e80e6b760587fbde94e866f43c353a10788b9a6857` |
| Active method asset content | `f9389eaf905e85a2fb61be524f5ee9e019d73888f3f82c9bb7fbe2023549e619` |

Every model request also records the exact prompt, schema, model config, runner,
selection file, execution batch, sandbox profile, response ID, token usage, and
analysis-lock binding. Style-generation subprocesses cannot read repository files
outside the constructed request.

### Analysis-lock state

The active pre-generation lock is:

```text
protocols/pre_style_analysis_lock.v1.d086827f31d3fab39e5ef9129ac07422bcf81c5232d2baee6c06fa07e4e0af2a.json
```

| Lock property | Value |
| --- | --- |
| Content SHA-256 | `d086827f31d3fab39e5ef9129ac07422bcf81c5232d2baee6c06fa07e4e0af2a` |
| File SHA-256 | `f72616bc5cbfdaff623a51868dafbbfadfe8385af0246a8a7f8f9c041a336db3` |
| Source bindings | 57 |
| Artifact bindings | 53 |
| Active style outputs before lock | 0 |
| Executable gate checks | 13/13 passed |

Pre-outcome execution checks found that global paragraph IDs could exhaust the
frozen retry limit. The active replacement lock therefore binds block-local model
IDs, deterministic restoration of original IDs, and recursive bisection of only
a failed block. A separate audit also found and prompted correction of one stale
nested model-config prompt hash. A narrow independent recheck issued GO for the
exact `d086...` lock before admissible generation. The same lock remained active
through generation, deterministic evaluation, reranking, and
provisional-shortlist selection.

## Methods Tested

### Overview

| Method | Meaning | Evidence visible to `gpt-5.5` | Hypothesis | Promotable? |
| --- | --- | --- | --- | --- |
| `neutral_only:none` | Score neutral Chinese unchanged. | English and neutral Chinese | Paired negative control | No |
| `aligned_pairs_light:light` | Infer recurring transformations from four aligned windows. | English, neutral, four pair windows | Direct transformation evidence raises style signal. | Yes |
| `aligned_pairs_edit_plan_light:light` | Infer a private plan restricted to low-risk operations. | Same four windows plus edit contract | Explicit planning lowers fidelity failures. | Yes |
| `microcards_only_light:light` | Apply only activated statistical microcards. | English, neutral, two cards; no retrieved pairs | Small interpretable rules are safer and sufficient for some rows. | Yes |
| `rule_linked_microcards_light:light` | Combine four aligned windows and at most two cards. | Both evidence sources | Explicit rules improve pair generalization when they agree. | Yes |
| `candidate_rerank:light` | Select a frozen candidate or fall back to neutral. | Candidate outputs and frozen meter only | Candidate families have complementary row-level successes. | Diagnostic only |

Every generated arm receives byte-identical English and neutral Chinese.

### `neutral_only:none`

No style rewrite is performed. It supplies the paired baseline for target margin,
flow metrics, fidelity, readability, and semantic-judge comparisons. Because it
cannot have strictly positive lift over itself, it is not promotable.

### `aligned_pairs_light:light`

The model receives four retrieved target-author neutral/style windows. It may
infer repeated local transformations but must apply at most one clearly supported
operation per changed paragraph. Dialogue-tag placement and Latin tokens are
frozen. Weak evidence means leave the paragraph unchanged.

Primary contrast: does aligned evidence reproduce Iteration 2's positive lift
after prompt compaction and stricter invariants?

### `aligned_pairs_edit_plan_light:light`

The evidence is identical to `aligned_pairs_light`; only the reasoning contract
changes. The model privately infers a plan, does not output it, and may use only:

- clause compression;
- sentence-boundary adjustment;
- plain speech-tag substitution without relocation;
- function-word reduction;
- reaction-beat placement without adding an action.

An operation must be supported by at least two pairs. Dialogue-tag relocation,
quote-topology change, Latin-token change, invented content, and reference-phrase
copying are forbidden.

Primary contrast: same examples, stricter transformation selection.

### `microcards_only_light:light`

No full aligned pair is exposed. The model receives only the two activated rule
descriptions, isolated examples, counterexamples, and semantic invariants. This is
the cleanest ablation of explicit interpretable style knowledge versus retrieved
target prose.

Primary contrast: can author signal move through narrow function-word/dialogue
operations with lower content leakage and lower fidelity risk?

### `rule_linked_microcards_light:light`

The model receives the same four pair windows as the aligned arm plus at most two
microcards. Pair evidence is the baseline; a card may be applied only when its
trigger is present and its examples agree with the local source-supported
opportunity.

Primary contrast: complementarity versus prompt competition.

### `candidate_rerank:light`

Reranking is deterministic and runs only after all four generator outputs are
frozen. Per row it:

1. rejects paragraph/order, number, placeholder, Latin-token, dialogue-start,
   quote, JSON-residue, length, and introduced-reference-copy failures;
2. requires strictly positive target-margin lift over neutral;
3. finds the highest target-author margin;
4. among candidates within `0.03` margin of the maximum, chooses minimum
   character edit ratio;
5. breaks any remaining tie by method ID;
6. falls back to neutral when no candidate qualifies.

Because the same style meter selects and diagnoses this output, rerank success is
selection-conditioned. It cannot satisfy the independent 80% endpoint or be
reported as an independently generated arm.

## Style Meter and Calibration

### Frozen classifier

The meter is the selected Stage-1 class-balanced linear SGD hinge classifier over
exact Chinese character 2-4 gram TF-IDF features with `min_df=20`, evaluated on
`entity_masked_v3` text.

| Binding | Value |
| --- | --- |
| Scorer ID | `class_balanced_sgd_hinge_exact_char_ngrams_min_df_20.v1` |
| Input view | `entity_masked_v3` |
| Classifier SHA-256 | `e53d99ace2917ce18acf0b1040c542539c10cef417b61c06ae371ab5a8a8920a` |
| Scorer config SHA-256 | `d05b95e71d091c7a86df77fadd1ec4af06c6f8c00d66f37be0256b41ab0c63a8` |
| Masking artifact SHA-256 | `035be1c8791b560e7db887d6ca120820300627f181b1a4e503acc667527c1102` |

The classifier is not retrained on proxy outputs. The target-author decision
margin is an uncalibrated hinge margin, not a probability or human style rating.

### Threshold derivation

The 32 calibration rows compare held-out original target-author Chinese against
their own neutral translations after masking. The algorithm enumerates all unique
observed margins, keeps thresholds with original-positive sensitivity at least
80% and neutral false-positive rate at most 10%, maximizes balanced accuracy, and
chooses the higher threshold on a tie.

| Quantity | Value |
| --- | ---: |
| Calibration rows | 32 |
| Candidate thresholds | 64 |
| Qualifying thresholds | 10 |
| Selected margin threshold | `0.354900062084198` |
| Original-positive sensitivity | 100% |
| Neutral false-positive rate | 0% |
| Specificity | 100% |
| Balanced accuracy | 100% |
| Leave-one safety-replacement-out threshold | `0.354900062084198` |
| Absolute sensitivity-analysis delta | `0.0` |

These values describe the frozen calibration cohort. They do not imply 100%
future method accuracy.

## Evaluation Rules

### Per-row deterministic style success

A row is a deterministic style success only when **all** conditions hold:

1. target-author decision margin is at or above the frozen threshold;
2. target-author rank is at most five among the 50 authors;
3. paired margin lift over that row's neutral control is strictly positive;
4. no hard fidelity or reference-copy failure is present.

Deterministic success is provisional until a valid independent judgment is bound.

### Hard-fidelity failures

The protocol counts the row as failed for any of the following:

- missing/duplicate output or paragraph ID/order mismatch;
- entity, number, role, or speaker mismatch;
- changed causality, chronology, negation, modality, or intensity;
- unsupported event, motive, lore, joke, intimacy, injury, or setting detail;
- introduced copy of eight or more CJK characters from reference evidence;
- leaked reference book/name/place/lore material;
- stale or mispaired independent judgment;
- high-severity blind semantic failure;
- high-severity blind readability failure.

Surface checks also track quote balance, dialogue-start topology, Latin tokens,
JSON residue, aggregate/paragraph CJK-length ratios, edit ratio, sentence flow,
paragraph flow, punctuation, and dialogue metrics.

### Independent semantic and readability layer

For each generated method that enters judged evaluation, a separate blind call
receives:

- the English semantic source;
- the neutral Chinese control;
- one candidate Chinese output;
- no author name, book title, hidden original Chinese, style score, method label,
  or retrieved evidence.

It returns paragraph-aligned candidate and neutral reviews with `pass`, `minor`,
or `major` status and explicit fidelity/readability issue arrays. Non-LLM code
validates IDs, reconstructs the exact request hash, binds source/style runs,
candidate/neutral/critique files, ledgers, configs, model, prompt, selection, and
analysis lock, then freezes judgment rows. A major candidate semantic or
readability issue is a hard failure. Neutral major semantic failures are retained
for paired noninferiority analysis.

### How to read the reported metrics

| Metric | Meaning | What it does not mean |
| --- | --- | --- |
| Target margin | Hinge decision value for 非天夜翔 after masking | Probability of authorship |
| Target rank | Rank of 非天夜翔 among 50 author classes | Human similarity ranking |
| Target top-class share | Fraction predicted exactly as 非天夜翔 | Style-success rate |
| Paired margin lift | Candidate margin minus same-row neutral margin | Absolute success |
| Deterministic style success | Threshold/rank/lift/fidelity conjunction | Final judged success |
| Final style success | Deterministic success plus valid no-major-issue judgment | Proof of production quality |
| Hard-fidelity rate | Rows failing any registered hard gate | Fraction with semantic drift only |
| Cluster bootstrap interval | Resampling books or authors, not individual rows | Independent external replication |

### Screening promotion rule

At most three fixed non-control generators may proceed. Each must have:

- valid threshold-based style success reporting;
- positive mean paired lift in own- and cross-author arms;
- hard-fidelity failure rate at most 10% in each arm;
- zero reference-copy failures;
- completed independent semantic/readability evaluation;
- no stale binding or missing row.

For 24 own-author screening rows, at most two hard failures are allowed. For 12
cross-author rows, at most one is allowed. The 36-row screen supports method
selection and rejection, not the final 80% claim.

### Confirmation endpoints

| Endpoint | Frozen requirement |
| --- | --- |
| Own-author, 80 rows | >=80% success; >=70% in every development book; Wilson 95% lower bound >=70%; book-cluster bootstrap lower bound >=70%; paired margin-lift interval excludes zero |
| Cross-author, 36 rows | Report separately; >=80% success; Wilson lower bound >=70%; positive lift in >=10/12 authors; author-cluster bootstrap lower bound >=70% |
| Semantic noninferiority | 95% cluster-bootstrap upper bound for candidate-minus-neutral high-severity semantic failure <=0; missing outputs fail |
| Readability noninferiority | No high-severity candidate readability regression versus neutral; missing outputs fail |

### One-time final validation

Only a confirmation winner may create the 80-row final set: 20 chunks each from
`星辰骑士`, `夺梦`, `定海浮生录`, and
`国家一级注册驱魔师上岗培训通知`. A pre-reveal lock must bind all prompts,
schemas, code, model config, scorer, method payload, source pipeline, promotion,
and confirmation. The same method must achieve at least 80% own-author success
with a book-cluster bootstrap lower bound of at least 70%.

## Execution Validity

Several pre-outcome execution attempts exposed request-size, cross-run
provenance, and paragraph-ID adherence defects. They were used only to harden the
runner; their prose was never scored, compared, or used to change hypotheses,
thresholds, evidence, or promotion gates. The admissible execution therefore:

1. reads frozen English and neutral prerequisites through explicit source hashes;
2. sends compact aligned windows and block-local paragraph IDs to the model;
3. restores original IDs deterministically without modifying Chinese prose;
4. bisects and retries only an invalid block, retaining failed-parent provenance;
5. starts every fixed method from zero and excludes all superseded candidates.

Three deterministic tests cover short-block restoration, failed-parent bisection
with ordered reassembly, and irrecoverable one-paragraph failure. This history is
relevant only to denominator completeness and reproducibility; it is not an
additional method comparison.

## Screening Results

### Endpoint and interpretation

The primary screening endpoint is the fraction of rows satisfying the frozen
deterministic style-success rule. A row must reach target-author margin
`>= 0.354900062084198`, rank the target author in the top five, improve over its
paired neutral input, and pass the registered artifact, paragraph, fidelity, and
no-copy gates. The style margin is an SVM-like classifier decision value, not a
probability. `Mean lift` is method margin minus neutral margin on the same row.

Every fixed method recorded **0/24 own-author successes and 0/12 cross-author
successes**. The corresponding 95% Wilson intervals are 0.0%-13.8% and
0.0%-24.2%. Thus even the upper uncertainty limits are far below the 80%
research target. This is a decisive screening rejection, not an inconclusive
near miss.

![Mean target-author margin versus frozen threshold](../../generated/style_research/style_transfer_experiments/iterations/constrained_rerank_v1/audits/charts/iteration3_mean_margin_vs_threshold.svg)

### Generation reliability

All four fixed generators finished 36/36 samples. Block-local IDs and recursive
split/retry recovered every failed block attempt, so no final row was omitted
from the denominator. Final-sample completeness should not be confused with a
perfect request history: 18 of 636 attempted block segments failed initially and
were recovered.

| Fixed generator | Final samples | Valid blocks | Failed/recovered blocks | Timeout/no response | Invalid response |
| --- | ---: | ---: | ---: | ---: | ---: |
| `aligned_pairs_light` | 36/36 | 159 | 9 | 2 | 7 |
| `aligned_pairs_edit_plan_light` | 36/36 | 152 | 2 | 0 | 2 |
| `microcards_only_light` | 36/36 | 154 | 4 | 2 | 2 |
| `rule_linked_microcards_light` | 36/36 | 153 | 3 | 0 | 3 |

![Block-generation reliability](../../generated/style_research/style_transfer_experiments/iterations/constrained_rerank_v1/audits/charts/iteration3_generation_reliability.svg)

### Aggregate method results

`Target top-1` is the count classified as Feitian Yexiang among 50 authors. It
is reported separately from threshold success because a passage can rank first
with a weak, sub-threshold margin. `Fidelity fail` counts deterministic surface
failures; it is not an English-grounded semantic judgment.

| Method | Arm | Mean margin | Gap to threshold | Mean lift | Positive lift | Target top-1 | Style success | Fidelity fail |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Neutral control | Own | -0.4898 | -0.8447 | +0.0000 | 0/24 | 22/24 | 0/24 | 0/24 |
| Neutral control | Cross | -1.2449 | -1.5998 | +0.0000 | 0/12 | 1/12 | 0/12 | 1/12 |
| Aligned pairs | Own | -0.5042 | -0.8591 | -0.0144 | 11/24 | 21/24 | 0/24 | 2/24 |
| Aligned pairs | Cross | -1.2443 | -1.5992 | +0.0005 | 5/12 | 1/12 | 0/12 | 1/12 |
| Pairs + edit plan | Own | -0.5131 | -0.8680 | -0.0233 | 9/24 | 20/24 | 0/24 | 1/24 |
| Pairs + edit plan | Cross | -1.2313 | -1.5862 | +0.0135 | 6/12 | 1/12 | 0/12 | 1/12 |
| Microcards only | Own | -0.4911 | -0.8460 | -0.0013 | 13/24 | 20/24 | 0/24 | 7/24 |
| Microcards only | Cross | -1.2229 | -1.5778 | +0.0219 | 8/12 | 1/12 | 0/12 | 3/12 |
| Pairs + microcards | Own | -0.5210 | -0.8759 | -0.0312 | 9/24 | 19/24 | 0/24 | 4/24 |
| Pairs + microcards | Cross | -1.2231 | -1.5780 | +0.0218 | 7/12 | 1/12 | 0/12 | 2/12 |
| Candidate rerank | Own | -0.4442 | -0.7991 | +0.0456 | 13/24 | 23/24 | 0/24 | 0/24 |
| Candidate rerank | Cross | -1.2036 | -1.5585 | +0.0413 | 9/12 | 1/12 | 0/12 | 1/12 |

Paired-lift uncertainty was estimated with the preregistered 10,000-resample
book-cluster bootstrap. Every fixed-method interval includes zero. Only the
selection-conditioned rerank excludes zero, which cannot establish independent
method efficacy.

| Method | Own mean lift [95% CI] | Cross mean lift [95% CI] | Interval result |
| --- | --- | --- | --- |
| Neutral control | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | control |
| Aligned pairs | -0.0144 [-0.0405, +0.0133] | +0.0005 [-0.0460, +0.0587] | both include zero |
| Pairs + edit plan | -0.0233 [-0.0670, +0.0171] | +0.0135 [-0.0298, +0.0671] | both include zero |
| Microcards only | -0.0013 [-0.0394, +0.0348] | +0.0219 [-0.0095, +0.0602] | both include zero |
| Pairs + microcards | -0.0312 [-0.0753, +0.0101] | +0.0218 [-0.0096, +0.0619] | both include zero |
| Candidate rerank | +0.0456 [+0.0233, +0.0688] | +0.0413 [+0.0120, +0.0858] | excludes zero; diagnostic only |

All methods had 0 reference-copy failures. Complete row-level results, including
book and author breakdowns, are in each method directory under the frozen
evaluation. The following evaluator-produced figures show the paired changes and
the zero-success endpoint:

![Mean paired margin lift](../../generated/style_research/style_transfer_experiments/iterations/constrained_rerank_v1/evaluations/iteration3_gpt55_v6/screening_initial/charts/mean_paired_lift.svg)

![Deterministic style success](../../generated/style_research/style_transfer_experiments/iterations/constrained_rerank_v1/evaluations/iteration3_gpt55_v6/screening_initial/charts/deterministic_style_success.svg)

### Threshold attainability control

A post-outcome diagnostic scored the hidden original Chinese with the same
frozen meter. This check does not alter the threshold or count toward promotion;
it asks whether the endpoint is attainable by genuine target-author prose.

| Original-text control | Mean margin | Median | Range | Threshold passes | Target top-1 |
| --- | ---: | ---: | --- | ---: | ---: |
| Target-author originals, own arm | +0.9802 | +0.9572 | +0.3138 to +1.6846 | 22/24 (91.7%) | 24/24 |
| Comparison-author originals, cross arm | -1.0941 | -1.1208 | -1.4410 to -0.6708 | 0/12 (0.0%) | 1/12 |

The threshold therefore separates real target-author prose in this screen. The
failure is in generated style movement: neutral own-author reconstructions began
0.8447 margin units below the threshold, and cross-author inputs began 1.5998
units below it.

### Edit magnitude and flow

Iteration 3 deliberately reduced rewrite intensity to repair Iteration 2's
25.0%-29.2% own-author hard-fidelity failure rates. The reduction was too large:
the fixed methods changed only 0.62%-1.27% of characters on average.

| Method | Mean character edit | Paragraphs changed | Output/neutral length | Own lift | Cross lift |
| --- | ---: | ---: | ---: | ---: | ---: |
| Aligned pairs | 1.25% | 53.5% | 0.989 | -0.0144 | +0.0005 |
| Pairs + edit plan | 1.27% | 59.3% | 0.986 | -0.0233 | +0.0135 |
| Microcards only | 0.71% | 28.8% | 0.994 | -0.0013 | +0.0219 |
| Pairs + microcards | 0.62% | 34.4% | 0.993 | -0.0312 | +0.0218 |

![Edit magnitude versus style lift](../../generated/style_research/style_transfer_experiments/iterations/constrained_rerank_v1/audits/charts/iteration3_edit_intensity_vs_lift.svg)

The earlier Iteration 2 medium methods changed approximately 6.76%-9.28% of
characters and 81.3%-85.2% of paragraphs. They achieved own-arm lifts of
+0.149 to +0.194 and cross-arm lifts of +0.135 to +0.147, but failed fidelity.
Iteration 3 overcorrected: it preserved meaning more conservatively by making
mostly local lexical and punctuation-adjacent edits, but nearly eliminated the
style signal. Registered sentence/paragraph-flow deltas are correspondingly
small and do not support a discourse-level transformation claim.

### Method-by-method interpretation

#### Aligned pairs

Four compact neutral-to-target windows did not teach a transferable rewrite at
light intensity. Own-author mean margin moved backward by 0.0144; cross-author
movement was effectively zero (+0.0005). The method retained a high own-arm
top-1 rate because the neutral inputs were already reconstructed from target
books, but none reached the absolute threshold. **H1 is rejected at light
intensity.**

#### Aligned pairs plus constrained edit plan

The private edit plan reduced hard-fidelity failures to 1/24 own and 1/12 cross,
but own-author lift worsened to -0.0233 and cross-author lift remained only
+0.0135. It was operationally safer than the medium Iteration 2 plan, not
stylistically effective. **H2 is not supported as a joint style-and-fidelity
improvement.**

#### Microcards only

Two corpus-derived microcards produced the best fixed-method cross-author lift
(+0.0219) but almost no own-author movement (-0.0013). They also produced the
worst fidelity rates: 7/24 own and 3/12 cross. A two-card, light-edit prompt is
neither sufficient nor safer. **H3 is rejected.** This experiment does not test
a comprehensive style card with broad descriptions and scene-stratified
examples; that remains a distinct method family for the next iteration.

#### Aligned pairs plus microcards

Combining the evidence sources did not create complementarity. It had the most
negative own-author lift (-0.0312), only +0.0218 cross-author lift, and six
surface-fidelity failures. **H4 is rejected at the tested prompt size and light
intensity.**

#### Candidate rerank diagnostic

The reranker selected neutral for 14/36 rows, aligned pairs for 9, microcards for
6, pairs plus microcards for 4, and pairs plus plan for 3. It improved descriptive
mean lift to +0.0456 own and +0.0413 cross, supporting limited row-level
complementarity, but still achieved zero threshold successes. Because the same
style meter helped select and score candidates, this result is
selection-conditioned and cannot be promoted. **H5 receives descriptive support
only, not independent validation.**

### Fidelity and semantic status

Most hard-fidelity failures were unbalanced dialogue quotes. One microcard row
also failed CJK-ratio and dialogue-turn checks. These are deterministic surface
failures, not confirmed changes in English meaning. Conversely, passing them is
not proof of semantic equivalence.

The preregistered independent semantic/readability stage was conditional on a
nonempty provisional shortlist. Because no method qualified, no per-output
semantic judgments were generated or loaded. Its status is **not applicable
after zero shortlist**, rather than “semantic pass.” The separate outcome agent
performed a stratified English-grounded close read for audit purposes, but did
not convert those observations into the frozen per-output judgment endpoint.

### Promotion decision

| Method | Positive lift both arms | Fidelity <=10% both arms | Copy gate | Required style success | Decision |
| --- | --- | --- | --- | --- | --- |
| Aligned pairs | fail | pass | pass | 0/24 own; 0/12 cross | reject |
| Pairs + edit plan | fail | pass | pass | 0/24 own; 0/12 cross | reject |
| Microcards only | fail | fail | pass | 0/24 own; 0/12 cross | reject |
| Pairs + microcards | fail | fail | pass | 0/24 own; 0/12 cross | reject |
| Candidate rerank | descriptive pass | pass | pass | 0/24 own; 0/12 cross | diagnostic only; reject |

The frozen provisional-shortlist artifact records `no_method_qualified` with
zero promoted methods. H6 is not tested at confirmation/final scale because the
study correctly stopped at screening.

## Independent Evaluation Record

### Completed pre-generation reviews

| Review | Separate agent/session | Scope | Verdict | Outcome authority |
| --- | --- | --- | --- | --- |
| Base Iteration 3 methodology audit | Separate academic reviewer | Cohort, leakage, threshold plan, method registry, promotion logic, lock | GO for pre-generation only | None |
| Block-geometry amendment audit | `019f5998-9e28-7982-a250-5ffa851aac86` | Block geometry and unchanged hypotheses | GO for amended execution only | None |
| Compact-reference audit | `019f59a3-56aa-7633-b46a-75dc222b1e67` | Pair/window identity, microcard isolation, inventory, gates | GO for amended execution only | None |
| Cross-run/adjudication audit | `019f59ca-3657-77e2-9d25-6b2088a2892b` | Source/style separation, English adjudication, threshold identity, excluded-output inventory, and lock | GO for amended execution only | None |
| ID-safe block-contract audit | `019f59eb-c62d-7a80-808f-75593b290ac8` | Local-ID remap, recursive recovery, tests, threshold/lock, and 13 gates | Conditional GO; found one stale nested prompt hash | None |
| Prompt-hash recheck | Same separate evaluator, new review artifact | Verify the metadata fix and exact replacement `d086...` lock without reopening outcomes | GO before admissible generation | None |

Every GO is limited to the exact content-addressed lock it audited. It is not a
method result or production approval.

### Completed outcome review

A new separate evaluator (`019f5a2b-6e21-7661-a78a-513cdd3ab25b`)
independently audited the frozen outcome. It recomputed method and arm counts,
hashes, margins, lifts, target shares, deterministic successes, fidelity
failures, and shortlist logic. It then inspected 64 stratified method-arm cells
covering 22 unique sample IDs, every method, both arms, positive and non-positive
lift, closest-to-threshold and worst rows, and all hard-fidelity failures.

The audit explicitly verified:

- 36/36 final rows for each fixed generator and no missing denominator rows;
- 18 failed block attempts were recovered, rather than erased from provenance;
- zero deterministic successes for every fixed method and the rerank diagnostic;
- candidate rerank is selection-conditioned and cannot be promotion evidence;
- surface quote/ratio failures are not mislabeled as semantic failures;
- no independent per-output semantic judgment was required after the empty
  shortlist;
- all 11 registered statistical-fallacy checks were considered.

Its verdict is **NO-GO for promotion, confirmation, final validation, or
production**. The audit characterizes the frozen-artifact decision as solid, but
correctly marks a full fresh rerun and unperformed semantic judging as
unverified. The orchestration thread read and reconciled the full audit; the
canonical artifact is
`audits/iteration3_screening_outcome_audit.md`.

## Threats to Validity

1. **Style meter construct validity.** Exact masked character n-grams capture a
   strong author signal but may emphasize lexical/morphological habits more than
   discourse flow or human-perceived literary style.
2. **Masking is incomplete content control.** `entity_masked_v3` reduces names
   and concentrated topic terms but cannot remove every setting, relationship,
   or book-specific cue.
3. **Proxy English is synthetic.** Reconstructed English approximates production
   direction but not the actual Eternal Gate translator's omissions, syntax, or
   errors.
4. **Small screening set.** The 24/12 screen is suitable for rejection and method
   selection, not a precise final success estimate.
5. **One generation per method/sample.** The study does not estimate model
   sampling variance or robustness to a future model snapshot.
6. **Target final books influenced scorer development.** They are held out from
   transfer development, but some were visible during Stage-1 classifier
   benchmarking; a stronger replication needs fully external target books.
7. **Pair pool scene imbalance.** Interpersonal care and worldbuilding are sparse
   or absent in the 29-pair evidence pool even though the screen includes them.
8. **Compact windows change evidence density.** Pair identity is preserved, but
   selecting the highest-change window may overrepresent visibly edited passages
   compared with unchanged author prose.
9. **Microcard count is small.** Two cards isolate interpretable signals but do
   not represent the full grammar, rhythm, narrator stance, or scene flow of an
   author.
10. **Repeated method iteration risks selection bias.** Content-addressed locks,
    fresh rows, frozen thresholds, unopened confirmation, and one-time final
    validation reduce but do not eliminate researcher degrees of freedom.
11. **LLM judge limitations.** Blind English-grounded critiques can miss subtle
    Chinese awkwardness or accept plausible additions; deterministic bindings and
    paired neutral review prevent neither all false positives nor false negatives.
12. **The 80% endpoint is benchmark-specific.** Passing would justify the frozen
    pipeline for this benchmark and a production pilot, not prove universal
    imitation of the author across genres or models.

## Reproduction

### Validate the prepared package

```bash
uv run python experiments/iteration3/prepare.py validate

uv run python experiments/iteration1/verify_style_analysis_gates.py \
  --experiment-root generated/style_research/style_transfer_experiments/iterations/constrained_rerank_v1 \
  --analysis-lock generated/style_research/style_transfer_experiments/iterations/constrained_rerank_v1/protocols/pre_style_analysis_lock.v1.d086827f31d3fab39e5ef9129ac07422bcf81c5232d2baee6c06fa07e4e0af2a.json
```

### Generate each fixed screening arm

Run these four commands with absolute paths in this environment. Each uses the
same source run, selection, model, lock, block size, and concurrency.

```bash
uv run python experiments/iteration3/run_style_transfer_block_generation.py run \
  --experiment-root generated/style_research/style_transfer_experiments/iterations/constrained_rerank_v1 \
  --sample-set iteration3_proxy_v1 \
  --stage style_transfer \
  --run-id iteration3_gpt55_v6 \
  --input-run-id iteration3_gpt55_v2 \
  --selection-file generated/style_research/style_transfer_experiments/iterations/constrained_rerank_v1/sample_sets/iteration3_proxy_v1.screening_v1_ids.json \
  --method-id aligned_pairs_light \
  --intensity light \
  --analysis-lock generated/style_research/style_transfer_experiments/iterations/constrained_rerank_v1/protocols/pre_style_analysis_lock.v1.d086827f31d3fab39e5ef9129ac07422bcf81c5232d2baee6c06fa07e4e0af2a.json \
  --jobs 3
```

Repeat only `--method-id` for:

```text
aligned_pairs_edit_plan_light
microcards_only_light
rule_linked_microcards_light
```

### Build the registered rerank diagnostic

```bash
uv run python experiments/iteration3/build_candidate_rerank.py \
  --experiment-root generated/style_research/style_transfer_experiments/iterations/constrained_rerank_v1 \
  --sample-set iteration3_proxy_v1 \
  --run-id iteration3_gpt55_v6 \
  --input-run-id iteration3_gpt55_v2 \
  --selection-file generated/style_research/style_transfer_experiments/iterations/constrained_rerank_v1/sample_sets/iteration3_proxy_v1.screening_v1_ids.json \
  --analysis-lock generated/style_research/style_transfer_experiments/iterations/constrained_rerank_v1/protocols/pre_style_analysis_lock.v1.d086827f31d3fab39e5ef9129ac07422bcf81c5232d2baee6c06fa07e4e0af2a.json
```

### Run deterministic screening evaluation

```bash
uv run python experiments/iteration3/evaluate_style_transfer_methods.py evaluate \
  --experiment-root generated/style_research/style_transfer_experiments/iterations/constrained_rerank_v1 \
  --sample-set iteration3_proxy_v1 \
  --run-id iteration3_gpt55_v6 \
  --input-run-id iteration3_gpt55_v2 \
  --selection-file generated/style_research/style_transfer_experiments/iterations/constrained_rerank_v1/sample_sets/iteration3_proxy_v1.screening_v1_ids.json \
  --analysis-lock generated/style_research/style_transfer_experiments/iterations/constrained_rerank_v1/protocols/pre_style_analysis_lock.v1.d086827f31d3fab39e5ef9129ac07422bcf81c5232d2baee6c06fa07e4e0af2a.json \
  --threshold generated/style_research/style_transfer_experiments/iterations/constrained_rerank_v1/calibration/style_meter_threshold.v1.json \
  --scorer-mode load \
  --screening-phase initial \
  --output-dir generated/style_research/style_transfer_experiments/iterations/constrained_rerank_v1/evaluations/iteration3_gpt55_v6/screening_initial
```

The threshold argument is mandatory here. Omitting it would use a stale global
default from an earlier iteration and correctly produce `threshold_status:
invalid`; that run is not an Iteration 3 result.

### Reproduce shortlist decision and report figures

```bash
uv run python experiments/iteration1/select_style_transfer_promotions.py \
  --screening-evaluation generated/style_research/style_transfer_experiments/iterations/constrained_rerank_v1/evaluations/iteration3_gpt55_v6/screening_initial/evaluation_summary.json \
  --analysis-lock generated/style_research/style_transfer_experiments/iterations/constrained_rerank_v1/protocols/pre_style_analysis_lock.v1.d086827f31d3fab39e5ef9129ac07422bcf81c5232d2baee6c06fa07e4e0af2a.json \
  --experiment-root generated/style_research/style_transfer_experiments/iterations/constrained_rerank_v1 \
  --sample-set iteration3_proxy_v1 \
  --provisional

uv run python experiments/iteration3/build_report_artifacts.py \
  --experiment-root generated/style_research/style_transfer_experiments/iterations/constrained_rerank_v1
```

The registered judged-evaluation and confirmation commands were not run because
the provisional shortlist was empty. Neither may substitute a custom sample
subset for the official frozen selection.

## Primary Artifacts

### Design and samples

- Iteration preregistration:
  `generated/style_research/style_transfer_experiments/iterations/constrained_rerank_v1/protocols/iteration3_preregistration.v1.json`
- Evaluation protocol:
  `generated/style_research/style_transfer_experiments/iterations/constrained_rerank_v1/protocols/evaluation_protocol.v1.json`
- Sample summary:
  `generated/style_research/style_transfer_experiments/iterations/constrained_rerank_v1/sample_sets/iteration3_proxy_v1.summary.json`
- Runner manifest, evaluator allocation, and hidden targets:
  `sample_sets/iteration3_proxy_v1.runner_manifest.jsonl`,
  `iteration3_proxy_v1.evaluator_allocation.jsonl`, and
  `iteration3_proxy_v1.hidden_targets.jsonl`
- Frozen calibration, screening, and confirmation selections:
  `sample_sets/iteration3_proxy_v1.screening_calibration_v1_ids.json`,
  `iteration3_proxy_v1.screening_v1_ids.json`, and
  `iteration3_proxy_v1.confirmation_v1_ids.json`

### Methods and scorer

- Method registry and method configs:
  `method_registry/style_methods.v1.json` and `method_registry/methods/`
- Active method-asset lock and content-addressed asset:
  `method_assets/style_transfer_payloads.v1.lock.json` and
  `method_assets/style_transfer_payloads.v1/assets.f9389eaf905e85a2fb61be524f5ee9e019d73888f3f82c9bb7fbe2023549e619.json`
- Frozen scorer:
  `scorers/class_balanced_sgd_hinge_exact_char_ngrams_min_df_20.v1/`
- Calibration scores and threshold:
  `calibration/style_meter_scores.v1.jsonl` and
  `calibration/style_meter_threshold.v1.json`
- Threshold sensitivity:
  `calibration/style_meter_threshold.sensitivity_without_replacement.v1.json`

### Execution provenance

- Frozen source/neutral run:
  `runs/iteration3_proxy_v1/iteration3_gpt55_v2/`
- English-source adjudication:
  `audits/english_source_adjudication.s_b7fb1a5137a9d09e1295f845.v1.json`

### Independent audits

- Base pre-generation audit:
  `audits/pre_style_generation_methodology_audit.md`
- Block-12 amendment audit:
  `audits/pre_style_generation_methodology_audit.block12_amendment.md`
- Compact-reference audit:
  `audits/pre_style_generation_methodology_audit.compact_references.md`
- Cross-run/adjudication audit:
  `audits/pre_style_generation_methodology_audit.cross_run_contract.md`
- ID-safe block-contract audit:
  `audits/pre_style_generation_methodology_audit.id_safe_blocks.md`
- Prompt-hash recheck:
  `audits/pre_style_generation_methodology_audit.id_safe_blocks.prompt_hash_recheck.md`
- Screening outcome audit:
  `audits/iteration3_screening_outcome_audit.md`

### Outcome artifacts

The admissible outcome artifacts are:

- complete fixed-method outputs and ledgers:
  `runs/iteration3_proxy_v1/iteration3_gpt55_v6/`
- deterministic evaluation, per-method reports, CSVs, JSON, and SVGs:
  `evaluations/iteration3_gpt55_v6/screening_initial/`
- frozen empty shortlist decision:
  `promotions/screening_v1.provisional_shortlist.v1.json`
- reproducible report tables:
  `audits/tables/iteration3_screening_method_arm_metrics.csv`,
  `iteration3_edit_intensity_metrics.csv`, and
  `iteration3_generation_reliability.csv`
- post-outcome threshold-attainability control:
  `audits/tables/iteration3_original_control_scores.csv` and
  `iteration3_original_control_summary.csv`
- report figures:
  `audits/charts/iteration3_mean_margin_vs_threshold.svg`,
  `iteration3_edit_intensity_vs_lift.svg`, and
  `iteration3_generation_reliability.svg`

No `independent_judgments`, refinement contract, confirmed promotion, or final
validation artifact exists because the registered gating path stopped at the
empty shortlist.

## Method Sources

- Krishna, Wieting, and Iyyer (2020), STRAP, style transfer through paraphrase
  generation: https://aclanthology.org/2020.emnlp-main.55/
- Patel, Andrews, and Callison-Burch (2024 revision), Styll low-resource
  authorship transfer with neutral-to-target in-context pairs:
  https://arxiv.org/abs/2212.08986
- Suzgun, Melas-Kyriazi, and Jurafsky (2022), Prompt-and-Rerank:
  https://aclanthology.org/2022.emnlp-main.141/
- Zhu et al. (2023), StoryTrans for Chinese and English long-story author
  transfer: https://aclanthology.org/2023.acl-long.827/
- Tao et al. (2024/2025), CAT-LLM Chinese text style definitions:
  https://arxiv.org/abs/2401.05707
- Horvitz et al. (2024), TinyStyler authorship embeddings and reranking:
  https://aclanthology.org/2024.findings-emnlp.781/
- Liu and May (2025), iterative preference optimization with pseudo-parallel
  data: https://aclanthology.org/2025.naacl-long.135/
- Ostheimer et al. (2024), LLM-based text style-transfer evaluation:
  https://aclanthology.org/2024.lrec-main.1373/
- Mir et al. (2019), content preservation, style strength, and fluency as
  separate evaluation dimensions: https://aclanthology.org/N19-1049/

## Conclusion and Iteration 4 Decision

Iteration 3 is complete and negative. Compact aligned pairs, a constrained edit
plan, two statistical microcards, and their combination were all tested at light
rewrite intensity on 24 own-author and 12 cross-author rows. Every method
finished generation, every row was retained, and every fixed method scored 0%
deterministic style success in both arms. The separate outcome evaluator issued
NO-GO. No method is approved for confirmation, final validation, Eternal Gate
pass 2, or production.

The central diagnosis is not an unattainable style threshold: 91.7% of hidden
target-author originals passed it. The tested methods changed too little text to
bridge the neutral-to-target gap. Iteration 2 demonstrated that stronger aligned
rewrites can move the meter but damage fidelity; Iteration 3 reduced edits until
both style movement and discourse-level transformation nearly disappeared.

Iteration 4 must therefore test stronger **full-regeneration** methods while
making fidelity enforcement explicit and English-grounded. The next method
study should compare:

1. full regeneration from English plus aligned neutral-to-target pairs;
2. a comprehensive corpus-derived style card with descriptions,
   scene-stratified examples, counterexamples, and semantic invariants;
3. aligned pairs plus the comprehensive style card;
4. multi-candidate generation with a selector independent of the frozen outcome
   meter.

Deterministic quote/dialogue repair should run before scoring, and a small blind
semantic/readability pilot should reject unsafe high-intensity prompts before a
new official screen. Iteration 3 screening rows are now consumed and cannot be
reused for Iteration 4 method selection. Confirmation and the four reserved
final books remain unopened until a newly frozen design earns promotion.
