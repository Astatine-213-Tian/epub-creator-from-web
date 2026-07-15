# Eternal Gate Style-Transfer Research: Iteration 4

> **Current validity note:** the target-margin and threshold fields in this frozen
> report came from the retired pre-normalization class-balanced scorer. They must not
> be compared with the current normalized benchmark. The source-fidelity, readability,
> model-output, and direct-reader evidence remain part of the method decision.

## Executive Status

> **ITERATION 4 DEVELOPMENT COMPLETE - NO-GO TO SCREENING.**
>
> Five full-regeneration methods and one blind selector were evaluated on the
> frozen 16-row development pilot. The best own-author result was 2/8 (25%);
> every cross-author result was 0/8. No method reached the registered 80%
> criterion, and a post-outcome construct audit found material topic/entity
> contamination in the frozen character-ngram meter. Screening, confirmation,
> final validation, and Eternal Gate production use remain unopened.

This is the sole canonical human-readable report for Iteration 4. It contains
the design, data lineage, development decisions, frozen method definitions,
results, graphs, separate-agent evaluations, and final conclusion.
Internal run directories and audit artifacts are reproducibility evidence, not
additional iteration reports.

| Stage | Evidence | Final Iteration 4 status |
| --- | --- | --- |
| Literature and prior-iteration synthesis | Chinese long-text transfer, authorship transfer, candidate selection, multidimensional evaluation | complete |
| Fresh cohort allocation | 214 rows: 16 development + 32 calibration + 44 screening + 122 confirmation; final remains unmaterialized | complete and hash-addressed |
| Corpus-grounded style definition | 15 retained features, 7 validated dimensions, and 6 masked scene examples from distinct train books | built and hash-addressed |
| English semantic source | 92 isolated source rows plus independent QA and monotonic repair | 92/92 approved after five shrinking repair rounds and one bilingual adjudication |
| Neutral Chinese baseline | Production-compatible English-to-Chinese prompt | 92/92 generated and validated |
| Generated-domain calibration | 32 fresh target rows | complete; threshold `0.25251173973083496` |
| Development generation | Five fixed methods x 16 rows | 80/80 valid outputs with `gpt-5.5` |
| Blind candidate selector | 16 rows, hidden method labels, hidden originals, hidden style meter | 16/16 decisions with `gpt-5.4` high |
| Development deterministic evaluation | Style, fidelity, copying, and flow diagnostics | complete as post-lock exploratory evidence; best own 25%, every cross 0% |
| Registered semantic/readability endpoint | Separate blind judgment against English | not executable under the frozen development admission contract; not measured |
| Executable downstream rehearsal | Disposable synthetic data through the real admission, asset, scoring, calibration, lock, selector, and evaluation code | passed; readiness evidence only, with no real Iteration 4 outcome written |
| Pre-generation methodology audit | Separate agent, exact content-addressed lock | GO on the final pre-style lock |
| Protocol-path audit | Separate agent checks whether registered development judgments can run | NO supported command; semantic claim unavailable |
| Style-meter construct audit | Top-300 positive-feature inspection with book/author dispersion | concern: 37.7% of audited positive weight is risk-flagged |
| Independent outcome audit | New separate agent after outcomes freeze | NO-GO; no method advances |
| Official screening | Frozen style, fidelity, semantic, readability, and reliability endpoints | not opened |
| Confirmation and final validation | Conditional on registered promotion gates | not opened |
| Production decision | Eternal Gate pass 2 | not authorized |

The earlier quota pause was operationally misdiagnosed. A historical usage
manifest was treated as if it described the current weekly allowance. Live
model calls remained available, the source run resumed, and all development
work listed above completed. The stale quota record remains only as provenance
for the interrupted attempt; it is not the final experiment status.

## Research Question

Can full Chinese regeneration from the authoritative English semantic source,
guided by train-only evidence of Feitian Yexiang's recurring style, reach at
least 80% frozen author-style success while remaining semantically and
readability-noninferior to the neutral Chinese baseline?

Iteration 4 tests whether stronger generation solves the under-editing failure
of Iteration 3 without repeating the semantic and surface-fidelity failures of
Iteration 2.

## Why a New Iteration Is Required

| Prior evidence | Result | Consequence for Iteration 4 |
| --- | --- | --- |
| Iteration 1: generic prompts, cards, sentence-flow rules, examples, and close-reading descriptions | Every tested method failed the frozen deterministic success endpoint; examples and descriptions were applied as medium recasts without comprehensive cross-book validation. | A style description or example set is not sufficient by itself; evidence construction and generation mode must both change. |
| Iteration 2: aligned neutral-to-target pairs | Mean target-author margin improved in both benchmark arms, but own-author hard-fidelity failures reached 25.0%-29.2%. | Retain aligned transformation evidence, but make English content planning and fidelity evaluation first-class. |
| Iteration 3: compact pairs, edit plan, two microcards, and combinations | All fixed methods scored 0/24 own and 0/12 cross successes; average character edits were only 0.62%-1.27%. | Light local editing is rejected. Regenerate complete paragraphs and test the style/fidelity tradeoff directly. |
| Iteration 3 original-text control | 22/24 hidden target originals passed the same threshold. | The classifier recognizes in-domain target originals, but this does not establish that its threshold is content-independent. Iteration 4 therefore treats it as a proxy, not a complete style definition. |

The Iteration 4 style-definition method is distinct from earlier card arms. It
uses a comprehensive word-, syntax-, sentence-, paragraph-, dialogue-, and
discourse-level definition; requires directional recurrence across target train
books; includes masked examples from distinct books and explicit
counterexamples; and drives full regeneration rather than optional local edits.

## Literature-Grounded Design

The design follows five findings from primary research:

1. CAT-LLM defines Chinese long-text style at both word and sentence levels and
   supplies the definition to an LLM, motivating a comprehensive corpus-grounded
   style definition rather than two isolated rules:
   https://arxiv.org/abs/2401.05707
2. StoryTrans treats long-story author style as a discourse-level problem and
   uses explicit content preservation, motivating paragraph-flow evidence and an
   English-grounded content ledger:
   https://aclanthology.org/2023.acl-long.827/
3. Styll and TinyStyler show the utility of few-shot authorship evidence, while
   TinyStyler and Prompt-and-Rerank motivate multiple candidates and explicit
   selection:
   https://arxiv.org/abs/2212.08986,
   https://aclanthology.org/2024.findings-emnlp.781/, and
   https://aclanthology.org/2022.emnlp-main.141/
4. Large-scale authorship-imitation evidence shows that demonstrations alone
   often miss subtle implicit style, so Iteration 4 does not treat example-only
   prompting as sufficient evidence:
   https://aclanthology.org/2025.findings-emnlp.532/
5. Text-style-transfer evaluation must separate style strength, content
   preservation, and naturalness; a single classifier score is not a complete
   quality claim:
   https://aclanthology.org/N19-1049/ and
   https://aclanthology.org/2024.lrec-main.1373/

These papers motivate the method families. They do not establish that any
family will work on this corpus; that is the empirical question.

## Data and Leakage Controls

### Corpus roles

| Corpus role | Source | Allowed use |
| --- | --- | --- |
| Target style discovery | 29 `非天夜翔` train books, `entity_masked_v3` | Statistics, masked close reading, aligned examples, style-definition construction |
| Comparison style discovery | Train books from 49 other authors, `entity_masked_v3` | Contrastive statistics and counterexamples only |
| Development pilot | Fresh target development chunks plus comparison-author train chunks | Prompt safety and method feasibility; never official efficacy evidence |
| Calibration | Fresh chunks from eight target development books | Freeze the generated-domain style threshold only |
| Screening | Fresh target development chunks plus comparison dev chunks from authors absent from the other execution stages | Official method-family selection |
| Confirmation | Fresh target development chunks plus comparison dev chunks from authors absent from pilot and screening | Replication after promotion only |
| Final validation | Four reserved target test books | One-time validation after confirmation only |
| Eternal Gate | English application text | Excluded from method selection and threshold construction |

### Frozen pre-generation geometry

| Partition | Own-author rows | Cross-author rows | Total | Cross-author policy |
| --- | ---: | ---: | ---: | --- |
| Development pilot | 8 | 8 | 16 | Train-book inputs; authors excluded from official cross partitions |
| Calibration | 32 | 0 | 32 | Four fresh rows per target development book |
| Screening | 24 | 20 | 44 | One dev-book row from each of 20 authors |
| Confirmation | 80 | 42 | 122 | Two dev-book rows from each of 21 different authors |
| Final validation | 80 | 0 | 80 | Twenty rows from each reserved target test book |

All 214 Iteration 4 development, calibration, screening, and confirmation chunks
are at least five chunks from every earlier transfer allocation and from one
another within a book. Their overlap with earlier allocated chunk IDs is zero.
The cross-author partitions contain 8 development, 20 screening, and 21
confirmation authors with no author overlap. Original Chinese evaluation targets
are exposed one sample at a time only to the one-way English source-construction
and QA stages. They remain hidden from neutral translation, style generation,
candidate selection, and all model-visible method evaluation.

The 49 comparison authors are stage-disjoint for pilot, screening, and
confirmation, but they are not unseen during evidence construction: train books
from all 49 contribute to the aggregate contrastive style definition. Cross-arm
claims therefore concern transfer to held-out dev-book chunks and execution-stage
author partitions, not generalization to wholly unseen authors.

### Development benchmark roster

The development pilot contains one chunk from each listed book. The own arm asks
whether a neutral reconstruction of target-author content can be restored toward
the target style. The cross arm is the harder test: it asks whether content from
another author can acquire target style without importing target names, lore, or
events.

| Arm | Author | Book | Chunk | Source role |
| --- | --- | --- | --- | --- |
| Own | 非天夜翔 | 天宝伏妖录 | `0242` | proxy-transfer |
| Own | 非天夜翔 | 山有木兮 | `0421` | dev |
| Own | 非天夜翔 | 清平梦华录 | `0335` | proxy-transfer |
| Own | 非天夜翔 | 相见欢 | `0367` | dev |
| Own | 非天夜翔 | 万物风华录 | `0516` | proxy-transfer |
| Own | 非天夜翔 | 骑士之歌 | `0228` | proxy-transfer |
| Own | 非天夜翔 | 图灵密码 | `0203` | dev |
| Own | 非天夜翔 | 乱世为王 | `0057` | dev |
| Cross | 望三山 | 我靠美颜稳住天下 | `0300` | train |
| Cross | 骑鲸南去 | 反派他过分美丽[穿书] | `0253` | train |
| Cross | 稚楚 | 可爱过敏原 | `0287` | train |
| Cross | 西西特 | 我有一个秘密 | `0037` | train |
| Cross | 梦溪石 | 千秋 | `0030` | train |
| Cross | 吕天逸 | 禁止犯规 | `0053` | train |
| Cross | 墨西柯 | 怎么还不哄我[娱乐圈] | `0034` | train |
| Cross | 比卡比 | 两个皇帝怎么谈恋爱 | `0015` | train |

This is a feasibility sample, not an accuracy-estimation sample. With eight
rows per arm, 80% cannot be observed exactly: the prospective operational gate
was 7/8 (87.5%). The official 80% claim was reserved for the much larger
confirmation and final partitions.

The same production-compatible neutral English-to-Chinese prompt used by the
eventual Eternal Gate pass 1 remains the neutral baseline. Using a different
neutral prompt here would invalidate transfer to production.

### English source and neutral construction

Source construction is limited to the 16 development, 32 calibration, and 44
screening rows (92 total). Confirmation remains unopened. A one-way model call
reconstructs English from one isolated original Chinese sample; a separate QA
call checks every English paragraph against that same source. Only approved
English can enter neutral translation.

An English repair round is not a new style-transfer method and does not have a
different research purpose. Every round uses the same frozen repair prompt on
only the rows rejected by the preceding QA stage, then runs the same independent
QA again. The subset therefore shrinks monotonically; round numbers record
provenance rather than increasingly aggressive behavior. Repair stops when all
rows are approved or an unresolved row is independently adjudicated and
reported. The table below lists the exact input, pass, and remaining counts for
every round rather than referring to an unexplained fixed number of rounds.

The source pipeline completed. Each repair round used the same frozen repair
instruction and only the rows rejected by the immediately preceding QA pass.
The round number therefore records lineage; it does not mean that the model was
given a more aggressive or differently optimized instruction.

| Source stage | Rows entering stage | Approved | Rejected / carried forward |
| --- | ---: | ---: | ---: |
| Initial independent QA | 92 | 34 | 58 |
| Repair round 1 QA | 58 | 31 | 27 |
| Repair round 2 QA | 27 | 17 | 10 |
| Repair round 3 QA | 10 | 7 | 3 |
| Repair round 4 QA | 3 | 2 | 1 |
| Repair round 5 QA | 1 | 0 | 1 |
| Independent bilingual adjudication | 1 | 1 | 0 |
| **Final** | **92** | **92** | **0** |

The last row failed repeated automated QA because one insult was rendered too
literally. A separate bilingual adjudicator reviewed the original Chinese and
English together, required the repair `Cerebral-palsy trash doesn't count as
human.`, and then issued PASS. This was source-equivalence repair, not Chinese
style generation.

Neutral translation then generated and validated 92/92 rows with the same
neutral English-to-Chinese prompt intended for Eternal Gate pass 1. Only the 16
development rows and 32 calibration rows were exposed to the later stages;
screening English/neutral artifacts exist but their method outputs and outcome
scores were never opened. Confirmation and final source construction remain
unmaterialized.

The historical interruption manifest remains at
`runs/iteration4_proxy_v1/iteration4_source_gpt54_official/resume_manifests/english_source_repair_qa.round_01.usage_limit.json`
as an audit trail. It correctly records that one attempt stopped, but its reset
message was later mistaken for current account capacity. That stale operational
record no longer describes the experiment's completion state.

## Corpus-Grounded Style Definition

The definition is built before development style outputs and uses only masked
train text.

1. Compute target-versus-comparison contrasts for lexical-function, punctuation,
   sentence flow, paragraph flow, dialogue, speech attribution, and simple
   syntactic-order proxies.
2. Split the 29 target train books into discovery and validation books. Retain a
   tendency only when it has adequate support, the expected direction recurs
   across books, and it remains present in validation books.
3. Ask an LLM to close-read an anonymized masked source packet stratified by book
   and scene. Every proposed claim must cite registered feature IDs and remain
   grounded in that frozen packet; free-form author stereotypes are rejected.
   The inherited close-reading result does not contain claim-level passage IDs,
   so it is supporting synthesis rather than passage-level traceable evidence.
4. Retain only actionable dimensions supported by the validation gates. Each
   dimension contains a description,
   source trigger, target direction/range, semantic guardrail, failure mode, and
   at least one masked example or counterexample.
5. Select examples from distinct target train books and scene strata. No
   development, confirmation, final, or Eternal Gate text may appear.
6. Run a separate evidence audit before the definition is admitted to an
   official style-generation lock.

The current asset retains 15 features and groups them into seven dimensions:

| Dimension | Validated signals |
| --- | --- |
| Dialogue-led turn architecture | Dialogue punctuation and plain speech tags |
| Local reaction timing | Source-supported micro-reactions |
| Compact laughter pivots | Source-supported laughter tags |
| Short modular sentence flow | Short-sentence share, mean sentence length, comma sequencing |
| Light explicit scaffolding | Connective, modal/aspect, preposition, particle, and deictic rates |
| Direct interrogative pressure | Question-tag rate |
| Source-licensed marked punctuation | Colon and exclamation rates |

Pause/silence was proposed by close reading but was not retained as an
independent dimension: `dialogue.silence_beats` reached only 0.50 validation-book
recurrence, while `punctuation.ellipsis` had insufficient discovery contrast
(`abs(z)=0.3945`). This is evidence-gated omission, not manual preference.
Six masked examples cover action, dialogue, reflection, interpersonal care,
travel, and worldbuilding, with one distinct target train book per example.

The definition is guidance, not a quota. The generator must not invent
questions, exclamations, gestures, dialogue, or sentence breaks merely to match a
corpus rate.

## Registered Methods

Every generated method receives byte-identical authoritative English and neutral
Chinese. All generated arms regenerate every paragraph; none is a light edit.

| Method | Intervention | Purpose | Promotable? |
| --- | --- | --- | --- |
| `neutral_only` | Score neutral Chinese unchanged. | Paired baseline. | No |
| `generic_full_regeneration` | Regenerate from English plus neutral terminology/content anchors, with no target-author evidence. | Measure style movement caused by full regeneration and model defaults alone. | No |
| `aligned_pairs_full_regeneration` | Add retrieved masked neutral-to-target transformations from distinct train books. | Test direct transformation evidence at adequate rewrite strength. | Yes |
| `style_definition_examples_full_regeneration` | Add the validated comprehensive style definition, masked scene examples, and counterexamples; no aligned pairs. | Test description-plus-example transfer as a complete method. | Yes |
| `aligned_pairs_style_definition_full_regeneration` | Combine aligned transformations with the validated definition and examples. | Test whether explicit interpretation helps pair generalization. | Yes |
| `content_plan_combined_full_regeneration` | Build an English-grounded fact/event/speaker ledger, then regenerate using aligned pairs plus the style definition. | Test whether explicit content planning allows stronger style without semantic drift. | Yes |
| `independent_candidate_selector` | Blindly select among registered fixed candidates using English fidelity, naturalness, and style-definition adherence; method labels and the frozen style meter are unavailable. | Test candidate complementarity without selecting on the outcome metric. | Yes, only if frozen before screening |

The 29 aligned pairs were inherited from an earlier `entity_masked_v2` study but
are not reused in that view. Iteration 4 binds each pair to its original train
chunk, aligns every target paragraph exactly to `entity_masked_v3`, applies the
current book mask plan to the neutral side, resolves legacy placeholders using
the aligned target mask (with a pair-modal mask fallback when the neutral recast
mentions an entity more often), and rejects any unresolved paragraph or
placeholder. The rebuilt asset records this conversion and its source hashes.

During this rebuild, a preprocessing defect was found before any Iteration 4
style output existed: already masked `<NUM>` and `<LATIN>` placeholders could be
consumed by the Latin-token regular expression and normalized incorrectly.
Retrieval normalization, evaluator masking, and pair conversion now protect
existing placeholders before masking new surfaces. The active 29-pair asset
contains four `<NUM>` and 49 `<LATIN>` occurrences, no unresolved `<TERM>`, and
is covered by the combined contract tests.

### Full-regeneration contract

The English source is authoritative. Neutral Chinese supplies terminology,
entity choices, and a content-preservation anchor, but its syntax is not a
template. Each output must preserve paragraph IDs and order, facts, event order,
causality, negation, modality, intensity, numbers, Latin tokens, speaker
attribution, and paragraph-level dialogue topology. Reference text may teach
transformations but may not supply lore, imagery, wording, or events.

The content-plan method must record a compact ledger per paragraph containing
events/states, participants, speaker, negation/modality, temporal/causal links,
and protected tokens. The ledger is derived from English before target-style
evidence is applied.

### Deterministic surface repair

A registered postprocessor may repair only mechanically detectable Chinese quote
pairing and spacing around existing dialogue marks. It may not add, remove, or
rephrase lexical content; change paragraph boundaries; infer a speaker; or alter
punctuation type for style. The development evaluation did not use a lexical
repair to hide failures; unbalanced quotes remained explicit fidelity errors.

## Development Pilot

The 16-row pilot can compare prompt safety, schema reliability, raw style-margin
movement, deterministic fidelity, and blind English-grounded semantic/readability
judgments. It cannot establish accuracy or promote a method. Pilot changes are
not allowed after the pre-style analysis lock: the pilot is a frozen feasibility
gate, not an adaptive prompt-tuning set. A failed method is rejected or moved to
a newly registered iteration rather than silently changed inside Iteration 4.
Official screening rows, labels, original targets, and scores stay unopened
during the pilot.

The frozen pilot rejection conditions were:

- missing output or paragraph-ID/order failure;
- more than 10% hard deterministic fidelity failure in either pilot arm;
- high-severity English-grounded semantic error;
- recurrent invented dialogue, gesture, emotion, causality, or imagery;
- severe readability regression against neutral;
- no positive mean target-margin movement for a promotable family.

## Frozen Evaluation

### Style endpoint

The historical Stage-1 class-balanced exact character 2-4 gram SGD hinge model was the
frozen author-style meter. Iteration 4 calibrated an absolute generated-domain
threshold on its 32 calibration rows before style generation. The selected
threshold and allocation hashes were bound into the analysis lock.

The copied historical scorer was independently reloaded and replayed against its
then-current benchmark before Iteration 4 generation, confirming that the frozen
decision used the intended artifact. That benchmark and scorer have now been
retired after corpus cleanup and punctuation normalization; their numerical
performance is deliberately not carried into the current result. See the current
authorship-meter report for the replacement benchmark.

A deterministic style success requires all of:

- target-author margin at or above the fresh threshold;
- target author ranked in the top five;
- strictly positive paired margin lift over neutral;
- no artifact, paragraph, fidelity, or reference-copy failure.

Style success is a proxy endpoint, not proof of human-identical authorship.

### Multidimensional endpoints

| Dimension | Evidence |
| --- | --- |
| Style strength | Frozen classifier success, margin, rank, paired lift, sentence/paragraph-flow deltas |
| Content preservation | Deterministic protected-token/dialogue checks plus blind English-grounded LLM judgment |
| Naturalness/readability | Blind candidate-versus-neutral LLM judgment with error labels and severity |
| Generalization | Own and cross arms reported separately; cross screening and confirmation authors are disjoint |
| Reliability | Missing outputs fail; block/request failures and recovery are reported without denominator deletion |
| Copy safety | No eight-character reference copying; copied candidates fail |

### Screening promotion gate

A promotable fixed method or the independently registered selector must satisfy:

- at least 80% deterministic style success in both screening arms;
- positive paired mean lift in both arms with cluster-bootstrap intervals;
- no more than 10% hard-fidelity failures in either arm;
- zero reference-copy failures;
- completed blind semantic/readability evaluation with no high-severity
  candidate regression;
- complete rows and matching frozen hashes.

Screening can reject or select a family. It cannot establish the final 80% claim.

### Confirmation and final claim

Confirmation requires at least 80% success separately on 80 own-author rows and
42 cross-author rows, Wilson 95% lower bounds of at least 70%, cluster-bootstrap
lower bounds of at least 70%, positive lift intervals, and semantic/readability
noninferiority. Only a confirmation winner may open the 80-row final target-book
set. The same frozen method must then reach at least 80% with a target-book
cluster-bootstrap lower bound of at least 70%.

## Statistical Controls

- Report own and cross arms separately; do not hide a weak arm in a pooled mean.
- Cluster uncertainty by source book for target rows and by author for cross rows.
- Count missing or invalid outputs as failures.
- Freeze method roster, threshold, selector, sample IDs, prompts, schemas,
  evidence assets, postprocessor, and analysis code before official generation.
- Do not tune on screening labels, target originals, style scores, or semantic
  judgments.
- The independent selector cannot access the frozen outcome style meter.
- Multiple fixed families are reported individually; selection-adjusted claims
  require untouched confirmation and final replication.

## Independent Evaluation Requirement

Iteration 4 requires two separate-agent reviews:

1. a methodology auditor must issue GO for the exact pre-generation lock; and
2. after outcomes freeze, a new evaluator must independently recompute metrics,
   inspect stratified positive/negative/failure cases against English, audit
   statistical fallacies and leakage, and issue GO/NO-GO.

The orchestration agent reconciled those evaluations into this report without
silently overriding them. Supporting audit Markdown stays under generated audit
artifacts; this file remains the only final iteration report.

### Executable downstream rehearsal (nonresearch)

Before resuming paid model calls, the complete downstream path was rehearsed in
a disposable copy of the Iteration 4 experiment. The rehearsal used
deterministic fake remote responses but the real registered runner, frozen asset
loader, classifier, calibration code, analysis-lock builder, selector builder,
and evaluator. It tested whether the frozen protocol could actually be executed;
it did **not** test any transfer method's efficacy.

| Rehearsed stage | Synthetic rows accepted by the real code | Contract exercised | Result |
| --- | ---: | --- | --- |
| English semantic source | 92 | source schema, provenance, and complete registered selection | pass |
| English-source QA | 92 | QA schema and one-to-one source binding | pass |
| Neutral translation | 92 | production-compatible neutral prompt and source lineage | pass |
| Development generation | 16 rows for each of 5 fixed methods | registered development admission, frozen method assets, and output provenance | 80/80 pass |
| Screening generation | 44 rows for each of 5 fixed methods | registered screening admission, frozen assets, and output provenance | 220/220 pass |
| Generated-domain calibration | 32 | frozen style-meter loading and threshold-calibration input contract | pass |
| Analysis lock | 22 source records and 51 bound artifacts | exact pre-style artifact inventory and lock validation | pass |
| Development selector | 16 | development-only candidate admission and selection-specific artifact paths | pass |
| Screening selector | 44 | preregistered candidate admission and selection-specific artifact paths | pass |
| Evaluator | 1 development and 1 screening summary | stage-specific inputs and summary construction | pass |

The rehearsal found five deterministic defects before they could corrupt a real
run. Each was repaired and covered by executable tests:

| Defect found | Risk | Repair and regression coverage |
| --- | --- | --- |
| No development-stage admission branch in the shared generator | The blind pilot could not run through the registered execution path. | Added an explicit frozen development roster limited to the five fixed methods at strong intensity. |
| Iteration 4 asset schema was incompatible with the shared loader | Real method calls would fail or bypass the intended frozen payload bundle. | Added a strict Iteration 4 adapter and hash-bound source projections for the style prompt and output schema. |
| Method configs used `id` while the evaluator requires `method_id` | Valid outputs could not be evaluated consistently. | Standardized method configs on `method_id` and added registry/config identity and hash checks. |
| Registered selection labels were accepted without exact path, hash, and row-count validation | A substituted or partial cohort could masquerade as the registered cohort. | Centralized exact selection validation for development, screening, confirmation, and final stages. |
| Selector admission and output paths were not sufficiently stage-specific | Later stages could run without promotion evidence, and one selection could overwrite another's artifacts. | Added stage gates, required confirmation/final evidence, and selection-specific run-config and summary paths. |

The rehearsal produced no research accuracy, style-success rate, semantic
judgment, promotion decision, or final claim. Its temporary workspace was
deleted on exit, and it wrote no real Iteration 4 method output. The current
regression suite contains 18 passing contract tests.

The readiness result is reproducible with:

```bash
uv run python experiments/iteration4/rehearse_downstream_pipeline.py \
  --bootstrap-resamples 40
uv run python -m unittest discover -s tests/experiments/iteration4 -p 'test_*.py'
uv run python experiments/iteration4/prepare.py validate
```

### Frozen execution snapshot

| Artifact | SHA-256 | State |
| --- | --- | --- |
| Iteration preregistration | `c49d8cef17b893bdc2456fad1da3a6c084db3f8a0b949b9be458882c3caa45e7` | Frozen before style generation |
| Evaluation protocol | `50e265052e42ba07ffe717296d9f02efd0382314444233fcb5e91e1ca804ac7a` | Stage admission, endpoints, and promotion gates |
| Style-definition lock | `deb004d4372ddf4239dfab19a5321e5c755fc1151eeef6f7c73132c93610f732` | 15 features, 7 dimensions, 6 masked examples |
| Method-payload lock | `a33b99de2cd7bb7dc3f5caecbe9d41e6ef82b0be19e7645fd1ca4162521706d1` | Content hash `497a0db8919ccc9cfd97f6a729ff0528953d2d97477e83e07e55c42e4bb994d8` |
| Method registry | `95754844a46397e01607aff594e9101dc15461b8b32a7195b09e6a7451476733` | Seven methods, including control and selector |
| Style-meter replay | `1e67c086c48f3e3e22fd66794e3ebd9a975c889a516b580e58042896c9b459b5` | Exact benchmark replay passed |
| Calibration threshold | `0b7213394be7221630e98a968e926f929226c48458056d10a2e67e30c8b604b5` | 32 rows; threshold `0.25251173973083496` |
| Pre-style analysis lock | `8f3711d73e59461dd8da8813a193b340035c691d43bed56c95071f6f5c3123f9` | Content hash `7abfe5a9858a21a271df83d193e626dd910f38ad68095ef26dbb879869e0159e` |
| Development selection | `20abe94a06bd0b6712114d66f6c387303078762d5484f6eaf8f82955ccf46719` | Exact 16-row pilot |
| Style model config | `87d5273e850f3a5bc56d63c67341ca0640692b14f37a08786ea9f6f8ed163595` | `gpt-5.5`, strong regeneration |

After the rehearsal repairs, a fresh separate methodology auditor, **Avicenna**
(`019f5cc5-b657-7300-9ceb-6d340f9f2572`), rechecked the exact preregistration,
registry, threshold, source/neutral readiness, and analysis-lock hashes above.
It issued **GO** for the development generation snapshot. That verdict
authorized the pilot only; it did not authorize screening automatically.

## Results

### Development execution

The real development run used `gpt-5.5` for all five fixed generation methods.
Each method received the same 16 English sources and neutral Chinese baselines.
All methods completed all rows; there was no denominator deletion.

| Output family | Expected | Complete | Missing / invalid |
| --- | ---: | ---: | ---: |
| Generic full regeneration | 16 | 16 | 0 |
| Aligned pairs | 16 | 16 | 0 |
| Style definition + examples | 16 | 16 | 0 |
| Aligned pairs + style definition | 16 | 16 | 0 |
| Content plan + combined evidence | 16 | 16 | 0 |
| **Fixed-method total** | **80** | **80** | **0** |
| Independent selector | 16 | 16 | 0 |

The independent selector used `gpt-5.4` at high reasoning. It saw the English
source, neutral baseline, anonymous eligible candidates, and the frozen style
definition. It did not see method labels, original Chinese targets, or style-
meter scores. Generic regeneration was a nonpromotable control and was not in
the selector roster. Candidates that failed the deterministic hard gate were
removed before selection.

| Selected candidate family | Selections | Share |
| --- | ---: | ---: |
| Neutral baseline | 5 | 31.25% |
| Aligned pairs | 4 | 25.00% |
| Content plan + combined evidence | 3 | 18.75% |
| Style definition + examples | 2 | 12.50% |
| Aligned pairs + style definition | 2 | 12.50% |

Selecting neutral on five rows is meaningful: on nearly one-third of the pilot,
the selector judged every eligible style rewrite worse than leaving the neutral
translation unchanged.

### Deterministic style-proxy result

The following are development diagnostics, not official screening estimates.
`Success` requires target margin >= `0.252512`, target rank <= 5, positive lift
over the paired neutral text, and all deterministic fidelity/copy gates. `Mean
margin` is the output's absolute target-class margin; `mean lift` is its paired
change from the neutral baseline. Hard-fidelity counts are failed rows, not
individual error instances.

![Deterministic success by method and arm](../../generated/style_research/style_transfer_experiments/iterations/full_regeneration_v1/runs/iteration4_proxy_v1/iteration4_style_gpt55_v1/evaluations/development_v1_amended/charts/deterministic_style_success.svg)

![Mean paired style-margin lift by method and arm](../../generated/style_research/style_transfer_experiments/iterations/full_regeneration_v1/runs/iteration4_proxy_v1/iteration4_style_gpt55_v1/evaluations/development_v1_amended/charts/mean_paired_lift.svg)

| Method | Own success | Own margin | Own lift | Own fidelity fails | Cross success | Cross margin | Cross lift | Cross fidelity fails |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Neutral control | 0/8 (0%) | -0.23579 | 0 | 0 | 0/8 (0%) | -1.34652 | 0 | 0 |
| Generic regeneration | 0/8 (0%) | -0.32166 | -0.08587 | 2 | 0/8 (0%) | -1.34695 | -0.00043 | 2 |
| Aligned pairs | **2/8 (25%)** | -0.07615 | +0.15964 | 3 | 0/8 (0%) | -1.23350 | +0.11302 | 0 |
| Style definition + examples | 0/8 (0%) | **-0.04781** | **+0.18798** | 2 | 0/8 (0%) | -1.26451 | +0.08202 | 1 |
| Aligned pairs + definition | 1/8 (12.5%) | -0.08195 | +0.15384 | 2 | 0/8 (0%) | -1.22748 | +0.11905 | 1 |
| Content plan + combined | 1/8 (12.5%) | -0.07510 | +0.16070 | 3 | 0/8 (0%) | **-1.21694** | **+0.12959** | 3 |
| Independent selector | **2/8 (25%)** | -0.09368 | +0.14211 | **0** | 0/8 (0%) | -1.24144 | +0.10509 | **0** |

The best observed own-arm rate was 25%, whose Wilson 95% interval is 7.1%-59.1%.
Every cross result was 0%; the 0/8 Wilson upper bound is 32.4%. The registered
pilot gate was 7/8 in each arm. The result is therefore not a near miss.

Positive mean lift shows that four target-informed fixed methods moved in the
intended direction, but lift was not consistent enough and the absolute gap was
large. Positive-lift row counts were 6/8 own and 6/8 cross for aligned pairs,
7/8 and 6/8 for style definition, 7/8 and 6/8 for their combination, and 6/8
and 7/8 for content planning. No cross row reached the absolute threshold.

### What each method showed

**Neutral control.** Seven of eight own rows were already classified as target
author, but their mean margin remained below the generated-domain threshold.
No cross row was classified as target. This is the reference point, not a style
method.

**Generic full regeneration.** Rewriting every paragraph without target evidence
made the own margin worse and left the cross margin unchanged. Full regeneration
alone does not induce target style and is rejected as a causal explanation for
the positive movement in the target-informed arms.

**Aligned pairs.** This produced the highest fixed-method own success (2/8) and
positive lift in both arms. It is the strongest evidence that concrete neutral-
to-target transformations help. It still missed every cross threshold and had
three own-arm fidelity failures, so the 29-example prompt is not an adequate
transfer method.

**Style definition + examples.** This produced the largest own mean lift
(+0.18798) and the least-negative own mean margin, but no row satisfied the full
success gate. A descriptive card plus six examples can steer aggregate surface
features without reliably producing classifier-level or fidelity-safe transfer.

**Aligned pairs + style definition.** Combining the two evidence types did not
add their gains: it scored one own success and remained at zero cross successes.
The longer heterogeneous prompt likely diluted actionable transformations rather
than producing complementary control.

**Content plan + combined evidence.** The English event/speaker ledger produced
the best cross mean lift (+0.12959), but the best cross mean margin was still
-1.21694, almost 1.47 margin units below threshold. It also had the most failed
rows. Planning helps organize content but does not itself solve style realization.

**Independent selector.** Hard gating eliminated deterministic fidelity failures
and retained the best 2/8 own rate, but it could not improve the cross result.
An oracle selecting any available candidate could reach only 3/8 distinct own
successes and 0/8 cross successes. Candidate generation, not ranking, is the
main bottleneck.

### Fidelity and copy safety

| Method | Failed rows: own / cross | Error instances observed |
| --- | ---: | --- |
| Generic regeneration | 2 / 2 | 5 Latin-token mismatches |
| Aligned pairs | 3 / 0 | 5 Latin-token mismatches, 1 dialogue-turn mismatch, 1 unbalanced quote |
| Style definition + examples | 2 / 1 | 2 Latin-token mismatches, 1 unbalanced quote |
| Aligned pairs + definition | 2 / 1 | 5 Latin-token mismatches |
| Content plan + combined | 3 / 3 | 13 Latin-token mismatches, 1 dialogue-turn mismatch |
| Independent selector | 0 / 0 | none after candidate hard gating |

An error-instance count can exceed the number of failed rows. All fixed methods
exceeded the registered 10% fidelity ceiling in at least one arm. No method
introduced an exact reference-copy sequence under the frozen eight-character
rule, and all 80 fixed outputs preserved the required artifact and paragraph
schema.

The dominant Latin-token failure identifies a tractable engineering problem:
names and protected strings were enforced by instruction, not by entity-slot
delexicalization and deterministic restoration. Fixing it would improve fidelity,
but it would not rescue the style result: ignoring fidelity, the fixed methods
still achieved at most 2/8 own threshold passes and 0/8 cross passes.

### Selector close-reading diagnostics

The selector scored each eligible anonymous candidate from 1-5 for semantic
fidelity, naturalness, and adherence to the corpus-grounded style definition.
These scores explain its decisions; they are not the registered independent
semantic endpoint because the same call both judged and selected candidates.

| Candidate family | Eligible judged appearances | Hard semantic flags | Mean semantic | Mean naturalness | Mean style adherence | Mean rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Aligned pairs | 13 | 0 | 4.31 | 4.23 | 4.38 | **2.38** |
| Aligned pairs + definition | 12 | 2 | 4.17 | 4.25 | 4.00 | 2.83 |
| Content plan + combined | 9 | 1 | **4.56** | **4.33** | 4.22 | 2.56 |
| Neutral baseline | 16 | 1 | 4.19 | 4.00 | 3.63 | 2.75 |
| Style definition + examples | 12 | 2 | 4.08 | 4.17 | 4.17 | 2.92 |

The aligned-pair candidate had the best mean rank and no selector-identified
hard semantic errors. Content planning had the highest mean semantic and
naturalness scores but was frequently ineligible because of deterministic
token failures. These are useful qualitative clues for subsequent research, not proof
of semantic noninferiority.

## Protocol Deviations and Limits

### Development evaluator admission omission

The frozen shared evaluator initially scored zero generated rows because its
allowed admission-stage table omitted `development_v1`, although the generator
correctly recorded `preregistered_development_pilot`. A validation-only wrapper
was frozen before generated-method style scores were exposed. It removes only
that exact mismatch when the ledger proves the frozen 16-row selection, its
hash/count, `official_efficacy_evidence=false`, and the registered pilot stage.
It changes no text, score, threshold, fidelity, copy, or cohort rule.

The amendment is recorded at
`protocols/development_evaluator_admission_amendment.v1.json` (SHA-256
`1a7f18e9e2d38a89d4608323687a76e0c707fbb52d59c1c0f189780ef0a37c42`).
The amended evaluation summary has SHA-256
`22a134951dfd44da0892ea137c79973ac12ff1688c29f3ec9e1ec05efb7c0148`.
Six focused contract tests pass. Because the wrapper was not in the original
preregistration, all numbers above are explicitly **post-lock exploratory
development diagnostics**, not promotion evidence.

### Missing registered semantic/readability path

The written protocol requires blind semantic/readability judgment during the
development pilot. The only judgment builder consumes `style_critique` outputs,
but the frozen runner rejects `style_critique` for `development_v1`. A separate
protocol-path auditor, **Mendel**
(`019f5d14-dcf4-7502-9853-1eda8172b499`), verified that no supported command can
produce the registered endpoint. Reusing selector ratings would be post-
selection conditioning, not independent evaluation.

This contradiction cannot be repaired retrospectively. Iteration 4 therefore
makes no registered semantic-noninferiority claim. The least-biasing remedy is a
new prospective iteration with the critique roster, admission stage, rehearsal,
and analysis lock fixed before generation.

## Style-Meter Construct Validity

The frozen historical exact character 2-4 gram classifier passed its
then-current authorship benchmark, but those pre-normalization performance
figures are retired. The independent outcome evaluator also observed substantial
target-class false positives. Even before the later corpus correction, these
results measured predictive authorship discrimination rather than a
content-independent style construct.

A deterministic post-outcome audit inspected the target class's top 300 positive
features across 5,650 target train chunks from 29 books and 19,307 comparison
chunks from 49 authors.

![Frozen meter positive weight by diagnostic category](../../generated/style_research/style_transfer_experiments/iterations/full_regeneration_v1/audits/style_meter_construct_validity/category_weight_mass.svg)

| Construct diagnostic | Result |
| --- | ---: |
| Risk-flagged features in top 20 | 7/20 |
| Risk-flagged features in top 50 | 20/50 |
| Risk-flagged positive-weight share in top 300 | **37.7%** |
| Recurrent ambiguous lexical weight | 26.2% |
| Mask-artifact weight | 18.1% |
| Known entity/topic-fragment weight | 11.7% |
| Low-book-dispersion lexical weight | 7.9% |
| Dialogue-structure weight | 17.3% |
| Function-grammar weight | 10.4% |

Useful high-weight cues include `，说：`, `答道`, `只得`, and `继而`. Material
content cues include `小悦`, `魔法`, `克里`, `杜景`, `骑士`, `吕布`, `曹天裁`,
`阿加斯`, and `佣兵`; placeholder patterns such as `与某` also carry weight.
Book-disjoint accuracy therefore did not remove enough theme, entity, and mask-
pipeline identity.

The correct interpretation is not that the classifier is useless. Paired margin
lift remains a useful secondary diagnostic because each generated output shares
source content with its neutral baseline. However, the absolute cross-author
threshold is not a valid standalone primary style endpoint: an honest transfer
must not gain score by importing target names or world vocabulary. The 80%
research objective remains valid, but its primary operationalization must be
replaced before another efficacy claim.

The reproducible audit command and full feature table are in
`generated/style_research/style_transfer_experiments/iterations/full_regeneration_v1/audits/style_meter_construct_validity/report.md`.

## Independent Evaluation

| Separate agent | Scope | Verdict | Consequence |
| --- | --- | --- | --- |
| Plato (`019f5cba-0eda-7313-9627-d90c1080d1e7`) | Final bilingual source adjudication | PASS | Last English row admitted after a bounded semantic repair |
| Avicenna (`019f5cc5-b657-7300-9ceb-6d340f9f2572`) | Exact pre-style methodology snapshot | GO | Development generation authorized |
| Mendel (`019f5d14-dcf4-7502-9853-1eda8172b499`) | Registered development judgment path | NO supported path | No registered semantic/readability claim is available |
| Godel (`019f5d19-c554-7ba3-a01b-3d09a8da2dd9`) | Frozen outcomes, amendment, statistics, leakage, and construct validity | **NO-GO** | No method advances to formal screening |

The outcome evaluator independently checked the 16-row binding, 80 successful
generation-ledger entries, amendment scope, per-method summaries, selector, and
top target-class features. Its central findings were:

1. the amended figures are understandable descriptive diagnostics but not
   confirmatory or official screening evidence;
2. no method meets the style or fidelity gate, and semantic noninferiority is
   unmeasured;
3. the weak candidate-pool oracle proves that selection is not the main
   bottleneck;
4. generator weakness is real, but cross-arm failure cannot be attributed solely
   to generation because the absolute meter is construct-contaminated; and
5. A subsequent study must prospectively repair both measurement and evaluation before
   testing a materially stronger generation mechanism.

## Conclusion

**Iteration 4 is complete at the development stage and is a NO-GO.** None of the
five tested full-regeneration methods, nor the blind selector, reached the
registered target. The strongest own result was 25% against a 7/8 pilot gate;
all cross results were 0%. No method should be integrated into Eternal Gate pass
2, and no Iteration 4 screening, confirmation, or final cohort should be opened.

The experiment nevertheless establishes four actionable results:

1. target-informed prompting produces real paired movement, while generic full
   regeneration does not;
2. aligned transformations are more actionable than description alone, but 29
   prompt examples are too sparse and fidelity is not mechanically protected;
3. a selector cannot repair a candidate pool in which no cross candidate reaches
   the endpoint; and
4. the current ngram classifier is suitable only as a historical attribution and
   paired-lift diagnostic, not as the next primary cross-content style meter.

Any subsequent transfer study must be prospectively new rather than an in-place repair. Its
minimum research design is:

1. build and freeze a content-resistant primary style representation using
   stricter entity/lore masking, leave-one-book-out validation, feature
   concentration filtering, counterfactual name/theme swaps, and blinded human
   convergent validation;
2. mechanically delexicalize protected names, numbers, and Latin tokens before
   generation and restore them afterward;
3. replace sparse prompt imitation with dense pseudo-parallel reconstruction:
   neutralize many target-train passages, then learn or demonstrate the mapping
   back to their original style without exposing raw references at inference;
4. generate a fixed multi-candidate set and apply semantic/dialogue/copy gates
   before a selector driven only by the content-resistant style representation;
5. preregister the missing independent critique path and require 7/8 successes
   in each fresh development arm, zero hard-fidelity failures at this sample
   size, zero reference copying, and blind semantic acceptance before screening;
6. run a new separate methodology agent before generation and a different
   outcome agent after every frozen iteration.

This moves the research from prompt-card combinations toward the reconstruction
and contrastive-control methods supported by TinyStyler-style self-distillation
and authorship-transfer policy optimization. Iteration 4's screening and later
partitions remain preserved as untouched evidence; any new transfer study must
use a fresh development allocation and its own lock.
