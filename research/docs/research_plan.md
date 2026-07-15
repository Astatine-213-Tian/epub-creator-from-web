# Design Doc: Dataset-First Author Style Transfer Research

## Purpose

Design a proper research workflow for textless back-translation of `永恒之门`.

`永恒之门` is available to us only as an English translation. The original work was
written in Chinese by 顾雪柔, an alias of 非天夜翔. Our goal is not exact recovery of
the lost Chinese source. The goal is to build and evaluate a defensible pipeline
that converts the English text into Chinese that is:

1. faithful to the English meaning;
2. natural as Chinese web-novel prose;
3. measurably closer to the target author's Chinese style than a neutral draft;
4. auditable through dataset controls, benchmarks, and paragraph-level validation.

The production implementation now uses the three-pass generation flow:

```text
English source
  -> Pass 1: neutral Chinese translation
  -> Pass 2: author-style transfer
  -> Pass 3: English-grounded validation and repair
```

The active pass-2 implementation is
`content_plan_combined_full_regeneration`: English-grounded paragraph content
planning, neutral Chinese as a terminology/content anchor, retrieved aligned
masked examples, a validated style definition, and complete paragraph
regeneration. This operational selection was made after direct reader review of
the Eternal Gate application artifact. It does not retroactively turn the
Iteration 4 NO-GO into an 80% research success claim; research conclusions and
production preference remain distinct.

## Superseded Work

The current single-author profile build and cluster artifacts are not the method we
want to reuse.

The profile intermediates, style cards/contracts, probe outputs, and retired
`book-translate profile build`, `profile cluster`, and `profile close-read`
commands have been removed. Their methodological history remains summarized in
the literature notes; none of those artifacts is an active baseline or fallback.

Reason: they were built primarily from the target author's corpus. They do not prove
that the extracted patterns distinguish author style from topic, genre, setting,
proper nouns, or general BL/webnovel conventions.

Reusable infrastructure:

- the pass-1 neutral translation command surface;
- run directories, prompt/output JSON shape, and validation merge logic;
- EPUB build logic;
- semantic-compression QA as the production pass-3 review component.

Everything author-style-specific must be re-tested from the multi-author dataset.

## Research Questions

### RQ1: style identification

Which features or representations can distinguish 非天夜翔/顾雪柔 from other BL or
webnovel authors after controlling for content, genre, setting, and book-specific
vocabulary?

### RQ2: style transfer

Which LLM-assisted style-transfer method moves neutral Chinese prose toward the
target author style while preserving meaning and readability?

### RQ3: production applicability

Can the best method from RQ1 and RQ2 improve `永恒之门` over a neutral translation
without introducing semantic drift, artificial Chinese, copied phrasing, or
style-guide leakage?

## Methodology

This is a pragmatic mixed-methods computational-literary study:

- quantitative authorship/style benchmarks decide whether a style signal is
  measurable;
- controlled LLM style-transfer experiments compare candidate methods;
- qualitative close reading and sampled human review judge readability and literary
  acceptability;
- paragraph-level QA protects the English source meaning.

The project should proceed as a research pipeline, not as prompt tuning.

## Data Strategy

### Primary dataset

Use the local multi-author corpus under `datasets/raw/`, indexed by
`datasets/dataset_manifest.json`.

Current observed scale after expansion and cleanup:

- 198 books;
- 50 canonical author labels, all with at least 3 books;
- 40 books labeled `非天夜翔`;
- 198 usable books at `>=50k` cleaned CJK characters;
- 195 primary books at `>=120k` cleaned CJK characters.

Raw files remain audit evidence only. All modeling and sampling use generated clean
texts plus the masked/unmasked chunk views.

### Target application data

Use the `永恒之门` English crawl and its existing neutral Chinese translation runs
only after method selection. `永恒之门` should not be used to train the author-style
identifier because it has no known Chinese original in this project.

### Proxy evaluation data

Because `永恒之门` has no Chinese gold original, use held-out Chinese novels for
controlled proxy experiments:

1. hold out whole target-author books and sample spaced chunks within them;
2. generate a plain English semantic source from the held-out Chinese in an isolated,
   one-way model session;
3. run the production-compatible neutral English-to-Chinese prompt, then style
   transfer;
4. freeze outputs before the evaluator opens the hidden original;
5. repeat the transfer test on a separate non-target-content arm.

This gives a measurable approximation of the real task while avoiding leakage.

## Stage 0: Dataset Audit And Cleaning

No style method should be selected before the dataset is audited.

### Cleaning goals

Build a clean, reproducible corpus suitable for author identification and transfer
experiments.

### Required checks

- file exists for every manifest row;
- author aliases are canonicalized, especially `顾雪柔` -> `非天夜翔`;
- encoding is valid UTF-8 after export;
- book title and author headers are normalized;
- non-novel boilerplate is removed or flagged;
- repeated site disclaimers, update notices, menus, comments, and source metadata
  are removed;
- duplicate chapters and duplicated books are detected;
- extremely short, incomplete, or fragmentary texts are flagged;
- fanwai, afterwords, author notes, and extras are identified;
- mixed-language or non-main-text spans are flagged;
- source quality issues are recorded rather than silently cleaned away.
- typographically equivalent Chinese and ASCII punctuation is canonicalized before
  any clean or masked chunks are generated, while numeric forms such as `3.14` and
  `12:30` are preserved.

### Clean corpus artifacts

Proposed layout:

```text
generated/style_research/corpus/
  raw_manifest_snapshot.json
  cleaned_manifest.json
  cleaning_report.json
  cleaning_report.md
  author_aliases.json
  quality_flags.json
  texts/
    <author>/<book_id>.clean.txt
```

### Inclusion rules

For the first benchmark:

- include authors with at least 3 full-length usable books when possible;
- keep one-book and two-book authors for later stress tests, not primary training;
- keep held-out target-author books for proxy style-transfer evaluation;
- split by book, never by random paragraph.

## Stage 1: Chunking, Masking, And Splits

### Chunking

Build author-classification chunks from cleaned text:

- 800-1,500 CJK characters per chunk;
- preserve source book, chapter, paragraph offsets when available;
- discard chunks dominated by metadata, poems, lists, or non-prose unless explicitly
  testing those forms;
- keep a balanced sample per book so long books do not dominate.

### Content masking

Maintain unmasked and masked versions.

Mask or normalize:

- character names;
- place names;
- title-specific invented terms;
- rare high-TF-IDF content words;
- book titles and chapter titles;
- unique ranks, sects, dynasties, organizations, and setting nouns;
- numbers when they are plot-specific;
- source-site residue.

Do not mask:

- punctuation;
- sentence boundaries;
- common function words;
- dialogue punctuation;
- common verbs;
- paragraph length and sentence rhythm.

### Splits

Use book-level splits:

```text
train: books used to train style identifiers
dev: books used for model selection and threshold tuning
test: held-out books used once for final reporting
proxy_transfer: held-out target-author passages used for textless reconstruction tests
```

No chunks from the same book may appear in more than one split.

## Stage 2: Author-Style Identification Benchmark

The first central question is whether we can measure target-author style under
content controls.

### Baseline methods

The current selected supervised baseline is exact character 2-4 gram TF-IDF with
an SGD hinge classifier. In plain language, it measures which short adjacent
character sequences recur in each author's prose. The benchmark compares an
unweighted classifier with a class-balanced companion on both clean and
entity-masked text. Exploratory punctuation, dialogue, length, function-character,
function-word, hashed n-gram, logistic, Naive Bayes, and passive-aggressive variants
remain diagnostic evidence; they are not the selected meter.

Feature families:

- character n-grams 1-4;
- Chinese function characters and function words;
- punctuation and punctuation n-grams;
- dialogue punctuation patterns;
- paragraph and sentence length distributions;
- clause count and connective count;
- type-token and lexical diversity diagnostics.

### Representation methods to test after baselines

Only add these after baselines are working:

- Chinese sentence or paragraph embeddings;
- contrastive author-style embeddings with same-genre hard negatives;
- LLM-authorship verification prompts adapted from PromptAV-style methods;
- LLM-extracted style attributes adapted from LISA-style methods.

### Hard negatives

Compare target author against authors with similar genre, scene, and audience. Do not
let the benchmark be solved by comparing fantasy to unrelated modern slice-of-life
or very short personal essays.

Hard-negative controls should include:

- same broad BL/webnovel domain;
- similar genre or `article_type`;
- similar dialogue ratio;
- similar chunk length;
- similar scene tags where available;
- same source site/export format when possible.

### Required metrics

Report both unmasked and masked metrics. The current hinge classifier exposes
decision margins rather than calibrated probabilities, so ROC AUC and probability
calibration remain future work.

```text
top1_author_accuracy
target_author_precision
target_author_recall
target_author_roc_auc
target_author_probability_calibration
confusion_matrix_by_author
confusion_matrix_by_genre_or_article_type
per_book_score_distribution
feature_importance_or_explanation
```

The style identifier is usable as a style meter only if:

- masked target-author recall is well above random and majority baselines;
- held-out target-author books are recognized consistently;
- errors are interpretable;
- feature importance is not dominated by names, titles, or setting terms;
- confidence scores are calibrated enough for neutral-vs-styled comparisons.

Current milestone result after punctuation normalization, train-only global-mask
fitting, and removal of mask-marker features: exact character n-gram TF-IDF with
unweighted SGD hinge loss reaches 89.8% test accuracy and 87.9% balanced accuracy
over 50 authors. The book-local `entity_masked_v3` view remains a diagnostic, not
the selected attribution proxy. Interpretable feature families remain diagnostic
guardrails. See `docs/reports/02_authorship_style_meter.md`.

## Stage 3: Style-Transfer Method Comparison

Do not choose a style-transfer method until it is tested.

### Candidate transfer methods

The versioned registry at
`generated/style_research/style_transfer_experiments/method_registry/style_methods.v1.json`
is authoritative. It preregisters 13 methods spanning the neutral control, weak and
strong generic prompts, global and scene-routed cards, sentence-flow controls,
function-word/punctuation/dialogue controls, retrieval, contrastive retrieval,
hybrids, LLM close reading, statistically gated close reading, and a verifier-loop
repair method. Failed methods remain in the report.

The current single-author card/contract method is not the default and is not reused
as the evidence source for the new registry.

### Proxy task

For held-out target-author Chinese passages:

```text
original Chinese passage
  -> plain English semantic source
  -> neutral Chinese draft
  -> candidate style transfer method
  -> evaluation against hidden original, style identifier, semantic QA, readability
```

For non-target authors:

```text
non-target content
  -> neutral Chinese draft
  -> transfer toward target author style
  -> evaluate style movement and semantic preservation
```

This separates "recovering author's own hidden passage" from "applying target style
to different content."

### Style intensity

Each method should test at least two intensities:

- light: minimal recasting, prioritize readability and fidelity;
- medium: stronger rhythm and diction transfer;
- strong: stress test for style overreach.

Production should use the lightest method that gives a measurable and readable
style gain.

## Stage 4: Evaluation Framework

Evaluate every candidate on separate axes.

### Style similarity

- masked target-author raw decision margin and rank;
- style lift over neutral draft;
- distance to target-author centroid;
- nearest-author margin;
- style-card or attribute opportunity use, if the method uses cards.

### Semantic faithfulness

- English-source meaning preservation;
- entities, numbers, roles, speaker turns, causality, negation, modality, and
  intensity;
- no invented motives, lore, injuries, jokes, intimacy, or setting detail;
- neutral-vs-styled semantic compression or expansion.

### Readability and naturalness

- natural Chinese fiction cadence;
- no translationese;
- no report-like style-analysis prose;
- no prompt/style-guide leakage;
- no clipped pseudo-literary compression;
- no repetitive favorite phrases.

### No-copy and contamination

- no long overlap with target-author reference text;
- no rare phrase copying;
- no reference-book names or setting concepts imported into `永恒之门`;
- no dynasty, xianxia, or court vocabulary unless the English source supports it.

### Human review

For sampled passages, present side-by-side:

```text
English source
neutral Chinese draft
candidate styled output
hidden original Chinese, only for proxy benchmark review
evaluation warnings
```

Human review should answer:

- Which version is most faithful?
- Which version is most natural Chinese?
- Which version is closest to target-author style?
- Which version sounds artificial or over-styled?

## Stage 5: Production Three-Pass Pipeline

The production pipeline is wired for `永恒之门` using the operationally selected
content-plan combined method. Continued benchmark research may replace it later,
but historical profile/card methods must not be reintroduced as fallbacks.

### Pass 1: neutral translation

Keep the existing semantic-draft idea:

- English is the semantic authority;
- output natural Simplified Chinese;
- preserve paragraph boundaries, names, numbers, speaker turns, event order, and
  glossary;
- do not imitate author style.

### Pass 2: selected style-transfer method

Use `content_plan_combined_full_regeneration` at strong intensity.

Inputs must include:

- English paragraph;
- neutral Chinese draft;
- glossary-preserving neutral Chinese;
- hash-locked aligned masked examples and validated style definition;
- no-copy constraints;
- paragraph IDs.

The style transfer pass must produce metadata for evaluation:

- method name;
- style intensity;
- style cues used;
- cues skipped;
- warnings or uncertainty.

### Pass 3: validation and repair

Pass 3 is mandatory. The current production path combines deterministic
fidelity/no-copy checks during generation, semantic-compression candidate
detection, LLM review, conservative repair acceptance, and final structural
validation. Style-meter scoring remains a research diagnostic rather than an
automatic production repair trigger.

The build gate binds pass 3 to the exact styled chunk state, source English,
neutral translations, snapshot and semantic manifests, maintained config and
glossary, review contract, candidate report, batch prompts, and parsed review
results. A stale, dry-run, failed, rejected, or unresolved `true_loss` review is
not publishable.

### Independent production audit, 2026-07-14

An independent `gpt-5.6-sol` audit initially returned FAIL because unresolved
`true_loss` findings could publish, current source inputs were not fully rehashed,
and QA provenance was incomplete. After those gates were repaired, a separate
`gpt-5.6-luna` re-audit returned PASS on all three findings. The re-audit verified
fail-closed unresolved-loss handling, per-block English/neutral source binding,
mandatory production QA, complete QA artifact provenance, and prompt-hash-gated
reuse of parsed review results.

Validation outputs:

```text
generated/translation_runs/<run_id>/author_style_transfer/
  style_transfer_summary.json
  validation_summary.json
  semantic_compression/
    semantic_compression_candidates.json
    semantic_compression_summary.json
    semantic_compression_summary.md
```

Repair rule:

- repair only flagged paragraphs;
- preserve English meaning over style;
- accept only low-risk repairs that clear deterministic signal checks;
- unresolved paragraphs go to manual review and block EPUB construction.

## Analysis Plan

### Primary quantitative outcomes

For author identification:

```text
masked target-author recall
masked target-author precision
masked ROC-AUC
calibrated target-author score stability across held-out books
```

For style transfer:

```text
style_lift = target_author_score(styled) - target_author_score(neutral)
semantic_error_rate
readability_warning_rate
no_copy_warning_rate
human_preference_rate
```

### Statistical reporting

- report per-book and per-author results, not only pooled chunk scores;
- use bootstrap confidence intervals over books or chapters;
- use paired comparisons for neutral vs styled outputs;
- report all failed or inconclusive methods, not only the winner.

## Proposed Artifacts

```text
generated/style_research/
  corpus/
    cleaned_manifest.json
    cleaning_report.md
    splits.json
  author_identification/
    benchmark_config.json
    benchmark_summary.json
    benchmark_summary.md
    confusion_matrix.masked.csv
    per_book_scores.csv
    feature_importance.csv
  style_transfer_experiments/
    sample_sets/development_proxy_v1.runner_manifest.jsonl
    sample_sets/development_proxy_v1.evaluator_allocation.jsonl
    sample_sets/development_proxy_v1.hidden_targets.jsonl
    sample_sets/development_proxy_v1.summary.json
    prompts/english_semantic_source.v1.md
    prompts/neutral_translation.v1.md
    prompts/style_transfer_method.v1.md
    schemas/*.schema.json
    protocols/model_run_config.v1.json
    protocols/evaluation_protocol.v1.json
    method_registry/style_methods.v1.json
    method_registry/methods/*.v1.json
    runs/development_proxy_v1/<run_id>/<stage>/
    evaluations/setup_review_history.v1.json
  decisions/
    method_selection_memo.md
```

The cleaned and masked chunk views are stored outside generated artifacts at
`datasets/unmasked/chunks.clean.jsonl` and
`datasets/masked/chunks.entity_masked_v3.jsonl`.

Stable config can live under:

```text
book_specs/style_research/author_style_transfer_config.json
book_specs/eternal_gate/style_transfer_config.json
```

## Implementation Order

1. **Complete:** freeze the old single-author profile path as a legacy baseline.
2. **Complete:** audit and clean the corpus; build masked/unmasked chunks and
   leakage-free book splits.
3. **Complete:** train author-identification baselines and select the masked exact
   n-gram classifier as the provisional generated-text style meter.
4. **Complete:** freeze 160 target-author and 60 cross-author chunks with opaque
   runner IDs, evaluator-only allocation, spacing controls, and residue checks.
5. **Complete:** version prompts, schemas, evaluation protocol, method configs, and
   the stateless generation runner with an append-only ledger, external repository
   read isolation, strict provenance validation, and English-source quarantine.
6. **Complete:** generate controlled English semantic sources and neutral Chinese;
   obtain independent methodology reviews.
7. **Complete through Iteration 4:** compare candidate LLM style-transfer methods
   and preserve failed results. The registered research gate remained NO-GO.
8. **Incomplete research gate:** no method reached the preregistered 80% threshold
   on a clean official validation sequence.
9. **Operational selection:** reader review selected
   `content_plan_combined_full_regeneration` at strong intensity for Eternal Gate,
   with the research limitation recorded explicitly.
10. **Complete:** replace the production pass-2 pipeline, require pass-3 semantic
    QA, and retain separate-agent evaluation before release.

## Decision Gates

Do not proceed from dataset work to style-transfer experiments unless:

- the cleaned corpus report is reproducible;
- author labels and aliases are canonicalized;
- enough comparison authors pass inclusion filters;
- book-level splits are leakage-free.

Do not use an author classifier as a style meter unless:

- masked performance is meaningfully above baseline;
- target-author recall is stable across held-out books;
- feature importance is not content-dominated.

Do not choose a transfer method unless:

- it improves masked target-author style score over neutral;
- it does not increase high-severity semantic errors;
- it does not reduce readability in sampled review;
- it passes no-copy checks.

Do not build the final `永恒之门` EPUB unless:

- pass 3 has no unresolved high-severity semantic errors;
- style-transfer exceptions are documented;
- manual review accepts representative high-risk samples.

## Risks And Mitigations

| Risk | Mitigation |
| --- | --- |
| Classifier learns content instead of style | masked corpus, hard negatives, feature audits |
| Dataset contains noisy scraped text | cleaning report, quality flags, inclusion filters |
| Target author dominates dataset size | balanced chunk sampling per book and author |
| Style transfer damages meaning | mandatory pass-3 English-grounded validation |
| Output sounds like the style guide | avoid long prose style guides; test leakage explicitly |
| LLM judge bias | combine deterministic checks, classifier scores, and human review |
| Reference text copying | overlap checks and no-copy prompt constraints |
| Proxy task differs from `永恒之门` | evaluate multiple domains and report external-validity limits |

## What Not To Do

- Do not reuse the current single-author profile/cluster as the chosen method.
- Do not start by writing more style cards from target-author-only evidence.
- Do not optimize for author score without semantic and readability gates.
- Do not train/test with chunks from the same book in both splits.
- Do not treat unmasked authorship accuracy as proof of style.
- Do not paste long analytical style reports into generation prompts.
- Do not claim exact recovery of the original Chinese text.
