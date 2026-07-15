# Note 07: Content Masking And Content-Control Methods

## Purpose

This note records how existing authorship/style research controls for topic, theme,
characters, and book-specific content. The implementation should not simply "randomly
mask names"; it should compare multiple content-control views and report whether
author-style signal survives.

Source status: checked against primary paper pages or PDFs on 2026-07-08.

## Why Masking Is Necessary

Authorship attribution models can learn topic or book identity instead of author
style. For `永恒之门`, this is especially dangerous because 非天夜翔's works vary by
setting, but the target reconstruction is a fantasy novel. A classifier might learn
`骑士`, `圣光`, `长安`, `驱魔`, dynastic ranks, character names, or worldbuilding terms
instead of cadence, punctuation, function-word rhythm, dialogue handling, or scene
mechanics.

Therefore every author-style benchmark needs at least two views:

```text
unmasked view: measures upper-bound authorship signal, including content shortcuts
masked view: measures whether author signal remains after content/theme suppression
```

Only masked or content-controlled results should be used as a style meter.

## Methods In Existing Research

### 1. POS / lexical-class masking

Sundararajan and Woodard (2018) directly test which linguistic content represents
style by masking all words or topic words by part of speech: nouns, verbs,
adjectives, adverbs, and especially proper nouns. Their finding is nuanced:

- proper nouns are strongly content-driven and should be masked in cross-domain
  attribution;
- common nouns can heavily influence attribution in single-domain data;
- some common nouns, verbs, adjectives, and adverbs can still carry useful
  stylometric signal in cross-domain data;
- syntax alone is not enough, but can help in cross-genre settings.

Design implication: for our project, mask proper names and book-specific terms
aggressively, but compare lighter and heavier masks rather than deleting all lexical
content in one pass.

Source: https://aclanthology.org/C18-1238/

### 2. Text distortion

Stamatatos (2017) introduces text distortion before feature extraction. The goal is
to transform text into a more topic-neutral form while preserving stylometric
structure. The extended JASIST version tests several variants:

- **DV-MA**: mask least frequent words, preserving token length with repeated mask
  characters;
- **DV-SA**: mask without preserving token length;
- **DV-EX**: keep exterior characters/letters while masking interiors;
- **DV-L2**: preserve a small amount of suffix/ending information;
- tune parameter `k`, the number of frequent words left visible.

Important principles:

- preserve punctuation, sentence shape, and frequent/function words;
- mask low-frequency words because they are more likely topic-specific;
- choose mask strength by experiment, not by intuition;
- different genres may need different distortion strength.

For Chinese web novels, the equivalent is not word-letter masking. A usable
adaptation is:

```text
keep: punctuation, sentence boundaries, common function characters/words
mask: low-frequency or low-dispersion content terms, names, places, invented terms
optionally preserve: CJK token length or first/last character for rhythm diagnostics
```

Sources:

- https://aclanthology.org/E17-1107/
- https://icsdweb.aegean.gr/stamatatos/papers/jasist-2018-preprint.pdf

### 3. Pre-processing for cross-topic character n-grams

Markov, Stamatatos, and Sidorov (2017/2018) show that character n-gram attribution
can be improved with simple pre-processing:

- replace each digit with `0`, preserving number format;
- split punctuation so punctuation frequency/patterns are visible;
- replace named entities with a common symbol, preserving occurrence but not identity;
- optionally replace highly frequent words with distinct symbols;
- tune feature count and compare typed n-gram categories.

They emphasize that character n-grams capture both content and form. Pre-processing
tries to make them capture more personal style and less theme.

Design implication: our chunk builder should emit at least a normal cleaned view and
a preprocessed/masked view. For feature extraction, punctuation should remain
visible even if content words are masked.

Source: https://www.cic.ipn.mx/~sidorov/CICLing-Markov-Preprint.pdf

### 4. Cross-topic and hard-negative splits

Wang and Riddell's CCTAA corpus is directly relevant because it is Chinese. Their
design uses cross-topic Chinese authorship attribution with predefined splits. The
paper reports that both Chinese RoBERTa sequence classification and function-character
n-gram SVM baselines perform below expectation, which is a warning: Chinese
cross-topic authorship attribution is genuinely hard.

Design implication:

- do not rely only on random splits;
- use book-level splits;
- test cross-genre or cross-theme conditions where possible;
- report predefined split files so results are reproducible.

Source: https://aclanthology.org/2022.lrec-1.633/

### 5. Content-controlled representation learning

Wegmann, Schraagen, and Nguyen (2022) argue that authorship verification training
can learn content because authors often write about recurring topics. They introduce
content control using conversation or domain labels and find conversation-controlled
training better than domain/no control for content-independent style representations.

Design implication for novels:

- we do not have conversation labels, so use proxy controls:
  - same genre bucket;
  - same scene type;
  - same dialogue ratio;
  - same chunk length;
  - same chapter position;
  - similar named-entity density;
- for contrastive training or evaluation, pick hard negatives matched on these
  proxies rather than random negatives.

Source: https://aclanthology.org/2022.repl4nlp-1.26/

### 6. Content-controlled style evaluation

STEL evaluates whether a model compares style rather than content by using
content-controlled sentence pairs. This is not an authorship benchmark, but the
evaluation principle is useful: style measures need tasks where content is held
constant or explicitly controlled.

Design implication:

- build proxy pairs where content is similar but style should differ;
- evaluate whether the style scorer prefers target-author style over same-topic
  content;
- for `永恒之门`, do not trust a style metric unless it wins on masked or
  content-controlled examples.

Source: https://aclanthology.org/2021.emnlp-main.569/

### 7. Topic-or-style feature comparison

Sari, Stevenson, and Vlachos (2018) frame authorship attribution as a mix of writing
style and preferred topics, and compare feature usefulness under different dataset
conditions. This supports treating feature choice as empirical: character n-grams,
word features, syntactic/style features, and representations should be compared under
our own split conditions.

Design implication:

- do not pick one mask level or feature family upfront;
- run a matrix of feature families against unmasked and masked views;
- use the method that survives content controls, not the method with the highest
  unmasked accuracy.

Source: https://aclanthology.org/C18-1029/

## Recommended Masking Views For This Project

Generate multiple views from the same cleaned chunks:

### View A: `clean`

Cleaned novel text after removing metadata, chapter headings, author notes, URLs,
and obvious boilerplate. This is the readable source for inspection and a high-signal
upper-bound benchmark.

### View B: `entity_masked`

Mask likely names, places, organizations, invented terms, and title-specific terms.

Candidate sources:

- book title and metadata terms;
- repeated 2-5 CJK terms with high TF-IDF within one book;
- terms highly concentrated in one book or one author;
- capitalized/Latin terms;
- glossary-like invented words;
- manually supplied known names for target-author proxy books if needed.

Use placeholders by type when possible:

```text
<NAME>
<PLACE>
<ORG>
<TERM>
<NUM>
```

If type is unknown, use a neutral `<CONTENT>` placeholder.

### View C: `topic_distorted`

A stronger Stamatatos-style distortion view:

- keep common function characters/words;
- keep punctuation and paragraph/sentence boundaries;
- normalize numbers;
- mask low-frequency CJK content terms;
- preserve approximate token length if useful for rhythm features.

This view is useful for asking: is there still author signal in punctuation,
function-word rhythm, sentence length, dialogue shape, and structural habits?

### View D: `structure_only`

A stress-test view:

- keep punctuation;
- keep paragraph and sentence boundaries;
- keep normalized dialogue marks;
- replace most CJK content with generic placeholders.

This will probably be too severe for production scoring, but it tells us how much
signal comes from structure alone.

## Recommended Evaluation Matrix

For each split, train/evaluate:

| View | Purpose | Expected Use |
| --- | --- | --- |
| `clean` | upper-bound authorship signal | diagnostic only |
| `entity_masked` | remove obvious character/theme shortcuts | primary first style-meter candidate |
| `topic_distorted` | stronger content control | robustness check |
| `structure_only` | severe stress test | diagnostic, not production |

For each view, compare:

- char n-grams;
- punctuation and dialogue punctuation;
- function characters/words;
- sentence/paragraph length;
- combined interpretable baseline;
- optional embeddings after baseline.

Report:

```text
clean_accuracy
entity_masked_accuracy
topic_distorted_accuracy
structure_only_accuracy
target_author_recall_by_view
feature_importance_by_view
```

If performance collapses only after masking, the clean score was probably content
shortcut learning.

## Adaptation To One-Book Authors

One-book authors may be used only in auxiliary experiments.

If we split one book by content, do not use random chapter chunks as primary
evidence. Use contiguous block splits and apply stronger masking. Label the result
as `within_book_sanity`, because it tests consistency within one book, not
generalizable author style.

Main benchmark policy remains:

- primary labels need at least 3 usable books;
- one-book authors are out-of-distribution or auxiliary;
- if more data is collected, prioritize adding books to existing 3-book comparison
  authors.

## Implementation Guidance

Before modeling, emit:

```text
chunks.clean.jsonl
chunks.entity_masked.jsonl
chunks.topic_distorted.jsonl
chunks.structure_only.jsonl
mask_terms.json
masking_report.md
```

Each chunk record should preserve:

- source `book_id`, `author`, split, and offsets;
- original clean text path;
- masked text;
- mask counts by type;
- top masked terms;
- CJK length before and after masking.

Do not delete the unmasked view. The unmasked-vs-masked gap is one of the most
important diagnostics.
