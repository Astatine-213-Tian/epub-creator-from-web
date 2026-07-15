# Note 05: Historical Author-Profile Methodology (Superseded)

> **Status:** Historical research note only. The single-author profile/card/
> cluster design described below was rejected after the multi-author benchmark
> exposed content-confounding risk. Do not implement its production-builder
> recommendations. The active method and evidence are in
> [`../research_plan.md`](../research_plan.md) and the numbered reports.

## Recommendation

Build the author profile with a hybrid method:

1. **Statistical discovery first**: extract robust Chinese style features and cluster
   them with reproducible methods.
2. **LLM interpretation second**: use LLM close reading to name, merge, split, and
   reject candidate clusters.
3. **Prompt distillation last**: convert only stable, interpretable clusters into
   compact style cards.

Do not let an LLM read the corpus and directly "summarize the author's style" as the
primary method. That produces plausible-sounding but weakly grounded style guides.

## Research Basis

### Chinese stylometry needs cross-topic and cross-genre controls

Yu (2012) shows that Chinese function words are useful for authorship attribution, but
also that genre and authorship can interfere with each other. The reported function-word
method worked best for novels and less well for noisier genres. This supports using
function words, but only with genre/domain controls and distribution reporting.

Source: https://aclanthology.org/W12-2506/

Wang and Riddell's CCTAA corpus paper argues that Chinese authorship research needs
reproducible testbeds and cross-topic design. Their baselines with Chinese RoBERTa
and function character n-grams performed below expectations, which is a useful warning:
style attribution under topic shift is hard.

Source: https://aclanthology.org/2022.lrec-1.633/

### Character n-grams are powerful diagnostics, not directly promptable rules

Kestemont (2014) explains why function words and character n-grams are attractive in
stylometry while also warning against treating them as magic. Sapkota et al. (2015)
show that not all character n-grams contribute equally; categories such as affix-like,
punctuation, and word-boundary n-grams can carry different kinds of signal.

Sources:

- https://aclanthology.org/W14-0908/
- https://aclanthology.org/N15-1010/

For Chinese, Zhang's *Authorship Analysis in Chinese Social Media Texts* uses
one- to four-character n-grams alongside word length, type-token ratio, and sentence
length. This supports a Chinese feature range of character n-grams 1-4, but these
features should become diagnostics and validators rather than raw prompt text.

Source: https://www.cambridge.org/core/elements/authorship-analysis-in-chinese-social-media-texts/CDFBDEE28085274DE8E3FEA3921AFD06

### Sub-lexical and rhythm-like Chinese signals may help, but are second-stage

Hou and Huang (2017/2020) explore tone, rime, and word-length motifs for Chinese
stylometry without annotation. This is interesting because it captures rhythm-like
signals missed by word features.

Sources:

- https://aclanthology.org/Y17-1011/
- https://www.cambridge.org/core/journals/natural-language-engineering/article/abs/robust-stylometric-analysis-and-author-attribution-based-on-tones-and-rimes/7276F866631E3F34456C3D4EB71A2E0B

For this project, these should be later diagnostics. The production design should
first stabilize sentence length, punctuation, phrase frames, and dialogue/action
mechanics.

### Corpus stylistics turns counts into usable literary mechanics

Mahlberg's Dickens work treats repeated clusters as pointers to local textual functions,
not as proof by themselves. This is the right model for fiction style transfer: a phrase
cluster is useful when we know the local function it performs.

Source: https://www.euppublishing.com/doi/abs/10.3366/cor.2007.2.1.1

For Eternal Gate, the equivalent output should be "evidence cards" such as:

- non-answer as dialogue action;
- gaze as relationship judgment;
- body-position blocking in intimacy;
- action vector plus result;
- weather/light establishing shot;
- mythic concept as causal system.

### Chinese long-text style transfer needs explicit style definitions

CAT-LLM proposes a Text Style Definition module for Chinese long-text style transfer,
analyzing style at word and sentence levels and keeping style definitions extensible.
Its reported evaluation separates transfer accuracy and content preservation.

Source: https://arxiv.org/abs/2401.05707

This supports a `style_definition` artifact rather than a prose "style bible" in the
prompt. For us, the artifact should be a structured tree of evidence cards and
constraints.

### Style transfer evaluation must separate style, content, and fluency

Prabhumoye et al. frame style transfer as rephrasing with stylistic properties while
preserving intent. Mir et al. specify three evaluation axes: style intensity, content
preservation, and naturalness. Ostheimer et al. show that LLM evaluation can correlate
with human judgments, especially with prompt ensembling, but it remains a proxy.

Sources:

- https://aclanthology.org/P18-1080/
- https://aclanthology.org/N19-1049/
- https://aclanthology.org/2024.lrec-main.1373/

For this project, every profile method must be tied to separate validators:
semantic fidelity, Chinese fluency, author-style fit, and no-copy/no-overstyle checks.

## Proposed Profile-Building Pipeline

### 1. Corpus Manifest

Create an explicit manifest before feature extraction:

```json
{
  "book_id": "knight_song",
  "path": "books/非天夜翔/骑士之歌.epub",
  "author_alias": "非天夜翔",
  "language": "zh-Hans",
  "genre_bucket": "secondary_world_epic_fantasy",
  "setting_family": "western_medieval_secondary_world",
  "cultural_frame": "knights_church_magic_undead_dragons",
  "historical_anchor": "secondary_world_no_real_period",
  "quality_flags": [],
  "weight_for_eternal_gate": 1.5
}
```

Required book-level metadata:

- author alias;
- edition/source path;
- genre bucket;
- setting family;
- cultural frame;
- historical anchor, if any;
- known defects: duplicate chapters, fanwai, OCR-like corruption, abridgment;
- inclusion weight for Eternal Gate.

Assign these fields after an intro/TOC/opening survey. Avoid broad residual buckets:
they are too vague for prompt filtering and can hide the difference between modern
urban fantasy, secondary-world fantasy, and Chinese historical fantasy.

### 2. Paragraph Records

Extract paragraph-level records, not one flattened string:

```json
{
  "paragraph_id": "knight_song:chap_01_004:017",
  "book_id": "knight_song",
  "chapter_id": "chap_01_004",
  "paragraph_index": 17,
  "chapter_position": "opening|body|closing",
  "text": "...",
  "char_len": 48,
  "sentence_count": 3,
  "dialogue_ratio": 0.66,
  "scene_tags_heuristic": ["dialogue", "gaze"],
  "quality_flags": []
}
```

The record layer should preserve enough metadata to check whether a pattern is:

- author-wide;
- book-specific;
- genre/setting-specific;
- scene-specific;
- an artifact of one long book or repeated material.

### 3. Feature Families

Extract multiple feature families. Keep each family separate before synthesis.

#### A. Stylometric Diagnostics

Use normalized frequencies, z-scores, and distribution by book:

- character n-grams 1-4;
- Chinese function words and function characters;
- punctuation n-grams, including dialogue punctuation;
- sentence length and paragraph length distributions;
- type-token ratio and lexical diversity variants;
- word length distribution if segmentation is stable enough;
- optional POS/function-tag n-grams if a Chinese tokenizer is introduced later.

These features answer: "Does this sample statistically sit near the author's corpus?"
They do not directly answer: "What should the translator write?"

#### B. Corpus-Stylistic Phrase Evidence

Mine:

- multi-character phrase frames;
- repeated clause fragments;
- collocation windows;
- key clusters versus baseline;
- dispersion across books and scenes;
- left/right context around anchors such as `没有回答`, `抬眼`, `低声`, `冲向`, `叹了口气`.

These features produce interpretable candidate mechanisms.

#### C. Scene Mechanics

Compute scene-lens features as broad retrieval aids, not final categories:

- dialogue and silence;
- gaze/appearance;
- action/combat;
- weather/light/space;
- politics/procedure;
- myth/worldbuilding;
- injury/care;
- intimacy/body blocking;
- humor/facework;
- openings/closings.

Let clusters merge, split, or reject these lenses.

#### D. Semantic Embedding Clusters

Use embeddings to group paragraphs by semantic scene function, not author style alone.
Then compare stylistic features inside each cluster.

Recommended procedure:

1. embed paragraph or 3-paragraph windows;
2. cluster with HDBSCAN or agglomerative clustering;
3. label clusters with top lexical features and representative snippets;
4. ask an LLM to propose a scene-function label;
5. reject clusters that are merely plot/topic/name clusters.

BERTopic-style methods are useful as a model: embeddings plus clustering plus
class-based TF-IDF-style labels. The important part is interpretable cluster labels,
not adopting a specific package immediately.

### 4. Statistics and Stability Tests

Each candidate pattern should receive a validity packet:

```json
{
  "support_n": 430,
  "books_present": 6,
  "min_book_count": 3,
  "dominant_book_share": 0.32,
  "genre_distribution": {"historical_court_war": 0.42, "secondary_world_epic_fantasy": 0.18},
  "dispersion": 0.71,
  "contrastive_lift_vs_baseline": 2.3,
  "bootstrap_stability": 0.84,
  "leave_one_book_out_stable": true,
  "topic_contamination_risk": "medium"
}
```

Recommended metrics:

- **normalized frequency** per 10k Chinese characters;
- **log-likelihood or log-ratio keyness** versus same-genre baseline or cross-author
  baseline;
- **dispersion** so one book does not dominate;
- **dominant-book share** cap;
- **bootstrap stability** over random paragraph samples;
- **leave-one-book-out stability** for author-wide claims;
- **within-domain lift** for Eternal Gate's European-fantasy target domain;
- **negative control** against non-author books when available.

Acceptance rules for promptable cards:

- appears in at least 3 reference books, unless explicitly scoped to one domain;
- dominant book contributes less than 50% of support, unless domain-scoped;
- survives leave-one-book-out or is downgraded;
- has interpretable local function verified by examples;
- has an explicit "do not apply when..." limit.

### 5. LLM Clustering and Close Reading

Use LLMs after statistics have produced candidate packets.

LLM tasks:

- name the mechanism;
- describe local textual function;
- merge overlapping clusters;
- split mixed clusters;
- identify topic/name contamination;
- propose source-English triggers and Chinese target frames;
- write "avoid" rules.

LLM input should be small and evidence-shaped:

```json
{
  "candidate_id": "dialogue.non_answer",
  "stats": {...},
  "top_frames": ["没有回答", "没有说话", "沉默片刻"],
  "left_context": ["听到这话", "X问道"],
  "right_context": ["又问", "叹了口气"],
  "short_examples": [
    {"book": "...", "chapter": "...", "snippet": "..."}
  ],
  "counterexamples": [...]
}
```

The LLM should output:

- `accept | merge | split | reject`;
- mechanism name;
- evidence strength;
- narrative function;
- promptable constraint;
- source trigger examples;
- target phrase-frame options;
- limits and contamination warnings.

### 6. Style Cards

The final profile should produce `style_cards.json`, not only a prose report.

```json
{
  "card_id": "dialogue.non_answer_turn",
  "scope": {
    "author": "非天夜翔/顾雪柔",
    "domain": "global",
    "scene": "dialogue"
  },
  "evidence_strength": "strong",
  "mechanism": "silence or non-answer functions as a dialogue turn",
  "source_triggers_en": ["did not answer", "said nothing", "after a pause"],
  "target_frames_zh": ["没有回答", "没有说话", "沉默片刻"],
  "prompt_constraint": "When the English contains a real non-answer beat, use a compact silence frame and let the next action or question carry pressure.",
  "avoid": ["do not infer romance, refusal, grief, or guilt without source evidence"],
  "stats": {...},
  "examples": [...]
}
```

### 7. Profile Layers

Produce separate layers:

- `style_profile.global.json`: author-wide metrics and cards.
- `style_profile.domain.<bucket>.json`: genre/setting-specific cards.
- `style_profile.scene.<scene>.json`: scene-mechanics cards.
- `style_cards.json`: accepted promptable cards.
- `style_diagnostics.json`: stylometric validators.
- `style_report.md`: human-readable synthesis and limitations.

For Eternal Gate, prompt assembly should prefer:

1. domain-matched cards;
2. scene-matched cards;
3. global cards;
4. retrieved paragraph examples only after card selection.

## Build Stages

Implement these stages as parts of one production profile builder.

### Stage A: Valid Feature Extraction

- Add corpus manifest.
- Rebuild paragraph corpus with metadata.
- Add feature packets:
  - character n-grams 1-4;
  - function words/characters;
  - punctuation/dialogue punctuation;
  - sentence and paragraph length;
  - repeated phrases and clause frames;
  - book/genre dispersion.

### Stage B: Evidence Cards

- Cluster repeated phrase/co-occurrence evidence.
- Add statistical validity packets.
- Run LLM close-reading on top candidates only.
- Output `style_cards.json`.

### Stage C: Profile-To-Prompt Distillation

- Convert accepted cards into compact `style_contract`.
- Remove long style-report prose from translation prompts.
- Add evaluator warnings for:
  - style missed;
  - style over-applied;
  - content drift;
  - GPT-like wording;
  - copied reference text.

## What Not To Do

- Do not build the profile from single-character frequency tables.
- Do not let one book dominate author-wide style.
- Do not prompt with long Chinese analytical paragraphs.
- Do not use LLM labels as evidence unless backed by statistics and examples.
- Do not treat semantic scene clusters as style clusters.
- Do not use style-distance scores as an optimization target without source-fidelity
  checks.

## Practical Decision

For the rewrite, the right author-style profile is:

> a reproducible, paragraph-level, domain-aware corpus model that combines
> stylometric diagnostics with corpus-stylistic evidence cards; LLMs interpret and
> clean clusters, but statistics and examples decide whether a pattern becomes
> promptable.
