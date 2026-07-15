# Note 06: Authorship-Style Benchmark and LLM Method Inventory

## Purpose

This note records methods that are useful for turning the author-style work into a
separate benchmarked project. The immediate goal is not to implement every method.
The goal is to keep a reviewed menu of candidate methods, prompt patterns, and
evaluation designs so we can run controlled experiments and choose what actually
improves `永恒之门`.

Source status: checked against primary paper pages or PDFs on 2026-07-04 where
available. Exact prompt wording is summarized rather than copied wholesale; use
the linked papers for full templates.

## Working Research Question

Can a style-transfer pipeline move a neutral Chinese translation measurably closer
to a target author's Chinese prose style while preserving the source meaning and
remaining natural to read?

The benchmark should measure three axes separately:

1. **Author-similarity gain**: target-author score increases from neutral draft to
   styled draft.
2. **Semantic faithfulness**: no lost, invented, compressed, or shifted content.
3. **Chinese readability**: output reads like natural fiction, not analysis prose,
   translationese, or style-guide leakage.

## Core Design Implication

An authorship classifier is useful as a style meter, but only if it is trained and
tested with content controls. Otherwise it may learn title-specific names, genre
vocabulary, setting terms, or plot topics instead of style.

The benchmark should therefore report both:

```text
unmasked_author_accuracy
masked_author_accuracy
style_lift = target_author_score(styled) - target_author_score(neutral)
```

and reject style lift when semantic drift or readability regressions are high.

## Literature Matrix

| Source | Method Family | What To Reuse | Main Caution |
| --- | --- | --- | --- |
| Eder, Rybicki & Kestemont (2016), `stylo` | Traditional stylometry toolkit | MFW, n-grams, frequency tables, culling, Burrows Delta, PCA/MDS/clustering, supervised classifiers, rolling Delta/classify | Parameter-sensitive; shallow features explain similarity but not literary function |
| Yu (2012), Chinese function words | Chinese low-semantic features | Per-thousand rates for Chinese function characters; test author vs genre effects | Function words also encode genre; small corpus and EM clustering instability |
| Sundararajan & Woodard (2018) | Style vs content diagnosis | Compare character LM strength against syntax/POS/topic-masked variants; proper-noun masking as content-control idea | English parser/POS findings do not transfer directly to Chinese |
| Wegmann, Schraagen & Nguyen (2022) | Content-controlled style representations | Authorship verification with hard negatives matched by conversation/domain/topic; STEL-Or-Content logic | Conversation labels do not exist for novels; need proxy controls |
| Patel et al. (2023), LISA | Interpretable style embeddings | LLM-generated style attributes, SFAM agreement model, 768-dimensional interpretable style vector | English Reddit domain; LLM annotation hallucination/content leakage |
| Kim & Jurgens (2026) | Style-eliciting prompts | Decode latent style vectors into actionable prompts; evaluate prompt recovery and style steering | English QA domain; possible misuse for imitation/deanonymization |
| Zhu et al. (2023), StoryTrans | Long story author-style transfer | Discourse representations, sentence-order objective, style classifier loss, mask-and-fill content preservation | Heavy model training; content preservation remains the main failure mode |
| Tao et al. (2024/2025), CAT-LLM | Chinese long-text style definition prompts | Structured Text Style Definition from word-level and sentence-level features; prompt includes task, source, style definition, output slot | Article-style transfer, not novel translation; style definitions can still hallucinate if too mechanical |
| Wegmann & Nguyen (2021), STEL | Style representation evaluation | Content-controlled style-similarity tasks with paraphrase pairs | Current STEL dimensions are mostly English and sentence-level |
| Rivera-Soto et al. (2021), LUAR | Universal authorship representations | Cross-domain authorship verification and contrastive representation training | Authorship embeddings can still entangle domain/content |
| Wang & Riddell (2022), CCTAA | Chinese cross-topic benchmark | Book/document split and cross-topic Chinese authorship testbed idea | Newswire genre is not webnovel fiction |
| Hung et al. (2023), PromptAV | LLM authorship verification | LLM prompt asks for a 0-1 same-author score with explicit stylometric variables and reasoning | Explanation may be illusory; cost and latency high |
| Horvitz et al. (2024), TinyStyler | Few-shot authorship style transfer | Use authorship embeddings from target examples; mean-pool examples; interpolate source/target style strength | Embedding bottleneck may miss literary style; automatic metrics imperfect |
| StyleDistance (2024/2025 preprint) | Synthetic content-independent style embeddings | LLM-generated parallel examples with controlled content/style; synthetic STEL-like evaluation over many style features | Sentence-level and English; synthetic bias/repetitive generations |

## Method Inventory

### A. Traditional Baselines

Use first because they are cheap and interpretable.

1. **MFW / most frequent characters and words**
   - Build a document or chunk x feature matrix.
   - Try culling to remove features not shared across enough documents.
   - For Chinese, start with characters, punctuation, and segmented words only if
     tokenizer quality is acceptable.

2. **Character n-grams**
   - Use 1-4 character n-grams.
   - Keep a masked variant that removes names, places, and unique setting terms.
   - Treat high-scoring n-grams as diagnostics, not prompt text.

3. **Burrows Delta / distance family**
   - Z-score features across documents.
   - Score chunk-to-author distance.
   - Use for visual sanity checks and nearest-author reports.

4. **Function-character vector**
   - Start from Yu's single-character list:
     `的 是 不 了 在 有 这 为 地 也 得 就 那 以 着 之 可 么 而 然 没 于 还 只 无 又 如 但 其 此 与 把 全 被 却`
   - Extend later with multi-character Chinese function words and sentence-final
     particles.

5. **Punctuation and dialogue rhythm**
   - Quote mark usage, colon/semicolon density, ellipses, dash patterns, short
     sentence clusters, paragraph length, sentence count per paragraph.
   - Especially relevant for fiction dialogue and action pacing.

### B. Supervised Authorship Benchmark

This should be the first real benchmark milestone.

Recommended setup:

```text
unit: 800-1500 Chinese characters per chunk
split: by book, not by paragraph
authors: target author + 10-20 comparison authors
books per comparison author: 2-4
target books: all available, or at least 5
features: char n-grams + punctuation + function characters
models: logistic regression, linear SVM, random forest as baseline
neural optional: Chinese RoBERTa or contrastive encoder after baseline
```

Required controls:

- no train/test leakage from the same book;
- title, character, place, and proper-noun masking;
- same-genre hard negatives;
- per-book and per-author confusion matrix;
- masked vs unmasked accuracy;
- calibration curve for target-author probability.

Acceptance threshold for using the classifier as a style meter:

```text
masked_top1_author_accuracy > random baseline by a large margin
target_author_recall is stable across held-out books
confusions are interpretable, not random
classifier confidence is calibrated enough for relative neutral-vs-styled comparisons
```

### C. Content-Controlled Contrastive Representation

Inspired by Wegmann et al. and StyleDistance.

Training triplets:

```text
anchor: target author chunk A
positive: same author, different book or different topic chunk B
negative: different author, same genre / similar scene / similar topic chunk C
```

Novel-specific content-control proxies:

- same genre bucket;
- same scene tag: dialogue, combat, intimacy, court/politics, exposition;
- similar named-entity density;
- similar chapter position: opening/body/closing;
- similar source length and dialogue ratio.

Evaluation:

- authorship verification accuracy under hard negatives;
- STEL-like pair ranking adapted to Chinese fiction;
- whether neutral/styled drafts move in embedding space toward target author.

### D. LLM Authorship Verification

Inspired by PromptAV.

Prompt pattern to test, paraphrased:

```text
Given Text 1 and Text 2, estimate from 0 to 1 whether they were written by the
same author. Consider punctuation, capitalization/special characters, abbreviations,
idioms, tone, sentence structure, and other relevant style variables. Reason step
by step and output the confidence score.
```

For Chinese fiction, replace English-specific variables with:

- punctuation and dialogue punctuation;
- function characters/particles;
- sentence and paragraph rhythm;
- idiom/chengyu density;
- dialogue action beats;
- metaphor density;
- narrator stance and emotional restraint;
- scene-mechanics cues.

Use this as a secondary judge, not as the main metric. PromptAV itself notes that
generated explanations can be illusory.

### E. LLM Style-Attribute Extraction

Inspired by LISA.

Prompt pattern:

1. Ask an LLM to describe a passage's grammar/vocabulary/punctuation/stylometric
   features while avoiding topic specifics.
2. Standardize the result into short declarative attributes such as `"The author uses X"`.
3. Train or score a model that maps `(text, style_attribute)` to an agreement score.

For our project:

- use Chinese prompts;
- make attributes scene-scoped;
- reject attributes that mention content, named entities, book-specific settings,
  or generic literary praise;
- treat attributes as candidate features that require statistical support.

Useful adaptation:

```json
{
  "attribute": "The author uses short sentence afterbeats after emotional dialogue.",
  "scene_scope": ["dialogue", "intimacy"],
  "agreement_score": 0.0,
  "evidence_chunks": []
}
```

### F. Style-Eliciting Prompt Discovery

Inspired by Kim & Jurgens.

Key idea: style explanations should be actionable prompts, not just descriptions.

Prompt-pattern registry:

- generate a curated set of small imperative style features;
- combine 1-10 features into style prompts;
- generate stylized text from those prompts;
- train a decoder from style vector to style prompt;
- evaluate whether recovered prompts produce new text close to the reference style.

For this project, a lighter version is enough:

```text
style_vector/reference evidence -> compact Chinese imperative prompt card
prompt card + neutral draft + English source -> styled output
```

A style card should be executable:

```json
{
  "prompt_rule": "After emotionally loaded dialogue, use one short physical or silence beat before explanation.",
  "apply_when": ["dialogue", "emotional turn"],
  "do_not_apply_when": ["source gives no emotional reaction", "action sequence needs speed"],
  "evidence": ["stats", "short examples", "counterexamples"]
}
```

### G. Chinese Style Definition Prompting

Inspired by CAT-LLM.

Prompt structure:

```text
Task: transfer the original text to the following style definition while retaining content.
Original text: <source/styleless text>
Style definition:
  Words: POS tendencies, word length, mono/polysyllabic words, modal particles, idioms.
  Sentences: length, long-short balance, emotion, sentence structure, rhetoric.
Output: transferred text only.
```

CAT-LLM also uses a style-removal prompt to create styleless text. The prompt role
asks the model to behave like a Chinese university literary scholar, remove the
original work's style, make the expression more modern/colloquial, preserve meaning,
and output only the styleless text.

For our pipeline, the neutral translation pass already plays the "styleless" role,
so we should not add a separate Chinese style-removal pass unless we need an
ablation.

Important CAT-LLM ablation to test:

```text
word-level style definitions before sentence-level definitions
```

The paper reports this order worked better than sentence-first prompting.

### H. Long-Text Author-Style Transfer

Inspired by StoryTrans.

Reusable ideas without full model training:

- style transfer should operate above sentence level;
- discourse/order features matter;
- use a style classifier loss or post-hoc classifier score as a target-style signal;
- protect content by masking or locking names, places, proper nouns, numbers, and
  source-specific keywords;
- evaluate style and content separately because strong style transfer can damage
  content.

For `永恒之门`, this maps to:

```text
locked entities + glossary + paragraph map
neutral draft
style rewrite with English visible
semantic drift check
style score check
```

## Prompt Pattern Registry

| Paper | Prompt Type | Reusable Pattern | Use In Our Pipeline |
| --- | --- | --- | --- |
| LISA | open-ended style annotation | describe grammar/vocabulary/punctuation/stylometry while avoiding topic | candidate style-attribute mining |
| LISA | standardization | rewrite descriptions into `"The author uses/is X"` attributes | normalize LLM style observations |
| Kim & Jurgens | style feature generation | generate short imperative style features that are independent, concrete, minimal | build an executable style-feature bank |
| Kim & Jurgens | stylized generation | answer a content prompt while following hidden style instructions; do not mention the instructions | controlled synthetic style examples |
| Kim & Jurgens | style decoding | map style vectors to a single-sentence prompt listing concrete traits | convert representations into style cards |
| CAT-LLM | style-enhanced transfer | task + original text + structured style definition + output slot | stage-2 style transfer prompt skeleton |
| CAT-LLM | style removal | remove source style, preserve meaning, output styleless text | possible ablation; likely redundant with neutral translation |
| PromptAV | authorship verification | score same-author probability with explicit stylometric variables and reasoning | LLM secondary verifier |
| TinyStyler | few-shot LLM baseline | show examples by one author, ask model to rewrite source in that author's style | baseline to compare against structured style cards |

## Evaluation Protocol

For each selected chapter/chunk, generate:

```text
A. neutral translation
B. structured-card style transfer
C. few-shot example-only style transfer
D. strong style transfer
```

Run:

1. **Authorship classifier score**
   - target-author probability;
   - nearest peer;
   - margin over nearest peer;
   - masked and unmasked variants.

2. **Style representation distance**
   - distance to target-author corpus centroid;
   - distance to same-genre comparison authors;
   - style lift over neutral draft.

3. **Semantic QA**
   - source-vs-neutral;
   - source-vs-styled;
   - neutral-vs-styled;
   - entity, number, speaker, event, and intensity checks.

4. **Readability QA**
   - unnatural Chinese;
   - prompt-guide leakage;
   - over-compression;
   - generic GPT literary phrasing;
   - overuse of detected style tokens.

5. **Blind human/LLM preference**
   - Which version is more faithful?
   - Which version is more natural Chinese?
   - Which version is closer to target author?
   - Which version sounds artificial?

## Experiment Queue

### Phase 1: Benchmark Baseline

- Build author/book manifest from available EPUBs and crawled comparison corpus.
- Extract chunk records with book-level split.
- Train simple char-ngram/function/punctuation classifiers.
- Report masked vs unmasked accuracy and confusion matrix.

### Phase 2: Style-Transfer Scoring

- Score existing `永恒之门` neutral/styled outputs.
- Compute style lift and semantic-risk correlations.
- Identify whether style lift corresponds to human readability improvements.

### Phase 3: Prompt Ablations

Test:

1. long prose style guide;
2. compact structured style cards;
3. word-level card first, sentence-level card second;
4. few-shot author examples only;
5. LISA-like attributes;
6. strong vs light style intensity.

### Phase 4: Representation Upgrade

Only after Phase 1-3 show value:

- train contrastive author-style embeddings with hard negatives;
- build STEL-like Chinese fiction style tasks;
- experiment with style-eliciting prompt decoder or lightweight SFAM-style model.

## Recommended Near-Term Decision

Do not start with a neural style-transfer model. Start with the benchmark:

```text
Can we classify authors reliably after masking content-heavy terms?
```

If yes, use the classifier as a relative style meter for neutral-vs-styled drafts.
If no, the corpus or feature design is not strong enough yet, and style transfer
experiments will be hard to interpret.

## Sources

- Eder, M., Rybicki, J., & Kestemont, M. (2016). Stylometry with R: A package for computational text analysis. The R Journal. https://journal.r-project.org/articles/RJ-2016-007/
- Yu, B. (2012). Function Words for Chinese Authorship Attribution. ACL Anthology. https://aclanthology.org/W12-2506/
- Sundararajan, K., & Woodard, D. (2018). What represents "style" in authorship attribution? ACL Anthology. https://aclanthology.org/C18-1238/
- Wegmann, A., Schraagen, M., & Nguyen, D. (2022). Same Author or Just Same Topic? ACL Anthology. https://aclanthology.org/2022.repl4nlp-1.26/
- Patel, A., Rao, D., Kothary, A., McKeown, K., & Callison-Burch, C. (2023). Learning Interpretable Style Embeddings via Prompting LLMs. ACL Anthology. https://aclanthology.org/2023.findings-emnlp.1020/
- Kim, J., & Jurgens, D. (2026). Interpreting Style Representations via Style-Eliciting Prompts. ACL Anthology. https://aclanthology.org/2026.findings-acl.2039/
- Zhu, X., Guan, J., Huang, M., & Liu, J. (2023). StoryTrans: Non-Parallel Story Author-Style Transfer with Discourse Representations and Content Enhancing. ACL Anthology. https://aclanthology.org/2023.acl-long.827/
- Tao, Z., Xi, D., Li, Z., Tang, L., & Xu, W. (2024/2025). CAT-LLM: Style-enhanced Large Language Models with Text Style Definition for Chinese Article-style Transfer. arXiv / ACM. https://arxiv.org/abs/2401.05707
- Wegmann, A., & Nguyen, D. (2021). Does It Capture STEL? ACL Anthology. https://aclanthology.org/2021.emnlp-main.569/
- Rivera-Soto, R. A., et al. (2021). Learning Universal Authorship Representations. ACL Anthology. https://aclanthology.org/2021.emnlp-main.70/
- Wang, H., & Riddell, A. (2022). CCTAA: A Reproducible Corpus for Chinese Authorship Attribution Research. ACL Anthology. https://aclanthology.org/2022.lrec-1.633/
- Hung, C.-Y., Hu, Z., Hu, Y., & Lee, R. (2023). Who Wrote it and Why? Prompting Large-Language Models for Authorship Verification. ACL Anthology. https://aclanthology.org/2023.findings-emnlp.937/
- Horvitz, Z., Patel, A., Singh, K., Callison-Burch, C., McKeown, K., & Yu, Z. (2024). TinyStyler: Efficient Few-Shot Text Style Transfer with Authorship Embeddings. https://www.cis.upenn.edu/~ccb/publications/tinystyler.pdf
- StyleDistance: Stronger Content-Independent Style Embeddings. arXiv. https://arxiv.org/abs/2410.12757
