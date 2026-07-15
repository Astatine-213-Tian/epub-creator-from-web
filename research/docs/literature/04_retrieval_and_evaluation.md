# Note 04: Retrieval and Evaluation

## Useful Insight

More reference text is not automatically better. The retrieval layer should select the
right kind of evidence: same scene function, compatible setting/genre, diverse books,
and no long copied passages. Evaluation should be multi-axis: source fidelity,
target-language fluency, author-style fit, and anti-template/GPT wording.

## Retrieval Notes

### Legacy local pipeline

The removed local vector pipeline used:

- corpus: `generated/vector_indexes/eternal_gate/corpus.json`
- vectors: `generated/vector_indexes/eternal_gate/vectors.npy`
- metadata: `generated/vector_indexes/eternal_gate/metadata.json`
- embedding model: `intfloat/multilingual-e5-small`
- reference books: eight Feitian/Guxuerou EPUBs

This was removed from the active pipeline because semantic retrieval by itself did
not give enough explicit control over:

- genre/setting match,
- scene-function match,
- book balance after style-domain weighting,
- lexical phrase-frame evidence,
- source-vs-style separation.

### LlamaIndex option

Source: LlamaIndex Python documentation, queried via Context7 on 2026-07-03.
https://developers.llamaindex.ai/python/

Relevant current concepts:

- `VectorStoreIndex` can be built from documents/nodes or from an existing vector
  store.
- `StorageContext` connects an index to a vector store.
- Retrievers can use `MetadataFilters` / `MetadataFilter` with operators such as
  equality and inclusion.
- Vector-store integrations can be backed by remote stores, while the application
  still retrieves nodes with metadata.

Pipeline implication:

- Do not force LlamaIndex into the core implementation. The local NumPy index is faster
  to maintain.
- Design the metadata schema so that local retrieval and a later LlamaIndex/vector-db
  backend can share the same fields:
  `book`, `author_alias`, `genre_bucket`, `setting_family`, `cultural_frame`,
  `historical_anchor`, `scene_tags`, `chapter_position`, `dialogue_ratio_bucket`,
  `style_card_ids`, `source_path`.

## Evaluation Notes

### TST evaluation is multi-dimensional

Sources:

- "Evaluating Style Transfer for Text", ACL 2019. https://aclanthology.org/N19-1049/
- "Text Style Transfer Evaluation Using Large Language Models", LREC-COLING 2024.
  https://aclanthology.org/2024.lrec-main.1373/
- "Evaluating Text Style Transfer Evaluation: Are There Any Reliable Metrics?",
  NAACL SRW 2025. https://aclanthology.org/2025.naacl-srw.41/

Relevant finding:

- TST evaluation is commonly separated into style transfer/intensity, content
  preservation, and naturalness/fluency.
- Human evaluation is preferred but costly; LLM-based and metric ensembles can provide
  useful proxies if treated as proxies.

Pipeline implication:

- Add a local evaluator with four channels:
  1. semantic fidelity: entities, numbers, glossary, comments, paragraph boundaries,
     and a back-translation or LLM comparison sample;
  2. fluency: Chinese punctuation, sentence length outliers, malformed JSON, awkward
     translationese;
  3. style fit: distance to author profile by genre/scene bucket, phrase-frame
     opportunities used/skipped, dialogue/scene rhythm;
  4. negative style: GPT/report-like wording, generic four-character padding, abstract
     evaluation nouns, over-explaining motives.

### Evaluation should produce actionable repairs

The evaluator should not output only a score. It should identify:

- paragraphs where style was over-applied,
- paragraphs where style opportunities were missed,
- paragraphs whose retrieved references were off-domain,
- paragraphs where source fidelity blocked style application,
- suspicious reuse of raw reference wording.

## Design Rule

Retrieval should feed the translator with "style evidence relevant to this paragraph",
not "lots of author text". Evaluation should ask whether the output is:

- faithful to the English,
- fluent as Chinese,
- closer to the selected author/genre style,
- not contaminated by generic GPT-style prose.
