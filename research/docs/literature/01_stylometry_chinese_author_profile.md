# Note 01: Stylometry and Chinese Author Profiles

## Useful Insight

Stylometry is useful here only if it stays interpretable. A classifier that says
"this sounds like author X" is less useful than a profile showing which stable
signals make the output author-like: function words, character n-grams, clause
rhythm, punctuation, dialogue density, phrase frames, scene functions, and distribution
across books or genres.

## Source Notes

### Yu 2012: Chinese function words are useful, but genre interferes

Source: Bei Yu, "Function Words for Chinese Authorship Attribution", ACL CLfL 2012.
https://aclanthology.org/W12-2506/

Relevant finding:

- The paper tests Chinese function-word authorship attribution across novels,
  essays, and blogs.
- It explicitly treats genre as a possible interference factor.
- It asks whether function-word usage is stable across time periods.

Pipeline implication:

- Do not build one undifferentiated "Feitian/Guxuerou style". Split style evidence
  by broad domain: dynasty/historical, ancient fantasy, modern, European fantasy,
  comedy/dialogue-heavy, and spectacle/action.
- Function words can be part of the profile, but they should be normalized and
  compared inside genre or setting buckets.

### Kestemont 2014: function words are attractive but not magic

Source: Mike Kestemont, "Function Words in Authorship Attribution: From Black Magic
to Theory?", ACL CLfL 2014.
https://aclanthology.org/W14-0908/

Relevant finding:

- Function words are frequent, shared by writers, and comparatively less topic-driven.
- The paper warns that function words can still correlate with genre, narrative
  perspective, demographics, and topic.
- Character n-grams often work well but are harder to interpret because they capture
  "a bit of everything."

Pipeline implication:

- Use function words and character n-grams as diagnostic signals, not as prompt
  content.
- Convert them into human-readable constraints only when the signal can be explained:
  sentence rhythm, connective preference, dialogue turn cadence, punctuation, or
  repeated clause structure.
- If a feature is predictive but not interpretable, keep it in the validator, not in
  the translator prompt.

### Burrows 2002 and Argamon 2008: use distance as a validator, not as style itself

Sources:

- John Burrows, "`Delta': A Measure of Stylistic Difference and a Guide to Likely
  Authorship", 2002. https://www.semanticscholar.org/paper/1081e505f03a9ae0dc3d55881b59a67e264f2d1f
- Shlomo Argamon, "Interpreting Burrows's Delta", 2008. https://doi.org/10.1093/llc/fqn003

Relevant finding:

- Burrows's Delta compares texts using relative frequencies of common words.
- Argamon provides a geometric/probabilistic interpretation, making the method's
  assumptions and limits clearer.

Pipeline implication:

- Add a style-distance validator for generated Chinese, but do not optimize only for
  Delta-style closeness.
- Use chunked samples and book-level cross-validation. A translation should move
  closer to the target author cluster without collapsing into the nearest reference
  book or into topic contamination.

### Eder, Rybicki, and Kestemont 2016: reproducible exploratory stylometry matters

Source: "Stylometry with R: A Package for Computational Text Analysis", R Journal
2016. https://journal.r-project.org/articles/RJ-2016-007/

Relevant finding:

- The `stylo` package is presented as a flexible exploratory tool for authorship
  verification and historical/literary research.
- It emphasizes reproducible pipelines, similarity metrics, and exploratory
  visualization.

Pipeline implication:

- Save profile artifacts with exact corpus paths, preprocessing settings, feature
  families, and per-book distributions.
- Treat profile generation like a reproducible build step, not a one-off LLM
  summary.

### Hou and Huang 2019: Chinese style can include sub-lexical/phonological signals

Source: "Robust stylometric analysis and author attribution based on tones and
rimes", Natural Language Engineering 2019.
https://doi.org/10.1017/S135132491900010X

Relevant finding:

- The paper proposes stylometry without annotation, using Mandarin tones/rimes,
  tone motifs, rime motifs, and word-length motifs.

Pipeline implication:

- For a lightweight version, do not implement full phonological stylometry now.
- Keep an optional future feature bucket for rhythm-like Chinese signals: sentence
  length, pause punctuation, repetition, and possibly pinyin tone/rime motifs for
  high-confidence prose passages.

## Design Rule

An interpretable author profile should separate:

- **Predictive diagnostics**: function words, character n-grams, Delta-like distance,
  classifier scores.
- **Promptable mechanisms**: phrase frames, dialogue rhythm, punctuation cadence,
  action blocking, weather/light openings, title/court/procedure language.
- **Validation metrics**: style distance, genre-bucket closeness, source-fidelity
  checks, anti-GPT wording checks.

Only promptable mechanisms should enter the translation prompt.
