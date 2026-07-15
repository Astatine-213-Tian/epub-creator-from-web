# Note 02: Corpus Stylistics and Evidence Cards

## Useful Insight

Corpus stylistics provides the bridge missing from pure stylometry: it turns counts
into interpretable literary mechanisms. The key is to mine repeated forms first, then
use close reading to decide what local textual function each form performs.

## Source Notes

### Mahlberg 2007: clusters point to local textual functions

Source: Michaela Mahlberg, "Clusters, key clusters and local textual functions in
Dickens", Corpora 2007.
https://nottingham-repository.worktribe.com/output/1017589/clusters-key-clusters-and-local-textual-functions-in-dickens

Relevant finding:

- Repeated word clusters can be read as pointers to local textual functions.
- The method identifies functional groups such as labels, speech clusters, body-part
  clusters, and time/place clusters.

Pipeline implication:

- This matches the current problem better than a generic "write in author's style"
  instruction.
- Mine recurring Chinese phrase frames and cluster them by local function:
  dialogue turn, non-answer, gaze, body contact, weather opening, action vector,
  procedural authority, mythic causal explanation, scene close.
- Each cluster should become an evidence card with counts, book distribution, examples,
  and explicit limits.

### Biber and corpus stylistics: quantitative + qualitative reading

Source: Douglas Biber, "Corpus linguistics and the study of literature: Back to the
future?" / Cambridge handbook chapter on literary style and literary texts.
https://resolve.cambridge.org/core/services/aop-cambridge-core/content/view/CDF598153F16040DD877C6C75D46D5A0/9781139764377c19_p346-361_CBO.pdf/literary_styleand_literary_texts.pdf

Relevant finding:

- Corpus-stylistic work combines quantitative corpus techniques with qualitative
  interpretation of literary text.
- Keywords, clusters, collocations, and concordance-style evidence are common tools.

Pipeline implication:

- Counts alone should not be promoted to rules.
- Every strong style rule should include local examples and an interpretation of
  narrative function.
- For Chinese, single characters are too weak as style claims unless they participate
  in phrase frames or syntactic positions.

### Removed Eternal Gate profile: useful lesson, unsafe prompt surface

Local context:

- Current reviewed source index: `book_specs/author_styles/feitianyexiang/reference_book_index.high_medium.json`
- Current profile-builder code: `src/translation/profile_builder.py`
- The previous `style_profile.json`, synthesis report, prompt rules, fixed
  `SCENE_SPECS`, and old style-analysis module were removed from the active pipeline.

Pipeline implication:

- Rebuild discovery from the new manifest/paragraph-corpus artifacts.
- Use broad lenses only as evidence discovery aids, not prompt categories.
- Generate the final style contract from accepted evidence cards, not from generic
  prose such as "prefer restrained atmosphere" or "use concrete verbs" unless the
  card shows when and why.

## Proposed Evidence Card Shape

```json
{
  "card_id": "dialogue.non_answer_turn",
  "label": "non-answer as dialogue action",
  "scene_bucket": "dialogue",
  "genre_buckets": ["dynasty", "fantasy", "modern"],
  "support": {
    "paragraph_count": 1140,
    "book_distribution": {"相见欢.epub": 230, "天宝伏妖录.epub": 310},
    "contrastive_lift_vs_baseline": 2.4,
    "chapter_position_distribution": {"opening": 0.08, "body": 0.82, "closing": 0.10}
  },
  "forms": {
    "priority_frames": ["没有回答", "没有说话", "沉默片刻"],
    "left_context": ["听到这话", "X问", "Y道"],
    "right_context": ["又问", "叹了口气", "抬眼看"]
  },
  "narrative_function": "silence behaves as a turn-taking action and lets pressure move to the next speaker",
  "source_triggers_en": ["did not answer", "said nothing", "after a pause"],
  "prompt_constraint": "When the English has a true non-answer beat, prefer a compact Chinese silence frame and let the next action or question carry the emotion.",
  "limits": ["do not infer romance, refusal, or sadness without source evidence"]
}
```

## Design Rule

The prompt should receive evidence-card summaries, not the research report. A good
prompt rule has:

- source trigger,
- target Chinese mechanism,
- when to use,
- when not to use,
- evidence strength,
- two or three short target frames.

It should not include long paragraphs of analytical prose, which the model may copy
as GPT-like narration.
